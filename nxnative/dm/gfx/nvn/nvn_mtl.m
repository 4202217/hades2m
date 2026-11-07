// Metal execution layer of the NVN null device, see nvn_mtl.h.
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <QuartzCore/CAMetalDisplayLink.h>
#import <ImageIO/ImageIO.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#include <os/proc.h>   // os_proc_available_memory: for the memory readout line
#endif
#include "nvn_mtl.h"
#include "nvn_obj.h"
#include "../tex/nvn_fmt_mtl.h"
#include "../tex/nvn_deswizzle.h"
#include "../tex/nvn_deswizzle_gpu.h"
#include "../tex/nvn_bcdec.h"
#include "nvn_builtin.h"
#include "nvn_dirty.h"
#include "nvn_refl.h"
#include "nvn_cover.h"
#include "dmsh_chain.h"   // DMSH_CHAIN: the translation chain version of the B shader library expected by this source tree (in the game profile folder, -I from the Makefile)
#include "game_profile.h"   // game profile constants (NVN driver constant segment bases, profile id)
// T5: the texture switches default from the game profile (game.tsv keys tex_reclaim / tex_half, generated into game_profile.h);
// the environment overrides them per run. The fallbacks keep the host building with a profile header generated before those keys existed.
#ifndef DM_GAME_TEX_RECLAIM
#define DM_GAME_TEX_RECLAIM 0   // off
#endif
#ifndef DM_GAME_TEX_HALF
#define DM_GAME_TEX_HALF 0      // off
#endif
#ifndef DM_GAME_POOL_RENEW
#define DM_GAME_POOL_RENEW 0    // off (game.tsv key pool_renew on / off; see mtl_pool_del)
#endif
_Static_assert(DM_GAME_NVN_TEX_SLOTS <= 128 && DM_GAME_NVN_IMG_SLOTS <= 32 && DM_GAME_NVN_TEX_BASE + 8 * DM_GAME_NVN_TEX_SLOTS <= 0xA00 && DM_GAME_NVN_IMG_BASE + 8 * DM_GAME_NVN_IMG_SLOTS <= 0xA00 && DM_GAME_NVN_SAMP_BASE + 8 * 128 <= 0xA00, "游戏档案的 NVN 句柄表超出驱动段缓冲（0xA00）或绑定表 | the profile's NVN handle tables exceed the driver segment buffer (0xA00) or the binding tables");
#include "nvn_hmap.h"
#include "../../host/dm.h"
#import <CommonCrypto/CommonDigest.h>
#include <stdarg.h>
#include <mach-o/dyld.h>
#include <libgen.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <stdatomic.h>
#include <errno.h>
#include <execinfo.h>
#include <os/lock.h>
// Function parameters in this file are never retained again in the callee (the caller / global tables already hold them, and lifetime is guaranteed by defer_obj_release): saves a retain / release pair per call on the hot path
#pragma clang attribute push(__attribute__((objc_externally_retained)), apply_to = function)

#define MLOG(...) do { fprintf(stderr, "[dm] [nvn-mtl] " __VA_ARGS__); fputc('\n', stderr); } while (0)
#define MDIE(...) do { MLOG("FATAL " __VA_ARGS__); abort(); } while (0)

static void warm_start(void);
static void warm_claim(void);

static id<MTLDevice> g_dev;
static NSMutableArray *g_l3_gpu_leaks;
static int g_nvn_deswz_gpu;
static id<MTLCommandQueue> g_q;
static id<MTL4CommandQueue> g_q4;
static dispatch_queue_t g_feedback_q;
static _Atomic uint32_t g_present_qos;
#define ALLOC_MAX 8
static id<MTL4CommandAllocator> g_alloc4[ALLOC_MAX]; static int g_alloc_n, g_alloc_cur;   // Command allocator pool of the main submission: at start pick one the GPU has finished with, create a new one if there is none (wait only at the cap)
static id<MTL4CommandAllocator> g_up_alloc4[3];
static id<MTL4CommandAllocator> g_present_alloc4[3];
static id<MTL4CommandBuffer> g_present_cb[3];
static id<MTL4ArgumentTable> g_vt, g_ft, g_ct, g_up_ct;
static id<MTL4ArgumentTable> g_present_args;
static id<MTLBuffer> g_present_uv[3];
int g_bloom_q = -1;   // Bloom 1/4 starting point (non-static: real-device same-process alternation sets it to 0 / 1 directly)
static id<MTL4ArgumentTable> g_bloomq_args;
static id<MTLBuffer> g_bloomq_const[3];
static id<MTLSamplerState> g_bloomq_samp;
static id<MTLRenderPipelineState> g_bloomq_pso;
static MTLPixelFormat g_bloomq_fmt;
static id<MTLTexture> g_bloomq_view;
static int g_bloomq_i;
static struct { int state; uint64_t s_ser, b0_ser, b1_ser, s_h, samp_h, b1_h; const tex_t *s_tex; float threshold, coeff_g, coeff_b; } g_bloomq;
static uint64_t g_bloomq_last_b0, g_bloomq_last_b1h, g_bloomq_last_b1s;   // Most recent bloom 1/4 overlay read redirection: B0 serial → B1 handle (kept across frames, see the 2e2d6700a2569795 branch)
static uint64_t g_present_alloc_fence[3];
static pthread_mutex_t g_pq_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_pq_cv = PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_link_cv = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t g_link_mu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_link_tick_ns, g_link_tick_n;
static uint64_t g_link_dropped;
// The link's MEASURED rate, not the requested one. The system may hand out 60 Hz for a 120 Hz request (low power mode, ProMotion downshift, the app cannot keep up), and the host has to know, because
// the tick count per frame depends on it. Measured over a window of ticks on the display-link callback thread; a new rate is only accepted after it holds for 30 consecutive ticks, so a
// single slow or fast tick does not make the step flapping (the same rule Touchable uses for its display-rate change).
static uint64_t g_link_hz_win_t0, g_link_hz_win_n;   // the window's first tick time and the tick count in it
static int g_link_hz_meas;                           // the measured rate in Hz, rounded to the nearest whole number (0 = not measured yet)
static int g_link_hz_cand, g_link_hz_cand_n;         // the rate candidate and how many ticks it has held for
static uint64_t g_link_hz_log;                       // when the measured rate last changed (for the log line)

// The link's measured rate in Hz (0 until there are enough ticks). Called with g_link_mu NOT held.
static int g_link_hz_measured(void) {
    pthread_mutex_lock(&g_link_mu);
    int hz = g_link_hz_meas;
    pthread_mutex_unlock(&g_link_mu);
    return hz;
}
static double g_link_hold_until; static uint64_t g_link_held;   // Changed only on the display-link callback thread: the time the previous frame must stay on screen until by its present interval (0 = no hold), and the ticks that took no new frame because of it
static int g_pq_inflight;   // Under g_pq_mu: number of presents committed to the GPU whose completion callback has not arrived yet
int g_dm_present_link;   // Present pacing mode (runtime value of DM_PRESENT_LINK, modified directly during same-process alternating measurement): 0 = acquire the drawable itself; 1 = display link at 60 Hz; 2 = display link at 120 Hz, the game presents a frame every other display-link tick
static int g_present_latency = 1;
static uint64_t g_pq_done;
static uint64_t g_pq_read;
static id<MTLResidencySet> g_res;
static int g_res_dirty;
static id<MTLSharedEvent> g_ev;
static uint64_t g_fence;                        // Highest committed fence value (incremented exclusively by the submit thread)
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_cv = PTHREAD_COND_INITIALIZER;
static uint64_t g_done;                         // Highest completed fence value (written by the completion callback)
static uint64_t g_arena_made;
static int g_allocprof = -1; static uint64_t g_ap_bytes[64], g_ap_fn[8];
#define AP_B uint64_t _ap0 = g_allocprof > 0 ? g_dev.currentAllocatedSize : 0
#define AP_E(i) do { if (g_allocprof > 0) { uint64_t _n = g_dev.currentAllocatedSize; if (_n > _ap0) g_ap_fn[i] += _n - _ap0; } } while (0)   // Per-frame arenas created so far (4 MiB each; for the memory report)
static _Atomic int g_last_commit_error;
static _Atomic int g_up_pending;

static pthread_mutex_t g_res_mu = PTHREAD_MUTEX_INITIALIZER;   // Dedicated lock for the residency set: res_add is called on paths that hold g_mu (arena_new), so it cannot share g_mu
static NSHashTable *g_res_objects;
static NSHashTable *g_res_ever; static int g_res_census = -1;
static void res_add(id<MTLResource> r) {
    if (!r) return;
    pthread_mutex_lock(&g_res_mu);
    [g_res addAllocation:(id<MTLAllocation>)r];
    if (!g_res_objects) g_res_objects = [NSHashTable weakObjectsHashTable];
    [g_res_objects addObject:r];
    if (g_res_census < 0) g_res_census = getenv("DM_RES_CENSUS") != NULL;
    if (g_res_census > 0) {
        if (!g_res_ever) g_res_ever = [NSHashTable weakObjectsHashTable];
        [g_res_ever addObject:r];
    }
    g_res_dirty = 1;
    pthread_mutex_unlock(&g_res_mu);
}
static void res_remove(id<MTLResource> r) {
    if (!r) return;
    pthread_mutex_lock(&g_res_mu);
    [g_res removeAllocation:(id<MTLAllocation>)r];
    [g_res_objects removeObject:r];
    g_res_dirty = 1;
    pthread_mutex_unlock(&g_res_mu);
}
// DM_RES_CHECK=1 (diagnostic): if textures / buffers the shader accesses through the argument block are not in the residency set, print the first 40 (textures are attributed to the actual allocation via parent texture / underlying buffer)
static int g_res_check_on;
static void res_check(id r, const char *what, int reg) {
    static int nbad;
    if (!g_res_check_on || !r || nbad >= 40) return;
    int is_tex = [r conformsToProtocol:@protocol(MTLTexture)], is_buf = [r conformsToProtocol:@protocol(MTLBuffer)];
    if (!is_tex && !is_buf) return;   // Samplers etc. are not allocations
    id<MTLResource> a = r; int hops = 0;
    if (is_tex) { id<MTLTexture> t = (id<MTLTexture>)a; while (t.parentTexture) { t = t.parentTexture; hops++; } a = t.buffer ? (id<MTLResource>)t.buffer : (id<MTLResource>)t; }
    pthread_mutex_lock(&g_res_mu); int ok = [g_res_objects containsObject:a]; pthread_mutex_unlock(&g_res_mu);
    if (!ok) { nbad++; id<MTLTexture> t = is_tex ? (id<MTLTexture>)r : nil;
        MLOG("【红】驻留缺失 #%d：%s 槽 %d %s %p（父链 %d 层）分配 %lu B 存储 %lu%s", nbad, what, reg, is_tex ? "贴图" : "缓冲", (__bridge void *)a, hops, (unsigned long)a.allocatedSize, (unsigned long)a.storageMode,
             t ? [NSString stringWithFormat:@" 格式 %lu %lux%lu 类型 %lu 用途 %#lx", (unsigned long)t.pixelFormat, (unsigned long)t.width, (unsigned long)t.height, (unsigned long)t.textureType, (unsigned long)t.usage].UTF8String : ""); }
}
static void res_commit(void) {
    pthread_mutex_lock(&g_res_mu);
    if (g_res_dirty) { [g_res commit]; g_res_dirty = 0; }
    pthread_mutex_unlock(&g_res_mu);
}

@interface NvDeferredRelease : NSObject { @public id object; uint64_t fence; int keep_resident; } @end
@implementation NvDeferredRelease @end
static NSMutableArray<NvDeferredRelease *> *g_deferred;
static pthread_mutex_t g_deferred_mu = PTHREAD_MUTEX_INITIALIZER;
// Deferred release has two meanings that must be kept apart (the root of the GPU memory fault on the first frame on entering Travincal):
//   defer_obj_release = this object is retired: drop the reference after the submissions that used it complete, and remove it from the residency set;
//   defer_obj_unref   = only the reference I hold is no longer wanted, the object itself may still be in use (the view cache of a texture ID may hold the base texture itself): only drop the reference, leave the residency set alone.
// Previously only the former existed: when the view cache was invalidated, a texture that 'is still the current storage of some serial' was also removed from the residency set. Invalidation used to happen only when the base texture was really replaced, so it was never hit;
// after the transparency chain's in-place composite swapped the storage of two textures, each of the two live textures was removed once this way, and afterwards they are used as attachments and sampled every frame as usual - the GPU accesses non-resident memory.
static void defer_obj_release_ex(id obj, int keep_resident) {
    if (!obj) return;
    NvDeferredRelease *d = [NvDeferredRelease new]; d->object = obj; d->fence = g_fence + 1; d->keep_resident = keep_resident;
    pthread_mutex_lock(&g_deferred_mu);
    if (!g_deferred) g_deferred = [NSMutableArray array];
    [g_deferred addObject:d];
    pthread_mutex_unlock(&g_deferred_mu);
}
static void defer_obj_release(id obj) { defer_obj_release_ex(obj, 0); }
static void defer_obj_unref(id obj) { defer_obj_release_ex(obj, 1); }
static void defer_drain(void) {
    NSMutableArray<NvDeferredRelease *> *ready = [NSMutableArray array];
    pthread_mutex_lock(&g_deferred_mu);
    for (NSInteger i = g_deferred.count - 1; i >= 0; i--) {
        NvDeferredRelease *d = g_deferred[(NSUInteger)i];
        if (d->fence <= g_done) { [ready addObject:d]; [g_deferred removeObjectAtIndex:(NSUInteger)i]; }
    }
    pthread_mutex_unlock(&g_deferred_mu);
    for (NvDeferredRelease *d in ready) {
        id obj = d->object;
        if (!d->keep_resident && [obj conformsToProtocol:@protocol(MTLResource)]) {
            pthread_mutex_lock(&g_res_mu);
            int registered = [g_res_objects containsObject:obj];
            pthread_mutex_unlock(&g_res_mu);
            if (registered) res_remove((id<MTLResource>)obj);
        }
        d->object = nil;
    }
}

static pthread_mutex_t g_bf_want_mu = PTHREAD_MUTEX_INITIALIZER;
static char **g_bf_want_keys; static size_t g_bf_want_n, g_bf_want_cap; static int g_bf_want_loaded;
void bf_want_note(const char *key) {
    const char *path = getenv("DM_NVN_BF_WANT");
    if (!path || !key || strlen(key) != 16) return;
    for (int i = 0; i < 16; i++) if (!((key[i] >= '0' && key[i] <= '9') || (key[i] >= 'a' && key[i] <= 'f') || (key[i] >= 'A' && key[i] <= 'F'))) return;
    pthread_mutex_lock(&g_bf_want_mu);
    if (!g_bf_want_loaded) {
        FILE *f = fopen(path, "r"); char line[64];
        if (f) { while (fgets(line, sizeof line, f)) { char *e = strpbrk(line, "\r\n"); if (e) *e = 0; if (strlen(line) != 16) continue;
                if (g_bf_want_n == g_bf_want_cap) { g_bf_want_cap = g_bf_want_cap ? g_bf_want_cap * 2 : 64; g_bf_want_keys = realloc(g_bf_want_keys, g_bf_want_cap * sizeof *g_bf_want_keys); }
                g_bf_want_keys[g_bf_want_n++] = strdup(line); } fclose(f); }
        g_bf_want_loaded = 1;
    }
    for (size_t i = 0; i < g_bf_want_n; i++) if (!strcmp(g_bf_want_keys[i], key)) { pthread_mutex_unlock(&g_bf_want_mu); return; }
    if (g_bf_want_n == g_bf_want_cap) { g_bf_want_cap = g_bf_want_cap ? g_bf_want_cap * 2 : 64; g_bf_want_keys = realloc(g_bf_want_keys, g_bf_want_cap * sizeof *g_bf_want_keys); }
    g_bf_want_keys[g_bf_want_n++] = strdup(key);
    FILE *f = fopen(path, "a"); if (f) { fprintf(f, "%s\n", key); fclose(f); }
    pthread_mutex_unlock(&g_bf_want_mu);
}

// ---------- Memory pools ----------
// DM_POOL_CENSUS=1 (diagnostic): make the guest memory pools attributable. The registry only knew [lo, hi) and whether the no-copy MTLBuffer is in the residency set,
// so nothing could say which pool carries the 2.2 GiB guest resident heap, which pools are wired only because a texture view reads them, or which pools exist only as the
// uncompressed source of a packaged (block-linear) texture that was already copied into a GPU-private texture.
// Per pool this keeps cheap per-hit bookkeeping under g_mu (address hit count and last hit frame, first hit frame / guest offset / caller, guest offset range) plus the
// NVN flags and pool object address needed for texture attribution. With the switch off an address hit is one integer increment and one branch; the table walks in
// pool_census only run with the switch on.
typedef struct {
    uintptr_t lo, hi; uintptr_t base;
    __unsafe_unretained id<MTLBuffer> buf;
    uint64_t gpu;
    const void *key; int res;                 // res: already in the residency set (on demand, see pool_res)
    uint32_t flags;                           // NVNmemoryPool flags from nvnMemoryPoolInitialize
    uint64_t size, buflen;                    // guest size requested / MTLBuffer length (page-expanded)
    const void *obj;                          // NVNmemoryPool object address (tex_t.pool points at it); NULL when unknown
    uint64_t hits, last_hit_frame;            // address lookups that resolved to this pool / frame of the last one
    uint64_t first_frame, first_off;          // first address use: frame and guest offset
    uintptr_t first_site; uint8_t first_kind; // first address use: caller return address; 0 = mtl_addr (CPU side), 1 = mtl_gpu_addr (GPU address)
    uint64_t min_off, max_off;                // guest offset range ever addressed by the GPU
    // T5 static-source reclaim state (see "T5: packaged-texture source page reclaim" below); kept in this struct so the census can report it.
    uint8_t rc_state;                         // RC_* state of this pool
    uint8_t rc_no_reclaim;                    // permanent: invalidated after a reclaim, or rejected for a hard reason
    uint8_t rc_queued;                        // a reclaim evaluation is pending for this pool
    uint8_t rc_why;                           // last rejection reason (RC_R_*), for the dry-run aggregate
    uint8_t rc_rejected;                      // this pool has been counted once in the cumulative rejection counters
    uint64_t rc_hits_at_upload;               // pool.hits at the last source upload of one of its textures
    uint64_t rc_last_upload_frame;            // frame of that upload
    uint64_t rc_stamp;                        // nvn_dirty_now() when the pool's source pages were reclaimed
    uint64_t rc_bytes;                        // guest bytes actually reclaimed
    uint64_t rc_next_eval;                    // next frame at which this pool may be evaluated again (throttle)
    uint64_t rc_up_gen, rc_done_gen;          // upload batches started for this pool / completed on the GPU: reclaim waits until they are equal
    uint64_t rc_nontex_hits;                  // address uses that were not the texture-source upload (buffer bind / copy / share view)
    uint8_t rc_up_state;                      // upload record: bit0 dynamic descriptor, bit1 serial re-uploaded, bit2 record full
    uint8_t rc_up_nser, rc_up_nrg;            // distinct uploaded serials / distinct source ranges recorded
    uint64_t rc_up_bytes;                     // sum of source bytes seen for this pool (diagnostic)
    uint64_t rc_up_ser[12];                   // distinct uploaded texture serials (re-upload detection)
    struct { uintptr_t mem; uint64_t size, stamp; } rc_up_rg[12];   // distinct source ranges + content stamp at upload
} pool_ent;
static pool_ent *g_pools; static int g_npool, g_cap;
static int g_pidx_ok, g_plast = -1;   // whether the address→pool index is valid, the last hit (pool_find)
// A per-thread cache of the pool lookups. Every address use (a texture upload, a buffer update, a draw's vertex buffer) went through
// the global pool lock and the pool index search. The game resolves the same few pools over and over (the uniform buffers, the
// render targets, the constant pool), so a small per-thread cache of the last few hits takes the lock and the search off that path.
// g_pool_gen changes whenever a pool is added or removed: a stale entry is then ignored.
static _Atomic uint64_t g_pool_gen = 1;
typedef struct { uint64_t lo, hi, base, gpu; __unsafe_unretained id<MTLBuffer> buf; int idx; } pc_ent;
static __thread struct { uint64_t gen; unsigned n, next; pc_ent e[8]; } t_pc;
static void pc_put(int idx, uint64_t base, uint64_t gpu) {
    uint64_t gen = atomic_load_explicit(&g_pool_gen, memory_order_relaxed);
    if (t_pc.gen != gen) { t_pc.gen = gen; t_pc.n = 0; t_pc.next = 0; }
    unsigned k = t_pc.next++ & 7;   // round robin: the eight entries hold the most recent pools
    t_pc.e[k].idx = idx; t_pc.e[k].lo = g_pools[idx].lo; t_pc.e[k].hi = g_pools[idx].hi; t_pc.e[k].base = base; t_pc.e[k].gpu = gpu;
    t_pc.e[k].buf = g_pools[idx].buf;   // the buffer is kept in the entry, so a hit does not read the pool table
    if (t_pc.n < 8) t_pc.n++;
}
static uint64_t g_pc_hit, g_pc_miss;   // the pool cache hits and misses
static int pc_get(uint64_t addr, uint64_t len) {   // the cache slot, or -1
    if (t_pc.gen != atomic_load_explicit(&g_pool_gen, memory_order_relaxed)) return -1;
    for (unsigned i = 0; i < t_pc.n; i++) if (addr >= t_pc.e[i].lo && addr + len <= t_pc.e[i].hi) { g_pc_hit++; return (int)i; }
    g_pc_miss++;
    return -1;
}
static NSMutableArray *g_keep;                  // Holds the buffer references (pool_ent has non-owning pointers)
static uint64_t g_pool_adds, g_pool_dels;        // cumulative pool creations / destructions (DM_POOL_CENSUS per-census delta)
static uint64_t g_pool_add_bytes, g_pool_del_bytes;   // cumulative guest bytes created / destroyed
static int g_pool_res_n;   // registered pools already in the residency set (for the memory readout; under g_mu)
static int g_pool_census_on;   // DM_POOL_CENSUS, read once in mtl_init before any pool is added
static int g_pool_census_prints;
static int pool_renew_on(void);   // D1 pool page renew switch (defined with mtl_pool_del; read in mtl_init already)
static _Atomic uint64_t g_cur_frame;   // the frame counter (defined below with the timing state): the first address hit records the frame
// ================= T5: static packaged-texture source page reclaim =================
// The port keeps packaged (has_layout) textures twice: the block-linear source bytes in the game's NVN memory pool, and a GPU-private MTLTexture
// built from them by pkg_upload_. After the upload batch that consumed a source pool the guest copy is dead: everything samples the private copy;
// the pool is referenced again only by a re-upload of the same storage (suppressed below) or by an engine buffer binding (an address hit, which
// disqualifies the pool). The source pools are wired into the MTLResidencySet by the on-demand residency path, so they also pin guest pages.
// Reclaim = evict the pool's residency and drop its guest pages. What actually returns pages on Darwin is NOT madvise(MADV_DONTNEED) (a no-op on
// anonymous memory there): mmap(MAP_FIXED|MAP_ANON) over the aligned interior range replaces the pages with fresh zero pages and the footprint drops.
// A no-copy MTLBuffer kept across such a remap keeps a stale GPU mapping (probed), so the buffer is released first and recreated on demand if the
// pool is ever addressed again (rc_unreclaim_pool through pool_res).
// Safety: CPU_NO_ACCESS pools only (the game cannot dereference them); pool storage inside the guest heap (dm_guest_heap_range: alias/stack pages
// must never be replaced); pages fully covered by the union of eligible registered-texture ranges, never boundary pages; only after the upload
// command buffers completed, with no address hit since the upload and a grace window; every later write/use invalidates the pool.
// Env: DM_TEX_RECLAIM=1 enables, DM_TEX_RECLAIM_DRYRUN=1 only reports. Default off, so every other game is unaffected.
enum { RC_NONE = 0, RC_WAIT, RC_EVICTING, RC_RECLAIMED };
enum { RC_R_NONE = 0, RC_R_HIT, RC_R_DIRTY, RC_R_COVER, RC_R_ALIAS, RC_R_OVERLAP, RC_R_NOTREADY, RC_R_HEAP, RC_R_BUDGET, RC_R_WRITEUSE,
       RC_R_ALIGN, RC_R_UNTRACKED, RC_R_REUPLOAD, RC_R_OBJ, RC_R_N };
static int g_tex_reclaim, g_tex_reclaim_dry;                       // env switches, read once in mtl_init
static int g_rc_min_mb = 8, g_rc_purity = 90, g_rc_backlog_mb = 64, g_rc_grace = 0, g_rc_max_pools = 16, g_rc_max_mb = 128;
static int g_rc_view_idle = 300;   // the frames a zero-copy view must go unused before the reclaim may take the pool behind it (0 = never)
static uint64_t g_rc_views_dropped;   // the zero-copy views dropped with their pool so far
// The per-tick cap actually used: with a big backlog of candidate pools the cap is raised, because the fixed 16-pool / 128 MB cap retires
// the loading phase's 260 pools too slowly to keep the peak resident under the gate (the cap still bounds the work of a single tick).
static int g_rc_max_pools_eff = 16; static uint64_t g_rc_max_mb_eff = 128u << 20;   // grace: frames after the upload feedback; 0 = reclaim inside the same frame (primary path)
static uint64_t g_rc_budget_mb = 1536;   // enough for a level's streamed sources (the city batch alone is ~570 MB on device); DM_TEX_RECLAIM_MB overrides
static uint64_t g_rc_pools, g_rc_bytes;                            // pools / guest bytes reclaimed so far (cumulative, for the readout)
// Bytes that are reclaimed RIGHT NOW: + when a pool's pages are replaced, - when that pool is finalized (its block goes back to the game's
// allocator, so the next level may reclaim again). The budget bounds this, not the cumulative total -- with the cumulative total the reclaim
// switched itself off for good once a session had reclaimed g_rc_budget_mb in sum (Hades II: the city batch alone is ~570 MB, so after two
// or three scene loads every later attempt hit RC_R_BUDGET and the memory climbed for the rest of the run with nothing reclaiming it).
static _Atomic uint64_t g_rc_live_bytes;
static uint64_t g_rc_cand_pools, g_rc_cand_bytes;                  // candidate pool appearances / source bytes seen (dry-run opportunity)
static uint64_t g_rc_invalid, g_rc_suppress_reup, g_rc_ticks, g_rc_tick_ns;
static uint64_t g_rc_block[RC_R_N];                                // rejections by reason, cumulative over the whole run (each pool is counted once, the first time it is rejected)
static uint64_t g_rc_events, g_rc_last_print_frame;
static uint64_t g_rc_acc_now_pools, g_rc_acc_now_bytes;   // in-batch would-reclaim bytes accumulated between prints (dry-run)
static uint64_t g_rc_batch_prints;                          // immediate in-batch dry-run lines (a device run dies inside the transition frame)
static uint64_t g_rc_log_seen[256]; static int g_rc_log_n;   // per-pool candidate log dedupe (hash of pool address + last upload frame)
static _Atomic int g_rc_tick_wanted;   // a finished upload batch is waiting for a submit-thread safe point (set from the feedback thread / up_wait)
static _Atomic int g_rc_addr_tx;       // 1 while pkg_upload_ resolves its source address: an address use then is the texture-source upload itself, not an engine use
// Reclaimed ranges for the lock-free "is this source gone?" checks on texture creation / readback paths. Entries are immutable once published;
// a finalized pool's entry is tombstoned by setting lo = hi. Appends happen under g_mu.
// Tombstoned entries are never reused, and the table was 1024 entries: a long session (each reclaim takes one entry, a level's batch is
// dozens of pools) filled it and then the reclaim refused every later pool for the rest of the run. 4096 pushes that wall far out; the
// scan is per texture creation, not per frame (16 bytes an entry, so the whole table is 64 KB).
#define RC_RG_MAX 4096
static struct { _Atomic uintptr_t lo, hi; } g_rc_rg[RC_RG_MAX];
static _Atomic uint64_t g_rc_rg_n;
static int rc_src_reclaimed(uintptr_t a, uint64_t n) {   // was any byte of [a, a+n) inside a reclaimed pool range?
    uint64_t m = atomic_load_explicit(&g_rc_rg_n, memory_order_acquire);
    if (!m) return 0;
    if (m > RC_RG_MAX) m = RC_RG_MAX;
    uintptr_t e = a + (n ? n : 1);
    for (uint64_t i = 0; i < m; i++) {
        uintptr_t lo = atomic_load_explicit(&g_rc_rg[i].lo, memory_order_relaxed), hi = atomic_load_explicit(&g_rc_rg[i].hi, memory_order_relaxed);
        if (lo < e && a < hi) return 1;
    }
    return 0;
}
// Candidate pools of the upload batch being built (submit thread only): deduped by pool storage address; handed to the feedback handler in up_flush.
typedef struct { uintptr_t lo; uint64_t size; uint64_t bytes; uint64_t gen; } rc_src;   // gen: pool upload generation when this batch flushed (feedback marks it completed)
static rc_src *g_up_rc; static int g_up_rc_n, g_up_rc_cap;
// Completed upload batches waiting for the submit thread to evaluate them (cross-thread: the feedback handler appends, up_wait drains).
static rc_src *g_rc_pending; static int g_rc_npend, g_rc_capp; static pthread_mutex_t g_rc_mu = PTHREAD_MUTEX_INITIALIZER;
static void rc_pending_add(const rc_src *e, int n) {
    if (n <= 0) return;
    pthread_mutex_lock(&g_rc_mu);
    if (g_rc_npend + n > g_rc_capp) { int nc = (g_rc_npend + n) * 2 + 16; rc_src *p = realloc(g_rc_pending, (size_t)nc * sizeof *p); if (p) { g_rc_pending = p; g_rc_capp = nc; } }
    for (int i = 0; i < n && g_rc_npend < g_rc_capp; i++) g_rc_pending[g_rc_npend++] = e[i];
    pthread_mutex_unlock(&g_rc_mu);
    atomic_store_explicit(&g_rc_tick_wanted, 1, memory_order_relaxed);   // the next submit-thread safe point evaluates this batch
}
// Write-use tracker: a texture used as a render-target attachment / copy destination / clear destination can change after its upload, so a pool that
// holds one is not a static source. serial -> use count, fixed open addressing; only touched when the feature is on.
#define RC_WU_MAX 8192
static _Atomic uint64_t g_rc_wu_key[RC_WU_MAX];
static _Atomic uint32_t g_rc_wu_cnt[RC_WU_MAX];
static void rc_write_use(uint64_t serial) {
    if (!serial || (!g_tex_reclaim && !g_tex_reclaim_dry)) return;
    uint64_t h = (serial * 0x9E3779B97F4A7C15ull) >> 40;
    for (unsigned k = 0; k < 64; k++) {
        uint64_t idx = (h + k) & (RC_WU_MAX - 1);
        uint64_t cur = atomic_load_explicit(&g_rc_wu_key[idx], memory_order_relaxed);
        if (cur == serial) { atomic_fetch_add_explicit(&g_rc_wu_cnt[idx], 1, memory_order_relaxed); return; }
        if (!cur) { uint64_t exp = 0; if (atomic_compare_exchange_strong(&g_rc_wu_key[idx], &exp, serial)) { atomic_fetch_add(&g_rc_wu_cnt[idx], 1); return; } if (exp == serial) { atomic_fetch_add(&g_rc_wu_cnt[idx], 1); return; } }
    }
}
static int rc_write_used(uint64_t serial) {
    if (!serial) return 0;
    uint64_t h = (serial * 0x9E3779B97F4A7C15ull) >> 40;
    for (unsigned k = 0; k < 64; k++) {
        uint64_t idx = (h + k) & (RC_WU_MAX - 1);
        uint64_t cur = atomic_load_explicit(&g_rc_wu_key[idx], memory_order_relaxed);
        if (!cur) return 0;
        if (cur == serial) return 1;
    }
    return 0;
}
static unsigned long long rc_footprint_mb(void) {
    task_vm_info_data_t vi; mach_msg_type_number_t c = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vi, &c) != KERN_SUCCESS) return 0;
    return (unsigned long long)(vi.phys_footprint >> 20);
}
static unsigned long long rc_rss_mb(void) {
    mach_task_basic_info_data_t bi; mach_msg_type_number_t c = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&bi, &c) != KERN_SUCCESS) return 0;
    return (unsigned long long)(bi.resident_size >> 20);
}
// DM_MADV_SELFTEST=1: settle on the actual machine which Darwin mechanism returns anonymous pages. madvise(MADV_DONTNEED) is a no-op on Darwin
// anonymous memory (measured on the Mac), so the reclaim path uses a MAP_FIXED remap; this line proves it on the device before reclaim is enabled.
static void rc_madv_selftest(void) {
    if (!getenv("DM_MADV_SELFTEST")) return;
    size_t pg = (size_t)getpagesize(), sz = 64u << 20;
    unsigned char *m = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (m == MAP_FAILED) { MLOG("内存机制自检：mmap 失败 errno=%d", errno); return; }
    memset(m, 0xAB, sz);
    unsigned long long r0 = rc_rss_mb(), f0 = rc_footprint_mb();
    int rc1 = madvise(m, sz / 2, MADV_DONTNEED);
    unsigned long long r1 = rc_rss_mb(), f1 = rc_footprint_mb();
    void *mr = mmap(m + sz / 4, sz / 2, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
    unsigned long long r2 = rc_rss_mb(), f2 = rc_footprint_mb();
    int zero_ok = m[sz / 4] == 0 && m[sz / 4 + pg] == 0;
    MLOG("内存机制自检（%zu MB，页 %zu 字节）：MADV_DONTNEED rc=%d 常驻 %llu→%llu MB footprint %llu→%llu MB；MAP_FIXED 替换 %zu MB 后 常驻 %llu MB footprint %llu MB，替换区读回零 %d（返回 %p）",
         sz >> 20, pg, rc1, r0, r1, f0, f1, (sz / 2) >> 20, r2, f2, zero_ok, mr);
    munmap(m, sz);
}
static void rc_env_init(void) {
    // Profile default (tex_reclaim off/dryrun/on = 0/1/2), then the environment: DM_TEX_RECLAIM=1/0 forces on/off,
    // DM_TEX_RECLAIM_DRYRUN=1 selects dry-run (device verification cycles, tests).
    int rcmode = DM_GAME_TEX_RECLAIM;
    { const char *er = getenv("DM_TEX_RECLAIM"); if (er) rcmode = atoi(er) ? 2 : 0; }
    { const char *ed = getenv("DM_TEX_RECLAIM_DRYRUN"); if (ed && atoi(ed)) rcmode = 1; }
    g_tex_reclaim = (rcmode == 2);
    g_tex_reclaim_dry = (rcmode == 1);
    const char *e;
    if ((e = getenv("DM_TEX_RECLAIM_MB")) && atoi(e) > 0) g_rc_budget_mb = (uint64_t)atoi(e);
    if ((e = getenv("DM_TEX_RECLAIM_MIN_MB")) && atoi(e) >= 0) g_rc_min_mb = atoi(e);
    if ((e = getenv("DM_TEX_RECLAIM_PURITY")) && atoi(e) >= 1 && atoi(e) <= 100) g_rc_purity = atoi(e);
    if ((e = getenv("DM_TEX_RECLAIM_BACKLOG_MB")) && atoi(e) > 0) g_rc_backlog_mb = atoi(e);
    if ((e = getenv("DM_TEX_RECLAIM_GRACE_FRAMES")) && atoi(e) >= 0) g_rc_grace = atoi(e);
    if ((e = getenv("DM_TEX_RECLAIM_MAX_POOLS")) && atoi(e) > 0) g_rc_max_pools = atoi(e);
    if ((e = getenv("DM_TEX_RECLAIM_MAX_MB")) && atoi(e) > 0) g_rc_max_mb = atoi(e);
    if (g_tex_reclaim || g_tex_reclaim_dry)
        MLOG("静态源回收：%s（预算 %llu MB，最小池 %d MB，纯度 %d%%，静默期 %d 帧，批次积压 %d MB）",
             g_tex_reclaim_dry && !g_tex_reclaim ? "试运行" : "开", (unsigned long long)g_rc_budget_mb, g_rc_min_mb, g_rc_purity, g_rc_grace, g_rc_backlog_mb);
}
static int rc_unreclaim_pool(int i);      // under g_mu: recreate the buffer of a reclaimed pool that is addressed again
static int rc_note_upload_src(const tex_t *t);   // remember a packaged-texture source pool for this batch (0 = not a candidate)
static void rc_reclaim_tick(void);        // submit thread: evaluate upload-batch candidates, reclaim when enabled
static void rc_reclaim_maybe(void);       // cheap per-pass gate for the replay safe points
// Residency / attribution class of a pool by its NVN flags: 0 = CPU cannot access it, 1 = CPU visible (uncached / cached), 2 = other / no CPU access bit
static int pool_class(uint32_t flags) {
    if (flags & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS) return 0;
    if (flags & (NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_CPU_CACHED)) return 1;
    return 2;
}

#define C1_BANK_SIZE (4u << 20)
#define C1_DECL_SIZE (64u << 10)
static NSMutableArray *g_c1_banks;
static size_t g_c1_bank_off, g_c1_alloc_bytes, g_c1_data_bytes;
static uint64_t g_c1_count, g_c1_exclusive;
static NSMutableSet<NSString *> *g_c1_full;
static int g_c1_full_valid;
static os_unfair_lock g_c1_lock = OS_UNFAIR_LOCK_INIT;

// DM_UDC_SELFTEST=1 (diagnostic, checks DontCare behavior on a real A19 device): private-storage Depth32Float_Stencil8 / RGBA8 (1952x900) are first cleared to write known values,
// then Load + DontCare without drawing, read back and compared pixel by pixel, printing one result line
static void udc_selftest(void) {
    if (!getenv("DM_UDC_SELFTEST")) return;
    id<MTLCommandQueue> q = [g_dev newCommandQueue];
    for (int depth = 1; depth >= 0; depth--) {
        NSUInteger w = 1952, h = 900;
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:depth ? MTLPixelFormatDepth32Float_Stencil8 : MTLPixelFormatRGBA8Unorm width:w height:h mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead; td.storageMode = MTLStorageModePrivate;
        id<MTLTexture> t = [g_dev newTextureWithDescriptor:td];
        id<MTLCommandBuffer> cb = [q commandBuffer];
        for (int pass = 0; pass < 2; pass++) {
            MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
            MTLLoadAction la = pass ? MTLLoadActionLoad : MTLLoadActionClear; MTLStoreAction sa = pass ? MTLStoreActionDontCare : MTLStoreActionStore;
            if (depth) { rp.depthAttachment.texture = t; rp.depthAttachment.loadAction = la; rp.depthAttachment.storeAction = sa; rp.depthAttachment.clearDepth = 0.3125;
                         rp.stencilAttachment.texture = t; rp.stencilAttachment.loadAction = la; rp.stencilAttachment.storeAction = sa; rp.stencilAttachment.clearStencil = 0x5a; }
            else { rp.colorAttachments[0].texture = t; rp.colorAttachments[0].loadAction = la; rp.colorAttachments[0].storeAction = sa; rp.colorAttachments[0].clearColor = MTLClearColorMake(0.2, 0.4, 0.6, 0.8); }
            [[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
        }
        id<MTLBuffer> out = [g_dev newBufferWithLength:w * h * 4 options:MTLResourceStorageModeShared];
        id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
        [b copyFromTexture:t sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(w, h, 1) toBuffer:out destinationOffset:0
            destinationBytesPerRow:w * 4 destinationBytesPerImage:w * h * 4 options:depth ? MTLBlitOptionDepthFromDepthStencil : MTLBlitOptionNone];
        [b endEncoding]; [cb commit]; [cb waitUntilCompleted];
        const uint32_t *px = out.contents; NSUInteger bad = 0; for (NSUInteger i = 0; i < w * h; i++) if (px[i] != px[0]) bad++;
        MLOG("UDC自检 %s %s %lux%lu 首值 %08x 与首值不同 %lu / %lu", g_dev.name.UTF8String, depth ? "Depth32F_S8" : "RGBA8", (unsigned long)w, (unsigned long)h, px[0], (unsigned long)bad, (unsigned long)(w * h));
    }
}
void mtl_init(void) {
    if (g_dev) return;
    rc_env_init();   // T5: read the reclaim switches before any pool / texture exists
    pool_renew_on();   // D1: likewise, so the log says whether the pool page renew is on before the first pool is finalized
    g_pool_census_on = getenv("DM_POOL_CENSUS") != NULL;   // read here, before the game creates any memory pool: the per-hit statistics and table walks stay off when unset
    const char *link = getenv("DM_PRESENT_LINK");
    g_dm_present_link = link ? atoi(link) : 1; if (g_dm_present_link < 0 || g_dm_present_link > 2) g_dm_present_link = 1;   // Default 1 (real-device same-process alternation: when load exceeds one display-link period, long frames 15% → 0 and frame intervals even out; no difference from 0 at a steady 60); DM_PRESENT_LINK=0 turns it off   // 1 = present on every tick of a 60 Hz display link; 2 = 120 Hz display link, the game presents a frame every other display-link tick (when behind, it can present on the next 8.33 ms display-link tick)
    const char *latency = getenv("DM_PRESENT_LATENCY");
    if (latency && atoi(latency) > 0) g_present_latency = atoi(latency);
    g_res_check_on = getenv("DM_RES_CHECK") != NULL;
    g_dev = MTLCreateSystemDefaultDevice();
    if (!g_dev) MDIE("没有 Metal 设备");
    if (!g_dev.hasUnifiedMemory) MDIE("需要统一内存设备（零拷贝内存池）");
    g_q = [g_dev newCommandQueue];
    MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
    rd.initialCapacity = 4096;
    NSError *re = nil;
    g_res = [g_dev newResidencySetWithDescriptor:rd error:&re];
    if (!g_res) MDIE("建立 MTLResidencySet 失败");
    [g_q addResidencySet:g_res];
    { void crash_set_wait_dump(void (*fn)(void)); void mtl_wait_state_dump(void); crash_set_wait_dump(mtl_wait_state_dump); }
    dispatch_queue_attr_t feedback_attr = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0);
    g_feedback_q = dispatch_queue_create("dm.nvn.mtl4-feedback", feedback_attr);
    MTL4CommandQueueDescriptor *q4d = [MTL4CommandQueueDescriptor new]; q4d.feedbackQueue = g_feedback_q;
    g_q4 = [g_dev newMTL4CommandQueueWithDescriptor:q4d error:&re];
    if (!g_q4) MDIE("建立 MTL4CommandQueue 失败");
    [g_q4 addResidencySet:g_res];
    g_alloc_n = 3;
    for (int i = 0; i < 3; i++) {
        MTL4CommandAllocatorDescriptor *ad = [MTL4CommandAllocatorDescriptor new];
        ad.label = [NSString stringWithFormat:@"nvn-mtl allocator %d", i];
        NSError *ae = nil;
        g_alloc4[i] = [g_dev newCommandAllocatorWithDescriptor:ad error:&ae];
        if (!g_alloc4[i]) MDIE("建立 MTL4CommandAllocator[%d] 失败：%s", i, ae.localizedDescription.UTF8String);
        g_up_alloc4[i] = [g_dev newCommandAllocatorWithDescriptor:ad error:&ae];
        if (!g_up_alloc4[i]) MDIE("建立上传 MTL4CommandAllocator[%d] 失败：%s", i, ae.localizedDescription.UTF8String);
        MTL4CommandAllocatorDescriptor *pad = [MTL4CommandAllocatorDescriptor new];
        pad.label = [NSString stringWithFormat:@"nvn-mtl present allocator %d", i];
        g_present_alloc4[i] = [g_dev newCommandAllocatorWithDescriptor:pad error:&ae];
        if (!g_present_alloc4[i]) MDIE("建立出图 MTL4CommandAllocator[%d] 失败：%s", i, ae.localizedDescription.UTF8String);
        g_present_cb[i] = [g_dev newCommandBuffer];
        if (!g_present_cb[i]) MDIE("建立出图 MTL4CommandBuffer[%d] 失败", i);
    }
    MTL4ArgumentTableDescriptor *atd = [MTL4ArgumentTableDescriptor new];
    atd.maxBufferBindCount = 31; atd.maxTextureBindCount = 1;
    NSError *ate = nil;
    g_vt = [g_dev newArgumentTableWithDescriptor:atd error:&ate]; if (!g_vt) MDIE("建立 vertex argument table 失败：%s", ate.localizedDescription.UTF8String);
    g_ft = [g_dev newArgumentTableWithDescriptor:atd error:&ate]; if (!g_ft) MDIE("建立 fragment argument table 失败：%s", ate.localizedDescription.UTF8String);
    g_ct = [g_dev newArgumentTableWithDescriptor:atd error:&ate]; if (!g_ct) MDIE("建立 compute argument table 失败：%s", ate.localizedDescription.UTF8String);
    g_up_ct = [g_dev newArgumentTableWithDescriptor:atd error:&ate]; if (!g_up_ct) MDIE("建立上传 compute argument table 失败：%s", ate.localizedDescription.UTF8String);
    MTL4ArgumentTableDescriptor *pad = [MTL4ArgumentTableDescriptor new];
    pad.maxTextureBindCount = 1; pad.maxSamplerStateBindCount = 1; pad.maxBufferBindCount = 1;
    g_present_args = [g_dev newArgumentTableWithDescriptor:pad error:&ate];
    if (!g_present_args) MDIE("建立出图 argument table 失败：%s", ate.localizedDescription.UTF8String);
    g_bloomq_args = [g_dev newArgumentTableWithDescriptor:pad error:&ate];
    if (!g_bloomq_args) MDIE("建立泛光 1/4 argument table 失败：%s", ate.localizedDescription.UTF8String);
    for (int i = 0; i < 3; i++) {
        g_present_uv[i] = [g_dev newBufferWithLength:sizeof(float) * 8 options:MTLResourceStorageModeShared];
        if (!g_present_uv[i]) MDIE("建立出图 UV 缓冲[%d] 失败", i);
        res_add(g_present_uv[i]);
        g_bloomq_const[i] = [g_dev newBufferWithLength:sizeof(float) * 8 options:MTLResourceStorageModeShared];
        if (!g_bloomq_const[i]) MDIE("建立泛光 1/4 常量缓冲[%d] 失败", i);
        res_add(g_bloomq_const[i]);
    }
    g_ev = [g_dev newSharedEvent];
    g_keep = [NSMutableArray array];
    g_nvn_deswz_gpu = nvn_deswz_gpu_init(g_dev) == 0;
    dm_res_report_register(mtl_res_counts);
    if (!g_nvn_deswz_gpu) MLOG("【红】显卡解排布初始化失败，预排布贴图回退 CPU");
    MLOG("Metal 设备 %s", g_dev.name.UTF8String);
    rc_madv_selftest();   // T5: which pages-return mechanism works on this machine (DM_MADV_SELFTEST=1)
    udc_selftest();
}

void dm_mtl_l3_tick(uint32_t tick) {
    static int leak = -1;
    if (leak < 0) leak = getenv("DM_TEST_LEAK_GPU") != NULL;
    if (leak && tick % 100 == 0) {
        if (!g_l3_gpu_leaks) g_l3_gpu_leaks = [NSMutableArray array];
        id<MTLBuffer> leak = [g_dev newBufferWithLength:1 << 20 options:MTLResourceStorageModeShared];
        if (leak) [g_l3_gpu_leaks addObject:leak];
    }
}

uint64_t dm_mtl_l3_sample(uint32_t tick) {
    (void)tick;
    return g_dev ? g_dev.currentAllocatedSize : 0;
}

// No-copy buffers of finalized pools (weak references, under g_mu): the memory report counts how many are still alive and their length -- a buffer that outlives its pool is still strongly referenced by something else (a texture built directly on the pool, etc.)
static NSHashTable *g_dead_pools;
// D1 probe: what a pool buffer looked like at the moment its pool was deleted (guest address, size, whether the pages were renewed, and how
// many references it still had once the host dropped its own). A surviving dead buffer whose "delete-time external references" is 0 got its
// holders AFTER the delete, which points at something entirely different from a holder present all along. Keyed weakly by the buffer, so
// this probe never keeps a pool buffer alive itself.
@interface NvDeadMeta : NSObject { @public uintptr_t lo; uint64_t size, frame, hits; unsigned long rc; int renewed, was_res; } @end
@implementation NvDeadMeta @end
static NSMapTable *g_dead_meta;   // weak buffer -> NvDeadMeta (under g_mu)
// D1 probe: follow a few dead pool buffers through the phases of every frame, so the moment their reference count climbs can be pinned to a
// part of the frame (before the submission / after it / before the present). A line prints only when a count CHANGES, so the log grows with
// the event, not with the frame rate. The entries are weak: watching must not be the thing that keeps a buffer alive.
#define DWATCH_N 3
static __weak id g_dwatch[DWATCH_N];
static unsigned long g_dwatch_rc[DWATCH_N];
static int g_dwatch_n, g_dwatch_prints;
// Taken in mtl_pool_del right after the pages were renewed: the first few buffers of the run are followed from the moment they become dead,
// which is the only window in which the climb can be attributed to a phase of the frame rather than to "somewhere in the level load".
static void dead_watch_note(id b) {
    if (!b || g_dwatch_n >= DWATCH_N) return;
    g_dwatch[g_dwatch_n] = b; g_dwatch_rc[g_dwatch_n] = 0; g_dwatch_n++;
}
static void dead_watch(const char *where) {
    if (!g_dwatch_n || g_dwatch_prints >= 60) return;
    unsigned long long f = atomic_load_explicit(&g_cur_frame, memory_order_relaxed);
    @autoreleasepool {
        for (int i = 0; i < g_dwatch_n; i++) {
            id b = g_dwatch[i]; if (!b) continue;   // reading a __weak hands back +1 (retained + autoreleased); that one is subtracted below
            unsigned long rc = (unsigned long)CFGetRetainCount((__bridge CFTypeRef)b) - 1;
            if (rc != g_dwatch_rc[i]) { MLOG("死缓冲观察：%s 帧 %llu 缓冲 %p 计数 %lu → %lu", where, f, (__bridge void *)b, g_dwatch_rc[i], rc); g_dwatch_rc[i] = rc; g_dwatch_prints++; }
        }
    }
}
static void dead_meta_put(id b, uintptr_t lo, uint64_t size, int renewed, unsigned long rc, int was_res, uint64_t hits) {
    if (!b) return;
    if (!g_dead_meta) g_dead_meta = [NSMapTable weakToStrongObjectsMapTable];
    NvDeadMeta *m = [NvDeadMeta new];
    m->lo = lo; m->size = size; m->frame = atomic_load_explicit(&g_cur_frame, memory_order_relaxed); m->rc = rc; m->renewed = renewed;
    m->was_res = was_res; m->hits = hits;
    [g_dead_meta setObject:m forKey:b];
}
static _Atomic uint64_t g_reg_gen = 1;   // Change generation of memory pool / texture / sampler registration: one of the invalidation conditions for argument buffer reuse
// On-demand residency of memory pools (from d2r-mobile e2a0c9a9b): putting a zero-copy buffer into the residency set makes iOS read the whole pool into memory and pin it as wired, even a pool the game never uses
// (they measured on a 17 Pro Max a 512 MB render-target pool addressed 0 times in a run, with about 530 MB more device wired memory; a 4 GB A14 was ended by the system within seconds of launch because of it).
// So a pool is not put into the residency set when created; it is put in, and the residency set committed on the spot, the first time the GPU gets its address through mtl_addr / mtl_gpu_addr -- these are the only two ways the GPU gets pool addresses,
// and committing on the spot is earlier than any command buffer commit that uses it. CPU-side reads of pool contents through mtl_addr trigger it too (conservative, never misses). Caller holds g_mu
static int pool_res(int i) {   // 1 = the pool has a usable buffer in the residency set
    if (g_pools[i].res) return 1;
    if (g_pools[i].rc_state == RC_RECLAIMED) {   // T5: addressed again after its source pages were reclaimed; the restore already re-added the residency
        if (!rc_unreclaim_pool(i)) return 0;
        return 1;
    }
    if (!g_pools[i].buf) return 0;
    res_add(g_pools[i].buf); res_commit(); g_pools[i].res = 1; g_pool_res_n++;
    return 1;
}
// DM_POOL_CENSUS: record one address use of pool i (the caller of mtl_addr / mtl_gpu_addr passes its own return address). All of it is a few integer writes under g_mu;
// off by default it is one branch. min_off / max_off give the guest byte range ever addressed (enough to tell "whole pool read at once" from "a few bytes touched").
// Caller holds g_mu.
static void pool_note_hit(int i, uint64_t addr, uint64_t len, int kind, uintptr_t site) {
    pool_ent *p = &g_pools[i];
    // T5: the cheap fields are always recorded -- static-source reclaim treats any address use after an upload as a hard disqualifier and must
    // not depend on the census switch. An address use that is not the texture-source upload itself (pkg_upload_) also disqualifies the pool.
    p->hits++;
    p->last_hit_frame = atomic_load(&g_cur_frame);
    if (!atomic_load_explicit(&g_rc_addr_tx, memory_order_relaxed)) p->rc_nontex_hits++;
    if (!g_pool_census_on) return;
    uint64_t off = addr - p->lo, end = off + (len ? len : 1);
    if (!p->first_site) { p->first_site = site; p->first_frame = p->last_hit_frame; p->first_off = off; p->first_kind = (uint8_t)kind; }
    if (off < p->min_off) p->min_off = off;
    if (end > p->max_off) p->max_off = end;
}
void mtl_pool_add_ex(const void *mem, uint64_t size, uint32_t flags, const void *obj) {
    mtl_init(); g_reg_gen++;
    uintptr_t pg = (uintptr_t)getpagesize(), lo = (uintptr_t)mem, hi = lo + size;
    uintptr_t a = lo & ~(pg - 1), b = (hi + pg - 1) & ~(pg - 1);
    id<MTLBuffer> buf = [g_dev newBufferWithBytesNoCopy:(void *)a length:b - a options:MTLResourceStorageModeShared deallocator:nil];
    if (!buf) MDIE("内存池 %p+%#llx 建零拷贝缓冲失败", mem, (unsigned long long)size);
    pthread_mutex_lock(&g_mu);
    if (g_npool == g_cap) { g_cap = g_cap ? g_cap * 2 : 64; g_pools = realloc(g_pools, g_cap * sizeof *g_pools); }
    g_pidx_ok = 0; g_plast = -1;   // the address index is invalid (pool_find)
    atomic_fetch_add_explicit(&g_pool_gen, 1, memory_order_relaxed);   // the per-thread pool caches are stale
    pool_ent *e = &g_pools[g_npool++];
    memset(e, 0, sizeof *e);
    e->lo = lo; e->hi = hi; e->base = a; e->buf = buf; e->gpu = buf.gpuAddress; e->key = mem;   // not in the residency set yet: put in the first time it is addressed (pool_res)
    e->flags = flags; e->size = size; e->buflen = b - a; e->obj = obj; e->min_off = ~(uint64_t)0;
    e->rc_state = RC_NONE; e->rc_queued = 0; e->rc_up_gen = e->rc_done_gen = 0;   // T5: no reclaim bookkeeping for a fresh pool (memset already covers this)
    g_pool_adds++; g_pool_add_bytes += size;
    [g_keep addObject:buf];
    pthread_mutex_unlock(&g_mu);
}
void mtl_pool_add(const void *mem, uint64_t size) { mtl_pool_add_ex(mem, size, 0, NULL); }   // unit tests / callers without NVN pool flags

// D1: pool page renew on Finalize. A zero-copy MTLBuffer is built on the guest's pool pages, and Metal keeps the buffer (and its GPU
// pages) alive for as long as those pages stay mapped -- releasing every host reference is not enough (device: 558 dead pool buffers,
// none with a host holder). The guest never unmaps: the heap is one mmap and a heap shrink is only mprotect. So each "enter save ->
// quit to menu" left ~119 dead buffers (~174 MB of Metal memory) behind for good, and the next session's pools at the same guest
// addresses overlapped the stale GPU pages. Both the reclaim remap (17 pools) and the game's own exit frame showed the dead count
// falling exactly when the pages' mapping changed. So on Finalize the pool's interior pages (fully inside the pool storage; boundary
// pages may hold a neighbour) are replaced with fresh anonymous pages and the bytes are copied across: the guest sees the same
// contents (its allocator may keep free-list data inside the block), and Metal's old pages are orphaned and freed with the buffer.
// Only guest heap memory that is open read-write (dm_guest_renewable; alias, stack and module pages are never touched), never pages
// that another live pool's buffer covers, and never a reclaimed pool (its pages were already replaced). Views still queued for release
// do not block the remap (the reclaim has always remapped with them pending): after the GPU wait no submitted command reads the pages,
// the bytes are kept, and the buffer dies when its own fence passes.
// Profile key pool_renew (on / off), env DM_POOL_RENEW overrides: 1 = renew with the bytes copied back, 2 = renew and leave the pages
// zeroed (MEASUREMENT ONLY: the game's allocator may keep free-list links inside a pool, so this can corrupt it; it exists to bound
// how much of the win is the copy and how much is the remap).
static int g_pool_renew = -1;
static _Atomic uint64_t g_renew_pools, g_renew_bytes, g_renew_ns, g_renew_skip_mem, g_renew_skip_overlap, g_renew_skip_gpu, g_renew_fail;
// Why a finalized pool was NOT renewed, for the pools that never even become a candidate: a reclaimed pool has no buffer (T5), and a pool
// with no interior page (e <= a) has nothing whole to replace. Neither is a "skip" the gate counters can show, and together they are the
// difference between 删池 and 换页 in the burst line -- without them "删池 175、换页 162" could be read as 13 pools stopped by a gate.
static _Atomic uint64_t g_renew_nobuf_rc, g_renew_nobuf, g_renew_skip_tiny;
// D1 probe: ONE line per burst of pool finalizes. The per-pool line above is useless for reading a session (the game finalizes ~143 pools in
// one frame on the way back to the menu, so a burst is 143 lines and they scroll away); this summarizes the burst instead, with the dead-buffer
// count before and after it -- which is the number the whole fix is judged on. Printed from the frame hook (mtl_mem_stat), one frame after
// the last delete of the burst, so a burst that spans a few frames still comes out as one line.
static _Atomic uint64_t g_d1_del;         // pools deleted since the last report
static _Atomic uint64_t g_d1_renew;       // ... of those, the ones whose pages were renewed
static void dead_holder_report(id<MTLBuffer> b);   // defined with the texture-ID registry below
static _Atomic uint64_t g_drop_died, g_drop_alive;   // D1 probe: what the buffer did the instant the host let go -- deallocated, or still held by someone else
static _Atomic uint64_t g_del_hits[5];   // D1 probe: deleted pools by address-hit count at delete (0 / 1 / 2-4 / 5-16 / >16); survivors all sit at 0-1
static _Atomic uint64_t g_renew_droponly;   // DM_POOL_RENEW_DROP: pools released without the page remap (measurement)
static _Atomic uint64_t g_d1_renew_b;     // bytes renewed since the last report
static _Atomic uint64_t g_d1_frame;       // frame of the last delete
static uint64_t g_d1_skip_mem0, g_d1_skip_ov0, g_d1_skip_gpu0, g_d1_fail0;   // reporter-side snapshots, for the deltas of the cumulative counters
static uint64_t g_d1_nobuf0, g_d1_nobuf_rc0, g_d1_tiny0;   // ... and of the three "never a candidate" counters
static long g_d1_dead0 = -1;              // dead-buffer count at the previous report (-1 = no previous report)
static int pool_renew_full(void) {   // DM_POOL_RENEW_FULL (measurement only, see the call site in mtl_pool_del)
    static int v = -1;
    if (v < 0) { const char *e = getenv("DM_POOL_RENEW_FULL"); v = e && *e && strcmp(e, "0") ? 1 : 0; }
    return v;
}
static int pool_renew_droponly(void) {   // DM_POOL_RENEW_DROP=1 (measurement): drop every host reference but leave the pages alone.
    static int v = -1;                     // Tells "the remap is what frees the buffer" from "the reference drop alone frees it".
    if (v < 0) { const char *e = getenv("DM_POOL_RENEW_DROP"); v = e && *e && strcmp(e, "0") ? 1 : 0; }
    return v;
}
static int pool_purge_mode(void) {   // DM_POOL_PURGE=1: set the buffer Empty before dropping it; 2: Volatile. (measurement)
    static int v = -1;
    if (v < 0) { const char *e = getenv("DM_POOL_PURGE"); v = e ? atoi(e) : 0; }
    return v;
}
static int pool_renew_on(void) {
    if (g_pool_renew < 0) {
        const char *e = getenv("DM_POOL_RENEW"); int v = e ? atoi(e) : DM_GAME_POOL_RENEW;
        g_pool_renew = v <= 0 ? 0 : (v >= 2 ? 2 : 1);
        if (g_pool_renew) MLOG("池换页（D1）：开，%s", g_pool_renew >= 2 ? "只换页不清零（【测量用】会把池内容清零，可能踩坏游戏的分配器）" : "换页后拷回内容");
    }
    return g_pool_renew;
}
// Replace [a, b) (page aligned) with fresh read-write anonymous pages, in 4 MB chunks so the bounce buffer stays small. copy: put the old
// bytes back (all-zero pages are skipped -- the fresh pages are already zero, so untouched pages cost no footprint). Returns the bytes
// replaced. No lock held.
static uint64_t pool_renew_pages(uintptr_t a, uintptr_t b, int copy) {
    enum { CH = 4u << 20 };   // a multiple of every host page size
    size_t pg = (size_t)getpagesize();
    if (b <= a || ((a | b) & (pg - 1))) return 0;   // the range is page aligned by construction: refuse anything else rather than touch a neighbour
    size_t tn = copy ? (b - a < CH ? (size_t)(b - a) : CH) : 0;
    void *tmp = copy ? malloc(tn) : NULL;
    if (copy && !tmp) return 0;
    uint64_t done = 0;
    for (uintptr_t p = a; p < b; ) {
        size_t n = b - p < CH ? (size_t)(b - p) : CH;
        if (copy) memcpy(tmp, (const void *)p, n);
        void *m = mmap((void *)p, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
        if (m == MAP_FAILED) {
            static _Atomic int nf; if (atomic_fetch_add(&nf, 1) < 4) MLOG("【红】池换页：替换 %#llx+%#zx 失败 errno=%d（原映射保留）", (unsigned long long)p, n, errno);
            atomic_fetch_add(&g_renew_fail, 1);
            break;
        }
        if (copy) for (size_t o = 0; o < n; o += pg) {
            const uint64_t *w = (const uint64_t *)((const uint8_t *)tmp + o); size_t k = 0, nw = pg / sizeof *w;
            while (k < nw && !w[k]) k++;
            if (k < nw) memcpy((void *)(p + o), w, pg);
        }
        done += n; p += n;
    }
    free(tmp);
    return done;
}
void mtl_pool_del(const void *mem) {
    g_reg_gen++;
    int renew = pool_renew_on();
    uintptr_t rlo = 0, rhi = 0; int rres = 0, was_res = 0; uint64_t phits = 0; id<MTLBuffer> rb = nil;   // D1: interior page range to renew, the pool's residency, its buffer
    pthread_mutex_lock(&g_mu);
    // The no-copy buffer goes through deferred release (out of the residency set and unreferenced after submitted commands complete). It used to be removed only from g_keep while the residency set still held a strong reference: after the pool was freed the buffer stayed alive wrapping that guest memory,
    // and a new pool at the same address wrapped it again. Engage's title demo cycles scenes; GPU allocated grew to 5.4 GB and resident memory to 3.6 GB in 850 s
    for (int i = 0; i < g_npool; i++)
        if (g_pools[i].key == mem) { id<MTLBuffer> b = g_pools[i].buf; g_pools[i].buf = nil;
            { uint64_t h = g_pools[i].hits; atomic_fetch_add(&g_del_hits[h == 0 ? 0 : h == 1 ? 1 : h <= 4 ? 2 : h <= 16 ? 3 : 4], 1); }   // T5: a reclaimed pool has no buffer
            uintptr_t pg = (uintptr_t)getpagesize(), lo = g_pools[i].lo, hi = g_pools[i].hi;
            was_res = g_pools[i].res; phits = g_pools[i].hits;   // hits: how often the pool was addressed (survivor vs. average)
            // D1: renew only whole pages inside the pool's own storage (a boundary page may also hold a neighbour), only guest memory the
            // guest's allocators own and that is open read-write, and never a reclaimed pool (its pages were replaced already).
            int cand = renew && b && g_pools[i].rc_state != RC_RECLAIMED;
            uintptr_t a = (lo + pg - 1) & ~(pg - 1), e = hi & ~(pg - 1);
            // DM_POOL_RENEW_FULL=1 (measurement): renew the buffer's WHOLE page-aligned range, boundary pages included, when nothing else
            // claims those pages. The buffer Metal built covers that whole range, so this tests "does a buffer stay alive while any of its
            // own pages is still mapped". Off by default: a boundary page can belong to a neighbour, which is why D1 stops at the interior.
            if (cand && pool_renew_full()) {
                uintptr_t xa = g_pools[i].base, xe = xa + g_pools[i].buflen;
                if (xe > xa && dm_guest_renewable(xa, xe - xa)) { a = xa; e = xe; }
            }
            if (!b) atomic_fetch_add(g_pools[i].rc_state == RC_RECLAIMED ? &g_renew_nobuf_rc : &g_renew_nobuf, 1);
            if (cand && (e <= a || !dm_guest_renewable(a, e - a))) { cand = 0; atomic_fetch_add(e > a ? &g_renew_skip_mem : &g_renew_skip_tiny, 1); }
            if (b) { if (!g_dead_pools) g_dead_pools = [NSHashTable weakObjectsHashTable]; [g_dead_pools addObject:b]; [g_keep removeObject:b]; }
            if (cand) { rb = b; rlo = a; rhi = e; rres = was_res; }
            else if (b) { dead_meta_put(b, lo, hi - lo, 0, (unsigned long)CFGetRetainCount((__bridge CFTypeRef)b) - 1, was_res, phits); defer_obj_release(b); }
            if (was_res) g_pool_res_n--;
            if (g_pools[i].rc_state == RC_RECLAIMED) {   // T5: tombstone the reclaimed range so "source gone" checks stop matching a freed pool
                int rgn = (int)atomic_load_explicit(&g_rc_rg_n, memory_order_relaxed);
                for (int r = 0; r < rgn && r < RC_RG_MAX; r++) if (atomic_load(&g_rc_rg[r].lo) == g_pools[i].lo) { atomic_store(&g_rc_rg[r].lo, (uintptr_t)0); atomic_store(&g_rc_rg[r].hi, (uintptr_t)0); }
            }
            // The block goes back to the game's allocator: whatever this pool still counted against the reclaim budget is released, so the next
            // level can reclaim again (this is what makes the budget a bound on the memory reclaimed at one time rather than on the whole run)
            if (g_pools[i].rc_bytes) { atomic_fetch_sub(&g_rc_live_bytes, g_pools[i].rc_bytes); g_pools[i].rc_bytes = 0; }
            g_pool_dels++; g_pool_del_bytes += g_pools[i].size;   // DM_POOL_CENSUS per-census create / destroy delta
            if (atomic_fetch_add(&g_d1_del, 1) == 0) {   // D1 probe: first delete of this burst -- where the cumulative counters stand
                g_d1_skip_mem0 = atomic_load(&g_renew_skip_mem); g_d1_skip_ov0 = atomic_load(&g_renew_skip_overlap);
                g_d1_skip_gpu0 = atomic_load(&g_renew_skip_gpu); g_d1_fail0 = atomic_load(&g_renew_fail);
                g_d1_nobuf0 = atomic_load(&g_renew_nobuf); g_d1_nobuf_rc0 = atomic_load(&g_renew_nobuf_rc); g_d1_tiny0 = atomic_load(&g_renew_skip_tiny);
            }
            atomic_store(&g_d1_frame, atomic_load_explicit(&g_cur_frame, memory_order_relaxed));
            g_pools[i] = g_pools[--g_npool]; g_pidx_ok = 0; g_plast = -1; atomic_fetch_add_explicit(&g_pool_gen, 1, memory_order_relaxed); break; }
    if (rb) {   // D1: a live pool whose buffer covers any page to renew would keep a stale (old-page) GPU mapping: then the pages stay as they are
        for (int k = 0; k < g_npool; k++)
            if (g_pools[k].base < rhi && rlo < g_pools[k].base + g_pools[k].buflen) { rlo = rhi = 0; atomic_fetch_add(&g_renew_skip_overlap, 1); break; }
        if (rhi <= rlo) { defer_obj_release(rb); rb = nil; }
    }
    pthread_mutex_unlock(&g_mu);
    if (!rb) return;   // the buffer is already queued for deferred release; nothing to renew
    // D1: wait (no lock held) until the GPU has completed every submission committed so far -- the point the deferred release would have
    // waited for anyway. (The reclaim path releases its buffer without a wait because it runs right after up_wait; a Finalize can happen at
    // any time.) A no-copy buffer kept across a remap holds a STALE GPU mapping (the probe note in rc_act), so the pages must not be
    // replaced while any submitted command may still read them. The game finalizes a whole batch of pools between two submissions, so
    // g_fence does not move in between and the batch pays this wait once.
    uint64_t t0 = now_ns(), f = g_fence;
    if (f && !mtl_wait(f, 2000000000ull)) {   // the GPU is stuck: fall back to the ordinary deferred release and leave the pages alone
        static _Atomic int nto; if (!atomic_fetch_add(&nto, 1)) MLOG("【红】池换页：等 GPU 到 fence %llu 超时，本池按原路延迟释放、不换页", (unsigned long long)f);
        pthread_mutex_lock(&g_mu); dead_meta_put(rb, rlo, rhi - rlo, 2, (unsigned long)CFGetRetainCount((__bridge CFTypeRef)rb) - 1, was_res, phits); pthread_mutex_unlock(&g_mu);
        defer_obj_release(rb); atomic_fetch_add(&g_renew_skip_gpu, 1);   // the residency entry is still in the set; defer_obj_release takes it out when the fence completes (as it always did)
        return;
    }
    // D1: re-check the overlap now that the lock was dropped for the wait: another thread may have created a pool on this very guest block
    // while we waited, and its buffer would wrap the pages about to be replaced. Under g_mu again, but briefly (no wait inside).
    pthread_mutex_lock(&g_mu);
    int taken = 0;
    for (int k = 0; k < g_npool; k++)
        if (g_pools[k].base < rhi && rlo < g_pools[k].base + g_pools[k].buflen) { taken = 1; break; }
    pthread_mutex_unlock(&g_mu);
    if (taken) {   // a live pool moved onto the range: leave the pages alone, release the buffer the ordinary way
        static _Atomic int nre; if (!atomic_fetch_add(&nre, 1)) MLOG("【红】池换页：等 GPU 期间有新池落到了池 %p 的地址上，本次不换页、按原路延迟释放", mem);
        pthread_mutex_lock(&g_mu); dead_meta_put(rb, rlo, rhi - rlo, 2, (unsigned long)CFGetRetainCount((__bridge CFTypeRef)rb) - 1, was_res, phits); pthread_mutex_unlock(&g_mu);
        defer_obj_release(rb); atomic_fetch_add(&g_renew_skip_overlap, 1);
        return;
    }
    if (rres) { res_remove(rb); res_commit(); }   // the residency set's strong reference goes too, or the buffer could never die (see the note at the top of this function)
    {   // D1 probe: how many references the buffer still has the instant the host drops its own -- i.e. holders that are NOT in any table the
        // dead-buffer probe can see. Measured while rb is still the only reference this function holds, so the probe's own +1 is subtracted.
        unsigned long rc = (unsigned long)CFGetRetainCount((__bridge CFTypeRef)rb) - 1;
        pthread_mutex_lock(&g_mu); dead_meta_put(rb, rlo, rhi - rlo, 1, rc, was_res, phits); pthread_mutex_unlock(&g_mu);
        dead_watch_note(rb);   // D1 probe: follow this one from now on, phase by phase (the slot is weak: it does not keep the buffer alive)
    }
    {   // DM_POOL_PURGE: ask the driver to let go of the allocation before the last host reference goes (a bytesNoCopy buffer's storage is not Metal's, so this may be a no-op -- the measurement says which)
        int pm = pool_purge_mode();
        if (pm) {
            MTLPurgeableState was = [rb setPurgeableState:pm == 2 ? MTLPurgeableStateVolatile : MTLPurgeableStateEmpty];
            static _Atomic int pn; if (atomic_fetch_add(&pn, 1) < 3) MLOG("池换页：置 %s 前缓冲 %p 的可释放状态是 %d", pm == 2 ? "Volatile" : "Empty", (__bridge void *)rb, (int)was);
        }
    }
    {   // D1 probe: the weak slot is the only way to ask "did that nil deallocate it" -- it answers before anything else in the burst can get a reference
        __weak id<MTLBuffer> w = rb;
        rb = nil;   // the last host reference: the driver's view of the old pages goes before the mapping is replaced
        // Still held at that instant? Counted, not investigated here: dead_holder_report() calls [g_res allAllocations], whose snapshot array
        // autoreleases every allocation in the set -- on the guest thread, with no pool to drain it, a dozen of those pinned 150-200 MB of GPU
        // allocations for good (measured as a +180 MB GPU regression against the previous build). Never call it from this path.
        if (w) atomic_fetch_add(&g_drop_alive, 1); else atomic_fetch_add(&g_drop_died, 1);
    }
    // A view or texture over this buffer may still hold it for another frame (they go through the deferred queue, and their fence is the
    // next commit). That no longer blocks the remap: nothing submitted can read the pages any more (the wait above), the bytes are kept,
    // and the reclaim has always remapped with such views still queued. The buffer simply dies when its fence passes, a frame later.
    // The remap itself is synchronous inside the game's own Finalize call, and the guest cannot reuse the block before Finalize returns
    // (NVN forbids touching a finalized pool), so no guest write can race the copy.
    uint64_t done = (rhi > rlo && !pool_renew_droponly()) ? pool_renew_pages(rlo, rhi, renew < 2) : 0;
    if (pool_renew_droponly()) atomic_fetch_add(&g_renew_droponly, 1);
    atomic_fetch_add(&g_renew_ns, now_ns() - t0);
    if (done) {
        atomic_fetch_add(&g_d1_renew, 1); atomic_fetch_add(&g_d1_renew_b, done);   // D1 probe: burst accounting
        uint64_t np = atomic_fetch_add(&g_renew_pools, 1) + 1, nb = atomic_fetch_add(&g_renew_bytes, done) + done;
        if (np <= 8 || (np & (np - 1)) == 0)   // the first few, then powers of two
            MLOG("池换页：池 %p 换页 %llu KB（累计 %llu 池 %llu MB，跳过 非堆 %llu / 与活池重叠 %llu / GPU 超时 %llu，失败 %llu，耗时累计 %.1f ms）", mem,
                 (unsigned long long)(done >> 10), (unsigned long long)np, (unsigned long long)(nb >> 20),
                 (unsigned long long)atomic_load(&g_renew_skip_mem), (unsigned long long)atomic_load(&g_renew_skip_overlap),
                 (unsigned long long)atomic_load(&g_renew_skip_gpu),
                 (unsigned long long)atomic_load(&g_renew_fail), atomic_load(&g_renew_ns) / 1e6);
    }
}

// Address → pool: the last hit first, then a binary search over indices sorted by start address, and a full linear scan only when that fails (pools overlapping).
// The Legend of Zelda: Breath of the Wild keeps about 1760 memory pools alive and every buffer binding of every draw looks one up; the old full scan took about 4% of the submit thread's samples (mtl_gpu_addr 222 / 6 s). Adding / removing pools invalidates the index, which is re-sorted when next used. Caller holds g_mu
static int *g_pidx; static int g_pidx_cap;
static int pidx_cmp(const void *a, const void *b) { uintptr_t x = g_pools[*(const int *)a].lo, y = g_pools[*(const int *)b].lo; return x < y ? -1 : x > y; }
static int pool_find(uint64_t addr, uint64_t len) {
    if (g_plast >= 0 && g_plast < g_npool && addr >= g_pools[g_plast].lo && addr + len <= g_pools[g_plast].hi) return g_plast;
    if (!g_pidx_ok) {
        if (g_pidx_cap < g_npool) { g_pidx_cap = g_npool * 2; g_pidx = realloc(g_pidx, (size_t)g_pidx_cap * sizeof *g_pidx); }
        for (int i = 0; i < g_npool; i++) g_pidx[i] = i;
        qsort(g_pidx, (size_t)g_npool, sizeof *g_pidx, pidx_cmp); g_pidx_ok = 1;
    }
    int lo = 0, hi = g_npool - 1, hit = -1;   // the last one whose start ≤ addr
    while (lo <= hi) { int m = (lo + hi) / 2; if (g_pools[g_pidx[m]].lo <= addr) { hit = m; lo = m + 1; } else hi = m - 1; }
    if (hit >= 0 && addr + len <= g_pools[g_pidx[hit]].hi) return g_plast = g_pidx[hit];
    for (int i = 0; i < g_npool; i++) if (addr >= g_pools[i].lo && addr + len <= g_pools[i].hi) return g_plast = i;
    return -1;
}
// noinline: the census records __builtin_return_address(0) as the first address user, which must be the real caller (see pool_note_hit). The call overhead is negligible
// next to the pool lookup itself, and the pool address paths are not inlined across this function's callers today anyway.
__attribute__((noinline)) int mtl_addr(uint64_t addr, uint64_t len, void **buf, uint64_t *off) {
    int ok = 0; uintptr_t site = (uintptr_t)__builtin_return_address(0);
    // The per-thread cache: the same pools over and over. The pool's buffer and its base offset are in the cache entry, so a hit does
    // not need the pool lock or the index search. A pool whose source pages were reclaimed (T5) must not be served from the cache:
    // its buffer may be gone and the restore (rc_unreclaim_pool) has to run under the lock.
    int ci = pc_get(addr, len);
    if (ci >= 0) {
        pc_ent *c = &t_pc.e[ci];
        if (c->buf) { *buf = (__bridge void *)c->buf; *off = addr - c->base; return 1; }
    }
    pthread_mutex_lock(&g_mu);
    int i = pool_find(addr, len);
    if (i >= 0) { pool_note_hit(i, addr, len, 0, site); int r = pool_res(i); if (r && g_pools[i].buf) { *buf = (__bridge void *)g_pools[i].buf; *off = addr - g_pools[i].base; ok = 1; pc_put(i, g_pools[i].base, g_pools[i].gpu); } }
    pthread_mutex_unlock(&g_mu);
    return ok;
}
__attribute__((noinline)) int mtl_gpu_addr(uint64_t addr, uint64_t len, void **buf, uint64_t *off, uint64_t *gpu) {
    int ok = 0; uintptr_t site = (uintptr_t)__builtin_return_address(0);
    int ci = pc_get(addr, len);   // the per-thread cache: see mtl_addr
    if (ci >= 0) {
        pc_ent *c = &t_pc.e[ci];
        if (c->buf) { *buf = (__bridge void *)c->buf; *off = addr - c->base; *gpu = c->gpu + *off; return 1; }
    }
    pthread_mutex_lock(&g_mu);
    int i = pool_find(addr, len);
    if (i >= 0) { pool_note_hit(i, addr, len, 1, site); int r = pool_res(i); if (r && g_pools[i].buf) { *buf = (__bridge void *)g_pools[i].buf; *off = addr - g_pools[i].base; *gpu = g_pools[i].gpu + *off; ok = 1; pc_put(i, g_pools[i].base, g_pools[i].gpu); } }
    pthread_mutex_unlock(&g_mu);
    return ok;
}

// T5: a packaged texture was just uploaded into its private copy; record its source pool as a candidate of this upload batch. The per-pool record
// (distinct serials + source ranges + content stamps) is the evidence the eligibility scan uses, so pools whose textures are not in the texture-ID
// registry (e.g. used as render targets only) are covered too. No lock held; called on the submit thread after the upload address was resolved.
static int tex_cpu_written(uint64_t serial);   // defined with the CPU-written texture set further below
static int rc_note_upload_src(const tex_t *t) {
    if (!g_tex_reclaim && !g_tex_reclaim_dry) return 0;   // off for titles without the feature: no per-upload bookkeeping at all
    if (!t || !t->mem || !t->L.size || !t->pool || !t->pool->mem) return 0;
    if (!(t->pool->flags & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS)) return 0;   // v1: CPU-inaccessible sources only (the game cannot read/write them itself)
    int noted = 0;
    pthread_mutex_lock(&g_mu);
    int i = pool_find((uintptr_t)t->mem, t->L.size);
    if (i >= 0) {
        pool_ent *p = &g_pools[i];
        p->rc_up_gen++;                                        // a new source upload started; rc_done_gen only catches up when its batch completed
        p->rc_hits_at_upload = p->hits;                        // the upload's own address use is included; any later use disqualifies the pool
        p->rc_last_upload_frame = atomic_load(&g_cur_frame);
        p->rc_up_bytes += t->L.size;
        // A dynamic descriptor (storage-write target, linear / zero-copy, CPU-written, multisample, unsupported target) means the pool can never be a static source.
        if (!t->has_layout || tex_cpu_written(t->serial)) p->rc_up_state |= 1;
        if ((t->d.flags & NVN_TEXTURE_FLAGS_IMAGE) && !t->has_layout) p->rc_up_state |= 1;   // packaged IMAGE is a spurious flag on Hades' ASTC assets, see rc_desc_static
        if (t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET | NVN_TEXTURE_FLAGS_DISPLAY |
                          NVN_TEXTURE_FLAGS_VIDEO_DECODE | NVN_TEXTURE_FLAGS_SPARSE)) p->rc_up_state |= 1;
        if (t->d.samples > 1) p->rc_up_state |= 1;
        switch (t->d.target) { case 0: case 1: case 3: case 4: case 8: case 9: break; default: p->rc_up_state |= 1; break; }
        // The same serial uploaded again means the content changed (streamed re-upload): the pool is dynamic.
        int seen = 0;
        for (int k = 0; k < (int)p->rc_up_nser; k++) if (p->rc_up_ser[k] == t->serial) { seen = 1; break; }
        if (seen) p->rc_up_state |= 2;
        else if (p->rc_up_nser < 12) p->rc_up_ser[p->rc_up_nser++] = t->serial;
        else p->rc_up_state |= 4;                              // record full: registered-texture evidence may still qualify the pool
        // Distinct source ranges (10 views over the same storage must count once): they carry coverage and the dirty stamp.
        int rseen = 0;
        for (int k = 0; k < (int)p->rc_up_nrg; k++) if (p->rc_up_rg[k].mem == (uintptr_t)t->mem && p->rc_up_rg[k].size == t->L.size) { rseen = 1; break; }
        if (!rseen) {
            if (p->rc_up_nrg < 12) { int k = p->rc_up_nrg++; p->rc_up_rg[k].mem = (uintptr_t)t->mem; p->rc_up_rg[k].size = t->L.size; p->rc_up_rg[k].stamp = nvn_dirty_now(); }
            else p->rc_up_state |= 4;
        }
        int first_pool = (p->rc_up_nser == 1 && !seen);   // this note added the pool's first serial: count the pool once
        if (!rseen) g_rc_cand_bytes += t->L.size;          // distinct source range: count its bytes once
        if (first_pool) g_rc_cand_pools++;
        // Batch list, deduped by pool storage address.
        uintptr_t key = p->lo;
        for (int k = 0; k < g_up_rc_n; k++) if (g_up_rc[k].lo == key) { g_up_rc[k].bytes += t->L.size; g_up_rc[k].gen = p->rc_up_gen; noted = 1; break; }
        if (!noted) {
            if (g_up_rc_n == g_up_rc_cap) { g_up_rc_cap = g_up_rc_cap ? g_up_rc_cap * 2 : 32; rc_src *n = realloc(g_up_rc, (size_t)g_up_rc_cap * sizeof *n); if (n) g_up_rc = n; else g_up_rc_cap = g_up_rc_n; }
            if (g_up_rc && g_up_rc_n < g_up_rc_cap) { g_up_rc[g_up_rc_n++] = (rc_src){key, p->size, t->L.size, p->rc_up_gen}; noted = 1; }
        }
    }
    pthread_mutex_unlock(&g_mu);
    return noted;
}
// T5: recreate the no-copy buffer of a pool and put it back into the residency set. Caller holds g_mu. 1 = usable again.
// Used by the invalidation path (pool addressed after reclaim) and by rc_act when replacing the pages failed completely.
static int rc_buf_restore_locked(int i) {
    pool_ent *p = &g_pools[i];
    id<MTLBuffer> b = [g_dev newBufferWithBytesNoCopy:(void *)p->base length:p->buflen options:MTLResourceStorageModeShared deallocator:nil];
    if (!b) return 0;
    p->buf = b; p->gpu = b.gpuAddress;
    atomic_fetch_add_explicit(&g_pool_gen, 1, memory_order_relaxed);   // the per-thread pool caches hold the old buffer: they are stale now
    [g_keep addObject:b];
    res_add(b); res_commit();
    p->res = 1; g_pool_res_n++;
    p->rc_state = RC_WAIT;
    return 1;
}
// T5: recreate the buffer of a reclaimed pool that is addressed again and give up reclaiming that pool.
// Caller holds g_mu. Returns 1 when the pool has a usable buffer again.
static int rc_unreclaim_pool(int i) {
    if (!rc_buf_restore_locked(i)) {
        MLOG("【红】静态源回收：池 %p 的零拷贝缓冲重建失败（%llu MB），本次寻址按失败处理", (void *)(uintptr_t)g_pools[i].lo, (unsigned long long)(g_pools[i].buflen >> 20));
        g_pools[i].rc_state = RC_WAIT;
        return 0;
    }
    pool_ent *p = &g_pools[i];
    p->rc_state = RC_NONE;
    if (!p->rc_no_reclaim) {
        p->rc_no_reclaim = 1; g_rc_invalid++;
        MLOG("【红】静态源回收：已回收池 %p（%llu MB）又被寻址，重建零拷贝缓冲并重新驻留；不再回收该池（累计失效 %llu）",
             (void *)(uintptr_t)p->lo, (unsigned long long)(p->size >> 20), (unsigned long long)g_rc_invalid);
    }
    return 1;
}

// ---------- Submission ----------
static void defer_reset(void); static void defer_fallback(void);
typedef struct { int kind; volatile uint32_t *val; uint32_t mode, v; uint64_t addr; uint32_t type; void *vis; } post_op;   // vis: the visibility slot table for kind 3
// ---------- Occlusion queries (NVN_COUNTER_TYPE_SAMPLES_PASSED) ----------
// NVN semantics: samples passed is a cumulative counter on the queue - ResetCounter zeroes it, every draw afterwards adds to it, and ReportCounter writes out the cumulative value at that moment (+ timestamp) when the GPU completes.
// Metal: every render pass attaches a visibility result buffer; draws during counting use counting mode, with a new slot per pass (no reliance on accumulation across passes); a reset empties the 'slot table since the reset',
// and a report hands the slot table together with its buffer (retained) to the completion callback to sum and write back; after a report a new slot is used, so later draws are not counted into the reported value.
// (Previously always wrote 0: occlusion query = 'fully occluded', campfires / light sources got culled - the character select screen calls ResetCounter 21 times per frame)
typedef struct { void *buf; uint32_t slot; } vis_slot;
typedef struct { int n; vis_slot s[]; } vis_list;
#define VIS_SLOTS 8192
typedef struct { __strong id<MTLBuffer> buf; uint64_t fence; } vis_buf;
static vis_buf g_vis_ring[4]; static int g_vis_ring_n, g_vis_ring_i = -1;
static id<MTLBuffer> g_vis; static uint32_t g_vis_n;   // This submission's visibility buffer and number of slots used
static int vis_in_ring(id<MTLBuffer> b) { for (int i = 0; i < g_vis_ring_n; i++) if (g_vis_ring[i].buf == b) return 1; return 0; }
static int g_vis_on, g_vis_counting, g_vis_cur = -1; static uint64_t g_vis_cur_pass;
static vis_slot *g_vs; static int g_nvs, g_vscap;      // Slots since the last reset
static uint64_t g_cnt_report[32], g_cnt_reset[32];       // Report / reset counts per counter type (diagnostics)
static void vis_clear(void) { for (int i = 0; i < g_nvs; i++) CFRelease(g_vs[i].buf); g_nvs = 0; g_vis_cur = -1; }
static vis_list *vis_snapshot(void) {
    vis_list *l = malloc(sizeof *l + g_nvs * sizeof(vis_slot)); l->n = g_nvs;
    for (int i = 0; i < g_nvs; i++) { l->s[i] = g_vs[i]; CFRetain(g_vs[i].buf); }
    return l;
}
// Target ranges of buffer writes (copies / inline uniform updates) already encoded in this submission but not yet executed on the GPU: reading these addresses on the CPU side at encode time yields old values (uv_proof avoids them when reading uniforms)
#define PEND_MAX 256
static uint64_t g_pend_lo[PEND_MAX], g_pend_hi[PEND_MAX]; static int g_npend, g_pend_over;
static int pend_hits(uint64_t a, uint64_t n) { if (g_pend_over) return 1; for (int i = 0; i < g_npend; i++) if (a < g_pend_hi[i] && a + n > g_pend_lo[i]) return 1; return 0; }
typedef struct { uint64_t ser; uint16_t level, slice; } zskip_zero;
static zskip_zero g_zskip_zero[4096]; static int g_zskip_n, g_zskip_count;
static zskip_zero g_zskip_cand[8]; static uint8_t g_zskip_cand_ok[8];
static void zskip_clear(void);
static int tbdr_on(void);
static id g_cb;
static id<MTL4ComputeCommandEncoder> g_blit;
static NSMutableArray<dispatch_block_t> *g_commit_blocks;
static uint64_t g_alloc_fence[ALLOC_MAX]; static uint64_t g_alloc_waits, g_alloc_grown;
static void up_flush(void);
static void up_wait(void);
// The upload batch belongs only to the submit thread (the thread that called mtl_sub_begin): the batch's encoder cannot be used across threads. Every entry point asserts current thread == submit thread,
// and on mismatch prints the stack and aborts on the spot - so that 'another thread touched the batch' is exposed deterministically on Mac too, not by timing (real iOS device stuck at the main menu: the present thread flushed the submit thread's batch via q_cb).
static pthread_t g_sub_thread; static _Atomic int g_sub_thread_set;
static int on_sub_thread(void) { return atomic_load(&g_sub_thread_set) && pthread_equal(pthread_self(), g_sub_thread); }
static void up_owner_check(const char *where) {
    if (!atomic_load(&g_sub_thread_set) || on_sub_thread()) return;
    char name[64] = ""; pthread_getname_np(pthread_self(), name, sizeof name);
    MLOG("【红】上传批只属于提交线程：%s 在线程「%s」上被调用", where, name);
    void *bt[32]; int n = backtrace(bt, 32); backtrace_symbols_fd(bt, n, 2);
    abort();
}
// Every command buffer created outside the replay / upload batch goes through this. On the submit thread: flush this batch's uploads first, so that commands that then read or write these textures are ordered after the uploads;
// other threads (the present thread etc.) creating a command buffer do not touch the batch: they do not read or write textures uploaded in this batch; commands that need these textures all come after this replay submission (the batch is flushed before it)
// Old-queue command buffers (readback / diagnostics / present; to be deleted once M4-4 and M4-5 are migrated away): uploads are already on the Metal 4 queue, and there is no ordering guarantee between the two queues -
// wait for the upload to complete after flushing it, and only then can the old queue read the upload result (mtl_read_texels waited first and flushed afterwards, the readback beat the upload, 4 failures on t_nvn upload content)
static id<MTLCommandBuffer> q_cb(void) { if (on_sub_thread()) { up_flush(); while (atomic_load(&g_up_pending)) usleep(200); } return [g_q commandBuffer]; }
static post_op *g_post; static int g_npost, g_postcap;
// One autorelease pool per submission: submissions run on a guest thread (long-lived, no run loop, never drains the implicit pool), so objects autoreleased in the Metal replay
// (command buffers, encoders, pass descriptors, etc.) are never released unless put in a pool. Measured on the character select screen of a real iOS device: the host heap grows by about 47 objects per frame, 0.22 MiB/s.
void *objc_autoreleasePoolPush(void); void objc_autoreleasePoolPop(void *);
static void *g_sub_pool;
static int g_ruler_truth = -1, g_ruler_pending;
static int tbdr_on(void);

static uint64_t g_sub_serial;   // Submission index (used by the parallel precompile's 'predict only once per submission')
enum { ARPOOL_OFF, ARPOOL_SUB, ARPOOL_FRAME };   // DM_ARPOOL (see arpool_mode, defined with the frame-hook probes further down)
static int arpool_mode(void);
static void arpool_roll(void **slot);
void mtl_sub_begin(void) {
    // DM_ARPOOL=sub: one submission wide rolling pool (see guest_frame_arpool for why the guest thread needs one at all). mtl_sub_begin is
    // the only per-submission entry point that is always reached on the guest thread, so the previous submission's autoreleases are drained
    // here instead of at the end of mtl_sub_commit (which returns from several places).
    { static void *ar; if (arpool_mode() == ARPOOL_SUB) arpool_roll(&ar); }
    dead_watch("提交前");
    mtl_init(); g_sub_serial++;
    if (tbdr_on() == 0 && g_ruler_pending) { g_ruler_truth = 0; g_ruler_pending = 0; MLOG("DM_NVN_RULER：切到被测模式"); }
    defer_drain();
    defer_reset();
    warm_claim();
    if (g_tex_reclaim || g_tex_reclaim_dry) rc_reclaim_tick();   // T5: safe point (no encoder running, no lock held)
    if (g_cb) MDIE("提交重入");
    g_sub_thread = pthread_self(); atomic_store(&g_sub_thread_set, 1);
    atomic_store(&g_last_commit_error, 0);
    g_sub_pool = objc_autoreleasePoolPush();
    g_cb = [g_dev newCommandBuffer];
    // Do not wait for the GPU mid-frame (hot real device: with a fixed rotation of 3, when the GPU cannot keep up every frame blocks here, and the submit thread runs at low frequency after waking, instruction count −18% but CPU time +49%):
    // pick an allocator the GPU has finished with; if all are in use create a new one; only at the cap wait for the oldest one. How far ahead it runs is bounded by the wait where the game acquires the swapchain texture, same as the old interface.
    int alloc_i = -1; uint64_t done = g_done;
    for (int i = 0; i < g_alloc_n; i++) if (g_alloc_fence[i] <= done) { alloc_i = i; break; }
    if (alloc_i < 0 && g_alloc_n < ALLOC_MAX) { NSError *ae = nil; MTL4CommandAllocatorDescriptor *ad = [MTL4CommandAllocatorDescriptor new];
        g_alloc4[g_alloc_n] = [g_dev newCommandAllocatorWithDescriptor:ad error:&ae]; if (!g_alloc4[g_alloc_n]) MDIE("新建 MTL4CommandAllocator 失败：%s", ae.localizedDescription.UTF8String);
        alloc_i = g_alloc_n++; g_alloc_grown++; }
    if (alloc_i < 0) { alloc_i = 0; for (int i = 1; i < g_alloc_n; i++) if (g_alloc_fence[i] < g_alloc_fence[alloc_i]) alloc_i = i; g_alloc_waits++; mtl_wait(g_alloc_fence[alloc_i], UINT64_MAX); }
    g_alloc_cur = alloc_i;
    [g_alloc4[alloc_i] reset];
    [g_cb beginCommandBufferWithAllocator:g_alloc4[alloc_i]];
    g_commit_blocks = [NSMutableArray array];
    // The overflow buffer created when the previous submission ran out of slots (see g_vis_n >= VIS_SLOTS): retire it. It used to be just cleared here and stayed in the residency set for good (from d2r-mobile 2eb575877)
    if (g_vis && !vis_in_ring(g_vis)) defer_obj_release(g_vis);
    g_npost = 0; g_vis = nil; g_vis_n = 0; g_vis_cur = -1; g_npend = 0; g_pend_over = 0;
    zskip_clear(); g_zskip_count = 0;
    int pick = -1;
    for (int i = 0; i < g_vis_ring_n; i++) if (g_vis_ring[i].fence <= g_done) { pick = i; break; }
    if (pick < 0 && g_vis_ring_n < 4) pick = g_vis_ring_n++;
    if (pick < 0) { uint64_t oldest = UINT64_MAX; for (int i = 0; i < 4; i++) if (g_vis_ring[i].fence < oldest) { oldest = g_vis_ring[i].fence; pick = i; } if (oldest > g_done) mtl_wait(oldest, UINT64_MAX); }
    g_vis_ring_i = pick; g_vis_ring[pick].fence = UINT64_MAX;
    if (!g_vis_ring[pick].buf) { g_vis_ring[pick].buf = [g_dev newBufferWithLength:VIS_SLOTS * 8 options:MTLResourceStorageModeShared]; res_add(g_vis_ring[pick].buf); }
    g_vis = g_vis_ring[pick].buf;
    void pg_sub_begin_ext(uint64_t fence); pg_sub_begin_ext(g_fence + 1);
    void arena_begin(void); arena_begin();
}

static void rp_end(void);
extern int g_frz_on; static int g_frz_state;   // World freeze: the state latched for this frame (0 normal / 1 full frame / 2 frozen frame), see the note above frz_op
// DM_NVN_PASSGPU_SERIAL=1 (for measurement): before creating each encoder on the main command buffer, encode 'signal n+1 → wait for signal n+1' - the signal is only raised once all prior work has completed,
// so the following encoder starts only after the previous one has fully finished; on TBDR passes no longer overlap in the pipeline, and the per-pass start-to-end span of DM_NVN_PASSGPU is that pass's exclusive time (the whole-frame total is inflated; look only at relative cost)
static void ser_gate(void) {
    static int on = -1; static id<MTLEvent> ev; static uint64_t n;
    if (on < 0) { on = getenv("DM_NVN_PASSGPU_SERIAL") != NULL; if (on) { ev = [g_dev newEvent]; MLOG("DM_NVN_PASSGPU_SERIAL：通道串行化（测量用，耗时偏大）"); } }
    if (!on) return;
    (void)ev; (void)n;
}
// Metal 4 conservative barriers (first version): a consumer barrier at the start of an encoder (all prior work → this encoder) and a producer barrier at the end (this encoder and earlier → later encoders).
// Only the one at the start is not enough: t_rt_range 'pass 1 → pass 2 → copy' read back black, and after adding the end barrier it is 19/19. Old-queue encoders (upload / diagnostics / present) do not go through these barriers.
static inline void m4_end(id e) {
    if ([e conformsToProtocol:@protocol(MTL4CommandEncoder)]) [(id<MTL4CommandEncoder>)e barrierAfterStages:MTLStageAll beforeQueueStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice];
    [e endEncoding];
}
static inline id<MTL4RenderCommandEncoder> enc_r_begin(MTL4RenderPassDescriptor *d) { id<MTL4RenderCommandEncoder> e = [g_cb renderCommandEncoderWithDescriptor:d]; [e barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice]; [e setArgumentTable:g_vt atStages:MTLRenderStageVertex]; [e setArgumentTable:g_ft atStages:MTLRenderStageFragment]; return e; }
static inline void enc_r_end(id e) { m4_end(e); }
static inline void enc_r_pso(id<MTL4RenderCommandEncoder> e, id<MTLRenderPipelineState> p) { [e setRenderPipelineState:p]; }
static inline void enc_r_dss(id<MTL4RenderCommandEncoder> e, id<MTLDepthStencilState> s) { [e setDepthStencilState:s]; }
static inline void enc_r_viewport(id<MTL4RenderCommandEncoder> e, MTLViewport v) { [e setViewport:v]; }
static inline void enc_r_scissor(id<MTL4RenderCommandEncoder> e, MTLScissorRect r) { [e setScissorRect:r]; }
static inline void enc_r_cull(id<MTL4RenderCommandEncoder> e, MTLCullMode m) { [e setCullMode:m]; }
static inline void enc_r_winding(id<MTL4RenderCommandEncoder> e, MTLWinding w) { [e setFrontFacingWinding:w]; }
static inline void enc_r_fill(id<MTL4RenderCommandEncoder> e, MTLTriangleFillMode m) { [e setTriangleFillMode:m]; }
static inline void enc_r_depth_bias(id<MTL4RenderCommandEncoder> e, float u, float s, float c) { [e setDepthBias:u slopeScale:s clamp:c]; }
static inline void enc_r_depth_clip(id<MTL4RenderCommandEncoder> e, BOOL v) { [e setDepthClipMode:v ? MTLDepthClipModeClip : MTLDepthClipModeClamp]; }
static inline void enc_r_stencil_ref(id<MTL4RenderCommandEncoder> e, uint32_t f, uint32_t b) { (void)b; [e setStencilReferenceValue:f]; }
static inline void enc_r_blend_color(id<MTL4RenderCommandEncoder> e, float r, float g, float b, float a) { [e setBlendColorRed:r green:g blue:b alpha:a]; }
static inline void enc_r_visibility(id<MTL4RenderCommandEncoder> e, MTLVisibilityResultMode m, NSUInteger o) { [e setVisibilityResultMode:m offset:o]; }
static inline void enc_r_vbuf(id<MTL4RenderCommandEncoder> e, NSUInteger i, uint64_t gpu) { (void)e; [g_vt setAddress:gpu atIndex:i]; }
static inline void enc_r_args(id<MTL4RenderCommandEncoder> e, MTLRenderStages s, uint64_t gpu, NSUInteger i) { (void)e; [(s == MTLRenderStageVertex ? g_vt : g_ft) setAddress:gpu atIndex:i]; }
static inline void enc_r_draw(id<MTL4RenderCommandEncoder> e, MTLPrimitiveType p, NSUInteger v, NSUInteger n) { [e drawPrimitives:p vertexStart:v vertexCount:n]; }
static inline void enc_r_draw_instanced(id<MTL4RenderCommandEncoder> e, MTLPrimitiveType p, NSUInteger v, NSUInteger n, NSUInteger i, NSUInteger b) { [e drawPrimitives:p vertexStart:v vertexCount:n instanceCount:i baseInstance:b]; }
static inline void enc_r_draw_indexed(id<MTL4RenderCommandEncoder> e, MTLPrimitiveType p, NSUInteger n, MTLIndexType t, uint64_t gpu, NSUInteger len, NSUInteger i, NSInteger bv, NSUInteger bi) { [e drawIndexedPrimitives:p indexCount:n indexType:t indexBuffer:gpu indexBufferLength:len instanceCount:i baseVertex:bv baseInstance:bi]; }
static inline id<MTL4ComputeCommandEncoder> enc_c_begin(MTLComputePassDescriptor *d) { (void)d; id<MTL4ComputeCommandEncoder> e = [g_cb computeCommandEncoder]; [e barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice]; [e setArgumentTable:g_ct]; return e; }
static inline id<MTLComputeCommandEncoder> enc_c_begin_concurrent(id<MTLCommandBuffer> cb) { return [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]; }
static inline void enc_c_end(id e) { m4_end(e); }
static inline void enc_c_pso(id e, id<MTLComputePipelineState> p) { [e setComputePipelineState:p]; }
static inline void enc_c_args(id e, uint64_t gpu, NSUInteger i) { (void)e; [g_ct setAddress:gpu atIndex:i]; }
static inline void enc_c_dispatch(id e, MTLSize n, MTLSize t) { [e dispatchThreadgroups:n threadsPerThreadgroup:t]; }
static inline id enc_b_begin(id cb) { return [cb respondsToSelector:@selector(beginCommandBufferWithAllocator:)] ? [cb computeCommandEncoder] : [cb blitCommandEncoder]; }
static inline void enc_b_end(id e) { m4_end(e); }
static inline void enc_b_copy_buffer(id e, id<MTLBuffer> s, NSUInteger so, id<MTLBuffer> d, NSUInteger doff, NSUInteger n) { [e copyFromBuffer:s sourceOffset:so toBuffer:d destinationOffset:doff size:n]; }
// Copy regions come from the guest descriptor, but the Metal texture can be smaller (DM_TEX_DEMOTE drops the top
// mip level of large assets; some views also differ in shape). Metal rejects an out-of-range blit, so clip every
// region to the level that actually exists instead of trusting the guest.
static int tex_clip(id<MTLTexture> t, NSUInteger lv, MTLOrigin o, MTLSize *z) {
    if (!t || lv >= t.mipmapLevelCount || !z->width || !z->height) return 0;
    NSUInteger w = MAX(t.width >> lv, 1u), h = MAX(t.height >> lv, 1u), d = t.textureType == MTLTextureType3D ? MAX(t.depth >> lv, 1u) : 1;
    if (o.x >= (NSInteger)w || o.y >= (NSInteger)h || (t.textureType == MTLTextureType3D && o.z >= (NSInteger)d)) return 0;
    z->width = MIN(z->width, w - (NSUInteger)o.x);
    z->height = MIN(z->height, h - (NSUInteger)o.y);
    z->depth = t.textureType == MTLTextureType3D ? MIN(z->depth, d - (NSUInteger)o.z) : MIN(z->depth, 1);
    return z->width && z->height && z->depth;
}
static inline void enc_b_copy_buffer_texture(id e, id<MTLBuffer> s, NSUInteger so, NSUInteger row, NSUInteger img, MTLSize z, id<MTLTexture> d, NSUInteger sl, NSUInteger lv, MTLOrigin o) { if (!tex_clip(d, lv, o, &z)) return; [e copyFromBuffer:s sourceOffset:so sourceBytesPerRow:row sourceBytesPerImage:img sourceSize:z toTexture:d destinationSlice:sl destinationLevel:lv destinationOrigin:o]; }
static inline void enc_b_copy_texture_buffer(id e, id<MTLTexture> s, NSUInteger sl, NSUInteger lv, MTLOrigin o, MTLSize z, id<MTLBuffer> d, NSUInteger doff, NSUInteger row, NSUInteger img) { if (!tex_clip(s, lv, o, &z)) return; [e copyFromTexture:s sourceSlice:sl sourceLevel:lv sourceOrigin:o sourceSize:z toBuffer:d destinationOffset:doff destinationBytesPerRow:row destinationBytesPerImage:img]; }
static inline void enc_b_copy_texture(id e, id<MTLTexture> s, NSUInteger sl, NSUInteger lv, MTLOrigin so, MTLSize z, id<MTLTexture> d, NSUInteger dl, NSUInteger dv, MTLOrigin doo) { if (!tex_clip(s, lv, so, &z) || !tex_clip(d, dv, doo, &z)) return; [e copyFromTexture:s sourceSlice:sl sourceLevel:lv sourceOrigin:so sourceSize:z toTexture:d destinationSlice:dl destinationLevel:dv destinationOrigin:doo]; }
static void blit_note(void);   // Per-pass dump: a copy / buffer update ended the render pass (defined after PD)
// Ordering inside the copy encoder: Metal 4 does not order copies / fills within one encoder (no automatic hazard tracking); an operation reading or writing a range an earlier one wrote needs an intra-encoder barrier.
// Without it, SMBW's two updates per frame to the same uniform slot ("write 4 bytes −1, then 16 bytes of coefficients") occasionally landed in reverse order; the fountain water wave simulation took −1 as its self coefficient, diverged, and the water vertices were pushed to the top of the screen.
// Buffer operations check conflicts by guest address range (barrier only when reading or writing a range written earlier); texture operations get one whenever the encoder already holds an operation.
enum { BW_MAX = 64 };
static uint64_t g_bw_lo[BW_MAX], g_bw_hi[BW_MAX]; static int g_bw_n, g_blit_ops; static uint64_t g_blit_barriers;
static uint64_t g_pass_split;   // the copies / buffer updates that ended an open render pass
static id<MTL4ComputeCommandEncoder> blit_raw(void) { blit_note(); rp_end(); if (!g_blit) { ser_gate(); g_blit = enc_b_begin(g_cb); [g_blit barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice]; g_bw_n = 0; g_blit_ops = 0; } return g_blit; }
static void blit_barrier(void) {
    static int off = -1; if (off < 0) { const char *e = getenv("DM_NVN_BLIT_ORDER"); off = e && !strcmp(e, "0"); if (off) MLOG("DM_NVN_BLIT_ORDER=0：拷贝编码器内不加屏障（诊断，会复现喷泉竖条）"); }
    // 计算编码器只有这些阶段：这里写 MTLStageAll 过不了 Metal 验证层（"must be a valid combination of MTLStages for the current encoder type"，MTL_DEBUG_LAYER=1 直接 abort）
    if (!off) [g_blit barrierAfterEncoderStages:MTLStageDispatch | MTLStageBlit beforeEncoderStages:MTLStageDispatch | MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
    g_bw_n = 0; g_blit_ops = 0; g_blit_barriers++;
}
uint64_t mtl_blit_barriers(void) { return g_blit_barriers; }
uint64_t mtl_blit_splits(void) { return g_pass_split; }
static int bw_hit(uint64_t lo, uint64_t hi) { for (int i = 0; i < g_bw_n; i++) if (lo < g_bw_hi[i] && g_bw_lo[i] < hi) return 1; return 0; }
// buffer → buffer / fill: reads [rlo, rhi) (0, 0 when nothing is read), writes [wlo, whi)
static id<MTL4ComputeCommandEncoder> blit_buf(uint64_t rlo, uint64_t rhi, uint64_t wlo, uint64_t whi) {
    id<MTL4ComputeCommandEncoder> e = blit_raw();
    if (g_blit_ops && (g_bw_n == BW_MAX || (rhi > rlo && bw_hit(rlo, rhi)) || bw_hit(wlo, whi))) blit_barrier();
    g_bw_lo[g_bw_n] = wlo; g_bw_hi[g_bw_n++] = whi; g_blit_ops++;
    return e;
}
// texture copies / readbacks: barrier if anything came before, and later operations must wait for it too
static id<MTL4ComputeCommandEncoder> blit(void) {
    id<MTL4ComputeCommandEncoder> e = blit_raw();
    if (g_blit_ops) blit_barrier();
    g_bw_lo[0] = 0; g_bw_hi[0] = ~0ull; g_bw_n = 1; g_blit_ops = 1;   // record as "everything written": the next operation always gets a barrier
    return e;
}

void mtl_sub_copy(uint64_t src, uint64_t dst, uint64_t n) {
    if (g_npend < PEND_MAX) { g_pend_lo[g_npend] = dst; g_pend_hi[g_npend++] = dst + n; } else g_pend_over = 1;
    void *sb, *db; uint64_t so, dof;
    if (!mtl_addr(src, n, &sb, &so) || !mtl_addr(dst, n, &db, &dof))
        MDIE("缓冲拷贝地址不在任何内存池内 src=%#llx dst=%#llx n=%#llx", (unsigned long long)src, (unsigned long long)dst, (unsigned long long)n);
    enc_b_copy_buffer(blit_buf(src, src + n, dst, dst + n), (__bridge id<MTLBuffer>)sb, so, (__bridge id<MTLBuffer>)db, dof, n);
}

// Inline uniform update (UpdateUniformBuffer): the data is copied into the host's own shared ring at encode time, then copied in order to the destination.
// Previously it was read from command memory only when the GPU executed - if the game recycles command memory before the GPU runs, it reads the bytes of the next batch of commands (SMBW's fountain water wave simulation got the wrong coefficients and diverged to ±65504).
static id<MTLBuffer> g_ubs_buf; static size_t g_ubs_off, g_ubs_size;
// Inline updates inside an open render pass are deferred instead of ending the pass. Metal cannot copy into a buffer inside a render pass, so each update used to end it: all attachments stored,
// reloaded by the next pass, new encoders. NVN applies the update in command order, so the draws after it in the pass must read the new data: a draw whose uniform binding overlaps a deferred update reads a shadow made only of the updates
// deferred in this pass, latest first per byte, and only when they cover the whole bound range (then the shadow is exactly what the GPU would read, whatever earlier frames still have in flight). The real copies are encoded in order when the pass ends
// (rp_end), so later passes and the CPU see them as before. The pass still ends before a draw that would need more: a binding the deferred updates cover only in part, or a read of the deferred data through a vertex, index, indirect or storage
// buffer (ubd_conflict). Off in TBDR mode (it reorders passes). DM_NVN_UB_DEFER=1 turns it on (default off: the local tree's behaviour must not change until it is measured)
static int enc_open(void);
static size_t arena_alloc(size_t n, id<MTLBuffer> *buf); static inline void args_dirty_all(void);
typedef struct { uint64_t lo, hi, gpu; size_t off; __unsafe_unretained id<MTLBuffer> buf; } ubd_ent;   // data at buf + off (the ubs ring; a ring block replaced meanwhile lives until this submission completes)
enum { UBD_MAX = 512, UBD_SHADOW_MAX = 64 };
static ubd_ent g_ubd[UBD_MAX]; static int g_nubd;
typedef struct { uint64_t lo, hi, gpu; int nubd; } ubd_shadow_t;   // a laid-out shadow in the arena, for one (range, deferred-set)
static ubd_shadow_t g_ubd_shadow[UBD_SHADOW_MAX]; static int g_ubd_nshadow;
static uint64_t g_ubd_deferred, g_ubd_conflicts, g_ubd_flushes;   // updates deferred, passes ended before a draw that needed more, deferred batches flushed at a pass end
static int ubd_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_NVN_UB_DEFER"); v = e && !strcmp(e, "1"); if (v) MLOG("DM_NVN_UB_DEFER=1：通道内的 uniform 更新推迟到通道结束再拷贝"); else MLOG("DM_NVN_UB_DEFER 缺省关（DM_NVN_UB_DEFER=1 开）"); } return v && !tbdr_on(); }
void mtl_ubd_stats(uint64_t *deferred, uint64_t *conflicts, uint64_t *flushes) { *deferred = g_ubd_deferred; *conflicts = g_ubd_conflicts; *flushes = g_ubd_flushes; }
static int ubd_hits(uint64_t lo, uint64_t hi) { for (int i = 0; i < g_nubd; i++) if (lo < g_ubd[i].hi && g_ubd[i].lo < hi) return 1; return 0; }
// Whether the deferred updates cover [lo, hi) entirely (any order of overlap): the union of their ranges clipped to it, swept from lo
static int ubd_covers(uint64_t lo, uint64_t hi) {
    for (uint64_t at = lo; at < hi;) { uint64_t next = at;
        for (int i = 0; i < g_nubd; i++) if (g_ubd[i].lo <= at && g_ubd[i].hi > next) next = g_ubd[i].hi;
        if (next == at) return 0; at = next; }
    return 1;
}
// GPU address of the shadow of uniform range [lo, hi) (ubd_covers holds): one deferred update holding all of it and suitably aligned is used in place; otherwise the updates are laid into the arena in order
// The laid-out shadow is memoized per (range, deferred-set): many draws in one pass read the same uniform range, and a fresh arena block per draw would fill the arena
static uint64_t ubd_shadow(uint64_t lo, uint64_t hi) {
    for (int i = g_nubd - 1; i >= 0; i--) { const ubd_ent *e = &g_ubd[i];
        if (e->lo <= lo && hi <= e->hi) { int later = 0; for (int j = i + 1; j < g_nubd; j++) later |= lo < g_ubd[j].hi && g_ubd[j].lo < hi;
            uint64_t g = e->gpu + (lo - e->lo); if (!later && !(g & 15)) return g; break; } }
    for (int i = 0; i < g_ubd_nshadow; i++) { ubd_shadow_t *s = &g_ubd_shadow[i]; if (s->lo == lo && s->hi == hi && s->nubd == g_nubd) return s->gpu; }
    id<MTLBuffer> ab; size_t o = arena_alloc((size_t)(hi - lo), &ab); uint8_t *d = (uint8_t *)ab.contents + o;
    for (int i = 0; i < g_nubd; i++) { const ubd_ent *e = &g_ubd[i]; uint64_t a = e->lo > lo ? e->lo : lo, b = e->hi < hi ? e->hi : hi;
        if (a < b) memcpy(d + (a - lo), (const uint8_t *)e->buf.contents + e->off + (a - e->lo), (size_t)(b - a)); }
    uint64_t g = ab.gpuAddress + o;
    if (g_ubd_nshadow < UBD_SHADOW_MAX) { ubd_shadow_t *s = &g_ubd_shadow[g_ubd_nshadow++]; s->lo = lo; s->hi = hi; s->nubd = g_nubd; s->gpu = g; }
    return g;
}
// Encode the deferred copies in order (rp_end, once the pass has ended)
static void ubd_flush(void) {
    int n = g_nubd; g_nubd = 0; g_ubd_nshadow = 0;   // the laid-out shadows belong to the deferred set that is now gone
    if (!n) return;
    g_ubd_flushes++;
    for (int i = 0; i < n; i++) { const ubd_ent *e = &g_ubd[i]; void *db; uint64_t dof;
        if (!mtl_addr(e->lo, e->hi - e->lo, &db, &dof)) continue;
        enc_b_copy_buffer(blit_buf(0, 0, e->lo, e->hi), e->buf, e->off, (__bridge id<MTLBuffer>)db, dof, e->hi - e->lo); }
}
// The bytes [lo, lo+n) as the deferred updates leave them (ubd_covers holds): for bindless handles read from uniform memory (ubo_handle)
static void ubd_read(uint64_t lo, size_t n, void *out) {
    for (int i = 0; i < g_nubd; i++) { const ubd_ent *e = &g_ubd[i]; uint64_t a = e->lo > lo ? e->lo : lo, b = e->hi < lo + n ? e->hi : lo + n;
        if (a < b) memcpy((uint8_t *)out + (a - lo), (const uint8_t *)e->buf.contents + e->off + (a - e->lo), (size_t)(b - a)); }
}
void mtl_sub_update(const void *data, uint64_t dst, uint64_t n) {
    if (!n) return;
    if (g_npend < PEND_MAX) { g_pend_lo[g_npend] = dst; g_pend_hi[g_npend++] = dst + n; } else g_pend_over = 1;
    void *db; uint64_t dof;
    if (!mtl_addr(dst, n, &db, &dof)) MDIE("uniform 更新目标不在任何内存池内 dst=%#llx n=%#llx", (unsigned long long)dst, (unsigned long long)n);
    int defer = enc_open() && g_nubd < UBD_MAX && ubd_on();
    size_t need = (n + 255u) & ~(size_t)255u;   // 256-aligned slots: a deferred update is read in place by draws as a uniform buffer
    if (!g_ubs_buf || g_ubs_off + need > g_ubs_size) {
        if (g_ubs_buf) defer_obj_release(g_ubs_buf);   // copies already encoded still reference the old block: release after this submission completes
        g_ubs_size = MAX((size_t)1 << 20, need);
        g_ubs_buf = [g_dev newBufferWithLength:g_ubs_size options:MTLResourceStorageModeShared];
        if (!g_ubs_buf) MDIE("uniform 更新环分配失败 %#zx", g_ubs_size);
        res_add(g_ubs_buf); g_ubs_off = 0;
    }
    memcpy((uint8_t *)g_ubs_buf.contents + g_ubs_off, data, n);
    if (defer) { g_ubd[g_nubd++] = (ubd_ent){dst, dst + n, g_ubs_buf.gpuAddress + g_ubs_off, g_ubs_off, g_ubs_buf}; g_ubd_deferred++; g_ubd_nshadow = 0; args_dirty_all(); }   // see ubd_shadow
    else enc_b_copy_buffer(blit_buf(0, 0, dst, dst + n), g_ubs_buf, g_ubs_off, (__bridge id<MTLBuffer>)db, dof, n);
    g_ubs_off += need;
}

// In-order buffer fill within the submission: all four bytes equal → fillBuffer; otherwise copied in chunks from a 64 KiB pattern buffer
void mtl_sub_fill(uint64_t dst, uint64_t n, uint32_t value) {
    if (!n) return;
    if (g_npend < PEND_MAX) { g_pend_lo[g_npend] = dst; g_pend_hi[g_npend++] = dst + n; } else g_pend_over = 1;
    void *db; uint64_t dof;
    if (!mtl_addr(dst, n, &db, &dof)) MDIE("缓冲填充地址不在任何内存池内 dst=%#llx n=%#llx", (unsigned long long)dst, (unsigned long long)n);
    uint8_t b0 = (uint8_t)value;
    if (value == (uint32_t)b0 * 0x01010101u) { [blit_buf(0, 0, dst, dst + n) fillBuffer:(__bridge id<MTLBuffer>)db range:NSMakeRange(dof, n) value:b0]; return; }
    enum { PAT = 0x10000 }; static __strong id<MTLBuffer> pat; static uint32_t pat_v; static int pat_ok;
    if (!pat_ok || pat_v != value) { uint32_t *tmp = malloc(PAT); for (unsigned i = 0; i < PAT / 4; i++) tmp[i] = value;
        pat = [g_dev newBufferWithBytes:tmp length:PAT options:MTLResourceStorageModeShared]; free(tmp); pat_v = value; pat_ok = 1; res_add(pat); }
    for (uint64_t off = 0; off < n; off += PAT) { uint64_t k = n - off < PAT ? n - off : PAT; enc_b_copy_buffer(blit_buf(0, 0, dst + off, dst + off + k), pat, 0, (__bridge id<MTLBuffer>)db, dof + off, k); }
}

static void post(post_op o) {
    if (g_npost == g_postcap) { g_postcap = g_postcap ? g_postcap * 2 : 64; g_post = realloc(g_post, g_postcap * sizeof *g_post); }
    g_post[g_npost++] = o;
}
void mtl_sub_event(volatile uint32_t *val, uint32_t mode, uint32_t v) { post((post_op){1, val, mode, v, 0, 0, NULL}); }
void mtl_sub_counter(uint64_t addr, uint32_t type) { post((post_op){2, NULL, 0, 0, addr, type, NULL}); }
void mtl_counter_stats(uint64_t rep[32], uint64_t rst[32]) { memcpy(rep, g_cnt_report, sizeof g_cnt_report); memcpy(rst, g_cnt_reset, sizeof g_cnt_reset); }

// Counter timestamps (19.2 MHz) are taken from the guest clock (hle_os.c guest_ns, the same source as svc GetSystemTick / cntpct): on hardware GPU timestamps and the CPU counter share one timebase, and games compare them.
// They used to come from CLOCK_MONOTONIC, which keeps running while the Mac sleeps while the guest clock (mach_absolute_time) does not, so the two differed by the sleep time; once timestamps were converted to ns, Mario halved its frame rate. In graphics unit tests guest_ns is a weak definition returning 0; fall back to the monotonic clock
uint64_t guest_ns(void);
static uint64_t gpu_ticks(void) { uint64_t g = guest_ns(); if (!g) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); g = (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec; } return g * 12 / 625; }

// ---------- Boot instrumentation (DM_BOOTPROF=1): metallib loading, pipeline creation (both happen synchronously during replay on the submit thread), texture upload, first-frame present ----------
static uint64_t bp_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec; }
static uint64_t g_bp_t0;   // Time of process start (when this module is loaded)
__attribute__((constructor)) static void bp_t0(void) { g_bp_t0 = bp_now(); }
static int bootprof(void) { static int v = -1; if (v < 0) v = getenv("DM_BOOTPROF") != NULL; return v; }
static uint64_t g_bp_lib_n, g_bp_lib_ns, g_bp_pso_n, g_bp_pso_ns, g_bp_up_bytes, g_bp_first_present;
static uint64_t g_up_total_ns;   // Cumulative upload time (same source as the per-frame @upload_ms)
// ---------- Per-frame GPU time aggregation ----------
#define GRING 1024
static struct { _Atomic uint64_t frame, ns, t0, t1; _Atomic int pending; } g_gpu[GRING];   // t0 / t1: the frame's earliest GPUStart / latest GPUEnd (mach absolute time, ns)
static _Atomic uint64_t g_cur_frame;
// ---------- Same-process alternating measurement (DM_NVN_UB, off by default: without the environment variable the code below only adds one 'is it enabled' check) ----------
// On a hot real device the system keeps adjusting power / frequency, and segment comparison (segments A and B each run for a minute) cannot resolve differences of <10%; within the same run, toggle by phase in turn and take differences of adjacent phase pairs, so the drift cancels.
// DM_NVN_UB="name=item,item;name=item", one phase every DM_NVN_UB_N frames (default 180, minimum 60): even phases = baseline (nothing changed), odd phases take one category in turn. Each item of a category:
//   @set:<global variable name>=<value>   on entering the phase set this int global variable to the value, on leaving restore it (address found by symbol name; switches to be used on a real device must have a line added to the registration table below)
//   <fragment shader key prefix> or *   the phase drops all draws of these fragment shaders (the picture will be wrong; it only measures 'how much this kind of draw is worth'; a nearly idle category needs a round of its own)
// Readings: each phase discards its first 45 frames (transition + present lag); for the remaining frames, the mean of the actual present interval (same source as perf.txt) and the frame counts in the three buckets 16.7 / 25 / ≥33 ms, one log line 「UB 上屏」(UB present).
// Table output: $DM_ROOT/run/ruler/ub_table.py; real-device script: $DM_ROOT/run/ruler/ub_alt.sh.
static int g_ub_n = -1, g_ub_N = 180; static _Atomic int g_ub_cur = -1; static char g_ub_name[16][16], g_ub_keys[16][16][48]; static int g_ub_nk[16];
static _Atomic int g_ubp_on, g_ubp_c[4]; static _Atomic uint64_t g_ubp_sum_us;
void mtl_ub_present(double iv) {   // Called once per frame by the present thread: actual present interval in ms (−1 = dropped, 0 = no reading)
    if (!atomic_load(&g_ubp_on) || iv == 0) return;
    if (iv < 0) { atomic_fetch_add(&g_ubp_c[3], 1); return; }
    atomic_fetch_add(&g_ubp_c[iv <= 20.8 ? 0 : iv <= 29.2 ? 1 : 2], 1); atomic_fetch_add(&g_ubp_sum_us, (uint64_t)(iv * 1000));
}
// Registered runtime switches: name → address. Main-program symbols of an iOS release build are not in the dynamic symbol table and dlsym cannot find them (in the first real-device round all four categories were 「找不到全局变量」(global variable not found) and the whole round was wasted),
// so a switch to be measured on a real device must have a line added here; on Mac / in unit tests unregistered ones are still looked up by symbol name.
static int *ub_var(const char *nm) {
    extern int g_bloom_q, g_dm_present_link;
    static const struct { const char *n; int *p; } t[] = { {"g_bloom_q", &g_bloom_q}, {"g_dm_present_link", &g_dm_present_link}, {"g_frz_on", &g_frz_on} };
    for (size_t i = 0; i < sizeof t / sizeof t[0]; i++) if (!strcmp(nm, t[i].n)) return t[i].p;
    return dlsym(RTLD_DEFAULT, nm);
}
static void ub_frame(uint64_t f) {
    if (g_ub_n < 0) { g_ub_n = 0; const char *e = getenv("DM_NVN_UB"), *n = getenv("DM_NVN_UB_N"); if (n && atoi(n) >= 60) g_ub_N = atoi(n);
        for (const char *q = e; q && *q && g_ub_n < 16; ) { const char *eq = strchr(q, '='), *end = strchr(q, ';'); if (!end) end = q + strlen(q); if (!eq || eq > end) break;
            int c = g_ub_n++; size_t l = (size_t)(eq - q); if (l > 15) l = 15; memcpy(g_ub_name[c], q, l);
            for (const char *k = eq + 1; k < end && g_ub_nk[c] < 16; ) { size_t kl = 0; while (k + kl < end && k[kl] != ',') kl++; size_t cl = kl > 47 ? 47 : kl; memcpy(g_ub_keys[c][g_ub_nk[c]++], k, cl); k += kl; if (k < end) k++; }
            q = *end ? end + 1 : end; }
        if (g_ub_n) MLOG("DM_NVN_UB：%d 类，每相 %d 帧（同进程交替测量；丢绘制的相画面不对）", g_ub_n, g_ub_N); }
    if (!g_ub_n) return;
    uint64_t ph = f / (uint64_t)g_ub_N, r = f % (uint64_t)g_ub_N;
    if (r == 45) { for (int i = 0; i < 4; i++) atomic_store(&g_ubp_c[i], 0); atomic_store(&g_ubp_sum_us, 0); atomic_store(&g_ubp_on, 1); }
    if (r != 0) return;
    if (f && atomic_load(&g_ubp_on)) { atomic_store(&g_ubp_on, 0); uint64_t pp = ph - 1; int pc = (pp & 1) ? (int)((pp / 2) % (uint64_t)g_ub_n) : -1;
        int c0 = atomic_load(&g_ubp_c[0]), c1 = atomic_load(&g_ubp_c[1]), c2 = atomic_load(&g_ubp_c[2]), c3 = atomic_load(&g_ubp_c[3]), n_ = c0 + c1 + c2;
        MLOG("UB 上屏 %llu %s 帧 %d 均 %.3f ms 2拍 %d 3拍 %d 4拍以上 %d 丢弃 %d", (unsigned long long)pp, pc < 0 ? "base" : g_ub_name[pc], n_, n_ ? atomic_load(&g_ubp_sum_us) / 1000.0 / n_ : 0.0, c0, c1, c2, c3); }
    int nc = (ph & 1) ? (int)((ph / 2) % (uint64_t)g_ub_n) : -1;
    static int *sp[16]; static int sv[16], ns;   // Variables set by the previous phase: restore first
    for (int i = 0; i < ns; i++) *sp[i] = sv[i]; ns = 0;
    for (int i = 0; nc >= 0 && i < g_ub_nk[nc] && ns < 16; i++) { const char *k = g_ub_keys[nc][i]; if (strncmp(k, "@set:", 5)) continue;
        char nm[48]; const char *eq = strchr(k, '='); if (!eq) continue; size_t l = (size_t)(eq - (k + 5)); if (l >= sizeof nm) continue; memcpy(nm, k + 5, l); nm[l] = 0;
        int *a_ = ub_var(nm); static int bad, ok;
        if (!a_) { if (bad++ < 8) MLOG("【红】DM_NVN_UB：找不到全局变量 %s（真机上要先在 ub_var 的表里登记）", nm); continue; }
        sp[ns] = a_; sv[ns] = *a_; ns++; *a_ = atoi(eq + 1); if (ok++ < 8) MLOG("DM_NVN_UB：%s = %d（原值 %d）", nm, atoi(eq + 1), sv[ns - 1]); }
    atomic_store(&g_ub_cur, nc);
}
static int ub_skip(const char *key) {   // Whether this phase should drop draws of this fragment shader
    int uc = atomic_load(&g_ub_cur); if (uc < 0) return 0;
    for (int i = 0; i < g_ub_nk[uc]; i++) { const char *k = g_ub_keys[uc][i]; if (k[0] == '@') continue; if (k[0] == '*' || !strncmp(key, k, strlen(k))) return 1; }
    return 0;
}
void mtl_set_frame(uint64_t f) { atomic_store(&g_cur_frame, f); if (g_ub_n) ub_frame(f); }
// GPU timing registration: gpu_tag_begin(frame) before commit, gpu_tag_done(frame, GPU start, GPU end) on completion (seconds on the CACurrentMediaTime base).
// Old-queue command buffers go through gpu_tag_f; the Metal 4 main submission backfills in the feedback of mtl_sub_commit (when switching to Metal 4 all the call sites were lost and the GPU column of the 「计时」(timing) line became empty)
static void gpu_tag_begin(uint64_t f) {
    unsigned k = f % GRING;
    if (atomic_load(&g_gpu[k].frame) != f) { atomic_store(&g_gpu[k].frame, f); atomic_store(&g_gpu[k].ns, 0); atomic_store(&g_gpu[k].pending, 0); atomic_store(&g_gpu[k].t0, UINT64_MAX); atomic_store(&g_gpu[k].t1, 0); }
    atomic_fetch_add(&g_gpu[k].pending, 1);
}
static void gpu_tag_done(uint64_t f, double g0, double g1) {
    unsigned k = f % GRING;
    if (atomic_load(&g_gpu[k].frame) != f) return;
    double d = g1 - g0;
    if (d > 0) {
        atomic_fetch_add(&g_gpu[k].ns, (uint64_t)(d * 1e9));
        uint64_t s0 = (uint64_t)(g0 * 1e9), s1 = (uint64_t)(g1 * 1e9), v;
        for (v = atomic_load(&g_gpu[k].t0); s0 < v && !atomic_compare_exchange_weak(&g_gpu[k].t0, &v, s0);) {}
        for (v = atomic_load(&g_gpu[k].t1); s1 > v && !atomic_compare_exchange_weak(&g_gpu[k].t1, &v, s1);) {}
    }
    atomic_fetch_sub(&g_gpu[k].pending, 1);
}
static void gpu_tag_f(id<MTLCommandBuffer> cb, uint64_t f) { gpu_tag_begin(f); [cb addCompletedHandler:^(id<MTLCommandBuffer> c) { gpu_tag_done(f, c.GPUStartTime, c.GPUEndTime); }]; }
void gpu_tag(id<MTLCommandBuffer> cb) { gpu_tag_f(cb, atomic_load(&g_cur_frame)); }
double mtl_gpu_ms(uint64_t f) {
    unsigned k = f % GRING;
    if (atomic_load(&g_gpu[k].frame) != f || atomic_load(&g_gpu[k].pending)) return -1;
    return atomic_load(&g_gpu[k].ns) / 1e6;
}
int mtl_gpu_span(uint64_t f, uint64_t *t0, uint64_t *t1) {
    unsigned k = f % GRING;
    if (atomic_load(&g_gpu[k].frame) != f || atomic_load(&g_gpu[k].pending)) return 0;
    *t0 = atomic_load(&g_gpu[k].t0); *t1 = atomic_load(&g_gpu[k].t1);
    return *t1 >= *t0 && *t1;
}
static void rp_flush(void);
static const char *cctx_snap(void);   // What is in this submission (printed into the failure line on error; defined before mtl_exec)
uint64_t mtl_sub_commit(void) {
    defer_fallback();   // Deferred merged pass: at the end of the submission there are still P draws not replayed (the normal path never gets here)
    rp_flush();
    if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
    up_flush();
    up_wait();   // The upload's staging blocks are resident until its feedback arrives; free them before the main segment is submitted
    uint64_t fence = ++g_fence;
    if (g_vis_ring_i >= 0) g_vis_ring[g_vis_ring_i].fence = fence;
    // CPU-visible effects after completion: copy the op table into the callback
    int n = g_npost; post_op *ops = n ? malloc(n * sizeof *ops) : NULL;
    if (n) memcpy(ops, g_post, n * sizeof *ops);
    NSArray<dispatch_block_t> *commit_blocks = [g_commit_blocks copy];
    MTL4CommitOptions *o = [MTL4CommitOptions new];
    [o addFeedbackHandler:^(id<MTL4CommitFeedback> f) { (void)f; for (dispatch_block_t block in commit_blocks) block(); }];
    uint64_t gpu_f = atomic_load(&g_cur_frame); gpu_tag_begin(gpu_f);
    NSString *cctx = @(cctx_snap());
    [o addFeedbackHandler:^(id<MTL4CommitFeedback> cb) {
        gpu_tag_done(gpu_f, cb.GPUStartTime, cb.GPUEndTime);
        if (cb.error) { atomic_store(&g_last_commit_error, 1); static int n, nv; int victim = [cb.error.description containsString:@"InnocentVictim"]; if (victim ? nv++ < 1 : n++ < 8) MLOG("【红】命令缓冲错误详情 fence=%llu：%s | GPU %.3f→%.3f", (unsigned long long)fence, cb.error.description.UTF8String, cb.GPUStartTime, cb.GPUEndTime); if (!victim) MLOG("【红】出错那次提交的现场 fence=%llu：%s", (unsigned long long)fence, cctx.UTF8String); MLOG("【红】命令缓冲执行出错：%s", cb.error.localizedDescription.UTF8String); }
        for (int i = 0; i < n; i++) {
            if (ops[i].kind == 1) { if (ops[i].mode == 1) *ops[i].val += ops[i].v; else *ops[i].val = ops[i].v; }
            else if (ops[i].kind == 3) {   // Samples passed: the sum of all slots since the reset
                vis_list *l = ops[i].vis; uint64_t sum = 0;
                for (int k = 0; k < l->n; k++) { sum += ((const uint64_t *)((__bridge id<MTLBuffer>)l->s[k].buf).contents)[l->s[k].slot]; CFRelease(l->s[k].buf); }
                free(l); uint64_t rec[2] = {sum, gpu_ticks()}; memcpy((void *)ops[i].addr, rec, sizeof rec);
            }
            else { uint64_t t = gpu_ticks(), rec[2] = {ops[i].type == 0 || ops[i].type == 0x10 ? t : 0, t}; memcpy((void *)ops[i].addr, rec, sizeof rec); }
        }
        free(ops);
        pthread_mutex_lock(&g_mu); if (fence > g_done) g_done = fence; pthread_cond_broadcast(&g_cv); pthread_mutex_unlock(&g_mu);
        defer_drain();
    }];
    void arena_commit(uint64_t fence); arena_commit(fence);
    res_commit();
    [g_cb endCommandBuffer];
    id<MTL4CommandBuffer> buffers[] = {g_cb};
    [g_q4 commit:buffers count:1 options:o];
    [g_q4 signalEvent:g_ev value:fence];
    g_alloc_fence[g_alloc_cur] = fence;
    g_cb = nil;
    g_commit_blocks = nil;
    objc_autoreleasePoolPop(g_sub_pool); g_sub_pool = NULL;
    dead_watch("提交后");
    return fence;
}

uint64_t mtl_last_fence(void) { return g_fence; }
int mtl_last_commit_error(void) { return atomic_load(&g_last_commit_error); }

int mtl_wait(uint64_t fence, uint64_t timeout_ns) {
    struct timespec dl; clock_gettime(CLOCK_REALTIME, &dl);
    uint64_t ns = (uint64_t)dl.tv_nsec + (timeout_ns > 3600000000000ull ? 3600000000000ull : timeout_ns);
    dl.tv_sec += ns / 1000000000ull; dl.tv_nsec = ns % 1000000000ull;
    pthread_mutex_lock(&g_mu);
    int r = 0;
    while (g_done < fence) if ((r = pthread_cond_timedwait(&g_cv, &g_mu, &dl))) break;
    int ok = g_done >= fence;
    pthread_mutex_unlock(&g_mu);
    return ok;
}

static void mtl_finish_gpu(void) {
    if (g_fence) mtl_wait(g_fence, UINT64_MAX);
    while (atomic_load(&g_up_pending)) usleep(1000);
}
void mtl_present_flush(void);
void mtl_finish(void) { mtl_present_flush(); mtl_finish_gpu(); }   // Queue semantics: already-enqueued presents also count as queue work
static void present_drop_for_suspend(void);
void mtl_drain(void) { present_drop_for_suspend(); mtl_present_flush(); mtl_init(); id<MTLCommandBuffer> cb = q_cb(); [cb commit]; [cb waitUntilCompleted]; mtl_finish_gpu(); }

// ================= Render replay =================
#include "../../host/front.h"
// The front end's picture-rect hook: the default says the picture fills the frame (no bars). The iOS front end implements it
// (the on-screen pad moves the shoulder buttons into the bars). Weak, so a front end that does not care does not have to.
__attribute__((weak)) void front_set_picture_rect(double x, double w) { (void)x; (void)w; }
static int g_red_once[96];
#define RED_ONCE(k, ...) do { if (!g_red_once[k]++) MLOG("【红】" __VA_ARGS__); } while (0)
// NVN vertex streams 0–15 → Metal vertex buffer slots: 0–13 go to 16–29 (30 is the constant fallback buffer), 14 and 15 to 14 and 15. Metal has only the 31 slots 0–30; translated shaders use only buffer(0) (the argument buffer),
// and the host's own vertex functions read no vertex buffers. 13 Sentinels uses stream 15, and the old 16 + stream went out of range (Metal assertion: buffer index (31) must be < 31)
#define VB_SLOT(st) ((st) < 14 ? 16 + (st) : (st))

// ---------- Texture → MTLTexture ----------
static int fmt_compressed(int f) { return (f >= 0x41 && f <= 0x50) || (f >= 0x79 && f <= 0x94); }
static int fmt_depth(int f) { return f >= 0x31 && f <= 0x36; }
typedef struct { MTLPixelFormat a, b; } pfv_pair;
static const pfv_pair g_pfv_srgb[] = {
    {MTLPixelFormatRGBA8Unorm, MTLPixelFormatRGBA8Unorm_sRGB},
    {MTLPixelFormatBGRA8Unorm, MTLPixelFormatBGRA8Unorm_sRGB},
    {MTLPixelFormatBC1_RGBA, MTLPixelFormatBC1_RGBA_sRGB},
    {MTLPixelFormatBC2_RGBA, MTLPixelFormatBC2_RGBA_sRGB},
    {MTLPixelFormatBC3_RGBA, MTLPixelFormatBC3_RGBA_sRGB},
    {MTLPixelFormatBC7_RGBAUnorm, MTLPixelFormatBC7_RGBAUnorm_sRGB},
};
static int tex_need_pfv(MTLPixelFormat base, MTLPixelFormat view) {
    if (base == view) return 0;
    for (size_t i = 0; i < sizeof g_pfv_srgb / sizeof *g_pfv_srgb; i++)
        if ((base == g_pfv_srgb[i].a && view == g_pfv_srgb[i].b) || (base == g_pfv_srgb[i].b && view == g_pfv_srgb[i].a)) return 0;
    return 1;
}
// Cache tables on the replay hot path always use nvn_hmap (C hash, keys not boxed); the table stores CFBridgingRetain'ed objects, CFRelease'd on replace / delete
static void hm_rel(void *v) { CFRelease(v); }
static __unsafe_unretained id hm_get_id(const nvn_hmap *m, const void *k) { return (__bridge id)nvn_hm_get(m, k); }
static void hm_put_id(nvn_hmap *m, const void *k, id o) { void *old = nvn_hm_put(m, k, (void *)CFBridgingRetain(o)); if (old) CFRelease(old); }
static void hm_del_id(nvn_hmap *m, const void *k) { void *old = nvn_hm_del(m, k); if (old) CFRelease(old); }
static nvn_hmap g_by_serial = {.klen = 8};     // Serial → MTLTexture / NvPkg / NvProg
static pthread_mutex_t g_smu = PTHREAD_MUTEX_INITIALIZER;
// Serial table generation: +1 when entries are added / removed or a packaged texture is replaced. Downstream caches (texture views, programs) record the generation at fetch time; if it is unchanged they can be used directly without another table lookup
static _Atomic uint64_t g_ser_gen = 1;
// Textures shared by storage (non-packaged): key = memory pool + offset + texture description. An NVN texture object is only a descriptor; an object rebuilt over the same storage with the same description is the same texture;
// SMBW rebuilds its transient render targets on new objects every frame and never Finalizes them, so building one per serial leaked about 13 MB of GPU memory per frame. Textures here are kept resident by the sharing table,
// and when the serial table replaces / deletes them it only drops its reference (defer_obj_unref), never removing them from the residency set. The sharing table does not evict (entries = distinct storage + description combinations).
typedef struct { uint64_t pool; int64_t off; nvn_texdesc d; } alias_key;
static nvn_hmap g_alias = {.klen = sizeof(alias_key)};
// The last frame each serial was used, for the shared views (the private textures keep it in their NvPkg). A view the game has not drawn
// for a long time is a view of a scene the player has left: the alias table keeps it alive, and with it the pool's buffer.
#define VU_MAX 4096
static _Atomic uint64_t g_vu_key[VU_MAX];
static _Atomic uint64_t g_vu_frame[VU_MAX];
// The per-view last use, keyed by the view's pointer: a view can be shared by several serials, and the game drawing it through any of
// them is a use. The serial table maps serial -> the view, so the use site knows the view.
static void view_use_obj(id view) {
    if (!view) return;
    uintptr_t p = (uintptr_t)view;
    uint64_t h = (p * 0x9E3779B97F4A7C15ull) >> 40;
    for (unsigned k = 0; k < 64; k++) {
        uint64_t idx = (h + k) & (VU_MAX - 1);
        uint64_t cur = atomic_load_explicit(&g_vu_key[idx], memory_order_relaxed);
        if (cur == p) { atomic_store_explicit(&g_vu_frame[idx], atomic_load(&g_cur_frame), memory_order_relaxed); return; }
        if (!cur) {
            uint64_t exp = 0;
            if (atomic_compare_exchange_strong(&g_vu_key[idx], &exp, p)) { atomic_store_explicit(&g_vu_frame[idx], atomic_load(&g_cur_frame), memory_order_relaxed); return; }
            if (exp == p) { atomic_store_explicit(&g_vu_frame[idx], atomic_load(&g_cur_frame), memory_order_relaxed); return; }
        }
    }
}
static uint64_t view_last_use_obj(id view) {
    if (!view) return 0;
    uintptr_t p = (uintptr_t)view;
    uint64_t h = (p * 0x9E3779B97F4A7C15ull) >> 40;
    for (unsigned k = 0; k < 64; k++) {
        uint64_t idx = (h + k) & (VU_MAX - 1);
        uint64_t cur = atomic_load_explicit(&g_vu_key[idx], memory_order_relaxed);
        if (!cur) return 0;
        if (cur == p) return atomic_load_explicit(&g_vu_frame[idx], memory_order_relaxed);
    }
    return 0;
}

static CFMutableSetRef g_alias_set;   // Textures in the sharing table (non-owning; used to ask "is this shared")
static uint64_t g_alias_evicted;   // shared textures evicted on pool Finalize so far (for the memory report)
static int alias_has(id o) { return g_alias_set && o && CFSetContainsValue(g_alias_set, (__bridge const void *)o); }
// On memory pool Finalize, evict the textures built on that pool from the sharing table (collect the keys first, then delete: deletion shifts entries back). The sharing table used not to evict: Mario's pools stay for good, so entries are bounded;
// Engage frees pools on scene changes and rebuilds on new ones, so the old pools' render targets and other private textures stayed in the table forever (about 0.8 GB more GPU memory over 850 s of the title demo), and linear textures built directly on a pool also kept the whole pool buffer alive.
// Also, when a pool object's address is reused, old entries would hand a texture on the old storage to the new pool's texture at the same offset and description. Evicted textures go through deferred release; if the serial table still holds one, that is just one more reference
static void treg_drop_pool_views(const void *pool);
// Drop the zero-copy views over a pool. Returns how many were dropped. The views leave the residency set here: they are Metal
// resources like any other, and the pool's buffer is about to go too.
int mtl_alias_drop_pool(const void *pool) {
    if (!pool) return 0;
    pthread_mutex_lock(&g_smu);
    uint32_t n = 0; alias_key *ks = NULL;
    for (uint32_t i = 0; i < g_alias.cap; i++) {
        if (!g_alias.vals[i]) continue;
        alias_key k; memcpy(&k, g_alias.keys + (size_t)i * g_alias.klen, sizeof k);
        if (k.pool != (uint64_t)(uintptr_t)pool) continue;
        alias_key *nk = realloc(ks, (n + 1) * sizeof *ks); if (!nk) break; ks = nk; ks[n++] = k;
    }
    for (uint32_t j = 0; j < n; j++) {
        void *old = nvn_hm_del(&g_alias, &ks[j]);
        if (!old) continue;
        id o = CFBridgingRelease(old);
        if (g_alias_set) CFSetRemoveValue(g_alias_set, (__bridge const void *)o);
        if ([o conformsToProtocol:@protocol(MTLResource)]) res_remove((id<MTLResource>)o);   // the residency set: the view's GPU memory goes back
        // The game's texture objects that point at this view: their serial entries go too, so the game cannot draw a view that is
        // no longer resident. A later use rebuilds the view (the alias table no longer has it) over the pool's bytes.
        for (uint32_t i = 0; i < g_by_serial.cap; i++) {
            if (!g_by_serial.vals[i]) continue;
            if ((__bridge id)g_by_serial.vals[i] != o) continue;
            uint64_t key; memcpy(&key, g_by_serial.keys + (size_t)i * g_by_serial.klen, sizeof key);
            void *sv = nvn_hm_del(&g_by_serial, &key);
            if (sv) CFRelease(sv);
        }
        defer_obj_release(o);
    }
    if (n) { g_ser_gen++; g_alias_evicted += n; }
    pthread_mutex_unlock(&g_smu);
    free(ks);
    treg_drop_pool_views(pool);   // outside g_smu: the texture ID registry lock g_rmu is taken before g_smu
    return (int)n;
}
static __unsafe_unretained id serial_get(uint64_t k) { pthread_mutex_lock(&g_smu); __unsafe_unretained id o = hm_get_id(&g_by_serial, &k); pthread_mutex_unlock(&g_smu); return o; }
static void serial_put(uint64_t k, id o) { pthread_mutex_lock(&g_smu); id old = hm_get_id(&g_by_serial, &k); if (old) { if (alias_has(old)) defer_obj_unref(old); else defer_obj_release(old); } hm_put_id(&g_by_serial, &k, o); g_ser_gen++; pthread_mutex_unlock(&g_smu); }
static void serial_del(uint64_t k) { pthread_mutex_lock(&g_smu); id old = hm_get_id(&g_by_serial, &k); if (old) { if (alias_has(old)) defer_obj_unref(old); else defer_obj_release(old); } hm_del_id(&g_by_serial, &k); g_ser_gen++; pthread_mutex_unlock(&g_smu); }
// Host record of a packaged texture: the current MTLTexture, the upload stamp, and the global write generation at the last check
@interface NvPkg : NSObject { @public id<MTLTexture> tex; uint64_t stamp, seen, hash; uint64_t uses, reuploads; uint64_t up_frame; uint64_t lasthash; uint64_t last_use_frame;
    uint64_t *phash;   // the hash of each page of the source at the time of the upload (the reference for the content comparison)
    int nphash;   // the number of pages covered
} @end   // hash: full hash of the uploaded content under DM_NVN_DIRTYCHECK; up_frame: frame of the last upload (DM_POOL_CENSUS); lasthash: the content hash at the previous reupload (DM_NVN_TEXREUPLOAD=hash)
@implementation NvPkg
- (void)dealloc { res_remove(tex); }
@end
static uint64_t g_pkg_uploads, g_pkg_reuploads, g_pkg_miss, g_pkg_inplace, g_pkg_sameskip;   // g_pkg_sameskip: reuploads skipped because the content had not changed
static uint64_t g_pfv_upgrades;
// DM_NVN_DIRTYCHECK=1 (diagnostic, compared against ground truth): fully hash the pool content on every use; changed but not detected by dirty-page tracking = a missed detection
static int dirtycheck(void) { static int v = -1; if (v < 0) v = getenv("DM_NVN_DIRTYCHECK") != NULL; return v; }
static int texreupload_diag(void) { static int v = -1; if (v < 0) v = getenv("DM_NVN_TEXREUPLOAD") != NULL; return v; }
// DM_NVN_TEXREUPLOAD=count: log every reupload (the first four of a texture, then the powers of two), so a run can tell
// "one texture reuploaded over and over" from "many textures once each" -- the first-reupload-only diagnostic cannot.
static int texreupload_count_diag(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_NVN_TEXREUPLOAD"); v = e && strcmp(e, "count") == 0; } return v; }
// DM_NVN_TEXREUPLOAD=hash: hash the texture content at every reupload and compare it with the previous one. The dirty tracker only knows a page
// was written, not whether the bytes differ. If the hash is unchanged, the game rewrote the same bytes and the reupload was pointless.
static int texreupload_hash_diag(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_NVN_TEXREUPLOAD"); v = e && strcmp(e, "hash") == 0; } return v; }
static uint64_t full_hash(const uint8_t *m, uint64_t n) { uint64_t h = 1469598103934665603ull; for (uint64_t i = 0; i < n; i++) h = (h ^ m[i]) * 1099511628211ull; return h; }
// The GPU does not support BC compressed formats (A15 / A16 etc.): BC textures are created in the decompressed format and decompressed on the CPU at upload / buffer→texture copy time (nvn_bcdec.c).
// BC3 among the game textures was converted to ASTC offline by the Legacy Device Pack (dm/tools/astc_pack.py); what reaches here is the small remainder of BC. The test switch DM_TEST_NO_BC=1 forces this path on any machine
static int no_bc(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("DM_TEST_NO_BC"); v = (e && *e && strcmp(e, "0")) || ![g_dev supportsBCTextureCompression];
        if (v) MLOG("显卡不支持 BC 压缩格式%s：BC 贴图建成解压格式（BC1–5 在 GPU 上解压，BC6H / BC7 / 3D 在 CPU 上）", e && *e && strcmp(e, "0") ? "（测试开关 DM_TEST_NO_BC）" : ""); }
    return v;
}
static int bc_unpack(uint32_t nvn_fmt, uint32_t *pf, int *bpp) { return no_bc() && nvn_bc_unpacked(nvn_fmt, pf, bpp); }
static MTLTextureDescriptor *tex_desc(const tex_t *t, int packaged) {
    uint32_t pf = 0; uint8_t swz[4];
    if (!nvn_fmt_to_mtl(t->d.format, &pf, swz) || !pf) { RED_ONCE(0, "贴图格式 %#x Metal 无对应（首例）", t->d.format); return nil; }
    int unp = 0; { uint32_t upf; int ubpp; if (bc_unpack(t->d.format, &upf, &ubpp)) { pf = upf; unp = 1; } }
    // Metal's limits: 16384 in 2D, 2048 in 3D, 2048 array layers; beyond them Metal aborts the process on an assertion, so they are caught here with a red line
    if (t->d.width > 16384 || t->d.height > 16384 || t->d.depth > 2048 || t->d.levels > 15) { RED_ONCE(71, "贴图尺寸超出 Metal 上限：%u×%u×%u 级 %u 格式 %#x 目标 %d（首例 ser=%llu）", t->d.width, t->d.height, t->d.depth, t->d.levels, t->d.format, t->d.target, (unsigned long long)t->serial); return nil; }
    MTLTextureDescriptor *d = [MTLTextureDescriptor new];
    d.pixelFormat = (MTLPixelFormat)pf;
    d.width = t->d.width ? t->d.width : 1; d.height = t->d.height ? t->d.height : 1; d.depth = 1; d.arrayLength = 1;
    d.mipmapLevelCount = t->d.levels ? t->d.levels : 1;
    switch (t->d.target) {
        // 1D / 1D array are created as one-row 2D / 2D array textures: translated shaders never declare texture1d (0 in all five games' libraries) and always sample them as 2D; a Metal 1D texture bound to a texture2d argument is a type mismatch with undefined reads.
        // Super Mario 3D World + Bowser's Fury's material shaders sample 64x1 R32F 1D lookup tables heavily every frame (on Maxwell a 2D sample of a 1D texture reads that one row)
        case 0: d.textureType = MTLTextureType2D; d.height = 1; break;
        case 2: d.textureType = MTLTextureType3D; d.depth = t->d.depth; break;
        case 3: d.textureType = MTLTextureType2DArray; d.arrayLength = t->d.height; d.height = 1; break;
        case 4: d.textureType = MTLTextureType2DArray; d.arrayLength = t->d.depth; break;
        case 5: d.textureType = MTLTextureType2DMultisample; d.sampleCount = t->d.samples; d.mipmapLevelCount = 1; break;
        case 6: d.textureType = MTLTextureType2DMultisampleArray; d.sampleCount = t->d.samples; d.arrayLength = t->d.depth; d.mipmapLevelCount = 1; break;
        // Cube textures are created as 6-layer 2D arrays and sampled through cube / cube array views made by tex_fit: Metal only allows cube array views of a 2D array base texture (not of a cube base).
        // The Legend of Zelda: Breath of the Wild has 1512 shaders that sample the environment light probes as a cube array while a single cube texture is bound (texture ID 1064 onwards): with mismatched types the reads are undefined, and every lit surface came out black
        case 8: d.textureType = MTLTextureType2DArray; d.arrayLength = 6; break;
        case 9: d.textureType = MTLTextureTypeCubeArray; d.arrayLength = t->d.depth; break;
        case 10: RED_ONCE(1, "缓冲贴图（TARGET_BUFFER）未实现"); return nil;
        default: d.textureType = MTLTextureType2D; break;
    }
    d.storageMode = MTLStorageModePrivate;   // Render targets and assets both go in GPU private storage (lossless compression available)
    d.usage = MTLTextureUsageShaderRead | (fmt_compressed(t->d.format) || packaged ? 0 : MTLTextureUsageRenderTarget)
            | (t->d.flags & NVN_TEXTURE_FLAGS_IMAGE ? MTLTextureUsageShaderWrite : 0)   // Image usage: shader-writable (compute kernels generating the 3D color lookup table, etc.)
            | (unp ? MTLTextureUsageShaderWrite | MTLTextureUsagePixelFormatView : 0);   // No BC: textures in the decompressed format are written by the GPU decompression kernel (up_dec) / receive whole-block copies from ASTC sources (copy_t2t_astc)
    return d;
}
// DM_TEX_DEMOTE=<n>: drop up to n top mip levels of large compressed 2D asset textures. Switch assets are already
// half-resolution against the PC originals (see dm/gfx/tex/tests/t_tex.c B), the phone screen is small, and on iOS
// the per-process memory limit is what kills Hades II: one 4096×4096 ASTC 4x4 mip chain is 21.8 MB and the scene
// transition uploads a few dozen of them in one frame. Skipping level 0 costs no decode / re-encode work: source
// levels 1..n-1 move down one slot byte for byte.
static int tex_demote_skip(const tex_t *t) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("DM_TEX_DEMOTE"); v = e ? atoi(e) : 0; }
    // NVN texture targets: 1D = 0, 2D = 1, 3D = 2. The gate used to compare with 0 (= 1D), so it never accepted a 2D asset and the switch never fired on a real run.
    if (v <= 0 || t->d.target != NVN_TEXTURE_TARGET_2D || t->d.samples > 1 || !fmt_compressed(t->d.format)) return 0;
    int skip = 0;
    while (skip < v && t->d.levels > skip + 1 && (t->d.width >> (skip + 1)) >= 1024 && (t->d.height >> (skip + 1)) >= 1024) skip++;
    return skip;
}
// DM_TEX_HALF=<1|2>: halve large single-level ASTC assets by dropping every other 4×4 block row / column. This is a
// byte-level operation on the compressed blocks (no decode / re-encode): the Switch data is already half-res vs the
// PC originals and the phone screen is small, while a 4096×4096 ASTC 4x4 texture is 16 MB. 1 = only textures with a
// side >= 4096, 2 = also 2048. Only the packaged (block-linear upload) path below implements halving, and only for samples assets: zero-copy linear textures change
// behaviour under halving, and a non-packaged texture with NVN_TEXTURE_FLAGS_IMAGE may be shader-written. Hades II sets the IMAGE flag on its imported ASTC assets
// too (measured: 4096x4096 ASTC 4x4 single level has flags 0x4 while carrying a packaged layout), so a packaged layout overrides the IMAGE veto.
static int fmt_astc(int f) { return f >= 0x79 && f <= 0x94; }
static int tex_half_skip(const tex_t *t) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("DM_TEX_HALF"); v = e ? atoi(e) : DM_GAME_TEX_HALF; }   // profile tex_half default, env override
    // NVN target 2D = 1 (this gate also compared with 0 = 1D, which is why DM_TEX_HALF never fired on a real run)
    if (v <= 0 || t->d.target != NVN_TEXTURE_TARGET_2D || t->d.samples > 1 || t->d.levels != 1 || !fmt_astc(t->d.format)) return 0;
    if ((t->d.flags & NVN_TEXTURE_FLAGS_IMAGE) && !t->has_layout) return 0;   // not a shipped packaged asset: may be shader-written
    if (t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) return 0;   // zero-copy linear path, not the packed upload path
    uint32_t mn = MIN(t->d.width, t->d.height), mx = MAX(t->d.width, t->d.height);
    if (mx < (v >= 2 ? 2048u : 4096u) || (mn >> 1) < 512) return 0;
    return 1;
}
// On Apple GPUs each standalone private texture can consume a much larger allocation granule than its pixel size
// (Hades II has twenty-odd menu textures whose 740 MB resident total is mostly padding). DM_TEX_HEAP=1 places guest
// textures in reusable 64 MB MTLHeaps, which keeps several render targets / ASTC pages in one allocation.
static pthread_mutex_t g_tex_heap_mu = PTHREAD_MUTEX_INITIALIZER;
static NSMutableArray<id<MTLHeap>> *g_tex_heaps;
static int g_tex_heap_mode = -1;
// Low-memory guard: an empty automatic MTLHeap still holds its whole slab allocation on the GPU side, so drop heaps that no longer contain a live texture.
static void tex_heap_purge_empty(void) {
    if (g_tex_heap_mode <= 0) return;
    NSInteger freed = 0;
    pthread_mutex_lock(&g_tex_heap_mu);
    if (g_tex_heaps) for (NSInteger i = (NSInteger)g_tex_heaps.count - 1; i >= 0; i--) {
        id<MTLHeap> h = g_tex_heaps[(NSUInteger)i];
        if (h && h.usedSize == 0) { [g_tex_heaps removeObjectAtIndex:(NSUInteger)i]; freed++; }
    }
    pthread_mutex_unlock(&g_tex_heap_mu);
    if (freed) MLOG("低内存保护：释放 %ld 个空纹理堆", (long)freed);
}
static id<MTLTexture> tex_new_private(MTLTextureDescriptor *d, const char *what) {
    if (g_tex_heap_mode < 0) {
        const char *e = getenv("DM_TEX_HEAP");
        g_tex_heap_mode = e && atoi(e);
    }
    if (!g_tex_heap_mode || d.storageMode != MTLStorageModePrivate)
        return [g_dev newTextureWithDescriptor:d];

    MTLSizeAndAlign need = [g_dev heapTextureSizeAndAlignWithDescriptor:d];
    if (!need.size) return [g_dev newTextureWithDescriptor:d];
    id<MTLTexture> mt = nil;
    pthread_mutex_lock(&g_tex_heap_mu);
    if (!g_tex_heaps) g_tex_heaps = [NSMutableArray array];
    for (id<MTLHeap> h in g_tex_heaps) {
        if ([h maxAvailableSizeWithAlignment:need.align] < need.size) continue;
        mt = [h newTextureWithDescriptor:d];
        if (mt) break;
    }
    if (!mt) {
        size_t slab = MAX((size_t)64u << 20, need.size + need.align);
        MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
        // The non-offset newTextureWithDescriptor: is only valid for automatic heaps; a placement heap requires
        // explicit offsets. Automatic layout is also the right choice for persistent, rarely written textures.
        hd.storageMode = MTLStorageModePrivate; hd.type = MTLHeapTypeAutomatic; hd.size = slab;
        id<MTLHeap> h = [g_dev newHeapWithDescriptor:hd];
        if (h) {
            mt = [h newTextureWithDescriptor:d];
            if (mt) {
                [g_tex_heaps addObject:h];
                static int n;
                if (n++ < 20) MLOG("纹理堆 #%d：%zu MB 区块，容纳 %s %lux%lu 格式 %lu（需要 %zu MB，对齐 %zu KB）", n, slab >> 20, what,
                                   (unsigned long)d.width, (unsigned long)d.height, (unsigned long)d.pixelFormat,
                                   need.size >> 20, need.align >> 10);
            }
        }
    }
    pthread_mutex_unlock(&g_tex_heap_mu);
    if (!mt) RED_ONCE(80, "纹理堆分配失败，回退独立贴图：%s", what);
    return mt ?: [g_dev newTextureWithDescriptor:d];
}
// In-place upgrade: when the original texture was not created with PixelFormatView (off by default, to keep Apple GPU lossless compression) but a different-format view is needed, create a new one with the original spec + PixelFormatView, copy over every slice × level, and replace the registration.
// Count: across the 4 screens character select / camp / wilderness / combat, 17473 textures needed a different-format view 0 times, so this is a fallback and does not rely on predicting which textures need it
static id<MTLTexture> tex_upgrade_pfv(const tex_t *t, id<MTLTexture> base, MTLPixelFormat view_pf) {
    MTLTextureDescriptor *d = [MTLTextureDescriptor new];
    d.textureType = base.textureType; d.pixelFormat = base.pixelFormat; d.width = base.width; d.height = base.height; d.depth = base.depth;   // Keep the original format, only add the format-view usage
    d.arrayLength = base.arrayLength; d.mipmapLevelCount = base.mipmapLevelCount; d.sampleCount = base.sampleCount;
    d.storageMode = base.storageMode; d.cpuCacheMode = base.cpuCacheMode; d.hazardTrackingMode = base.hazardTrackingMode;
    d.resourceOptions = base.resourceOptions; d.usage = base.usage | MTLTextureUsagePixelFormatView;
    // 线性贴图别名在客体池上：加用途必须保住 buffer 后端——换成独立贴图只把像素拷一次，之后 GPU 的写入到不了 CPU 映射
    id<MTLBuffer> backing = base.buffer;
    id<MTLTexture> upgraded = backing ? [backing newTextureWithDescriptor:d offset:base.bufferOffset bytesPerRow:base.bufferBytesPerRow] : tex_new_private(d, "格式视图");
    if (!upgraded) { RED_ONCE(45, "贴图升级失败 ser=%llu 格式 %lu → %lu", t->serial, (unsigned long)base.pixelFormat, (unsigned long)view_pf); return base; }
    res_add(upgraded);
    id<MTL4ComputeCommandEncoder> b = backing ? nil : blit();   // 共用后端里像素已经在：再拷会自己叠自己
    NSUInteger slices = base.textureType == MTLTextureType3D ? 1 : base.arrayLength * (base.textureType == MTLTextureTypeCube || base.textureType == MTLTextureTypeCubeArray ? 6 : 1);
    for (NSUInteger l = 0; !backing && l < base.mipmapLevelCount; l++) {
        NSUInteger w = MAX(base.width >> l, 1u), h = MAX(base.height >> l, 1u), z = base.textureType == MTLTextureType3D ? MAX(base.depth >> l, 1u) : 1;
        if (base.textureType == MTLTextureType3D)
            enc_b_copy_texture(b, base, 0, l, MTLOriginMake(0, 0, 0), MTLSizeMake(w, h, z), upgraded, 0, l, MTLOriginMake(0, 0, 0));
        else for (NSUInteger s = 0; s < slices; s++)
            enc_b_copy_texture(b, base, s, l, MTLOriginMake(0, 0, 0), MTLSizeMake(w, h, 1), upgraded, s, l, MTLOriginMake(0, 0, 0));
    }
    id holder = serial_get(t->serial);
    if ([holder isKindOfClass:[NvPkg class]]) { id<MTLTexture> old = ((NvPkg *)holder)->tex; ((NvPkg *)holder)->tex = upgraded; defer_obj_release(old); g_ser_gen++; }
    else {
        serial_put(t->serial, upgraded);
        alias_key ak; memset(&ak, 0, sizeof ak); ak.pool = (uint64_t)(uintptr_t)t->pool; ak.off = t->off; ak.d = t->d;   // Switch the sharing table to the upgraded one too, so later objects on the same storage get it directly
        pthread_mutex_lock(&g_smu); if (hm_get_id(&g_alias, &ak) == base) { hm_put_id(&g_alias, &ak, upgraded); CFSetAddValue(g_alias_set, (__bridge const void *)upgraded); } pthread_mutex_unlock(&g_smu);
    }
    MLOG("贴图升级为可换格式视图 ser=%llu 基格式 %lu → 视图格式 %lu（升级次数=%llu）", t->serial, (unsigned long)base.pixelFormat, (unsigned long)view_pf, ++g_pfv_upgrades);
    return upgraded;
}
// Packaged texture upload: first set read-only, then take the stamp, then read the pool memory and deswizzle into a staging buffer, and blit into a new private texture.
// Batching: uploads within one replay submission share one upload command buffer (compute encoder deswizzle + a unified blit), flushed before the replay command buffer is committed;
// the same queue executes in commit order ⇒ uploads are always finished before the first sample. Uploads outside a submission are flushed on the spot. The staging area is cut from 32 MB private blocks, held by the command buffer until completion.
static _Atomic uint64_t g_upload_ns;   // Cumulative time of deswizzle + upload encoding on the submit thread (timing instrumentation)
static _Atomic uint64_t g_upload_gpu_ns;
static id<MTL4CommandBuffer> g_up_cb;
static id<MTL4ComputeCommandEncoder> g_up_ce;
static _Atomic int g_up_alloc_busy[3];
static unsigned g_up_seq;
#define UP_BLOCK_SIZE (32u * 1024u * 1024u)
typedef struct {
    __unsafe_unretained id<MTLBuffer> src;
    size_t off, row, image;
    MTLSize size;
    __unsafe_unretained id<MTLTexture> dst;   // Strongly referenced by g_up_texs until the flush is encoded
    NSUInteger slice, level;
} up_blit;
static up_blit *g_up_blits; static int g_nup_blits, g_up_blitcap;
// Without BC, BC1–5 are decompressed on the GPU (A16 logs: CPU deswizzle + CPU decompression took over 300 ms in a single frame): the GPU deswizzle writes the compressed blocks into the staging area, and at flush time, after the barrier,
// the compute kernel bcd decodes them into the texture one thread per block (the target is a 2D view in the decompressed format; an sRGB base texture is written with raw values through an RGBA8Unorm view). BC6H / BC7 / 3D textures are very few and still go through the CPU (nvn_bcdec.c)
typedef struct { __unsafe_unretained id<MTLBuffer> src, params; size_t off, poff; uint32_t bw, bh; __unsafe_unretained id<MTLTexture> view; } up_dec;   // view: held by g_up_texs within the batch, and afterwards by defer_obj_unref until the GPU finishes
static up_dec *g_up_decs; static int g_nup_decs, g_up_deccap;
static const char k_bcd_msl[] =
"#include <metal_stdlib>\nusing namespace metal;\n"
"struct P { uint off, pitch, w, h, fmt; };\n"   // fmt: 1 BC1, 2 BC2, 3 BC3, 4 BC4, 5 BC4 signed, 6 BC5, 7 BC5 signed
"static float3 c565(uint c) { return float3((c >> 11) & 31, (c >> 5) & 63, c & 31) / float3(31, 63, 31); }\n"
"static void col(device const uchar *b, thread float4 *o, bool four) {\n"
"  uint c0 = b[0] | (b[1] << 8), c1 = b[2] | (b[3] << 8), ix = b[4] | (b[5] << 8) | (b[6] << 16) | (b[7] << 24); float3 e0 = c565(c0), e1 = c565(c1); float4 p[4];\n"
"  p[0] = float4(e0, 1); p[1] = float4(e1, 1);\n"
"  if (four || c0 > c1) { p[2] = float4((2 * e0 + e1) / 3, 1); p[3] = float4((e0 + 2 * e1) / 3, 1); } else { p[2] = float4((e0 + e1) / 2, 1); p[3] = float4(0); }\n"
"  for (int i = 0; i < 16; i++) o[i] = p[(ix >> (2 * i)) & 3]; }\n"
"static void a8(device const uchar *b, thread float *o, bool sn) {\n"   // Single-channel block of BC4 / BC3 alpha / BC5
"  float a0 = sn ? max(float(as_type<char>(b[0])), -127.0) / 127 : b[0] / 255.0, a1 = sn ? max(float(as_type<char>(b[1])), -127.0) / 127 : b[1] / 255.0;\n"
"  bool eight = sn ? as_type<char>(b[0]) > as_type<char>(b[1]) : b[0] > b[1]; float lo = sn ? -1 : 0; float p[8]; p[0] = a0; p[1] = a1;\n"
"  for (int k = 2; k < 8; k++) p[k] = eight ? ((8 - k) * a0 + (k - 1) * a1) / 7 : (k < 6 ? ((6 - k) * a0 + (k - 1) * a1) / 5 : (k == 6 ? lo : 1));\n"
"  uint lo24 = b[2] | (b[3] << 8) | (b[4] << 16), hi24 = b[5] | (b[6] << 8) | (b[7] << 16);\n"
"  for (int i = 0; i < 16; i++) o[i] = p[i < 8 ? (lo24 >> (3 * i)) & 7 : (hi24 >> (3 * (i - 8))) & 7]; }\n"
"kernel void bcd(device const uchar *src [[buffer(0)]], device const P &p [[buffer(1)]], texture2d<float, access::write> dst [[texture(0)]], uint2 g [[thread_position_in_grid]]) {\n"
"  uint nbx = (p.w + 3) / 4, nby = (p.h + 3) / 4; if (g.x >= nbx || g.y >= nby) return;\n"
"  uint bb = (p.fmt == 1 || p.fmt == 4 || p.fmt == 5) ? 8 : 16; device const uchar *b = src + p.off + g.y * p.pitch + g.x * bb; float4 o[16]; float a[16], c[16];\n"
"  if (p.fmt == 1) col(b, o, false);\n"
"  else if (p.fmt == 2) { col(b + 8, o, true); for (int i = 0; i < 16; i++) o[i].a = ((b[i / 2] >> (4 * (i & 1))) & 15) / 15.0; }\n"
"  else if (p.fmt == 3) { col(b + 8, o, true); a8(b, a, false); for (int i = 0; i < 16; i++) o[i].a = a[i]; }\n"
"  else if (p.fmt == 4 || p.fmt == 5) { a8(b, a, p.fmt == 5); for (int i = 0; i < 16; i++) o[i] = float4(a[i], 0, 0, 1); }\n"
"  else { a8(b, a, p.fmt == 7); a8(b + 8, c, p.fmt == 7); for (int i = 0; i < 16; i++) o[i] = float4(a[i], c[i], 0, 1); }\n"
"  for (int i = 0; i < 16; i++) { uint2 q = g * 4 + uint2(i & 3, i >> 2); if (q.x < p.w && q.y < p.h) dst.write(o[i], q); } }\n";
static int bcd_fmt(uint32_t f) {   // NVN format → bcd fmt; 0 = not decoded on the GPU (BC6H / BC7 / non-BC)
    switch (f) { case 0x41: case 0x42: case 0x45: case 0x46: return 1; case 0x43: case 0x47: return 2; case 0x44: case 0x48: return 3;
                 case 0x49: return 4; case 0x4a: return 5; case 0x4b: return 6; case 0x4c: return 7; default: return 0; }
}
static id<MTLComputePipelineState> bcd_pso(void) {
    static id<MTLComputePipelineState> ps; static int failed;
    if (!ps && !failed) { NSError *e = nil; id<MTLLibrary> lib = [g_dev newLibraryWithSource:@(k_bcd_msl) options:nil error:&e]; id<MTLFunction> fn = lib ? [lib newFunctionWithName:@"bcd"] : nil;
        ps = fn ? [g_dev newComputePipelineStateWithFunction:fn error:&e] : nil;
        if (!ps) { failed = 1; MLOG("【红】无 BC：GPU 解压管线建不出 %s（退回 CPU 解压）", e.localizedDescription.UTF8String); } else MLOG("无 BC：BC1–5 改在 GPU 上解压"); }
    return ps;
}
static NSMutableArray *g_up_texs;   // Target textures in the batch: before the flush, textures may be replaced / released within the same submission
static id<MTLBuffer> g_up_block;
static size_t g_up_block_off, g_up_block_size;
static NSMutableArray *g_up_keep;
static size_t g_up_stage_bytes;
#define UP_BATCH_LIMIT (64u * 1024u * 1024u)
// Low-memory guard (DM_LOWMEM_MB=<mb> threshold, DM_LOWMEM_HYST_MB=<mb> release hysteresis, DM_LOWMEM_HARD_MB=<mb> hard threshold; DM_LOWMEM_MB=0 turns it off):
// when os_proc_available_memory falls below the threshold, the upload batch is cut to 16 MiB, every upload flush waits for the GPU, and the deferred-release queue is
// drained every frame, so staging buffers and newly created private textures cannot pile up inside one frame while the system is on its way to killing the process.
// Hysteresis keeps it from flapping: the guard leaves once available memory is above threshold + hysteresis. Correctness is unchanged: every still-rendered texture is
// still uploaded, only in smaller batches. iOS only; on the Mac the guard is a no-op unless DM_LOWMEM_SIM_AVAIL_MB=<mb> simulates a low-memory device for testing.
#define UP_BATCH_LIMIT_LOWMEM (16u * 1024u * 1024u)
static _Atomic int g_lowmem_active;   // guard state: read as one atomic load on the upload paths
static int g_lowmem_mb = -1, g_lowmem_hyst_mb = -1, g_lowmem_hard_mb = -1;   // thresholds, read from the environment on the first poll
static int g_lowmem_enters, g_lowmem_exits, g_lowmem_hard_warned;
static uint64_t g_lowmem_frame, g_lowmem_adds0, g_lowmem_addb0, g_lowmem_dels0, g_lowmem_delb0;   // guard start frame and pool counters at that moment
static _Atomic unsigned long long g_lowmem_avail_mb;   // last available-memory reading in MB (0 = unavailable / guard off); for the memory readout
// Available memory left for this process, in MB; 0 = no source (Mac without simulation)
static unsigned long long lowmem_avail_mb(void) {
#if TARGET_OS_IPHONE
    return (unsigned long long)(os_proc_available_memory() >> 20);
#else
    static int sim = -1;
    if (sim == -1) { const char *e = getenv("DM_LOWMEM_SIM_AVAIL_MB"); sim = e ? atoi(e) : 0; }
    return sim > 0 ? (unsigned long long)sim : 0;
#endif
}
// Called once per frame from mtl_mem_stat. Cheap: one available-memory read, and while the guard is active one deferred-queue drain (that queue is usually a few entries).
static void lowmem_poll(unsigned long long frame) {
    if (g_lowmem_mb < 0) {   // thresholds, first call only
        const char *e = getenv("DM_LOWMEM_MB"); g_lowmem_mb = e ? atoi(e) : 384; if (g_lowmem_mb < 0) g_lowmem_mb = 0;
        e = getenv("DM_LOWMEM_HYST_MB"); g_lowmem_hyst_mb = e ? atoi(e) : 128; if (g_lowmem_hyst_mb < 0) g_lowmem_hyst_mb = 0;
        e = getenv("DM_LOWMEM_HARD_MB"); g_lowmem_hard_mb = e && atoi(e) > 0 ? atoi(e) : 192;
        if (g_lowmem_mb > 0) MLOG("低内存保护：阈值 %d MB（解除 %d MB，硬阈值 %d MB）", g_lowmem_mb, g_lowmem_mb + g_lowmem_hyst_mb, g_lowmem_hard_mb);
    }
    if (g_lowmem_mb <= 0) return;
    unsigned long long av = lowmem_avail_mb();
    if (!av) return;   // no source: macOS without DM_LOWMEM_SIM_AVAIL_MB
    atomic_store(&g_lowmem_avail_mb, av);
    int was = atomic_load(&g_lowmem_active), now = was;
    if (!was && av < (unsigned long long)g_lowmem_mb) {
        now = 1; g_lowmem_enters++; g_lowmem_frame = frame;
        pthread_mutex_lock(&g_mu); g_lowmem_adds0 = g_pool_adds; g_lowmem_addb0 = g_pool_add_bytes; g_lowmem_dels0 = g_pool_dels; g_lowmem_delb0 = g_pool_del_bytes; pthread_mutex_unlock(&g_mu);
        MLOG("低内存保护：可用 %llu MB < %d MB，上传批次降到 16 MB、每次上传等待完成、每帧清延迟释放队列（帧 %llu）", av, g_lowmem_mb, frame);
        tex_heap_purge_empty();   // heaps with no live texture (DM_TEX_HEAP) hold their whole slab: drop them while it still counts
        mtl_tex_reclaim_hint();   // let the static-source reclaim chance a pass at the next submit-thread safe point (source pools are the biggest dead guest bytes)
    } else if (was && av >= (unsigned long long)(g_lowmem_mb + g_lowmem_hyst_mb)) {
        now = 0; g_lowmem_exits++;
        MLOG("低内存保护解除：可用 %llu MB >= %d MB（帧 %llu）", av, g_lowmem_mb + g_lowmem_hyst_mb, frame);
    }
    if (now != was) atomic_store(&g_lowmem_active, now);
    if (!now) return;
    @autoreleasepool { defer_drain(); }   // retire finished objects now instead of at the next submission; while active this keeps the deferred queue from holding a frame's worth of buffers (defer_drain autoreleases on this pool-less guest thread)
    if (!g_lowmem_hard_warned && av < (unsigned long long)g_lowmem_hard_mb) {
        g_lowmem_hard_warned = 1;
        MLOG("【红】低内存保护：可用 %llu MB < 硬阈值 %d MB（帧 %llu）：保持 16 MB 上传批次；本机没有投机性预拷贝路径可停，未做额外内容裁剪", av, g_lowmem_hard_mb, frame);
    }
    static unsigned long long last_warn;
    if (frame >= last_warn + 300) {   // rate-limited reminder while the guard stays active, with the pool delta since the guard came on
        last_warn = frame;
        uint64_t adds, addb, dels, delb;
        pthread_mutex_lock(&g_mu); adds = g_pool_adds; addb = g_pool_add_bytes; dels = g_pool_dels; delb = g_pool_del_bytes; pthread_mutex_unlock(&g_mu);
        MLOG("低内存保护持续：可用 %llu MB（帧 %llu，自帧 %llu 起）；自保护开始新建池 %llu 个 %llu MB、销毁 %llu 个 %llu MB", av, frame, (unsigned long long)g_lowmem_frame,
             (unsigned long long)(adds - g_lowmem_adds0), (unsigned long long)((addb - g_lowmem_addb0) >> 20), (unsigned long long)(dels - g_lowmem_dels0), (unsigned long long)((delb - g_lowmem_delb0) >> 20));
    }
}

// Upload staging buffers are referenced only by the upload command buffer. Once its feedback arrives they are
// finished, so retire them immediately instead of waiting for the main replay fence (g_fence + 1).
static void up_retire_staging(NSArray *objects) {
    if (!objects.count) return;
    for (id obj in objects) if ([obj conformsToProtocol:@protocol(MTLResource)]) res_remove((id<MTLResource>)obj);
    res_commit();
}
static void up_wait(void) {
    while (atomic_load(&g_up_pending)) usleep(200);
    if (g_tex_reclaim || g_tex_reclaim_dry) { atomic_store_explicit(&g_rc_tick_wanted, 1, memory_order_relaxed); rc_reclaim_tick(); }
}


// The upload command buffer was discarded by the system (dropped along during GPU error recovery; the first frame on entering Travincal drops one every run): the objects of this batch of textures are built and the timestamps updated, but the content was never written in,
// and if the pool memory on the game side no longer changes it will never be re-uploaded. Record this batch of target textures, and the next time any of them is fetched treat it as dirty and re-upload once.
// DM_TEST_UP_LOST=<batch index> (for testing): treat that batch of uploads as dropped, to verify the re-upload path.
static NSHashTable *g_up_lost; static os_unfair_lock g_up_lost_lk = OS_UNFAIR_LOCK_INIT; static _Atomic int g_up_lost_n; static _Atomic uint64_t g_up_lost_redo;   // Weak-reference table: if a texture in the record is never fetched again and is released normally, this table does not keep it alive
static int up_lost_take(id tex) {   // This texture's upload was dropped: take it out of the record and return 1
    if (!tex) return 0; os_unfair_lock_lock(&g_up_lost_lk); int hit = [g_up_lost containsObject:tex]; if (hit) { [g_up_lost removeObject:tex]; atomic_store(&g_up_lost_n, (int)g_up_lost.allObjects.count); atomic_fetch_add(&g_up_lost_redo, 1); } os_unfair_lock_unlock(&g_up_lost_lk); return hit;
}
static void up_diag(void);
static void up_flush(void) {
    up_owner_check("up_flush");
    if (!g_up_cb) return;
    if (g_up_ce) {
        m4_end(g_up_ce); g_up_ce = nil;
    }
    id<MTL4ComputeCommandEncoder> b = g_nup_blits || g_nup_decs ? g_up_ce = [g_up_cb computeCommandEncoder] : nil;
    if (b) [b barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice];
    for (int i = 0; b && i < g_nup_blits; i++) {
        up_blit *x = &g_up_blits[i];
        enc_b_copy_buffer_texture(b, x->src, x->off, x->row, x->image, x->size, x->dst, x->slice, x->level, MTLOriginMake(0, 0, 0));
    }
    if (b && g_nup_decs) {   // GPU decompression without BC (see up_dec): offsets in the parameter block are relative to the start of the staging block
        [b setArgumentTable:g_up_ct]; [b setComputePipelineState:bcd_pso()];
        for (int i = 0; i < g_nup_decs; i++) { up_dec *x = &g_up_decs[i];
            [g_up_ct setAddress:x->src.gpuAddress atIndex:0]; [g_up_ct setAddress:x->params.gpuAddress + x->poff atIndex:1]; [g_up_ct setTexture:x->view.gpuResourceID atIndex:0];
            [b dispatchThreadgroups:MTLSizeMake((x->bw + 7) / 8, (x->bh + 7) / 8, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)]; }
    }
    if (b) { m4_end(b); g_up_ce = nil; }
    id<MTL4CommandBuffer> cb = g_up_cb;
    unsigned slot = (g_up_seq - 1) % 3;
    NSMutableArray *dsts = [NSMutableArray arrayWithCapacity:(NSUInteger)g_nup_blits]; for (int i = 0; i < g_nup_blits; i++) if (g_up_blits[i].dst) [dsts addObject:g_up_blits[i].dst];
    static int lost_at = -2; static uint64_t batch_n; if (lost_at == -2) { const char *e = getenv("DM_TEST_UP_LOST"); lost_at = e ? atoi(e) : -1; } int fake_lost = dsts.count && lost_at > 0 && (int)++batch_n == lost_at;
    MTL4CommitOptions *o = [MTL4CommitOptions new];
    NSMutableArray *staging = [NSMutableArray array];
    if (g_up_block) [staging addObject:g_up_block];
    if (g_up_keep.count) [staging addObjectsFromArray:g_up_keep];
    // T5: the candidate source pools of this upload batch are recorded when the upload is encoded; pass them to the feedback handler and let the
    // submit thread evaluate them once the GPU finished the upload (the only GPU reader of those pools).
    rc_src *rcb = NULL; int nrcb = g_up_rc_n;
    if (nrcb > 0) { rcb = malloc((size_t)nrcb * sizeof *rcb); if (rcb) memcpy(rcb, g_up_rc, (size_t)nrcb * sizeof *rcb); else nrcb = 0; g_up_rc_n = 0; }
    atomic_fetch_add(&g_up_pending, 1);
    [o addFeedbackHandler:^(id<MTL4CommitFeedback> f) {
        if (f.error) MLOG("【红】上传命令缓冲执行出错：%s", f.error.description.UTF8String);
        if (f.error || fake_lost) {
            os_unfair_lock_lock(&g_up_lost_lk); if (!g_up_lost) g_up_lost = [NSHashTable weakObjectsHashTable]; for (id d in dsts) [g_up_lost addObject:d]; int n = (int)g_up_lost.allObjects.count; atomic_store(&g_up_lost_n, n); os_unfair_lock_unlock(&g_up_lost_lk);
            pthread_mutex_lock(&g_smu); g_ser_gen++; pthread_mutex_unlock(&g_smu);   // Make the view cache of every texture ID take the slow path once (on a cache hit tex_mtl is not passed through and this record is not seen)
            MLOG("%s上传命令缓冲被丢：这一批 %lu 张贴图记为内容未写，下次取用时重传（待重传共 %d 张）", fake_lost ? "DM_TEST_UP_LOST：" : "", (unsigned long)dsts.count, n);
        }
        double d = f.GPUEndTime - f.GPUStartTime;
        if (d > 0) atomic_fetch_add(&g_upload_gpu_ns, (uint64_t)(d * 1e9));
        up_retire_staging(staging);
        if (rcb) { if (!f.error && !fake_lost) rc_pending_add(rcb, nrcb); free(rcb); }
        atomic_store(&g_up_alloc_busy[slot], 0);
        atomic_fetch_sub(&g_up_pending, 1);
    }];
    res_commit();
    [cb endCommandBuffer];
    id<MTL4CommandBuffer> buffers[] = {cb};
    [g_q4 commit:buffers count:1 options:o];
    g_up_cb = nil; g_up_block = nil; g_up_block_off = g_up_block_size = 0;
    [g_up_keep removeAllObjects];
    up_diag();
    g_nup_blits = 0; g_nup_decs = 0;
    [g_up_texs removeAllObjects];
    g_up_stage_bytes = 0;
    if (atomic_load(&g_lowmem_active)) up_wait();   // low-memory guard: do not overlap this batch's staging + new private textures with the next batch in time
}

static void up_begin(void) {
    up_owner_check("up_begin");
    if (g_up_cb && g_up_stage_bytes >= (atomic_load(&g_lowmem_active) ? (size_t)UP_BATCH_LIMIT_LOWMEM : (size_t)UP_BATCH_LIMIT)) { up_flush(); up_wait(); }   // low-memory guard: smaller batch so staging + new private textures cannot pile up
    if (g_up_cb) return;
    unsigned slot = g_up_seq++ % 3;
    while (atomic_load(&g_up_alloc_busy[slot])) usleep(1000);
    atomic_store(&g_up_alloc_busy[slot], 1);
    [g_up_alloc4[slot] reset];
    g_up_cb = [g_dev newCommandBuffer];
    [g_up_cb beginCommandBufferWithAllocator:g_up_alloc4[slot]];
    g_up_ce = [g_up_cb computeCommandEncoder];
    [g_up_ce barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice];
    [g_up_ce setArgumentTable:g_up_ct];
    if (!g_up_texs) g_up_texs = [NSMutableArray array];
    if (!g_up_keep) g_up_keep = [NSMutableArray array];
}

// Diagnostic readout (low frequency; printed only when the block switch count changed and ≥ 1 second since the last line, at most 200 lines per game): how many times an upload batch ran out of staging block and switched blocks midway; GPU allocated, residency set object count, deferred-release queue length.
// If the old block is not retired on a switch it stays in the residency set and can never be freed, and "GPU allocated" climbs one block at a time with the switch count
static uint64_t g_up_blocks_new, g_up_swaps, g_up_swap_bytes; static NSHashTable *g_up_old;   // g_up_old: old blocks switched out (weak references); the diagnostic line counts how many are still alive
static void up_diag(void) {
    // Print a line when the block switch count changes; then one more line 5 seconds later (that batch's commit is long done and deferred retirement drained, so "GPU allocated" is the settled number)
    static uint64_t said_swaps = ~0ull, last_ns; static int lines, follow; uint64_t now = clock_gettime_nsec_np(CLOCK_MONOTONIC);
    int chg = g_up_swaps != said_swaps, fol = follow && now - last_ns >= 5000000000ull; if ((!chg && !fol) || lines >= 200) return;
    if (chg && last_ns && now - last_ns < 1000000000ull) return; last_ns = now; said_swaps = g_up_swaps; lines++; follow = chg;
    unsigned long alive = 0, inres = 0; uint64_t abytes = 0; NSArray *olds = g_up_old.allObjects;
    pthread_mutex_lock(&g_res_mu); unsigned long nres = (unsigned long)g_res_objects.allObjects.count; for (id<MTLBuffer> o in olds) { alive++; abytes += o.length; inres += [g_res_objects containsObject:o]; } pthread_mutex_unlock(&g_res_mu);
    pthread_mutex_lock(&g_deferred_mu); unsigned long ndef = (unsigned long)g_deferred.count; pthread_mutex_unlock(&g_deferred_mu);
    MLOG("上传暂存块%s：累计新建 %llu 块、其中一批内换块 %llu 次（换下的旧块共 %llu MB）；换下的旧块此刻还活着 %lu 块 %llu MB、仍在驻留集 %lu 块；显卡已分配 %llu MB、驻留集 %lu 个对象、延迟释放队列 %lu", fol && !chg ? "（落定后）" : "",
         (unsigned long long)g_up_blocks_new, (unsigned long long)g_up_swaps, (unsigned long long)(g_up_swap_bytes >> 20), alive, (unsigned long long)(abytes >> 20), inres, (unsigned long long)(g_dev.currentAllocatedSize >> 20), nres, ndef);
}
static id<MTLBuffer> up_alloc(size_t n, size_t *off) {
    size_t need = (n + 255u) & ~((size_t)255u);
    if (!g_up_block || g_up_block_off + need > g_up_block_size) {
        // Block switch: the old block is still referenced by copy / decompress commands already recorded in this batch and cannot be freed on the spot; hand it to g_up_keep, to be retired later with the other staging after the flush commits (removed from the residency set on retire).
        // Previously g_up_block was simply replaced with the new block here: nobody retired the old block, yet it stayed in the residency set, so each extra block switch in an upload batch leaked one block (usually 32 MB) of GPU memory.
        // The old block and the "current block" retired at flush are two objects, so nothing is retired twice
        g_up_blocks_new++; if (g_up_block) { g_up_swaps++; g_up_swap_bytes += g_up_block_size; if (!g_up_old) g_up_old = [NSHashTable weakObjectsHashTable]; [g_up_old addObject:g_up_block]; static int leak = -1; if (leak < 0) leak = getenv("DM_TEST_UP_LEAK") != NULL;   // test switch DM_TEST_UP_LEAK=1: do not retire the old block, as before the fix (for comparison, to measure the leak)
            if (!leak) { if (!g_up_keep) g_up_keep = [NSMutableArray array]; [g_up_keep addObject:g_up_block]; } }
        g_up_block_size = n > UP_BLOCK_SIZE ? need : UP_BLOCK_SIZE;
        g_up_block = [g_dev newBufferWithLength:g_up_block_size options:MTLResourceStorageModePrivate];
        if (!g_up_block) MDIE("建预排布上传暂存块失败 %#zx", g_up_block_size);
        res_add(g_up_block);
        g_up_block_off = 0;
    }
    *off = g_up_block_off;
    g_up_block_off += need;
    g_up_stage_bytes += need;
    return g_up_block;
}

// Deswizzle parameter blocks: 64 bytes per cell, cut sequentially from a 64 KB shared-storage block; a full block is replaced by a new one, and the old block is released lazily by fence
static id<MTLBuffer> g_up_pblock; static size_t g_up_pblock_off;
static id<MTLBuffer> up_param_alloc(size_t *off) {
    if (!g_up_pblock || g_up_pblock_off + 64 > g_up_pblock.length) {
        if (g_up_pblock) [g_up_keep addObject:g_up_pblock];
        g_up_pblock = [g_dev newBufferWithLength:64 << 10 options:MTLResourceStorageModeShared];
        if (!g_up_pblock) MDIE("建解排布参数块失败");
        res_add(g_up_pblock); g_up_pblock_off = 0;
    }
    *off = g_up_pblock_off; g_up_pblock_off += 64;
    return g_up_pblock;
}
static void up_blit_add(id<MTLBuffer> src, size_t off, size_t row, size_t image, MTLSize size,
                        id<MTLTexture> dst, NSUInteger slice, NSUInteger level) {
    if (g_nup_blits == g_up_blitcap) {
        g_up_blitcap = g_up_blitcap ? g_up_blitcap * 2 : 256;
        g_up_blits = realloc(g_up_blits, g_up_blitcap * sizeof *g_up_blits);
    }
    g_up_blits[g_nup_blits++] = (up_blit){src, off, row, image, size, dst, slice, level};
}

static id<MTLTexture> pkg_upload_(const tex_t *t, uint64_t *stamp, id<MTLTexture> reuse);
static id<MTLTexture> pkg_upload(const tex_t *t, uint64_t *stamp, id<MTLTexture> reuse) {
    struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
    id<MTLTexture> r = pkg_upload_(t, stamp, reuse);   // (a pool around this call was measured and did NOT free the survivors: the autorelease is made elsewhere in the submission)
    clock_gettime(CLOCK_MONOTONIC, &b); uint64_t d_ = (uint64_t)((b.tv_sec - a.tv_sec) * 1000000000ll + b.tv_nsec - a.tv_nsec);
    atomic_fetch_add(&g_upload_ns, d_); g_up_total_ns += d_;
    return r;
}
uint64_t mtl_upload_ns_take(void) { return atomic_exchange(&g_upload_ns, 0); }
uint64_t mtl_upload_gpu_ns_take(void) { return atomic_exchange(&g_upload_gpu_ns, 0); }
void mtl_create_stats(uint64_t *lib, uint64_t *pso) { *lib = g_bp_lib_n; *pso = g_bp_pso_n; }
void mtl_create_ns(uint64_t *lib_ns, uint64_t *pso_ns) { *lib_ns = g_bp_lib_ns; *pso_ns = g_bp_pso_ns; }   // Cumulative time spent synchronously creating shader libraries / pipelines on the spot (the long-frame log takes the per-frame difference)
uint64_t mtl_upload_bytes(void) { return g_bp_up_bytes; }   // Cumulative texture upload bytes
// The hash of every page of the texture's source, taken right after the upload. The dirty check compares the pages that were written
// against these, so a write that puts the same bytes back does not trigger a re-upload.
static void pk_record_pages(NvPkg *pk, const tex_t *t) {
    if (!pk || !t->mem || !t->L.size) return;
    // The page grid is the pool's, so the range may start and end inside a page: the count is the pages the range touches.
    int np = (int)((t->L.size + 65535) / 16384) + 2;   // the host page is 16 KB; allow for the start and the end being inside a page
    if (np > NVN_DIRTY_MAX_PAGES) return;   // a huge texture: fall back to always re-uploading
    if (pk->nphash != np) { free(pk->phash); pk->phash = calloc((size_t)np, sizeof *pk->phash); pk->nphash = pk->phash ? np : 0; }
    if (!pk->phash) { pk->nphash = 0; return; }
    int got = nvn_dirty_all_hashes(t->mem, t->L.size, pk->phash, np);
    if (got < 0 || got > np) { pk->nphash = 0; return; }
    pk->nphash = got;   // the number of pages the reference covers: the dirty check indexes into it
}
static id<MTLTexture> pkg_upload_(const tex_t *t, uint64_t *stamp, id<MTLTexture> reuse) {
    MTLTextureDescriptor *d = tex_desc(t, 1);
    if (!d) return nil;
    int demote = tex_demote_skip(t);
    if (demote) {
        d.width = MAX(t->d.width >> demote, 1u); d.height = MAX(t->d.height >> demote, 1u);
        d.mipmapLevelCount = MAX((NSUInteger)t->d.levels - (NSUInteger)demote, 1);
        static int ndemote; if (ndemote++ < 48) MLOG("贴图降级 #%d ser=%llu %ux%u %u 级 → %lux%lu %u 级（跳过顶层 mip，约省 %d%%）", ndemote, (unsigned long long)t->serial,
            (unsigned)t->d.width, (unsigned)t->d.height, (unsigned)t->d.levels, (unsigned long)d.width, (unsigned long)d.height, (unsigned)d.mipmapLevelCount, demote == 1 ? 75 : 94);
    }
    int half = tex_half_skip(t);
    if (half) {   // ASTC block grid shrinks by half in each direction; the upload loop below subsamples the blocks
        int bw0, bh0; nvn_fmt_bpb(t->d.format, &bw0, &bh0);
        d.width = MAX((((t->d.width + (uint32_t)bw0 - 1) / (uint32_t)bw0) >> 1) * (uint32_t)bw0, (uint32_t)bw0);
        d.height = MAX((((t->d.height + (uint32_t)bh0 - 1) / (uint32_t)bh0) >> 1) * (uint32_t)bh0, (uint32_t)bh0);
        static int nhalf; if (nhalf++ < 48) MLOG("贴图半分辨率 #%d ser=%llu %ux%u → %lux%lu（ASTC 块隔行隔列，约省 75%%）", nhalf, (unsigned long long)t->serial,
            (unsigned)t->d.width, (unsigned)t->d.height, (unsigned long)d.width, (unsigned long)d.height);
    }
    // Streaming pools are rewritten in place and the same serial is re-uploaded hundreds of times per level. When the
    // shape is unchanged, overwrite the existing storage: the upload queue is committed in order with the replay
    // queue, so this is the same hazard ordering the guest already guarantees. It also avoids holding two full
    // generations of the texture while the old one sits in the deferred-release queue.
    id<MTLTexture> mt = nil;
    if (reuse && reuse.pixelFormat == d.pixelFormat && reuse.textureType == d.textureType && reuse.sampleCount == d.sampleCount
        && reuse.width == d.width && reuse.height == d.height && reuse.depth == d.depth && reuse.arrayLength == d.arrayLength
        && reuse.mipmapLevelCount == d.mipmapLevelCount) { mt = reuse; g_pkg_inplace++; }
    else mt = tex_new_private(d, "预排布贴图");
    if (!mt) { RED_ONCE(2, "建 MTLTexture 失败 %ux%u 格式 %#x 目标 %d", t->d.width, t->d.height, t->d.format, t->d.target); return nil; }
    if (g_allocprof > 0) { static int n; if (n++ < 30 || n % 500 == 0) MLOG("新建贴图（%s）#%d ser=%llu %ux%ux%u 格式 %#x 目标 %d 级 %u 存储 %p", __func__, n, (unsigned long long)t->serial, t->d.width, t->d.height, t->d.depth, t->d.format, t->d.target, t->d.levels, (void *)t->mem); }
    if (mt != reuse) res_add(mt);   // Both first upload and re-upload register residency here (Metal 4 does not make resources resident automatically; a missed registration = GPU address page fault)
    nvn_dirty_protect(t->mem, t->L.size);
    *stamp = nvn_dirty_now();
    int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh);
    unsigned layers = (t->d.target == 4 || t->d.target == 9) ? t->d.depth : t->d.target == 8 ? 6 : t->d.target == 3 ? t->d.height : 1;
    if (t->d.target == 9) layers *= 6;
    uint32_t upf; int ubpp = 0, unp = bc_unpack(t->d.format, &upf, &ubpp);   // No BC: decompress on the CPU after deswizzling, with the staging area sized for the decompressed data
    size_t total = 0;
    for (unsigned l = 0; l < d.mipmapLevelCount; l++) {
        uint32_t w = MAX(t->d.width >> (l + demote), 1u), h = MAX(t->d.height >> (l + demote), 1u), z = t->d.target == 2 ? MAX(t->d.depth >> (l + demote), 1u) : 1;
        if (half) total += (size_t)MAX(((w + bw - 1) / bw) >> 1, 1u) * bpb * MAX(((h + bh - 1) / bh) >> 1, 1u) * z * layers;
        else total += unp ? (size_t)w * ubpp * h * z * layers : (size_t)((w + bw - 1) / bw) * bpb * ((h + bh - 1) / bh) * z * layers;
    }
    g_bp_up_bytes += total;
    void *mb; uint64_t mo;
    int gdec = unp && t->d.target != 2 && bcd_fmt(t->d.format) && bcd_pso();   // BC1–5 without BC: GPU deswizzle + GPU decompression
    size_t ctotal = 0;   // Staging size of the compressed blocks (with GPU decompression the staging area holds compressed blocks)
    if (gdec) for (unsigned l = 0; l < d.mipmapLevelCount; l++) { uint32_t w = MAX(t->d.width >> (l + demote), 1u), h = MAX(t->d.height >> (l + demote), 1u); ctotal += (size_t)((w + bw - 1) / bw) * bpb * ((h + bh - 1) / bh) * layers; }
    atomic_store_explicit(&g_rc_addr_tx, 1, memory_order_relaxed);   // T5: the following address use is the texture-source upload itself
    int use_gpu = !half && (!unp || gdec) && g_nvn_deswz_gpu && !rc_src_reclaimed((uintptr_t)t->mem, t->L.size) && mtl_addr((uint64_t)t->mem, t->L.size, &mb, &mo);
    atomic_store_explicit(&g_rc_addr_tx, 0, memory_order_relaxed);   // T5: a reclaimed source is served from the CPU path (never re-wired)
    if (g_nvn_deswz_gpu && !half && !use_gpu && !unp) RED_ONCE(40, "预排布贴图存储不在内存池里，回退 CPU");
    if (use_gpu) {
        // The GPU reads the memory pool at execution time, which is closer to a real Switch than a snapshot at CPU upload time; the dirty-page stamp is still taken before encoding, and later writes still trigger a re-upload.
        up_begin();
        size_t stg_off;
        id<MTLBuffer> stg = up_alloc(gdec ? ctotal : total ? total : 16, &stg_off);
        size_t off = 0; int enc_ok = g_up_ce != nil, nb0 = g_nup_blits, nd0 = g_nup_decs;
        for (unsigned l = 0; enc_ok && l < d.mipmapLevelCount; l++) {
            uint32_t w = MAX(t->d.width >> (l + demote), 1u), h = MAX(t->d.height >> (l + demote), 1u), z = t->d.target == 2 ? MAX(t->d.depth >> (l + demote), 1u) : 1;
            size_t pitch = (size_t)((w + bw - 1) / bw) * bpb, rows = (h + bh - 1) / bh;
            for (unsigned y = 0; y < layers; y++) {
                size_t poff; id<MTLBuffer> params = up_param_alloc(&poff);   // Metal 4 has no setBytes: the parameter block of each deswizzle is cut from a shared block (no new GPU buffer each time)
                if (nvn_deswz_gpu_encode_m4(g_up_ce, g_up_ct, params, poff, (__bridge id<MTLBuffer>)mb, mo, &t->d, l + demote, y, stg, stg_off + off, pitch)) { enc_ok = 0; break; }
                if (gdec) {   // Decompression job: dispatched at flush time after the barrier (see up_flush)
                    MTLPixelFormat vf = mt.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB ? MTLPixelFormatRGBA8Unorm : mt.pixelFormat;
                    id<MTLTexture> v = [mt newTextureViewWithPixelFormat:vf textureType:MTLTextureType2D levels:NSMakeRange(l, 1) slices:NSMakeRange(y, 1)];
                    size_t qoff; id<MTLBuffer> q = up_param_alloc(&qoff);
                    if (!v || !q) { enc_ok = 0; break; }
                    uint32_t *pp = (uint32_t *)((uint8_t *)q.contents + qoff); pp[0] = (uint32_t)(stg_off + off); pp[1] = (uint32_t)pitch; pp[2] = w; pp[3] = h; pp[4] = (uint32_t)bcd_fmt(t->d.format);
                    if (g_nup_decs == g_up_deccap) { g_up_deccap = g_up_deccap ? g_up_deccap * 2 : 256; g_up_decs = realloc(g_up_decs, g_up_deccap * sizeof *g_up_decs); }
                    g_up_decs[g_nup_decs++] = (up_dec){stg, q, 0, qoff, (w + 3) / 4, (h + 3) / 4, v}; [g_up_texs addObject:v]; defer_obj_unref(v);   // View: released after this submission finishes running, residency not revoked (what is resident is the base texture)
                } else
                up_blit_add(stg, stg_off + off, pitch, pitch * rows, MTLSizeMake(w, h, z), mt, y, l);
                off += pitch * rows * z;
            }
        }
        if (enc_ok) {
            rc_note_upload_src(t);   // T5: this batch's source pool is now a reclaim candidate
            [g_up_texs addObject:mt];
            if (!g_cb) up_flush();
            return mt;
        }
        g_nup_blits = nb0; g_nup_decs = nd0;   // Roll back the copies / decompressions already recorded for this texture: the already-encoded deswizzle only writes the staging area and is harmless; this texture takes the CPU path below instead
        if (!g_cb) up_flush();
        RED_ONCE(41, "预排布贴图 GPU 解排布不支持格式 %#x 目标 %d，回退 CPU", t->d.format, t->d.target);
    }

    up_begin();
    // CPU deswizzle needs a CPU-writable staging buffer, so it cannot come from up_alloc (private storage). It is retired with the batch once the GPU reports completion,
    // but each of these buffers is as large as the asset: with DM_TEX_HALF every large asset takes this path (GPU deswizzle is skipped for halving), and a city transition
    // would otherwise keep ~100 MB of finished staging buffers alive. The flush+wait at the end of this function bounds that when halving is on; other games' CPU
    // uploads are small and keep the previous batching behaviour.
    id<MTLBuffer> stg = [g_dev newBufferWithLength:total ? total : 16 options:MTLResourceStorageModeShared];
    res_add(stg); [g_up_keep addObject:stg];
    size_t off = 0;
    for (unsigned l = 0; l < d.mipmapLevelCount; l++) {
        uint32_t w = MAX(t->d.width >> (l + demote), 1u), h = MAX(t->d.height >> (l + demote), 1u), z = t->d.target == 2 ? MAX(t->d.depth >> (l + demote), 1u) : 1;
        size_t pitch = (size_t)((w + bw - 1) / bw) * bpb, rows = (h + bh - 1) / bh;
        for (unsigned y = 0; y < layers; y++) {
            if (unp) {   // No BC: first deswizzle into a temporary area (compressed blocks), then decompress layer by layer into the staging area
                uint8_t *cb = malloc(pitch * rows * z); size_t up = (size_t)w * ubpp;
                if (!cb || nvn_deswizzle(&t->d, l + demote, y, t->mem, cb, pitch)) { free(cb); RED_ONCE(9, "解排布失败 格式 %#x 目标 %d", t->d.format, t->d.target); break; }
                for (uint32_t zi = 0; zi < z; zi++) nvn_bc_decode(t->d.format, cb + zi * pitch * rows, pitch, w, h, (uint8_t *)stg.contents + off + zi * up * h, up);
                free(cb);
                up_blit_add(stg, off, up, up * h, MTLSizeMake(w, h, z), mt, y, l);
                off += up * h * z;
                continue;
            }
            if (half) {   // Deswizzle the full level, then keep every other block row / column (nearest-neighbour halving of the compressed image)
                size_t dbw = MAX((((w + bw - 1) / bw) >> 1), 1u), drow = MAX((((h + bh - 1) / bh) >> 1), 1u);
                uint8_t *lin = malloc(pitch * rows * z);
                if (!lin || nvn_deswizzle(&t->d, l + demote, y, t->mem, lin, pitch)) { free(lin); RED_ONCE(9, "解排布失败 格式 %#x 目标 %d", t->d.format, t->d.target); break; }
                for (uint32_t zi = 0; zi < z; zi++) {
                    const uint8_t *sp = lin + (size_t)zi * pitch * rows, *dp = (const uint8_t *)stg.contents + off + (size_t)zi * dbw * bpb * drow;
                    uint8_t *dout = (uint8_t *)dp;
                    for (size_t r = 0; r < drow; r++) { const uint8_t *sr = sp + (r << 1) * pitch; uint8_t *dr = dout + r * dbw * bpb;
                        for (size_t c = 0; c < dbw; c++) memcpy(dr + c * bpb, sr + (c << 1) * bpb, bpb); }
                }
                free(lin);
                up_blit_add(stg, off, dbw * bpb, dbw * bpb * drow, MTLSizeMake((uint32_t)(dbw * bw), (uint32_t)(drow * bh), z), mt, y, l);
                off += dbw * bpb * drow * z;
                continue;
            }
            if (nvn_deswizzle(&t->d, l + demote, y, t->mem, (uint8_t *)stg.contents + off, pitch)) { RED_ONCE(9, "解排布失败 格式 %#x 目标 %d", t->d.format, t->d.target); break; }
            up_blit_add(stg, off, pitch, pitch * rows, MTLSizeMake(w, h, z), mt, y, l);
            off += pitch * rows * z;
        }
    }
    rc_note_upload_src(t);   // T5: CPU-deswizzle path: same candidate bookkeeping (no address hit, but the source was read on the CPU)
    up_flush();
    if (half) up_wait();   // DM_TEX_HALF: one shared staging buffer per asset (see above); wait so a scene transition cannot pile another one up before this one retires
    return mt;
}
// Zero newly created non-packaged textures: a new Metal texture's content is undefined, while a freshly mapped memory pool on the Switch reads as 0. Games depend on it: SMBW's fountain water surface uses a 512×1 RGBA16F wave texture
// that every frame is "read last frame → render into a render target → copy back" and is never cleared; the undefined initial value keeps swinging through the feedback and pushes the water vertices to the top of the screen (the tall panels above the fountain).
// Copied level by level / slice by slice from one shared all-zero buffer, executed with the upload batch (ordered before this replay). Multisample, depth / stencil and block-compressed formats are not cleared (games always clear them before use, and they cannot be copied into anyway).
static size_t mtl_pf_bytes(MTLPixelFormat pf) {
    switch ((int)pf) {
        case 10: case 11: case 12: case 13: case 14: return 1;                                   // R8
        case 20: case 22: case 23: case 24: case 25: case 30: case 31: case 32: case 33: case 34: return 2;   // R16 / RG8
        case 40: case 41: case 42: case 43: return 2;                                            // packed 16-bit colour
        case 53: case 54: case 55: case 60: case 62: case 63: case 64: case 65:
        case 70: case 71: case 72: case 73: case 74: case 80: case 81:
        case 90: case 91: case 92: case 93: case 94: return 4;                                   // R32 / RG16 / RGBA8 / BGRA8 / 10-10-10-2 / 11-11-10 / 9-9-9-5
        case 103: case 104: case 105: case 110: case 112: case 113: case 114: case 115: return 8;   // RG32 / RGBA16
        case 123: case 124: case 125: return 16;                                                 // RGBA32
    }
    return 0;
}
// Shared all-zero buffer (for zeroing new textures and scissored zero clears): replaced by a bigger one when too small; the old one is released after in-flight submissions finish
static id<MTLBuffer> zero_buf(size_t need) {
    static id<MTLBuffer> zb; static size_t zn;
    if (need > zn) {
        size_t n = MAX(need, (size_t)4 << 20);
        id<MTLBuffer> b = [g_dev newBufferWithLength:n options:MTLResourceStorageModeShared];
        if (!b) return nil;
        memset(b.contents, 0, n); res_add(b);
        if (zb) defer_obj_release(zb);
        zb = b; zn = n;
    }
    return zb;
}
static uint64_t g_tex_zeroed;
static void tex_zero_fill(id<MTLTexture> mt) {
    static int warned;
    if (!atomic_load(&g_sub_thread_set) || !on_sub_thread()) return;   // the upload batch belongs to the submit thread only
    MTLTextureType tt = mt.textureType;
    if (tt == MTLTextureType2DMultisample || tt == MTLTextureType2DMultisampleArray || tt == MTLTextureTypeTextureBuffer) return;
    size_t bpp = mtl_pf_bytes(mt.pixelFormat);
    if (!bpp) { if (!warned++) MLOG("新建贴图清零：像素格式 %lu 不清（深度 / 模板 / 压缩等）", (unsigned long)mt.pixelFormat); return; }
    NSUInteger slices = mt.arrayLength * (tt == MTLTextureTypeCube || tt == MTLTextureTypeCubeArray ? 6 : 1);
    size_t need = (size_t)mt.width * bpp * mt.height * mt.depth;   // level 0, one slice is the largest
    if (need > (64u << 20)) { static int big; if (!big++) MLOG("新建贴图清零：%lux%lux%lu 超过 64 MB，不清", (unsigned long)mt.width, (unsigned long)mt.height, (unsigned long)mt.depth); return; }
    id<MTLBuffer> zb = zero_buf(need); if (!zb) return;
    up_begin();
    for (NSUInteger l = 0; l < mt.mipmapLevelCount; l++) {
        NSUInteger w = MAX(mt.width >> l, 1u), h = MAX(mt.height >> l, 1u), z = tt == MTLTextureType3D ? MAX(mt.depth >> l, 1u) : 1;
        for (NSUInteger y = 0; y < slices; y++) up_blit_add(zb, 0, w * bpp, w * bpp * h, MTLSizeMake(w, h, z), mt, y, l);
    }
    [g_up_texs addObject:mt];
    if (!g_cb) up_flush();
    if (!g_tex_zeroed++) MLOG("新建贴图清零：开（Metal 新贴图内容未定义，Switch 新内存读 0）");
}

// D1 diagnostic: every MTLTexture created DIRECTLY on a pool's buffer (the two zero-copy paths below and the buffer-texture path). Weak, so it
// only ever lists the ones still alive. A texture retains the MTLBuffer it was built on, so this is the only way to name the holder of a pool
// buffer that outlived its pool when every host table (alias / serial / texture ID / residency set) says it holds nothing.
static NSHashTable *g_pool_tex;
static pthread_mutex_t g_ptx_mu = PTHREAD_MUTEX_INITIALIZER;
static void pool_tex_note(id<MTLTexture> t) {
    if (!t) return;
    pthread_mutex_lock(&g_ptx_mu);
    if (!g_pool_tex) g_pool_tex = [NSHashTable weakObjectsHashTable];
    [g_pool_tex addObject:t];
    pthread_mutex_unlock(&g_ptx_mu);
}
static uint64_t pool_tex_over(id b, char *out, size_t on) {   // the live pool-backed textures wrapping b, and the shape / retain count of the first three
    uint64_t n = 0; size_t o = 0; if (on) out[0] = 0;
    pthread_mutex_lock(&g_ptx_mu);
    for (id<MTLTexture> t in g_pool_tex) {
        if (t.buffer != b) continue;
        if (n < 3 && o < on) o += (size_t)snprintf(out + o, on - o, "%s%lux%lu 格式%lu 类型%lu 用途%#lx 计数%lu", n ? "、" : "",
                                                   (unsigned long)t.width, (unsigned long)t.height, (unsigned long)t.pixelFormat,
                                                   (unsigned long)t.textureType, (unsigned long)t.usage,
                                                   (unsigned long)CFGetRetainCount((__bridge CFTypeRef)t));
        n++;
    }
    pthread_mutex_unlock(&g_ptx_mu);
    return n;
}

static int clear_rect_zero(int i);
// Linear textures (LINEAR) in a CPU-visible pool: the game writes the pool memory directly from the CPU (SMBW's glyph cache: 2048×8192 R8, IMAGE|LINEAR, pool CPU_UNCACHED|GPU_CACHED,
// the font engine rasterizes glyphs into it on the CPU). A GPU-private texture never sees those writes → all menu / dialog text is blank. On unified memory, create a linear texture directly on the pool buffer (zero copy): the GPU reads the current content at execution time, same as NVN.
static id<MTLTexture> tex_linear(const tex_t *t) {
    // LINEAR_RENDER_TARGET is a linear layout too (SMBW's sky colour ramps: four 64×1 RGBA16F, written by the CPU every frame)
    if (!(t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) || !t->mem || !t->pool || (t->pool->flags & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS)) return nil;
    if (t->d.target != NVN_TEXTURE_TARGET_2D || t->d.levels > 1 || t->d.samples > 1) return nil;
    uint32_t pf = 0; uint8_t swz[4]; int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh);
    if (!nvn_fmt_to_mtl(t->d.format, &pf, swz) || !pf || !bpb || bw != 1 || bh != 1) return nil;
    size_t row = t->d.stride ? t->d.stride : (size_t)t->d.width * bpb;
    void *mb; uint64_t mo;
    if (!mtl_addr((uint64_t)t->mem, (uint64_t)row * (t->d.height ? t->d.height : 1), &mb, &mo)) { RED_ONCE(64, "线性贴图的存储不在内存池里（%ux%u 格式 %#x）", t->d.width, t->d.height, t->d.format); return nil; }
    id<MTLBuffer> b = (__bridge id<MTLBuffer>)mb;
    NSUInteger al = [g_dev minimumLinearTextureAlignmentForPixelFormat:(MTLPixelFormat)pf];
    if (mo % al || row % al) { RED_ONCE(65, "线性贴图偏移 %#llx / 行距 %zu 不满足 %lu 字节对齐，按私有贴图建（CPU 写入不可见）", (unsigned long long)mo, row, (unsigned long)al); return nil; }
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:(MTLPixelFormat)pf width:t->d.width ? t->d.width : 1 height:t->d.height ? t->d.height : 1 mipmapped:NO];
    d.resourceOptions = b.resourceOptions;
    d.usage = MTLTextureUsageShaderRead | (t->d.flags & NVN_TEXTURE_FLAGS_IMAGE ? MTLTextureUsageShaderWrite : 0) | (t->d.flags & NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET ? MTLTextureUsageRenderTarget : 0);
    id<MTLTexture> lt = [b newTextureWithDescriptor:d offset:mo bytesPerRow:row]; pool_tex_note(lt); return lt;   // zero copy: the texture retains the pool's buffer
}
// CPU-written block-linear textures (WriteTexels): the game gave no packaged layout and changes the content from the CPU each frame (Engage's movie frames: a Y plane R8 + a UV plane RG8, WriteTexels every frame).
// Only packaged textures used to be uploaded from guest memory, so such a texture became a GPU-only texture whose content stayed 0. Their serials are recorded and they resolve through the packaged path (upload + re-upload on dirty pages)
static pthread_mutex_t g_cpu_tex_mu = PTHREAD_MUTEX_INITIALIZER; static CFMutableSetRef g_cpu_tex;
void mtl_tex_cpu_written(const tex_t *t) {
    uint64_t serial = t->serial;
    if (!serial) return;
    pthread_mutex_lock(&g_cpu_tex_mu);
    if (!g_cpu_tex) g_cpu_tex = CFSetCreateMutable(NULL, 0, NULL);
    int added = 0;
    if (!CFSetContainsValue(g_cpu_tex, (const void *)(uintptr_t)serial)) {
        CFSetAddValue(g_cpu_tex, (const void *)(uintptr_t)serial); added = 1;
        static int n; if (n++ < 16) MLOG("CPU 写贴图 ser=%llu %ux%ux%u 格式 %#x 目标 %d 级 %u：按预排布贴图上传 / 重传", (unsigned long long)serial, t->d.width, t->d.height, t->d.depth, t->d.format, t->d.target, t->d.levels);
    }
    pthread_mutex_unlock(&g_cpu_tex_mu);
    if (added) { pthread_mutex_lock(&g_smu); g_ser_gen++; pthread_mutex_unlock(&g_smu); }   // the texture IDs that resolve to it take the slow path, which now treats it as packaged
}
static int tex_cpu_written(uint64_t serial) {
    pthread_mutex_lock(&g_cpu_tex_mu); int r = g_cpu_tex && CFSetContainsValue(g_cpu_tex, (const void *)(uintptr_t)serial); pthread_mutex_unlock(&g_cpu_tex_mu);
    return r;
}
static id<MTLTexture> tex_mtl_(const tex_t *ct);
static id<MTLTexture> tex_mtl(const tex_t *ct) { AP_B; id<MTLTexture> r = tex_mtl_(ct); AP_E(4); return r; }
static uint64_t g_texinfo_ser[8]; static int g_texinfo_n;
static id<MTLTexture> tex_mtl_(const tex_t *ct) {
    tex_t *t = (tex_t *)ct;
    if (!t || !t->serial) return nil;
    if (t->magic != MAGIC('T', 'E', 'X', 0)) { RED_ONCE(70, "贴图对象已失效（magic %#x，首例 ser=%llu）：不建 MTLTexture", t->magic, (unsigned long long)t->serial); return nil; }
    int packaged = (t->has_layout || tex_cpu_written(t->serial)) && t->mem;
    {   // Diagnostic DM_NVN_TEXINFO=WxH: log storage and description of textures of that size on first resolution (to tell whether two textures share storage and why they were not merged)
        static int ti = -1; static uint32_t tw, th; static uint64_t last[64]; static int nl;
        if (ti > 0 && t->d.width == tw && t->d.height == th && g_texinfo_n < 8) { int dup = 0; for (int i = 0; i < g_texinfo_n; i++) dup |= g_texinfo_ser[i] == t->serial; if (!dup && !(t->d.flags & 8)) g_texinfo_ser[g_texinfo_n++] = t->serial; }   // remember the non-render-target ones (no COMPRESSIBLE) for DM_NVN_TEXRAW=info
        if (ti < 0) { const char *e = getenv("DM_NVN_TEXINFO"); ti = e && sscanf(e, "%ux%u", &tw, &th) == 2; }
        if (ti && t->d.width == tw && t->d.height == th) {
            int seen = 0; for (int i = 0; i < nl; i++) if (last[i] == t->serial) seen = 1;
            static uint64_t n; if (!seen && (n++ < 40 || n % 600 == 0)) { if (nl < 64) last[nl++] = t->serial; else last[n % 64] = t->serial;
                uint64_t nz = 0; if (t->mem && t->L.size <= (64u << 20)) for (uint64_t i = 0; i < t->L.size; i++) nz += t->mem[i] != 0;   // non-zero bytes in guest memory: content the CPU wrote
                MLOG("TEXINFO ser=%llu 预排布=%d has_layout=%d mem=%p pool=%p(%#x) off=%#llx 目标=%d 格式=%#x 标志=%#x 采样=%d %ux%ux%u 级=%u 跨距=%u 块高=%d 存储 %llu 字节 非零 %llu",
                     (unsigned long long)t->serial, packaged, t->has_layout, t->mem, t->pool, t->pool ? t->pool->flags : 0, (unsigned long long)t->off, t->d.target, t->d.format, t->d.flags, t->d.samples,
                     t->d.width, t->d.height, t->d.depth, t->d.levels, t->d.stride, t->d.pkg_block_h, (unsigned long long)t->L.size, (unsigned long long)nz); }
        }
    }
    id have = serial_get(t->serial);
    if (packaged) {
        NvPkg *pk = [have isKindOfClass:[NvPkg class]] ? have : nil;   // resolved as GPU-only before the CPU wrote it: dropped and rebuilt as packaged
        if (pk) {   // Already uploaded: check the covered pages only if there has been a guest write since the last check
            pk->uses++; pk->last_use_frame = atomic_load(&g_cur_frame); view_use_obj(pk->tex);
            uint64_t now = nvn_dirty_now();
            int tracked = nvn_dirty_tracked(t->mem, t->L.size);
            int clean = tracked && (now == pk->seen || !nvn_dirty_since(t->mem, t->L.size, pk->stamp));
            // A page that was written is not necessarily a page whose content changed. The game re-writes the same bytes into the same pool
            // pages (the audio banks and the GPK assets it loads on demand land there over and over). Measured on the device: the content of
            // the two hot textures never changed, yet they were re-uploaded 4096 and 8192 times -- 1.4 GB/s of pointless upload traffic and one
            // mprotect per fault. So before re-uploading, compare the content of the pages that were written with the content at upload time.
            // Only the dirty pages are compared, so the cost is proportional to what was written (the measured hot texture: ONE page of 1024).
            if (!clean && tracked && pk->phash && pk->nphash > 0) {
                static _Thread_local uint64_t cur[NVN_DIRTY_MAX_PAGES * 2];   // the page index and its current hash, per dirty page
                int nd = nvn_dirty_content_hashes(t->mem, t->L.size, pk->stamp, cur, NVN_DIRTY_MAX_PAGES, NULL, NULL);
                if (nd > 0 && nd <= pk->nphash) {
                    int same = 1;
                    for (int i = 0; i < nd; i++) {   // cur[2i] is the index of the page within the range: the reference hash is at that index
                        uint64_t idx = cur[2 * i];
                        if (idx >= (uint64_t)pk->nphash || cur[2 * i + 1] != pk->phash[idx]) { same = 0; break; }
                    }
                    if (same) {   // the same bytes: the uploaded texture is still the right one, so there is nothing to upload
                        clean = 1;
                        pk->stamp = nvn_dirty_now();   // the pages are dirty, but the content is not: the next check starts from here
                        g_pkg_sameskip++;
                        if (g_pkg_sameskip <= 8 || (g_pkg_sameskip % 500) == 0)
                            MLOG("贴图 ser=%llu 内容没变，跳过重传（脏页 %d 个；累计 %llu 次）", t->serial, nd, (unsigned long long)g_pkg_sameskip);
                    }
                }
            }
            pk->seen = now;
            if (clean && atomic_load_explicit(&g_up_lost_n, memory_order_relaxed) && up_lost_take(pk->tex)) { clean = 0; uint64_t k = atomic_load(&g_up_lost_redo); if (k <= 3 || !atomic_load(&g_up_lost_n)) MLOG("上传被丢的贴图 ser=%llu 重传（累计 %llu 张，记录里还剩 %d 张）", (unsigned long long)t->serial, (unsigned long long)k, atomic_load(&g_up_lost_n)); }
            if (clean) {
                if (!rc_src_reclaimed((uintptr_t)t->mem, t->L.size) && dirtycheck() && full_hash(t->mem, t->L.size) != pk->hash && !g_pkg_miss++)
                    MLOG("【红】脏页漏检：贴图 ser=%llu 池内容已变但未检出", t->serial);
                return pk->tex;
            }
            if (rc_src_reclaimed((uintptr_t)t->mem, t->L.size)) {   // T5: the source pages are gone (zeros); the private copy is the only content left -- keep it
                pk->stamp = nvn_dirty_now(); pk->seen = pk->stamp;
                if (++g_rc_suppress_reup <= 8 || (g_rc_suppress_reup % 500) == 0)
                    MLOG("静态源回收：贴图 ser=%llu 的源已回收，暂缓重传（私有副本保留；累计拦截 %llu 次）", (unsigned long long)t->serial, (unsigned long long)g_rc_suppress_reup);
                return pk->tex;
            }
            uint64_t old_stamp = pk->stamp, st; id<MTLTexture> nt = pkg_upload(t, &st, pk->tex);   // Pool content changed (streamed assets written into the same storage): overwrite in place when the shape matches, otherwise switch to a new texture
            if (nt) {
                if (nt != pk->tex) { id<MTLTexture> old = pk->tex; pk->tex = nt; defer_obj_release(old); res_add(nt); g_ser_gen++; }
                pk->stamp = st; pk->seen = nvn_dirty_now(); g_pkg_reuploads++; pk->reuploads++; pk->up_frame = atomic_load(&g_cur_frame);
                if (texreupload_diag() && pk->reuploads == 1) {
                    uint64_t pages = 0, first = 0, last = 0; int span = nvn_dirty_span_since(t->mem, t->L.size, old_stamp, &pages, &first, &last);
                    MLOG("TEXREUPLOAD ser=%llu 帧=%llu 使用=%llu 首次重传=%llu 预排布=%d has_layout=%d mem=%p pool=%p(%#x) off=%#llx 目标=%d 格式=%#x 标志=%#x 采样=%d %ux%ux%u 级=%u 跨距=%u 块高=%d 存储=%llu 跟踪=%d 脏页=%llu 脏字节=%llu..%llu 查询=%d",
                         (unsigned long long)t->serial, (unsigned long long)atomic_load(&g_cur_frame), (unsigned long long)pk->uses, (unsigned long long)pk->reuploads, packaged, t->has_layout,
                         (void *)t->mem, (void *)t->pool, t->pool ? t->pool->flags : 0, (unsigned long long)t->off, t->d.target, t->d.format, t->d.flags, t->d.samples,
                         t->d.width, t->d.height, t->d.depth, t->d.levels, t->d.stride, t->d.pkg_block_h, (unsigned long long)t->L.size, tracked,
                         (unsigned long long)pages, (unsigned long long)first, (unsigned long long)last, span);
                }
                // DM_NVN_TEXREUPLOAD=count: every reupload of a texture, with its serial and how many times that texture has been reuploaded.
                // The first-reupload-only line above cannot answer "is it one texture over and over, or many textures once each?" -- this one can.
                if (texreupload_count_diag()) {
                    if ((pk->reuploads & (pk->reuploads - 1)) == 0 || pk->reuploads <= 4)   // the first four, then the powers of two
                        MLOG("TEXREUPCOUNT 帧=%llu ser=%llu 第 %llu 次重传 使用=%llu 尺寸 %ux%u 存储 %llu 字节 池 %p",
                             (unsigned long long)atomic_load(&g_cur_frame), (unsigned long long)t->serial, (unsigned long long)pk->reuploads,
                             (unsigned long long)pk->uses, t->d.width, t->d.height, (unsigned long long)t->L.size, (void *)t->pool);
                }
                // DM_NVN_TEXREUPLOAD=hash: does the content actually change between two reuploads? The dirty tracker only knows that a page was
                // written, not whether the bytes differ. If the hash is the same, the reupload was pointless: the game rewrote the same bytes.
                // The first eight, then every 64th: the climbing reuploads happen much later than the first ones, so a probe that only covers
                // the first few would never see them.
                if (texreupload_hash_diag() && (pk->reuploads <= 8 || (pk->reuploads % 64) == 0)) {
                    uint64_t h = full_hash(t->mem, t->L.size);
                    MLOG("TEXREUPHASH ser=%llu 帧=%llu 第 %llu 次重传 哈希=%#llx 上次=%#llx 内容%s 存储 %llu 字节",
                         (unsigned long long)t->serial, (unsigned long long)atomic_load(&g_cur_frame), (unsigned long long)pk->reuploads,
                         (unsigned long long)h, (unsigned long long)pk->lasthash, h == pk->lasthash ? "没变" : "变了", (unsigned long long)t->L.size);
                    pk->lasthash = h;
                }
                if (dirtycheck()) pk->hash = full_hash(t->mem, t->L.size);   // the whole-texture hash: only the diagnostic needs it (it costs 5.6 ms for a 16 MB texture)
                pk_record_pages(pk, t);   // the hash of every page of the source: the next dirty check compares the pages that were written against these
            }
            return pk->tex;
        }
        pk = [NvPkg new];
        pk->tex = pkg_upload(t, &pk->stamp, nil);
        if (!pk->tex) return nil;
        pk->seen = nvn_dirty_now(); g_pkg_uploads++; pk->up_frame = atomic_load(&g_cur_frame);
        if (dirtycheck()) pk->hash = full_hash(t->mem, t->L.size);   // the whole-texture hash: only the diagnostic needs it (it costs 5.6 ms for a 16 MB texture)
        pk_record_pages(pk, t);   // the hash of every page of the source: the next dirty check compares the pages that were written against these
        serial_put(t->serial, pk);
        return pk->tex;
    }
    if (have) {
        // The fast path: the texture is already resolved, so the alias table is not consulted. A shared view must still be marked as
        // used here, or the reclaim would think a view the game binds every frame is idle and take the pool behind it.
        if (alias_has(have)) view_use_obj(have);
        return have;
    }
    if (t->d.target == 10) {   // Buffer texture (TARGET_BUFFER): create the texture-buffer view directly on the memory pool buffer (zero copy; the GPU reads the pool's current content at execution time, same as NVN semantics)
        uint32_t pf = 0; uint8_t swz[4]; int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh);
        void *mb; uint64_t mo;
        if (!nvn_fmt_to_mtl(t->d.format, &pf, swz) || !pf || !bpb) { RED_ONCE(36, "缓冲贴图格式 %#x 未映射", t->d.format); return nil; }
        if (!t->mem || !mtl_addr((uint64_t)t->mem, (uint64_t)t->d.width * bpb, &mb, &mo)) { RED_ONCE(37, "缓冲贴图的存储不在内存池里"); return nil; }
        id<MTLBuffer> b = (__bridge id<MTLBuffer>)mb;
        NSUInteger al = [g_dev minimumLinearTextureAlignmentForPixelFormat:(MTLPixelFormat)pf];
        if (mo % al) { RED_ONCE(38, "缓冲贴图偏移 %#llx 不满足 %lu 字节对齐", (unsigned long long)mo, (unsigned long)al); return nil; }
        MTLTextureDescriptor *d = [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:(MTLPixelFormat)pf width:t->d.width resourceOptions:b.resourceOptions
                                                                                         usage:MTLTextureUsageShaderRead | (t->d.flags & NVN_TEXTURE_FLAGS_IMAGE ? MTLTextureUsageShaderWrite : 0)];
        size_t row = ((size_t)t->d.width * bpb + al - 1) / al * al;
        id<MTLTexture> mt = [b newTextureWithDescriptor:d offset:mo bytesPerRow:row];
        if (!mt) { RED_ONCE(39, "缓冲贴图建不出：%u 元素 格式 %#x", t->d.width, t->d.format); return nil; }
        pool_tex_note(mt);
        res_add(mt);
        serial_put(t->serial, mt);
        return mt;
    }
    alias_key ak; memset(&ak, 0, sizeof ak); ak.pool = (uint64_t)(uintptr_t)t->pool; ak.off = t->off; ak.d = t->d;
    pthread_mutex_lock(&g_smu); __unsafe_unretained id shared = hm_get_id(&g_alias, &ak); pthread_mutex_unlock(&g_smu);
    if (shared) { serial_put(t->serial, shared); view_use_obj(shared); return shared; }
    id<MTLTexture> mt = tex_linear(t);
    if (!mt) {
        MTLTextureDescriptor *d = tex_desc(t, 0);
        if (!d) return nil;
        mt = tex_new_private(d, "游戏贴图");
        if (!mt) { RED_ONCE(2, "建 MTLTexture 失败 %ux%u 格式 %#x 目标 %d", t->d.width, t->d.height, t->d.format, t->d.target); return nil; }
        res_add(mt); tex_zero_fill(mt);   // register residency before zeroing (the copy destination must be resident too)
    }
    if (g_allocprof > 0) { static int n; if (n++ < 30 || n % 500 == 0) MLOG("新建贴图（%s）#%d ser=%llu %ux%ux%u 格式 %#x 目标 %d 级 %u 存储 %p", __func__, n, (unsigned long long)t->serial, t->d.width, t->d.height, t->d.depth, t->d.format, t->d.target, t->d.levels, (void *)t->mem); }
    if (getenv("DM_NVN_TEXCPU") && t->pool && !(t->pool->flags & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS)) MLOG("CPU 可见池上的非预排布贴图 ser=%llu %ux%ux%u 格式 %#x 目标 %d 级 %u 贴图标志 %#x 池标志 %#x 存储 %p", (unsigned long long)t->serial, t->d.width, t->d.height, t->d.depth, t->d.format, t->d.target, t->d.levels, t->d.flags, t->pool->flags, (void *)t->mem);
    res_add(mt);
    pthread_mutex_lock(&g_smu);
    if (!g_alias_set) g_alias_set = CFSetCreateMutable(NULL, 0, NULL);
    hm_put_id(&g_alias, &ak, mt); CFSetAddValue(g_alias_set, (__bridge const void *)mt);
    pthread_mutex_unlock(&g_smu);
    serial_put(t->serial, mt);
    return mt;
}
// DM_POOL_CENSUS=1: per-pool guest memory census; defined after the texture-ID registry because it walks that table (forward declaration for mtl_mem_stat)
static void pool_census(unsigned long long frame, unsigned long long rss_mb, const char *why);   // why != NULL = forced print (e.g. low-memory guard)
// Memory readout line (always on, low frequency, capped): resident memory, GPU allocated, deferred-release queue, residency set object count, registered memory pools / peak / tracking table capacity, cumulative texture re-uploads.
// Purpose: pin down "crashes with no log". The system gives no notice before killing a process for memory, so only readouts left beforehand can show which number kept climbing.
// Called by the frame hook every 300 frames; printed only when resident or GPU allocated changed by ≥ 64 MB since the last line, otherwise one line every 3000 frames; capped at 400 lines per game. All integers (the frame hook runs on the guest thread's small stack, so no floating-point formatting)
// DM_RES_CENSUS=1: break down the strong references that keep Metal resources alive, and print the largest allocations.
// This separates useful residency from resources we retired but that are still retained on a guest thread.
static void res_census(unsigned long long frame, unsigned long long rss_mb) {
    @autoreleasepool {
        unsigned long live_n[2] = {0, 0}, retired_n[2] = {0, 0};
        unsigned long long live_b[2] = {0, 0}, retired_b[2] = {0, 0};
        unsigned long tex_priv_n = 0, tex_shared_n = 0; unsigned long long tex_priv_b = 0, tex_shared_b = 0;   // Shared textures are views over pool buffers and their allocatedSize repeats the buffer's size
        enum { TOP_N = 12 };
        __strong id top_obj[TOP_N] = {nil};
        unsigned long long top_b[TOP_N] = {0};
        int top_live[TOP_N] = {0};

        pthread_mutex_lock(&g_res_mu);
        NSArray *all = g_res_ever.allObjects;
        for (id o in all) {
            if (![o conformsToProtocol:@protocol(MTLResource)]) continue;
            int kind = [o conformsToProtocol:@protocol(MTLBuffer)] ? 1 : 0;
            unsigned long long bytes = (unsigned long long)[(id<MTLResource>)o allocatedSize];
            int live = [g_res_objects containsObject:o];
            if (live) { live_n[kind]++; live_b[kind] += bytes; }
            else { retired_n[kind]++; retired_b[kind] += bytes; }
            if (live && kind == 0) { if ([(id<MTLResource>)o storageMode] == MTLStorageModePrivate) { tex_priv_n++; tex_priv_b += bytes; } else { tex_shared_n++; tex_shared_b += bytes; } }
            for (int i = 0; i < TOP_N; i++) if (bytes > top_b[i]) {
                for (int j = TOP_N - 1; j > i; j--) { top_obj[j] = top_obj[j - 1]; top_b[j] = top_b[j - 1]; top_live[j] = top_live[j - 1]; }
                top_obj[i] = o; top_b[i] = bytes; top_live[i] = live; break;
            }
        }
        pthread_mutex_unlock(&g_res_mu);

        MLOG("驻留普查 至帧 %llu：常驻 %llu MB，显卡已分配 %llu MB；驻留集里 贴图 %lu 个 %llu MB、缓冲 %lu 个 %llu MB；已退役未销毁 贴图 %lu 个 %llu MB、缓冲 %lu 个 %llu MB",
             frame, rss_mb, (unsigned long long)(g_dev.currentAllocatedSize >> 20),
             live_n[0], live_b[0] >> 20, live_n[1], live_b[1] >> 20,
             retired_n[0], retired_b[0] >> 20, retired_n[1], retired_b[1] >> 20);
        MLOG("  驻留贴图按存储：私有 %lu 个 %llu MB、共享/池视图 %lu 个 %llu MB（后者与缓冲重复计费）", tex_priv_n, tex_priv_b >> 20, tex_shared_n, tex_shared_b >> 20);
        {   // A diagnostic: the shared views grouped by their pool, with that pool's state and how long ago it was last used. A view over a
            // pool the game has not touched for a long time is a view the game dropped and the host never released.
            pthread_mutex_lock(&g_smu);
            uint32_t acap = g_alias.cap;
            uint32_t *pool_of = NULL; uint64_t *bytes_of = NULL; uint32_t npool = 0;
            if (acap) { pool_of = calloc(acap, sizeof *pool_of); bytes_of = calloc(acap, sizeof *bytes_of); }
            for (uint32_t i = 0; i < acap && pool_of; i++) {
                if (!g_alias.vals[i]) continue;
                alias_key k; memcpy(&k, g_alias.keys + (size_t)i * g_alias.klen, sizeof k);
                id o = (__bridge id)g_alias.vals[i];
                unsigned long long b = [o conformsToProtocol:@protocol(MTLResource)] ? (unsigned long long)[(id<MTLResource>)o allocatedSize] : 0;
                uint32_t j; for (j = 0; j < npool; j++) if (pool_of[j] == k.pool) break;
                if (j == npool) { if (npool >= acap) break; pool_of[npool] = k.pool; npool++; }
                bytes_of[j] += b;
            }
            pthread_mutex_unlock(&g_smu);
            pthread_mutex_lock(&g_mu);
            for (uint32_t j = 0; j < npool; j++) {
                const void *obj = (const void *)(uintptr_t)pool_of[j];
                int pi = -1, last_hit = 0, ntex = 0;
                for (int i = 0; i < g_npool; i++) if (g_pools[i].obj == obj) { pi = i; last_hit = (int)g_pools[i].last_hit_frame; ntex = (int)g_pools[i].rc_up_nser; break; }
                uint64_t now = atomic_load(&g_cur_frame);
                if (bytes_of[j] > 4u * 1024 * 1024)
                    MLOG("    共享视图池 %p：%llu MB、%d 个池、%d 个已登记贴图、上次被游戏访问在第 %d 帧（距今 %llu 帧）%s",
                         obj, bytes_of[j] >> 20, pi >= 0 ? 1 : 0, ntex, last_hit, now - (uint64_t)last_hit, (pi < 0 || now - (uint64_t)last_hit > 300) ? "  <- 游戏早就不用了" : "");
            }
            pthread_mutex_unlock(&g_mu);
            free(pool_of); free(bytes_of);
        }
        {   // A diagnostic: the full per-object list of the residency set, so two points can be diffed (which object appeared, which went away).
            // DM_RES_DUMP=<path>: one line per resident object, sorted by size, so a diff of two dumps shows what changed.
            static const char *dump_path; static int dump_done;
            if (!dump_done) {
                dump_done = 1;
                const char *e = getenv("DM_RES_DUMP");
                if (e && *e) {
                    // A relative name lands in the front end's Documents folder (on iOS that path is per-install and random).
                    if (e[0] == '/') dump_path = strdup(e);
                    else {
                        const char *doc = front_documents_dir();
                        if (doc) { char buf[1024]; snprintf(buf, sizeof buf, "%s/%s", doc, e); dump_path = strdup(buf); }
                    }
                }
            }
            if (dump_path) {
                NSMutableString *out = [NSMutableString string];
                pthread_mutex_lock(&g_smu);   // alias_has reads the alias set
                for (id o in all) {
                    if (![o conformsToProtocol:@protocol(MTLResource)]) continue;
                    if (![g_res_objects containsObject:o]) continue;
                    unsigned long long bytes = (unsigned long long)[(id<MTLResource>)o allocatedSize];
                    if ([o conformsToProtocol:@protocol(MTLTexture)]) {
                        id<MTLTexture> t = (id<MTLTexture>)o;
                        int shared = alias_has(o) ? 1 : 0;
                        [out appendFormat:@"T\t%llu\t%lux%lux%lu\tfmt%lu\ttype%lu\tparent%d\tshared%d\n", bytes, (unsigned)t.width, (unsigned)t.height, (unsigned)t.depth,
                                    (unsigned)t.pixelFormat, (unsigned)t.textureType, t.parentTexture ? 1 : 0, shared];
                    } else {
                        id<MTLBuffer> b = (id<MTLBuffer>)o;
                        [out appendFormat:@"B\t%llu\tlen%lu\n", bytes, (unsigned long)b.length];
                    }
                }
                NSArray *lines = [out componentsSeparatedByString:@"\n"];
                NSMutableArray *sorted = [NSMutableArray array];
                for (NSString *l in lines) if (l.length) [sorted addObject:l];
                [sorted sortUsingComparator:^NSComparisonResult(NSString *a, NSString *b) {
                    unsigned long long x = [[a componentsSeparatedByString:@"\t"][1] longLongValue], y = [[b componentsSeparatedByString:@"\t"][1] longLongValue];
                    return x < y ? 1 : (x > y ? -1 : [a compare:b]);
                }];
                NSMutableString *body = [NSMutableString string];
                for (NSString *l in sorted) { [body appendString:l]; [body appendString:@"\n"]; }
                pthread_mutex_unlock(&g_smu);
                [body writeToFile:@(dump_path) atomically:YES encoding:NSUTF8StringEncoding error:NULL];
                MLOG("  驻留清单已写 %s（%lu 行）", dump_path, (unsigned long)sorted.count);
            }
        }
        for (int i = 0; i < TOP_N && top_obj[i]; i++) {
            id o = top_obj[i];
            if ([o conformsToProtocol:@protocol(MTLTexture)]) {
                id<MTLTexture> t = (id<MTLTexture>)o;
                MLOG("  驻留大对象 #%d：贴图 %llu MB %s %lux%lux%lu 格式 %lu 类型 %lu 级 %lu 用途 %#lx 存储 %lu 父贴图 %d",
                     i + 1, top_b[i] >> 20, top_live[i] ? "驻留" : "已退役", (unsigned long)t.width, (unsigned long)t.height,
                     (unsigned long)t.depth, (unsigned long)t.pixelFormat, (unsigned long)t.textureType, (unsigned long)t.mipmapLevelCount,
                     (unsigned long)t.usage, (unsigned long)t.storageMode, t.parentTexture != nil);
            } else {
                id<MTLBuffer> b = (id<MTLBuffer>)o;
                MLOG("  驻留大对象 #%d：缓冲 %llu MB %s 长度 %lu 存储 %lu", i + 1, top_b[i] >> 20,
                     top_live[i] ? "驻留" : "已退役", (unsigned long)b.length, (unsigned long)b.storageMode);
            }
        }
    }
}
// D1 probe: one line per burst of pool finalizes (see the note at g_d1_del). Called from the frame hook (mtl_mem_stat, once a frame); the
// burst is reported one frame after its last delete, so a burst spread over a few frames still comes out as a single line.
// The numbers: how many pools the game destroyed, how many of them had their pages renewed and how much that was, what the three gates
// skipped in this burst (deltas of the cumulative counters), and the dead-buffer count before -> after -- the number D1 is judged on.
static void d1_probe(unsigned long long frame) {
    static unsigned long long last_report;   // reporter thread only
    if (!atomic_load_explicit(&g_d1_del, memory_order_relaxed)) return;
    // Wait for the burst to end (the game finalizes a whole batch within a frame or two), but never starve: if deletions keep coming every
    // frame (streaming), report anyway after 10 s.
    if (frame <= atomic_load_explicit(&g_d1_frame, memory_order_relaxed) && frame < last_report + 600) return;
    last_report = frame;
    uint64_t del = atomic_exchange(&g_d1_del, 0), ren = atomic_exchange(&g_d1_renew, 0), ren_b = atomic_exchange(&g_d1_renew_b, 0);
    uint64_t sm = atomic_load(&g_renew_skip_mem), so = atomic_load(&g_renew_skip_overlap), sg = atomic_load(&g_renew_skip_gpu), fl = atomic_load(&g_renew_fail);
    // The dead-buffer count is the same walk the memory line does, just at this moment: a buffer whose pool was deleted but whose MTLBuffer is
    // still alive. @autoreleasepool: objects read from a weak table come back retained + autoreleased and this runs on a guest thread.
    long dead_n = 0; unsigned long long dead_mb = 0;
    @autoreleasepool { pthread_mutex_lock(&g_mu); for (id<MTLBuffer> b in g_dead_pools) { dead_n++; dead_mb += b.length; } pthread_mutex_unlock(&g_mu); }
    dead_mb >>= 20;
    unsigned long long gpu = g_dev ? (unsigned long long)(g_dev.currentAllocatedSize >> 20) : 0;
    // The other three things a menu return can leave behind besides pool buffers: the serial table (a texture / package / program object the
    // host still holds), the shared/alias views, and the texture-heap slabs. If the dead count is flat but GPU allocated still climbs, one of
    // these is the holder -- the serial count is the first suspect.
    unsigned ns = 0, na = 0;
    pthread_mutex_lock(&g_smu); ns = g_by_serial.n; na = g_alias.n; pthread_mutex_unlock(&g_smu);
    static long g_d1_ser0 = -1; static long long g_d1_gpu0 = -1;   // reporter thread only: the same numbers at the previous report
    int ds = g_d1_ser0 < 0 ? 0 : (int)ns - (int)g_d1_ser0;
    long long dg = g_d1_gpu0 < 0 ? 0 : (long long)gpu - g_d1_gpu0;
    char before[24];
    if (g_d1_dead0 < 0) snprintf(before, sizeof before, "-"); else snprintf(before, sizeof before, "%ld", g_d1_dead0);
    MLOG("池换页批次（至帧 %llu）：删池 %llu 个，换页 %llu 个 %llu MB；本批跳过 非堆 %llu / 与活池重叠 %llu / GPU 超时 %llu，失败 %llu；死缓冲 %s → %ld 个（%llu MB）；序号对象 %u（%+d）、共用贴图 %u；显卡已分配 %llu MB（较上次 %+lld）；未换页 已回收 %llu / 无缓冲 %llu / 太小 %llu；只放手 累计 %llu；删池命中分布 %llu/%llu/%llu/%llu/%llu；放手即死 %llu / 仍被持 %llu；上传未完成 %d",
         frame, (unsigned long long)del, (unsigned long long)ren, (unsigned long long)(ren_b >> 20),
         (unsigned long long)(sm - g_d1_skip_mem0), (unsigned long long)(so - g_d1_skip_ov0), (unsigned long long)(sg - g_d1_skip_gpu0),
         (unsigned long long)(fl - g_d1_fail0), before, dead_n, dead_mb, ns, ds, na, gpu, dg,
         (unsigned long long)(atomic_load(&g_renew_nobuf_rc) - g_d1_nobuf_rc0), (unsigned long long)(atomic_load(&g_renew_nobuf) - g_d1_nobuf0),
         (unsigned long long)(atomic_load(&g_renew_skip_tiny) - g_d1_tiny0),
         (unsigned long long)atomic_load(&g_renew_droponly),
         (unsigned long long)atomic_load(&g_del_hits[0]), (unsigned long long)atomic_load(&g_del_hits[1]),
         (unsigned long long)atomic_load(&g_del_hits[2]), (unsigned long long)atomic_load(&g_del_hits[3]),
         (unsigned long long)atomic_load(&g_del_hits[4]),
         (unsigned long long)atomic_load(&g_drop_died), (unsigned long long)atomic_load(&g_drop_alive), atomic_load(&g_up_pending));
    g_d1_dead0 = dead_n; g_d1_ser0 = (long)ns; g_d1_gpu0 = (long long)gpu;   // the skip snapshots are taken at the first delete of the next burst
}

// D1 diagnostic: a pool buffer that outlived its pool -- WHO still holds it? The first version checked three places (the shared views in
// g_alias, the host's g_keep table, the deferred-release queue) and answered "Metal itself" whenever all three were empty. That answer is
// incomplete by construction: a resource registered in the residency set is retained BY THE SET, which none of the three looked at, and a
// view over the buffer is a strong holder of it (MTLTexture keeps its MTLBuffer). The measured survivors (5 per menu return, none with a
// holder by the old check) are what these extra columns are for.
// No lock is held by the caller: g_rmu is taken before g_mu elsewhere (tex_by_id holds g_rmu and mtl_addr takes g_mu), so every probe below
// is a scope of its own, and the dead buffers are collected under g_mu first and inspected after it is dropped.
// A view made from a texture ([base newTextureViewWithPixelFormat:...]) reports NO buffer of its own: only the root of its parent chain does.
// So every "does this texture hold that buffer" test has to walk to the root, or a view-of-a-pool-view (treg's alt, the bloom slice view, the
// per-level views) looks like it belongs to nothing and the holder reads as 0.
static id<MTLBuffer> tex_root_buf(id o) {
    if (!o || ![o conformsToProtocol:@protocol(MTLTexture)]) return nil;
    id<MTLTexture> t = (id<MTLTexture>)o;
    for (int hops = 0; t.parentTexture && hops < 8; hops++) t = t.parentTexture;
    return t.buffer;
}
static uint64_t dead_treg_views(id b);   // defined with the texture-ID registry below (its tables are declared further down)
static void dead_holder_report(id<MTLBuffer> b) {
    uint64_t keep = 0, av = 0, ser_v = 0, deferred = 0, in_res = 0, res_v = 0;
    // What the buffer looked like when its pool was deleted (recorded in mtl_pool_del, before any of the tables below could be consulted):
    // guest address, size, whether D1 renewed its pages, at which frame, and how many references it had once the host let go. Read first,
    // because the maps below take locks that must not be taken around g_mu.
    NvDeadMeta *dm_ = nil;
    pthread_mutex_lock(&g_mu); if (g_dead_meta) dm_ = [g_dead_meta objectForKey:b]; pthread_mutex_unlock(&g_mu);
    pthread_mutex_lock(&g_mu); for (id o in g_keep) if (o == b) keep++; pthread_mutex_unlock(&g_mu);
    pthread_mutex_lock(&g_smu);
    for (uint32_t i = 0; i < g_alias.cap; i++) {
        id o = (__bridge id)g_alias.vals[i];
        if (o && (o == b || tex_root_buf(o) == b)) av++;   // identity too: the tables can hold a buffer itself, and tex_root_buf returns nil for one
    }
    for (uint32_t i = 0; i < g_by_serial.cap; i++) {
        id o = (__bridge id)g_by_serial.vals[i];
        if (o && (o == b || tex_root_buf(o) == b)) ser_v++;
    }
    pthread_mutex_unlock(&g_smu);
    pthread_mutex_lock(&g_deferred_mu);
    if (g_deferred) for (NvDeferredRelease *dd in g_deferred) if (dd->object == b) deferred++;
    pthread_mutex_unlock(&g_deferred_mu);
    // The residency set is the one holder the first version missed, and it is a strong one. A view in it is counted too: that is the shape a
    // leak takes after "defer_obj_unref" (drop my reference, keep the residency entry), which never removes the view and so never frees it.
    pthread_mutex_lock(&g_res_mu);
    in_res = [g_res_objects containsObject:b];
    for (id o in g_res_objects)
        if (tex_root_buf(o) == b) res_v++;
    // Ground truth from the set itself, next to our own bookkeeping: res_remove drops the bookkeeping whether or not the set lets go of the
    // allocation, so g_res_objects alone cannot rule the set out as the holder. allAllocations is a snapshot of what the set really holds.
    NSArray *allocs = g_res ? [g_res allAllocations] : nil;
    pthread_mutex_unlock(&g_res_mu);
    uint64_t set_b = 0, set_v = 0;
    for (id o in allocs) { if ((id)o == b) set_b++; else if (tex_root_buf(o) == b) set_v++; }
    uint64_t treg_v = dead_treg_views(b);
    // The last resort: the retain count says whether ANY holder exists at all. Two are the probe's own (the sample array and the
    // retain+autorelease that reading a weak table hands back); anything above that is a holder none of the checks above can name.
    unsigned long rc = (unsigned long)CFGetRetainCount((__bridge CFTypeRef)b);
    char pt[256]; uint64_t ptn = pool_tex_over(b, pt, sizeof pt);
    MLOG("  死缓冲诊断：缓冲 %p、%llu 字节；视图 共用表 %llu / 序号表 %llu / 贴图号表 %llu；驻留集 账上 缓冲 %s 视图 %llu / 实有 缓冲 %llu 视图 %llu；g_keep %llu、延迟队列 %llu、引用计数 %lu；池上贴图 %llu 个（%s）；删池时 外持 %lu、%s、第 %llu 帧、客址 %p、驻留 %s、命中 %llu、池 %llu KB",
         (__bridge void *)b, (unsigned long long)b.length, av, ser_v, treg_v, in_res ? "里" : "-", res_v, set_b, set_v, keep, deferred, rc, ptn, pt,
         dm_ ? dm_->rc : 0UL, dm_ ? (dm_->renewed == 1 ? "换页" : dm_->renewed == 2 ? "退路" : "未换") : "无记录", dm_ ? dm_->frame : 0ULL,
         dm_ ? (void *)dm_->lo : NULL, dm_ ? (dm_->was_res ? "是" : "否") : "?", dm_ ? dm_->hits : 0ULL, dm_ ? (dm_->size >> 10) : 0ULL);
}

// The guest thread is a plain pthread, and nothing on it ever drains an autorelease pool. Every object a host call hands back "+0
// autoreleased" therefore stays alive for the life of the process. A pool-finalize burst is where that becomes visible: a few of the pool
// buffers (the ones that went through the texture-upload path) come back autoreleased, so `rb = nil` does not free them. Measured on the Mac
// before this existed: 8 dead buffers after the first burst, 43 after five load/quit cycles, 15 MB per return to the title -- and every
// table (residency set, g_keep, deferred queue, alias / serial / texture-ID tables) answered "not mine" because the holder is an autorelease
// entry, not a reference anyone keeps.
// One pool per guest frame fixes it exactly: draining at the start of frame N+1 releases what frame N autoreleased, which is what Cocoa's own
// per-event-loop pool does on a thread that has one. Measured with it on: 0 dead buffers at every burst over five cycles, GPU allocated flat.
// ON BY DEFAULT since the device round of 2026-10-22 (handover 4.10): four quit-to-title cycles on the phone with it on left 0 dead
// buffers at every burst and the GPU at exactly 1988 MB after each of the four; the same build without it climbed to 36 dead buffers and
// +30 MB over four loads. It still changes object lifetimes across the whole guest thread, so the switch stays: DM_ARPOOL=0 turns it off,
// =frame is the wider variant, =sub (the default) is the one both surfaces were measured with.
static int arpool_mode(void) {
    static int v = -1;
    if (v >= 0) return v;
    const char *e = getenv("DM_ARPOOL");   // unset or "sub" = on (one pool per submission) / "frame" = one per guest frame / "0", "off" = off
    v = ARPOOL_SUB;
    if (e && *e) v = (!strcmp(e, "0") || !strcmp(e, "off")) ? ARPOOL_OFF : ((!strcmp(e, "frame") || !strcmp(e, "1")) ? ARPOOL_FRAME : ARPOOL_SUB);
    return v;
}
// One rolling pool per scope: push at the start of this scope, pop at the start of the next one, so what the previous scope autoreleased is
// released before this scope runs. Two scopes can share it -- each caller has its own static slot.
static void arpool_roll(void **slot) {
    if (*slot) objc_autoreleasePoolPop(*slot);
    *slot = objc_autoreleasePoolPush();
}
static void guest_frame_arpool(void) {
    static void *ar;
    if (arpool_mode() != ARPOOL_FRAME) return;
    arpool_roll(&ar);
}

// DM_POOL_SELFTEST=1 (measurement): does a bytesNoCopy MTLBuffer ever die on this driver once the host's last reference is gone? Builds one
// over this process's own page-aligned memory with exactly the options a pool uses, next to an ordinary shared buffer as a control, drops both,
// and reports at the following memory lines how many of each are still alive. If the no-copy one outlives the ordinary one, the survivor class in
// the dead-buffer report is the driver's own bookkeeping -- not a holder inside this host -- and no amount of table-clearing will free it.
static NSHashTable *g_st_nocopy, *g_st_owned; static int g_st_stage;
static void pool_selftest(unsigned long long frame) {
    static int on = -1;
    if (on < 0) on = getenv("DM_POOL_SELFTEST") && *getenv("DM_POOL_SELFTEST") != '0';
    if (!on) return;
    if (g_st_stage == 0 && frame >= 30) {
        g_st_nocopy = [NSHashTable weakObjectsHashTable]; g_st_owned = [NSHashTable weakObjectsHashTable];
        size_t pg = (size_t)getpagesize(), len = pg * 4;
        for (int i = 0; i < 4; i++) {
            void *m = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
            if (m == MAP_FAILED) break;
            memset(m, i + 1, len);
            id<MTLBuffer> a = [g_dev newBufferWithBytesNoCopy:m length:len options:MTLResourceStorageModeShared deallocator:nil];
            id<MTLBuffer> b = [g_dev newBufferWithLength:len options:MTLResourceStorageModeShared];
            if (a) [g_st_nocopy addObject:a];
            if (b) [g_st_owned addObject:b];
            MLOG("池缓冲自检：第 %d 个 无拷贝 %p(计数 %lu) / 自持 %p(计数 %lu)", i, (__bridge void *)a,
                 a ? (unsigned long)CFGetRetainCount((__bridge CFTypeRef)a) : 0UL, (__bridge void *)b,
                 b ? (unsigned long)CFGetRetainCount((__bridge CFTypeRef)b) : 0UL);
            a = nil; b = nil;
        }
        g_st_stage = 1;
        MLOG("池缓冲自检：已放开 4 个无拷贝缓冲与 4 个自持缓冲（帧 %llu），此刻 无拷贝 %lu / 自持 %lu", frame, (unsigned long)g_st_nocopy.count, (unsigned long)g_st_owned.count);
    } else if (g_st_stage == 1 && (frame == 60 || frame == 120 || frame == 600)) {
        MLOG("池缓冲自检：帧 %llu 仍活 无拷贝 %lu / 自持 %lu", frame, (unsigned long)g_st_nocopy.count, (unsigned long)g_st_owned.count);
        if (frame == 600) g_st_stage = 2;
    }
}

void mtl_mem_stat(unsigned long long frame, unsigned long long rss_mb) {
    guest_frame_arpool();
    dead_watch("帧前");
    pool_selftest(frame);
    d1_probe(frame);   // D1: one line per burst of pool finalizes (before the throttle below, so a menu return is never thinned out)
    if (g_res_census > 0 && frame % 300 == 0) res_census(frame, rss_mb);   // mtl_mem_stat itself now runs every frame; the census walks every tracked resource and stays on the old cadence
    pool_census(frame, rss_mb, NULL);   // DM_POOL_CENSUS: no-op unless enabled; every 100 frames while on (a real-device run can die around frame 400-600)
    lowmem_poll(frame);   // low-memory guard: one available-memory read per frame, deferred-queue drain while active (iOS; no-op on Mac unless DM_LOWMEM_SIM_AVAIL_MB is set)
    if (g_pool_census_on) {
        // Forced census prints, so the fatal transition state is captured even though the process dies inside that frame:
        //  - DM_POOL_CENSUS_FORCE_FRAME=<n>: one print at that frame (any platform; used to verify this path on the Mac);
        //  - iOS low memory: when os_proc_available_memory drops below DM_POOL_CENSUS_LOWMEM_MB (default 260), force up to 4 prints, 30 frames apart (the second one
        //    only happens if the process survives that long; the first one is the transition state that killed it).
        static long force_frame = -2;
        if (force_frame == -2) { const char *e = getenv("DM_POOL_CENSUS_FORCE_FRAME"); force_frame = e ? atol(e) : 0; }
        if (force_frame > 0 && frame == (unsigned long long)force_frame) pool_census(frame, rss_mb, "DM_POOL_CENSUS_FORCE_FRAME");
#if TARGET_OS_IPHONE
        static unsigned long long last_force_frame; static int forces; static int lowmem_mb = -1;
        if (lowmem_mb < 0) { const char *e = getenv("DM_POOL_CENSUS_LOWMEM_MB"); lowmem_mb = e && atoi(e) > 0 ? atoi(e) : 260; }
        unsigned long long av = (unsigned long long)(os_proc_available_memory() >> 20);
        if (av < (unsigned long long)lowmem_mb && forces < 4 && (!last_force_frame || frame >= last_force_frame + 30)) {
            char why[96]; snprintf(why, sizeof why, "低内存强制：可用 %llu MB < %d MB", av, lowmem_mb);
            pool_census(frame, rss_mb, why); last_force_frame = frame; forces++;
        }
#endif
        // The guard went active (threshold above): capture the transition state too, so the first census is not lost to the print cadence
        if (atomic_load(&g_lowmem_active)) {
            static unsigned long long last_guard_frame; static int guard_prints;
            if (guard_prints < 4 && (!last_guard_frame || frame >= last_guard_frame + 120)) {
                char why[96]; snprintf(why, sizeof why, "低内存保护生效：可用 %llu MB", atomic_load(&g_lowmem_avail_mb));
                pool_census(frame, rss_mb, why); last_guard_frame = frame; guard_prints++;
            }
        }
    }
    static unsigned long long last_rss, last_gpu, last_frame; static int lines; unsigned long long gpu = (unsigned long long)(g_dev.currentAllocatedSize >> 20);
    int lowmem_act = atomic_load(&g_lowmem_active);   // low-memory guard state for the readout below
    unsigned long long dr = rss_mb > last_rss ? rss_mb - last_rss : last_rss - rss_mb, dg = gpu > last_gpu ? gpu - last_gpu : last_gpu - gpu;
    if (lines >= 400 || (lines && dr < 64 && dg < 64 && frame - last_frame < 3000)) return; lines++; last_rss = rss_mb; last_gpu = gpu; last_frame = frame;
    pthread_mutex_lock(&g_deferred_mu); unsigned long nd = (unsigned long)g_deferred.count; pthread_mutex_unlock(&g_deferred_mu);
    pthread_mutex_lock(&g_res_mu); unsigned long nr = (unsigned long)g_res_objects.count; pthread_mutex_unlock(&g_res_mu);
    int live = 0, peak = 0, cap = 0; nvn_dirty_stat(&live, &peak, &cap);
    pthread_mutex_lock(&g_smu); unsigned ns = g_by_serial.n, ashared = g_alias.n; pthread_mutex_unlock(&g_smu);
    unsigned long dead_n = 0; unsigned long long dead_mb = 0;
    // Residency of the guest memory pools: total buffer length wired through the residency set, split by pool class (the wired guest pages are exactly what the guest
    // resident class sees; textures are charged separately). Only walked when this line actually prints, not every frame.
    int pres = 0; unsigned long long pres_buf = 0, cls_buf[3] = {0, 0, 0}; int cls_pres[3] = {0, 0, 0};
    // Objects read from a weak table come back retained + autoreleased: the frame hook runs on a guest thread with no autorelease pool that drains, so without this wrapper every buffer counted would be pinned for good (the readout creating the leak itself)
    id<MTLBuffer> live_buf = nil; unsigned long live_rc = 0;   // calibration: the count on a pool buffer the host is STILL holding, next to the dead ones
    @autoreleasepool { pthread_mutex_lock(&g_mu); pres = g_pool_res_n;
        if (g_npool && g_pools[0].buf) { live_buf = (id<MTLBuffer>)g_pools[0].buf; live_rc = (unsigned long)CFGetRetainCount((__bridge CFTypeRef)live_buf) - 1; }
        for (int i = 0; i < g_npool; i++) if (g_pools[i].res) { int c = pool_class(g_pools[i].flags); pres_buf += g_pools[i].buflen; cls_buf[c] += g_pools[i].buflen; cls_pres[c]++; }
        id<MTLBuffer> samp[48]; int nsamp = 0;
        for (id<MTLBuffer> b in g_dead_pools) {
            dead_n++; dead_mb += b.length;
            if (nsamp < 48) samp[nsamp++] = b;   // every dead buffer, so a single print covers the whole set
        }
        pthread_mutex_unlock(&g_mu);
        // The holder probe needs locks of its own and must not run under g_mu (see dead_holder_report).
        static int dn_pr, diag = -1;
        if (diag < 0) { const char *e = getenv("DM_DEAD_DIAG"); diag = !(e && *e == '0'); }   // DM_DEAD_DIAG=0: no holder reports, so the watch can be read without them
        int per = dn_pr < 144 ? 48 : 1;   // the first few prints cover every sampled buffer, later ones one per line
        if (diag) for (int s = 0; s < nsamp && s < per; s++, dn_pr++) dead_holder_report(samp[s]);
    } dead_mb >>= 20;   // host objects held by serial (textures / packaged textures / programs): if it climbs along with GPU allocated, textures are not being released
    // os_proc_available_memory: how far this process is from being terminated by the system for memory, iOS only (from d2r-mobile 704147218); the Mac writes 「无」(none). On a phone, a crash with no log is most likely this number reaching 0
    char rcs[448] = "";
    if (g_tex_reclaim || g_tex_reclaim_dry) {
        // The pools by reclaim state: a pool that is still a candidate is waiting for its turn (it will be reclaimed), a pool that was rejected hard will never be.
        int rq = 0, rwait = 0, rrej = 0, rdone = 0;
        for (int i = 0; i < g_npool; i++) {
            if (g_pools[i].rc_state == RC_RECLAIMED) { rdone++; continue; }
            if (g_pools[i].rc_no_reclaim) rrej++;
            else if (g_pools[i].rc_queued) rq++;
            else rwait++;
        }
        snprintf(rcs, sizeof rcs, "，静态源回收 %llu 池 %llu MB（候选 %llu 池 %llu MB，失效 %llu，拦截重传 %llu；池状态 待评估 %d / 排队中 %d / 已硬拒 %d / 已回收 %d；预算 在用 %llu/%llu MB，区间表 %llu/%d）",
                 (unsigned long long)g_rc_pools, (unsigned long long)(g_rc_bytes >> 20), (unsigned long long)g_rc_cand_pools,
                 (unsigned long long)(g_rc_cand_bytes >> 20), (unsigned long long)g_rc_invalid, (unsigned long long)g_rc_suppress_reup,
                 rwait, rq, rrej, rdone,
                 (unsigned long long)(atomic_load(&g_rc_live_bytes) >> 20), (unsigned long long)g_rc_budget_mb,
                 (unsigned long long)atomic_load(&g_rc_rg_n), RC_RG_MAX);
    }
    if (g_pool_renew > 0) {   // D1: pool page renew, next to the "dead buffers still alive" count it is meant to bring down
        char pr[256];
        snprintf(pr, sizeof pr, "，池换页 %llu 池 %llu MB（跳过 非堆 %llu / 与活池重叠 %llu / GPU 超时 %llu，失败 %llu，累计耗时 %.1f ms）",
                 (unsigned long long)atomic_load(&g_renew_pools), (unsigned long long)(atomic_load(&g_renew_bytes) >> 20),
                 (unsigned long long)atomic_load(&g_renew_skip_mem), (unsigned long long)atomic_load(&g_renew_skip_overlap),
                 (unsigned long long)atomic_load(&g_renew_skip_gpu),
                 (unsigned long long)atomic_load(&g_renew_fail), atomic_load(&g_renew_ns) / 1e6);
        strncat(rcs, pr, sizeof rcs - strlen(rcs) - 1);
    }
    char avail[32] = "无";
#if TARGET_OS_IPHONE
    snprintf(avail, sizeof avail, "%llu MB", (unsigned long long)(os_proc_available_memory() >> 20));
#endif
    MLOG("内存 至帧 %llu：常驻 %llu MB，显卡已分配 %llu MB，延迟释放队列 %lu，驻留集 %lu 个，序号对象 %u 个，内存池 %d 个（其中已驻留 %d；峰值 %d / 跟踪表容量 %d；已释放仍活 %lu 个 %llu MB），共用贴图 %u 张（随池淘汰累计 %llu），贴图重传累计 %llu 次（原地 %llu），每帧竞技场累计新建 %llu 块，驻留池缓冲 %d 个 %llu MB（CPU不可见 %d 个 %llu MB / CPU可见 %d 个 %llu MB / 其他 %d 个 %llu MB）；低内存保护 %s（进入 %d 次）；离系统内存上限还有 %s%s；活池缓冲 %p 计数 %lu", frame, rss_mb, gpu, nd, nr, ns, live, pres, peak, cap, dead_n, dead_mb, ashared, (unsigned long long)g_alias_evicted, (unsigned long long)g_pkg_reuploads, (unsigned long long)g_pkg_inplace, (unsigned long long)g_arena_made,
         pres, (unsigned long long)(pres_buf >> 20), cls_pres[0], (unsigned long long)(cls_buf[0] >> 20), cls_pres[1], (unsigned long long)(cls_buf[1] >> 20), cls_pres[2], (unsigned long long)(cls_buf[2] >> 20), lowmem_act ? "开" : "关", g_lowmem_enters, avail, rcs, (__bridge void *)live_buf, live_rc);
    if (g_allocprof > 0) { char ab[256]; int o = 0; for (int i = 0; i < g_alloc_n && o < 200; i++) o += snprintf(ab + o, sizeof ab - o, " #%d %llu MB", i, (unsigned long long)(g_alloc4[i].allocatedSize >> 20));
        for (int i = 0; i < 3 && o < 240; i++) o += snprintf(ab + o, sizeof ab - o, " 传%d %llu 出%d %llu", i, (unsigned long long)(g_up_alloc4[i].allocatedSize >> 20), i, (unsigned long long)(g_present_alloc4[i].allocatedSize >> 20));
        MLOG("命令分配器占用：%s", ab);
        MLOG("显卡分配增长按函数：结束通道 %llu MB，开通道 %llu MB，取管线 %llu MB，参数缓冲 %llu MB，取贴图 %llu MB", g_ap_fn[0] >> 20, g_ap_fn[1] >> 20, g_ap_fn[2] >> 20, g_ap_fn[3] >> 20, g_ap_fn[4] >> 20); }
    if (g_allocprof > 0) { int top[4] = {-1, -1, -1, -1}; for (int o = 0; o < 64; o++) for (int k = 0; k < 4; k++) if (top[k] < 0 || g_ap_bytes[o] > g_ap_bytes[top[k]]) { for (int j = 3; j > k; j--) top[j] = top[j - 1]; top[k] = o; break; }
        MLOG("显卡分配增长按操作：op%d %llu MB，op%d %llu MB，op%d %llu MB，op%d %llu MB", top[0], g_ap_bytes[top[0]] >> 20, top[1], g_ap_bytes[top[1]] >> 20, top[2], g_ap_bytes[top[2]] >> 20, top[3], g_ap_bytes[top[3]] >> 20); }
}
void mtl_pkg_stats(uint64_t *uploads, uint64_t *reuploads) { *uploads = g_pkg_uploads; *reuploads = g_pkg_reuploads; }
uint64_t mtl_pkg_sameskip_count(void) { return g_pkg_sameskip; }   // the reuploads that were skipped because the content had not changed
uint64_t mtl_pkg_miss(void) { return g_pkg_miss; }
void mtl_tex_release(tex_t *t) { if (t->serial) serial_del(t->serial); }

// ---------- Render passes: SetRenderTargets opens a new pass; clears before the first command in the pass are folded into the load action ----------
static struct {
    p_set_rt rt; int has_rt;
    uint8_t clear_c[8]; MTLClearColor cc[8];
    uint8_t clear_d, clear_s; double cd; uint32_t cs;
    p_scissor sc; int has_sc;
} R;

// ---------- Zero-source skip ----------
// Shader fact table (statically proven keys): 'the textures it samples are all 0 and the coefficient is finite ⇒ the output is all 0'. Coefficient = component comp of vector vec of fragment-stage constant buffer register ubo_slot.
//   4a819818b8e489c5 ps_outline_blur_p: output = (sum of several samples of the same texture along a row / column) × fp_c7._m0[1].x (fp_c7 = cbuf reg 4), homogeneous linear in the sampled values, no constant term
//   (source of the proof: dmsh_b/<chain version>/4a819818b8e489c5_b.metal, read line by line). Add other shaders here once they are proven.
typedef struct { char key[17]; int ubo_slot, vec, comp; } zskip_fact;
static zskip_fact g_zskip_facts[16] = {{"4a819818b8e489c5", 4, 1, 0}};
static int g_zskip_nf = 1;
void zskip_test_add(const char *key, int ubo_slot, int vec, int comp) {
    if (!key || g_zskip_nf >= 16) return;
    snprintf(g_zskip_facts[g_zskip_nf].key, sizeof g_zskip_facts[0].key, "%s", key);
    g_zskip_facts[g_zskip_nf].ubo_slot = ubo_slot; g_zskip_facts[g_zskip_nf].vec = vec; g_zskip_facts[g_zskip_nf].comp = comp; g_zskip_nf++;
}
int zskip_test_count(void) { return g_zskip_count; }
static int g_zskip_env = -1;   // Cache of DM_NVN_ZSKIP (−1 = not read). This function is called on every draw, and getenv once took about 8% of the submit path; unit tests set it back to −1 after toggling the switch mid-run
static int zskip_on(void) { if (g_zskip_env < 0) { const char *e = getenv("DM_NVN_ZSKIP"); g_zskip_env = !e || strcmp(e, "0"); } return g_zskip_env && tbdr_on(); }
static void zskip_clear(void) { g_zskip_n = 0; memset(g_zskip_cand_ok, 0, sizeof g_zskip_cand_ok); }
static void zskip_del(uint64_t ser) { for (int i = 0; i < g_zskip_n;) if (g_zskip_zero[i].ser == ser) g_zskip_zero[i] = g_zskip_zero[--g_zskip_n]; else i++; }
static int zskip_has(uint64_t ser, int lv, int sl) { for (int i = 0; i < g_zskip_n; i++) if (g_zskip_zero[i].ser == ser && g_zskip_zero[i].level == lv && g_zskip_zero[i].slice == sl) return 1; return 0; }
static void zskip_add(uint64_t ser, int lv, int sl) { if (!ser || zskip_has(ser, lv, sl) || g_zskip_n >= 4096) return; g_zskip_zero[g_zskip_n++] = (zskip_zero){ser, (uint16_t)lv, (uint16_t)sl}; }
static int zskip_zero_clear(int i) { return R.clear_c[i] && R.cc[i].red == 0 && R.cc[i].green == 0 && R.cc[i].blue == 0 && R.cc[i].alpha == 0; }
static void zskip_commit_clears(void) { for (int i = 0; i < 8; i++) if (g_zskip_cand_ok[i]) zskip_add(g_zskip_cand[i].ser, g_zskip_cand[i].level, g_zskip_cand[i].slice); }
static int zskip_target_zero(const tex_t *t, const view_t *v, int hv) { int lv = hv && v->set_levels ? v->base_level : 0, sl = hv && v->set_layers ? v->min_layer : 0; return t && zskip_has(t->serial, lv, sl); }
#ifdef NVN_RT_RANGE_TEST
static int g_rr_checking;
static void rr_begin(void), rr_finish(int ranged);
static void rr_snap(const p_draw *, const p_draw_idx *);
static void rr_reset(void);
#endif
static id<MTL4RenderCommandEncoder> g_enc;
static int enc_open(void) { return g_enc != nil; }   // Whether a render pass is open (an inline uniform update inside one can be deferred instead of ending the pass, see ubd_*)
// Target info of the current render pass: determined once from the attachments at rp_begin and shared by all draws in the pass (pipeline key, depth-stencil, viewport / scissor all depend only on it)
static struct { uint32_t cfmt[8], dfmt, sfmt, samples; int hd, hs; NSUInteger tw, th; uint64_t serial, reg_gen, ser_gen, range_dirty; int range_external, range_depth_nonro; __unsafe_unretained id<MTLTexture> range_color, range_depth; } g_pass;
static id<MTLRenderPipelineState> g_enc_pso; static id<MTLDepthStencilState> g_enc_dss;   // Already set on this encoder (not set again if identical)
static int g_enc_vis, g_enc_has_vis;   // This encoder's current visibility count mode (0 off / slot number + 1), and whether a visibility buffer is attached
static int g_pso_dirty = 1, g_dss_dirty = 1;   // Pipeline / depth-stencil related state was changed during replay
static uint64_t g_stat_args_enc, g_stat_args_reuse;   // Argument buffer: new-encode / reuse counts
static uint64_t g_stat_draw, g_stat_pass, g_bw_load, g_bw_store, g_pass_full_rw, g_udc_saved_bytes;   // Bandwidth ledger: attachment load / store bytes, number of full load-store passes
uint64_t mtl_pass_count(void) { return g_stat_pass; }   // the render passes opened so far (cumulative)
uint64_t mtl_pc_stats(uint64_t *hit, uint64_t *miss) { *hit = g_pc_hit; *miss = g_pc_miss; }   // the pool cache hits and misses
// DM_NVN_PASSDUMP: records pass attachments / samples / presents one by one (textures by serial) for offline data-flow analysis; the pass numbers of P / D / R / S lines agree (= g_stat_pass after the pass is opened)
// DM_NVN_PASSDUMP=frame number, or start-end (inclusive at both ends)
static int g_kd_force;   // the draw DM_NVN_KEYDUMP matched: force the per-binding records on (see kd_draw)
static int passdump(void) {
    if (g_kd_force) return 1;
#ifdef NVN_RT_RANGE_TEST
    if (g_rr_checking) return 0;
#endif
    static long a0 = -2, a1; if (a0 == -2) { const char *e = getenv("DM_NVN_PASSDUMP"); a0 = e ? atol(e) : -1; a1 = e && strchr(e, '-') ? atol(strchr(e, '-') + 1) : a0; }
    long f = (long)atomic_load(&g_cur_frame); return a0 >= 0 && f >= a0 && f <= a1;
}
#define PD(...) do { if (passdump()) { fprintf(stderr, "[pass] F%llu ", (unsigned long long)atomic_load(&g_cur_frame)); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)
static void blit_note(void) { if (g_enc) { g_pass_split++; PD("E %llu 拷贝 / 缓冲更新结束渲染通道", g_stat_pass); } }
// DM_GFXUSE=<file>: once the character select screen is stable (starting 1 second after the first OpenPanel MainMenuButtonRibbon, requires DM_BOOTPROF=1), over 30 consecutive frames,
// the ranges in guest memory of the resources the GPU actually reads (texture storage, UBO / SSBO, vertex / index buffers), deduplicated by (kind, address, length), one line recorded on first sight.
// For intersecting with the inflate output ranges (which data decompressed before character select is really used by the character select screen). Columns: kind  guest_addr  len  first_frame  tex_id  fmt  w×h
__attribute__((weak)) uint64_t dm_prof_ribbon_ns(void) { return 0; }   // diag.c provides the strong definition; when not merged it is always 0 → the time window does not open, and a failure is reported at the end
__attribute__((weak)) void dm_shot_written(const char *path, int ok) { (void)path; (void)ok; }
__attribute__((weak)) void dm_guest_frame_tick(void) {}   // hle_os.c provides the strong definition (DM_DETCLOCK deterministic guest clock); a no-op when graphics unit tests do not link the host   // diag.c provides the strong definition (DM_TICK_HOLD=E:SHOT unfreeze); a no-op when graphics unit tests do not link the host
__attribute__((weak)) int dm_detclock_enabled(void) { return 0; }   // Same as above: the host's hle_os.c provides the strong definition; when graphics unit tests do not link the host, the deterministic clock is treated as off
__attribute__((weak)) uint64_t guest_ns(void) { return 0; }          // Same as above: read only when the deterministic clock is on; never reached in graphics unit tests
enum { USE_OFF, USE_WAIT, USE_REC, USE_DONE };
static int g_use = -1; static FILE *g_use_f; static long g_use_f0, g_use_rows; static nvn_hmap g_use_seen = {.klen = 24};
#define USE_FRAMES 30
static void use_exit(void) {
    if (g_use == USE_WAIT) fprintf(stderr, "[dm] 【红】DM_GFXUSE 时间窗始终没开：dm_prof_ribbon_ns() = %llu（需 DM_BOOTPROF=1 且宿主含面板打点）\n", (unsigned long long)dm_prof_ribbon_ns());
    else if (g_use == USE_REC) fprintf(stderr, "[dm] 【红】DM_GFXUSE 只记了 %ld 帧（不足 %d）即退出，文件未完整\n", (long)atomic_load(&g_cur_frame) - g_use_f0, USE_FRAMES);
    if (g_use_f) fflush(g_use_f);
}
static void use_tick(void) {   // Called on every present
    if (g_use < 0) { const char *e = getenv("DM_GFXUSE"); g_use = USE_OFF;
        if (e) { g_use_f = fopen(e, "w"); if (!g_use_f) { MLOG("【红】DM_GFXUSE 打不开 %s", e); return; }
            fprintf(g_use_f, "kind\tguest_addr\tlen\tfirst_frame\ttex_id\tfmt\twxh\n"); fflush(g_use_f); g_use = USE_WAIT; atexit(use_exit);
            MLOG("[gfxuse] 等面板事件：首次 OpenPanel MainMenuButtonRibbon 后 1 秒起记 %d 帧 → %s", USE_FRAMES, e); } }
    long f = (long)atomic_load(&g_cur_frame);
    if (g_use == USE_WAIT) {
        uint64_t t = dm_prof_ribbon_ns();
        if (t && now_ns() - t >= 1000000000ull) { g_use = USE_REC; g_use_f0 = f; MLOG("[gfxuse] 开始记录：帧 %ld 起 %d 帧", f, USE_FRAMES); }
    } else if (g_use == USE_REC && f - g_use_f0 >= USE_FRAMES) {
        g_use = USE_DONE; fclose(g_use_f); g_use_f = NULL;
        MLOG("[gfxuse] 记录完成：帧 %ld–%ld，去重后 %ld 个区间", g_use_f0, f - 1, g_use_rows);
    }
}
static __attribute__((always_inline)) inline void use_rec(const char *kind, uint64_t addr, uint64_t len, int tid, const tex_t *t) {
    if (g_use != USE_REC || !addr || !len) return;
    uint64_t k[3] = {addr, len, (uint64_t)(uint8_t)kind[0]};
    if (nvn_hm_get(&g_use_seen, k)) return;
    nvn_hm_put(&g_use_seen, k, (void *)1);
    g_use_rows++;
    if (t) fprintf(g_use_f, "%s\t%#llx\t%llu\t%lld\t%d\t%#x\t%ux%u\n", kind, (unsigned long long)addr, (unsigned long long)len, (long long)atomic_load(&g_cur_frame), tid, t->d.format, t->d.width, t->d.height);
    else fprintf(g_use_f, "%s\t%#llx\t%llu\t%lld\t\t\t\n", kind, (unsigned long long)addr, (unsigned long long)len, (long long)atomic_load(&g_cur_frame));
}
static uint64_t att_bytes(id<MTLTexture> t, NSUInteger level) {
    if (!t) return 0;
    NSUInteger w = MAX(t.width >> level, 1), h = MAX(t.height >> level, 1);
    NSUInteger bpp = t.pixelFormat == MTLPixelFormatDepth32Float_Stencil8 ? 5 : t.pixelFormat == MTLPixelFormatRGBA16Float || t.pixelFormat == MTLPixelFormatRG32Float ? 8 :
                     t.pixelFormat == MTLPixelFormatRGBA32Float ? 16 : t.pixelFormat == MTLPixelFormatR8Unorm || t.pixelFormat == MTLPixelFormatStencil8 ? 1 :
                     t.pixelFormat == MTLPixelFormatRG8Unorm || t.pixelFormat == MTLPixelFormatR16Float ? 2 : 4;
    return (uint64_t)w * h * bpp * t.sampleCount;
}
// The render attachment selects a particular mip, while MTLTexture.width/height still describe the base level. Empty attachments do not take part in the bounds.
static NSUInteger mip_extent(NSUInteger size, NSUInteger level) {
    return level < sizeof(size) * 8 ? MAX(size >> level, 1) : 1;
}
static void pass_dims(MTL4RenderPassDescriptor *pd, NSUInteger nc, NSUInteger *w, NSUInteger *h) {
    *w = *h = 0;
    for (NSUInteger i = 0; i < nc + 2; i++) {
        MTLRenderPassAttachmentDescriptor *a = i < nc ? pd.colorAttachments[i] : i == nc ? pd.depthAttachment : pd.stencilAttachment;
        if (!a.texture) continue;
        NSUInteger aw = mip_extent(a.texture.width, a.level), ah = mip_extent(a.texture.height, a.level);
        *w = *w ? MIN(*w, aw) : aw; *h = *h ? MIN(*h, ah) : ah;
    }
    if (!*w) *w = *h = 1;
}

// ---------- DM_NVN_PASSGPU=<file>: per-pass GPU time (stage-boundary timestamp counters) ----------
// Each render pass samples 4 points (vertex start/end, fragment start/end), a compute pass 2 points; resolved after the command buffer completes and accumulated by 'index of the pass within the frame'.
// Each line: frame idx kind gpu_ms inc_ms w h fmt0 dfmt load draws t0_ns t1_ns (kind: R render / C compute; load: L load / C clear for each color attachment + depth)
// On TBDR passes overlap in the pipeline (the later pass's vertex stage is earlier than the end of the earlier pass's fragment stage): gpu_ms = this pass's start-to-end span (including queueing) and cannot be summed;
// attribution uses the timeline increment inc_ms = t1 − max(t0, latest t1 of the passes completed so far); the sum of the increments of all passes = total time the GPU is busy (command buffers complete in submission order)
typedef struct { uint64_t frame, pass; int idx, kind, w, h, draws, slot; uint32_t cfmt0, dfmt; char load[12]; } pg_pass;   // pass: global pass number (same as the pass number in the DM_NVN_PASSDUMP lines, for joining the two records)
#define PG_SLOTS 4096   // Counter sample buffer limit 32768 bytes = 4096 timestamps (measured from the error on this machine)
#define PG_RING 8
static int g_pg = -1; static FILE *g_pg_f;
static id<MTLCounterSampleBuffer> g_pg_buf[PG_RING]; static int g_pg_b; static uint32_t g_pg_used;
static pg_pass *g_pg_pass; static int g_pg_np, g_pg_cap, g_pg_open = -1;
static uint64_t g_pg_frame = ~0ull; static int g_pg_idx;
static pthread_mutex_t g_pg_mu = PTHREAD_MUTEX_INITIALIZER;
static inline void args_dirty_all(void);
typedef struct { double sum, last; uint64_t n; pg_pass meta; } pg_stat;
static pg_stat g_pg_st[256]; static uint64_t g_pg_rep_frame;
static int pg_on(void) {
    if (g_pg >= 0) return g_pg;
    const char *e = getenv("DM_NVN_PASSGPU"); g_pg = 0;
    if (!e) return 0;
    if (![g_dev supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) { MLOG("【红】DM_NVN_PASSGPU：本设备不支持阶段边界计数采样"); return 0; }
    id<MTLCounterSet> ts = nil; for (id<MTLCounterSet> c in g_dev.counterSets) if ([c.name isEqualToString:MTLCommonCounterSetTimestamp]) ts = c;
    if (!ts) { MLOG("【红】DM_NVN_PASSGPU：没有时间戳计数器集"); return 0; }
    MTLCounterSampleBufferDescriptor *d = [MTLCounterSampleBufferDescriptor new];
    d.counterSet = ts; d.storageMode = MTLStorageModeShared; d.sampleCount = PG_SLOTS;
    for (int i = 0; i < PG_RING; i++) { NSError *er = nil; g_pg_buf[i] = [g_dev newCounterSampleBufferWithDescriptor:d error:&er]; if (!g_pg_buf[i]) { MLOG("【红】计数采样缓冲建不出：%s", er.localizedDescription.UTF8String); return 0; } }
    if (!(g_pg_f = fopen(e, "w"))) { MLOG("【红】DM_NVN_PASSGPU 打不开 %s", e); return 0; }
    fprintf(g_pg_f, "frame\tidx\tkind\tgpu_ms\tinc_ms\tw\th\tfmt0\tdfmt\tload\tdraws\tt0_ns\tt1_ns\tpass\n");
    MLOG("[passgpu] 逐通道 GPU 计时 → %s", e);
    return g_pg = 1;
}
static void pg_sub_begin(uint64_t fence) {   // This submission uses block fence % PG_RING of the ring; first wait for the submission that used it PG_RING submissions ago to complete
    if (!pg_on()) return;
    if (fence > PG_RING) mtl_wait(fence - PG_RING, UINT64_MAX);
    g_pg_b = (int)(fence % PG_RING); g_pg_used = 0; g_pg_np = 0; g_pg_open = -1;
}
static pg_pass *pg_new(int kind, int nslot) {
    if (!pg_on() || g_pg_used + nslot > PG_SLOTS) return NULL;
    uint64_t f = atomic_load(&g_cur_frame); if (f != g_pg_frame) { g_pg_frame = f; g_pg_idx = 0; }
    if (g_pg_np == g_pg_cap) { g_pg_cap = g_pg_cap ? g_pg_cap * 2 : 128; g_pg_pass = realloc(g_pg_pass, g_pg_cap * sizeof *g_pg_pass); }
    pg_pass *p = &g_pg_pass[g_pg_np++]; memset(p, 0, sizeof *p);
    p->frame = f; p->pass = g_stat_pass; p->idx = g_pg_idx++; p->kind = kind; p->slot = (int)g_pg_used; p->draws = -(int)g_stat_draw;
    g_pg_used += nslot;
    return p;
}
static void pg_render(MTL4RenderPassDescriptor *pd, NSUInteger nc) {
    (void)pd; (void)nc;
    return;
#if 0
    pg_pass *p = pg_new(0, 4); if (!p) return;
    MTLRenderPassSampleBufferAttachmentDescriptor *a = pd.sampleBufferAttachments[0];
    a.sampleBuffer = g_pg_buf[g_pg_b];
    a.startOfVertexSampleIndex = p->slot; a.endOfVertexSampleIndex = p->slot + 1; a.startOfFragmentSampleIndex = p->slot + 2; a.endOfFragmentSampleIndex = p->slot + 3;
    id<MTLTexture> t0 = nil; int k = 0;
    for (NSUInteger i = 0; i < nc && i < 8; i++) { id<MTLTexture> t = pd.colorAttachments[i].texture; if (!t) continue;
        if (!t0) t0 = t; if (k < 10) p->load[k++] = pd.colorAttachments[i].loadAction == MTLLoadActionClear ? 'C' : 'L'; }
    id<MTLTexture> dt = pd.depthAttachment.texture;
    if (dt) { if (k < 10) { p->load[k++] = '|'; p->load[k++] = pd.depthAttachment.loadAction == MTLLoadActionClear ? 'C' : 'L'; } p->dfmt = (uint32_t)dt.pixelFormat; }
    if (!t0) t0 = dt;
    NSUInteger pw, ph; pass_dims(pd, nc, &pw, &ph);
    p->w = (int)pw; p->h = (int)ph; p->cfmt0 = t0 && t0 != dt ? (uint32_t)t0.pixelFormat : 0;
    g_pg_open = g_pg_np - 1;
#endif
}
static id<MTLComputeCommandEncoder> pg_compute(void) {
    pg_pass *p = pg_new(1, 2);
    ser_gate();
    if (!p) { args_dirty_all(); return enc_c_begin(NULL); }
    MTLComputePassDescriptor *cd = [MTLComputePassDescriptor computePassDescriptor];
    cd.sampleBufferAttachments[0].sampleBuffer = g_pg_buf[g_pg_b];
    cd.sampleBufferAttachments[0].startOfEncoderSampleIndex = p->slot; cd.sampleBufferAttachments[0].endOfEncoderSampleIndex = p->slot + 1;
    p->draws = 0;
    args_dirty_all();
    return enc_c_begin(cd);
}
static void pg_end_render(void) { if (g_pg_open >= 0) { g_pg_pass[g_pg_open].draws += (int)g_stat_draw; g_pg_open = -1; } }
static void pg_commit(id<MTLCommandBuffer> cb) {   // Hand this submission's pass table to the completion callback for resolving
    if (!g_pg || !g_pg_np) return;
    int n = g_pg_np; pg_pass *ps = malloc(n * sizeof *ps); memcpy(ps, g_pg_pass, n * sizeof *ps);
    id<MTLCounterSampleBuffer> buf = g_pg_buf[g_pg_b]; uint32_t used = g_pg_used;
    [cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
        (void)c;
        NSData *d = [buf resolveCounterRange:NSMakeRange(0, used)];
        const MTLCounterResultTimestamp *ts = d.bytes;
        pthread_mutex_lock(&g_pg_mu);
        for (int i = 0; d && i < n; i++) {
            pg_pass *p = &ps[i]; uint64_t a, b;
            #define OK(x) ((x) != MTLCounterErrorValue && (x) != 0)
            if (p->kind == 0) { a = OK(ts[p->slot].timestamp) ? ts[p->slot].timestamp : ts[p->slot + 2].timestamp; b = OK(ts[p->slot + 3].timestamp) ? ts[p->slot + 3].timestamp : ts[p->slot + 1].timestamp; }
            else { a = ts[p->slot].timestamp; b = ts[p->slot + 1].timestamp; }
            if (!OK(a) || !OK(b) || b < a) continue;
            #undef OK
            static uint64_t prev_end;   // Latest end time of the passes completed so far (completion callbacks are serialized in submission order)
            double ms = (b - a) / 1e6;   // Timestamps are in ns, same as the CPU (measured 1:1 via sampleTimestamps on this machine)
            double inc = b > prev_end ? (b - (a > prev_end ? a : prev_end)) / 1e6 : 0;
            if (b > prev_end) prev_end = b;
            fprintf(g_pg_f, "%llu\t%d\t%c\t%.4f\t%.4f\t%d\t%d\t%u\t%u\t%s\t%d\t%llu\t%llu\t%llu\n", (unsigned long long)p->frame, p->idx, p->kind ? 'C' : 'R', ms, inc,
                    p->w, p->h, p->cfmt0, p->dfmt, p->load, p->draws, (unsigned long long)a, (unsigned long long)b, (unsigned long long)p->pass);
            if (p->idx < 256 && p->frame > 300) { pg_stat *st = &g_pg_st[p->idx]; st->sum += inc; st->n++; st->last = inc; st->meta = *p; }
            if (p->frame >= g_pg_rep_frame + 300 && p->frame > 300) {   // Report the top five every 300 frames (accumulated from frame 300)
                g_pg_rep_frame = p->frame; double tot = 0; int top[5] = {-1, -1, -1, -1, -1};
                for (int k = 0; k < 256; k++) if (g_pg_st[k].n) { double m = g_pg_st[k].sum / g_pg_st[k].n; tot += m;
                    for (int j = 0; j < 5; j++) if (top[j] < 0 || m > g_pg_st[top[j]].sum / g_pg_st[top[j]].n) { for (int q = 4; q > j; q--) top[q] = top[q - 1]; top[j] = k; break; } }
                char line[1024]; size_t o = 0;
                for (int j = 0; j < 5 && top[j] >= 0; j++) { pg_stat *t = &g_pg_st[top[j]];
                    o += snprintf(line + o, sizeof line - o, " #%d %.2f ms %dx%d fmt%u d%u %s %d 绘制；", top[j], t->sum / t->n, t->meta.w, t->meta.h, t->meta.cfmt0, t->meta.dfmt, t->meta.load, t->meta.draws); }
                MLOG("[passgpu] 至帧 %llu 每帧通道均值合计 %.2f ms；前五：%s", (unsigned long long)p->frame, tot, line);
                fflush(g_pg_f);
            }
        }
        pthread_mutex_unlock(&g_pg_mu);
        free(ps);
    }];
}
void ts_after(void);
void ms_after(void);
// ---------- Store lookahead: at pass end, decide from the later commands of the same submission whether each attachment's write-back is redundant ----------
// On TBDR every attachment is written back to memory in full when the pass ends; Switch (Maxwell immediate-mode rendering) has no such cost. If an attachment is cleared in full before being read, the write-back is redundant (DontCare).
// For the decision see mtl_sub_index / sdx_decide; only what can be proven within the same submission is accepted, and anything still undecided at the end of the submission is always written back.
typedef struct { uint64_t ser, bytes; uint32_t lv, sl; int idx; uint8_t depth, state, ridx, drawn, why, load, wrote, wwhy; const char *wkey; int wslot; } sd_att;   // depth: 0 color 1 depth 2 stencil (the two parts of a combined depth-stencil texture are decided separately)   // state 0 undecided 1 needed 2 redundant; idx: color attachment index, -1 for depth
static sd_att g_sd[9]; static int g_nsd;
static int g_udc_ab = -1, g_udc_ab_on;
// Attachments this pass did not write (depth / stencil test only, color write mask 0, etc.) get their write-back changed to DontCare: the content is already in memory, saving one full write-back.
// On by default: on Mac, same-process on / off screenshots follow the same distribution as the null control; on A19 Pro, private Depth32F_S8 / RGBA8 read back after DontCare differ in 0 places.
// DM_NVN_UNWRITTEN_DC=0 turns it off (for equivalence comparison)
static int udc_on(void) {
    if (g_udc_ab < 0) { const char *e = getenv("DM_NVN_UNWRITTEN_DC"); g_udc_ab = !(e && !strcmp(e, "0"));
        const char *a = getenv("DM_NVN_UDC_AB"); if (a && !strcmp(a, "1")) { g_udc_ab = 0; g_udc_ab_on = 0; } }
    return g_udc_ab || g_udc_ab_on;
}
// Diagnostic DM_NVN_TBDR=0: turns off all TBDR optimizations (store lookahead, load elision, same-target continuation, clear-only pass hoist, on-demand depth-stencil attach); replay takes the plainest 'attach all, load all, store all' path.
// Only for equivalence comparison (the same replay run on / off, comparing the picture byte by byte), not a product configuration
// Mario: off by default (only DM_NVN_TBDR=1 turns it on). These optimizations infer read / write relations from "which textures are bound in texture slots", while 38% of this game's shaders use bindless textures (handles in uniform buffers, NVN_RB_TEXU),
//   so the inference would miss readers - e.g. deciding an attachment is never read afterwards and not storing it. Turn them back on once they recognize bindless readers.
static int tbdr_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_NVN_TBDR"), *r = getenv("DM_NVN_RULER"); v = e && !strcmp(e, "1"); g_ruler_truth = r && !strcmp(r, "1"); if (!v) MLOG("TBDR 优化缺省关（本作有无绑定贴图；DM_NVN_TBDR=1 开）"); } return v && !g_ruler_truth; }
int dm_nvn_ruler_truth(void) { tbdr_on(); return g_ruler_truth; }   // Visual ruler ground-truth phase (host-side settings under test yield accordingly, e.g. the 3D render scale)
static int rt_range_on(void) { static int v = -1; if (v < 0) { const char *e=getenv("DM_NVN_RT_RANGE"); v=e && !strcmp(e,"1"); } return v; }   // Narrow candidate, off by default
static uint64_t g_sd_dead_bytes, g_sd_n_dead, g_sd_n_all;
// ---------- Cross-submission attachment validity ledger ----------
// Entry = subresource (texture serial, level, layer) + state, surviving across submissions:
//   XV_INVALID       content was not written back / not generated and must not be read. The next touch must be a full Clear, or use as an attachment with load action DontCare / Clear; otherwise it is a violation.
//   XV_PENDING_CLEAR reserved for 'clear-only pending across submissions' (not implemented): accepts only full, unscissored clears with the write mask fully open; clearing again while pending = replace the clear value without landing it;
//                    as an attachment with load DontCare / Clear → the pending clear is voided, with load Load → changed to a Clear load; depth-stencil is recorded per plane.
// Violation = report a failure + that stable key (format / width / height / level) permanently disabled (in memory + persisted to DM_XV_FILE, read at the next launch) + all entries of the same texture deleted. The fallback can only detect, not undo (in that frame the attachment holds old content).
// Its first use: color attachments 'scanned to the end undecided' at the end of a submission are not written back (sd_finish) - conditions: its first touch in this submission is a full clear / DontCare load (reset every frame before use), it is not a window texture, and the key is not disabled.
// Switch DM_NVN_XV=0 turns it off; has no effect when tbdr_on()==0 (visual ruler ground-truth mode).
// Trial period (XV_PROBE): for a key's first XV_TRIAL times it is only 'shadow-accounted' - written back as usual, but its next touch is tracked all the same; only after XV_TRIAL consecutive compliant touches does the write-back really start being skipped.
//   On a read / Load during the trial period: only disable that key (within this process, not persisted, no failure), the picture is intact - in practice there are readers as early as the launch / menu screens (upscale target 36), and not every fresh install should get one wrong frame.
enum { XV_INVALID = 1, XV_PENDING_CLEAR = 2, XV_PROBE = 3 };
#define XV_TRIAL 3
typedef struct { uint64_t ser, tag; uint32_t lv, sl; uint8_t state; } xv_ent;
static xv_ent g_xv[64]; static int g_xv_n, g_xv_block, g_xv_init;
static uint64_t g_xv_disabled[128]; static int g_xv_ndis;
static struct { uint64_t tag; int good; } g_xv_trial[128]; static int g_xv_ntrial;   // Number of consecutive compliant touches for each key
static int xv_trial_get(uint64_t tag, int add) { for (int i = 0; i < g_xv_ntrial; i++) if (g_xv_trial[i].tag == tag) return g_xv_trial[i].good += add; if (g_xv_ntrial < 128) { g_xv_trial[g_xv_ntrial].tag = tag; return g_xv_trial[g_xv_ntrial++].good = add; } return 0; }
static uint64_t g_xv_present[128]; static int g_xv_npresent;
static uint64_t g_xv_dc, g_xv_viol;
typedef struct { uint64_t ser; uint32_t lv, sl; uint8_t good; } xv_touch;
static xv_touch g_xv_touch[128]; static int g_xv_ntouch;
static uint64_t xv_tag(const tex_t *t, uint32_t lv) { return ((uint64_t)t->d.format << 40) ^ ((uint64_t)t->d.width << 24) ^ ((uint64_t)t->d.height << 8) ^ lv; }
// Stable key (unchanged across launches, does not use the texture serial): format / width / height / level, then mixed with the formats of the color attachments of the pass where the write-back is skipped and this attachment's index - textures of the same format and size but different purpose (e.g. transmittance 27 and upscale target 36) do not share a key
static uint64_t xv_tag_pass(const tex_t *t, uint32_t lv, int idx, const p_set_rt *rt) { uint64_t h = xv_tag(t, lv) ^ ((uint64_t)(idx + 1) << 60); for (int i = 0; i < rt->n && i < 8; i++) h = (h ^ (rt->color[i] ? (uint64_t)rt->color[i]->d.format + 1 : 0)) * 0x100000001b3ull; return h; }
static void xv_init(void) {
    if (g_xv_init) return; g_xv_init = 1;
    const char *p = getenv("DM_XV_FILE"); if (!p || !*p) return;
    FILE *f = fopen(p, "r"); if (!f) return;
    char line[256], why[160]; unsigned long long tag; int ch, n = 0;
    while (fgets(line, sizeof line, f) && g_xv_ndis < 128) if (sscanf(line, "%llx\t%159[^\t]\t%d", &tag, why, &ch) >= 1) g_xv_disabled[g_xv_ndis++] = tag, n++;
    fclose(f); MLOG("读到 %d 条禁用记录", n);
}
static int xv_disabled(uint64_t tag) { xv_init(); for (int i = 0; i < g_xv_ndis; i++) if (g_xv_disabled[i] == tag) return 1; return 0; }
static xv_ent *xv_find(uint64_t ser, uint32_t lv, uint32_t sl) { for (int i = 0; i < g_xv_n; i++) if (g_xv[i].ser == ser && g_xv[i].lv == lv && g_xv[i].sl == sl) return &g_xv[i]; return NULL; }
static void xv_drop(uint64_t ser, uint32_t lv, uint32_t sl) { for (int i = 0; i < g_xv_n; i++) if (g_xv[i].ser == ser && g_xv[i].lv == lv && g_xv[i].sl == sl) { g_xv[i] = g_xv[--g_xv_n]; return; } }
static void xv_drop_all_of(uint64_t ser) { for (int i = 0; i < g_xv_n;) if (g_xv[i].ser == ser) g_xv[i] = g_xv[--g_xv_n]; else i++; }
static void xv_set(uint64_t ser, uint32_t lv, uint32_t sl, uint8_t state, uint64_t tag) {
    xv_init(); if (g_xv_block || xv_disabled(tag)) return;
    xv_ent *e = xv_find(ser, lv, sl); if (e) { e->state = state; e->tag = tag; return; }
    if (g_xv_n < 64) g_xv[g_xv_n++] = (xv_ent){ser, tag, lv, sl, state};
}
static void xv_violation(const xv_ent *e, const char *why) {
    xv_init();
    if (e->state == XV_PROBE) {   // Trial period: the content is still there and the picture is intact - only disable the key (this process)
        if (!xv_disabled(e->tag) && g_xv_ndis < 128) g_xv_disabled[g_xv_ndis++] = e->tag;
        PD("V xv 试用期见读者：键=%016llx 贴图=%llu 级=%u 原因=%s → 该键不做", (unsigned long long)e->tag, (unsigned long long)e->ser, e->lv, why);
        xv_drop_all_of(e->ser); return; }
    g_xv_viol++;
    int seen = xv_disabled(e->tag); if (!seen && g_xv_ndis < 128) g_xv_disabled[g_xv_ndis++] = e->tag;
    MLOG("【红】xv 标签=%016llx 贴图=%llu 级=%u 层=%u 原因=%s 通道=%llu", (unsigned long long)e->tag, (unsigned long long)e->ser, e->lv, e->sl, why, (unsigned long long)g_stat_pass);
    const char *p = getenv("DM_XV_FILE"); if (p && *p) { FILE *f = fopen(p, "a"); if (f) { fprintf(f, "%016llx\t%s\t%llu\n", (unsigned long long)e->tag, why, (unsigned long long)g_stat_pass); fclose(f); } }
    xv_drop_all_of(e->ser);
}
static void xv_check(uint64_t ser, uint32_t lv, uint32_t sl, const char *why) { xv_ent *e = xv_find(ser, lv, sl); if (e && (e->state == XV_INVALID || e->state == XV_PROBE)) { xv_ent copy = *e; xv_violation(&copy, why); } }
static void xv_good(xv_ent *e) { if (e->state == XV_PROBE) xv_trial_get(e->tag, 1); xv_drop(e->ser, e->lv, e->sl); }   // Compliant touch: counts once toward the trial period, and the entry returns to valid
static void xv_touch_add(uint64_t ser, uint32_t lv, uint32_t sl, int good) { for (int i = 0; i < g_xv_ntouch; i++) if (g_xv_touch[i].ser == ser && g_xv_touch[i].lv == lv && g_xv_touch[i].sl == sl) return; if (g_xv_ntouch < 128) g_xv_touch[g_xv_ntouch++] = (xv_touch){ser, lv, sl, (uint8_t)good}; }
static int xv_first_good(uint64_t ser, uint32_t lv, uint32_t sl) { for (int i = 0; i < g_xv_ntouch; i++) if (g_xv_touch[i].ser == ser && g_xv_touch[i].lv == lv && g_xv_touch[i].sl == sl) return g_xv_touch[i].good; return 0; }
static int xv_present(uint64_t ser) { for (int i = 0; i < g_xv_npresent; i++) if (g_xv_present[i] == ser) return 1; return 0; }
static void xv_forward(void);
static int tbdr_on(void);
static int xv_on(void) { static int off = -1; if (off < 0) { const char *e = getenv("DM_NVN_XV"); off = e && !strcmp(e, "0"); } return !off && tbdr_on(); }
void mtl_xv_window_tex(uint64_t serial) { if (!xv_present(serial) && g_xv_npresent < 128) g_xv_present[g_xv_npresent++] = serial; }   // Window (swapchain) texture: never enters the ledger
void xv_test_counts(int *dc, int *viol) { if (dc) *dc = (int)g_xv_dc; if (viol) *viol = (int)g_xv_viol; }
static void sd_finish(void);
// Load elision (fully covering first draw): if the draw that opens the pass can be proven to cover a color attachment in full (first_draw_covers), that attachment's load becomes DontCare -
// the first draw does not blend and writes all channels, every covered pixel is rewritten by it, and the original content is not visible (Switch immediate-mode rendering has no such cost; on TBDR load = reading the whole thing from memory into the tile)
static const p_draw *g_fd_dw; static const p_draw_idx *g_fd_di;   // The draw that opens the pass (one of non-indexed / indexed); non-null only while draw() is opening a pass
static uint64_t g_fd_n;         // Cumulative number of attachments judged elidable
static int g_fd_why;            // Reason for the most recent failure (Q line of PASSDUMP: 1 no program 2 shader 3 attachment 4 blend/mask/raster 5 depth-stencil test 6 vertex count 7 position attribute 8 format/stream 9 out of bounds 10 address 11 geometry)
static int first_draw_covers(int i, id<MTLTexture> t, NSUInteger level);
static int pf_is_float(MTLPixelFormat f) {   // Float color formats that can store NaN (sentinel choice for the load-elision probe; RGB9E5 cannot store NaN and does not count)
    switch (f) { case MTLPixelFormatR16Float: case MTLPixelFormatRG16Float: case MTLPixelFormatRGBA16Float: case MTLPixelFormatR32Float:
        case MTLPixelFormatRG32Float: case MTLPixelFormatRGBA32Float: case MTLPixelFormatRG11B10Float: return 1; default: return 0; }
}
static int fz_load_covers(int i, id<MTLTexture> t, NSUInteger level);
static NSMutableArray *g_lp;    // DM_NVN_LOADPROBE=1 / 2 / 3 (two constant sentinel values / NaN - under float formats normal shader output cannot reach NaN, ruling out numeric coincidence): attachments judged elidable are cleared to the sentinel color instead, and at pass end the remaining sentinel pixels are read back and counted (diagnostic: should be 0)
static void lp_finish(void);
// Same-target continuation: when the game calls SetRenderTargets again on the same set of targets (with no copy / dispatch / clear in between) the pass is not ended - on TBDR this saves one full write-back + load.
// Basis for equivalence: NVN's Barrier between two passes requires the latter to see all writes of the former. Within one Metal render encoder, later draws' reads and writes of the attachment itself (blend, depth / stencil test)
// are ordered with earlier draws in submission order; but 'sampling' the attachment texture is not covered (reading and writing the same attachment within one pass is undefined). So continuation happens only when later draws do not sample these attachments:
// if during continuation a draw needs to sample them, or a clear arrives, the pass is ended before it, returning to the separate semantics. Copies / dispatches end the render encoder anyway
static int g_enc_mode;          // How the current encoder was opened (rp_begin_ex argument: 0 all attachments 1 only for clears 2 depth omitted)
static int g_rt_range_active, g_rt_range_pending;
static int g_rt_merged;         // The current pass has already been continued (at least one same-target SetRenderTargets was merged in)
static uint64_t g_rt_merge_n;   // Cumulative continuation count
static int g_defer_replaying;
static void fz_end(void);   // Wrap-up of the readable-tile pass merge (defined before draw)
static void defer_fallback(void);
// DM_NVN_ENC_LABEL=1: label every render encoder with 'p<index of the render pass within the frame> <width>x<height> <first 8 digits of the first draw's fragment key> d<draw count>'; in Instruments (xctrace Game Performance / Metal System Trace)
// the encoder track and the GPU interval table carry this name directly, and shader_map.tsv maps it back to the shader; off by default (one string allocation per pass)
static int g_el_w, g_el_h, g_el_draws; static char g_el_key[12];
static int enc_label_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_NVN_ENC_LABEL"); v = e && atoi(e); } return v; }
static void ghost_flush(void);
static void rp_end(void) { if (g_enc) { AP_B;
    ghost_flush();
    if (enc_label_on()) { static uint64_t lf = ~0ull, base; uint64_t f = atomic_load(&g_cur_frame); if (f != lf) { lf = f; base = g_stat_pass - 1; }   // Intra-frame serial: the same pass has the same name every frame, so the table can aggregate by name
        [(id)g_enc setLabel:[NSString stringWithFormat:@"p%02llu %dx%d %s d%d", (unsigned long long)(g_stat_pass - base), g_el_w, g_el_h, g_el_key[0] ? g_el_key : "-", g_el_draws]]; }
#ifdef NVN_RT_RANGE_TEST
    int ranged=g_rt_range_active && !g_rt_range_pending;
#endif
    sd_finish(); enc_r_end(g_enc); zskip_commit_clears(); g_enc = nil; g_rt_merged = 0; g_rt_range_active = g_rt_range_pending = 0; pg_end_render(); ts_after(); ms_after(); if (g_lp.count) lp_finish(); fz_end();
    if (g_nubd) ubd_flush();   // the inline updates deferred in this pass, in order (see mtl_sub_update)
#ifdef NVN_RT_RANGE_TEST
    rr_finish(ranged);
#endif
    AP_E(0);
} }
void pg_sub_begin_ext(uint64_t fence) { pg_sub_begin(fence); }
void pg_commit_ext(id<MTLCommandBuffer> cb) { pg_commit(cb); }
// The format an attachment is actually attached with: the view's format if the view changes the format and can be attached, otherwise the base texture's. Shared by attaching (attach) and pipeline prediction (pso_predict_rest) so both agree --
// prediction used the base format, so for a pass written through an sRGB view (Mario's last pass) the predicted pipeline had the wrong format and the real one was compiled at the first draw (from d2r-mobile e2a0c9a9b)
// A compressed texture as a render target (The Legend of Zelda: Breath of the Wild: a 32x32 RGTC1 (BC4) texture written through an RGBA16UI view, one texel = one 8-byte block, i.e. BC4 encoded on the GPU):
//   Metal can neither attach a compressed format as a colour attachment nor create compressed <-> uncompressed views (the pipeline descriptor validation aborts the process). For now the attachment is left out (a pass with no attachment at all skips its draws), with a red line;
//   drawing it correctly needs an uncompressed proxy texture with the same block size, copied into the compressed texture through a buffer after the pass
static int mtl_fmt_block(MTLPixelFormat f) { return (f >= MTLPixelFormatBC1_RGBA && f <= MTLPixelFormatBC7_RGBAUnorm_sRGB) || (f >= MTLPixelFormatPVRTC_RGB_2BPP && f < MTLPixelFormatGBGR422); }
static MTLPixelFormat rt_view_fmt(id<MTLTexture> base, const view_t *v, int has_v) {
    if (base && mtl_fmt_block(base.pixelFormat)) {
        RED_ONCE(78, "渲染目标是压缩格式贴图（Metal 格式 %lu %lux%lu 层 %lu，视图换格式 %d → NVN %#x）：不挂，该附件的绘制落空", (unsigned long)base.pixelFormat, (unsigned long)base.width, (unsigned long)base.height,
                 (unsigned long)base.arrayLength, has_v ? v->set_format : 0, has_v ? v->format : 0);
        return MTLPixelFormatInvalid;
    }
    if (!base || !has_v || !v->set_format) return base.pixelFormat;
    uint32_t pf = 0; uint8_t swz[4];
    if (!nvn_fmt_to_mtl(v->format, &pf, swz) || !pf || pf == (uint32_t)base.pixelFormat) return base.pixelFormat;
    if (tex_need_pfv(base.pixelFormat, (MTLPixelFormat)pf) && !(base.usage & MTLTextureUsagePixelFormatView)) return base.pixelFormat;
    return (MTLPixelFormat)pf;
}
// The same texture level / layer attached at more than one colour slot: only the first is attached, later ones are not (their writes are dropped). The G-buffer passes of The Legend of Zelda: Breath of the Wild attach the albedo and another buffer twice each
// (c1 = c2, c3 = c4), and the channel write masks of c2 / c4 are always 0 (cmask 0x00f0f0ff). The Switch's ROP writes only the masked channels and the two slots do not interfere;
// on Apple's TBDR each attachment has its own tile memory and the whole tile is stored back at the end of the pass, so the copy stored later (loaded, never written) overwrites the writes of the earlier one -- the order differs per 32x32 tile,
// the albedo kept only scattered squares, and the whole 3D picture went black. A red line is printed when a dropped slot is used with a non-zero write mask (those writes are really lost)
static int rt_dup_slot(const p_set_rt *r, int i) {
    if (!r->color[i]) return 0;
    uint8_t lv = r->has_cview[i] && r->cview[i].set_levels ? r->cview[i].base_level : 0; uint16_t ly = r->has_cview[i] && r->cview[i].set_layers ? r->cview[i].min_layer : 0;
    for (int j = 0; j < i; j++) {
        if (!r->color[j] || r->color[j]->serial != r->color[i]->serial) continue;   // compare serials: records hold copies of the texture objects (tex_snap), so the pointers differ
        uint8_t lj = r->has_cview[j] && r->cview[j].set_levels ? r->cview[j].base_level : 0; uint16_t yj = r->has_cview[j] && r->cview[j].set_layers ? r->cview[j].min_layer : 0;
        if (lj == lv && yj == ly) return 1;
    }
    return 0;
}
static void attach(MTLRenderPassAttachmentDescriptor *a, const tex_t *t, const view_t *v, int has_v) {
    if (t) rc_write_use(t->serial);   // T5: an attached texture can change after its upload
    id<MTLTexture> base = tex_mtl(t); a.texture = base;
    if (base && mtl_fmt_block(base.pixelFormat)) { rt_view_fmt(base, v, has_v); a.texture = nil; return; }   // compressed formats cannot be attached (see rt_view_fmt)
    if (has_v && v->set_levels) a.level = v->base_level;
    if (has_v && v->set_layers) { if (t->d.target == 2) a.depthPlane = v->min_layer; else a.slice = v->min_layer; }
    // A view that changes the format is attached as a Metal view of that format (pipeline formats follow the attachment). SMBW's last pass writes the RGBA8 swapchain texture through an RGBA8_SRGB view, so the hardware encodes sRGB on write;
    // attaching the base texture instead stored the linear values as they were and the whole picture came out dark (every value decoded from sRGB once more, on the Mac and the iPhone). The view lives until this submission completes
    if (base && has_v && v->set_format) {
        MTLPixelFormat pf = rt_view_fmt(base, v, has_v);
        if (pf != base.pixelFormat) {
            id<MTLTexture> vw = [base newTextureViewWithPixelFormat:pf];
            if (vw) { a.texture = vw; defer_obj_unref(vw); }
            else RED_ONCE(67, "渲染目标视图建不出：格式 %#x（底图 Metal 格式 %lu）", v->format, (unsigned long)base.pixelFormat);
        } else { uint32_t want = 0; uint8_t swz[4];
            if (nvn_fmt_to_mtl(v->format, &want, swz) && want && want != (uint32_t)base.pixelFormat)
                RED_ONCE(68, "渲染目标视图格式 %#x 挂不上（底图 Metal 格式 %lu 建时没有换格式用途）：按底图格式画", v->format, (unsigned long)base.pixelFormat); }
    }
}
// mode 1 (clear_only): a pass opened only to land pending clears, attaching only the attachments that have clears; mode 2: the lookahead proves this pass does not use depth-stencil, so depth is not attached.
// clear_only: a pass opened only to land pending clears, attaching only the attachments that have clears - the other attachments are not touched by this pass, and attaching them costs a pointless load and store on TBDR
// (Mac character select screen: the color-clear pass drags along a 400x225 depth load and store)
static id<MTL4RenderCommandEncoder> rp_begin_ex(int clear_only);
static id<MTL4RenderCommandEncoder> rp_begin(void) { return rp_begin_ex(0); }
static id<MTL4RenderCommandEncoder> rp_begin_ex_(int clear_only);
static id<MTL4RenderCommandEncoder> rp_begin_ex(int clear_only) { AP_B; id<MTL4RenderCommandEncoder> e = rp_begin_ex_(clear_only); AP_E(1); return e; }
static id<MTL4RenderCommandEncoder> rp_begin_ex_(int clear_only) {
    if (g_enc) return g_enc;
    rc_reclaim_maybe();   // T5: reclaim between passes of a long transition frame (cheap gate; no-op unless an upload batch is waiting)
    memset(g_zskip_cand_ok, 0, sizeof g_zskip_cand_ok);
    g_enc_mode = clear_only;
    if (!R.has_rt) { RED_ONCE(3, "绘制前没有设置渲染目标"); return nil; }
    // The version of the attachment-range candidate is taken before capturing any attachment; if it changes during setup, there is no continuation.
    uint64_t reg0 = 0, ser0 = 0, dirty0 = 0;
    if (rt_range_on()) { reg0 = g_reg_gen; ser0 = g_ser_gen; }
    if (rt_range_on()) dirty0=nvn_dirty_now();
    if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
    MTL4RenderPassDescriptor *pd = [MTL4RenderPassDescriptor new];
    int any = 0;
    for (int i = 0; i < R.rt.n; i++) {
        if (!R.rt.color[i] || (clear_only == 1 && !R.clear_c[i])) continue;
        if (rt_dup_slot(&R.rt, i)) continue;   // a slot attached twice is not attached (see rt_dup_slot)
        attach(pd.colorAttachments[i], R.rt.color[i], &R.rt.cview[i], R.rt.has_cview[i]);
        if (!pd.colorAttachments[i].texture) continue;
        any = 1;
        pd.colorAttachments[i].loadAction = R.clear_c[i] ? MTLLoadActionClear : MTLLoadActionLoad;
        pd.colorAttachments[i].clearColor = R.cc[i];
        if (R.clear_c[i]) PD("C %llu c%d 清成 %g,%g,%g,%g", g_stat_pass + 1, i, R.cc[i].red, R.cc[i].green, R.cc[i].blue, R.cc[i].alpha);
        int cov = tbdr_on() && !R.clear_c[i] && (g_fd_dw || g_fd_di) && fz_load_covers(i, pd.colorAttachments[i].texture, pd.colorAttachments[i].level);
        if (!R.clear_c[i] && (g_fd_dw || g_fd_di) && !cov) PD("Q %llu c%d 读入不可省 why=%d", g_stat_pass + 1, i, g_fd_why);
        if (cov) {
            static int probe = -1; if (probe < 0) { probe = getenv("DM_NVN_LOADPROBE") ? atoi(getenv("DM_NVN_LOADPROBE")) : 0; if (probe) MLOG("DM_NVN_LOADPROBE：读入可省的附件改清成哨兵色并回读核对"); }
            if (probe) { pd.colorAttachments[i].loadAction = MTLLoadActionClear; int pm = probe == 1 && pf_is_float(pd.colorAttachments[i].texture.pixelFormat) ? 3 : probe;   // Float formats use a NaN sentinel by default: a fixed sentinel value can be hit by real output (it has been hit in practice)
                pd.colorAttachments[i].clearColor = pm == 3 ? MTLClearColorMake(NAN, NAN, NAN, NAN) : probe == 2 ? MTLClearColorMake(0.777, 0.222, 0.555, 0.888) : MTLClearColorMake(0.123, 0.456, 0.789, 0.321);
                if (!g_lp) g_lp = [NSMutableArray array];
                [g_lp addObject:@[pd.colorAttachments[i].texture, @(pd.colorAttachments[i].level), @(pd.colorAttachments[i].slice), @(g_stat_pass + 1), @(i), @(R.rt.color[i]->serial), @(pm)]]; }
            else pd.colorAttachments[i].loadAction = MTLLoadActionDontCare;
            g_fd_n++;
        }
        xv_touch_add(R.rt.color[i]->serial, (uint32_t)pd.colorAttachments[i].level, (uint32_t)pd.colorAttachments[i].slice,
                     pd.colorAttachments[i].loadAction != MTLLoadActionLoad);
        if (g_xv_n) { xv_ent *xe = xv_find(R.rt.color[i]->serial, (uint32_t)pd.colorAttachments[i].level, (uint32_t)pd.colorAttachments[i].slice);
            if (xe && (xe->state == XV_INVALID || xe->state == XV_PROBE)) { if (pd.colorAttachments[i].loadAction == MTLLoadActionLoad) { xv_ent copy = *xe; xv_violation(&copy, "作附件读入旧内容"); } else xv_good(xe); } }
        pd.colorAttachments[i].storeAction = MTLStoreActionStore;
    }
    // clear_only: 0 attach all 1 only for clears 2 neither depth nor stencil attached 3 stencil only 4 depth only (3 / 4: the pass_ds_need lookahead proves the whole pass uses only that one)
    if (R.rt.depth && clear_only == 3) {   // Stencil only: the depth plane of the combined format does not enter the tile and is neither read nor written (Metal allows an empty depth attachment with the combined texture on the stencil attachment)
        attach(pd.stencilAttachment, R.rt.depth, &R.rt.dview, R.rt.has_dview);
        MTLPixelFormat pf = pd.stencilAttachment.texture.pixelFormat;
        if (pf == MTLPixelFormatDepth32Float_Stencil8 || pf == MTLPixelFormatStencil8) {
            any = 1; pd.stencilAttachment.loadAction = R.clear_s ? MTLLoadActionClear : MTLLoadActionLoad;
            pd.stencilAttachment.clearStencil = R.cs; pd.stencilAttachment.storeAction = MTLStoreActionStore;
        } else pd.stencilAttachment.texture = nil;   // No stencil plane: the stencil test always passes, attach nothing
    } else if (R.rt.depth && clear_only != 2 && !(clear_only == 1 && !R.clear_d && !R.clear_s)) {
        attach(pd.depthAttachment, R.rt.depth, &R.rt.dview, R.rt.has_dview);
        if (pd.depthAttachment.texture) {
            any = 1;
            pd.depthAttachment.loadAction = R.clear_d ? MTLLoadActionClear : MTLLoadActionLoad;
            pd.depthAttachment.clearDepth = R.cd; pd.depthAttachment.storeAction = MTLStoreActionStore;
            MTLPixelFormat pf = pd.depthAttachment.texture.pixelFormat;
            if (clear_only != 4 && (pf == MTLPixelFormatDepth32Float_Stencil8 || pf == MTLPixelFormatStencil8)) {   // 4: depth only
                pd.stencilAttachment.texture = pd.depthAttachment.texture; pd.stencilAttachment.level = pd.depthAttachment.level;
                pd.stencilAttachment.slice = pd.depthAttachment.slice;
                pd.stencilAttachment.loadAction = R.clear_s ? MTLLoadActionClear : MTLLoadActionLoad;
                pd.stencilAttachment.clearStencil = R.cs; pd.stencilAttachment.storeAction = MTLStoreActionStore;
            }
        }
    }
    if (!any) {
        RED_ONCE(4, "渲染目标都建不出 MTLTexture");
        if (passdump()) {   // PASSDUMP: when a whole pass is skipped, list its targets (serial, format, compressed or not)
            char b[256]; size_t o = 0;
            for (int i = 0; i < R.rt.n && i < 8; i++) if (R.rt.color[i]) o += snprintf(b + o, sizeof b - o, " c%d=%llu/f%#x%s", i, (unsigned long long)R.rt.color[i]->serial, R.rt.color[i]->d.format, fmt_compressed(R.rt.color[i]->d.format) ? "(压缩)" : "");
            if (R.rt.depth) o += snprintf(b + o, sizeof b - o, " d=%llu", (unsigned long long)R.rt.depth->serial);
            PD("N %llu 通道跳过（附件都挂不上）mode=%d%s", g_stat_pass + 1, clear_only, b);
        }
        return nil;
    }
    for (int i = 0; i < R.rt.n && i < 8; i++) if (R.rt.color[i] && zskip_zero_clear(i)) {
        g_zskip_cand_ok[i] = 1; g_zskip_cand[i] = (zskip_zero){R.rt.color[i]->serial, (uint16_t)(R.rt.has_cview[i] && R.rt.cview[i].set_levels ? R.rt.cview[i].base_level : 0), (uint16_t)(R.rt.has_cview[i] && R.rt.cview[i].set_layers ? R.rt.cview[i].min_layer : 0)};
    }
    // Bandwidth ledger: attachments with Load count as load, all attachments count as store (currently always Store)
    int nload = 0, natt = 0;
    for (int i = 0; i < R.rt.n; i++) {
        MTLRenderPassColorAttachmentDescriptor *c = pd.colorAttachments[i];
        if (!c.texture) continue;
        uint64_t b = att_bytes(c.texture, c.level); natt++; g_bw_store += b; if (c.loadAction == MTLLoadActionLoad) { g_bw_load += b; nload++; }
        if (R.rt.has_cview[i] && R.rt.cview[i].set_format) PD("V %llu c%d 视图格式 %#x（基贴图 NVN 格式 %#x）", g_stat_pass + 1, i, R.rt.cview[i].format, R.rt.color[i]->d.format);
        PD("P %llu c%d ser=%llu fmt=%lu %lux%lu ms%lu lv%lu sl%lu %s mtl=%p", g_stat_pass + 1, i, R.rt.color[i]->serial, (unsigned long)c.texture.pixelFormat,
           (unsigned long)c.texture.width, (unsigned long)c.texture.height, (unsigned long)c.texture.sampleCount, (unsigned long)c.level, (unsigned long)c.slice,
           c.loadAction == MTLLoadActionClear ? "clear" : c.loadAction == MTLLoadActionDontCare ? "dontcare" : "load", (__bridge void *)(c.texture.parentTexture ? c.texture.parentTexture : c.texture));
    }
    {   // Depth and stencil planes are counted separately (combined format 5 bytes = depth 4 + stencil 1; when only one is attached only that one is counted)
        id<MTLTexture> dt = pd.depthAttachment.texture, stt = pd.stencilAttachment.texture;
        uint64_t whole = att_bytes(dt ? dt : stt, dt ? pd.depthAttachment.level : pd.stencilAttachment.level);
        int comb = (dt ? dt : stt).pixelFormat == MTLPixelFormatDepth32Float_Stencil8;
        uint64_t db = dt ? (comb ? whole / 5 * 4 : whole) : 0, sb = stt && stt != dt ? (comb ? whole / 5 : whole) : stt ? whole - db : 0;
        if (dt) { natt++; g_bw_store += db; if (pd.depthAttachment.loadAction == MTLLoadActionLoad) { g_bw_load += db; nload++; } }
        if (stt) { natt += !dt; g_bw_store += sb; if (pd.stencilAttachment.loadAction == MTLLoadActionLoad) { g_bw_load += sb; nload += !dt; } }
        if (dt) PD("P %llu d ser=%llu fmt=%lu %lux%lu %s", g_stat_pass + 1, R.rt.depth->serial, (unsigned long)dt.pixelFormat,
                   (unsigned long)dt.width, (unsigned long)dt.height, pd.depthAttachment.loadAction == MTLLoadActionClear ? "clear" : "load");
        else if (stt) PD("P %llu s ser=%llu fmt=%lu %lux%lu %s（只挂模板）", g_stat_pass + 1, R.rt.depth->serial, (unsigned long)stt.pixelFormat,
                   (unsigned long)stt.width, (unsigned long)stt.height, pd.stencilAttachment.loadAction == MTLLoadActionClear ? "clear" : "load");
    }
    if (natt && nload == natt) g_pass_full_rw++;
    g_nsd = 0;
    {
        for (int i = 0; i < R.rt.n && i < 8; i++) { MTLRenderPassColorAttachmentDescriptor *c = pd.colorAttachments[i]; if (!c.texture) continue;
            g_sd[g_nsd] = (sd_att){0}; g_sd[g_nsd].ser = R.rt.color[i]->serial; g_sd[g_nsd].bytes = att_bytes(c.texture, c.level); g_sd[g_nsd].lv = (uint32_t)c.level; g_sd[g_nsd].sl = (uint32_t)(c.texture.textureType == MTLTextureType3D ? c.depthPlane : c.slice); g_sd[g_nsd].idx = i; g_sd[g_nsd].load = c.loadAction == MTLLoadActionLoad; g_sd[g_nsd].wrote = R.clear_c[i] != 0; g_nsd++;
            c.storeAction = MTLStoreActionUnknown; }
        id<MTLTexture> dt = pd.depthAttachment.texture, stt = pd.stencilAttachment.texture;
        uint64_t db = att_bytes(dt, pd.depthAttachment.level), sb = 0;
        if (stt) { int comb = stt.pixelFormat == MTLPixelFormatDepth32Float_Stencil8; sb = att_bytes(stt, pd.stencilAttachment.level) / (comb ? 5 : 1); if (stt == dt) db -= sb; }   // Combined format 5 bytes = depth 4 + stencil 1 (when only stencil is attached, only 1 is counted)
        if (dt) { g_sd[g_nsd] = (sd_att){0}; g_sd[g_nsd].ser = R.rt.depth->serial; g_sd[g_nsd].bytes = db; g_sd[g_nsd].lv = (uint32_t)pd.depthAttachment.level; g_sd[g_nsd].sl = (uint32_t)pd.depthAttachment.slice; g_sd[g_nsd].idx = -1; g_sd[g_nsd].depth = 1; g_sd[g_nsd].load = pd.depthAttachment.loadAction == MTLLoadActionLoad; g_sd[g_nsd].wrote = R.clear_d != 0; g_nsd++;
                  pd.depthAttachment.storeAction = MTLStoreActionUnknown; }
        if (stt) { g_sd[g_nsd] = (sd_att){0}; g_sd[g_nsd].ser = R.rt.depth->serial; g_sd[g_nsd].bytes = sb; g_sd[g_nsd].lv = (uint32_t)pd.stencilAttachment.level; g_sd[g_nsd].sl = (uint32_t)pd.stencilAttachment.slice; g_sd[g_nsd].idx = -1; g_sd[g_nsd].depth = 2; g_sd[g_nsd].load = pd.stencilAttachment.loadAction == MTLLoadActionLoad; g_sd[g_nsd].wrote = R.clear_s != 0; g_nsd++;
                   pd.stencilAttachment.storeAction = MTLStoreActionUnknown; }
    }
    memset(R.clear_c, 0, sizeof R.clear_c); R.clear_d = R.clear_s = 0;
    {   // Pass info (same convention as the former per-draw fetch: colors take each attachment's base texture format; sample count takes the last color attachment, or depth if there is depth)
        uint64_t ser = g_pass.serial + 1; memset(&g_pass, 0, sizeof g_pass); g_pass.serial = ser;
        for (int i = 0; i < R.rt.n && i < 8; i++) { id<MTLTexture> t = R.rt.color[i] ? pd.colorAttachments[i].texture : nil;
            if (t) { g_pass.cfmt[i] = (uint32_t)t.pixelFormat; g_pass.samples = (uint32_t)t.sampleCount; } }
        id<MTLTexture> dt = R.rt.depth ? pd.depthAttachment.texture : nil, stt = R.rt.depth ? pd.stencilAttachment.texture : nil;   // Depth / stencil attachments are fetched separately (possibly only one of them is attached)
        g_pass.hd = dt && dt.pixelFormat != MTLPixelFormatStencil8; g_pass.hs = stt != nil;
        g_pass.dfmt = g_pass.hd ? (uint32_t)dt.pixelFormat : 0; g_pass.sfmt = stt ? (uint32_t)stt.pixelFormat : 0;
        if (dt || stt) g_pass.samples = (uint32_t)(dt ? dt : stt).sampleCount;
        pass_dims(pd, (NSUInteger)R.rt.n, &g_pass.tw, &g_pass.th);
        if (rt_range_on()) { g_pass.reg_gen = reg0; g_pass.ser_gen = ser0; }
        if (rt_range_on()) { g_pass.range_dirty=dirty0; g_pass.range_color=pd.colorAttachments[0].texture; g_pass.range_depth=pd.depthAttachment.texture; }
        g_enc_pso = nil; g_enc_dss = nil;
    }
    if (g_vis_on) {
        // Slots used up (one submission exceeds VIS_SLOTS counting passes): continue in a new overflow buffer. A ring buffer must not be retired (it is still used in rotation, and retiring removes it from the residency set); only a previous overflow buffer is
        if (!g_vis || g_vis_n >= VIS_SLOTS) { if (g_vis && !vis_in_ring(g_vis)) defer_obj_release(g_vis); g_vis = [g_dev newBufferWithLength:VIS_SLOTS * 8 options:MTLResourceStorageModeShared]; res_add(g_vis); g_vis_n = 0; }
        pd.visibilityResultBuffer = g_vis;
    }
    g_enc_vis = 0; g_enc_has_vis = g_vis_on;
    pg_render(pd, (NSUInteger)R.rt.n);
    { void td_note(id pd, NSUInteger nc); td_note(pd, (NSUInteger)R.rt.n);
      void ts_note(id pd, NSUInteger nc); ts_note(pd, (NSUInteger)R.rt.n);
      void ms_note(id pd, NSUInteger nc); ms_note(pd, (NSUInteger)R.rt.n); }
    ser_gate();
    g_enc = enc_r_begin(pd);
    args_dirty_all();
    g_stat_pass++;
    if (enc_label_on()) { id<MTLTexture> lt = pd.colorAttachments[0].texture ?: pd.depthAttachment.texture ?: pd.stencilAttachment.texture; g_el_w = (int)lt.width; g_el_h = (int)lt.height; g_el_draws = 0; g_el_key[0] = 0; }
    return g_enc;
}
// Empty-scissor precheck before opening a pass: the size of the selected level of the first valid target is an upper bound on the minimum bounds of the actual pass.
static void rt_dims(NSUInteger *tw, NSUInteger *th) {
    id<MTLTexture> t0 = nil; NSUInteger level = 0;
    for (int i = 0; i < R.rt.n && i < 8 && !t0; i++) if (R.rt.color[i]) {
        t0 = tex_mtl(R.rt.color[i]); level = R.rt.has_cview[i] && R.rt.cview[i].set_levels ? R.rt.cview[i].base_level : 0;
    }
    if (!t0 && R.rt.depth) { t0 = tex_mtl(R.rt.depth); level = R.rt.has_dview && R.rt.dview.set_levels ? R.rt.dview.base_level : 0; }
    *tw = t0 ? mip_extent(t0.width, level) : 1; *th = t0 ? mip_extent(t0.height, level) : 1;
}
// There are pending clears but no draw in the pass: open a pass once to land the clears
static void rp_flush(void) {
    int pend = R.clear_d || R.clear_s;
    for (int i = 0; i < 8; i++) pend |= R.clear_c[i];
    if (pend && !g_enc && g_cb) rp_begin_ex(1);
    rp_end();
}
static int scissor_full(const tex_t *t, const view_t *v, int has_v) {
    NSUInteger level = has_v && v->set_levels ? v->base_level : 0;
    NSUInteger w = mip_extent(t->d.width, level), h = mip_extent(t->d.target == 0 || t->d.target == 3 ? 1 : t->d.height, level);
    return !R.has_sc || (R.sc.x <= 0 && R.sc.y <= 0 && (int64_t)R.sc.x + R.sc.w >= (int64_t)w && (int64_t)R.sc.y + R.sc.h >= (int64_t)h);
}
// Scissored zero clear (see OP_CLEAR_COLOR): land pending clears first, then copy from the zero buffer into the scissor rectangle of the attachment's level / slice in the replay stream's copy encoder. Multisample, 3D, depth / compressed formats are not handled (returns 0 and takes the old path)
static int clear_rect_zero(int i) {
    const tex_t *t = R.rt.color[i]; id<MTLTexture> mt = tex_mtl(t); if (!mt) return 0;
    MTLTextureType tt = mt.textureType;
    if (tt == MTLTextureType2DMultisample || tt == MTLTextureType2DMultisampleArray || tt == MTLTextureType3D || tt == MTLTextureTypeTextureBuffer) return 0;
    size_t bpp = mtl_pf_bytes(mt.pixelFormat); if (!bpp) return 0;
    const view_t *v = &R.rt.cview[i]; int hv = R.rt.has_cview[i];
    NSUInteger lv = hv && v->set_levels ? v->base_level : 0, sl = hv && v->set_layers ? v->min_layer : 0;
    long lw = (long)MAX(mt.width >> lv, 1u), lh = (long)MAX(mt.height >> lv, 1u);
    long x0 = R.sc.x < 0 ? 0 : R.sc.x, y0 = R.sc.y < 0 ? 0 : R.sc.y, x1 = (long)R.sc.x + R.sc.w, y1 = (long)R.sc.y + R.sc.h;
    if (x1 > lw) x1 = lw; if (y1 > lh) y1 = lh;
    if (x1 <= x0 || y1 <= y0) return 1;
    size_t row = (size_t)(x1 - x0) * bpp, img = row * (size_t)(y1 - y0);
    id<MTLBuffer> zb = zero_buf(img); if (!zb) return 0;
    rp_flush();   // a pending full clear on the same attachment must land first, or it would overwrite this one
    enc_b_copy_buffer_texture(blit(), zb, 0, row, img, MTLSizeMake((NSUInteger)(x1 - x0), (NSUInteger)(y1 - y0), 1), mt, sl, lv, MTLOriginMake((NSUInteger)x0, (NSUInteger)y0, 0));
    static uint64_t n; if (!n++) MLOG("剪裁清零：首例 %ldx%ld @%ld,%ld（目标 %lux%lu 级 %lu）", x1 - x0, y1 - y0, x0, y0, (unsigned long)mt.width, (unsigned long)mt.height, (unsigned long)lv);
    return 1;
}


// ================= Shaders / descriptor pool / pipelines / draws =================
static pthread_mutex_t g_rmu = PTHREAD_MUTEX_INITIALIZER;
// ---------- Shader mapping table: Switch control section hash → PC metallib (produced by dm/gfx/tools/shader_map.py) ----------
static NSMutableDictionary<NSString *, NSArray *> *g_shmap;
static NSString *g_shmap_path, *g_msl_dir;
static NSString *exe_rel(const char *rel) {
    char buf[4096]; uint32_t n = sizeof buf;
    if (_NSGetExecutablePath(buf, &n)) return nil;
    char *dir = dirname(buf);
    return [NSString stringWithFormat:@"%s/%s", dir, rel];
}
// Shader mapping table (output of tools/shader_map.py): at runtime only 'Switch control section key → stage' is used; shaders are always translations of the Switch originals (tools/dmsh_build.py produces <key>_b.metallib + reflection offline)
static void shmap_load(void) {
    if (g_shmap) return;
    const char *e = getenv("DM_NVN_SHMAP");
    NSString *path = e ? @(e) : exe_rel("../games/" DM_GAME_ID "/shader_map.tsv");   // default: the current game profile folder (Mac: dm/games/<id>; on iOS main.m sets DM_NVN_SHMAP)
    NSString *txt = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:nil];
    if (!txt) MDIE("读不到着色器映射表 %s（DM_NVN_SHMAP 可指定）", path.UTF8String);
    g_shmap = [NSMutableDictionary dictionary];
    for (NSString *ln in [txt componentsSeparatedByString:@"\n"]) {
        if (!ln.length || [ln hasPrefix:@"#"] || [ln hasPrefix:@"package"]) continue;
        NSArray *c = [ln componentsSeparatedByString:@"\t"];
        if (c.count > 3) g_shmap[c[3]] = c[2];
    }
    g_shmap_path = path;
    const char *m = getenv("DM_MSL_DIR");
    // Library directory = <DM_MSL_DIR or default cache>/b-<translation chain version>: the version comes from the game profile's dmsh_chain.h (tools/dmsh_chain.py); a host built from an old source tree only accepts libraries of the old version
    g_msl_dir = [NSString stringWithFormat:@"%@/b-%s", m ? @(m) : [NSString stringWithFormat:@"%s%s/build/cache/metal_shader_cache", getenv("DM_ROOT") ?: getenv("HOME"), getenv("DM_ROOT") ? "" : "/dm-work"], DMSH_CHAIN];
    if (![NSFileManager.defaultManager fileExistsAtPath:g_msl_dir]) MLOG("【红】着色器库目录 %s 不存在：本源码树的翻译链版本 %s 还没出包（dm/gfx/tools/dmsh_build.py）", g_msl_dir.UTF8String, DMSH_CHAIN);
    NSString *cp = [[path stringByDeletingLastPathComponent] stringByAppendingPathComponent:@"c1_full.tsv"];   // Same directory as the mapping table (Mac: dm/games/<id>; iOS: dm/ inside the App, copied along by build_ipa.sh)
    NSString *ct = cp ? [NSString stringWithContentsOfFile:cp encoding:NSUTF8StringEncoding error:nil] : nil;
    NSString *head = ct ? [[ct componentsSeparatedByString:@"\n"] firstObject] : nil;
    NSString *want = [NSString stringWithFormat:@"# chain b-%s", DMSH_CHAIN];
    if (!head || ![head hasPrefix:want]) {
        MLOG("【红】c1_full.tsv 清单过期或缺失，全部 c1 独占 64 KiB");
    } else {
        g_c1_full_valid = 1; g_c1_full = [NSMutableSet set];
        for (NSString *line in [ct componentsSeparatedByString:@"\n"]) if (line.length && ![line hasPrefix:@"#"]) [g_c1_full addObject:[line stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet]];
    }
    MLOG("着色器路线 B（Switch 原版翻译）：映射表 %lu 条（%s），metallib 目录 %s", (unsigned long)g_shmap.count, path.UTF8String, g_msl_dir.UTF8String);
    warm_start();
}

@interface NvShader : NSObject { @public id<MTLFunction> fn; nvn_refl rf; id<MTLArgumentEncoder> ae; NSUInteger arg_len; int stage, range_ro, afr; const char *builtin; uint64_t logged; char key[24]; id<MTLBuffer> c1buf; uint64_t c1_gpu; size_t c1off;
    // The argument buffer encoded last time
    id<MTLBuffer> ab_buf; uint64_t ab_gpu; size_t ab_off; uint64_t ab_pass;   // ab_pass: serial of the render pass that encoded this argument block - reused only within the same pass (the arena holding a block from another pass / another submission may already have been reclaimed and rewritten)
    id<MTLLibrary> bflib; int bflib_tried; NSMutableDictionary<NSString *, id<MTLFunction>> *bffn; } @end   // bflib: readable-tile version <key>_bf (merge B step 2)
@implementation NvShader @end
#include "nvn_pso_skey.h"
@interface NvProg : NSObject { @public NvShader *sh[6]; } @end
@implementation NvProg @end
// Warm-up entry: built in the background, waiting to be taken by the submit thread. vs / fs are strong references (in pso they are only raw pointers, and the background dictionary is released when the task ends)
@interface WarmItem : NSObject { @public int kind; NSString *key, *vs_key, *fs_key; NvShader *shader, *vs, *fs; pso_key pso; id<MTLFunction> fsfn; id object; } @end
@implementation WarmItem @end
static NSMutableDictionary<NSString *, NvShader *> *g_shcache;

static void c1_store(NvShader *o) {
    size_t len = o->rf.c1_len;
    NSString *guardKey = [NSString stringWithUTF8String:o->key];
    if ([guardKey hasSuffix:@"_b"]) guardKey = [guardKey substringToIndex:guardKey.length - 2];
    os_unfair_lock_lock(&g_c1_lock);   // Both the warm-up thread and the submit thread load shaders
    if (!g_c1_full_valid || [g_c1_full containsObject:guardKey]) {   // Runtime index not proven bounded: exclusive 64 KiB of all zeros (same as the original implementation)
        o->c1buf = [g_dev newBufferWithLength:C1_DECL_SIZE options:MTLResourceStorageModeShared]; o->c1_gpu = o->c1buf.gpuAddress;
        if (!o->c1buf) { os_unfair_lock_unlock(&g_c1_lock); MDIE("c1 独占缓冲分配失败"); }
        res_add(o->c1buf);
        memset(o->c1buf.contents, 0, C1_DECL_SIZE); memcpy(o->c1buf.contents, o->rf.c1, len);
        g_c1_alloc_bytes += C1_DECL_SIZE; g_c1_data_bytes += len; g_c1_count++; g_c1_exclusive++;
        os_unfair_lock_unlock(&g_c1_lock);
        return;
    }
    size_t need = (len + 255u) & ~((size_t)255u);
    if (!g_c1_banks) g_c1_banks = [NSMutableArray array];
    if (!g_c1_banks.count || g_c1_bank_off + need + 256u + C1_DECL_SIZE > C1_BANK_SIZE) {
        id<MTLBuffer> b = [g_dev newBufferWithLength:C1_BANK_SIZE options:MTLResourceStorageModeShared];
        if (!b) { os_unfair_lock_unlock(&g_c1_lock); MDIE("c1 缓冲块分配失败"); }
        res_add(b);
        memset(b.contents, 0, C1_BANK_SIZE);
        [g_c1_banks addObject:b];
        g_c1_bank_off = 0;
        g_c1_alloc_bytes += C1_BANK_SIZE;
    }
    id<MTLBuffer> b = g_c1_banks.lastObject;
    size_t off = g_c1_bank_off;
    memcpy((uint8_t *)b.contents + off, o->rf.c1, len);
    g_c1_bank_off += need + 256u;
    g_c1_data_bytes += len;
    g_c1_count++;
    o->c1buf = b;
    o->c1_gpu = b.gpuAddress;
    o->c1off = off;
    os_unfair_lock_unlock(&g_c1_lock);
}

static NSMutableArray<WarmItem *> *g_warm_items;
static os_unfair_lock g_warm_lock = OS_UNFAIR_LOCK_INIT;
static _Atomic int g_warm_ready;
static int g_warm_started;
static uint64_t g_warm_t0;
static _Atomic int g_warm_running;   // Background warm-up has not finished yet (the loading screen on entering a run waits a little longer based on this, see mtl_warm_wait)
static uint64_t g_warm_shaders, g_warm_rps, g_warm_cps, g_warm_skip_fz;
static uint64_t g_warm_claim_shaders, g_warm_claim_rps, g_warm_claim_cps;
static id<MTLFunction> bf_fn(NvShader *fs, const int *fx, uint8_t fbmask, const uint8_t *lmap);
static int rt_range_bindings_ro(NSDictionary *j);
static FILE *g_pso_record; static int g_pso_record_checked;
static FILE *pso_record_file(void) {
    if (g_pso_record_checked) return g_pso_record;
    g_pso_record_checked = 1;
    const char *path = getenv("DM_PSO_RECORD");
    if (path && *path) g_pso_record = fopen(path, "a");
    return g_pso_record;
}
static void pso_record(const char *fmt, ...) {
    FILE *f = pso_record_file();
    if (!f) return;
    va_list ap; va_start(ap, fmt); vfprintf(f, fmt, ap); va_end(ap); fflush(f);
}
// User manifest: shaders / pipelines created synchronously on the submit thread while running on this machine, warmed up at the next launch together with the seed manifest.
// Path = DM_PSO_USER, or pso_user.tsv under the parent of the save directory (DM_SAVE, Documents/save on iOS); tests use a fresh save copy per run, so they do not pollute each other
static NSString *pso_user_path(void) {
    const char *u = getenv("DM_PSO_USER"), *save = getenv("DM_SAVE");
    if (u && *u) return @(u);
    return save && *save ? [[@(save) stringByDeletingLastPathComponent] stringByAppendingPathComponent:@"pso_user.tsv"] : nil;
}
static FILE *g_pso_user; static int g_pso_user_checked;
static NSMutableSet<NSString *> *g_pso_user_seen;
static FILE *pso_user_file(void) {
    if (g_pso_user_checked) return g_pso_user;
    g_pso_user_checked = 1;
    NSString *path = pso_user_path();
    if (path) g_pso_user = fopen(path.fileSystemRepresentation, "a");
    return g_pso_user;
}
static void pso_user_record(const char *fmt, ...) {
    FILE *f = pso_user_file();
    if (!f) return;
    char line[8192]; va_list ap; va_start(ap, fmt); vsnprintf(line, sizeof line, fmt, ap); va_end(ap);
    if (!g_pso_user_seen) {
        g_pso_user_seen = [NSMutableSet set];
        NSString *seed = g_shmap_path ? [g_shmap_path.stringByDeletingLastPathComponent stringByAppendingPathComponent:@"pso_manifest.tsv"] : nil;
        NSMutableArray<NSString *> *paths = [NSMutableArray array]; if (seed) [paths addObject:seed]; NSString *user = pso_user_path(); if (user) [paths addObject:user];
        for (NSString *path in paths) { NSString *txt = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:nil];
            for (NSString *old in [txt componentsSeparatedByString:@"\n"]) if (old.length && ![old hasPrefix:@"#"]) [g_pso_user_seen addObject:old]; }
    }
    size_t ll = strlen(line); NSString *s = [[NSString alloc] initWithBytes:line length:ll && line[ll - 1] == '\n' ? ll - 1 : ll encoding:NSUTF8StringEncoding];   // Compared with the manifest file line by line (without the newline)
    if (!s || [g_pso_user_seen containsObject:s]) return;
    [g_pso_user_seen addObject:s]; fputs(line, f); fflush(f);
}
// Build the library from a .metallib file without permanently occupying a file descriptor: newLibraryWithURL keeps that file open for as long as the library object lives (one per game shader - of the 258 descriptors the iOS app permanently holds in a run, 213 are these,
// while the default soft limit of an iOS process is 256; new shaders created in other acts raise it further). Here the file is mapped into memory and closed immediately, and the library is built from the in-memory data; the mapping is released when the library's data object is freed
static id<MTLLibrary> lib_from_file(NSString *path, NSError **err) {
    int fd = open(path.fileSystemRepresentation, O_RDONLY); struct stat st;
    if (fd < 0 || fstat(fd, &st) || st.st_size <= 0) { if (fd >= 0) close(fd); if (err) *err = [NSError errorWithDomain:NSPOSIXErrorDomain code:errno userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithFormat:@"打不开 %@", path]}]; return nil; }
    size_t len = (size_t)st.st_size; void *m = mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
    if (m == MAP_FAILED) { if (err) *err = [NSError errorWithDomain:NSPOSIXErrorDomain code:errno userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithFormat:@"映射失败 %@", path]}]; return nil; }
#ifdef DM_SHELL
    // Shell: when the user's computer has no Apple shader compiler (the no-compiler mode of tools/d2rm_pack.py, gfx/tools/nometal_run.py), the "library files" contain
    // shader source rather than compiled libraries (a compiled library starts with MTLB). They are compiled here on the spot; the system caches the compiled result, and from the second launch on the same shader no longer costs this time.
    // Measured on a real device (cold): about 80 ms each on average, about 450 ms for the longest single one; one session uses several hundred.
    if (len < 4 || memcmp(m, "MTLB", 4)) { NSString *src = [[NSString alloc] initWithBytes:m length:len encoding:NSUTF8StringEncoding]; munmap(m, len);
        if (!src) { if (err) *err = [NSError errorWithDomain:NSPOSIXErrorDomain code:EILSEQ userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithFormat:@"既不是编好的库也不是源码 %@", path]}]; return nil; }
        static _Atomic int n; static _Atomic long long us; CFAbsoluteTime t0 = CFAbsoluteTimeGetCurrent();
        id<MTLLibrary> l = [g_dev newLibraryWithSource:src options:nil error:err]; long long dt = (long long)((CFAbsoluteTimeGetCurrent() - t0) * 1e6);
        int k = atomic_fetch_add(&n, 1) + 1; long long tot = atomic_fetch_add(&us, dt) + dt;
        if (k == 1 || k % 100 == 0 || dt >= 300000) MLOG("着色器现编：第 %d 个 %.0f ms，累计 %.1f 秒", k, dt / 1e3, tot / 1e6);
        return l; }
#endif
    dispatch_data_t d = dispatch_data_create(m, len, dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{ munmap(m, len); });
    return [g_dev newLibraryWithData:d error:err];
}
static NvShader *shader_load(NSString *pc, int stage) {
    NSError *e = nil;
    NSString *lp = [NSString stringWithFormat:@"%@/%@.metallib", g_msl_dir, pc];
    id<MTLLibrary> lib = lib_from_file(lp, &e);
    NSData *jd = [NSData dataWithContentsOfFile:[NSString stringWithFormat:@"%@/%@.json", g_msl_dir, pc]];
    NSDictionary *j = jd ? [NSJSONSerialization JSONObjectWithData:jd options:0 error:nil] : nil;
    if (!lib || !j) { MLOG("【红】加载着色器失败 %s：%s", pc.UTF8String, e ? e.localizedDescription.UTF8String : "无反射 json"); return nil; }
    if (![j[@"chain"] isEqual:@DMSH_CHAIN]) { MLOG("【红】着色器 %s 的翻译链版本 %s 与本宿主期望的 %s 不符", pc.UTF8String, [[j[@"chain"] description] UTF8String], DMSH_CHAIN); return nil; }
    NvShader *o = [NvShader new];
    char err[160];
    if (nvn_refl_from_json(j, &o->rf, err, sizeof err)) { MLOG("【红】着色器 %s 反射转换失败：%s", pc.UTF8String, err); return nil; }
    if (rt_range_on()) o->range_ro=rt_range_bindings_ro(j);
    o->fn = [lib newFunctionWithName:j[@"entry"]]; o->stage = stage; strlcpy(o->key, pc.UTF8String, sizeof o->key);
    if (o->rf.nb) {
        o->ae = [o->fn newArgumentEncoderWithBufferIndex:o->rf.abi];
        int max_aid = -1;
        for (int i = 0; i < o->rf.nb; i++) max_aid = MAX(max_aid, o->rf.b[i].aid);
        o->arg_len = o->ae.encodedLength;
        if (o->arg_len != (NSUInteger)(8 * (max_aid + 1)) || (o->arg_len & 7))
            MDIE("着色器 %s 参数块布局异常：encodedLength=%lu maxAid=%d", o->key, (unsigned long)o->arg_len, max_aid);
    }
    if (o->rf.has_c1) c1_store(o);
    return o;
}

static int stage_of(NSString *zh) {
    if ([zh isEqualToString:@"顶点"]) return 0; if ([zh isEqualToString:@"像素"]) return 1; if ([zh isEqualToString:@"计算"]) return 5;
    if ([zh isEqualToString:@"外壳"]) return 3; if ([zh isEqualToString:@"域"]) return 4; return -1;
}
#define NVN_RT_RANGE_RO_IMPL 1
// Private continuation accepts only B reflection that is explicitly RO; a missing rw on sbuf does not let the default 0 count as proof; does not change the existing reflection / translation chain.
static int rt_range_bindings_ro(NSDictionary *j) {
    if (![j[@"route"] isEqual:@"B"] || ![j[@"bindings"] isKindOfClass:[NSArray class]] ||
        ![j[@"drv_reads"] isKindOfClass:[NSDictionary class]] || [j[@"drv_reads"] count]) return 0;
    for (NSDictionary *b in j[@"bindings"]) {
        if (![b isKindOfClass:[NSDictionary class]] || ![b[@"kind"] isKindOfClass:[NSString class]] || ![b[@"msl_type"] isKindOfClass:[NSString class]]) return 0;
        NSString *k=b[@"kind"], *t=b[@"msl_type"];
        if (b[@"rw"] && (![b[@"rw"] isKindOfClass:[NSNumber class]] || [b[@"rw"] boolValue])) return 0;
        if ([k isEqual:@"cbuf"] || [k isEqual:@"c1"]) { if (![t hasPrefix:@"constant "]) return 0; }
        else if ([k isEqual:@"sbuf"]) { if (![b[@"rw"] isKindOfClass:[NSNumber class]] || [b[@"rw"] boolValue] || ![t hasPrefix:@"const device "]) return 0; }
        else if ([k isEqual:@"samp"] || [k isEqual:@"sampt"]) { if (![t isEqual:@"sampler"]) return 0; }
        else if ([k isEqual:@"tex"] || [k isEqual:@"tbuf"]) {
            if (![t hasPrefix:@"texture"] || ([t containsString:@"access::"] && ![t containsString:@"access::sample>"] && ![t containsString:@"access::read>"])) return 0; }
        else return 0;
    }
    return 1;
}
static int rt_range_pr_ro(NvProg *p) {
    if (!p || !p->sh[0] || !p->sh[1]) return 0;
    for (int k=0;k<6;k++) if (p->sh[k] && (k>1 || p->sh[k]->stage != k || !p->sh[k]->range_ro || p->sh[k]->rf.drv_read[0] || p->sh[k]->rf.drv_read[2])) return 0;
    return 1;
}
static NvShader *shader_get(NSString *pc, int stage) {
    if (!g_shcache) g_shcache = [NSMutableDictionary dictionary];
    NvShader *o = g_shcache[pc];
    if (o) return o;
    uint64_t bt = bp_now(); o = shader_load(pc, stage);
    if (!o) return nil;
    g_shcache[pc] = o;
    pso_record("S\t%s\t%d\n", pc.UTF8String, stage);
    pso_user_record("S\t%s\t%d\n", pc.UTF8String, stage);
    g_bp_lib_n++; g_bp_lib_ns += bp_now() - bt;
    return o;
}
static id<MTLLibrary> g_builtin_lib;
static NSMutableDictionary<NSString *, NvShader *> *g_builtin;
static NvShader *builtin_get(const char *ctl) {
    for (unsigned i = 0; i < sizeof k_nvn_builtin_map / sizeof *k_nvn_builtin_map; i++) {
        if (strcmp(ctl, k_nvn_builtin_map[i].ctl)) continue;
        if (!g_builtin) g_builtin = [NSMutableDictionary dictionary];
        NvShader *o = g_builtin[@(ctl)];
        if (o) return o;
        if (!g_builtin_lib) {
            NSError *e = nil; g_builtin_lib = [g_dev newLibraryWithSource:@(k_nvn_builtin_msl) options:nil error:&e];
            if (!g_builtin_lib) MDIE("内嵌计算核 MSL 编译失败：%s", e.localizedDescription.UTF8String);
        }
        o = [NvShader new]; o->fn = [g_builtin_lib newFunctionWithName:@(k_nvn_builtin_map[i].fn)]; o->stage = 5; o->builtin = k_nvn_builtin_map[i].fn;
        g_builtin[@(ctl)] = o;
        return o;
    }
    return nil;
}
static __unsafe_unretained NvProg *prog_get(const prog_t *cp) {   // Called only on the submit thread
    prog_t *p = (prog_t *)cp;
    static const prog_t *lp; static uint64_t lser, lgen; static NvProg *lpr;   // Last result: same object, same serial and unchanged serial table means it is used directly
    uint64_t gen = g_ser_gen;
    if (p == lp && p->serial == lser && gen == lgen && lpr) return lpr;
    __unsafe_unretained NvProg *have = p->serial ? serial_get(p->serial) : nil;
    if (have) { lp = p; lser = p->serial; lgen = gen; lpr = have; return have; }
    shmap_load();
    NvProg *o = [NvProg new];
    for (unsigned i = 0; i < p->nstage; i++) {
        // Key = sha256(first 0x880 bytes of the control section + code area): many of Mario's shaders share a control section but differ in code (dm/gfx/tools/shader_split.py). It was computed at SetShaders (hle_nvn.c): by now the game may have freed the control section
        uint8_t md[8]; prog_key(p, i, md);
        NSString *k = [NSString stringWithFormat:@"%02x%02x%02x%02x%02x%02x%02x%02x", md[0], md[1], md[2], md[3], md[4], md[5], md[6], md[7]];
        NSString *zh = g_shmap[k];
        if (!zh) {   // Not in the package: look up the compute kernels embedded in the main body (nvn_builtin.h)
            NvShader *b = builtin_get(k.UTF8String);
            if (b) { o->sh[5] = b; continue; }
            MLOG("【红】着色器 %s 不在映射表里（DM_NVN_SHDUMP=目录 可把控制段与代码区存盘）", k.UTF8String);
            // DM_NVN_SHDUMP=directory: shaders missing from the map are saved as <key>.ctl (0x880-byte control section) and <key>.code (code area, length from control +0x704), to trace their source / add them to the map offline
            static const char *shd; static int shd_init; if (!shd_init) { shd = getenv("DM_NVN_SHDUMP"); shd_init = 1; }
            if (shd) { char path[1024]; FILE *f; uint32_t ncode; memcpy(&ncode, (const uint8_t *)p->sd[i].control + 0x704, 4);   // with SHDUMP set the control section is the host's copy
                snprintf(path, sizeof path, "%s/%s.ctl", shd, k.UTF8String); if ((f = fopen(path, "wb"))) { fwrite(p->sd[i].control, 1, 0x880, f); fclose(f); }
                snprintf(path, sizeof path, "%s/%s.code", shd, k.UTF8String);
                if (p->sd[i].data && ncode && ncode < (1u << 24) && (f = fopen(path, "wb"))) { fwrite((const void *)(uintptr_t)p->sd[i].data, 1, ncode, f); fclose(f); } }
            continue;
        }
        int st = stage_of(zh);
        NvShader *sh = st >= 0 ? shader_get([k stringByAppendingString:@"_b"], st) : nil;
        if (sh) o->sh[st] = sh; else MLOG("【红】着色器 %s（%s）加载失败：先跑 dm/gfx/tools/dmsh_build.py 出包", k.UTF8String, zh.UTF8String);
    }
    if (p->serial) serial_put(p->serial, o);
    return o;
}
void mtl_prog_release(prog_t *p) { if (p->serial) serial_del(p->serial); }

// ---------- Descriptor pool: texture ID / sampler ID → object ----------
// At registration the whole texture object is snapshotted (NVN semantics: the descriptor has been written into the pool); view: the MTLTexture for that ID (a view or the base texture, held via CFBridgingRetain) and the serial-table / guest-write generations at the time it was obtained
typedef struct { tex_t t; int ok; view_t v; int has_v; void *view; uint64_t view_id, view_rid, view_ser, view_dirty;
                 int view_pkg;   // the texture is CPU-written (packaged): only those depend on the guest writes (view_dirty). A GPU-only texture keeps its view across guest writes.
                 void *alt, *alt_of; uint64_t alt_rid; uint8_t alt_type;   // alt: compatible view used when the shader's declared type differs from view (tex_fit), alt_of = the view it was built from
                 void *ar, *ar_of; uint64_t ar_rid; } treg_t;              // ar: the alpha-from-red view for the profile's alpha_from_red shaders (tex_alpha_red), ar_of = the view it was built from
static treg_t *g_treg; static int g_ntreg;
// ---------- DM_POOL_CENSUS=1: guest memory pool census ----------
// Prints what the NVN memory pools are made of: request size / flags / residency, when and through which host path the GPU first took the pool's address, how many
// address hits it got and when the last one was, the guest offset range ever addressed, and how many textures are built on its guest storage (with their NVN storage bytes).
// Why: pool sizes alone cannot explain the guest resident class -- those pages are wired exactly when a pool's no-copy MTLBuffer is in the residency set -- and the residency
// set only ever adds a pool because mtl_addr / mtl_gpu_addr took its address, usually a packaged-texture upload reading its source bytes.
// Textures are attributed through two independent tables, each walked under its own lock and never nested here (the established order is g_rmu -> g_smu, so they are taken
// one at a time): the storage-sharing table g_alias (one entry per pool + offset + description, shared by every texture object with the same storage) and the
// registered-texture table g_treg (a snapshot per texture ID). A texture is attributed by its pool object, or by its guest storage address falling inside a pool (textures
// on a virtual pool point into their physical pool's range the same way). Aliases whose pool object is not a registered physical pool stay unattributed.
// A pool is "upload-source only" when every texture registered on it is packaged (has an NVN packaged layout, or was CPU-written and is re-uploaded as packaged), none of
// them changed since its upload (nvn_dirty_since), there is no zero-copy view on it (no alias entry, no linear texture) and the GPU has not addressed it in the last 300
// frames: the pool memory is then only the source copy, so its guest pages are the step-2 reclaim candidate.
// Called from mtl_mem_stat; prints outside all locks. Rounds are capped so a 60 s device run cannot flood the log.
typedef struct {
    uint32_t alias_n, treg_n, treg_pkg_n, treg_lin_n, up_n, dirty_n;
    uint64_t alias_b, treg_b, treg_pkg_b, up_b, dirty_b, last_up_frame, last_up_stamp, reup_n;
} pool_tex_use;
typedef struct { uint64_t key; int idx; } pc_slot;
static int pc_slot_cmp(const void *a, const void *b) { const pc_slot *x = a, *y = b; return x->key < y->key ? -1 : x->key > y->key ? 1 : 0; }
static int pc_obj_find(const pc_slot *o, int n, uintptr_t obj) {   // binary search by pool object address
    int lo = 0, hi = n - 1;
    while (lo <= hi) { int m = (lo + hi) / 2; if (o[m].key < obj) lo = m + 1; else if (o[m].key > obj) hi = m - 1; else return o[m].idx; }
    return -1;
}
static int pc_pool_at(const pool_ent *ps, const pc_slot *by_lo, int n, uint64_t addr, uint64_t len) {   // pool containing a guest range; by_lo is sorted by start address
    if (n <= 0) return -1;
    len = len ? len : 1;
    int lo = 0, hi = n - 1, hit = -1;
    while (lo <= hi) { int m = (lo + hi) / 2; if (ps[by_lo[m].idx].lo <= addr) { hit = m; lo = m + 1; } else hi = m - 1; }
    if (hit >= 0) { const pool_ent *p = &ps[by_lo[hit].idx]; if (addr + len <= p->hi) return by_lo[hit].idx; }
    for (int i = 0; i < n; i++) if (addr >= ps[i].lo && addr + len <= ps[i].hi) return i;   // overlapping pools (rare)
    return -1;
}
static void pool_flags_str(uint32_t f, char *o, size_t n) {
    static const struct { uint32_t bit; const char *name; } t[] = {
        {NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS, "CPU_NO_ACCESS"}, {NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED, "CPU_UNCACHED"}, {NVN_MEMORY_POOL_FLAGS_CPU_CACHED, "CPU_CACHED"},
        {NVN_MEMORY_POOL_FLAGS_GPU_NO_ACCESS, "GPU_NO_ACCESS"}, {NVN_MEMORY_POOL_FLAGS_GPU_UNCACHED, "GPU_UNCACHED"}, {NVN_MEMORY_POOL_FLAGS_GPU_CACHED, "GPU_CACHED"},
        {NVN_MEMORY_POOL_FLAGS_SHADER_CODE, "SHADER_CODE"}, {NVN_MEMORY_POOL_FLAGS_COMPRESSIBLE, "COMPRESSIBLE"},
        {NVN_MEMORY_POOL_FLAGS_PHYSICAL, "PHYSICAL"}, {NVN_MEMORY_POOL_FLAGS_VIRTUAL, "VIRTUAL"} };
    size_t p = 0; o[0] = 0;
    for (size_t i = 0; i < sizeof t / sizeof t[0]; i++) if (f & t[i].bit) p += (size_t)snprintf(o + p, p < n ? n - p : 0, "%s%s", p ? "|" : "", t[i].name);
    if (!p) snprintf(o, n, "0");
}
// Resolve a recorded return address: the host is linked without -rdynamic, so dladdr usually has no local symbol and this prints module+offset (dmhost+0x1234)
static void pool_site_str(uintptr_t ra, char *o, size_t n) {
    Dl_info di;
    if (!ra) { snprintf(o, n, "-"); return; }
    if (dladdr((void *)ra, &di) && di.dli_fname) {
        const char *fn = strrchr(di.dli_fname, '/'); fn = fn ? fn + 1 : di.dli_fname;
        if (di.dli_sname) snprintf(o, n, "%s+%#lx", di.dli_sname, (unsigned long)(ra - (uintptr_t)di.dli_saddr));
        else snprintf(o, n, "%s+%#lx", fn, (unsigned long)(ra - (uintptr_t)di.dli_fbase));
    } else snprintf(o, n, "%#lx", (unsigned long)ra);
}
static int pool_upload_source_only(const pool_ent *p, const pool_tex_use *u, unsigned long long frame) {
    if (!u->treg_n || u->treg_pkg_n != u->treg_n) return 0;   // only packaged textures registered on it
    if (u->treg_lin_n || u->alias_n) return 0;                 // a zero-copy view still reads the pool
    if (u->dirty_n) return 0;                                  // source changed since the upload: it will be re-uploaded
    if (p->last_hit_frame && frame < p->last_hit_frame + 300) return 0;   // GPU addressed it recently (the upload itself reads it; later draws do not)
    return 1;
}
static void pool_census(unsigned long long frame, unsigned long long rss_mb, const char *why) {
    if (!g_pool_census_on) return;
    if (!why && (frame == 0 || frame % 100)) return;   // every 100 frames while the switch is on; why != NULL forces a print (low-memory guard hits at the fatal transition)
    if (g_pool_census_prints >= 12) { if (!why) return; }   // capped: a 60 s device run must not flood the log; a forced print is always shown
    g_pool_census_prints++;
    if (why) MLOG("池普查 强制打印（帧 %llu）：%s", frame, why);
    // ---- pool table snapshot (g_mu only; no other lock is taken while it is held) ----
    pthread_mutex_lock(&g_mu);
    int np = g_npool;
    pool_ent *ps = np > 0 ? malloc((size_t)np * sizeof *ps) : NULL;
    if (ps) memcpy(ps, g_pools, (size_t)np * sizeof *ps);
    uint64_t adds = g_pool_adds, dels = g_pool_dels, addb = g_pool_add_bytes, delb = g_pool_del_bytes;
    pthread_mutex_unlock(&g_mu);
    if (!ps) return;
    pc_slot *by_size = malloc((size_t)np * sizeof *by_size), *by_lo = malloc((size_t)np * sizeof *by_lo), *by_obj = malloc((size_t)np * sizeof *by_obj);
    pool_tex_use *use = calloc((size_t)np, sizeof *use);
    if (!by_size || !by_lo || !by_obj || !use) { free(ps); free(by_size); free(by_lo); free(by_obj); free(use); return; }
    int nobj = 0; unsigned long long sum_size = 0, sum_buflen = 0;
    for (int i = 0; i < np; i++) {
        by_size[i].key = ~(uint64_t)ps[i].size; by_size[i].idx = i;   // qsort ascending -> largest first
        by_lo[i].key = ps[i].lo; by_lo[i].idx = i;
        if (ps[i].obj) { by_obj[nobj].key = (uintptr_t)ps[i].obj; by_obj[nobj].idx = i; nobj++; }
        sum_size += ps[i].size; sum_buflen += ps[i].buflen;
    }
    qsort(by_size, (size_t)np, sizeof *by_size, pc_slot_cmp);
    qsort(by_lo, (size_t)np, sizeof *by_lo, pc_slot_cmp);
    qsort(by_obj, (size_t)nobj, sizeof *by_obj, pc_slot_cmp);
    // ---- storage-sharing table snapshot (g_smu only): key copy + the MTLTexture allocatedSize that the residency census charges per view ----
    uint32_t na = 0, alias_cap = 0;
    alias_key *aks = NULL; uint64_t *aalloc = NULL;
    pthread_mutex_lock(&g_smu);
    alias_cap = g_alias.cap;
    if (alias_cap) {
        aks = malloc((size_t)alias_cap * sizeof *aks); aalloc = malloc((size_t)alias_cap * sizeof *aalloc);
        if (!aks || !aalloc) na = 0;
        else for (uint32_t i = 0; i < alias_cap; i++) if (g_alias.vals[i]) {
            memcpy(&aks[na], g_alias.keys + (size_t)i * g_alias.klen, sizeof *aks);
            aalloc[na] = (uint64_t)[(__bridge id)g_alias.vals[i] allocatedSize];
            na++;
        }
    }
    pthread_mutex_unlock(&g_smu);
    // ---- texture-ID registry snapshot (g_rmu only): pool object, guest storage / size, description ----
    typedef struct { const void *pool, *mem; uint64_t size, serial; uint32_t flags, width, height; uint8_t has_layout; } pc_tex;
    pc_tex *tx = NULL; int nt = 0;
    pthread_mutex_lock(&g_rmu);
    for (int i = 0; i < g_ntreg; i++) {
        const treg_t *r = &g_treg[i];
        if (!r->ok || !r->t.mem || r->t.magic != MAGIC('T', 'E', 'X', 0)) continue;
        if (!tx) { tx = malloc((size_t)(g_ntreg - i) * sizeof *tx); if (!tx) break; }
        tx[nt].pool = r->t.pool; tx[nt].mem = r->t.mem; tx[nt].size = r->t.L.size; tx[nt].serial = r->t.serial;
        tx[nt].flags = (uint32_t)r->t.d.flags; tx[nt].width = r->t.d.width; tx[nt].height = r->t.d.height; tx[nt].has_layout = r->t.has_layout;
        nt++;
    }
    pthread_mutex_unlock(&g_rmu);
    // packaged = has an NVN packaged layout, or CPU-written (re-uploaded as packaged). One lock for the whole batch (tex_cpu_written would lock per texture).
    uint8_t *pkg = nt ? calloc((size_t)nt, 1) : NULL;
    if (pkg) { pthread_mutex_lock(&g_cpu_tex_mu); for (int i = 0; i < nt; i++) pkg[i] = tx[i].has_layout || (g_cpu_tex && CFSetContainsValue(g_cpu_tex, (const void *)(uintptr_t)tx[i].serial)); pthread_mutex_unlock(&g_cpu_tex_mu); }
    // upload bookkeeping per registered texture: the NvPkg in the serial table (upload frame / stamp / re-uploads) and, for linear zero-copy views, the texture's
    // allocatedSize (= the parent pool buffer's length, which is what the residency census double-counts). One g_smu lock for all lookups.
    uint64_t *upf = NULL, *upst = NULL, *upre = NULL, *linalloc = NULL;
    if (nt) { upf = calloc((size_t)nt, sizeof *upf); upst = calloc((size_t)nt, sizeof *upst); upre = calloc((size_t)nt, sizeof *upre); linalloc = calloc((size_t)nt, sizeof *linalloc); }
    if (upf && upst && upre && linalloc) {
        pthread_mutex_lock(&g_smu);
        for (int i = 0; i < nt; i++) {
            void *v = nvn_hm_get(&g_by_serial, &tx[i].serial);
            if (!v) continue;
            id o = (__bridge id)v;
            if ([o isKindOfClass:[NvPkg class]]) { NvPkg *pk = (NvPkg *)o; upf[i] = pk->up_frame; upst[i] = pk->stamp; upre[i] = pk->reuploads; }
            else if (!(pkg && pkg[i]) && (tx[i].flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) && [o conformsToProtocol:@protocol(MTLTexture)])
                linalloc[i] = (uint64_t)[(id<MTLTexture>)o allocatedSize];
        }
        pthread_mutex_unlock(&g_smu);
    }
    // ---- attribution ----
    uint32_t alias_un = 0, treg_un = 0, treg_all = 0, pkg_all = 0, lin_all = 0;
    unsigned long long alias_all_b = 0, alias_un_b = 0, alias_alloc_all = 0, treg_un_b = 0, treg_all_b = 0, pkg_all_b = 0, lin_all_b = 0, lin_alloc_all = 0;
    enum { PC_TOP = 8 };
    typedef struct { uint64_t size, alloc, off, poolobj, mem; uint32_t w, h, levels, fmt, target; } pc_top;
    pc_top atop[PC_TOP], ltop[PC_TOP]; int natop = 0, nltop = 0;
    for (uint32_t i = 0; i < na; i++) {
        nvn_texlayout L; nvn_tex_layout(&aks[i].d, &L);
        alias_all_b += L.size; alias_alloc_all += aalloc[i];
        int idx = pc_obj_find(by_obj, nobj, (uintptr_t)aks[i].pool);
        if (idx >= 0) { use[idx].alias_n++; use[idx].alias_b += L.size; } else { alias_un++; alias_un_b += L.size; }
        if (L.size && (natop < PC_TOP || L.size > atop[PC_TOP - 1].size)) {
            int k = natop < PC_TOP ? natop++ : PC_TOP - 1;
            for (; k > 0 && atop[k - 1].size < L.size; k--) atop[k] = atop[k - 1];
            atop[k] = (pc_top){L.size, aalloc[i], (uint64_t)aks[i].off, aks[i].pool, 0, (uint32_t)aks[i].d.width, (uint32_t)aks[i].d.height,
                               (uint32_t)aks[i].d.levels, (uint32_t)aks[i].d.format, (uint32_t)aks[i].d.target};
        }
    }
    for (int i = 0; i < nt; i++) {
        int p = pkg ? pkg[i] : 0;
        int lin = !p && (tx[i].flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET));
        treg_all++; treg_all_b += tx[i].size; if (p) { pkg_all++; pkg_all_b += tx[i].size; }
        if (lin) { lin_all++; lin_all_b += tx[i].size; lin_alloc_all += linalloc ? linalloc[i] : 0; }
        int idx = pc_obj_find(by_obj, nobj, (uintptr_t)tx[i].pool);
        if (idx < 0) idx = pc_pool_at(ps, by_lo, np, (uintptr_t)tx[i].mem, tx[i].size ? tx[i].size : 1);
        if (idx >= 0) {
            use[idx].treg_n++; use[idx].treg_b += tx[i].size;
            if (p) use[idx].treg_pkg_n++, use[idx].treg_pkg_b += tx[i].size;
            if (lin) use[idx].treg_lin_n++;
            if (p && upf && upf[i]) {
                use[idx].up_n++; use[idx].up_b += tx[i].size;
                if (upf[i] > use[idx].last_up_frame) use[idx].last_up_frame = upf[i];
                if (upst && upst[i] > use[idx].last_up_stamp) use[idx].last_up_stamp = upst[i];
                if (upre) use[idx].reup_n += upre[i];
                int dirty = nvn_dirty_since(tx[i].mem, (size_t)tx[i].size, upst ? upst[i] : 0);   // 1 = written since the upload (or untracked: conservatively dirty)
                if (dirty) { use[idx].dirty_n++; use[idx].dirty_b += tx[i].size; }
            }
        } else { treg_un++; treg_un_b += tx[i].size; }
        if (lin && tx[i].size && (nltop < PC_TOP || tx[i].size > ltop[PC_TOP - 1].size)) {
            int k = nltop < PC_TOP ? nltop++ : PC_TOP - 1;
            for (; k > 0 && ltop[k - 1].size < tx[i].size; k--) ltop[k] = ltop[k - 1];
            ltop[k] = (pc_top){tx[i].size, linalloc ? linalloc[i] : 0, 0, (uint64_t)(uintptr_t)tx[i].pool, (uint64_t)(uintptr_t)tx[i].mem,
                               tx[i].width, tx[i].height, 1, 0, 0};
        }
    }
    (void)lin_all_b;
    // ---- totals by pool class ----
    int nres = 0, ntexpool = 0, npkgpool = 0, nupsrc = 0, lin_pools = 0, cnp_n = 0;   // cnp: CPU_NO_ACCESS pools whose registered textures are all packaged = the eviction-candidate class
    unsigned long long res_buf = 0, res_size = 0, texpool_size = 0, pkgpool_size = 0, upsrc_size = 0, upsrc_pkg_b = 0, lin_pool_buflen = 0, cnp_size = 0;
    unsigned long long cls_pool[3] = {0, 0, 0}, cls_res[3] = {0, 0, 0}, cls_pkgtex[3] = {0, 0, 0}, cls_tex[3] = {0, 0, 0};
    int cls_n[3] = {0, 0, 0}, cls_resn[3] = {0, 0, 0};
    for (int i = 0; i < np; i++) {
        int c = pool_class(ps[i].flags);
        cls_n[c]++; cls_pool[c] += ps[i].size; cls_tex[c] += use[i].treg_b; cls_pkgtex[c] += use[i].treg_pkg_b;
        if (ps[i].res) { nres++; res_size += ps[i].size; res_buf += ps[i].buflen; cls_resn[c]++; cls_res[c] += ps[i].buflen; }
        if (use[i].alias_n || use[i].treg_n) { ntexpool++; texpool_size += ps[i].size; }
        if (use[i].treg_pkg_n) { npkgpool++; pkgpool_size += ps[i].size; }
        if (use[i].treg_lin_n) { lin_pools++; lin_pool_buflen += ps[i].buflen; }
        if (pool_upload_source_only(&ps[i], &use[i], frame)) { nupsrc++; upsrc_size += ps[i].size; upsrc_pkg_b += use[i].treg_pkg_b; }
        if (c == 0 && use[i].treg_n && use[i].treg_pkg_n == use[i].treg_n) { cnp_n++; cnp_size += ps[i].size; }   // the class the step-2 reclaim work targets
    }
    // ---- report ----
    MLOG("池普查 至帧 %llu（常驻 %llu MB）：池 %d 个，请求 %llu MB / 缓冲 %llu MB，驻留集池 %d 个 缓冲 %llu MB（请求 %llu MB）；有贴图的池 %d 个 %llu MB（登记了预排布贴图的 %d 个 %llu MB、其中上传源池 %d 个 %llu MB）；CPU不可见且只含预排布贴图的池 %d 个 %llu MB",
         frame, rss_mb, np, (unsigned long long)(sum_size >> 20), (unsigned long long)(sum_buflen >> 20), nres,
         (unsigned long long)(res_buf >> 20), (unsigned long long)(res_size >> 20), ntexpool, (unsigned long long)(texpool_size >> 20),
         npkgpool, (unsigned long long)(pkgpool_size >> 20), nupsrc, (unsigned long long)(upsrc_size >> 20), cnp_n, (unsigned long long)(cnp_size >> 20));
    MLOG("  池普查贴图：登记贴图 %u 张 %llu MB（预排布 %u 张 %llu MB、线性零拷贝视图 %u 张 %llu MB）→ 归属池内 %u 张 %llu MB、未归属 %u 张 %llu MB；共用表 %u 项 NVN 存储 %llu MB / allocatedSize 合计 %llu MB（归属 %u 项、未归属 %u 项 %llu MB）；预排布贴图源合计 %llu MB，上传源池可回收候选 %llu MB（贴图戳 %llu）",
         treg_all, (unsigned long long)(treg_all_b >> 20), pkg_all, (unsigned long long)(pkg_all_b >> 20), lin_all, (unsigned long long)(lin_all_b >> 20),
         treg_all - treg_un, (unsigned long long)((treg_all_b - treg_un_b) >> 20), treg_un, (unsigned long long)(treg_un_b >> 20),
         na, (unsigned long long)(alias_all_b >> 20), (unsigned long long)(alias_alloc_all >> 20), na - alias_un, alias_un, (unsigned long long)(alias_un_b >> 20),
         (unsigned long long)(pkg_all_b >> 20), (unsigned long long)(upsrc_pkg_b >> 20), (unsigned long long)nvn_dirty_now());
    MLOG("  池普查按标志：CPU不可见 %d 个 %llu MB（驻留 %d 个 %llu MB，预排布源 %llu MB / 登记合计 %llu MB）/ CPU可见 %d 个 %llu MB（驻留 %d 个 %llu MB，预排布源 %llu MB / 登记合计 %llu MB）/ 其他 %d 个 %llu MB（驻留 %d 个 %llu MB，预排布源 %llu MB / 登记合计 %llu MB）；线性视图实际来自 %d 个池、缓冲合计 %llu MB（其 allocatedSize 合计 %llu MB 与之重复计费）",
         cls_n[0], (unsigned long long)(cls_pool[0] >> 20), cls_resn[0], (unsigned long long)(cls_res[0] >> 20), (unsigned long long)(cls_pkgtex[0] >> 20), (unsigned long long)(cls_tex[0] >> 20),
         cls_n[1], (unsigned long long)(cls_pool[1] >> 20), cls_resn[1], (unsigned long long)(cls_res[1] >> 20), (unsigned long long)(cls_pkgtex[1] >> 20), (unsigned long long)(cls_tex[1] >> 20),
         cls_n[2], (unsigned long long)(cls_pool[2] >> 20), cls_resn[2], (unsigned long long)(cls_res[2] >> 20), (unsigned long long)(cls_pkgtex[2] >> 20), (unsigned long long)(cls_tex[2] >> 20),
         lin_pools, (unsigned long long)(lin_pool_buflen >> 20), (unsigned long long)(lin_alloc_all >> 20));
    {   // per-census delta: pools created / destroyed and guest bytes since the previous print (the city batch creates ~110 pools in one frame)
        static uint64_t pc_prev_adds, pc_prev_dels, pc_prev_addb, pc_prev_delb;
        MLOG("  池普查增量（自上次）：新建 %llu 个 %llu MB，销毁 %llu 个 %llu MB，净 %+lld MB；上一次打印在此前累计 %llu 个新建 / %llu 个销毁",
             (unsigned long long)(adds - pc_prev_adds), (unsigned long long)((addb - pc_prev_addb) >> 20),
             (unsigned long long)(dels - pc_prev_dels), (unsigned long long)((delb - pc_prev_delb) >> 20),
             (long long)((addb - pc_prev_addb - (delb - pc_prev_delb)) >> 20), (unsigned long long)pc_prev_adds, (unsigned long long)pc_prev_dels);
        pc_prev_adds = adds; pc_prev_dels = dels; pc_prev_addb = addb; pc_prev_delb = delb;
    }
    // largest sharing entries / linear views with their owning pool: maps the resident "shared / pool view" textures (each reporting its parent buffer's allocatedSize)
    // to a real pool and a real NVN storage size.
    for (int k = 0; k < natop; k++) {
        int pi = pc_obj_find(by_obj, nobj, (uintptr_t)atop[k].poolobj);
        MLOG("  共用大项 #%d：NVN 存储 %llu KB，allocatedSize %llu MB，池 %s#%d %llu MB 驻留 %d，偏移 %#llx，%ux%u 级 %u 格式 %#x 目标 %u",
             k + 1, (unsigned long long)(atop[k].size >> 10), (unsigned long long)(atop[k].alloc >> 20),
             pi >= 0 ? "" : "（虚拟池，未归属）", pi, pi >= 0 ? (unsigned long long)(ps[pi].size >> 20) : 0ull, pi >= 0 ? ps[pi].res : 0,
             (unsigned long long)atop[k].off, atop[k].w, atop[k].h, atop[k].levels, atop[k].fmt, atop[k].target);
    }
    for (int k = 0; k < nltop; k++) {
        int pi = pc_obj_find(by_obj, nobj, (uintptr_t)ltop[k].poolobj);
        if (pi < 0) pi = pc_pool_at(ps, by_lo, np, ltop[k].mem, ltop[k].size);
        MLOG("  线性大项 #%d：NVN 存储 %llu KB，allocatedSize %llu MB，池 #%d %llu MB（缓冲 %llu MB）驻留 %d，%ux%u",
             k + 1, (unsigned long long)(ltop[k].size >> 10), (unsigned long long)(ltop[k].alloc >> 20),
             pi, pi >= 0 ? (unsigned long long)(ps[pi].size >> 20) : 0ull, pi >= 0 ? (unsigned long long)(ps[pi].buflen >> 20) : 0ull, pi >= 0 ? ps[pi].res : 0,
             ltop[k].w, ltop[k].h);
    }
    // pools by size: the largest 40 plus every pool >= 32 MiB (even when fewer than 40 pools exist)
    for (int r = 0; r < np; r++) {
        int i = by_size[r].idx;
        int large = ps[i].size >= (32u << 20);
        if (r >= 40 && !large) break;
        char fl[160], site[128];
        pool_flags_str(ps[i].flags, fl, sizeof fl);
        pool_site_str(ps[i].first_site, site, sizeof site);
        uint64_t mn = ps[i].hits && ps[i].min_off != ~(uint64_t)0 ? ps[i].min_off : 0, mx = ps[i].hits ? ps[i].max_off : 0;
        MLOG("  大池 #%d：%p %llu MB（缓冲 %llu MB）标志 %#x[%s] 驻留 %d 命中 %llu（末次帧 %llu）首址 帧 %llu 偏移 %#llx 经 %s %s，寻址 %llu..%llu MB；共用视图 %u 张 %llu MB；登记贴图 %u 张 %llu MB（预排布 %u 张 %llu MB，线性视图 %u 张）；上传 %u 张 %llu MB 末次帧 %llu 戳 %llu 脏 %u 张 %llu MB 重传 %llu%s",
             i, (void *)(uintptr_t)ps[i].lo, (unsigned long long)(ps[i].size >> 20), (unsigned long long)(ps[i].buflen >> 20),
             ps[i].flags, fl, ps[i].res, (unsigned long long)ps[i].hits, (unsigned long long)ps[i].last_hit_frame,
             (unsigned long long)ps[i].first_frame, (unsigned long long)ps[i].first_off, ps[i].first_kind ? "GPU地址" : "CPU地址", site,
             (unsigned long long)(mn >> 20), (unsigned long long)(mx >> 20),
             use[i].alias_n, (unsigned long long)(use[i].alias_b >> 20),
             use[i].treg_n, (unsigned long long)(use[i].treg_b >> 20), use[i].treg_pkg_n, (unsigned long long)(use[i].treg_pkg_b >> 20), use[i].treg_lin_n,
             use[i].up_n, (unsigned long long)(use[i].up_b >> 20), (unsigned long long)use[i].last_up_frame, (unsigned long long)use[i].last_up_stamp,
             use[i].dirty_n, (unsigned long long)(use[i].dirty_b >> 20), (unsigned long long)use[i].reup_n,
             pool_upload_source_only(&ps[i], &use[i], frame) ? " 上传源" : "");
    }
    // every pool currently wired through the residency set (compact): the resident-pool side of the 2.2 GiB guest pages
    if (frame >= 300 && frame % 600 == 0) {
        int shown = 0;
        for (int r = 0; r < np && shown < 256; r++) {
            int i = by_size[r].idx;
            if (!ps[i].res) continue;
            shown++;
            char fl[160]; pool_flags_str(ps[i].flags, fl, sizeof fl);
            MLOG("  驻留池 #%d：%p %llu MB（缓冲 %llu MB）标志 %#x[%s] 命中 %llu 末次帧 %llu 首帧 %llu 寻址 %llu..%llu MB 共用视图 %u 张 %llu MB 登记贴图 %u 张 %llu MB（预排布 %u 张 %llu MB，上传 %u 张 %llu MB）%s",
                 i, (void *)(uintptr_t)ps[i].lo, (unsigned long long)(ps[i].size >> 20), (unsigned long long)(ps[i].buflen >> 20), ps[i].flags, fl,
                 (unsigned long long)ps[i].hits, (unsigned long long)ps[i].last_hit_frame, (unsigned long long)ps[i].first_frame,
                 (unsigned long long)((ps[i].hits && ps[i].min_off != ~(uint64_t)0 ? ps[i].min_off : 0) >> 20), (unsigned long long)((ps[i].hits ? ps[i].max_off : 0) >> 20),
                 use[i].alias_n, (unsigned long long)(use[i].alias_b >> 20), use[i].treg_n, (unsigned long long)(use[i].treg_b >> 20),
                 use[i].treg_pkg_n, (unsigned long long)(use[i].treg_pkg_b >> 20), use[i].up_n, (unsigned long long)(use[i].up_b >> 20),
                 pool_upload_source_only(&ps[i], &use[i], frame) ? " 上传源" : "");
        }
    }
    free(ps); free(by_size); free(by_lo); free(by_obj); free(use);
    free(aks); free(aalloc); free(tx); free(pkg); free(upf); free(upst); free(upre); free(linalloc);
}
// ---------- T5: static-source reclaim core (submit thread only) ----------
// One snapshot per tick: registered textures (g_treg), memory pools (g_pools, including the per-pool upload record) and the sharing-table pool
// objects (g_alias). The eligibility scan runs outside all locks on these copies; acting re-validates the pool under g_mu.
typedef struct {
    uintptr_t mem; uint64_t size, serial; const void *pool; uint32_t flags; int32_t target, samples;
    uint8_t has_layout, pkg, cpu_written;
    uint64_t pk_stamp, pk_up_frame, pk_reups;
} rc_tsnap;
typedef struct {
    uintptr_t lo, hi; uint64_t size, buflen, hits, hits_at_upload, last_up_frame, last_hit_frame, stamp, next_eval;
    const void *obj; uint32_t flags; uint8_t state, no_reclaim, queued;
    uint64_t nontex_hits, up_bytes, up_gen, done_gen; uint8_t up_state, up_nser, up_nrg;
    uint64_t up_ser[12];
    struct { uintptr_t mem; uint64_t size, stamp; } up_rg[12];
} rc_psnap;
typedef struct {
    uintptr_t lo, hi; uint64_t hits, coverage, up_bytes, last_up_frame, last_hit_frame;
    int nreg, nup, nruns, nviews; uint8_t why, unattributed;
    struct { uintptr_t addr; uint64_t len; } runs[96];
} rc_plan;
static rc_tsnap *g_rc_ts; static int g_rc_nts, g_rc_cts;
static rc_psnap *g_rc_ps; static int g_rc_nps, g_rc_cps;
static uintptr_t *g_rc_ak; static int g_rc_nak, g_rc_cak;   // pool object addresses of sharing-table entries (sorted)
static uint64_t g_rc_next_scan, g_rc_scan_prints;
static uint64_t g_rc_frame(void) { return atomic_load(&g_cur_frame); }
static int rc_cmp_ts(const void *a, const void *b) { const rc_tsnap *x = a, *y = b; return x->mem < y->mem ? -1 : x->mem > y->mem ? 1 : 0; }
static int rc_cmp_uptr(const void *a, const void *b) { uintptr_t x = *(const uintptr_t *)a, y = *(const uintptr_t *)b; return x < y ? -1 : x > y; }
// The zero-copy views over a pool, and whether the game has stopped drawing all of them. A view the game has not drawn for a while
// belongs to a scene the player has left: the alias table keeps it alive, and with it the pool's buffer, so the Metal memory never
// comes back. The reclaim may take such a pool, dropping the views with it (the views show the pool's bytes, and the reclaim replaces
// those bytes with zeros, so the views are stale either way).
static int rc_pool_views_idle(const void *pool_obj, uint64_t idle_frames, int *nviews) {
    uintptr_t want = (uintptr_t)pool_obj;
    uint64_t now = atomic_load(&g_cur_frame);
    int n = 0, idle = 1;
    pthread_mutex_lock(&g_smu);
    for (uint32_t i = 0; i < g_alias.cap; i++) {
        if (!g_alias.vals[i]) continue;
        alias_key k; memcpy(&k, g_alias.keys + (size_t)i * g_alias.klen, sizeof k);
        if (k.pool != (uint64_t)want) continue;
        n++;
        id o = (__bridge id)g_alias.vals[i];
        uint64_t lu = view_last_use_obj(o);
        if (lu && now - lu < idle_frames) idle = 0;   // drawn recently: the game may still use it
    }
    pthread_mutex_unlock(&g_smu);
    if (nviews) *nviews = n;
    return idle;
}
static int rc_ak_has(uintptr_t obj) { int lo = 0, hi = g_rc_nak - 1; while (lo <= hi) { int m = (lo + hi) / 2; if (g_rc_ak[m] < obj) lo = m + 1; else if (g_rc_ak[m] > obj) hi = m - 1; else return 1; } return 0; }
static int rc_cmp_iv(const void *a, const void *b) { const uintptr_t *x = a, *y = b; return x[0] < y[0] ? -1 : x[0] > y[0] ? 1 : 0; }
static int rc_desc_static(const rc_tsnap *t) {   // can this texture's source storage be dead after its upload?
    if (!t->has_layout || t->cpu_written) return 0;
    // The IMAGE storage-write flag is NOT a disqualifier for a packaged texture: Hades II sets it on imported ASTC assets that have a packaged
    // layout and are only ever sampled (measured in the TEXINFO census: 4096x4096 ASTC 4x4, flags 0x4, has_layout). GPU writes go to the
    // private copy and never change the source bytes, so they do not affect reclaiming the source. For non-packaged (zero-copy) textures the
    // flag means real shader writes, but those are excluded by the has_layout check above anyway.
    if ((t->flags & NVN_TEXTURE_FLAGS_IMAGE) && !t->has_layout) return 0;
    if (t->flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET | NVN_TEXTURE_FLAGS_DISPLAY |
                    NVN_TEXTURE_FLAGS_VIDEO_DECODE | NVN_TEXTURE_FLAGS_SPARSE | NVN_TEXTURE_FLAGS_ADAPTIVE_ZCULL | NVN_TEXTURE_FLAGS_MINIMAL_LAYOUT)) return 0;
    if (t->samples > 1) return 0;
    switch (t->target) { case 0: case 1: case 3: case 4: case 8: case 9: return 1; default: return 0; }
}
// Refresh the per-tick snapshots. Called on the submit thread; each table is read under its own lock (never nested: the established g_rmu -> g_smu order is kept).
static int rc_snapshot(void) {
    if (pthread_mutex_trylock(&g_rmu)) return 0;   // a texture upload may be inside tex_by_id right now (g_rmu held): retry at the next safe point
    int need = 0;
    for (int i = 0; i < g_ntreg; i++) if (g_treg[i].ok && g_treg[i].t.mem) need++;
    if (need > g_rc_cts) { int nc = need + 256; rc_tsnap *n = realloc(g_rc_ts, (size_t)nc * sizeof *n); if (n) { g_rc_ts = n; g_rc_cts = nc; } }
    g_rc_nts = 0;
    if (g_rc_ts) for (int i = 0; i < g_ntreg; i++) {
        const treg_t *r = &g_treg[i];
        if (!r->ok || !r->t.mem || r->t.magic != MAGIC('T', 'E', 'X', 0)) continue;
        if (g_rc_nts >= g_rc_cts) break;
        rc_tsnap *o = &g_rc_ts[g_rc_nts++];
        o->mem = (uintptr_t)r->t.mem; o->size = r->t.L.size; o->serial = r->t.serial; o->pool = r->t.pool;
        o->flags = (uint32_t)r->t.d.flags; o->target = (int32_t)r->t.d.target; o->samples = (int32_t)r->t.d.samples;
        o->has_layout = r->t.has_layout; o->pkg = 0; o->cpu_written = 0; o->pk_stamp = o->pk_up_frame = o->pk_reups = 0;
    }
    pthread_mutex_unlock(&g_rmu);
    if (g_rc_ts) qsort(g_rc_ts, (size_t)g_rc_nts, sizeof *g_rc_ts, rc_cmp_ts);
    pthread_mutex_lock(&g_smu);
    for (int i = 0; i < g_rc_nts; i++) {
        void *v = nvn_hm_get(&g_by_serial, &g_rc_ts[i].serial);
        if (!v) continue;
        id o = (__bridge id)v;
        if ([o isKindOfClass:[NvPkg class]]) { NvPkg *pk = (NvPkg *)o; g_rc_ts[i].pkg = 1; g_rc_ts[i].pk_stamp = pk->stamp; g_rc_ts[i].pk_up_frame = pk->up_frame; g_rc_ts[i].pk_reups = pk->reuploads; }
    }
    {   // sharing-table pool objects: any zero-copy view on a pool disqualifies it
        uint32_t cap = g_alias.cap;
        if ((int)cap > g_rc_cak) { int nc = (int)cap + 64; uintptr_t *n = realloc(g_rc_ak, (size_t)nc * sizeof *n); if (n) { g_rc_ak = n; g_rc_cak = nc; } }
        g_rc_nak = 0;
        if (g_rc_ak) for (uint32_t i = 0; i < cap; i++) if (g_alias.vals[i] && g_rc_nak < g_rc_cak) {
            alias_key k; memcpy(&k, g_alias.keys + (size_t)i * g_alias.klen, sizeof k);
            g_rc_ak[g_rc_nak++] = (uintptr_t)k.pool;
        }
    }
    pthread_mutex_unlock(&g_smu);
    if (g_rc_ak) qsort(g_rc_ak, (size_t)g_rc_nak, sizeof *g_rc_ak, rc_cmp_uptr);
    pthread_mutex_lock(&g_cpu_tex_mu);
    for (int i = 0; i < g_rc_nts; i++) if (g_cpu_tex && CFSetContainsValue(g_cpu_tex, (const void *)(uintptr_t)g_rc_ts[i].serial)) g_rc_ts[i].cpu_written = 1;
    pthread_mutex_unlock(&g_cpu_tex_mu);
    pthread_mutex_lock(&g_mu);
    if (g_npool > g_rc_cps) { int nc = g_npool + 64; rc_psnap *n = realloc(g_rc_ps, (size_t)nc * sizeof *n); if (n) { g_rc_ps = n; g_rc_cps = nc; } }
    g_rc_nps = 0;
    if (g_rc_ps) for (int i = 0; i < g_npool; i++) {
        const pool_ent *p = &g_pools[i];
        rc_psnap *o = &g_rc_ps[g_rc_nps++];
        memset(o, 0, sizeof *o);
        o->lo = p->lo; o->hi = p->hi; o->size = p->size; o->buflen = p->buflen; o->hits = p->hits; o->hits_at_upload = p->rc_hits_at_upload;
        o->last_up_frame = p->rc_last_upload_frame; o->last_hit_frame = p->last_hit_frame; o->stamp = p->rc_stamp; o->next_eval = p->rc_next_eval;
        o->obj = p->obj; o->flags = p->flags; o->state = p->rc_state; o->no_reclaim = p->rc_no_reclaim; o->queued = p->rc_queued;
        o->nontex_hits = p->rc_nontex_hits; o->up_bytes = p->rc_up_bytes; o->up_state = p->rc_up_state; o->up_nser = p->rc_up_nser; o->up_nrg = p->rc_up_nrg;
        o->up_gen = p->rc_up_gen; o->done_gen = p->rc_done_gen;
        memcpy(o->up_ser, p->rc_up_ser, sizeof o->up_ser); memcpy(o->up_rg, p->rc_up_rg, sizeof o->up_rg);
    }
    pthread_mutex_unlock(&g_mu);
    return 1;
}
// Count a pool once, at its first rejection. The cumulative counter is about pools, not evaluations: a pool that is rejected
// again on a later tick (the same reason, or another one) must not be counted twice. Returns 1 when this call is the first rejection
// of this pool, 0 when the pool was already rejected before. The flag lives in the pool (rc_rejected), so a pool that is freed and
// a new pool that reuses the address starts over.
static int rc_pool_count_reject(uintptr_t lo, uintptr_t hi, uint8_t why) {
    int first = 0;
    pthread_mutex_lock(&g_mu);
    for (int i = 0; i < g_npool; i++) if (g_pools[i].lo == lo && g_pools[i].hi == hi) {
        if (!g_pools[i].rc_rejected) { g_pools[i].rc_rejected = 1; g_pools[i].rc_why = why; first = 1; }
        break;
    }
    pthread_mutex_unlock(&g_mu);
    return first;
}
// Update one pool's reclaim bookkeeping after an evaluation. Under g_mu, matched by storage range (indices move when a pool is finalized).
static void rc_pool_update(uintptr_t lo, uintptr_t hi, int set_no_reclaim, int set_queued, uint64_t next_eval, uint8_t why) {
    pthread_mutex_lock(&g_mu);
    for (int i = 0; i < g_npool; i++) if (g_pools[i].lo == lo && g_pools[i].hi == hi) {
        if (set_no_reclaim) g_pools[i].rc_no_reclaim = 1;
        g_pools[i].rc_queued = (uint8_t)set_queued;
        if (set_queued && g_pools[i].rc_state == RC_NONE) g_pools[i].rc_state = RC_WAIT;
        g_pools[i].rc_next_eval = next_eval;
        if (why) g_pools[i].rc_why = why;
        break;
    }
    pthread_mutex_unlock(&g_mu);
}
// Eligibility of one pool; fills the plan (coverage, page runs) when it returns RC_R_NONE. No lock held (works on the tick snapshots).
static uint8_t rc_eval_pool(const rc_psnap *p, rc_plan *pl) {
    memset(pl, 0, sizeof *pl); pl->lo = p->lo; pl->hi = p->hi; pl->hits = p->hits; pl->last_up_frame = p->last_up_frame; pl->last_hit_frame = p->last_hit_frame;
    uintptr_t pg = (uintptr_t)getpagesize(), rlo = (p->lo + pg - 1) & ~(pg - 1), rhi = p->hi & ~(pg - 1);
    uint64_t minb = (uint64_t)g_rc_min_mb << 20;
    uint64_t fr = g_rc_frame();
    if (p->state == RC_RECLAIMED) return RC_R_NONE;   // handled by the invalidation watch
    if (rhi <= rlo || rhi - rlo < minb) return RC_R_ALIGN;
    {   // heap-region only: alias / stack pages belong to the host's mapping bookkeeping and must never be replaced
        uint8_t *hb = NULL; size_t hs = 0;
        if (!dm_guest_heap_range(&hb, &hs) || p->lo < (uintptr_t)hb || p->hi > (uintptr_t)hb + hs) return RC_R_HEAP;
    }
    for (int q = 0; q < g_rc_nps; q++) {   // never touch a range that overlaps another live pool
        const rc_psnap *o = &g_rc_ps[q];
        if (o->lo == p->lo && o->hi == p->hi) continue;
        if (o->lo < rhi && rlo < o->hi) return RC_R_OVERLAP;
    }
    if (p->up_gen != p->done_gen) return RC_R_NOTREADY;         // a source upload batch for this pool has not completed yet (its GPU deswizzle may still read the pool)
    if (p->hits != p->hits_at_upload) return RC_R_HIT;          // an address use after the source upload: not a static source any more
    if (p->nontex_hits) return RC_R_HIT;                        // the pool is also used as data (buffer bind / copy / share view), not only as upload source
    if (p->up_state & 1) return RC_R_OBJ;                       // a dynamic descriptor was uploaded from it
    if (p->up_state & 2) return RC_R_REUPLOAD;                  // a texture re-uploaded through it: content streamed through the pool
    if (!p->up_nrg && !p->up_nser) return RC_R_OBJ;             // no upload evidence at all
    if (fr < p->last_up_frame + (uint64_t)g_rc_grace) return RC_R_NOTREADY;
    uintptr_t iv[128][2]; int niv = 0, nup = 0;
    for (int k = 0; k < (int)p->up_nrg && niv < 128; k++) {     // upload-record ranges: the bytes that were read to build private copies
        uintptr_t m = p->up_rg[k].mem; uint64_t sz = p->up_rg[k].size;
        if (!sz) continue;
        nup++;
        if (!nvn_dirty_tracked((const void *)m, (size_t)sz)) return RC_R_UNTRACKED;
        if (nvn_dirty_since((const void *)m, (size_t)sz, p->up_rg[k].stamp)) return RC_R_DIRTY;
        uintptr_t a = m > rlo ? m : rlo, b = m + sz; if (b > rhi) b = rhi;
        if (b > a) { iv[niv][0] = a; iv[niv][1] = b; niv++; }
    }
    for (int k = 0; k < (int)p->up_nser; k++) if (rc_write_used(p->up_ser[k])) return RC_R_WRITEUSE;
    // Registered textures on the pool are the completeness superset: every one of them must be static, uploaded and clean (up_n == treg_pkg_n).
    int nreg = 0;
    int lo = 0, hi = g_rc_nts - 1, first = g_rc_nts;
    while (lo <= hi) { int m = (lo + hi) / 2; if (g_rc_ts[m].mem < p->lo) lo = m + 1; else { first = m; hi = m - 1; } }
    for (int i = first; i < g_rc_nts && g_rc_ts[i].mem < p->hi; i++) {
        const rc_tsnap *t = &g_rc_ts[i];
        nreg++;
        if (!rc_desc_static(t)) return RC_R_OBJ;
        if (!t->pkg) return RC_R_NOTREADY;                     // not uploaded yet: retry after the next batch
        if (t->pk_reups) return RC_R_REUPLOAD;
        if (!nvn_dirty_tracked((const void *)t->mem, (size_t)(t->size ? t->size : 1))) return RC_R_UNTRACKED;
        if (nvn_dirty_since((const void *)t->mem, (size_t)(t->size ? t->size : 1), t->pk_stamp)) return RC_R_DIRTY;
        if (rc_write_used(t->serial)) return RC_R_WRITEUSE;
        uintptr_t a = t->mem > rlo ? t->mem : rlo, b = t->mem + (t->size ? t->size : 1); if (b > rhi) b = rhi;
        if (b > a && niv < 128) { iv[niv][0] = a; iv[niv][1] = b; niv++; }
    }
    pl->nreg = nreg; pl->nup = nup; pl->up_bytes = p->up_bytes; pl->unattributed = (uint8_t)(nreg == 0);
    if (!niv) return RC_R_COVER;
    {   // DM_RC_VIEW_IDLE=<frames>: how long a zero-copy view must go unused before the reclaim may take the pool behind it.
        static int vs = -1;
        if (vs < 0) { const char *e = getenv("DM_RC_VIEW_IDLE"); vs = e ? atoi(e) : g_rc_view_idle; if (vs < 0) vs = 0; }
        g_rc_view_idle = vs;
    }
    if (!p->obj) return RC_R_ALIAS;   // no pool object: nothing to key the views on
    if (rc_ak_has((uintptr_t)p->obj)) {
        // The pool has zero-copy views. If the game has stopped drawing all of them, the pool can be taken and the views dropped
        // with it (the views show the pool's bytes, and the reclaim replaces those bytes with zeros, so the views are stale anyway).
        int nv = 0;
        if (!rc_pool_views_idle(p->obj, (uint64_t)g_rc_view_idle, &nv)) return RC_R_ALIAS;   // the game still draws one of them
        pl->nviews = nv;
    }
    // Merge the ranges; only pages fully covered by the union are ever replaced (boundary pages and holes stay resident).
    qsort(iv, (size_t)niv, 2 * sizeof(uintptr_t), rc_cmp_iv);
    uint64_t cov = 0; int nr = 0;
    for (int i = 0; i < niv;) {
        uintptr_t a = iv[i][0], b = iv[i][1]; int j = i + 1;
        while (j < niv && iv[j][0] <= b) { if (iv[j][1] > b) b = iv[j][1]; j++; }
        cov += (uint64_t)(b - a);
        uintptr_t pa = (a + pg - 1) & ~(pg - 1), pb = b & ~(pg - 1);
        if (pb > pa && nr < 96) { pl->runs[nr].addr = pa; pl->runs[nr].len = (uint64_t)(pb - pa); pl->coverage += (uint64_t)(pb - pa); nr++; }
        i = j;
    }
    pl->nruns = nr;
    if (cov * 100 < (uint64_t)g_rc_purity * p->size) return RC_R_COVER;   // purity: the pool is (nearly) all uploaded texture-source storage
    if (!nr || pl->coverage < minb) return RC_R_COVER;
    return RC_R_NONE;
}
// Perform one reclaim. Runs on the submit thread: the upload command buffers that read this pool already completed (up_wait) and no address use
// was recorded since; any later use goes through pool_res -> rc_unreclaim_pool. Under g_mu so a concurrent pool_res cannot recreate the buffer
// while the pages are being replaced.
static void rc_act(const rc_plan *pl) {
    pthread_mutex_lock(&g_mu);
    int i = -1;
    for (int k = 0; k < g_npool; k++) if (g_pools[k].lo == pl->lo && g_pools[k].hi == pl->hi) { i = k; break; }
    if (i < 0 || (g_pools[i].rc_state != RC_WAIT && g_pools[i].rc_state != RC_NONE) || g_pools[i].hits != pl->hits || g_pools[i].rc_no_reclaim) { pthread_mutex_unlock(&g_mu); return; }
    if ((int)atomic_load_explicit(&g_rc_rg_n, memory_order_relaxed) >= RC_RG_MAX) {   // no room to remember the range: do not reclaim (needed by the safety checks)
        // The range table is full for now, not because this pool is bad: the pool stays a candidate and is retried later (the table drains as ranges are tombstoned).
        static _Atomic int nfull; if (!atomic_fetch_add(&nfull, 1)) MLOG("【红】静态源回收：已回收区间表满（%d 条，里面还有活条目时不会再回收），本次不回收；调大 RC_RG_MAX 或做整理", RC_RG_MAX);
        g_rc_block[RC_R_BUDGET]++;
        g_pools[i].rc_queued = 1; g_pools[i].rc_next_eval = g_rc_frame() + 30;
        pthread_mutex_unlock(&g_mu);
        return;
    }
    pool_ent *p = &g_pools[i];
    p->rc_state = RC_EVICTING;
    // The zero-copy views over this pool: the game has stopped drawing them, so they are dropped with the pool. They stay in the
    // residency set otherwise, and with them the pool's buffer, so the Metal memory never comes back.
    if (pl->nviews > 0) {
        int dropped = mtl_alias_drop_pool(p->obj);
        if (dropped) g_rc_views_dropped += (uint64_t)dropped;
    }
    id<MTLBuffer> b = p->buf;
    if (p->res) { res_remove(b); res_commit(); p->res = 0; g_pool_res_n--; }
    [g_keep removeObject:b];
    p->buf = nil; p->gpu = 0;
    atomic_fetch_add_explicit(&g_pool_gen, 1, memory_order_relaxed);   // the per-thread pool caches hold this pool's buffer: they are stale now
    b = nil;   // release the driver's view of the pages before the mapping is replaced (a kept no-copy buffer keeps a stale GPU mapping)
    uint64_t done = 0, psize = p->size;
    for (int r = 0; r < pl->nruns; r++) {
        void *m = mmap((void *)pl->runs[r].addr, (size_t)pl->runs[r].len, PROT_READ, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
        if (m == MAP_FAILED) { RED_ONCE(90, "静态源回收：替换池 %p 的页失败（%#llx+%#llx，errno=%d）", (void *)(uintptr_t)p->lo, (unsigned long long)pl->runs[r].addr, (unsigned long long)pl->runs[r].len, errno); continue; }
        done += pl->runs[r].len;
    }
    if (!done) {   // nothing was replaced: the source is still valid, so do not mark it reclaimed; restore the buffer and retry much later
        p->rc_state = RC_WAIT; p->rc_queued = 0; p->rc_next_eval = g_rc_frame() + 300;
        if (!rc_buf_restore_locked(i)) {
            p->rc_no_reclaim = 1;
            MLOG("【红】静态源回收：池 %p 的页替换全部失败，零拷贝缓冲也重建失败，按不可回收处理", (void *)(uintptr_t)pl->lo);
        } else MLOG("【红】静态源回收：池 %p 的页替换全部失败，缓冲与驻留已恢复，保留为候选稍后重试", (void *)(uintptr_t)pl->lo);
        pthread_mutex_unlock(&g_mu);
        return;
    }
    // At least one page run was replaced: the source is no longer reliable, publish the reclaim and the range (checked by the texture paths).
    p->rc_state = RC_RECLAIMED; p->rc_stamp = nvn_dirty_now(); p->rc_queued = 0; p->rc_no_reclaim = 0; p->rc_why = RC_R_NONE;
    uint64_t slot = atomic_load_explicit(&g_rc_rg_n, memory_order_relaxed);
    atomic_store_explicit(&g_rc_rg[slot].lo, (uintptr_t)p->lo, memory_order_relaxed);
    atomic_store_explicit(&g_rc_rg[slot].hi, (uintptr_t)p->hi, memory_order_relaxed);
    atomic_store_explicit(&g_rc_rg_n, slot + 1, memory_order_release);
    nvn_dirty_protect((const void *)p->lo, (size_t)p->size);   // fresh pages are mapped read-only; sync any tracker page that was not in PROT state
    p->rc_bytes = done;
    g_rc_pools++; g_rc_bytes += done; atomic_fetch_add(&g_rc_live_bytes, done);
    uint64_t np = g_rc_pools, nb = g_rc_bytes, frame = g_rc_frame();
    pthread_mutex_unlock(&g_mu);
    if (g_rc_events++ < 40)
        MLOG("静态源回收：池 #%d %p %llu MB → 移除驻留并回收 %llu MB（覆盖 %llu MB，登记贴图 %d 张 / 上传记录 %d 段 %llu MB，帧 %llu；累计 %llu 池 %llu MB）",
             i, (void *)(uintptr_t)pl->lo, (unsigned long long)(psize >> 20), (unsigned long long)(done >> 20),
             (unsigned long long)(pl->coverage >> 20), pl->nreg, pl->nup, (unsigned long long)(pl->up_bytes >> 20), (unsigned long long)frame,
             (unsigned long long)np, (unsigned long long)(nb >> 20));
}
// Invalidate a reclaimed pool when a guest write into its source range is seen after the reclaim: the static proof no longer holds.
static int rc_check_invalid(const rc_psnap *p) {
    int dirty = 0;
    for (int k = 0; k < (int)p->up_nrg && !dirty; k++)
        if (p->up_rg[k].size && nvn_dirty_tracked((const void *)p->up_rg[k].mem, (size_t)p->up_rg[k].size) && nvn_dirty_since((const void *)p->up_rg[k].mem, (size_t)p->up_rg[k].size, p->stamp)) dirty = 1;
    for (int i = 0; i < g_rc_nts && !dirty; i++) {
        const rc_tsnap *t = &g_rc_ts[i];
        if (t->mem < p->lo || t->mem >= p->hi || !t->size) continue;
        if (!nvn_dirty_tracked((const void *)t->mem, (size_t)t->size)) continue;
        if (nvn_dirty_since((const void *)t->mem, (size_t)t->size, p->stamp)) dirty = 1;
    }
    if (!dirty) return 0;
    pthread_mutex_lock(&g_mu);
    for (int i = 0; i < g_npool; i++) if (g_pools[i].lo == p->lo && g_pools[i].hi == p->hi && !g_pools[i].rc_no_reclaim) {
        g_pools[i].rc_no_reclaim = 1; g_rc_invalid++;
        MLOG("【红】静态源回收：已回收池 %p（%llu MB）的源内存被写入，不再回收该池（累计失效 %llu；该池纹理的源已零页，若再上传会从新内容走 CPU 路径）",
             (void *)(uintptr_t)p->lo, (unsigned long long)(p->size >> 20), (unsigned long long)g_rc_invalid);
        pthread_mutex_unlock(&g_mu);
        return 1;
    }
    pthread_mutex_unlock(&g_mu);
    return 0;
}
// Cheap gate for per-pass safe points in the replay: only enter the expensive tick when a finished upload batch is waiting.
static void rc_reclaim_maybe(void) {
    if ((!g_tex_reclaim && !g_tex_reclaim_dry) || !atomic_load_explicit(&g_rc_tick_wanted, memory_order_relaxed)) return;
    rc_reclaim_tick();
}
// Submit-thread tick: drain finished upload batches into "queued" pools, evaluate, reclaim (or only report in dry-run mode).
static void rc_reclaim_tick(void) {
    if (!g_tex_reclaim && !g_tex_reclaim_dry) return;
    uint64_t t0 = clock_gettime_nsec_np(CLOCK_MONOTONIC), fr = g_rc_frame();
    pthread_mutex_lock(&g_rc_mu);
    int npend = g_rc_npend; rc_src *pend = g_rc_pending; g_rc_pending = NULL; g_rc_npend = g_rc_capp = 0;
    pthread_mutex_unlock(&g_rc_mu);
    if (!npend && fr < g_rc_next_scan) { free(pend); return; }
    if (!npend) {   // steady-state re-evaluation cadence; a finished batch is consumed immediately
        static uint64_t last_ns;
        uint64_t now = clock_gettime_nsec_np(CLOCK_MONOTONIC);
        if (last_ns && now - last_ns < 1000000ull) { free(pend); return; }   // per-pass safe points call this: at most ~1 tick/ms
        last_ns = now;
    }
    // The re-evaluation cadence: with a backlog of candidates the steady state is not enough. The loading phase queues 260 pools in
    // 18 frames, and a 30-frame cadence (0.5 s at 60 fps) with a 16-pool cap can only retire 32 pools a second: the peak resident
    // forms before the reclaim catches up (measured: the peak 2713 MB at frame 803, the gate 2652 MB). The cadence therefore follows the
    // backlog: a big backlog is scanned every frame, and the per-tick pool cap is raised to match (the cap still bounds the work per tick).
    uint32_t backlog = 0;
    { pthread_mutex_lock(&g_mu); for (int i = 0; i < g_npool; i++) if (g_pools[i].rc_state == RC_WAIT) backlog++; pthread_mutex_unlock(&g_mu); }
    g_rc_next_scan = fr + (backlog > 64 ? 1 : backlog > 16 ? 4 : 30);
    g_rc_max_pools_eff = backlog > 64 ? 64 : backlog > 16 ? 32 : g_rc_max_pools;
    g_rc_max_mb_eff = backlog > 64 ? 512u << 20 : backlog > 16 ? 256u << 20 : (uint64_t)g_rc_max_mb << 20;
    if (npend) {   // a finished upload batch: mark its pools for evaluation (the upload record is already in the pool table)
        pthread_mutex_lock(&g_mu);
        for (int k = 0; k < npend; k++)
            for (int i = 0; i < g_npool; i++) if (g_pools[i].lo == pend[k].lo && g_pools[i].size == pend[k].size) {
                if (pend[k].gen > g_pools[i].rc_done_gen) g_pools[i].rc_done_gen = pend[k].gen;   // this batch's upload of the pool completed on the GPU
                g_pools[i].rc_queued = 1; g_pools[i].rc_next_eval = 0; if (g_pools[i].rc_state == RC_NONE) g_pools[i].rc_state = RC_WAIT; break;
            }
        pthread_mutex_unlock(&g_mu);
    }
    free(pend);
    if (!rc_snapshot()) {   // g_rmu busy (a texture upload inside tex_by_id): retry at the next safe point; the pool table keeps rc_queued
        atomic_store_explicit(&g_rc_tick_wanted, 1, memory_order_relaxed);
        g_rc_next_scan = fr;
        return;
    }
    atomic_store_explicit(&g_rc_tick_wanted, 0, memory_order_relaxed);
    rc_plan plans[8]; int nplans = 0;
    int neval = 0, n_now = 0, n_steady = 0, n_unattr = 0, nreclaim = 0;
    uint64_t eval_b = 0, b_now = 0, b_steady = 0, reclaim_b = 0;
    uint64_t block[RC_R_N]; memset(block, 0, sizeof block);
    for (int pi = 0; pi < g_rc_nps; pi++) {
        const rc_psnap *p = &g_rc_ps[pi];
        if (p->state == RC_RECLAIMED) { rc_check_invalid(p); continue; }
        if (!p->queued || p->no_reclaim || fr < p->next_eval) continue;
        neval++; eval_b += p->size;
        rc_plan pl; uint8_t why = rc_eval_pool(p, &pl);
        if (why == RC_R_NONE) {
            int in_batch = fr <= p->last_up_frame + 2;
            if (in_batch) { n_now++; b_now += pl.coverage; g_rc_acc_now_pools++; g_rc_acc_now_bytes += pl.coverage; } else { n_steady++; b_steady += pl.coverage; }
            if (pl.unattributed) n_unattr++;
            if (g_tex_reclaim_dry) {
                if (nplans < 8) plans[nplans++] = pl;
                rc_pool_update(p->lo, p->hi, 0, 1, fr + 60, RC_R_NONE);
            } else if (atomic_load(&g_rc_live_bytes) + pl.coverage <= (g_rc_budget_mb << 20) && nreclaim < g_rc_max_pools_eff && reclaim_b + pl.coverage <= g_rc_max_mb_eff) {
                rc_act(&pl); nreclaim++; reclaim_b += pl.coverage;
            } else why = RC_R_BUDGET;
        }
        if (why == RC_R_NOTREADY) rc_pool_update(p->lo, p->hi, 0, 1, fr + 10, why);
        else if (why == RC_R_BUDGET) {
            // The budget is full (or this tick's per-tick cap is reached): the pool is still a good candidate, it just has to wait. It must NOT be marked
            // rc_no_reclaim, or it would never be evaluated again and its source pages would stay resident for the rest of the level.
            block[why]++;
            g_rc_block[why]++;   // the budget is a shared limit, so this counts the events, not the pools
            rc_pool_update(p->lo, p->hi, 0, 1, fr + 10, why);
        }
        else if (why != RC_R_NONE) {
            block[why]++;
            // A hard rejection (rc_no_reclaim) is permanent, so the pool is counted once, at its first rejection.
            int first = rc_pool_count_reject(p->lo, p->hi, why);
            if (first) g_rc_block[why]++;
            rc_pool_update(p->lo, p->hi, 1, 0, 0, why);
        }
    }
    uint64_t dt = clock_gettime_nsec_np(CLOCK_MONOTONIC) - t0;
    g_rc_ticks++; g_rc_tick_ns += dt;
    if (g_tex_reclaim_dry && n_now && g_rc_batch_prints < 16) {   // immediate: the process may die inside this frame
        g_rc_batch_prints++;
        MLOG("静态源回收（试运行）批次内可回收：%d 池 %llu MB（本 tick，至帧 %llu，累计 %llu 池 %llu MB）", n_now, (unsigned long long)(b_now >> 20),
             (unsigned long long)fr, (unsigned long long)g_rc_acc_now_pools, (unsigned long long)(g_rc_acc_now_bytes >> 20));
    }
    if ((g_tex_reclaim_dry || g_rc_events || (g_tex_reclaim && neval)) && (nplans || neval) && fr >= g_rc_last_print_frame + 30 && g_rc_scan_prints < 24) {
        g_rc_scan_prints++; g_rc_last_print_frame = fr;
        MLOG("静态源回收（%s）至帧 %llu：自上次打印的批次内可回收 %llu 池 %llu MB；静默后候选 %d 池 %llu MB（其中未登记源池 %d）；本次评估 %d 池 %llu MB；累计候选 %llu 池 %llu MB；随池丢弃的零拷贝视图 %llu 个。预算 在用 %llu/%llu MB，区间表 %llu/%d。累计阻塞（硬拒绝按池计，每个池只记一次；预算按次计）：命中 %llu 脏 %llu 覆盖 %llu 共用视图 %llu 重叠 %llu 未上传 %llu 堆外 %llu 写用 %llu 未跟踪 %llu 重传 %llu 对象 %llu 预算 %llu；本 tick 阻塞 %llu；本 tick %.2f ms",
             g_tex_reclaim_dry ? "试运行" : "开", (unsigned long long)fr, (unsigned long long)g_rc_acc_now_pools, (unsigned long long)(g_rc_acc_now_bytes >> 20), n_steady, (unsigned long long)(b_steady >> 20), n_unattr,
             neval, (unsigned long long)(eval_b >> 20), (unsigned long long)g_rc_cand_pools, (unsigned long long)(g_rc_cand_bytes >> 20), (unsigned long long)g_rc_views_dropped,
             (unsigned long long)(atomic_load(&g_rc_live_bytes) >> 20), (unsigned long long)g_rc_budget_mb,
             (unsigned long long)atomic_load(&g_rc_rg_n), RC_RG_MAX,
             (unsigned long long)g_rc_block[RC_R_HIT], (unsigned long long)g_rc_block[RC_R_DIRTY], (unsigned long long)g_rc_block[RC_R_COVER],
             (unsigned long long)g_rc_block[RC_R_ALIAS], (unsigned long long)g_rc_block[RC_R_OVERLAP], (unsigned long long)g_rc_block[RC_R_NOTREADY],
             (unsigned long long)g_rc_block[RC_R_HEAP], (unsigned long long)g_rc_block[RC_R_WRITEUSE], (unsigned long long)g_rc_block[RC_R_UNTRACKED],
             (unsigned long long)g_rc_block[RC_R_REUPLOAD], (unsigned long long)g_rc_block[RC_R_OBJ], (unsigned long long)g_rc_block[RC_R_BUDGET],
             (unsigned long long)(block[RC_R_HIT] + block[RC_R_DIRTY] + block[RC_R_COVER] + block[RC_R_ALIAS] + block[RC_R_OVERLAP] + block[RC_R_NOTREADY] + block[RC_R_HEAP] + block[RC_R_WRITEUSE] + block[RC_R_UNTRACKED] + block[RC_R_REUPLOAD] + block[RC_R_OBJ] + block[RC_R_BUDGET]), (double)dt / 1e6);
        g_rc_acc_now_pools = g_rc_acc_now_bytes = 0;
    }
    for (int k = 0, shown = 0; k < nplans && shown < 10; k++) {
        const rc_plan *pl = &plans[k];
        uint64_t span = pl->hi - pl->lo;
        uint64_t h = (uint64_t)pl->lo * 0x9E3779B97F4A7C15ull ^ (pl->last_up_frame + 1) * 0xC2B2AE3D27D4EB4Full;
        int seen = 0; for (int i = 0; i < g_rc_log_n; i++) if (g_rc_log_seen[i] == h) { seen = 1; break; }
        if (seen) continue;
        if (g_rc_log_n < 256) g_rc_log_seen[g_rc_log_n++] = h;
        shown++;
        MLOG("  静态源候选 #%d：%p %llu MB %s 覆盖 %llu%%（页 %llu MB）登记贴图 %d 张 / 上传记录 %d 段 %llu MB 末次上传帧 %llu 末次命中帧 %llu → 可回收 %llu MB%s",
             k + 1, (void *)(uintptr_t)pl->lo, (unsigned long long)(span >> 20), (g_rc_frame() <= pl->last_up_frame + 2 ? "本帧批次" : "静默"),
             (unsigned long long)(pl->coverage * 100 / (span ? span : 1)), (unsigned long long)(pl->coverage >> 20), pl->nreg, pl->nup,
             (unsigned long long)(pl->up_bytes >> 20), (unsigned long long)pl->last_up_frame, (unsigned long long)pl->last_hit_frame,
             (unsigned long long)(pl->coverage >> 20), pl->unattributed ? "（未登记源池）" : "");
    }
}
// Low-memory guard -> static-source reclaim: ask the next submit-thread safe point to evaluate reclaim again. Called from the frame hook's low-memory poll
// (guest thread), so it only sets the atomic flag the submit thread already watches; no-op when the reclaim switches are off.
void mtl_tex_reclaim_hint(void) { if (g_tex_reclaim || g_tex_reclaim_dry) atomic_store_explicit(&g_rc_tick_wanted, 1, memory_order_relaxed); }
// Lock-free copy of texture ID → serial (the replay hot path only needs the serial: store lookahead index, binding set of the clear-only hoist): texture IDs are 20 bits, fixed 1M entries (BSS, only touched pages are used),
// release write at registration, relaxed atomic read on the reader side, without entering g_rmu (the former h_ser locked and unlocked on every lookup, thousands of times per frame in the camp)
static _Atomic uint64_t g_tser[1 << 20];
static samp_par *g_sreg; static uint8_t *g_sreg_ok; static void **g_sview; static uint64_t *g_srid; static int g_nsreg;   // g_sview: sampler ID → MTLSamplerState (retained)
// Memory pool Finalize: for texture IDs whose registered texture lives on this pool, drop the cached MTLTexture (view or base texture). The cache used to be released only when the ID was registered again,
// so a linear texture built directly on a pool kept the whole pool buffer alive (6 to 17 finalized 16 MB pool buffers stayed alive in Engage's title demo). If the ID is still used, the next lookup rebuilds it on the slow path
static void treg_drop_ar(treg_t *r) { if (r->ar) { defer_obj_unref((__bridge id)r->ar); CFRelease(r->ar); r->ar = r->ar_of = NULL; r->ar_rid = 0; } }
// alt is a texture view BUILT FROM the cached view (tex_fit, when the shader declares a type the texture does not have). A view keeps its
// parent texture alive, so an alt that outlives the cached view keeps that view alive, and through it the whole pool buffer behind it --
// dropping only `view` on Finalize left the pool pinned by an alt nobody would ask for again (the next tex_fit compares alt_of with the
// current view and rebuilds). Same for `ar` (tex_alpha_red), which is built from the base texture.
static void treg_drop_alt(treg_t *r) { if (r->alt) { defer_obj_unref((__bridge id)r->alt); CFRelease(r->alt); r->alt = r->alt_of = NULL; r->alt_rid = 0; } }
static void treg_drop_pool_views(const void *pool) {
    pthread_mutex_lock(&g_rmu);
    for (int i = 0; i < g_ntreg; i++) {
        treg_t *r = &g_treg[i];
        if (!r->ok || (const void *)r->t.pool != pool || (!r->view && !r->alt && !r->ar)) continue;
        if (r->view) { defer_obj_unref((__bridge id)r->view); CFRelease(r->view); r->view = NULL; r->view_id = r->view_rid = 0; }
        treg_drop_alt(r); treg_drop_ar(r);   // both are derived from the view that just went: they hold it, and it holds the pool's buffer
    }
    pthread_mutex_unlock(&g_rmu);
}
// D1 diagnostic: the texture-ID cache entries whose cached view (or the alternate view derived from it) lives on this buffer.
static uint64_t dead_treg_views(id b) {
    uint64_t n = 0;
    pthread_mutex_lock(&g_rmu);
    for (int i = 0; i < g_ntreg; i++) {
        if (!g_treg[i].ok) continue;
        id v = (__bridge id)g_treg[i].view, a = (__bridge id)g_treg[i].alt;
        if (v && tex_root_buf(v) == b) n++;
        if (a && tex_root_buf(a) == b) n++;
    }
    pthread_mutex_unlock(&g_rmu);
    return n;
}
void mtl_reg_tex(int tid, const tex_t *t, const view_t *v) {
    int id_ = tid; if (id_ < 0) return;
    pthread_mutex_lock(&g_rmu);
    if (id_ >= g_ntreg) { int n = id_ + 1024; g_treg = realloc(g_treg, n * sizeof *g_treg); memset(g_treg + g_ntreg, 0, (n - g_ntreg) * sizeof *g_treg); g_ntreg = n; }
    g_reg_gen++;
    if (g_treg[id_].view) { defer_obj_unref((__bridge id)g_treg[id_].view); CFRelease(g_treg[id_].view); g_treg[id_].view = NULL; g_treg[id_].view_id = g_treg[id_].view_rid = 0; }
    if (g_treg[id_].alt) { defer_obj_unref((__bridge id)g_treg[id_].alt); CFRelease(g_treg[id_].alt); g_treg[id_].alt = g_treg[id_].alt_of = NULL; g_treg[id_].alt_rid = 0; }
    treg_drop_ar(&g_treg[id_]);
    g_treg[id_].ok = t != NULL; if (t) g_treg[id_].t = *t; g_treg[id_].has_v = v != NULL; if (v) g_treg[id_].v = *v;
    if (id_ < (1 << 20)) atomic_store_explicit(&g_tser[id_], t ? t->serial : 0, memory_order_release);
    pthread_mutex_unlock(&g_rmu);
    {   // DM_NVN_TEXINFO=WxH also logs registrations of textures of that size: the channel swizzle from the builder and from the view
        static int ti = -1; static uint32_t tw, th; static uint64_t n;
        if (ti < 0) { const char *e = getenv("DM_NVN_TEXINFO"); ti = e && sscanf(e, "%ux%u", &tw, &th) == 2; }
        if (ti && t && t->d.width == tw && t->d.height == th && (n++ < 40 || n % 600 == 0))
            MLOG("TEXINFO 登记 id %d ser=%llu 格式=%#x 建时重排 %d:%d%d%d%d 视图 %d 视图重排 %d:%d%d%d%d 视图格式 %d:%#x", tid, (unsigned long long)t->serial, t->d.format,
                 t->set_swz, t->swz & 7, t->swz >> 3 & 7, t->swz >> 6 & 7, t->swz >> 9 & 7, v != NULL, v ? v->set_swizzle : 0, v ? v->swz[0] : 0, v ? v->swz[1] : 0, v ? v->swz[2] : 0, v ? v->swz[3] : 0,
                 v ? v->set_format : 0, v ? v->format : 0);
    }
}
void mtl_reg_samp(int sid, const samp_par *sp) {
    int id_ = sid; if (id_ < 0) return;
    pthread_mutex_lock(&g_rmu);
    if (id_ >= g_nsreg) { int n = id_ + 256; g_sreg = realloc(g_sreg, n * sizeof *g_sreg); g_sreg_ok = realloc(g_sreg_ok, n); g_sview = realloc(g_sview, n * sizeof *g_sview); g_srid = realloc(g_srid, n * sizeof *g_srid);
        memset(g_sreg_ok + g_nsreg, 0, n - g_nsreg); memset(g_sview + g_nsreg, 0, (n - g_nsreg) * sizeof *g_sview); memset(g_srid + g_nsreg, 0, (n - g_nsreg) * sizeof *g_srid); g_nsreg = n; }
    if (g_sview[id_]) { CFRelease(g_sview[id_]); g_sview[id_] = NULL; g_srid[id_] = 0; }
    g_reg_gen++;
    g_sreg[id_] = *sp; g_sreg_ok[id_] = 1;
    pthread_mutex_unlock(&g_rmu);
}
// Handles of deferred binds (BindTexturesDeferred) are read at execution: the record keeps the address of the handles (see hle_nvn.c); every reader of a bind record refreshes it first. Command memory is writable, so it is updated in place
static inline void bh_fresh(const p_bind_handles *b) {
    if (b->defer && b->src && b->n > 0 && b->n <= 32) memcpy((void *)b->h, (const void *)(uintptr_t)b->src, 8 * (size_t)b->n);
}
static MTLTextureType ttype(int target, int fallback) {
    switch (target) { case 0: return MTLTextureType2D; case 1: case 7: return MTLTextureType2D; case 2: return MTLTextureType3D;   // 1D as one-row 2D (see where textures are created)
        case 3: return MTLTextureType2DArray; case 4: return MTLTextureType2DArray; case 5: return MTLTextureType2DMultisample;
        case 6: return MTLTextureType2DMultisampleArray; case 8: return MTLTextureTypeCube; case 9: return MTLTextureTypeCubeArray; }
    return (MTLTextureType)fallback;
}
// Out-of-pool texture ID = empty slot: the game itself uses GetSeparateTextureHandle(0xFFFFF) to get a handle and bind it to unused slots (measured call site main+0x1746820,
// texture pool capacity 15256) and never registers that ID. Read 0 per D3D / empty-slot semantics (the same all-zero dummy textures as unbound), no failure; unregistered IDs inside the pool are still reported a failure
static int g_tpool_n; static uint64_t g_stat_tex_oob;
// NVN window origin. In lower-left mode (GL convention, NVN's default) NDC y = -1 lands on a render target's first memory row and window y grows with the row; Metal puts NDC y = +1 on the first row.
// Viewport and scissor rectangles, fragment coordinates, copies and texture coordinates all address the same memory rows in both conventions, so only the direction inside the viewport differs:
// lower-left draws get a viewport of negative height (and the opposite front-face winding), and the window image, stored bottom-up, is flipped at present (and in raw screenshots).
// Link's Awakening keeps the default: read as upper-left, every texture it renders and samples came out consistently flipped and cancelled out, except where the game compensates for the
// convention itself (text rendered into textures for the title and the menus came out upside down, and a MIRROR_Y copy of the frame)
static _Atomic int g_origin_ll;
void mtl_set_origin_lower_left(int on) { if (atomic_exchange(&g_origin_ll, on) != on) MLOG("窗口原点：%s", on ? "左下（GL，视口倒向、绕向反、出图翻转）" : "左上"); }
void mtl_set_tex_pool_n(int n) { g_tpool_n = n; }
static id<MTLTexture> tex_by_id(int tid) {   // Caller holds g_rmu
    if (g_tpool_n && tid >= g_tpool_n) { if (!g_stat_tex_oob++) MLOG("池外贴图号 %d（池容量 %d）按空槽处理：读出 0", tid, g_tpool_n); return nil; }
    if (tid < 0 || tid >= g_ntreg || !g_treg[tid].ok) { RED_ONCE(10, "贴图号 %d 未登记", tid); return nil; }
    treg_t *rp = &g_treg[tid];
    uint64_t sg = g_ser_gen, dg = nvn_dirty_now();   // Take the generations first and then look up: if a change happens in between, the next time must take the slow path
    // A GPU-only texture does not depend on the guest writes, so its view stays valid across them. A CPU-written (packaged) one does:
    // its source pages can be written and re-uploaded, so the dirty generation has to match.
    if (rp->view && rp->view_ser == sg && (!rp->view_pkg || rp->view_dirty == dg)) return (__bridge id<MTLTexture>)rp->view;
    id<MTLTexture> v = (__bridge id<MTLTexture>)rp->view;
    if (v) {   // The serial table or guest memory has changed: check whether the base texture was re-uploaded and replaced; if so the view is invalidated
        id<MTLTexture> cur = tex_mtl(&rp->t);
        if (cur == v || v.parentTexture == cur) { rp->view_ser = sg; rp->view_dirty = dg; return v; }
        defer_obj_unref(v); CFRelease(rp->view); rp->view = NULL; rp->view_rid = 0; v = nil;
        treg_drop_alt(rp);   // the alt was built from the view that just went; it holds it, so it would hold the old base texture for good
    }
    rp->view_pkg = (rp->t.has_layout || tex_cpu_written(rp->t.serial)) && rp->t.mem;   // only a CPU-written texture depends on the guest writes
    treg_t r = *rp;
    id<MTLTexture> base = tex_mtl(&rp->t);
    if (!base) return nil;
    v = base;
    // Channel swizzle: the view's if set, otherwise the one from texture building; identity builds no view (SMBW's font cache is (1 1 1 R): without the swizzle glyphs become white blocks)
    uint8_t sw[4] = {MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha}; int need_sw = 0;
    if (r.has_v && r.v.set_swizzle) memcpy(sw, r.v.swz, 4);
    else if (r.t.set_swz) for (int c = 0; c < 4; c++) sw[c] = (uint8_t)(r.t.swz >> (3 * c) & 7);
    for (int c = 0; c < 4; c++) { if (sw[c] > MTLTextureSwizzleAlpha) sw[c] = (uint8_t)(MTLTextureSwizzleRed + c); if (sw[c] != MTLTextureSwizzleRed + c) need_sw = 1; }
    if ((r.has_v && (r.v.set_format || r.v.set_levels || r.v.set_layers || r.v.set_target)) || need_sw) {
        if (!r.has_v) memset(&r.v, 0, sizeof r.v);
        uint32_t pf = (uint32_t)base.pixelFormat; uint8_t swz[4];
        if (r.v.set_format && !(nvn_fmt_to_mtl(r.v.format, &pf, swz) && pf)) pf = (uint32_t)base.pixelFormat;
        { uint32_t upf; int ubpp; if (r.v.set_format && bc_unpack(r.v.format, &upf, &ubpp)) pf = upf; }   // No BC: a BC view is switched to the decompressed format just like the base texture
        if (r.v.set_format && tex_need_pfv(base.pixelFormat, (MTLPixelFormat)pf) && !(base.usage & MTLTextureUsagePixelFormatView)) {
            base = tex_upgrade_pfv(&rp->t, base, (MTLPixelFormat)pf);
            sg = g_ser_gen;
        }
        NSUInteger l0 = r.v.set_levels ? r.v.base_level : 0, nl = r.v.set_levels && r.v.nlevels ? r.v.nlevels : base.mipmapLevelCount - l0;
        if (l0 >= base.mipmapLevelCount) { RED_ONCE(93, "贴图视图 id %d 起始 mip %lu 超出 %lu 级", tid, (unsigned long)l0, (unsigned long)base.mipmapLevelCount); return nil; }
        // 级数超出贴图本身的裁到它有的：deko3d 的 Tegra 描述符写入同样裁最大级（智慧的再现：七级贴图的视图取 1+7，Metal 验证拒绝）
        if (nl > base.mipmapLevelCount - l0) nl = base.mipmapLevelCount - l0;
        NSUInteger s0 = r.v.set_layers ? r.v.min_layer : 0;
        NSUInteger ns = base.textureType == MTLTextureType3D ? 1 : (r.v.set_layers && r.v.nlayers ? r.v.nlayers : base.arrayLength * (base.textureType == MTLTextureTypeCube || base.textureType == MTLTextureTypeCubeArray ? 6 : 1) - s0);
        MTLTextureType tt = r.v.set_target ? ttype(r.v.target, base.textureType) : base.textureType;
        if (tt == MTLTextureType2D && ns > 1) ns = 1;
        v = need_sw ? [base newTextureViewWithPixelFormat:(MTLPixelFormat)pf textureType:tt levels:NSMakeRange(l0, nl) slices:NSMakeRange(s0, ns) swizzle:MTLTextureSwizzleChannelsMake(sw[0], sw[1], sw[2], sw[3])]
                    : [base newTextureViewWithPixelFormat:(MTLPixelFormat)pf textureType:tt levels:NSMakeRange(l0, nl) slices:NSMakeRange(s0, ns)];
        if (!v) { RED_ONCE(11, "贴图视图建不出 id %d", tid); v = base; }
    }
    rp->view = (void *)CFBridgingRetain(v); rp->view_rid = v.gpuResourceID._impl; rp->view_ser = sg; rp->view_dirty = dg;
    return v;
}
// Hot-path object fetch: if the table entry is valid, return the table-held pointer directly (without ARC retain / autorelease), otherwise take the slow path to build / validate. Caller holds g_rmu.
static inline __attribute__((always_inline)) void *tex_fast(int tid) {
    if (tid > 0 && tid < g_ntreg && g_treg[tid].ok && g_treg[tid].view && g_treg[tid].view_ser == g_ser_gen && (!g_treg[tid].view_pkg || g_treg[tid].view_dirty == nvn_dirty_now())) return g_treg[tid].view;
    return (__bridge void *)tex_by_id(tid);
}
// The texture type the shader declares differs from the one the descriptor gives: use a compatible view. Maxwell sample instructions carry their own dimension, and a non-array instruction reading a 2D array texture reads layer 0;
// on Metal a 2D array bound to a texture2d argument is a type mismatch with undefined reads (measured: one fixed value). Super Mario 3D World + Bowser's Fury creates its pre-lighting buffer as a one-layer 2D array
// and the deferred lighting shader samples it as 2D: the ambient light of the whole picture became one colour, washed out white. Supports 2D array <-> 2D and 2D array -> cube / cube array; any other mismatch is reported once. Caller holds g_rmu
static id<MTLTexture> tex_fit(int tid, id<MTLTexture> t, MTLTextureType want, uint64_t *rid) {
    MTLTextureType have = t.textureType; NSUInteger ns = 0;
    if (want == MTLTextureType2D && have == MTLTextureType2DArray) ns = 1;
    else if (want == MTLTextureType2DArray && have == MTLTextureType2D) ns = 1;
    else if (want == MTLTextureTypeCube && have == MTLTextureType2DArray && t.arrayLength >= 6) ns = 6;
    else if (want == MTLTextureTypeCubeArray && have == MTLTextureType2DArray && t.arrayLength >= 6) ns = t.arrayLength / 6 * 6;
    else if (want == MTLTextureTypeCube && have == MTLTextureTypeCubeArray) ns = 6;   // a cube array read as a cube: the first cube
    if (!ns || tid <= 0 || tid >= g_ntreg) { RED_ONCE(76, "贴图类型不符：着色器声明 %lu，贴图 %lu（贴图号 %d），照原样绑", (unsigned long)want, (unsigned long)have, tid); return nil; }
    treg_t *r = &g_treg[tid];
    if (r->alt && r->alt_of == (__bridge void *)t && r->alt_type == (uint8_t)want) { *rid = r->alt_rid; return (__bridge id<MTLTexture>)r->alt; }
    if (r->alt) { defer_obj_unref((__bridge id)r->alt); CFRelease(r->alt); r->alt = r->alt_of = NULL; r->alt_rid = 0; }
    id<MTLTexture> v = [t newTextureViewWithPixelFormat:t.pixelFormat textureType:want levels:NSMakeRange(0, t.mipmapLevelCount) slices:NSMakeRange(0, ns) swizzle:t.swizzle];
    if (!v) { RED_ONCE(77, "贴图类型兼容视图建不出：%lu → %lu（贴图号 %d）", (unsigned long)have, (unsigned long)want, tid); return nil; }
    r->alt = (void *)CFBridgingRetain(v); r->alt_of = (__bridge void *)t; r->alt_type = (uint8_t)want; r->alt_rid = v.gpuResourceID._impl;
    *rid = r->alt_rid; return v;
}
// Alpha from red: the fragment shaders the game profile's alpha_from_red lists read a single-channel render target whose channel mapping is exactly (R, 0, 0, 1) as (R, 0, 0, R).
// Breath of the Wild's layout captures: a layout pane marked CaptureOn (user data "A8", "CaptureScale" 0.5, "MultiFilter"; about 140 layouts) is drawn into an RGBA8 target, blurred at half size,
// and its alpha copied into the red channel of an R8 texture the game builds with mapping (R, 0, 0, 1); the layout's panes then sample it in place of the FBLayout_00 placeholder.
// Each HUD prompt bar (PaGuideKey*) draws, per key, a cream glow P_Glow_00 (aa209f57b133e862: colour from the Kumo64 cloud texture, alpha = min(cloud alpha, capture alpha), animated in
// when the bar appears), then a black shadow P_Sh_00 (e6a88e12dbae9743: alpha = capture alpha), then the text. Read literally, the capture's alpha is always 1: the shadow is a hard-edged box
// and the glow fills the whole rectangle with cloud (both seen on the iPhone); in Switch footage both follow the text. On the Mac (shrine, bar fade-in) the full list leaves only a glow around the words.
// Only render targets qualify (captures always are; asset and CPU-written textures are not), so the list can name every layout shader variant. DM_NVN_ALPHA_RED=0 turns it off, <keys,...> replaces the list. Caller holds g_rmu
static char g_afr_keys[48][20]; static int g_afr_nk = -1;
static int afr_want(NvShader *sh) {
    if (!sh->afr) {
        if (g_afr_nk < 0) { g_afr_nk = 0; const char *e = getenv("DM_NVN_ALPHA_RED"), *src = e && *e ? e : DM_GAME_ALPHA_FROM_RED;
            if (strcmp(src, "0")) for (const char *q = src; *q && g_afr_nk < 48;) { size_t n = strcspn(q, ", "); if (n) { if (n > 16) n = 16; memcpy(g_afr_keys[g_afr_nk], q, n); g_afr_keys[g_afr_nk++][n] = 0; } q += n; q += strspn(q, ", "); }
            if (g_afr_nk) MLOG("单通道贴图 alpha 取红：%s%s", src, e && *e ? "（DM_NVN_ALPHA_RED）" : "（游戏档案 alpha_from_red）"); }
        sh->afr = 1; for (int i = 0; i < g_afr_nk; i++) if (!strncmp(sh->key, g_afr_keys[i], strlen(g_afr_keys[i]))) sh->afr = 2;
    }
    return sh->afr == 2;
}
static id<MTLTexture> tex_alpha_red(int tid, id<MTLTexture> t, uint64_t *rid) {
    MTLPixelFormat pf = t.pixelFormat; MTLTextureSwizzleChannels sw = t.swizzle;
    if (pf != MTLPixelFormatR8Unorm && pf != MTLPixelFormatR16Unorm && pf != MTLPixelFormatR16Float && pf != MTLPixelFormatR32Float) return nil;
    if (sw.red != MTLTextureSwizzleRed || sw.green != MTLTextureSwizzleZero || sw.blue != MTLTextureSwizzleZero || sw.alpha != MTLTextureSwizzleOne || tid <= 0 || tid >= g_ntreg) return nil;
    id<MTLTexture> base = t.parentTexture ? t.parentTexture : t;
    if (!(base.usage & MTLTextureUsageRenderTarget)) return nil;   // packaged and CPU-written textures are built without render-target usage
    treg_t *r = &g_treg[tid];
    if (r->ar && r->ar_of == (__bridge void *)t) { *rid = r->ar_rid; return (__bridge id<MTLTexture>)r->ar; }
    treg_drop_ar(r);
    NSUInteger l0 = t.parentTexture ? t.parentRelativeLevel : 0, s0 = t.parentTexture ? t.parentRelativeSlice : 0;
    NSUInteger ns = t.textureType == MTLTextureType3D ? 1 : t.arrayLength * (t.textureType == MTLTextureTypeCube || t.textureType == MTLTextureTypeCubeArray ? 6 : 1);
    id<MTLTexture> v = [base newTextureViewWithPixelFormat:pf textureType:t.textureType levels:NSMakeRange(l0, t.mipmapLevelCount) slices:NSMakeRange(s0, ns)
                                                   swizzle:MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleZero, MTLTextureSwizzleZero, MTLTextureSwizzleRed)];
    if (!v) { RED_ONCE(82, "alpha 取红的视图建不出（贴图号 %d 格式 %lu）", tid, (unsigned long)pf); return nil; }
    { static int once; if (!once++) MLOG("单通道贴图 alpha 取红：首例 贴图号 %d %lux%lu", tid, (unsigned long)t.width, (unsigned long)t.height); }
    r->ar = (void *)CFBridgingRetain(v); r->ar_of = (__bridge void *)t; r->ar_rid = v.gpuResourceID._impl;
    *rid = r->ar_rid; return v;
}
static inline __attribute__((always_inline)) uint64_t tex_fast_rid(int tid, void **out) {
    if (tid > 0 && tid < g_ntreg && g_treg[tid].ok && g_treg[tid].view && g_treg[tid].view_ser == g_ser_gen && (!g_treg[tid].view_pkg || g_treg[tid].view_dirty == nvn_dirty_now())) { if (out) *out = g_treg[tid].view; return g_treg[tid].view_rid; }
    id<MTLTexture> t = tex_by_id(tid); if (out) *out = (__bridge void *)t; return t ? t.gpuResourceID._impl : 0;
}
static MTLSamplerAddressMode wrapm(int w) {
    switch (w) { case 0: case 7: return MTLSamplerAddressModeClampToEdge; case 1: return MTLSamplerAddressModeRepeat;
        case 3: case 2: return MTLSamplerAddressModeMirrorClampToEdge; case 5: case 4: return MTLSamplerAddressModeClampToBorderColor;
        case 6: return MTLSamplerAddressModeMirrorRepeat; }
    return MTLSamplerAddressModeRepeat;
}
static id<MTLSamplerState> samp_by_id(int sid) {   // Caller holds g_rmu
    if (sid < 0 || sid >= g_nsreg || !g_sreg_ok[sid]) { RED_ONCE(12, "采样器号 %d 未登记", sid); return nil; }
    if (g_sview[sid]) return (__bridge id<MTLSamplerState>)g_sview[sid];
    samp_par p = g_sreg[sid];
    MTLSamplerDescriptor *d = [MTLSamplerDescriptor new];
    d.magFilter = p.magf ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    d.minFilter = (p.minf == 1 || p.minf == 3 || p.minf == 5) ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    d.mipFilter = p.minf <= 1 ? MTLSamplerMipFilterNotMipmapped : (p.minf == 2 || p.minf == 3) ? MTLSamplerMipFilterNearest : MTLSamplerMipFilterLinear;
    d.sAddressMode = wrapm(p.ws); d.tAddressMode = wrapm(p.wt); d.rAddressMode = wrapm(p.wr);
    d.lodMinClamp = p.lod_min < 0 ? 0 : p.lod_min; d.lodMaxClamp = p.lod_max;
    d.maxAnisotropy = p.aniso < 1 ? 1 : p.aniso > 16 ? 16 : (NSUInteger)p.aniso;
    if (p.cmp_mode) d.compareFunction = (MTLCompareFunction)(p.cmp_func - 1);
    d.borderColor = p.border[3] > 0.5f ? (p.border[0] > 0.5f ? MTLSamplerBorderColorOpaqueWhite : MTLSamplerBorderColorOpaqueBlack) : MTLSamplerBorderColorTransparentBlack;
    d.supportArgumentBuffers = YES;
    id<MTLSamplerState> s = [g_dev newSamplerStateWithDescriptor:d];
    if (s) { g_sview[sid] = (void *)CFBridgingRetain(s); g_srid[sid] = s.gpuResourceID._impl; }
    return s;
}
static inline __attribute__((always_inline)) void *samp_fast(int sid) {
    if (sid > 0 && sid < g_nsreg && g_sreg_ok[sid] && g_sview[sid]) return g_sview[sid];
    return (__bridge void *)samp_by_id(sid);
}

// ---------- Argument buffer blocks: one block per submission, reclaimed after the GPU completes ----------
@interface NvArena : NSObject { @public id<MTLBuffer> buf; uint64_t gpu, fence; size_t used; } @end
@implementation NvArena @end
static NSMutableArray<NvArena *> *g_arenas; static NvArena *g_cur_arena; static NSMutableArray<NvArena *> *g_sub_arenas;
static NvArena *arena_new(void) {
    // The for-in loop variable does not retain the object: take a strong reference first and then remove it from the array, otherwise what is returned after the array releases it is a dangling pointer
    for (NvArena *it in g_arenas) if (it->fence && it->fence <= g_done) { NvArena *r = it; [g_arenas removeObject:r]; r->fence = 0; r->used = 0; return r; }
    NvArena *a = [NvArena new]; a->buf = [g_dev newBufferWithLength:4 << 20 options:MTLResourceStorageModeShared]; a->gpu = a->buf.gpuAddress; res_add(a->buf); g_arena_made++; return a;
}
void arena_begin(void) { if (!g_arenas) { g_arenas = [NSMutableArray array]; g_sub_arenas = [NSMutableArray array]; } g_cur_arena = nil; }
void arena_commit(uint64_t fence) { for (NvArena *a in g_sub_arenas) { a->fence = fence; [g_arenas addObject:a]; } [g_sub_arenas removeAllObjects]; g_cur_arena = nil; }
static size_t arena_alloc(size_t n, id<MTLBuffer> *buf) {
    n = (n + 255) & ~(size_t)255;
    if (!g_cur_arena || g_cur_arena->used + n > g_cur_arena->buf.length) {
        pthread_mutex_lock(&g_mu); g_cur_arena = arena_new(); pthread_mutex_unlock(&g_mu);
        [g_sub_arenas addObject:g_cur_arena];
    }
    size_t off = g_cur_arena->used; g_cur_arena->used += n; *buf = g_cur_arena->buf; return off;
}

// ---------- Draw state ----------
static struct {
    const prog_t *prog;
    const prog_t *bst[5];      // program bound per stage (NVN stage bits 1 vertex 2 fragment 4 geometry 8 / 16 tessellation, indices as in NvProg.sh), see bind_stages
    blend_st blend[8]; color_st color; chmask_st chmask; ms_st ms; poly_st poly; ds_st ds;
    va_st attr[16]; int nattr; vs_st stream[16]; int nstream;
    NVNbufferRange vbuf[16];
    p_bind_buf ubo[6][16], ssbo[6][16];
    uint64_t tex[6][128], samp[6][128], img[6][32];
    uint32_t st_vmask[2], st_wmask[2], st_ref[2];
    p_poly_offset po; float vp[16][4]; float dr[2]; int has_vp;
    float blend_color[4]; int64_t copy_stride, copy_img_stride;
} S;
static void udc_note_draw(NvProg *pr) {
    if (R.has_rt) for (int i = 0; i < R.rt.n && i < 8; i++) if (R.rt.color[i]) { zskip_del(R.rt.color[i]->serial); g_zskip_cand_ok[i] = 0; }   // Zero-source set: once really drawn it is no longer all zeros
    if (pr) for (int st = 0; st < 6; st++) { __unsafe_unretained NvShader *sh = pr->sh[st]; if (!sh) continue; for (int i = 0; i < sh->rf.nb; i++) if (sh->rf.b[i].kind == NVN_RB_IMG) { zskip_clear(); break; } }   // Image slots are writable: conservatively clear
    int image_write = 0;
    uint32_t shader_color_mask = 0;
    if (pr) for (int st = 0; st < 6; st++) { __unsafe_unretained NvShader *sh = pr->sh[st]; if (!sh) continue;
        if (st == 1) shader_color_mask = sh->rf.color_mask;
        for (int i = 0; i < sh->rf.nb; i++) if (sh->rf.b[i].kind == NVN_RB_IMG && sh->rf.b[i].rw) { image_write = 1; break; }
        if (image_write) break;
    }
    if (R.has_rt) for (int i = 1; i < R.rt.n && i < 8; i++)   // a slot attached twice (left unattached) really written: the writes are lost (see rt_dup_slot)
        if ((shader_color_mask >> i & 1) && ((S.chmask.mask >> (4 * i)) & 0xf) && rt_dup_slot(&R.rt, i))
            RED_ONCE(79, "颜色槽 %d 与前面的槽挂同一张贴图（ser=%llu）且本次绘制写它（掩码 %#x）：这些写入丢了", i, (unsigned long long)R.rt.color[i]->serial, (S.chmask.mask >> (4 * i)) & 0xf);
    for (int i = 0; i < g_nsd; i++) {
        sd_att *t = &g_sd[i];
        if (image_write) { t->wrote = 1; t->wwhy = 1; }
        else if (t->depth == 0 && t->idx >= 0 && (((S.chmask.mask >> (4 * t->idx)) & 0xf) || (shader_color_mask >> t->idx & 1))) { t->wrote = 1; t->wwhy = (S.chmask.mask >> (4 * t->idx)) & 0xf ? 2 : 3; }
        else if (t->depth == 1 && S.ds.dw) { t->wrote = 1; t->wwhy = 4; }
        else if (t->depth == 2 && S.ds.st && (S.st_wmask[0] || S.st_wmask[1])
                && (S.ds.fail_f != NVN_STENCIL_OP_KEEP || S.ds.zfail_f != NVN_STENCIL_OP_KEEP || S.ds.zpass_f != NVN_STENCIL_OP_KEEP
                    || S.ds.fail_b != NVN_STENCIL_OP_KEEP || S.ds.zfail_b != NVN_STENCIL_OP_KEEP || S.ds.zpass_b != NVN_STENCIL_OP_KEEP)) { t->wrote = 1; t->wwhy = 5; }
    }
}
// NVN's BindProgram(program, stage bits) binds per stage: the stages in the bits take that program's shaders (a null program unbinds them), the other stages stay.
// Metroid Dread's depth prepass calls BindProgram(program, vertex | fragment), then BindProgram(null, fragment): only the vertex shader remains and depth is drawn.
// Every bind used to replace S.prog as a whole, a null program meant "no program bound", and the prepass's ~140 draws per frame were all dropped: depth stayed empty, the G-buffer's depth EQUAL test passed no pixel, and scenes and characters were black.
// When every stage comes from the same program, that program is the effective one (all of Mario's and Engage's binds are like this, so nothing changes for them); otherwise a composite is made: a host-made prog_t carrying only a serial, whose NvProg in the serial table takes each stage from its source program (a null fragment = a vertex-only pipeline)
static const prog_t *bind_stages(const prog_t *p, int stages) {
    for (int st = 0; st < 5; st++) if (stages >> st & 1) S.bst[st] = p;
    if (stages & 0x20) return p;   // compute: replaced as a whole as before
    const prog_t *v = S.bst[0];
    if (!v) return NULL;
    int same = 1; for (int st = 1; st < 5; st++) if (S.bst[st] && S.bst[st] != v) same = 0;
    if (same && S.bst[1] == v) return v;
    enum { NCMP = 64 };
    static prog_t cp[NCMP]; static const prog_t *ck[NCMP][5]; static uint64_t cs[NCMP][5]; static int ncp, next;
    for (int i = 0; i < ncp; i++) {
        int hit = 1; for (int st = 0; st < 5 && hit; st++) hit = ck[i][st] == S.bst[st] && (!S.bst[st] || cs[i][st] == S.bst[st]->serial);
        if (hit) return &cp[i];
    }
    int i = ncp < NCMP ? ncp++ : next++ % NCMP;
    if (cp[i].serial) serial_del(cp[i].serial);
    NvProg *o = [NvProg new];
    for (int st = 0; st < 5; st++) { ck[i][st] = S.bst[st]; cs[i][st] = S.bst[st] ? S.bst[st]->serial : 0; if (S.bst[st]) { __unsafe_unretained NvProg *src = prog_get(S.bst[st]); o->sh[st] = src ? src->sh[st] : nil; } }
    memset(&cp[i], 0, sizeof cp[i]); cp[i].magic = MAGIC('P', 'R', 'O', 'G'); cp[i].serial = nvn_new_serial(); serial_put(cp[i].serial, o);
    { static int once; if (!once++) MLOG("按阶段绑程序：首个合成程序（顶点 %s 片元 %s）", o->sh[0] ? o->sh[0]->key : "-", o->sh[1] ? o->sh[1]->key : "-"); }
    return &cp[i];
}
static void state_reset(void) {
    memset(&S, 0, sizeof S);
    g_pso_dirty = g_dss_dirty = 1;
    for (int i = 0; i < 8; i++) S.blend[i] = (blend_st){i, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ZERO, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ZERO, NVN_BLEND_EQUATION_ADD, NVN_BLEND_EQUATION_ADD};
    S.color = (color_st){0, NVN_LOGIC_OP_COPY, NVN_ALPHA_FUNC_ALWAYS}; S.chmask.mask = 0xffffffffu;
    S.poly = (poly_st){NVN_FACE_NONE, NVN_FRONT_FACE_CCW, NVN_POLYGON_MODE_FILL, 0};
    S.ds = (ds_st){0, 1, NVN_DEPTH_FUNC_LESS, 0, NVN_STENCIL_FUNC_ALWAYS, NVN_STENCIL_FUNC_ALWAYS, 1, 1, 1, 1, 1, 1};
    S.st_vmask[0] = S.st_vmask[1] = S.st_wmask[0] = S.st_wmask[1] = 0xff; S.dr[1] = 1;
}

// Correct the vertex format by the shader input's component type: Metal requires integer inputs to get integer formats of the same signedness, and float inputs float / normalized formats.
// Integer input (inferred at translation time as "only used as an integer", dmsh_build.py int_inputs) with an integer format: on a signedness mismatch switch to the same width with the other sign (bit pattern unchanged); with a 32-bit float format read it as an integer of the same width (bit pattern unchanged).
// Float input with a 32-bit integer format: use the float format with the same component count. Maxwell attribute registers are untyped, so the shader interprets the same 32-bit bits as float; Metal would otherwise reject the pipeline (Hades II character / enemy vertex inputs).
// Other mismatches (integer input with a normalized / half format, float input with an 8- or 16-bit integer format) cannot keep the bit pattern: report once and keep the format.
static MTLVertexFormat vfmt_for_ct(MTLVertexFormat f, int ct, int nvn_fmt) {
    static const MTLVertexFormat U[] = { MTLVertexFormatUChar, MTLVertexFormatUChar2, MTLVertexFormatUChar3, MTLVertexFormatUChar4, MTLVertexFormatUShort, MTLVertexFormatUShort2, MTLVertexFormatUShort3, MTLVertexFormatUShort4, MTLVertexFormatUInt, MTLVertexFormatUInt2, MTLVertexFormatUInt3, MTLVertexFormatUInt4 };
    static const MTLVertexFormat I[] = { MTLVertexFormatChar, MTLVertexFormatChar2, MTLVertexFormatChar3, MTLVertexFormatChar4, MTLVertexFormatShort, MTLVertexFormatShort2, MTLVertexFormatShort3, MTLVertexFormatShort4, MTLVertexFormatInt, MTLVertexFormatInt2, MTLVertexFormatInt3, MTLVertexFormatInt4 };
    static const MTLVertexFormat F[] = { MTLVertexFormatFloat, MTLVertexFormatFloat2, MTLVertexFormatFloat3, MTLVertexFormatFloat4 };
    int iu = -1, ii = -1, iff = -1;
    for (int i = 0; i < 12; i++) { if (U[i] == f) iu = i; if (I[i] == f) ii = i; }
    for (int i = 0; i < 4; i++) if (F[i] == f) iff = i;
    if (ct == NVN_CT_UINT) { if (iu >= 0) return f; if (ii >= 0) return U[ii]; if (iff >= 0) return U[8 + iff]; }
    else if (ct == NVN_CT_INT) { if (ii >= 0) return f; if (iu >= 0) return I[iu]; if (iff >= 0) return I[8 + iff]; }
    else {
        if (iu >= 8) return F[iu - 8];   // R32UI / RG32UI / RGB32UI / RGBA32UI
        if (ii >= 8) return F[ii - 8];   // R32I / RG32I / RGB32I / RGBA32I
        if (iu < 0 && ii < 0) return f;
    }
    RED_ONCE(66, "顶点属性格式 %#x 与着色器输入类型 %s 对不上，位模式保不住（照原格式）", nvn_fmt, ct == NVN_CT_UINT ? "uint" : ct == NVN_CT_INT ? "int" : "float");
    return f;
}
static MTLVertexFormat vfmt(int f) {
    switch (f) {
        case 0x01: return MTLVertexFormatUCharNormalized; case 0x02: return MTLVertexFormatCharNormalized; case 0x03: return MTLVertexFormatUChar; case 0x04: return MTLVertexFormatChar;
        case 0x05: return MTLVertexFormatHalf; case 0x06: return MTLVertexFormatUShortNormalized; case 0x07: return MTLVertexFormatShortNormalized;
        case 0x08: return MTLVertexFormatUShort; case 0x09: return MTLVertexFormatShort; case 0x0A: return MTLVertexFormatFloat; case 0x0B: return MTLVertexFormatUInt; case 0x0C: return MTLVertexFormatInt;
        case 0x0D: return MTLVertexFormatUChar2Normalized; case 0x0E: return MTLVertexFormatChar2Normalized; case 0x0F: return MTLVertexFormatUChar2; case 0x10: return MTLVertexFormatChar2;
        case 0x11: return MTLVertexFormatHalf2; case 0x12: return MTLVertexFormatUShort2Normalized; case 0x13: return MTLVertexFormatShort2Normalized; case 0x14: return MTLVertexFormatUShort2;
        case 0x15: return MTLVertexFormatShort2; case 0x16: return MTLVertexFormatFloat2; case 0x17: return MTLVertexFormatUInt2; case 0x18: return MTLVertexFormatInt2;
        case 0x19: return MTLVertexFormatUChar3Normalized; case 0x1A: return MTLVertexFormatChar3Normalized; case 0x1B: return MTLVertexFormatUChar3; case 0x1C: return MTLVertexFormatChar3;
        case 0x1D: return MTLVertexFormatHalf3; case 0x1E: return MTLVertexFormatUShort3Normalized; case 0x1F: return MTLVertexFormatShort3Normalized; case 0x20: return MTLVertexFormatUShort3;
        case 0x21: return MTLVertexFormatShort3; case 0x22: return MTLVertexFormatFloat3; case 0x23: return MTLVertexFormatUInt3; case 0x24: return MTLVertexFormatInt3;
        case 0x25: return MTLVertexFormatUChar4Normalized; case 0x26: return MTLVertexFormatChar4Normalized; case 0x27: return MTLVertexFormatUChar4; case 0x28: return MTLVertexFormatChar4;
        case 0x29: return MTLVertexFormatHalf4; case 0x2A: return MTLVertexFormatUShort4Normalized; case 0x2B: return MTLVertexFormatShort4Normalized; case 0x2C: return MTLVertexFormatUShort4;
        case 0x2D: return MTLVertexFormatShort4; case 0x2E: return MTLVertexFormatFloat4; case 0x2F: return MTLVertexFormatUInt4; case 0x30: return MTLVertexFormatInt4;
        case 0x3D: return MTLVertexFormatUInt1010102Normalized; case 0x69: return MTLVertexFormatInt1010102Normalized; case 0x3F: return MTLVertexFormatFloatRG11B10; case 0x9A: return MTLVertexFormatUChar4Normalized_BGRA;
    }
    return MTLVertexFormatInvalid;
}
static MTLBlendFactor bfac(int f) {
    switch (f) { case 1: return MTLBlendFactorZero; case 2: return MTLBlendFactorOne; case 3: return MTLBlendFactorSourceColor; case 4: return MTLBlendFactorOneMinusSourceColor;
        case 5: return MTLBlendFactorSourceAlpha; case 6: return MTLBlendFactorOneMinusSourceAlpha; case 7: return MTLBlendFactorDestinationAlpha; case 8: return MTLBlendFactorOneMinusDestinationAlpha;
        case 9: return MTLBlendFactorDestinationColor; case 10: return MTLBlendFactorOneMinusDestinationColor; case 11: return MTLBlendFactorSourceAlphaSaturated;
        case 0x10: return MTLBlendFactorSource1Color; case 0x11: return MTLBlendFactorOneMinusSource1Color; case 0x12: return MTLBlendFactorSource1Alpha; case 0x13: return MTLBlendFactorOneMinusSource1Alpha;
        case 0x61: return MTLBlendFactorBlendColor; case 0x62: return MTLBlendFactorOneMinusBlendColor; case 0x63: return MTLBlendFactorBlendAlpha; case 0x64: return MTLBlendFactorOneMinusBlendAlpha; }
    return MTLBlendFactorOne;
}
static MTLBlendOperation bop(int e) { switch (e) { case 2: return MTLBlendOperationSubtract; case 3: return MTLBlendOperationReverseSubtract; case 4: return MTLBlendOperationMin; case 5: return MTLBlendOperationMax; } return MTLBlendOperationAdd; }

// Pipeline: cached by (program, target formats, blend / write mask, vertex layout, sample count)
static const uint8_t *g_pso_lmap; static id<MTLFunction> g_pso_fsfn; static pso_fz_desc g_pso_fz, g_fz_desc;
static os_unfair_lock g_fz_fn_lock = OS_UNFAIR_LOCK_INIT;
static NSMutableDictionary<NSString *, id<MTLFunction>> *g_bf_fn_cache;
static NSMutableDictionary<NSString *, id<MTLLibrary>> *g_bf_lib_cache;
static nvn_hmap g_pso = {.klen = sizeof(pso_key)}, g_dss;
static nvn_hmap g_pso_fail = {.klen = sizeof(pso_key)};   // pipelines that failed to build: remembered and never retried (previously every draw recompiled it, over 10,000 times in one iPhone session)
static MTLRenderPipelineDescriptor *pso_desc(const pso_key *k, NvShader *vs, NvShader *fs, id<MTLFunction> fsfn, int merged) {
    MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = vs->fn; d.fragmentFunction = fsfn ? fsfn : fs ? fs->fn : nil;
    // Diagnostic DM_NVN_HILITE=fragment shader key,...: these draws use a solid-color fragment instead (keys 1–6 in order magenta / green / cyan / yellow / blue / white), to see where the geometry lands
    // Pipelines are built on several threads at once: the keys and the library are set up once (a lazy setup raced, and a thread used a library another had just replaced: Hades II, 6 keys)
    if (fs) { static char keys[6][24]; static int nk; static id<MTLLibrary> hl; static dispatch_once_t once;
        dispatch_once(&once, ^{ const char *e = getenv("DM_NVN_HILITE");
            for (const char *q = e; q && *q && nk < 6; q = strchr(q, ',') ? strchr(q, ',') + 1 : q + strlen(q)) { size_t n = strcspn(q, ","); if (n > 23) n = 23; memcpy(keys[nk], q, n); keys[nk++][n] = 0; }
            if (!nk) return;
            NSError *err = nil; hl = [g_dev newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal;\nstruct O { float4 c0 [[color(0)]]; float4 c1 [[color(1)]]; float4 c2 [[color(2)]]; };\n"
                "fragment O hl0() { return {float4(1,0,1,1),float4(1,0,1,1),float4(1,0,1,1)}; } fragment O hl1() { return {float4(0,1,0,1),float4(0,1,0,1),float4(0,1,0,1)}; }\n"
                "fragment O hl2() { return {float4(0,1,1,1),float4(0,1,1,1),float4(0,1,1,1)}; } fragment O hl3() { return {float4(1,1,0,1),float4(1,1,0,1),float4(1,1,0,1)}; }\n"
                "fragment O hl4() { return {float4(0,0,1,1),float4(0,0,1,1),float4(0,0,1,1)}; } fragment O hl5() { return {float4(1,1,1,1),float4(1,1,1,1),float4(1,1,1,1)}; }\n" options:nil error:&err];
            if (!hl) MLOG("【红】HILITE 着色器编译失败 %s", err.localizedDescription.UTF8String); });
        for (int i = 0; i < nk; i++) if (!strncmp(fs->key, keys[i], strlen(keys[i]))) {
            if (hl) d.fragmentFunction = [hl newFunctionWithName:[NSString stringWithFormat:@"hl%d", i]];
            break; } }
    d.rasterSampleCount = k->samples ? k->samples : 1;
    d.alphaToCoverageEnabled = k->a2c;
    for (int i = 0; i < 8; i++) {
        if (!k->cfmt[i]) continue;
        MTLRenderPipelineColorAttachmentDescriptor *c = d.colorAttachments[i];
        c.pixelFormat = (MTLPixelFormat)k->cfmt[i];
        int l = merged ? k->lmap[i] : i;
        uint32_t m = l == 0xff ? 0 : (k->chmask >> (4 * l)) & 0xf;
        // Output numbers the fragment shader does not write: zero write mask, no blending. Maxwell writes only the targets in the shader's output map and leaves the others untouched (whatever the blend state and channel mask);
        // on Metal an attachment the shader does not write has undefined contents, and with blending enabled that undefined value is blended in. Super Mario 3D World + Bowser's Fury's volume draws after the G-buffer (50d813761ca3f1f9 and others) write only outputs 0 and 2,
        // while targets 1 (normals) and 3 (pre-lighting) have SRC_ALPHA blending and all channels enabled
        int unwritten = fs && fs->rf.color_known && !fsfn && l != 0xff && !(fs->rf.color_mask >> l & 1);
        if (unwritten) m = 0;
        c.writeMask = (m & 1 ? MTLColorWriteMaskRed : 0) | (m & 2 ? MTLColorWriteMaskGreen : 0) | (m & 4 ? MTLColorWriteMaskBlue : 0) | (m & 8 ? MTLColorWriteMaskAlpha : 0);
        if (l != 0xff && k->blend_en >> l & 1 && !unwritten) {
            c.blendingEnabled = YES;
            c.sourceRGBBlendFactor = bfac(k->blend[l].src_rgb); c.destinationRGBBlendFactor = bfac(k->blend[l].dst_rgb);
            c.sourceAlphaBlendFactor = bfac(k->blend[l].src_a); c.destinationAlphaBlendFactor = bfac(k->blend[l].dst_a);
            c.rgbBlendOperation = bop(k->blend[l].eq_rgb); c.alphaBlendOperation = bop(k->blend[l].eq_a);
        }
    }
    d.depthAttachmentPixelFormat = (MTLPixelFormat)k->dfmt; d.stencilAttachmentPixelFormat = (MTLPixelFormat)k->sfmt;
    MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
    const nvn_refl *rf = &vs->rf;
    for (int v = 0; v < rf->nv; v++) {
        int loc = rf->v[v].loc, a = rf->v[v].nvn;   // Metal slot loc ← NVN attribute a
        MTLVertexFormat vf = a < 16 ? vfmt(k->attr[a].fmt) : MTLVertexFormatInvalid;
        // When a >= S.nattr, k.attr[a] keeps the 0 from memset, vfmt(0) is Invalid, the same fallback branch is taken and the RED_ONCE condition of fmt being non-0 does not hold.
        if (a < 16 && k->attr[a].fmt && vf == MTLVertexFormatInvalid) RED_ONCE(15, "顶点属性格式 %#x 未映射", k->attr[a].fmt);
        if (vf == MTLVertexFormatInvalid) {
            int ct = rf->v[v].ct;
            vd.attributes[loc].format = ct == NVN_CT_UINT ? MTLVertexFormatUInt4 : ct == NVN_CT_INT ? MTLVertexFormatInt4 : MTLVertexFormatFloat4;
            vd.attributes[loc].offset = 0; vd.attributes[loc].bufferIndex = 30;
            vd.layouts[30].stride = 16; vd.layouts[30].stepFunction = MTLVertexStepFunctionConstant; vd.layouts[30].stepRate = 0;
            continue;
        }
        int st = k->attr[a].stream;
        vf = vfmt_for_ct(vf, rf->v[v].ct, k->attr[a].fmt);
        vd.attributes[loc].format = vf; vd.attributes[loc].offset = k->attr[a].offset; vd.attributes[loc].bufferIndex = VB_SLOT(st);
        vd.layouts[VB_SLOT(st)].stride = k->stream[st].stride ? k->stream[st].stride : 4;
        vd.layouts[VB_SLOT(st)].stepFunction = k->stream[st].stride == 0 ? MTLVertexStepFunctionConstant : k->stream[st].divisor ? MTLVertexStepFunctionPerInstance : MTLVertexStepFunctionPerVertex;
        vd.layouts[VB_SLOT(st)].stepRate = k->stream[st].stride == 0 ? 0 : k->stream[st].divisor ? k->stream[st].divisor : 1;
    }
    d.vertexDescriptor = vd;
    return d;
}
// Pipeline key: current state S + pass formats (normal pass; a merged pass additionally carries lmap / specialized function, filled in by pso_get)
static void pso_key_fill(pso_key *k, NvShader *vs, NvShader *fs, const uint32_t cfmt[8], uint32_t dfmt, uint32_t sfmt, uint32_t samples, int lmap) {
    memset(k, 0, sizeof *k);
    k->vs = (__bridge void *)vs; k->fs = (__bridge void *)fs;
    memcpy(k->cfmt, cfmt, sizeof k->cfmt); k->dfmt = dfmt; k->sfmt = sfmt; k->samples = samples;
    k->blend_en = S.color.blend_en; k->chmask = S.chmask.mask; k->a2c = S.ms.a2c;
    for (int i = 0; i < 8; i++) if (k->cfmt[i] || lmap) k->blend[i] = S.blend[i];
    const nvn_refl *rf = &vs->rf;
    for (int v = 0; v < rf->nv; v++) { int a = rf->v[v].nvn; if (a < 16 && a < S.nattr) { k->attr[a] = S.attr[a]; int st = S.attr[a].stream; if (st < 16) k->stream[st] = S.stream[st]; } }
}
// ---- Parallel precompile: pipelines met for the first time used to be compiled synchronously one by one on the submit thread; 9 misses in a row in one frame = 284 ms (heavy scene on a real device).
// Now: on the first pipeline miss in a submission, scan ahead once along the remaining commands (replaying state only, no drawing) and kick all the normal-pass pipelines that will also miss later to the background to compile in parallel;
// when the submit thread reaches that draw it only waits for 'the remaining time not yet compiled'. The compiled pipelines are identical to synchronously compiled ones (same descriptor). Pipeline keys of merged passes / specialized fragments depend on the planning result at replay time, are not predicted, and are compiled synchronously as before.
#define PSO_PEND_MAX 384
static struct { pso_key k; id<MTLRenderPipelineState> ps; int used, done; } g_pp[PSO_PEND_MAX];
static pthread_mutex_t g_pp_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_pp_cv = PTHREAD_COND_INITIALIZER;
static uint64_t g_pp_submit_mark, g_pp_kicked, g_pp_taken;   // Whether this submission has already been predicted (by submission serial); cumulative kicked / taken counts
static void pso_predict_rest(void);
static int pso_pend_find(const pso_key *k) { for (int i = 0; i < PSO_PEND_MAX; i++) if (g_pp[i].used && !memcmp(&g_pp[i].k, k, sizeof *k)) return i; return -1; }
static void pso_pend_kick(const pso_key *k, NvShader *vs, NvShader *fs) {   // Called by the submit thread: kick only if it is neither in the cache nor in the table
    if (hm_get_id(&g_pso, k)) return;
    pthread_mutex_lock(&g_pp_mu);
    int at = pso_pend_find(k), slot = -1;
    if (at < 0) for (int i = 0; i < PSO_PEND_MAX; i++) if (!g_pp[i].used) { slot = i; break; }
    if (at < 0 && slot >= 0) { g_pp[slot].k = *k; g_pp[slot].ps = nil; g_pp[slot].done = 0; g_pp[slot].used = 1; }
    pthread_mutex_unlock(&g_pp_mu);
    if (at >= 0 || slot < 0) return;
    MTLRenderPipelineDescriptor *d = pso_desc(k, vs, fs, nil, 0); g_pp_kicked++;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        NSError *e = nil; id<MTLRenderPipelineState> ps = [g_dev newRenderPipelineStateWithDescriptor:d error:&e];
        pthread_mutex_lock(&g_pp_mu); g_pp[slot].ps = ps; g_pp[slot].done = 1; pthread_cond_broadcast(&g_pp_cv); pthread_mutex_unlock(&g_pp_mu);
    });
}
static id<MTLRenderPipelineState> pso_pend_take(const pso_key *k) {   // Called by the submit thread: if it is in the table, wait for it to finish compiling and take it; otherwise return nil
    pthread_mutex_lock(&g_pp_mu);
    int at = pso_pend_find(k); id<MTLRenderPipelineState> ps = nil;
    if (at >= 0) { while (!g_pp[at].done) pthread_cond_wait(&g_pp_cv, &g_pp_mu); ps = g_pp[at].ps; g_pp[at].ps = nil; g_pp[at].used = 0; }
    pthread_mutex_unlock(&g_pp_mu);
    return ps;
}
static __unsafe_unretained id<MTLRenderPipelineState> pso_get_(NvProg *pr, MTLRenderPassDescriptor *unused);
static id<MTLFunction> ms_fs_override(NvShader *fs);   // DM_NVN_MAGSCAN (defined before draw)
static __unsafe_unretained id<MTLRenderPipelineState> pso_get(NvProg *pr, MTLRenderPassDescriptor *unused) { AP_B; __unsafe_unretained id<MTLRenderPipelineState> r = pso_get_(pr, unused); AP_E(2); return r; }
static __unsafe_unretained id<MTLRenderPipelineState> pso_get_(NvProg *pr, MTLRenderPassDescriptor *unused) {
    (void)unused;
    __unsafe_unretained NvShader *vs = pr->sh[0], *fs = pr->sh[1];
    if (!vs) { RED_ONCE(13, "程序没有顶点着色器（或未映射）"); return nil; }
    pso_key k; pso_key_fill(&k, vs, fs, g_pass.cfmt, g_pass.dfmt, g_pass.sfmt, g_pass.samples, g_pso_lmap != NULL);
    if (g_pso_lmap) { memcpy(k.lmap, g_pso_lmap, 8); k.fsfn = (__bridge void *)g_pso_fsfn; }
    id<MTLFunction> ms_fn = ms_fs_override(fs); if (ms_fn) k.fsfn = (__bridge void *)ms_fn;   // DM_NVN_MAGSCAN phases 2 / 3
    __unsafe_unretained id<MTLRenderPipelineState> cached = hm_get_id(&g_pso, &k);
    if (cached) return cached;
    if (hm_get_id(&g_pso_fail, &k)) return nil;
    id<MTLRenderPipelineState> ps;
    NSError *e = nil; uint64_t bt = bp_now();
    ps = g_pso_lmap || ms_fn ? nil : pso_pend_take(&k);
    if (ps) g_pp_taken++;
    else {
        if (g_pp_submit_mark != g_sub_serial) { g_pp_submit_mark = g_sub_serial; pso_predict_rest(); }   // First miss in this submission: first kick the later ones out to compile in parallel, then synchronously compile the one at hand
        MTLRenderPipelineDescriptor *d = pso_desc(&k, vs, fs, ms_fn ? ms_fn : g_pso_fsfn, g_pso_lmap != NULL);
        ps = [g_dev newRenderPipelineStateWithDescriptor:d error:&e];
    }
    g_bp_pso_n++; g_bp_pso_ns += bp_now() - bt;
    if (!ps) { MLOG("【红】建渲染管线失败（顶点 %s 片元 %s）：%s", vs->key, fs ? fs->key : "-", e.localizedDescription.UTF8String); hm_put_id(&g_pso_fail, &k, NSNull.null); return nil; }
    if (!ms_fn) { pso_skey sk; char shex[sizeof sk * 2 + 1]; pso_skey_from(&k, vs, fs, g_pso_lmap != NULL, g_pso_lmap ? &g_pso_fz : NULL, &sk); pso_skey_hex(&sk, shex);
        pso_record("R\t%s\n", shex);
        pso_user_record("R\t%s\n", shex); }
    hm_put_id(&g_pso, &k, ps);
    return ps;
}
static NSString *warm_key(const char *s, size_t n) {
    size_t z = strnlen(s, n);
    return [[NSString alloc] initWithBytes:s length:z encoding:NSUTF8StringEncoding];
}
static NvShader *warm_shader(NSMutableDictionary<NSString *, NvShader *> *dict, NSString *key, int stage) {
    NvShader *sh = dict[key];
    if (!sh) { sh = shader_load(key, stage); if (sh) dict[key] = sh; }
    return sh;
}
static void warm_enqueue(WarmItem *item) {
    os_unfair_lock_lock(&g_warm_lock);
    [g_warm_items addObject:item];
    atomic_store_explicit(&g_warm_ready, 1, memory_order_release);
    os_unfair_lock_unlock(&g_warm_lock);
}
static void warm_worker(NSArray<NSString *> *paths) {
    @autoreleasepool {
        NSMutableDictionary<NSString *, NvShader *> *dict = [NSMutableDictionary dictionary];
        NSMutableSet<NSString *> *seen = [NSMutableSet set];
        for (NSString *path in paths) {
            NSString *txt = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:nil];
            if (!txt) { MLOG("【红】预热清单读取失败：%s", path.UTF8String); continue; }
            { unsigned long nl = 0; for (NSString *l in [txt componentsSeparatedByString:@"\n"]) nl += l.length && ![l hasPrefix:@"#"];
              MLOG("管线预热清单 %s：%lu 行", path.UTF8String, nl); }   // Diagnostics: how many lines each manifest gave (when a device warms up 0, shows which one was missing)
            for (NSString *line in [txt componentsSeparatedByString:@"\n"]) {
                if (!line.length || [line hasPrefix:@"#"] || [seen containsObject:line]) continue;
                [seen addObject:line];
                NSArray<NSString *> *c = [line componentsSeparatedByString:@"\t"];
                if (!c.count) continue;
                if ([c[0] isEqualToString:@"S"] && c.count >= 3) {
                    NSString *key = c[1]; NvShader *sh = warm_shader(dict, key, c[2].intValue);
                    if (!sh) continue;
                    WarmItem *item = [WarmItem new]; item->kind = 0; item->key = key; item->shader = sh;
                    warm_enqueue(item); g_warm_shaders++;
                } else if ([c[0] isEqualToString:@"R"] && c.count >= 2) {
                    pso_skey sk; if (pso_skey_parse(c[1].UTF8String, &sk)) continue;
                    NSString *vkey = warm_key(sk.vs, sizeof sk.vs), *fkey = warm_key(sk.fs, sizeof sk.fs);
                    NvShader *vs = vkey.length ? warm_shader(dict, vkey, sk.vs_stage) : nil;
                    NvShader *fs = fkey.length ? warm_shader(dict, fkey, sk.fs_stage) : nil;
                    if (!vs || (fkey.length && !fs)) continue;
                    id<MTLFunction> fsfn = nil;
                    if (sk.fz) {
                        if (sk.fz_desc.kind == 1 && fs) { int fx[32]; for (int i = 0; i < 32; i++) fx[i] = sk.fz_desc.fx[i]; fsfn = bf_fn(fs, fx, sk.fz_desc.fbmask, sk.k.lmap); }
                                                if (!fsfn) { g_warm_skip_fz++; continue; }
                    }
                    pso_key k = sk.k; k.vs = (__bridge void *)vs; k.fs = (__bridge void *)fs; k.fsfn = (__bridge void *)fsfn;
                    NSError *e = nil; id ps = [g_dev newRenderPipelineStateWithDescriptor:pso_desc(&k, vs, fs, fsfn, sk.merged) error:&e];
                    if (!ps) { MLOG("【红】后台建渲染管线失败：%s", e.localizedDescription.UTF8String); continue; }
                    WarmItem *item = [WarmItem new]; item->kind = 1; item->vs_key = vkey; item->fs_key = fkey; item->vs = vs; item->fs = fs; item->fsfn = fsfn; item->pso = k; item->object = ps;
                    warm_enqueue(item); g_warm_rps++;
                } else if ([c[0] isEqualToString:@"C"] && c.count >= 2) {
                    NSString *key = c[1]; if (!key.length) continue;
                    NvShader *cs = warm_shader(dict, key, 5);
                    if (!cs) continue;
                    NSError *e = nil; id ps = [g_dev newComputePipelineStateWithFunction:cs->fn error:&e];
                    if (!ps) { MLOG("【红】后台建计算管线失败：%s", e.localizedDescription.UTF8String); continue; }
                    WarmItem *item = [WarmItem new]; item->kind = 2; item->key = key; item->shader = cs; item->object = ps;
                    warm_enqueue(item); g_warm_cps++;
                }
            }
        }
        MLOG("管线预热完成：着色器 %llu 个 / 渲染管线 %llu 个（跳过特化片元 %llu）/ 计算管线 %llu 个，后台用时 %.1f ms",
             g_warm_shaders, g_warm_rps, g_warm_skip_fz, g_warm_cps, (bp_now() - g_warm_t0) / 1e6);
    }
    atomic_store(&g_warm_running, 0);
}
// Wait for the background warm-up to finish (at most timeout_ms): returns how many milliseconds were actually waited, 0 if it is not running. On first install the system shader cache is fully cold; if warm-up is slower than the player entering a run,
// pipelines not yet warmed are compiled on the spot on the submit thread = in-run stutter; the loading screen on entering a run blocks the player anyway, so let it load a little longer (called where hle_nvn.c presents).
int mtl_warm_wait(int timeout_ms) {
    if (!atomic_load(&g_warm_running)) return 0;
    uint64_t t0 = bp_now();
    while (atomic_load(&g_warm_running) && (bp_now() - t0) / 1000000ull < (uint64_t)timeout_ms) usleep(2000);
    return (int)((bp_now() - t0) / 1000000ull) + 1;
}
static void warm_start(void) {
    if (g_warm_started++) return;
    const char *e = getenv("DM_PSO_MANIFEST");
    if (e && !strcmp(e, "off")) return;
    NSArray<NSString *> *paths;
    if (e && *e) {
        paths = @[@(e)];
    } else {
        NSString *seed = [g_shmap_path.stringByDeletingLastPathComponent stringByAppendingPathComponent:@"pso_manifest.tsv"];
        NSMutableArray<NSString *> *all = [NSMutableArray arrayWithObject:seed];
        NSString *user = pso_user_path();
        if (user && [NSFileManager.defaultManager fileExistsAtPath:user]) [all addObject:user];
        else MLOG("管线预热：本机用户清单%s%s", user ? "还没有：" : "没有路径（DM_PSO_USER / DM_SAVE 都没设）", user ? user.UTF8String : "");
        if (![NSFileManager.defaultManager fileExistsAtPath:seed]) { MLOG("【红】管线预热：种子清单不存在：%s", seed.UTF8String); return; }
        paths = all;
    }
    if (!g_warm_items) g_warm_items = [NSMutableArray array];
    g_warm_t0 = bp_now();
    dispatch_queue_attr_t attr = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INITIATED, 0);   // Not optional background work: it decides whether the pipelines are ready before the player enters a run
    dispatch_queue_t q = dispatch_queue_create("dm.nvn.pso-warm", attr);
    atomic_store(&g_warm_running, 1);
    dispatch_async(q, ^{ warm_worker(paths); });
}
static int g_rt_ovr;   // Stencil override for runtime extrapolation: 0 no override; 1 pass only on the unit / effect / extrapolation-hole bits (0xD0), no stencil write; 3 depth test as usual, write the effect bit 0x40 where it passes
static id<MTLDepthStencilState> dss_get(int has_depth, int has_stencil) {
    struct { ds_st ds; uint32_t vm[2], wm[2]; int hd, hs, ovr; } k;
    memset(&k, 0, sizeof k); k.ds = S.ds; memcpy(k.vm, S.st_vmask, 8); memcpy(k.wm, S.st_wmask, 8); k.hd = has_depth; k.hs = has_stencil; k.ovr = has_stencil ? g_rt_ovr : 0;
    if (!g_dss.klen) g_dss.klen = sizeof k;
    id<MTLDepthStencilState> s = hm_get_id(&g_dss, &k);
    if (s) return s;
    MTLDepthStencilDescriptor *d = [MTLDepthStencilDescriptor new];
    if (has_depth && S.ds.dt) { d.depthCompareFunction = (MTLCompareFunction)(S.ds.df - 1); d.depthWriteEnabled = S.ds.dw; }
    if (has_stencil && S.ds.st) {
        MTLStencilDescriptor *f = [MTLStencilDescriptor new], *b = [MTLStencilDescriptor new];
        f.stencilCompareFunction = (MTLCompareFunction)(S.ds.sf_f - 1); f.stencilFailureOperation = (MTLStencilOperation)(S.ds.fail_f - 1);
        f.depthFailureOperation = (MTLStencilOperation)(S.ds.zfail_f - 1); f.depthStencilPassOperation = (MTLStencilOperation)(S.ds.zpass_f - 1);
        f.readMask = S.st_vmask[0]; f.writeMask = S.st_wmask[0];
        b.stencilCompareFunction = (MTLCompareFunction)(S.ds.sf_b - 1); b.stencilFailureOperation = (MTLStencilOperation)(S.ds.fail_b - 1);
        b.depthFailureOperation = (MTLStencilOperation)(S.ds.zfail_b - 1); b.depthStencilPassOperation = (MTLStencilOperation)(S.ds.zpass_b - 1);
        b.readMask = S.st_vmask[1]; b.writeMask = S.st_wmask[1];
        d.frontFaceStencil = f; d.backFaceStencil = b;
    }
    if (has_stencil && k.ovr) {
        MTLStencilDescriptor *f = [MTLStencilDescriptor new];
        if (k.ovr == 1) { f.stencilCompareFunction = MTLCompareFunctionNotEqual; f.readMask = 0xD0; f.writeMask = 0; }   // Reference value 0: passes only if (s & 0xD0) != 0 (unit 0x10, effect 0x40, extrapolation hole / screen edge 0x80)
        else { f.stencilCompareFunction = MTLCompareFunctionAlways; f.depthStencilPassOperation = MTLStencilOperationReplace; f.readMask = 0; f.writeMask = 0x40; }   // Reference value 0x40
        d.frontFaceStencil = f; d.backFaceStencil = f;
    }
    s = [g_dev newDepthStencilStateWithDescriptor:d];
    hm_put_id(&g_dss, &k, s);
    return s;
}
// Argument buffer: filled item by item per the B reflection (uniform slot r, separate texture / sampler slot r, storage buffer slot r, driver segments c0 / c2, c1, support_buffer)
// Unbound resources read 0: give all-zero dummy resources (NVN descriptors 0..255 are reserved for the driver; the game binding handle 0 means 'this slot is empty')
static uint64_t g_dummy_tex_rid[8][4], g_dummy_img_rid[8][4], g_dummy_tbuf_rid[4];
static id<MTLTexture> dummy_tex(int dtt, int dpf) {
    static __strong id<MTLTexture> cache[8][4];
    MTLTextureType tt = (MTLTextureType)dtt;
    MTLPixelFormat pf = dpf == NVN_DPF_DEPTH ? MTLPixelFormatDepth32Float : dpf == NVN_DPF_UINT ? MTLPixelFormatRGBA32Uint : dpf == NVN_DPF_SINT ? MTLPixelFormatRGBA32Sint : MTLPixelFormatRGBA8Unorm;
    id<MTLTexture> t = cache[dtt & 7][dpf & 3];
    if (t) return t;
    MTLTextureDescriptor *d = [MTLTextureDescriptor new];
    d.textureType = tt; d.pixelFormat = pf; d.width = d.height = d.depth = 1; d.storageMode = MTLStorageModeShared; d.usage = MTLTextureUsageShaderRead;
    t = [g_dev newTextureWithDescriptor:d];   // Shared storage is all zeros when newly created
    if (pf != MTLPixelFormatDepth32Float) { uint8_t z[64] = {0}; NSUInteger faces = tt == MTLTextureTypeCube || tt == MTLTextureTypeCubeArray ? 6 : 1;
        for (NSUInteger f = 0; f < faces; f++) [t replaceRegion:MTLRegionMake3D(0, 0, 0, 1, 1, 1) mipmapLevel:0 slice:f withBytes:z bytesPerRow:16 bytesPerImage:16]; }
    cache[dtt & 7][dpf & 3] = t; g_dummy_tex_rid[dtt & 7][dpf & 3] = t.gpuResourceID._impl; res_add(t);
    return t;
}
// Write sink for unbound image slots: kept separate from the dummy textures that read 0 (writes must not pollute them); one is shared per same type and same pixel class
static id<MTLTexture> dummy_img(int dtt, int dpf) {
    static __strong id<MTLTexture> cache[8][4];
    id<MTLTexture> t = cache[dtt & 7][dpf & 3]; if (t) return t;
    MTLTextureDescriptor *d = [MTLTextureDescriptor new];
    d.textureType = (MTLTextureType)dtt; d.pixelFormat = dpf == NVN_DPF_UINT ? MTLPixelFormatRGBA32Uint : dpf == NVN_DPF_SINT ? MTLPixelFormatRGBA32Sint : MTLPixelFormatRGBA16Float;
    d.width = d.height = d.depth = 1; d.storageMode = MTLStorageModePrivate; d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    t = [g_dev newTextureWithDescriptor:d]; cache[dtt & 7][dpf & 3] = t; g_dummy_img_rid[dtt & 7][dpf & 3] = t.gpuResourceID._impl; res_add(t); return t;
}
static id<MTLBuffer> dummy_buf(void);
static id<MTLTexture> dummy_tbuf(int dpf) {   // Unbound typed buffer: all zeros (pixel class matches the shader's texture_buffer<T>)
    static __strong id<MTLTexture> cache[4];
    id<MTLTexture> t = cache[dpf & 3]; if (t) return t;
    MTLPixelFormat pf = dpf == NVN_DPF_UINT ? MTLPixelFormatRGBA32Uint : dpf == NVN_DPF_SINT ? MTLPixelFormatRGBA32Sint : MTLPixelFormatRGBA32Float;
    id<MTLBuffer> b = dummy_buf();
    MTLTextureDescriptor *d = [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:pf width:b.length / 16 resourceOptions:b.resourceOptions usage:MTLTextureUsageShaderRead];
    t = [b newTextureWithDescriptor:d offset:0 bytesPerRow:b.length];
    cache[dpf & 3] = t; g_dummy_tbuf_rid[dpf & 3] = t.gpuResourceID._impl; return t;
}
static id<MTLSamplerState> dummy_samp(void) {
    static id<MTLSamplerState> s; if (!s) { MTLSamplerDescriptor *d = [MTLSamplerDescriptor new]; d.supportArgumentBuffers = YES; s = [g_dev newSamplerStateWithDescriptor:d]; }
    return s;
}
static uint64_t dummy_samp_rid(void) { static uint64_t rid; if (!rid) rid = dummy_samp().gpuResourceID._impl; return rid; }
static id<MTLBuffer> dummy_buf(void) { static id<MTLBuffer> b; if (!b) { b = [g_dev newBufferWithLength:64 << 10 options:MTLResourceStorageModeShared]; res_add(b); } return b; }
static uint64_t dummy_buf_gpu(void) { static uint64_t gpu; if (!gpu) gpu = dummy_buf().gpuAddress; return gpu; }
// useResource is idempotent for the same (resource, usage, stage) within one encoder: remember what this render encoder has already declared and skip repeats (a compute encoder is created anew for each dispatch, no dedup)
// Snapshot substitution for same-pass feedback sampling (registered by fb_prepare, used by encode_args; see the notes before draw)
static uint64_t g_draw_seq, g_fb_seq, g_fb_count; static int g_fb_n, g_fb_rebind;
static __unsafe_unretained id<MTLTexture> g_fb_src[4]; static __strong id<MTLTexture> g_fb_use[4]; static uint64_t g_fb_rid[4];
static uint8_t g_args_dirty[6] = {1, 1, 1, 1, 1, 1};
static uint64_t g_args_reg_seen, g_args_dirty_seen;
static inline void args_dirty_all(void) { memset(g_args_dirty, 1, sizeof g_args_dirty); }
static inline void args_sync_dirty(void) {
    uint64_t reg = g_reg_gen, dirty = nvn_dirty_now();
    if (reg != g_args_reg_seen || dirty != g_args_dirty_seen) {
        g_args_reg_seen = reg; g_args_dirty_seen = dirty; args_dirty_all();
    }
}
static void bind_args(id<MTLCommandEncoder> en, int nvn_stage, int metal_vertex, id<MTLBuffer> ab, uint64_t gpu, size_t off, NSUInteger ai) {
    if (nvn_stage == 5) enc_c_args((id<MTLComputeCommandEncoder>)en, gpu, ai);
    else enc_r_args((id<MTLRenderCommandEncoder>)en, metal_vertex ? MTLRenderStageVertex : MTLRenderStageFragment, gpu, ai);
}
// Static constant buffers of route B: the Ryujinx support_buffer (layout in Ryujinx.Graphics.Shader/SupportBuffer.cs: alpha_test u32, is_bgra u32[8],
// viewport_inverse f4, viewport_size f4, frag_scale_count i32, render_scale f32[73]..., each field aligned to 16 bytes) and the all-0 driver constant segment (64 KB = the upper bound declared by shaders)
static id<MTLBuffer> support_buf(void) {
    static __strong id<MTLBuffer> b; if (b) return b;
    b = [g_dev newBufferWithLength:65536 options:MTLResourceStorageModeShared]; res_add(b);   // All 0 when newly created
    const int render_scale_off = 192;   // Measured MSL layout: alpha_test 0, is_bgra[8] 16, viewport_inverse 144, viewport_size 160, frag_scale_count 176, render_scale float4[73] 192 (take .x)
    for (int i = 0; i < 73; i++) *(float *)((uint8_t *)b.contents + render_scale_off + 16 * i) = 1.0f;
    return b;
}
static uint64_t support_gpu(void) { static uint64_t gpu; if (!gpu) gpu = support_buf().gpuAddress; return gpu; }
static id<MTLBuffer> zero_bank(void) { static __strong id<MTLBuffer> b; if (!b) { b = [g_dev newBufferWithLength:65536 options:MTLResourceStorageModeShared]; res_add(b); } return b; }
static uint64_t zero_bank_gpu(void) { static uint64_t gpu; if (!gpu) gpu = zero_bank().gpuAddress; return gpu; }
// Shift all UI draws as a whole (iPad true fullscreen, the UI rect does not touch the top edge of the picture; and a canvas wider
// than 16:9, where the game anchors its UI box to the top-left corner): add a clip-space offset to the projection matrix at the
// start of the vertex stage's cb0, and bind a modified copy instead.
// Moving the viewport is not used: the viewport's top edge is the edge of the clip volume, so top-anchored widgets that the host
// lifts back to the real top of the picture (UI coordinate y negative) would be clipped away entirely.
// Both offsets add the w column into the x or y column, i.e. x_clip += o * w_clip, which after the shader's divide is exactly a
// screen-space translation of 2o pixels per unit of o. g_ui_moff / g_ui_mxoff are set per draw by draw() (0 = no change);
// the copies live in a ring buffer, and the same address in the same pass with the same offsets is copied only once.
static double g_ui_moff, g_ui_mxoff;
static int ui_mat_ok(void) { const p_bind_buf *u = &S.ubo[0][0]; void *mb; uint64_t mo; return u->addr && u->size >= 64 && u->size <= 4096 && !pend_hits(u->addr, u->size) && mtl_addr(u->addr, u->size, &mb, &mo); }
static uint64_t ui_mat_patch(const p_bind_buf *u, void *mb, uint64_t mo) {
    enum { RING = 4 << 20 }; static __strong id<MTLBuffer> ring; static uint64_t ring_gpu, la, ls, lp, lg; static size_t at; static double lo, lox;
    if (!ring) { ring = [g_dev newBufferWithLength:RING options:MTLResourceStorageModeShared]; res_add(ring); ring_gpu = ring.gpuAddress; }
    if (la == u->addr && ls == u->size && lp == g_pass.serial && lo == g_ui_moff && lox == g_ui_mxoff) return lg;
    const uint8_t *src = [(__bridge id<MTLBuffer>)mb contents]; if (!src || u->size < 64 || u->size > 4096) return 0;
    size_t n = (u->size + 255) & ~(size_t)255; if (at + n > RING) at = 0;
    uint8_t *dst = (uint8_t *)ring.contents + at; memcpy(dst, src + mo, u->size);
    float o = (float)g_ui_moff, ox = (float)g_ui_mxoff;
    // This game's UI vertex shader reads the block with the row-vector convention (x = sum_i v_i * row[i].x, v.w = 1), so a shift is
    // "row[i].xy += offset * row[i].w" over the whole column -- the same form the vertical offset has always used on the first block.
    // The shader carries TWO such transforms in the block (rows 0-3 and rows 4-7, picked per vertex by the sign of attribute 1's z),
    // so both axes are applied to every leading 4x4 block; a block of zeros is not a transform and is left alone.
    if (o != 0 || ox != 0) for (size_t bo = 0; bo + 64 <= u->size && bo < 256; bo += 64) {
        float *m = (float *)(dst + bo);
        if (!(m[0] || m[4] || m[8] || m[12])) continue;
        if (o != 0) { m[1] += o * m[3]; m[5] += o * m[7]; m[9] += o * m[11]; m[13] += o * m[15]; }
        if (ox != 0) { m[0] += ox * m[3]; m[4] += ox * m[7]; m[8] += ox * m[11]; m[12] += ox * m[15]; }
    }
    { // DM_UI_MATDUMP=1: print the uniform block the UI draws actually read (first two distinct blocks), to check the layout by hand
        static int md = -1; if (md < 0) md = getenv("DM_UI_MATDUMP") != NULL;
        static uint64_t ma[2], mp[2]; static int mn;
        if (md && mn < 2) { int hit = 0; for (int q = 0; q < mn; q++) hit |= ma[q] == u->addr && mp[q] == g_pass.serial;
            if (!hit) { ma[mn] = u->addr; mp[mn] = g_pass.serial; mn++;
                MLOG("UI cb0：%llu 字节，通道 %llu，偏移 y=%g x=%g（存盘地址 %#llx）", (unsigned long long)u->size, (unsigned long long)g_stat_pass, g_ui_moff, g_ui_mxoff, (unsigned long long)u->addr);
                for (size_t r = 0; r * 16 + 16 <= u->size && r < 16; r++) { const float *f = (const float *)(dst + r * 16), *g = (const float *)(src + mo + r * 16);
                    MLOG("  行%zu  %+.6f %+.6f %+.6f %+.6f   原 %+.6f", r, f[0], f[1], f[2], f[3], g[0]); } } }
    }
    la = u->addr; ls = u->size; lp = g_pass.serial; lo = g_ui_moff; lox = g_ui_mxoff; lg = ring_gpu + at; at += n;
    return lg;
}
// Bindless handle (TEXU / SAMPU): the 32-bit word at byte offset off of uniform slot reg = the low 32 bits of the NVN handle the game wrote there (low 20 bits texture ID, high 12 bits sampler ID).
// Read on the CPU side at replay time (the memory at the same moment a real GPU would read it at execution); slot unbound / out of range / unreadable = 0 (empty slot: dummy texture / dummy sampler).
// A location with a GPU-side write queued in this submit (pend_hits) is still read at its current value; the first case is reported red.
static uint32_t ubo_handle(int stage, int reg, int off) {
    if (stage < 0 || stage >= 6 || reg < 0 || reg >= 16) return 0;
    const p_bind_buf *u = &S.ubo[stage][reg];
    if (!u->addr || (uint64_t)off + 4 > u->size) return 0;
    if (g_nubd && ubd_hits(u->addr + off, u->addr + off + 4) && ubd_covers(u->addr + off, u->addr + off + 4)) { uint32_t h; ubd_read(u->addr + off, 4, &h); return h; }   // written by an inline update deferred in this pass
    if (pend_hits(u->addr + off, 4)) RED_ONCE(47, "无绑定句柄所在的 uniform 缓冲（阶段 %d 槽 %d +%#x）本提交里有 GPU 侧写入：读的是当前值", stage, reg, off);
    void *mb; uint64_t mo; const uint8_t *base;
    if (!mtl_addr(u->addr + off, 4, &mb, &mo) || !(base = [(__bridge id<MTLBuffer>)mb contents])) return 0;
    uint32_t h; memcpy(&h, base + mo, 4); return h;
}
static int encode_args_(id<MTLCommandEncoder> en, NvShader *sh, int nvn_stage, int metal_vertex);
static int encode_args(id<MTLCommandEncoder> en, NvShader *sh, int nvn_stage, int metal_vertex) { AP_B; int r = encode_args_(en, sh, nvn_stage, metal_vertex); AP_E(3); return r; }
static int encode_args_(id<MTLCommandEncoder> en, NvShader *sh, int nvn_stage, int metal_vertex) {
    if (!sh->ae) return 1;
    uint64_t slots[NVN_REFL_MAXB] = {0};
    args_sync_dirty();
    if (!passdump() && !g_args_dirty[nvn_stage] && sh->ab_buf && sh->ab_pass == g_pass.serial && nvn_stage != 5) {
        bind_args(en, nvn_stage, metal_vertex, sh->ab_buf, sh->ab_gpu, sh->ab_off, sh->rf.abi);
        g_stat_args_reuse++;
        return 1;
    }
    #define USE(r, u) do { (void)(u); if (__builtin_expect(g_res_check_on, 0)) res_check((r), sh->key, reg); } while (0)
    // DM_NVN_PASSDUMP: on the first new encode per pass per stage, records the shader key (K) and the actual source of each binding (B: UBO address / length / first 4 floats, texture ID and serial, or 「哑」(dummy))
    static uint64_t bl_pass[6]; static int bl_all = -1; if (bl_all < 0) bl_all = getenv("DM_NVN_PASSDUMP_ALL") != NULL;   // Record on every new encode (otherwise only the first time per pass per stage)
    int bl = passdump() && (bl_all || g_kd_force || bl_pass[nvn_stage] != g_stat_pass);
    if (bl) { bl_pass[nvn_stage] = g_stat_pass; PD("K %llu st%d key=%s nb=%d", g_stat_pass, nvn_stage, sh->key[0] ? sh->key : sh->builtin ? sh->builtin : "?", sh->rf.nb); }
    { // DM_UI_MATDUMP=1: the reflected bindings of the UI vertex shader, once (the matrix patch only fires on stage 0 / register 0)
        static int md = -1; if (md < 0) md = getenv("DM_UI_MATDUMP") != NULL;
        static int n; if (md && nvn_stage == 0 && n < 3 && DM_GAME_UI_VS_KEY[0] && !strcmp(sh->key, DM_GAME_UI_VS_KEY)) { n++;
            MLOG("UI 顶点着色器反射：%d 个绑定", sh->rf.nb);
            for (int i = 0; i < sh->rf.nb; i++) MLOG("  b%d kind=%d reg=%d off=%d aid=%d", i, (int)sh->rf.b[i].kind, sh->rf.b[i].reg, (int)sh->rf.b[i].off, (int)sh->rf.b[i].aid); } }
    pthread_mutex_lock(&g_rmu);   // The texture / sampler registration tables may be being grown or rewritten by the game thread
    for (int i = 0; i < sh->rf.nb; i++) {
        const nvn_rbind *b = &sh->rf.b[i];
        int reg = b->reg, aid = b->aid;
        if (b->kind == NVN_RB_CBUF) {
            const p_bind_buf *u = reg < 16 ? &S.ubo[nvn_stage][reg] : NULL;
            void *mb; uint64_t mo, gpu;
            if (!u || !u->addr || !mtl_gpu_addr(u->addr, 1, &mb, &mo, &gpu)) { RED_ONCE(16, "阶段 %d cb%d 没有绑定 uniform 缓冲", nvn_stage, reg); continue; }
            use_rec("buf", u->addr, u->size, 0, NULL);
            if (g_nubd && ubd_hits(u->addr, u->addr + u->size)) {   // inline updates deferred in this pass (see mtl_sub_update); ubd_conflict made sure they cover the binding
                if (ubd_covers(u->addr, u->addr + u->size)) gpu = ubd_shadow(u->addr, u->addr + u->size);
                else RED_ONCE(92, "通道内推迟的 uniform 更新只盖住绑定的一部分（阶段 %d cb%d）：读的是更新前的内容", nvn_stage, reg); }
            if (nvn_stage == 0 && reg == 0 && (g_ui_moff != 0 || g_ui_mxoff != 0)) { uint64_t pg = ui_mat_patch(u, mb, mo); if (pg) gpu = pg;
                { static int md = -1; if (md < 0) md = getenv("DM_UI_MATDUMP") != NULL; static uint64_t lp; static int n;
                  if (md && n < 24 && lp != g_stat_pass) { lp = g_stat_pass; n++;
                      MLOG("补丁生效：通道 %llu 着色器 %s cb0 %#llx → %#llx（%s）", (unsigned long long)g_stat_pass, sh->key, (unsigned long long)u->addr, (unsigned long long)gpu, pg ? "换缓冲" : "没换成"); } } }
            if (bl) {
                PD("B %llu st%d cb reg%d addr=%#llx size=%llu pool=%#llx+%#llx f=%g,%g,%g,%g", g_stat_pass, nvn_stage, reg, (unsigned long long)u->addr, (unsigned long long)u->size,
                   (unsigned long long)(u->addr - mo), (unsigned long long)mo, ((float *)u->addr)[0], ((float *)u->addr)[1], ((float *)u->addr)[2], ((float *)u->addr)[3]);
                // DM_NVN_CBDUMP=directory: also saves the whole uniform block to disk as-is (F<frame>_p<pass>_st<stage>_cb<slot>_<address>.bin), for the ground-truth side to compare byte by byte by guest address
                // DM_NVN_CBDUMP_KEY=shader key: only save for draws using that shader (any stage)
                static const char *cbd, *cbk; static int cbd_init; if (!cbd_init) { cbd = getenv("DM_NVN_CBDUMP"); cbk = getenv("DM_NVN_CBDUMP_KEY"); cbd_init = 1; }
                if (cbd && (!cbk || !strcmp(sh->key, cbk))) {
                    char p[1024]; snprintf(p, sizeof p, "%s/F%llu_p%llu_st%d_cb%d_%llx.bin", cbd, (unsigned long long)atomic_load(&g_cur_frame), g_stat_pass, nvn_stage, reg, (unsigned long long)u->addr);
                    FILE *f = fopen(p, "wb"); if (f) { fwrite((const void *)u->addr, 1, u->size, f); fclose(f); } else RED_ONCE(41, "CBDUMP 写不出 %s", p);
                }
            }
            slots[aid] = gpu;
            USE((__bridge id<MTLBuffer>)mb, MTLResourceUsageRead);
        } else if (b->kind == NVN_RB_SAMP || b->kind == NVN_RB_SAMPT || b->kind == NVN_RB_SAMPU) {   // SAMPT: the sampler part of a combined handle (texture ID | sampler ID << 20); SAMPU: the same, with the handle in a uniform buffer
            int sid = b->kind == NVN_RB_SAMPU ? (int)(ubo_handle(nvn_stage, reg, b->off) >> 20) :
                      reg < 128 ? (int)((b->kind == NVN_RB_SAMPT ? S.tex[nvn_stage][reg] : S.samp[nvn_stage][reg]) >> 20) : 0;
            __unsafe_unretained id<MTLSamplerState> ss = sid ? (__bridge id<MTLSamplerState>)samp_fast(sid) : nil;
            slots[aid] = sid && sid < g_nsreg && g_srid[sid] ? g_srid[sid] : dummy_samp_rid();
            USE(ss ? ss : dummy_samp(), MTLResourceUsageRead);
            if (bl && sid > 0) { int ok = sid < g_nsreg && g_sreg_ok[sid]; samp_par sp = ok ? g_sreg[sid] : (samp_par){0};   // g_rmu is already held in this section
                if (ok) PD("B %llu st%d samp reg%d sid=%d mag=%d min=%d（0 最近点 / 1 双线性；2–5 带 mip）wrap=%d,%d,%d border=%g,%g,%g,%g lod=%g..%g bias=%g aniso=%g cmp=%d/%d %s", g_stat_pass, nvn_stage, reg, sid, (int)sp.magf, (int)sp.minf,
                           (int)sp.ws, (int)sp.wt, (int)sp.wr, sp.border[0], sp.border[1], sp.border[2], sp.border[3], sp.lod_min, sp.lod_max, sp.lod_bias, sp.aniso, (int)sp.cmp_mode, (int)sp.cmp_func, ss ? "实" : "建不出"); }
        } else if (b->kind == NVN_RB_TEX || b->kind == NVN_RB_TEXU) {
            int tid = b->kind == NVN_RB_TEXU ? (int)(ubo_handle(nvn_stage, reg, b->off) & 0xfffff) : reg < 128 ? (int)(S.tex[nvn_stage][reg] & 0xfffff) : 0;
            void *tv = NULL; uint64_t trid = tid ? tex_fast_rid(tid, &tv) : 0; __unsafe_unretained id<MTLTexture> t = (__bridge id<MTLTexture>)tv;
            if (t && g_fb_n && g_fb_seq == g_draw_seq && nvn_stage == 1) for (int q = 0; q < g_fb_n; q++) if (t == g_fb_src[q]) { t = g_fb_use[q]; trid = g_fb_rid[q]; break; }   // same-pass feedback sampling: read the snapshot instead (see fb_prepare)
            if (t && tid < g_ntreg) use_rec("tex", (uint64_t)g_treg[tid].t.mem, g_treg[tid].t.L.size, tid, &g_treg[tid].t);
            if (t && passdump() && tid < g_ntreg) PD("R %llu st%d ser=%llu", g_stat_pass, nvn_stage, g_treg[tid].t.serial);
            if (bl) PD("B %llu st%d tex reg%d tid=%d %s ser=%llu fmt=%lu %lux%lux%lu view_lv=%lu+%lu view_sl=%lu array=%lu target=%lu swz=%d%d%d%d mtl=%p", g_stat_pass, nvn_stage, reg, tid, t ? "实" : "哑", t && tid < g_ntreg ? g_treg[tid].t.serial : 0ull,
                       (unsigned long)t.pixelFormat, (unsigned long)t.width, (unsigned long)t.height, (unsigned long)t.depth,
                       (unsigned long)(t.parentTexture ? t.parentRelativeLevel : 0), (unsigned long)t.mipmapLevelCount,
                       (unsigned long)(t.parentTexture ? t.parentRelativeSlice : 0), (unsigned long)t.arrayLength, (unsigned long)t.textureType, t ? (int)t.swizzle.red : 0, t ? (int)t.swizzle.green : 0, t ? (int)t.swizzle.blue : 0, t ? (int)t.swizzle.alpha : 0, (__bridge void *)(t.parentTexture ? t.parentTexture : t));
            if (t && t.textureType != (MTLTextureType)b->dtt) { uint64_t r2 = 0; id<MTLTexture> v = tex_fit(tid, t, (MTLTextureType)b->dtt, &r2); if (v) { t = v; trid = r2; } }
            if (t && nvn_stage == 1 && afr_want(sh)) { uint64_t r2 = 0; id<MTLTexture> v = tex_alpha_red(tid, t, &r2); if (v) { t = v; trid = r2; } }   // see tex_alpha_red
            if (!t) { t = dummy_tex(b->dtt, b->dpf); trid = g_dummy_tex_rid[b->dtt & 7][b->dpf & 3]; }
            slots[aid] = trid;
            USE(t, MTLResourceUsageRead);
        } else if (b->kind == NVN_RB_IMG) {   // Writable texture: NVN image slot r (BindImages) ↔ D3D u_r
            int tid = reg < 32 ? (int)(S.img[nvn_stage][reg] & 0xfffff) : 0;
            void *tv = NULL; uint64_t trid = tid ? tex_fast_rid(tid, &tv) : 0; __unsafe_unretained id<MTLTexture> t = (__bridge id<MTLTexture>)tv;
            if (passdump()) PD("W %llu st%d reg%d tid=%d %s ser=%llu", g_stat_pass, nvn_stage, reg, tid, t ? "实" : "哑", t && tid < g_ntreg ? g_treg[tid].t.serial : 0ull);
            if (t && !(t.usage & MTLTextureUsageShaderWrite)) { RED_ONCE(32, "图像槽绑的贴图没有可写用途（贴图号 %d）", tid); t = nil; }
            if (!t) { t = dummy_img(b->dtt, b->dpf); trid = g_dummy_img_rid[b->dtt & 7][b->dpf & 3]; }   // Unbound: D3D semantics discard the write - write into a dedicated dummy texture so the set that reads 0 is not polluted
            slots[aid] = trid;
            USE(t, MTLResourceUsageRead | MTLResourceUsageWrite);
        } else if (b->kind == NVN_RB_TBUF) {   // Typed buffer: the buffer texture (target BUFFER) in texture slot r
            int tid = reg < 128 ? (int)(S.tex[nvn_stage][reg] & 0xfffff) : 0;
            void *tv = NULL; uint64_t trid = tid ? tex_fast_rid(tid, &tv) : 0; __unsafe_unretained id<MTLTexture> t = (__bridge id<MTLTexture>)tv;
            if (t && t.textureType != MTLTextureTypeTextureBuffer) { RED_ONCE(40, "类型化缓冲槽绑的不是缓冲贴图（贴图号 %d 类型 %lu）", tid, (unsigned long)t.textureType); t = nil; }
            if (bl) PD("B %llu st%d tbuf reg%d tid=%d %s ser=%llu", g_stat_pass, nvn_stage, reg, tid, t ? "实" : "哑", t && tid < g_ntreg ? g_treg[tid].t.serial : 0ull);
            if (!t) { t = dummy_tbuf(b->dpf); trid = g_dummy_tbuf_rid[b->dpf & 3]; }
            slots[aid] = trid;
            USE(t, MTLResourceUsageRead);
        } else if (b->kind == NVN_RB_SBUFP || b->kind == NVN_RB_UBOP) {   // B: storage buffers / compute-stage uniforms are device pointers; the address is taken by NVN slot
            const p_bind_buf *u = reg < 16 ? (b->kind == NVN_RB_UBOP ? &S.ubo[nvn_stage][reg] : &S.ssbo[nvn_stage][reg]) : NULL;
            void *mb; uint64_t mo, gpu;
            if (u && u->addr && mtl_gpu_addr(u->addr, u->size ? u->size : 1, &mb, &mo, &gpu)) { use_rec("buf", u->addr, u->size, 0, NULL); slots[aid] = gpu; USE((__bridge id<MTLBuffer>)mb, b->rw ? MTLResourceUsageRead | MTLResourceUsageWrite : MTLResourceUsageRead); }
            else { slots[aid] = dummy_buf_gpu(); USE(dummy_buf(), MTLResourceUsageRead); }
            if (bl) PD("B %llu st%d sbufp reg%d %s addr=%#llx size=%llu %s", g_stat_pass, nvn_stage, reg, u && u->addr ? "实" : "哑", u ? (unsigned long long)u->addr : 0ull, u ? (unsigned long long)u->size : 0ull, b->rw ? "rw" : "ro");
        } else if (b->kind == NVN_RB_C1) {      // B: the shader's own constant segment, built when the shader is fetched
            id<MTLBuffer> cb = sh->c1buf ? sh->c1buf : zero_bank();   // c1 has length 0 (600 of them in the whole library) yet the shader still declares it: reads 0
            slots[aid] = (sh->c1buf ? sh->c1_gpu : zero_bank_gpu()) + (sh->c1buf ? sh->c1off : 0);
            USE(cb, MTLResourceUsageRead);
        } else if (b->kind == NVN_RB_SUPPORT) {   // support_buffer: scale 1.0, BGRA swap 0, viewport transform not disabled (viewport fields are not read)
            slots[aid] = support_gpu();
            USE(support_buf(), MTLResourceUsageRead);
        } else if (b->kind == NVN_RB_DRV0 || b->kind == NVN_RB_DRV2) {
            // NVN driver constant segments: those proven unread by static analysis at packaging time are supplied as 0; those read are filled on the spot from the current bindings - c0 = storage buffer table (base + 0x10 r: address 64 bits, size 32 bits),
            // c2 = texture handle table (0x168 + 8 r) and sampler handle table (0x568 + 8 r), the values being the NVN handles the game gave at bind time; positions read outside the tables are registered at packaging time and reported a failure at load time
            int bank = b->kind == NVN_RB_DRV2 ? 2 : 0;
            if (!sh->rf.drv_read[bank]) { slots[aid] = zero_bank_gpu(); USE(zero_bank(), MTLResourceUsageRead); }
            else {
                id<MTLBuffer> db; size_t doff = arena_alloc(0xA00, &db); uint8_t *d = (uint8_t *)db.contents + doff; memset(d, 0, 0xA00);
                if (bank == 0) {
                    int base = DM_GAME_NVN_SSBO_BASE(nvn_stage);   // game profile nvn_ssbo_bases (Mario: vertex 0x110, pixel 0x510, compute 0x310)
                    if (base < 0) RED_ONCE(45, "阶段 %d 的着色器读驱动段 c0，存储缓冲表基址未实测", nvn_stage);
                    else for (int r = 0; r < 16; r++) { uint64_t a = S.ssbo[nvn_stage][r].addr; uint32_t n = (uint32_t)S.ssbo[nvn_stage][r].size; memcpy(d + base + 16 * r, &a, 8); memcpy(d + base + 16 * r + 8, &n, 4); }
                } else {
                    for (int r = 0; r < DM_GAME_NVN_TEX_SLOTS; r++) { uint32_t th = (uint32_t)S.tex[nvn_stage][r]; memcpy(d + DM_GAME_NVN_TEX_BASE + 8 * r, &th, 4); }
                    for (int r = 0; r < DM_GAME_NVN_IMG_SLOTS; r++) { uint32_t ih = (uint32_t)S.img[nvn_stage][r]; memcpy(d + DM_GAME_NVN_IMG_BASE + 8 * r, &ih, 4); }   // This game (NVN 55): texture handle table at 0x20 + 8 r (dm/gfx/tools/dmsh_build.py TEX_BASE)
                    // separate sampler handle table (older driver layout such as Engage's: textures 0x168 + 8 r, samplers 0x568 + 8 r; Mario uses no separate samplers, so this is all zero)
                    for (int r = 0; r < 128; r++) { uint32_t sh_ = (uint32_t)S.samp[nvn_stage][r]; memcpy(d + DM_GAME_NVN_SAMP_BASE + 8 * r, &sh_, 4); }
                }
                slots[aid] = g_cur_arena->gpu + doff;
                USE(db, MTLResourceUsageRead);
            }
        } else RED_ONCE(46, "绑定种类 %d 未实现", b->kind);
    }
    #undef USE
    pthread_mutex_unlock(&g_rmu);
    NSUInteger arg_len = sh->arg_len ? sh->arg_len : sh->ae.encodedLength;
    id<MTLBuffer> ab; size_t off = arena_alloc(arg_len, &ab);
    memcpy((uint8_t *)ab.contents + off, slots, arg_len);
    bind_args(en, nvn_stage, metal_vertex, ab, g_cur_arena->gpu + off, off, sh->rf.abi);
    sh->ab_buf = ab; sh->ab_gpu = g_cur_arena->gpu + off; sh->ab_off = off; sh->ab_pass = g_pass.serial; g_args_dirty[nvn_stage] = 0;
    g_stat_args_enc++;
    return 1;
}
static id<MTLBuffer> g_quad_ib; static uint64_t g_quad_gpu;
static id<MTLBuffer> g_fan_ib; static uint64_t g_fan_gpu;   // non-indexed triangle fans: shared indices (0, i, i+1), base vertex = first
static id<MTL4RenderCommandEncoder> rp_begin(void);
// ---------- Pass lookahead: does this pass use depth / stencil ----------
// When opening a pass, scan forward from the op after the current draw, simulating the changes of depth-stencil state, up to where this pass must end (target reset, copy, compute dispatch -
// matching one-to-one the ops that end a pass in the replay). If no draw in between enables depth test / depth write / stencil test, this pass may leave the depth attachment unattached:
// not one byte of the depth content is touched, and on TBDR the load and store of the whole depth texture are saved (measured on a real device: a single-draw fullscreen pass takes half the GPU, with an extra D32S8 load and store attached).
// The scan follows real replay order across sub-list returns and across the remaining handles of this submission; end of submission = the pass must end. Undecidable cases such as a mid-pass clear or too-deep nesting always count as 'used'.
static const uint8_t *g_exec_next;   // The op following the current draw in the replay
static const uint8_t *g_ret[16]; static int g_ret_n;   // Call return stack of the replay (at OP_CALL the caller's next op is pushed)
static const uint64_t *g_rest; static int g_nrest;      // Handles in this submission still to be replayed after the current handle
void mtl_exec_rest(const uint64_t *h, int n) { g_rest = h; g_nrest = n; }
static uint64_t g_stat_ds_omit;      // Number of passes that attach less depth-stencil because the lookahead proved it unused (including stencil-only / depth-only)
static uint64_t g_stat_s_only, g_stat_d_only;   // Of those, the number of passes that attach only stencil / only depth
// Clear-only pass hoist (hoist_try): ops that were already executed early and must be skipped when replay reaches their original position (each group's SetRenderTargets and the clears in it).
// The exec loop, the store lookahead (sdx index) and the depth-stencil lookahead (ds_scan) all skip by it - otherwise the lookahead would take the clear at the original position as 'cleared again' and wrongly judge the early-cleared content redundant
#define HOIST_MAX 40   // At most 4 groups × (1 set-target op + 8 clears)
static const uint8_t *g_hoist_skip[HOIST_MAX]; static int g_hoist_n; static uint64_t g_hoist_groups;
static int hoist_skipped(const uint8_t *p) { for (int i = 0; i < g_hoist_n; i++) if (g_hoist_skip[i] == p) return 1; return 0; }
typedef struct { const uint8_t *p, *ret[8]; int ret_n, rest_i, bad; } fz_cursor;
static int fz_next(fz_cursor *c, const uint8_t **out, op_hdr *oh, const void **oa) {
    for (int guard = 0; guard < 1000000; guard++) {
        if (!c->p) return 0;
        op_hdr h; memcpy(&h, c->p, sizeof h); const void *a = c->p + 8;
        switch (h.op) {
            case OP_END:
                if (c->ret_n) c->p = c->ret[--c->ret_n];
                else if (c->rest_i < g_nrest) c->p = (const uint8_t *)g_rest[c->rest_i++];
                else { c->p = NULL; return 0; }
                continue;
            case OP_JUMP: memcpy(&c->p, a, 8); continue;
            case OP_CALL: { uint64_t t; memcpy(&t, a, 8); if (c->ret_n >= 8) { c->bad = 1; return 0; }
                c->ret[c->ret_n++] = c->p + h.size; c->p = (const uint8_t *)t; continue; }
            default: break;
        }
        const uint8_t *p = c->p; c->p += h.size;
        if (g_hoist_n && hoist_skipped(p)) continue;
        if (out) *out = p; if (oh) *oh = h; if (oa) *oa = a; return 1;
    }
    c->bad = 1; return 0;
}
// State of the readable-tile pass merge (merge B step 2): the earlier pass (phase 1) and the later pass (phase 2) share one render pass, and the attachments are the union
typedef struct {
    int active, phase, n1, nskip;
    int nreuse;
    const uint8_t *rt2_op, *reuse[8], *skip[12];
    p_set_rt rt1, rt2;                 // Logical targets of each of the two passes (R.rt is the union during the merge)
    int8_t p2map[8];                   // Logical color number of the later pass → physical attachment
    uint8_t lmap1[8], lmap2[8];        // Physical attachment → logical color number of this phase's draws (0xff = not written)
    uint8_t clr2[8], cd2f, cs2f; MTLClearColor cc2[8]; double cd2; uint32_t cs2;   // Clears at the start of the later pass (used as-is when the in-run dual render redraws)
    uint8_t wrote[8];                   // Physical attachments written by the later pass
    uint8_t cov2[8];                     // Physical attachments covered by the first draw of the later pass
} fuse_t;
static fuse_t g_fz; static uint64_t g_fz_n, g_fz_draws, g_fz_fallback;
typedef struct { int idx; p_draw dw; p_draw_idx di; p_scissor sc; int has_sc; void *S_; uint8_t pm[8]; } defer_snap;   // pm: group P target index → logical color number of this draw (identity for its own draws; a composed mapping for absorbed upstream groups)
typedef struct {
    int active, at_q, in_p, n, reason, reason_op, logged, reset_between; uint64_t born;   // born: creation serial (replay is ordered by it, not by slot)
    p_set_rt p_rt, q_rt;
    const uint8_t *p_end_op, *q_op;
    uint8_t clear_c[8], lmap[8];
    MTLClearColor cc[8];
    uint8_t clear_d, clear_s;
    double cd; uint32_t cs;
    uint64_t p_reads[512]; int nreads, p_dw, p_st;
    defer_snap snap[64];
} defer_t;
static defer_t g_defer_slots[4];
static defer_t *g_defer_cur = &g_defer_slots[0];
#define g_defer (*g_defer_cur)
static int sub_eq(const p_set_rt *a, int ka, const p_set_rt *b, int kb);
static uint64_t g_defer_n, g_defer_draws, g_defer_fallbacks;
static const uint8_t *g_defer_pend;   // The P-range end op of the group that is in its P range (NULL if none): the exec loop compares each op against just this one pointer
static int g_defer_nact, g_defer_natq;   // Number of active groups, number of groups that reached Q and await replay (the hot path only looks at the counts, no per-slot scan)
static int g_defer_ab = -1, g_defer_ab_on;
static void draw(const p_draw *dw, const p_draw_idx *di);
// How many of each operation the replay executes (a diagnostic): the per-operation counts say where the replay's work is.
static uint64_t g_op_count[128];
void mtl_op_count_stats(uint64_t *out) { memcpy(out, g_op_count, sizeof g_op_count); }
static int defer_has_in_p(void) { return g_defer_pend != NULL; }
static int defer_has_active(void) { return g_defer_nact > 0; }
static int defer_slot_free(void) { for (int i = 0; i < 4; i++) if (!g_defer_slots[i].active) return i; return -1; }
static int defer_same_q_depth(const defer_t *a, const p_set_rt *q) {
    if (!a->q_rt.depth || !q->depth) return 1;
    return sub_eq(&a->q_rt, 8, q, 8);
}
static int defer_next(int at_q_only, uint64_t after) {   // Slot number of the next active group (smallest creation serial that is > after); returns −1 if none
    int best = -1; for (int i = 0; i < 4; i++) { const defer_t *d = &g_defer_slots[i]; if (!d->active || (at_q_only && !d->at_q) || d->born <= after) continue; if (best < 0 || d->born < g_defer_slots[best].born) best = i; }
    return best;
}
static void defer_release_one(defer_t *d) { if (d->active) { g_defer_nact--; if (d->at_q) g_defer_natq--; if (d->in_p) g_defer_pend = NULL; } for (int i = 0; i < d->n; i++) free(d->snap[i].S_); memset(d, 0, sizeof *d); }
static void defer_clear_all(void) { for (int i = 0; i < 4; i++) defer_release_one(&g_defer_slots[i]); g_defer_nact = g_defer_natq = 0; g_defer_pend = NULL; }
static int hoist_take(const uint8_t *p) { for (int i = 0; i < g_hoist_n; i++) if (g_hoist_skip[i] == p) { g_hoist_skip[i] = g_hoist_skip[--g_hoist_n]; return 1; } return 0; }
enum { DS_FREE = 0, DS_USED = 1, DS_RET = 2 };
static int g_ds_why; static int g_ds_scanned; static int g_ds_need;   // g_ds_need: need accumulated during the scan, 1 depth 2 stencil   // Diagnostics: the opcode that triggered the 'used' verdict (for a draw, records 1000 + dt*100 + dw*10 + st), and the number of ops scanned
static int ds_scan(const uint8_t *p, ds_st *ds, int depth) {
    for (int guard = 0; guard < 1000000; guard++) {
        op_hdr h; memcpy(&h, p, sizeof h);
        const void *a = p + 8; g_ds_scanned++;
        if (g_hoist_n && hoist_skipped(p)) { p += h.size; continue; }   // Clear pass already executed early
        switch (h.op) {
            case OP_END: return DS_RET;
            case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); p = t; continue; }
            case OP_CALL: { uint64_t t; memcpy(&t, a, 8); if (depth >= 8) return DS_USED;
                int r = ds_scan((const uint8_t *)t, ds, depth + 1); if (r != DS_RET) return r; break; }
            case OP_SET_RT: case OP_COPY_B2B: case OP_CLEAR_BUF: case OP_UPDATE_UB: case OP_COPY_B2T: case OP_COPY_T2B: case OP_COPY_T2T: case OP_CLEAR_TEX: case OP_DISPATCH: return DS_FREE;
            case OP_CLEAR_COLOR: case OP_CLEAR_DS: g_ds_why = h.op; return DS_USED;
            case OP_BIND_STATE: { const p_bind_state *b = a; if (b->kind == 5) memcpy(ds, b->bytes, sizeof *ds); break; }
            case OP_DRAW: case OP_DRAW_IDX:   // Depth test / write → depth needed; stencil test → stencil needed; can stop once both are needed
                g_ds_need |= (ds->dt || ds->dw ? 1 : 0) | (ds->st ? 2 : 0);
                if (g_ds_need == 3) { g_ds_why = 1000 + ds->dt * 100 + ds->dw * 10 + ds->st; return DS_USED; } break;
            default: break;
        }
        p += h.size;
    }
    return DS_USED;
}
// Store lookahead decision (command by command, in real replay order):
//   Redundant: after being set as an attachment again and before any draw, it is cleared in full (color: all four channels of that index written; depth: depth clear; stencil: all 8 stencil bits cleared. Depth and stencil are decided separately and their write-back is set separately)
//   Needed: at a draw / dispatch it is bound to a texture or image slot of any stage (including ones already bound when the scan starts); used as copy source / copy destination / texture-clear target; drawn before cleared after becoming an attachment again (load)
// Different levels / layers of the same texture are decided separately (a downsample chain writes level k and reads level k−1); a sampler binding counts as a read of the whole texture (conservative)
static const uint8_t *g_exec_cur;   // The op currently being replayed (inclusive)
static void sd_need(sd_att *t, int why) { if (!t->state) { t->state = 1; t->why = (uint8_t)why; } }   // why: 1 a draw reads it as a texture 2 as an attachment, drawn before cleared 3 copy 4 texture clear 5 scanned to the end undecided 6 no start point / index unavailable
static int sdx_decide(uint8_t st[9]);   // Store lookahead decision (for the index see mtl_sub_index)
static void sd_finish(void) {
    if (!g_nsd) return;
    if (!tbdr_on()) {   // All optimizations off: always write back
        for (int i = 0; i < g_nsd; i++) { sd_att *t = &g_sd[i];
            if (t->depth == 1) [g_enc setDepthStoreAction:MTLStoreActionStore]; else if (t->depth == 2) [g_enc setStencilStoreAction:MTLStoreActionStore]; else [g_enc setColorStoreAction:MTLStoreActionStore atIndex:(NSUInteger)t->idx]; }
        g_nsd = 0; return; }
    uint8_t nst[9];
    if (sdx_decide(nst)) { for (int i = 0; i < g_nsd; i++) if (nst[i] == 2) g_sd[i].state = 2; else sd_need(&g_sd[i], nst[i] >> 2); }
    else for (int i = 0; i < g_nsd; i++) sd_need(&g_sd[i], 6);   // Pass ended outside the list / index unavailable: always write back
    for (int i = 0; i < g_nsd; i++) {
        sd_att *t = &g_sd[i]; int udc_dead = udc_on() && t->load && !t->wrote;
        // A color attachment written in the same pass as the window texture does not enter the ledger: it is a link in the present chain, not a transient attachment that is 'reset every frame before use'.
        // The in-run 'upscale' pass writes two textures at once - scene texture + window texture - and the UI composite was merged into the same pass, so in-run nobody reads back the scene texture; as soon as the game stops drawing 3D (loading screen)
        // a separate composite pass samples it afterwards instead. 'About to enter the loading screen' is only known after the previous frame was submitted, so prediction cannot stop it: on iPad true fullscreen every install hits it once on the first act change
        // (the 2732×2048 attachment of frame N was not written back and is sampled for 4 consecutive frames starting at frame N+1; the key is permanently disabled). At the iPhone aspect ratio the same-size key already sees a reader during the trial period and is silently disabled.
        int xv_win = 0; for (int k = 0; k < R.rt.n && k < 8; k++) if (R.rt.color[k] && xv_present(R.rt.color[k]->serial)) xv_win = 1;
        uint64_t xtag = 0; int xvcand = xv_on() && !xv_win && t->depth == 0 && t->state != 2 && t->why == 5 && xv_first_good(t->ser, t->lv, t->sl) && !xv_present(t->ser)
                                        && !xv_disabled(xtag = xv_tag_pass(R.rt.color[t->idx], t->lv, t->idx, &R.rt));
        int xvdead = xvcand && xv_trial_get(xtag, 0) >= XV_TRIAL;   // Only after the trial period is the write-back really skipped
        if (xvcand && !xvdead) xv_set(t->ser, t->lv, t->sl, XV_PROBE, xtag);   // Trial period: write back as usual, only track the next touch
        int dead = t->state == 2 || udc_dead || xvdead; MTLStoreAction sa = dead ? MTLStoreActionDontCare : MTLStoreActionStore;
        if (xvdead) { xv_set(t->ser, t->lv, t->sl, XV_INVALID, xtag); g_xv_dc++; }
        if (t->depth == 1) [g_enc setDepthStoreAction:sa]; else if (t->depth == 2) [g_enc setStencilStoreAction:sa]; else [g_enc setColorStoreAction:sa atIndex:(NSUInteger)t->idx];
        g_sd_n_all++; if (dead) { g_sd_n_dead++; g_sd_dead_bytes += t->bytes; g_bw_store -= t->bytes; if (udc_dead) g_udc_saved_bytes += t->bytes; }
        if (!t->state) t->why = 5;
        PD("Z %llu %s ser=%llu lv%u sl%u %s why=%d by=%s slot=%d", g_stat_pass, t->depth == 1 ? "d" : t->depth == 2 ? "s" : "c", (unsigned long long)t->ser, t->lv, t->sl, xvdead ? "dontcare-xv" : dead ? "dontcare" : "store", dead ? 0 : t->why, t->why == 1 && t->wkey ? t->wkey : "-", t->wslot);
        if (udc_on()) PD("U %llu %s ser=%llu lv%u sl%u runtime=%s reason=%s state=%d load=%d", g_stat_pass, t->depth == 1 ? "d" : t->depth == 2 ? "s" : "c", (unsigned long long)t->ser, t->lv, t->sl, t->wrote ? "写过" : "没写", t->wwhy == 1 ? "图像写" : t->wwhy == 2 ? "颜色写掩码" : t->wwhy == 3 ? "片元输出" : t->wwhy == 4 ? "深度写" : t->wwhy == 5 ? "模板写" : "无", t->state, t->load);
    }
    g_nsd = 0;
}
// Continue scanning in real replay order: current list -> callers at each level (return stack from the top down) -> remaining handles of this submit; the render pass necessarily ends when the submit ends (rp_flush before commit)
// Coverage proof (holds only if all are satisfied; anything unrecognized is treated as not holding):
//   Shader: the vertex position xyz is taken as-is from one vertex attribute, w is 1 or the same attribute's .w (dmsh_build raster_facts); the fragment does not discard, does not write sample_mask, and writes this color index
//   State: this attachment has no blending, a full write mask, no alpha-to-coverage, fill mode, no depth test (or ALWAYS), no stencil test, single sample, not 3D
//   Vertices: non-indexed or 16 / 32-bit indexed draws, with indices and positions both read from CPU-visible pool memory; the scissor / viewport / geometry / winding / depth clipping decisions are in nvn_cover.h (pure functions, with unit tests)
// Read out attribute a of all vertices of this draw (non-indexed or 16 / 32-bit indexed; vertex number = first + k or index value + base_vertex), with missing components filled with 0 / 1 per the fetch rules.
// Returns 0 on success, a reason code on failure (6 vertex count / index type, 7 attribute index, 8 format / stream, 9 out of bounds, 10 address)
static int attr_read(int a, const p_draw *dw, const p_draw_idx *di, float P[64][4]) {
    int n = dw ? dw->count : di->count;
    if (n < 1 || n > 64 || (dw && dw->first < 0) || (di && (di->itype < 1 || di->itype > 2))) return 6;
    if (a < 0 || a >= S.nattr || S.attr[a].stream >= 16) return 7;
    va_st at = S.attr[a]; vs_st sm = S.stream[at.stream]; NVNbufferRange vb = S.vbuf[at.stream];
    MTLVertexFormat vf = vfmt(at.fmt); int nc = vf == MTLVertexFormatFloat2 ? 2 : vf == MTLVertexFormatFloat3 ? 3 : vf == MTLVertexFormatFloat4 ? 4 : 0;
    if (!nc || !sm.stride || sm.divisor || !vb.address) return 8;
    const uint8_t *ib = NULL; int isz = di ? (di->itype == 1 ? 2 : 4) : 0;
    if (di) { void *m; uint64_t o; if (!mtl_addr(di->ibuf, (uint64_t)n * isz, &m, &o) || !(ib = [(__bridge id<MTLBuffer>)m contents])) return 10; ib += o; }
    for (int k = 0; k < n; k++) {
        int64_t vid = dw ? (int64_t)dw->first + k : (isz == 2 ? (int64_t)((const uint16_t *)ib)[k] : (int64_t)((const uint32_t *)ib)[k]) + di->base_vertex;
        uint64_t off = (uint64_t)vid * sm.stride + at.offset;
        if (vid < 0 || off + 4 * nc > vb.size) return 9;
        void *mb; uint64_t mo; const uint8_t *base;
        if (!mtl_addr(vb.address + off, 4 * nc, &mb, &mo) || !(base = [(__bridge id<MTLBuffer>)mb contents])) return 10;
        float v[4] = {0, 0, 0, 1}; memcpy(v, base + mo, 4 * nc); memcpy(P[k], v, sizeof v);
    }
    return 0;
}
static int first_draw_covers(int i, id<MTLTexture> t, NSUInteger level) {
    const p_draw *dw = g_fd_dw; const p_draw_idx *di = g_fd_di;
    int n = dw ? dw->count : di->count, prim = dw ? dw->prim : di->prim;
    __unsafe_unretained NvProg *pr = S.prog ? prog_get(S.prog) : nil; if (!pr) return g_fd_why = 1, 0;
    NvShader *vs = pr->sh[0], *fs = pr->sh[1];
    if (!vs || !fs || vs->rf.pos_attr < 0 || fs->rf.may_discard || fs->rf.writes_smask || !(fs->rf.color_mask >> i & 1)) return g_fd_why = 2, 0;
    if (!t || t.sampleCount != 1 || t.textureType == MTLTextureType3D) return g_fd_why = 3, 0;
    if ((S.color.blend_en >> i & 1) || ((S.chmask.mask >> (4 * i)) & 0xf) != 0xf || S.ms.a2c || S.poly.mode != NVN_POLYGON_MODE_FILL) return g_fd_why = 4, 0;
    if (R.rt.depth && ((S.ds.dt && S.ds.df != NVN_DEPTH_FUNC_ALWAYS) || S.ds.st)) return g_fd_why = 5, 0;
    if (n < 3 || n > 64) return g_fd_why = 6, 0;
    float P[64][4]; int why = attr_read(vs->rf.pos_attr, dw, di, P);
    if (why) return g_fd_why = why, 0;
    if (vs->rf.pos_w1) for (int k = 0; k < n; k++) P[k][3] = 1.0f;   // Position w is the constant 1 (the attribute's .w is not taken)
    int sc[4] = {R.sc.x, R.sc.y, R.sc.w, R.sc.h}; float vp[4] = {S.vp[0][0], S.vp[0][1], S.vp[0][2], S.vp[0][3]};
    g_fd_why = 11;
    return nvn_cover((const float (*)[4])P, n, prim, (int)S.poly.cull, S.poly.front == NVN_FRONT_FACE_CCW,
                     S.has_vp, vp, R.has_sc, sc, (long)MAX(t.width >> level, 1), (long)MAX(t.height >> level, 1));
}

static int state_op(uint32_t op, const void *a) {
    switch (op) {
        case OP_SCISSOR: g_op_count[12]++; R.sc = *(const p_scissor *)a; R.has_sc = 1; break;
        case OP_BIND_PROGRAM: g_op_count[14]++; { const p_bind_program *bp_ = a; const prog_t *np = bind_stages(bp_->prog, bp_->stages); if (np != S.prog) { S.prog = np; g_pso_dirty = 1; args_dirty_all(); } break; }
        case OP_BIND_STATE: {
            const p_bind_state *b = a; void *dst = NULL; size_t n = 0;
            blend_st bs;
            switch (b->kind) {
                case 0: memcpy(&bs, b->bytes, sizeof bs); dst = &S.blend[bs.target]; n = sizeof bs; break;
                case 1: dst = &S.color; n = sizeof S.color; break;
                case 2: dst = &S.chmask; n = sizeof S.chmask; break;
                case 3: dst = &S.ms; n = sizeof S.ms; break;
                case 4: dst = &S.poly; n = sizeof S.poly; break;
                case 5: dst = &S.ds; n = sizeof S.ds; break;
            }
            if (dst && memcmp(dst, b->bytes, n)) { memcpy(dst, b->bytes, n); if (b->kind == 5) g_dss_dirty = 1; else g_pso_dirty = 1; }
            break;
        }
        case OP_BIND_VATTR: g_op_count[16]++; { const p_bind_vattr *b = a; if (S.nattr != b->n || memcmp(S.attr, b->st, 4 * b->n)) { memcpy(S.attr, b->st, 4 * b->n); S.nattr = b->n; g_pso_dirty = 1; } break; }
        case OP_BIND_VSTREAM: g_op_count[17]++; { const p_bind_vstream *b = a; if (S.nstream != b->n || memcmp(S.stream, b->st, 8 * b->n)) { memcpy(S.stream, b->st, 8 * b->n); S.nstream = b->n; g_pso_dirty = 1; } break; }
        case OP_BIND_VBUF: g_op_count[18]++; { const p_bind_vbuf *b = a; memcpy(&S.vbuf[b->first], b->r, sizeof *b->r * b->n); break; }
        case OP_BIND_UBO: g_op_count[19]++; { const p_bind_buf *b = a; if (b->stage < 6 && b->index < 16 && memcmp(&S.ubo[b->stage][b->index], b, sizeof *b)) { S.ubo[b->stage][b->index] = *b; g_args_dirty[b->stage] = 1; } break; }
        case OP_BIND_SSBO: g_op_count[20]++; { const p_bind_buf *b = a; if (b->stage < 6 && b->index < 16 && memcmp(&S.ssbo[b->stage][b->index], b, sizeof *b)) { S.ssbo[b->stage][b->index] = *b; g_args_dirty[b->stage] = 1; } break; }
        case OP_BIND_TEX: g_op_count[21]++; { const p_bind_handles *b = a; bh_fresh(b); if (b->stage < 6 && b->first + b->n <= 128 && memcmp(&S.tex[b->stage][b->first], b->h, 8 * b->n)) { memcpy(&S.tex[b->stage][b->first], b->h, 8 * b->n); g_args_dirty[b->stage] = 1; } break; }
        case OP_BIND_SAMP: g_op_count[22]++; { const p_bind_handles *b = a; bh_fresh(b); if (b->stage < 6 && b->first + b->n <= 128 && memcmp(&S.samp[b->stage][b->first], b->h, 8 * b->n)) { memcpy(&S.samp[b->stage][b->first], b->h, 8 * b->n); g_args_dirty[b->stage] = 1; } break; }
        case OP_BIND_IMG: g_op_count[34]++; { const p_bind_handles *b = a; bh_fresh(b); if (b->stage < 6 && b->first >= 0 && b->first + b->n <= 32) { if (memcmp(&S.img[b->stage][b->first], b->h, 8 * b->n)) { memcpy(&S.img[b->stage][b->first], b->h, 8 * b->n); g_args_dirty[b->stage] = 1; } } else RED_ONCE(31, "图像槽越界 阶段 %d 起 %d 数 %d", b->stage, b->first, b->n); break; }
        case OP_STENCIL: g_op_count[23]++; {
            const p_stencil *b = a;
            if (b->which != 2) g_dss_dirty = 1;
            for (int f = 0; f < 2; f++) if (b->face & (1 << f)) { if (b->which == 0) S.st_vmask[f] = b->value_mask; else if (b->which == 1) S.st_wmask[f] = b->mask; else S.st_ref[f] = b->ref; }
            break;
        }
        case OP_POLY_OFFSET: g_op_count[24]++; S.po = *(const p_poly_offset *)a; break;
        case OP_VIEWPORTS: g_op_count[11]++; { const p_viewports *b = a; for (int i = 0; i < b->n; i++) memcpy(S.vp[b->first + i], b->v[i], 16); S.has_vp = 1; break; }
        case OP_DEPTH_RANGE: { const p_depth_range *b = a; S.dr[0] = b->n; S.dr[1] = b->f; break; }
        case OP_BLEND_COLOR: memcpy(S.blend_color, ((const p_blend_color *)a)->rgba, 16); break;
        case OP_COPY_STRIDE: { const p_copy_stride *cs = a; if (cs->image) S.copy_img_stride = cs->stride; else S.copy_stride = cs->stride; break; }
        default: return 0;
    }
    return 1;
}

static void rt_sub(const p_set_rt *r, int k, uint64_t *ser, int *lv, int *sl);

// Prediction for parallel precompilation (see the note before pso_get): scan from after the current operation to the end of this submit, replaying only state; at a draw, compute the pipeline key from "the state at that time + that render pass's attachment formats" and dispatch those not yet built to the background.
// Whether a render pass actually attaches depth / stencil is only decided at replay (depth-only / stencil-only / neither attached are omitted), so one key is computed for each possible attach mode -- guessing too many only means wasted background compiles and does not affect the result.
static void pso_predict_rest(void) {
    if (!g_exec_next) return;
    static typeof(S) Ssave; memcpy(&Ssave, &S, sizeof S);
    int pso_dirty = g_pso_dirty, dss_dirty = g_dss_dirty;
    uint32_t cf[8]; memcpy(cf, g_pass.cfmt, sizeof cf); uint32_t df = g_pass.dfmt, sf = g_pass.sfmt, samples = g_pass.samples, dfull = df, sfull = sf; int kicked0 = (int)g_pp_kicked, draws = 0;
    fz_cursor cur = {.p = g_exec_next};
    // Collect them all first, then dispatch: keys for "attachments attached as-is" go first (most likely to be used), the other attach modes after -- background concurrency is limited, and what is dispatched first is compiled first
    enum { QN = 160 }; static struct { pso_key k; void *vs, *fs; } qa[QN], qb[QN]; int na = 0, nb = 0;
    for (int guard = 0; guard < 200000 && na < QN; guard++) {
        const uint8_t *q; op_hdr hq; const void *aq;
        if (!fz_next(&cur, &q, &hq, &aq)) break;
        if (hq.op == OP_SET_RT) {
            const p_set_rt *r = aq; memset(cf, 0, sizeof cf); dfull = sfull = 0; samples = 1;
            for (int i = 0; i < r->n && i < 8; i++) { id<MTLTexture> t = r->color[i] && !rt_dup_slot(r, i) ? tex_mtl(r->color[i]) : nil; if (t) { cf[i] = (uint32_t)rt_view_fmt(t, &r->cview[i], r->has_cview[i]); samples = (uint32_t)t.sampleCount; } }
            id<MTLTexture> dt = r->depth ? tex_mtl(r->depth) : nil;
            if (dt) { MTLPixelFormat f = dt.pixelFormat; samples = (uint32_t)dt.sampleCount;
                dfull = f != MTLPixelFormatStencil8 ? (uint32_t)f : 0; sfull = (f == MTLPixelFormatStencil8 || f == MTLPixelFormatDepth32Float_Stencil8) ? (uint32_t)f : 0; }
            df = dfull; sf = sfull; continue;
        }
        if (hq.op == OP_DRAW || hq.op == OP_DRAW_IDX) {
            draws++;
            __unsafe_unretained NvProg *pr = S.prog ? prog_get(S.prog) : nil; if (!pr || !pr->sh[0]) continue;
            __unsafe_unretained NvShader *vs = pr->sh[0], *fs = pr->sh[1];
            const uint32_t dv[4] = {dfull, 0, 0, dfull}, sv[4] = {sfull, 0, sfull, 0};
            for (int v = 0; v < 4; v++) {
                if (v && dv[v] == dv[0] && sv[v] == sv[0]) continue;
                if (v == 2 && !sfull) continue; if (v == 3 && (!dfull || !sfull)) continue;
                pso_key k; pso_key_fill(&k, vs, fs, cf, dv[v], sv[v], samples, 0);
                if (hm_get_id(&g_pso, &k)) continue;
                int dup = 0; for (int i = 0; i < (v ? nb : na) && !dup; i++) dup = !memcmp(&(v ? qb : qa)[i].k, &k, sizeof k);
                if (dup) continue;
                if (!v && na < QN) { qa[na].k = k; qa[na].vs = (__bridge void *)vs; qa[na].fs = (__bridge void *)fs; na++; }
                else if (v && nb < QN) { qb[nb].k = k; qb[nb].vs = (__bridge void *)vs; qb[nb].fs = (__bridge void *)fs; nb++; }
            }
            continue;
        }
        if (hq.op == OP_CLEAR_COLOR || hq.op == OP_CLEAR_DS) continue;
        state_op(hq.op, aq);
    }
    (void)df; (void)sf;
    for (int i = 0; i < na; i++) pso_pend_kick(&qa[i].k, (__bridge NvShader *)qa[i].vs, (__bridge NvShader *)qa[i].fs);
    for (int i = 0; i < nb; i++) pso_pend_kick(&qb[i].k, (__bridge NvShader *)qb[i].vs, (__bridge NvShader *)qb[i].fs);
    memcpy(&S, &Ssave, sizeof S); args_dirty_all(); g_pso_dirty = pso_dirty; g_dss_dirty = dss_dirty;
    static int logged; if (g_pp_kicked > (uint64_t)kicked0 && logged++ < 20) MLOG("并行预编：本次提交余下 %d 次绘制，丢给后台 %d 条管线", draws, (int)g_pp_kicked - kicked0);
}

static void fz_probe_cov2(const p_set_rt *r2, const uint8_t *first, const int8_t *p2map, const uint8_t *clr, int n1, int n) {
    static typeof(S) Ssave; memcpy(&Ssave, &S, sizeof S); typeof(R) Rsave = R;
    const p_draw *fdw = g_fd_dw; const p_draw_idx *fdi = g_fd_di; int fdwhy = g_fd_why;
    int pso_dirty = g_pso_dirty, dss_dirty = g_dss_dirty;
    op_hdr fh; memcpy(&fh, first, sizeof fh);
    g_fd_dw = fh.op == OP_DRAW ? (const p_draw *)(first + 8) : NULL;
    g_fd_di = fh.op == OP_DRAW_IDX ? (const p_draw_idx *)(first + 8) : NULL;
    R.rt = *r2; R.has_rt = 1;
    fz_cursor cur = {.p = g_exec_next};
    for (;;) {
        const uint8_t *q; op_hdr hq; const void *aq;
        if (!fz_next(&cur, &q, &hq, &aq)) break;
        if (q == first) break;
        if (hq.op == OP_DRAW || hq.op == OP_DRAW_IDX || hq.op == OP_SET_RT || hq.op == OP_CLEAR_COLOR || hq.op == OP_CLEAR_DS) continue;
        state_op(hq.op, aq);
    }
    for (int j = n1; j < n && j < 8; j++) {
        int k = -1;
        for (int q = 0; q < 8; q++) if (p2map[q] == j) { k = q; break; }
        if (k < 0 || clr[k] || !r2->color[k]) continue;
        uint64_t ser; int level, slice;
        rt_sub(r2, k, &ser, &level, &slice);
        (void)ser; (void)slice;
        g_fz.cov2[j] = (uint8_t)first_draw_covers(k, tex_mtl(r2->color[k]), (NSUInteger)level);
    }
    memcpy(&S, &Ssave, sizeof S); args_dirty_all(); R = Rsave;
    g_fd_dw = fdw; g_fd_di = fdi; g_fd_why = fdwhy;
    g_pso_dirty = pso_dirty; g_dss_dirty = dss_dirty;
}

static int fz_load_covers(int i, id<MTLTexture> t, NSUInteger level) {
    if (g_fz.active && g_fz.phase == 1 && i >= g_fz.n1) return g_fz.cov2[i];
    return first_draw_covers(i, t, level);
}

// Probe wrap-up: for every attachment changed to clear to the sentinel color, read back that level / layer, compare pixel by pixel with a same-format 1x1 sentinel pixel, and count what remains (> 0 means the coverage proof is wrong, report a failure)
static void lp_finish(void) {
    id<MTLCommandBuffer> dcb = q_cb();
    NSArray *ps = g_lp; g_lp = nil;
    for (NSArray *e in ps) {
        id<MTLTexture> t = e[0]; NSUInteger lv = [e[1] unsignedIntegerValue], sl = [e[2] unsignedIntegerValue];
        NSUInteger W = MAX(t.width >> lv, 1), H = MAX(t.height >> lv, 1), bpp = (NSUInteger)(att_bytes(t, lv) / (W * H));
        MTLTextureDescriptor *rd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:t.pixelFormat width:1 height:1 mipmapped:NO];
        rd.usage = MTLTextureUsageRenderTarget; rd.storageMode = MTLStorageModePrivate;
        id<MTLTexture> ref = [g_dev newTextureWithDescriptor:rd];
        MTLRenderPassDescriptor *pd = [MTLRenderPassDescriptor renderPassDescriptor];
        pd.colorAttachments[0].texture = ref; pd.colorAttachments[0].loadAction = MTLLoadActionClear; pd.colorAttachments[0].storeAction = MTLStoreActionStore;
        pd.colorAttachments[0].clearColor = [e[6] intValue] == 3 ? MTLClearColorMake(NAN, NAN, NAN, NAN) : [e[6] intValue] == 2 ? MTLClearColorMake(0.777, 0.222, 0.555, 0.888) : MTLClearColorMake(0.123, 0.456, 0.789, 0.321);
        [[dcb renderCommandEncoderWithDescriptor:pd] endEncoding];
        id<MTLBuffer> bt = [g_dev newBufferWithLength:W * H * bpp options:MTLResourceStorageModeShared], br = [g_dev newBufferWithLength:bpp options:MTLResourceStorageModeShared];
        id<MTLBlitCommandEncoder> b = [dcb blitCommandEncoder];
        [b copyFromTexture:t sourceSlice:sl sourceLevel:lv sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(W, H, 1) toBuffer:bt destinationOffset:0 destinationBytesPerRow:W * bpp destinationBytesPerImage:W * H * bpp];
        [b copyFromTexture:ref sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(1, 1, 1) toBuffer:br destinationOffset:0 destinationBytesPerRow:bpp destinationBytesPerImage:bpp];
    enc_b_end(b);
        unsigned long long pass = [e[3] unsignedLongLongValue], ser = [e[5] unsignedLongLongValue]; int idx = [e[4] intValue];
        [dcb addCompletedHandler:^(id<MTLCommandBuffer> c) { (void)c;
            const uint8_t *p = bt.contents, *r = br.contents; uint64_t left = 0; char at[96] = ""; size_t ao = 0;
            for (NSUInteger k = 0; k < W * H; k++) if (!memcmp(p + k * bpp, r, bpp)) { if (left++ < 4) ao += snprintf(at + ao, sizeof at - ao, " (%lu,%lu)", (unsigned long)(k % W), (unsigned long)(k / W)); }
            if (left) MLOG("【红】读入可省探针：通道 %llu c%d ser=%llu lv%lu sl%lu %lux%lu 剩哨兵像素 %llu，前几个位置%s", pass, idx, ser, (unsigned long)lv, (unsigned long)sl, (unsigned long)W, (unsigned long)H, (unsigned long long)left, at);
            else MLOG("读入可省探针：通道 %llu c%d ser=%llu lv%lu sl%lu %lux%lu 全覆盖（哨兵剩 0）", pass, idx, ser, (unsigned long)lv, (unsigned long)sl, (unsigned long)W, (unsigned long)H);
        }];
    }
    [dcb commit];
}
static int rt_same(const p_set_rt *a, const p_set_rt *b) {   // Same set of targets: compare by texture serial and view (not by object pointer: the game may copy texture objects by value)
    if (a->n != b->n || !a->depth != !b->depth || (a->depth && a->depth->serial != b->depth->serial) || a->has_dview != b->has_dview) return 0;
    if (a->has_dview && memcmp(&a->dview, &b->dview, sizeof a->dview)) return 0;
    for (int k = 0; k < a->n && k < 8; k++) {
        if (!a->color[k] != !b->color[k] || (a->color[k] && a->color[k]->serial != b->color[k]->serial) || a->has_cview[k] != b->has_cview[k]) return 0;
        if (a->has_cview[k] && memcmp(&a->cview[k], &b->cview[k], sizeof a->cview[k])) return 0;
    }
    return 1;
}
// Experimental RT range candidate: only ignore positive nlevels differences not used by the actual render attachments; other fields stay word-for-word identical.
#define NVN_RT_RANGE_IMPL 1
static int rt_range_shape(const p_set_rt *a, const p_set_rt *b) {
    if (a->n != 1 || b->n != 1 || !a->color[0] || !b->color[0] || !a->has_cview[0] || !b->has_cview[0] ||
        !a->depth != !b->depth || a->has_dview != b->has_dview ||
        (a->has_dview && memcmp(&a->dview,&b->dview,sizeof a->dview))) return 0;
    const tex_t *x=a->color[0], *y=b->color[0];
    if (!x->serial || x->serial != y->serial || x->d.target != NVN_TEXTURE_TARGET_2D || y->d.target != NVN_TEXTURE_TARGET_2D ||
        x->d.samples > 1 || y->d.samples > 1 || (x->has_layout && x->mem) || (y->has_layout && y->mem) ||
        memcmp(&x->d,&y->d,sizeof x->d) || x->mem != y->mem || x->pool != y->pool || x->off != y->off) return 0;
    const view_t *v=&a->cview[0], *w=&b->cview[0]; int levels=x->d.levels;
    if (!v->set_levels || !w->set_levels || v->base_level < 0 || v->base_level >= levels ||
        v->nlevels <= 0 || v->nlevels > levels-v->base_level || w->nlevels <= 0 || w->nlevels > levels-v->base_level ||
        (v->set_layers && (v->min_layer || v->nlayers != 1)) || (v->set_format && v->format != x->d.format) ||
        (v->set_target && v->target != NVN_TEXTURE_TARGET_2D)) return 0;
    view_t cv=*w; cv.nlevels=v->nlevels;
    if (memcmp(v,&cv,sizeof cv)) return 0;
    if (a->depth) { const tex_t *d=a->depth,*e=b->depth;
        if (!d->serial || d->serial != e->serial || d->d.target != NVN_TEXTURE_TARGET_2D || e->d.target != NVN_TEXTURE_TARGET_2D ||
            d->d.samples > 1 || e->d.samples > 1 || (d->has_layout && d->mem) || (e->has_layout && e->mem) ||
            memcmp(&d->d,&e->d,sizeof d->d) || d->mem != e->mem || d->pool != e->pool || d->off != e->off) return 0;
        int dl=a->has_dview && a->dview.set_levels ? a->dview.base_level : 0;
        if (dl < 0 || dl >= d->d.levels || mip_extent(d->d.width,dl) < mip_extent(x->d.width,v->base_level) ||
            mip_extent(d->d.height,dl) < mip_extent(x->d.height,v->base_level)) return 0; }

    return v->nlevels != w->nlevels;
}
#define NVN_RT_RANGE_DEPTH_IMPL 1
static void rt_range_note_depth(void) { if (rt_range_on() && S.ds.dw) g_pass.range_depth_nonro=1; }
static int rt_range_same(const p_set_rt *a, const p_set_rt *b) {
    if (!rt_range_on() || !g_enc || g_fz.active || g_pass.range_external || g_pass.range_depth_nonro || !S.ds.dt || S.ds.dw || S.ds.st || g_enc_mode != 4 || !g_pass.hd || g_pass.hs ||
        (g_vis_on && !g_enc_has_vis) || g_pass.reg_gen != g_reg_gen || g_pass.ser_gen != g_ser_gen || g_pass.range_dirty != nvn_dirty_now() || !rt_range_shape(a,b) || !b->depth) return 0;
    id<MTLTexture> color=tex_mtl(b->color[0]), depth=tex_mtl(b->depth);
    int lv=b->cview[0].base_level;
    return mip_extent(color.width,lv) == g_pass.tw && mip_extent(color.height,lv) == g_pass.th &&
        color == g_pass.range_color && depth == g_pass.range_depth && color.textureType == MTLTextureType2D &&
        depth.textureType == MTLTextureType2D && color.sampleCount == 1 && depth.sampleCount == 1 &&
        g_pass.reg_gen == g_reg_gen && g_pass.ser_gen == g_ser_gen && g_pass.range_dirty == nvn_dirty_now();
}
static int draw_samples_rt(NvProg *pr) {   // Sampled attachments keep the original conservative ser-level guard, not split by level/layer
    uint64_t rs[9]; int nr = 0;
    for (int k = 0; k < R.rt.n && k < 8; k++) if (R.rt.color[k]) rs[nr++] = R.rt.color[k]->serial;
    if (R.rt.depth) rs[nr++] = R.rt.depth->serial;
    if (!pr) return 1;
    for (int st = 0; st < 5; st++) { __unsafe_unretained NvShader *sh = pr->sh[st]; if (!sh) continue;
        for (int i = 0; i < sh->rf.nb; i++) { const nvn_rbind *b = &sh->rf.b[i]; uint64_t h;
            if (b->kind == NVN_RB_IMG) h = b->reg < 32 && sh->stage < 6 ? S.img[sh->stage][b->reg] : 0; else if (b->kind == NVN_RB_TEX || b->kind == NVN_RB_TBUF) h = b->reg < 128 && sh->stage < 6 ? S.tex[sh->stage][b->reg] : 0; else continue;
            int tid = (int)(h & 0xfffff); if (!tid) continue;
            pthread_mutex_lock(&g_rmu); uint64_t ser = tid < g_ntreg && g_treg[tid].ok ? g_treg[tid].t.serial : 0; pthread_mutex_unlock(&g_rmu);
            for (int j = 0; j < nr; j++) if (ser && ser == rs[j]) return 1; } }
    return 0;
}
// The whole render pass's need for depth / stencil (bits: 1 depth 2 stencil; 3 = both needed or cannot be proven). Starting from the draw that opens the render pass, scan in real replay order to the end of the render pass:
// the attached clears and this draw's state are counted first; later draws accumulate by their depth-stencil state; anything unprovable such as a mid-pass clear or nesting too deep is treated as both needed
static int pass_ds_need(void) {
    int need = (R.clear_d || S.ds.dt || S.ds.dw ? 1 : 0) | (R.clear_s || S.ds.st ? 2 : 0);
    if (!R.rt.depth || need == 3 || !g_exec_next || g_ret_n > 16) {
        PD("S %llu 不前瞻 depth=%d clear_d=%d clear_s=%d dt=%d dw=%d st=%d next=%d ret=%d", g_stat_pass + 1, R.rt.depth != NULL, R.clear_d, R.clear_s, (int)S.ds.dt, (int)S.ds.dw, (int)S.ds.st, g_exec_next != NULL, g_ret_n);
        return 3;
    }
    ds_st ds = S.ds; g_ds_why = -1; g_ds_scanned = 0; g_ds_need = need;
    int r = ds_scan(g_exec_next, &ds, 0);
    for (int k = g_ret_n - 1; r == DS_RET && k >= 0; k--) r = ds_scan(g_ret[k], &ds, 0);
    for (int k = 0; r == DS_RET && k < g_nrest; k++) r = ds_scan((const uint8_t *)g_rest[k], &ds, 0);
    int out = r == DS_USED ? 3 : g_ds_need;
    PD("S %llu 前瞻 结果=%d 需求=%d（1 深度 2 模板）原因=%d 扫过 %d 条 返回栈 %d 余句柄 %d", g_stat_pass + 1, r, out, g_ds_why, g_ds_scanned, g_ret_n, g_nrest);
    return out;
}
// Pure-clear render pass hoisting: when a render pass is opened, scan forward, and if the shape is "several draws of this render pass -> several groups (SetRenderTargets + only clears) -> back to the same targets as this render pass",
// then hoist those groups of clears to execute before this render pass and skip them at their original positions, and on returning to the same targets the same-target continuation picks up -- this render pass's attachments are no longer stored whole and loaded again for the clears sandwiched in between
// (character select page: grunge and particles both draw 28 / 27 level 0 + depth, with only levels 1 and 2 cleared in between).
// Equivalence conditions (hoist only if all hold, otherwise leave as-is): the scan range has only draws, state and these clears, no copies / dispatches / texture clears / sub-lists; the cleared subresources (texture + level + layer)
// are disjoint from this render pass's attachments; the cleared textures are neither on any current texture / image slot nor bound within the scan range (conservative: regardless of whether the program samples them); the scissor at the time of the clear covers the whole block (same requirement as executing at the original position)
static uint64_t h_ser(uint64_t h) { int tid = (int)(h & 0xfffff); return tid ? atomic_load_explicit(&g_tser[tid], memory_order_relaxed) : 0; }
// ---------- Store lookahead index: at the start of a submit do one linear scan in replay order and record events per texture; when a render pass ends, binary-search each attachment for "the first valid event afterwards" to decide ----------
// Replaces the original "scan the whole remainder forward at every render pass end" (pass count x command count, 49 render passes per frame in camp). Cross-checked per attachment against the original scan: after 0 mismatches over the 17 pages of the screenshot gate + a camp run with about 340,000 attachment decisions (about 46,000 redundant), the original scan was removed.
// Events (seq = replay sequence number): READ a draw / dispatch reads this texture per program reflection; ATT it is a current attachment at a draw; CLEAR cleared as a whole block while an attachment; COPY / CLRTEX either end of a copy / the cleared texture.
// Decision (the first valid event decides): READ / COPY / CLRTEX -> needed; ATT whose SetRenderTargets is after the render pass end -> needed (load); CLEAR with the same -> redundant; no event -> needed.
// On any uncertainty (replay sequence number does not match the index, nesting too deep) the index is invalidated and all remaining render passes of this submit store (storing is always correct, it just saves less bandwidth).
typedef struct { uint32_t seq, rtseq; uint64_t ser; const uint8_t *op; uint16_t lv, sl; uint8_t kind, plane, anyls, why; } sdx_ev;   // plane 0 color 8 depth-stencil (ATT) 1 depth 2 stencil (CLEAR)
enum { SDX_READ = 1, SDX_NEED, SDX_ATT, SDX_CLEAR };
#define SDX_NONE 0xffffffffu
static sdx_ev *g_sdx, *g_sdx_tmp; static uint32_t g_sdx_n, g_sdx_cap; static int g_sdx_ok;   // Index valid
// Candidate textures (those that have been attachments) -> buckets: open addressing; a texture that has not yet been an attachment when an event occurs is irrelevant to any render pass that ended before then, so it is not recorded
#define SDX_HB 2048
static uint64_t g_sdx_hk[SDX_HB]; static uint16_t g_sdx_hb[SDX_HB]; static uint32_t g_sdx_nb, g_sdx_bcnt[SDX_HB + 1], g_sdx_bst[SDX_HB + 1];
static int sdx_bucket(uint64_t ser, int add) {   // Returns the bucket number; returns -1 if not in the set (and not adding); invalidates the index when full
    if (!ser) return -1;
    for (uint32_t i = (uint32_t)(ser * 0x9E3779B97F4A7C15ull >> 53), n = 0; n < SDX_HB; i = (i + 1) & (SDX_HB - 1), n++) {
        if (g_sdx_hk[i] == ser) return g_sdx_hb[i];
        if (!g_sdx_hk[i]) { if (!add) return -1; if (g_sdx_nb >= SDX_HB / 2) { g_sdx_ok = 0; return -1; } g_sdx_hk[i] = ser; g_sdx_hb[i] = (uint16_t)g_sdx_nb; return (int)g_sdx_nb++; }
    }
    return -1;
}
static const uint8_t **g_sdx_op; static uint32_t g_sdx_nop, g_sdx_opcap;
static uint32_t g_exec_seq;   // Sequence number of the current operation during replay
static uint64_t g_pp_slot[6][160]; static const prog_t *g_pp_prog; static const p_set_rt *g_pp_rt; static uint32_t g_pp_rtseq;
static uint32_t *g_rtx_seq, *g_drx_seq; static uint32_t g_rtx_n, g_rtx_cap, g_drx_n, g_drx_cap;   // This submit's SET_RT / draw sequence number tables (used by the pre-screen of the readable tile merge)
static void seq_push(uint32_t **v, uint32_t *n, uint32_t *cap, uint32_t x) { if (*n == *cap) { *cap = *cap ? *cap * 2 : 1024; *v = realloc(*v, *cap * sizeof **v); } (*v)[(*n)++] = x; }
static void sdx_ev_add(sdx_ev e) { if (sdx_bucket(e.ser, 0) < 0) return; if (g_sdx_n == g_sdx_cap) { g_sdx_cap = g_sdx_cap ? g_sdx_cap * 2 : 4096; g_sdx = realloc(g_sdx, g_sdx_cap * sizeof *g_sdx); g_sdx_tmp = realloc(g_sdx_tmp, g_sdx_cap * sizeof *g_sdx); } g_sdx[g_sdx_n++] = e; }
static void sdx_lvsl(const view_t *v, int hv, uint16_t *lv, uint16_t *sl) { *lv = (uint16_t)(hv && v->set_levels ? v->base_level : 0); *sl = (uint16_t)(hv && v->set_layers ? v->min_layer : 0); }
static void sdx_walk(const uint8_t *p, int depth) {
    if (depth > 64) { g_sdx_ok = 0; return; }
    for (uint32_t guard = 0; guard < 4000000 && p && g_sdx_ok; guard++) {
        op_hdr h; memcpy(&h, p, sizeof h); const void *a = p + 8;
        if (g_sdx_nop == g_sdx_opcap) { g_sdx_opcap = g_sdx_opcap ? g_sdx_opcap * 2 : 16384; g_sdx_op = realloc(g_sdx_op, g_sdx_opcap * sizeof *g_sdx_op); }
        uint32_t seq = g_sdx_nop; g_sdx_op[g_sdx_nop++] = p;
        switch (h.op) {
            case OP_END: return;
            case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); p = t; continue; }
            case OP_CALL: { uint64_t t; memcpy(&t, a, 8); sdx_walk((const uint8_t *)t, depth + 1); break; }
            case OP_SET_RT: { const p_set_rt *r = a; g_pp_rt = r; g_pp_rtseq = seq; seq_push(&g_rtx_seq, &g_rtx_n, &g_rtx_cap, seq);
                for (int k = 0; k < r->n && k < 8; k++) if (r->color[k]) sdx_bucket(r->color[k]->serial, 1);
                if (r->depth) sdx_bucket(r->depth->serial, 1);
                break; }
            case OP_CLEAR_COLOR: { const p_clear_color *c = a; const p_set_rt *r = g_pp_rt;
                if (!r || (c->mask & 0xf) != 0xf || c->index < 0 || c->index >= r->n || c->index >= 8 || !r->color[c->index]) break;
                uint16_t lv, sl; sdx_lvsl(&r->cview[c->index], r->has_cview[c->index], &lv, &sl); int last = -1;   // When the same subresource is attached at several indices, the last one counts
                for (int k = 0; k < r->n && k < 8; k++) { uint16_t l2, s2; if (!r->color[k] || r->color[k]->serial != r->color[c->index]->serial) continue; sdx_lvsl(&r->cview[k], r->has_cview[k], &l2, &s2); if (l2 == lv && s2 == sl) last = k; }
                if (last == c->index) sdx_ev_add((sdx_ev){seq, g_pp_rtseq, r->color[c->index]->serial, p, lv, sl, SDX_CLEAR, 0, 0, 0});
                break; }
            case OP_CLEAR_DS: { const p_clear_ds *c = a; const p_set_rt *r = g_pp_rt; if (!r || !r->depth) break;
                uint16_t lv, sl; sdx_lvsl(&r->dview, r->has_dview, &lv, &sl);
                if (c->depth_mask) sdx_ev_add((sdx_ev){seq, g_pp_rtseq, r->depth->serial, p, lv, sl, SDX_CLEAR, 1, !r->has_dview, 0});
                if ((c->stencil_mask & 0xff) == 0xff) sdx_ev_add((sdx_ev){seq, g_pp_rtseq, r->depth->serial, p, lv, sl, SDX_CLEAR, 2, !r->has_dview, 0});
                break; }
            case OP_BIND_TEX: case OP_BIND_IMG: { const p_bind_handles *b = a; bh_fresh(b); int base = h.op == OP_BIND_IMG ? 128 : 0, lim = h.op == OP_BIND_IMG ? 32 : 128;
                if (b->stage < 6 && b->first >= 0 && b->first + b->n <= lim) for (int k = 0; k < b->n; k++) g_pp_slot[b->stage][base + b->first + k] = h_ser(b->h[k]);
                break; }
            case OP_BIND_PROGRAM: g_pp_prog = ((const p_bind_program *)a)->prog; break;
            case OP_DRAW: case OP_DRAW_IDX: case OP_DISPATCH: {
                if (h.op != OP_DISPATCH) seq_push(&g_drx_seq, &g_drx_n, &g_drx_cap, seq);
                __unsafe_unretained NvProg *pr = g_pp_prog ? prog_get(g_pp_prog) : nil;
                if (!pr) { for (int st = 0; st < 6; st++) for (int k = 0; k < 160; k++) if (g_pp_slot[st][k]) sdx_ev_add((sdx_ev){seq, 0, g_pp_slot[st][k], p, 0, 0, SDX_READ, 0, 0, 1}); }
                else for (int st = 0; st < 6; st++) { __unsafe_unretained NvShader *sh = pr->sh[st]; if (!sh || (h.op == OP_DISPATCH) != (st == 5)) continue;
                    for (int i = 0; i < sh->rf.nb; i++) { const nvn_rbind *b = &sh->rf.b[i]; int k = b->kind == NVN_RB_IMG ? 128 + b->reg : (b->kind == NVN_RB_TEX || b->kind == NVN_RB_TBUF) ? b->reg : -1;
                        if (k < 0 || k >= 160 || sh->stage < 0 || sh->stage >= 6) continue; uint64_t ser = g_pp_slot[sh->stage][k];
                        if (ser) sdx_ev_add((sdx_ev){seq, 0, ser, p, 0, 0, SDX_READ, 0, 0, 1}); } }
                const p_set_rt *r = g_pp_rt;
                if (h.op != OP_DISPATCH && r) {
                    for (int k = 0; k < r->n && k < 8; k++) if (r->color[k]) { uint16_t lv, sl; sdx_lvsl(&r->cview[k], r->has_cview[k], &lv, &sl); sdx_ev_add((sdx_ev){seq, g_pp_rtseq, r->color[k]->serial, p, lv, sl, SDX_ATT, 0, 0, 2}); }
                    if (r->depth) { uint16_t lv, sl; sdx_lvsl(&r->dview, r->has_dview, &lv, &sl); sdx_ev_add((sdx_ev){seq, g_pp_rtseq, r->depth->serial, p, lv, sl, SDX_ATT, 8, !r->has_dview, 2}); }
                }
                break; }
            case OP_COPY_T2T: { const p_copy_t2t *c = a; if (c->src) sdx_ev_add((sdx_ev){seq, 0, c->src->serial, p, 0, 0, SDX_NEED, 0, 0, 3}); if (c->dst) sdx_ev_add((sdx_ev){seq, 0, c->dst->serial, p, 0, 0, SDX_NEED, 0, 0, 3}); break; }
            case OP_COPY_T2B: case OP_COPY_B2T: { const p_copy_b2t *c = a; if (c->tex) sdx_ev_add((sdx_ev){seq, 0, c->tex->serial, p, 0, 0, SDX_NEED, 0, 0, 3}); break; }
            case OP_CLEAR_TEX: { const p_clear_tex *c = a; if (c->tex) sdx_ev_add((sdx_ev){seq, 0, c->tex->serial, p, 0, 0, SDX_NEED, 0, 0, 4}); break; }
            default: break;
        }
        p += h.size;
    }
}
void mtl_sub_index(const uint64_t *h, int n) {   // Start of submit (hle_nvn.c QueueSubmitCommands): build this submit's store lookahead index
    g_xv_ntouch = 0; g_sdx_n = g_sdx_nop = 0; g_rtx_n = g_drx_n = 0; g_exec_seq = 0; g_sdx_ok = tbdr_on();
    if (!g_sdx_ok) { g_xv_n = 0; g_xv_block = 1; return; }   // Index unavailable: clear the ledger and add nothing in this submit (the next touch cannot be verified)
    g_xv_block = 0;
    memset(g_sdx_hk, 0, sizeof g_sdx_hk); g_sdx_nb = 0;
    for (int i = 0; i < g_xv_n; i++) sdx_bucket(g_xv[i].ser, 1);   // Textures in the ledger enter the candidate set: otherwise, when one is only sampled and not used as an attachment in this submit, its read events are not indexed and the fallback cannot see them
    if (R.has_rt) { for (int k = 0; k < R.rt.n && k < 8; k++) if (R.rt.color[k]) sdx_bucket(R.rt.color[k]->serial, 1); if (R.rt.depth) sdx_bucket(R.rt.depth->serial, 1); }   // The targets at the start of the submit (the first draw may reopen the render pass without going through SetRenderTargets)
    g_pp_prog = S.prog; g_pp_rt = NULL; g_pp_rtseq = SDX_NONE;
    for (int st = 0; st < 6; st++) { for (int k = 0; k < 128; k++) g_pp_slot[st][k] = S.tex[st][k] ? h_ser(S.tex[st][k]) : 0;
                                     for (int k = 0; k < 32; k++) g_pp_slot[st][128 + k] = S.img[st][k] ? h_ser(S.img[st][k]) : 0; }
    for (int i = 0; i < n && g_sdx_ok; i++) sdx_walk((const uint8_t *)h[i], 0);
    if (!g_sdx_ok) return;
    // Counting sort into buckets: events are appended in sequence-number order, and within the same sequence number READ precedes ATT (the same draw is judged for reads first, then attachments), so after a stable scatter each bucket is ordered
    memset(g_sdx_bcnt, 0, (g_sdx_nb + 1) * sizeof *g_sdx_bcnt);
    for (uint32_t i = 0; i < g_sdx_n; i++) g_sdx_bcnt[sdx_bucket(g_sdx[i].ser, 0)]++;
    for (uint32_t b = 0, acc = 0; b <= g_sdx_nb; b++) { g_sdx_bst[b] = acc; acc += b < g_sdx_nb ? g_sdx_bcnt[b] : 0; }
    for (uint32_t b = 0; b < g_sdx_nb; b++) g_sdx_bcnt[b] = g_sdx_bst[b];
    for (uint32_t i = 0; i < g_sdx_n; i++) g_sdx_tmp[g_sdx_bcnt[sdx_bucket(g_sdx[i].ser, 0)]++] = g_sdx[i];
    sdx_ev *t = g_sdx; g_sdx = g_sdx_tmp; g_sdx_tmp = t;
    xv_forward();
}
static void sdx_seq(const uint8_t *p) {   // Every operation in mtl_exec: check the sequence number against the index (on mismatch invalidate, and store everything for the rest)
    if (!g_sdx_ok) return;
    if (g_exec_seq >= g_sdx_nop || g_sdx_op[g_exec_seq] != p) { RED_ONCE(40, "写回前瞻索引与回放顺序不符（序号 %u / %u），本次提交余下通道一律写回", g_exec_seq, g_sdx_nop); g_sdx_ok = 0; return; }
    g_exec_seq++;
}
static int sdx_decide(uint8_t st[9]) {   // st[i]: 2 redundant; otherwise why << 2 (the reason it is needed, same as sd_need). Returns 0 = index unavailable
    if (!g_sdx_ok || !g_exec_cur || !g_exec_seq || g_sdx_op[g_exec_seq - 1] != g_exec_cur) return 0;
    uint32_t P = g_exec_seq - 1;
    for (int i = 0; i < g_nsd; i++) { const sd_att *t = &g_sd[i]; st[i] = 5 << 2;
        int b = sdx_bucket(t->ser, 0); if (b < 0) continue;   // Not in the candidate set: no related event at all in this submit -> undecided after scanning to the end, needed
        uint32_t lo = g_sdx_bst[b], end = g_sdx_bst[b + 1], hi = end;   // First seq >= P in the bucket
        while (lo < hi) { uint32_t m = (lo + hi) / 2; if (g_sdx[m].seq < P) lo = m + 1; else hi = m; }
        for (uint32_t j = lo; j < end; j++) { const sdx_ev *e = &g_sdx[j];
            if (g_hoist_n && hoist_skipped(e->op)) continue;
            if (e->kind == SDX_READ || e->kind == SDX_NEED) { st[i] = (uint8_t)(e->why << 2); break; }
            if (e->rtseq == SDX_NONE || e->rtseq < P || (g_hoist_n && hoist_skipped(g_sdx_op[e->rtseq]))) continue;   // This target was set before the render pass ended: not re-set as an attachment after the render pass ended, so it does not count
            int hit = t->depth ? (e->kind == SDX_ATT ? e->plane == 8 : e->plane == t->depth) && (e->anyls || (e->lv == t->lv && e->sl == t->sl))
                               : e->plane == 0 && e->lv == t->lv && e->sl == t->sl;
            if (!hit) continue;
            st[i] = e->kind == SDX_CLEAR ? 2 : (uint8_t)(2 << 2); break; }
    }
    return 1;
}
static void xv_forward(void) {
    if (!g_xv_n) return;
    if (!g_sdx_ok) { g_xv_n = 0; g_xv_block = 1; return; }
    for (int i = 0; i < g_xv_n;) {
        xv_ent *e = &g_xv[i]; int b = sdx_bucket(e->ser, 0), action = 0;
        if (b >= 0) for (uint32_t j = g_sdx_bst[b]; j < g_sdx_bst[b + 1]; j++) { sdx_ev *v = &g_sdx[j];
            if (v->kind == SDX_READ || v->kind == SDX_NEED) { xv_ent copy = *e; xv_violation(&copy, "提交前瞻读/需求"); action = 2; break; }
            if (v->lv != e->lv || v->sl != e->sl) continue;
            if (v->kind == SDX_CLEAR) { xv_good(e); action = 1; break; }
            if (v->kind == SDX_ATT) break;
        }
        if (!action) i++;
    }
}
static void rt_sub(const p_set_rt *r, int k, uint64_t *ser, int *lv, int *sl) {   // Subresource of attachment k (0–7 color, 8 depth)
    const tex_t *t = k == 8 ? r->depth : r->color[k]; const view_t *v = k == 8 ? &r->dview : &r->cview[k]; int hv = k == 8 ? r->has_dview : r->has_cview[k];
    *ser = t ? t->serial : 0; *lv = hv && v->set_levels ? v->base_level : 0; *sl = hv && v->set_layers ? v->min_layer : 0;
}
static void hoist_try(void) {
    if (g_hoist_n || !g_exec_next) return;
    typedef struct { const uint8_t *rt_op; const p_set_rt *rt; uint8_t clear_c[8]; MTLClearColor cc[8]; uint8_t clear_d, clear_s; double cd; uint32_t cs;
                     p_scissor sc; int has_sc; const uint8_t *ops[8]; int nops; } grp;
    grp G[4]; int ng = 0, in_grp = 0; p_scissor sc = R.sc; int has_sc = R.has_sc;
    // The binding set needed for "the cleared texture is not on any slot": the table is looked up only when a hoistable clear group is really found and we are back to this render pass's targets (the vast majority of render passes never get that far).
    // Handles bound within the scan range are noted first and converted to texture serials together with the current slots at lookup time; the scan only reads commands and the current slots S do not change meanwhile, so the result is the same as checking at the start
    uint64_t bh[512]; int nbh = 0;
    const uint8_t *p = g_exec_next;
    for (int guard = 0; guard < 200000; guard++) {
        op_hdr h; memcpy(&h, p, sizeof h); const void *a = p + 8;
        switch (h.op) {
            case OP_END: case OP_CALL: case OP_COPY_T2T: case OP_COPY_B2T: case OP_COPY_T2B: case OP_CLEAR_TEX: case OP_DISPATCH: return;
            case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); p = t; continue; }
            case OP_DRAW: case OP_DRAW_IDX: if (in_grp) return; break;
            case OP_SCISSOR: sc = *(const p_scissor *)a; has_sc = 1; break;
            case OP_BIND_TEX: case OP_BIND_IMG: { const p_bind_handles *b = a; bh_fresh(b); for (int k = 0; k < b->n && k < 32 && nbh < 512; k++) if (b->h[k]) bh[nbh++] = b->h[k]; break; }
            case OP_CLEAR_COLOR: case OP_CLEAR_DS: {
                if (!in_grp || G[ng - 1].nops >= 8) return;
                grp *g = &G[ng - 1];
                if (h.op == OP_CLEAR_COLOR) { const p_clear_color *c = a;
                    if (c->index < 0 || c->index >= g->rt->n || !g->rt->color[c->index] || c->mask != 0xf) return;
                    const tex_t *t = g->rt->color[c->index];
                    if (has_sc && !(sc.x <= 0 && sc.y <= 0 && sc.x + sc.w >= (int)t->d.width && sc.y + sc.h >= (int)t->d.height)) return;
                    g->clear_c[c->index] = 1; g->cc[c->index] = MTLClearColorMake(c->rgba[0], c->rgba[1], c->rgba[2], c->rgba[3]); }
                else { const p_clear_ds *c = a; if (!g->rt->depth) return;
                    if (c->depth_mask) { g->clear_d = 1; g->cd = c->depth; } if (c->stencil_mask) { g->clear_s = 1; g->cs = (uint32_t)c->stencil; } }
                g->sc = sc; g->has_sc = has_sc; g->ops[g->nops++] = p; break; }
            case OP_SET_RT: { const p_set_rt *r = a;
                if (rt_same(&R.rt, r)) {   // Back to this render pass's targets: execute if hoisting holds
                    if (!ng) return;
                    uint64_t bound[512]; int nb = 0;   // The order of the original "check at the start": the current slots (stage x textures 128 + images 32), followed by those bound within the scan range
                    for (int st = 0; st < 6; st++) { for (int k = 0; k < 128 && nb < 512; k++) if (S.tex[st][k]) bound[nb++] = h_ser(S.tex[st][k]);
                                                     for (int k = 0; k < 32 && nb < 512; k++) if (S.img[st][k]) bound[nb++] = h_ser(S.img[st][k]); }
                    for (int k = 0; k < nbh && nb < 512; k++) bound[nb++] = h_ser(bh[k]);
                    for (int gi = 0; gi < ng; gi++) {   // Disjoint and not bound
                        grp *g = &G[gi]; if (!g->nops) return;
                        for (int k = 0; k < 9; k++) { if (k < 8 ? !g->clear_c[k] : !(g->clear_d || g->clear_s)) continue;
                            uint64_t ser; int lv, sl; rt_sub(g->rt, k, &ser, &lv, &sl); if (!ser) return;
                            for (int j = 0; j < nb; j++) if (bound[j] == ser) return;
                            for (int m = 0; m < 9; m++) { uint64_t s2; int l2, sl2; if (m < 8 && m >= R.rt.n) continue; rt_sub(&R.rt, m, &s2, &l2, &sl2);
                                if (s2 == ser && l2 == lv && sl2 == sl) return; } } }
                    // Register all skip entries first, then execute: when a hoisted render pass ends the store lookahead scans forward, and the clears at their original positions must already be in the skip set
                    for (int gi = 0; gi < ng; gi++) { g_hoist_skip[g_hoist_n++] = G[gi].rt_op; for (int k = 0; k < G[gi].nops; k++) g_hoist_skip[g_hoist_n++] = G[gi].ops[k]; }
                    p_set_rt rt0 = R.rt; p_scissor sc0 = R.sc; int hs0 = R.has_sc;   // This render pass's targets, scissor and pending clears: restored as they were after execution
                    uint8_t cc0f[8]; MTLClearColor cc0[8]; uint8_t cd0f = R.clear_d, cs0f = R.clear_s; double cd0 = R.cd; uint32_t cs0 = R.cs;
                    memcpy(cc0f, R.clear_c, 8); memcpy(cc0, R.cc, sizeof cc0);
                    for (int gi = 0; gi < ng; gi++) {   // Execute: temporarily swap in that group's targets and clears and open a clear-only render pass
                        grp *g = &G[gi];
                        R.rt = *g->rt; memcpy(R.clear_c, g->clear_c, 8); memcpy(R.cc, g->cc, sizeof R.cc); R.clear_d = g->clear_d; R.clear_s = g->clear_s; R.cd = g->cd; R.cs = g->cs;
                        R.sc = g->sc; R.has_sc = g->has_sc;
                        PD("H %llu 前提纯清除通道（原位置在本通道之后）", g_stat_pass + 1);
                        rp_begin_ex(1); rp_end();
                    }
                    R.rt = rt0; R.sc = sc0; R.has_sc = hs0; memcpy(R.clear_c, cc0f, 8); memcpy(R.cc, cc0, sizeof cc0); R.clear_d = cd0f; R.clear_s = cs0f; R.cd = cd0; R.cs = cs0;
                    if (!g_hoist_groups) MLOG("纯清除通道前提：首例 %d 组（通道 %llu 前）", ng, g_stat_pass + 1);
                    g_hoist_groups += (uint64_t)ng;
                    return;
                }
                if (in_grp && !G[ng - 1].nops) return;   // The previous group has no clear (only set targets): not recognized
                if (ng >= 4) return;
                memset(&G[ng], 0, sizeof G[ng]); G[ng].rt_op = p; G[ng].rt = r; ng++; in_grp = 1; break; }
            default: break;
        }
        p += h.size;
    }
}
// ---------- Readable tile render pass merge (merge B step 2) ----------
// When a draw of the later pass samples, at the same pixel, a color attachment written by the earlier pass, the two passes are merged into one render pass: what the earlier pass wrote stays in the tile,
// and the later pass reads it directly with the readable tile version (<key>_bf, function-constant specialization), instead of first landing in main memory and being read back just to be sampled.
// Planning (fuse_try, at the start of the earlier pass) does only structural checks; the closed-form proof of same-pixel addressing is done at every draw of the later pass (fz_p2_draw), and if it cannot be proven the merged render pass is ended and
// a normal render pass is reopened with the later pass's original targets (the earlier pass's result is stored as usual, equivalent to not merging).
static void fz_unskip(void) { for (int i = 0; i < g_fz.nskip; i++) for (int j = 0; j < g_hoist_n; j++) if (g_hoist_skip[j] == g_fz.skip[i]) { g_hoist_skip[j] = g_hoist_skip[--g_hoist_n]; break; } }
static int fz_reuse_op(const uint8_t *p) { for (int i = 0; i < g_fz.nreuse; i++) if (g_fz.reuse[i] == p) return 1; return 0; }
// Diagnostic DM_NVN_FUSE_CHECK=1: in-run double rendering. For the later pass's draws that can be proven, keep a state snapshot per draw; after the merged render pass ends, make a copy of the attachments the later pass wrote (the merge path's result),
// then turn off all lookahead and redraw one by one through the normal path with the later pass's original targets and original clears, and compare the two byte by byte (same frame, same input, no dependence on cross-run determinism). The redraw result is left on the attachment (no effect if identical)
static int fz_check_on(void) { static int v = -1; if (v < 0) v = getenv("DM_NVN_FUSE_CHECK") != NULL; return v; }
typedef struct { int idx; p_draw dw; p_draw_idx di; p_scissor sc; int has_sc; void *S_; } fz_snap;
static fz_snap g_fzs[8]; static int g_nfzs, g_fzs_bad, g_fz_checking; static uint64_t g_fzc_same, g_fzc_diff;
static void draw(const p_draw *dw, const p_draw_idx *di);
static void fz_snap_take(const p_draw *dw, const p_draw_idx *di) {
    if (g_nfzs >= 8) { g_fzs_bad = 1; return; }
    fz_snap *z = &g_fzs[g_nfzs++]; z->idx = di != NULL; if (dw) z->dw = *dw; if (di) z->di = *di; z->sc = R.sc; z->has_sc = R.has_sc;
    z->S_ = malloc(sizeof S); memcpy(z->S_, &S, sizeof S);
}
static void fz_check_run(const fuse_t *fz) {
    if (!g_nfzs || g_fzs_bad) { if (g_fzs_bad) MLOG("【红】局内双渲染：后一个通道绘制超过 8 次，未比"); goto out; }
    {
    id<MTLTexture> att[8], scr[8]; uint64_t sers[8]; int na = 0;
    // fz_end has already restored R.rt from the physical union to the later pass's logical targets; wrote is still registered by physical slot and must be aligned through p2map.
    for (int j = 0; j < fz->rt2.n && j < 8; j++) if (fz->p2map[j] >= 0 && fz->wrote[fz->p2map[j]] && fz->rt2.color[j]) {
        id<MTLTexture> t = tex_mtl(fz->rt2.color[j]); if (!t) continue;
        MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:t.pixelFormat width:t.width height:t.height mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead; d.storageMode = MTLStorageModePrivate;
        id<MTLTexture> c = [g_dev newTextureWithDescriptor:d]; if (!c) continue;
        enc_b_copy_texture(blit(), t, 0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(t.width, t.height, 1), c, 0, 0, MTLOriginMake(0, 0, 0));
        att[na] = t; scr[na] = c; sers[na] = fz->rt2.color[j]->serial; na++; }
    if (!na) { MLOG("【红】局内双渲染：已留 %d 次绘制快照，但没有可比附件", g_nfzs); goto out; }
    static typeof(S) Ssave; memcpy(&Ssave, &S, sizeof S); typeof(R) Rsave = R; const uint8_t *en0 = g_exec_next;
    g_fz_checking = 1; g_exec_next = NULL;   // Turn off all lookahead during the redraw (continuation / hoisting / merging / depth-stencil on demand): normal path
    R.rt = fz->rt2; memcpy(R.clear_c, fz->clr2, 8); memcpy(R.cc, fz->cc2, sizeof R.cc); R.clear_d = fz->cd2f; R.clear_s = fz->cs2f; R.cd = fz->cd2; R.cs = fz->cs2;
    for (int i = 0; i < g_nfzs; i++) { fz_snap *z = &g_fzs[i]; memcpy(&S, z->S_, sizeof S); args_dirty_all(); R.sc = z->sc; R.has_sc = z->has_sc; g_pso_dirty = g_dss_dirty = 1;
        draw(z->idx ? NULL : &z->dw, z->idx ? &z->di : NULL); }
    rp_end();
    memcpy(&S, &Ssave, sizeof S); args_dirty_all(); R = Rsave; g_exec_next = en0; g_fz_checking = 0; g_pso_dirty = g_dss_dirty = 1;
    for (int a = 0; a < na; a++) {
        id<MTLTexture> t = att[a], c = scr[a]; NSUInteger W = t.width, H = t.height, bpp = (NSUInteger)(att_bytes(t, 0) / (W * H));
        id<MTLBuffer> b1 = [g_dev newBufferWithLength:W * H * bpp options:MTLResourceStorageModeShared], b2 = [g_dev newBufferWithLength:W * H * bpp options:MTLResourceStorageModeShared];
        id<MTL4ComputeCommandEncoder> b = blit();
        enc_b_copy_texture_buffer(b, c, 0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(W, H, 1), b1, 0, W * bpp, W * H * bpp);
        enc_b_copy_texture_buffer(b, t, 0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(W, H, 1), b2, 0, W * bpp, W * H * bpp);
        unsigned long long ser = sers[a], pass = g_stat_pass, frame = atomic_load(&g_cur_frame); NSUInteger nb = W * H * bpp;
        [g_commit_blocks addObject:^{
            const uint8_t *x = b1.contents, *y = b2.contents; uint64_t dpx = 0, maxd = 0;
            for (NSUInteger k = 0; k < nb; k += bpp) if (memcmp(x + k, y + k, bpp)) { dpx++; for (NSUInteger q = 0; q < bpp; q++) { uint64_t dd = (uint64_t)abs((int)x[k + q] - (int)y[k + q]); if (dd > maxd) maxd = dd; } }
            if (dpx) { g_fzc_diff++; MLOG("【红】局内双渲染：帧 %llu 通道 %llu 附件 ser=%llu 合并 vs 普通 不同像素 %llu / %lu（逐字节最大差 %llu）", frame, pass, ser, (unsigned long long)dpx, (unsigned long)(nb / bpp), (unsigned long long)maxd); }
            else { g_fzc_same++; MLOG("局内双渲染：帧 %llu 通道 %llu 附件 ser=%llu 合并 vs 普通 逐字节相同（%lu 像素）", frame, pass, ser, (unsigned long)(nb / bpp)); }
            if ((g_fzc_same + g_fzc_diff) % 300 == 0) MLOG("局内双渲染累计：逐字节相同 %llu 次、不同 %llu 次", (unsigned long long)g_fzc_same, (unsigned long long)g_fzc_diff);
        }];
    }
    }
out:
    for (int i = 0; i < g_nfzs; i++) free(g_fzs[i].S_);
    g_nfzs = 0; g_fzs_bad = 0;
}
static void fz_end(void) {   // End of a merged render pass (called by rp_end): interrupted before entering the later pass -> retract the skip entries and restore the targets to the earlier pass's; otherwise restore to the later pass's
    if (!g_fz.active) return;
    if (g_fz.phase == 1) { fz_unskip(); R.rt = g_fz.rt1; } else R.rt = g_fz.rt2;
    g_fz.active = 0;
    if (g_fz.phase == 2 && fz_check_on() && !g_fz_checking) { fuse_t fz = g_fz; R.rt = fz.rt2; fz_check_run(&fz); }
    else { for (int i = 0; i < g_nfzs; i++) free(g_fzs[i].S_); g_nfzs = 0; g_fzs_bad = 0; }
}
static int sub_eq(const p_set_rt *a, int ka, const p_set_rt *b, int kb) { uint64_t s1, s2; int l1, l2, x1, x2; rt_sub(a, ka, &s1, &l1, &x1); rt_sub(b, kb, &s2, &l2, &x2); return s1 && s1 == s2 && l1 == l2 && x1 == x2; }
static int prog_samples_ser(const prog_t *pg, uint64_t tex[6][128], uint64_t img[6][32], uint64_t ser) {   // Whether this texture is bound in any texture / image / typed buffer slot declared by the program's reflection
    __unsafe_unretained NvProg *pr = pg ? prog_get(pg) : nil; if (!pr) return 1;
    for (int st = 0; st < 5; st++) { __unsafe_unretained NvShader *sh = pr->sh[st]; if (!sh || sh->stage >= 6) continue;
        for (int i = 0; i < sh->rf.nb; i++) { const nvn_rbind *b = &sh->rf.b[i]; uint64_t h;
            if (b->kind == NVN_RB_IMG) h = b->reg < 32 ? img[sh->stage][b->reg] : 0; else if (b->kind == NVN_RB_TEX || b->kind == NVN_RB_TBUF) h = b->reg < 128 ? tex[sh->stage][b->reg] : 0; else continue;
            if (h && h_ser(h) == ser) return 1; } }
    return 0;
}
static int g_defer_env_ab, g_defer_env_off;   // Environment switches: read once at the start of each submit (defer_reset); the hot path does not call getenv
static void defer_env(void) { const char *x = getenv("DM_NVN_DEFER_AB"), *e = getenv("DM_NVN_DEFER"); g_defer_env_ab = x && !strcmp(x, "1"); g_defer_env_off = e && !strcmp(e, "0"); }
static int defer_on(void) {
    if (!g_defer_env_ab) return tbdr_on() && !g_defer_env_off;
    if (g_defer_ab < 0) { g_defer_ab = 0; g_defer_ab_on = 0; }
    return tbdr_on() && (g_defer_ab || g_defer_ab_on); }
static int defer_handle_sub(uint64_t h, const p_set_rt *r, int k, int cover) {
    int tid = (int)(h & 0xfffff); if (!tid || tid >= g_ntreg || !g_treg[tid].ok) return 0;
    uint64_t ser; int lv, sl; rt_sub(r, k, &ser, &lv, &sl); if (!ser || g_treg[tid].t.serial != ser) return 0;
    if (!g_treg[tid].has_v || !g_treg[tid].v.set_levels) return cover;
    view_t v = g_treg[tid].v; int n = v.nlevels > 0 ? v.nlevels : 1;
    return lv >= v.base_level && lv < v.base_level + n && (!v.set_layers || (sl >= v.min_layer && sl < v.min_layer + (v.nlayers > 0 ? v.nlayers : 1)));
}
static int defer_prog_hits(const prog_t *pg, uint64_t tex[6][128], uint64_t img[6][32], const p_set_rt *r, int color_only, int cover) {
    __unsafe_unretained NvProg *pr = pg ? prog_get(pg) : nil; if (!pr) return 1;
    for (int st = 0; st < 5; st++) { __unsafe_unretained NvShader *sh = pr->sh[st]; if (!sh || sh->stage < 0 || sh->stage >= 6) continue;
        for (int i = 0; i < sh->rf.nb; i++) { const nvn_rbind *b = &sh->rf.b[i]; uint64_t h = 0;
            if (b->kind == NVN_RB_IMG) h = b->reg < 32 ? img[sh->stage][b->reg] : 0;
            else if (b->kind == NVN_RB_TEX || b->kind == NVN_RB_TBUF) h = b->reg < 128 ? tex[sh->stage][b->reg] : 0;
            else continue;
            if (!h) continue;
            for (int k = 0; k < (color_only ? r->n : 9); k++) if ((color_only ? r->color[k] : (k < 8 ? r->color[k] : r->depth)) && defer_handle_sub(h, r, k, cover)) return 1;
        }
    }
    return 0;
}
static void defer_collect_reads(defer_t *d, const prog_t *pg, uint64_t tex[6][128], uint64_t img[6][32]) {
    __unsafe_unretained NvProg *pr = pg ? prog_get(pg) : nil; if (!pr) { d->nreads = 513; return; }
    for (int st = 0; st < 5; st++) { __unsafe_unretained NvShader *sh = pr->sh[st]; if (!sh || sh->stage < 0 || sh->stage >= 6) continue;
        for (int i = 0; i < sh->rf.nb; i++) { const nvn_rbind *b = &sh->rf.b[i]; uint64_t h = 0;
            if (b->kind == NVN_RB_IMG) h = b->reg < 32 ? img[sh->stage][b->reg] : 0;
            else if (b->kind == NVN_RB_TEX || b->kind == NVN_RB_TBUF) h = b->reg < 128 ? tex[sh->stage][b->reg] : 0;
            else continue;
            if (!h) continue; int seen = 0; for (int j = 0; j < d->nreads; j++) seen |= d->p_reads[j] == h;
            if (!seen && d->nreads < 512) d->p_reads[d->nreads++] = h;
        }
    }
}
static int defer_rt_hits_reads(const p_set_rt *r, int idx) {
    if (idx < 0 || idx >= r->n || !r->color[idx]) return 0;
    for (int i = 0; i < g_defer.nreads; i++) if (defer_handle_sub(g_defer.p_reads[i], r, idx, 1)) return 1;
    return 0;
}
static int defer_q_superset(const p_set_rt *q) {
    memset(g_defer.lmap, 0xff, sizeof g_defer.lmap);
    for (int i = 0; i < g_defer.p_rt.n; i++) if (g_defer.p_rt.color[i]) { int found = 0;
        // The indices must be the same: a normal fragment function's color(n) output is written to physical attachment n, and lmap only changes the write mask / blend and cannot move the output (only the _bf version with output mapping constants can).
        // With different indices, deferral would write P's output into a different attachment of Q.
        if (i < q->n && q->color[i] && sub_eq(&g_defer.p_rt, i, q, i)) { g_defer.lmap[i] = (uint8_t)i; found = 1; }
        if (!found) return 0;
    }
    return 1;
}
static int defer_take(const p_draw *dw, const p_draw_idx *di) {
    if (g_defer.n >= 64) { g_defer.reason = 20; return 0; }
    defer_snap *z = &g_defer.snap[g_defer.n++]; z->idx = di != NULL; if (dw) z->dw = *dw; if (di) z->di = *di;
    z->sc = R.sc; z->has_sc = R.has_sc; z->S_ = malloc(sizeof S); memcpy(z->S_, &S, sizeof S); for (int k = 0; k < 8; k++) z->pm[k] = (uint8_t)k; g_defer_draws++; return 1;
}
static void defer_release(void) { defer_release_one(&g_defer); }
static void defer_env(void);
static void defer_reset(void) { defer_clear_all(); g_defer_replaying = 0; defer_env(); }
static void defer_apply_p_clear(void) {
    memset(R.clear_c, 0, sizeof R.clear_c); R.clear_d = R.clear_s = 0;
    for (int j = 0; j < g_defer.q_rt.n; j++) { int i = g_defer.lmap[j]; if (i >= 0 && g_defer.clear_c[i]) { R.clear_c[j] = 1; R.cc[j] = g_defer.cc[i]; } }
    if (g_defer.clear_d) { R.clear_d = 1; R.cd = g_defer.cd; } if (g_defer.clear_s) { R.clear_s = 1; R.cs = g_defer.cs; }
}
static int defer_begin_q(const uint8_t *op) {
    int found = 0, first = -1;
    for (int i = 0; i < 4; i++) if (g_defer_slots[i].active && !g_defer_slots[i].at_q && g_defer_slots[i].q_op == op) { found = 1; if (first < 0) first = i; }
    if (!found) return 0;
    rp_flush();
    g_defer_cur = &g_defer_slots[first]; R.rt = g_defer.q_rt; R.has_rt = 1;
    for (int i = 0; i < 4 && !R.rt.depth; i++) { const defer_t *d = &g_defer_slots[i]; if (d->active && !d->at_q && d->q_op == op && d->q_rt.depth) { R.rt.depth = d->q_rt.depth; R.rt.dview = d->q_rt.dview; R.rt.has_dview = d->q_rt.has_dview; } }   // Attach the depth-stencil needed by any group (the depths of groups with the same Q have been checked compatible)
    memset(R.clear_c, 0, sizeof R.clear_c); R.clear_d = R.clear_s = 0;
    for (int i = 0; i < 4; i++) { defer_t *d = &g_defer_slots[i]; if (!d->active || d->at_q || d->q_op != op) continue;
        for (int j = 0; j < d->q_rt.n; j++) { int k = d->lmap[j]; if (k >= 0 && d->clear_c[k]) { R.clear_c[j] = 1; R.cc[j] = d->cc[k]; } }
        if (d->clear_d) { R.clear_d = 1; R.cd = d->cd; } if (d->clear_s) { R.clear_s = 1; R.cs = d->cs; }
        d->at_q = 1; g_defer_natq++;
    }
    return 1;
}
static void defer_replay(void) {
    if (g_defer_replaying) return;
    int any = 0; for (int i = 0; i < 4; i++) if (g_defer_slots[i].active && g_defer_slots[i].at_q) any = 1;
    if (!any) return;
    typeof(S) save; memcpy(&save, &S, sizeof S); p_scissor sc = R.sc; int has_sc = R.has_sc; int vis = g_vis_counting;
    const uint8_t *en0 = g_exec_next; g_defer_replaying = 1; g_exec_next = NULL;
    for (int gi, after = 0; (gi = defer_next(1, after)) >= 0;) { defer_t *d = &g_defer_slots[gi]; after = d->born;   // In creation order: the groups' P ranges do not overlap and are arranged in sequence, so the group order is the original draw order
        g_defer_cur = d; if (d->reset_between) g_vis_counting = 0;
        { static int once; if (!once++) MLOG("推迟合并通道：首例 通道 %llu，组 %d 回放绘制 %d 次", g_stat_pass, gi, d->n); }   // With multiple groups, several groups may already have been created before the first replay, so it cannot be judged by g_defer_n == 1
        for (int i = 0; i < d->n; i++) { defer_snap *z = &d->snap[i]; static uint8_t lm[8]; for (int j = 0; j < 8; j++) lm[j] = d->lmap[j] == 0xff || d->lmap[j] >= 8 ? 0xff : z->pm[d->lmap[j]];
            memcpy(&S, z->S_, sizeof S); args_dirty_all(); R.sc = z->sc; R.has_sc = z->has_sc; g_pso_dirty = g_dss_dirty = 1; g_pso_lmap = lm; draw(z->idx ? NULL : &z->dw, z->idx ? &z->di : NULL); }
        defer_release_one(d);
    }
    g_vis_counting = vis; memcpy(&S, &save, sizeof S); args_dirty_all(); R.sc = sc; R.has_sc = has_sc; g_pso_lmap = NULL; g_pso_fsfn = nil; memset(&g_pso_fz, 0, sizeof g_pso_fz); g_pso_dirty = g_dss_dirty = 1; g_exec_next = en0; g_defer_replaying = 0; g_defer_cur = &g_defer_slots[0];
}
static void defer_fallback(void) {
    if (g_defer_replaying || !defer_has_active()) return;
    g_defer_fallbacks++; MLOG("【红】推迟合并通道：提交结束仍有未回放组，按各组 P 原目标兜底");
    if (g_enc) { enc_r_end(g_enc); g_enc = nil; }
    typeof(S) save; memcpy(&save, &S, sizeof S); p_set_rt rt = R.rt; g_defer_replaying = 1;
    for (int gi, after = 0; (gi = defer_next(0, after)) >= 0;) { defer_t *d = &g_defer_slots[gi]; after = d->born; g_defer_cur = d; R.rt = d->p_rt; R.has_rt = 1; memcpy(R.clear_c, d->clear_c, 8); memcpy(R.cc, d->cc, sizeof R.cc);
        R.clear_d = d->clear_d; R.clear_s = d->clear_s; R.cd = d->cd; R.cs = d->cs;
        for (int i = 0; i < d->n; i++) { defer_snap *z = &d->snap[i]; memcpy(&S, z->S_, sizeof S); args_dirty_all(); R.sc = z->sc; R.has_sc = z->has_sc; g_pso_dirty = g_dss_dirty = 1; g_pso_lmap = z->pm; draw(z->idx ? NULL : &z->dw, z->idx ? &z->di : NULL); }
        rp_end(); defer_release_one(d);
    }
    memcpy(&S, &save, sizeof S); args_dirty_all(); R.rt = rt; g_defer_replaying = 0; g_defer_cur = &g_defer_slots[0];
}
static const uint8_t *defer_cont(int *ci) {   // The next segment after a list ends (within the same submit): first the caller return stack, then the remaining handles of this submit; returns NULL when there are none left
    for (;;) { int k = (*ci)++;
        if (k < g_ret_n) { if (g_ret_n > 16) return NULL; return g_ret[g_ret_n - 1 - k]; }
        k -= g_ret_n; if (k < g_nrest) return (const uint8_t *)g_rest[k];
        return NULL; }
}
// Deferral pre-screen (binary search on the store lookahead index, zero scanning): for every color subresource of P, the first "used as attachment / cleared" event after this draw must be "used as attachment under a different SetRenderTargets,
// whose targets cover all of P's colors"; cleared first / never used as attachment again in this submit -> deferral is impossible, return directly. Read events carry no level; handed to the full lookahead to check by level.
static int defer_pre(void) {
    if (!g_sdx_ok || !g_exec_cur || !g_exec_seq || g_sdx_op[g_exec_seq - 1] != g_exec_cur) return 0;
    uint32_t P = g_exec_seq - 1;
    for (int k = 0; k < R.rt.n; k++) { if (!R.rt.color[k]) continue;
        uint64_t ser; int lv, sl; rt_sub(&R.rt, k, &ser, &lv, &sl);
        int b = sdx_bucket(ser, 0); if (b < 0) return 0;
        uint32_t lo = g_sdx_bst[b], end = g_sdx_bst[b + 1], hi = end;
        while (lo < hi) { uint32_t m = (lo + hi) / 2; if (g_sdx[m].seq <= P) lo = m + 1; else hi = m; }
        int ok = 0, same = 0; uint32_t walk = 0, prt = lo > g_sdx_bst[b] && g_sdx[lo - 1].seq == P ? g_sdx[lo - 1].rtseq : SDX_NONE;   // SetRenderTargets sequence number of P's own render pass (taken from this draw's used-as-attachment event)
        for (uint32_t j = lo; j < end; j++) { const sdx_ev *e = &g_sdx[j];
            if (++walk > 64) return 0;   // Too many events (a heavily sampled texture): not taken as a candidate, conservative and avoids a long walk
            if (g_hoist_n && hoist_skipped(e->op)) continue;
            if (e->kind == SDX_READ || e->kind == SDX_NEED) continue;   // Read events carry no level (reading another level is not a conflict); handed to the full lookahead to check by level
            if (e->plane != 0 || e->lv != lv || e->sl != sl) continue;
            if (e->kind == SDX_CLEAR) return 0;
            if (e->rtseq == SDX_NONE || e->rtseq >= g_sdx_nop) return 0;
            const uint8_t *rop = g_sdx_op[e->rtseq]; const p_set_rt *r = (const p_set_rt *)(rop + 8);
            if (rt_same(r, &R.rt)) { if (e->rtseq != prt) same = 1; continue; }   // Same target: one immediately following is a continuation segment of P, one separated by other render passes is Q; the pre-screen does not distinguish them, and if there is still no other candidate at the end it is handed to the full lookahead
            for (int i = 0; i < R.rt.n; i++) { if (!R.rt.color[i]) continue; int f = 0;
                for (int q = 0; q < r->n && !f; q++) if (r->color[q] && sub_eq(&R.rt, i, r, q)) f = 1;
                if (!f) return 0; }
            ok = 1; break; }
        if (!ok && !same) return 0; }
    return 1;
}
static void defer_try(const p_draw *dw, const p_draw_idx *di) {
    if (!defer_on() || defer_has_in_p() || g_defer_replaying || g_fz.active || g_fz_checking || !g_exec_next || !R.rt.n) return;
    int slot = defer_slot_free(); if (slot < 0) return;
    g_defer_cur = &g_defer_slots[slot];
    int ncol = 0; for (int i = 0; i < R.rt.n; i++) ncol += R.rt.color[i] != NULL; if (!ncol) return;   // Render passes with only depth-stencil attached (shadow maps etc.): Q is defined as covering P's colors, so a P without color can never be deferred -- previously the pre-screen let it through idly and the full lookahead scanned dozens of draws for nothing
    int has_clear = R.clear_d || R.clear_s; for (int i = 0; i < 8; i++) has_clear |= R.clear_c[i]; if (!has_clear) return;
    if (!defer_pre()) return;
    memset(&g_defer, 0, sizeof g_defer); g_defer.p_rt = R.rt; memcpy(g_defer.clear_c, R.clear_c, 8); memcpy(g_defer.cc, R.cc, sizeof R.cc);
    g_defer.clear_d = R.clear_d; g_defer.clear_s = R.clear_s; g_defer.cd = R.cd; g_defer.cs = R.cs;
    uint64_t tex[6][128], img[6][32]; memcpy(tex, S.tex, sizeof tex); memcpy(img, S.img, sizeof img); const prog_t *pg = S.prog; ds_st ds = S.ds;
    const uint8_t *p = g_exec_next, *p_end = NULL, *qop = NULL; const p_set_rt *q = NULL; p_set_rt scan_rt = R.rt; int srt_hit = -1, ci = 0, draws = 1, p_has_draw = 1; defer_collect_reads(&g_defer, pg, tex, img);   // The current draw is P's first
    for (int guard = 0; guard < 200000; guard++) {
        op_hdr h; memcpy(&h, p, sizeof h); const void *a = p + 8;
        if (g_hoist_n && hoist_skipped(p)) { p += h.size; continue; }
        switch (h.op) {
            case OP_END: { const uint8_t *nx = defer_cont(&ci); if (!nx) { g_defer.reason = 1; goto fail; } p = nx; continue; }
            case OP_CALL: g_defer.reason = 2; goto fail;
            case OP_JUMP: memcpy(&p, a, 8); continue;
            case OP_BIND_PROGRAM: pg = ((const p_bind_program *)a)->prog; break;
            case OP_BIND_TEX: case OP_BIND_IMG: { const p_bind_handles *b = a; bh_fresh(b); int lim = h.op == OP_BIND_IMG ? 32 : 128;
                if (b->stage >= 0 && b->stage < 6 && b->first >= 0 && b->first + b->n <= lim) for (int k = 0; k < b->n; k++) { if (h.op == OP_BIND_IMG) img[b->stage][b->first + k] = b->h[k]; else tex[b->stage][b->first + k] = b->h[k]; } break; }
            case OP_BIND_STATE: { const p_bind_state *b = a; if (b->kind == 5) memcpy(&ds, b->bytes, sizeof ds); break; }
            case OP_DRAW: case OP_DRAW_IDX:
                if (defer_prog_hits(pg, tex, img, &g_defer.p_rt, 1, 1)) { /* Suggestion: here, after identifying that an output mapping is needed but _bf is missing, call bf_want_note. */ g_defer.reason = 7; goto fail; }
                if (p_end) {   // Draws in the intermediate range: must not write textures P reads, must not write P's depth-stencil (and must not test stencil when P writes stencil)
                    if (srt_hit < 0) { srt_hit = 0; for (int k = 0; k < scan_rt.n && !srt_hit; k++) if (scan_rt.color[k] && defer_rt_hits_reads(&scan_rt, k)) srt_hit = 1; }   // Attachments change only at SetRenderTargets: computed once per segment
                    if (srt_hit) { g_defer.reason = 12; goto fail; }
                    if (scan_rt.depth && g_defer.p_rt.depth && scan_rt.depth->serial == g_defer.p_rt.depth->serial && (ds.dw || (g_defer.p_st && ds.st))) { g_defer.reason = 13; goto fail; }
                    break; }
                draws++; if (draws > 64) { g_defer.reason = 20; goto fail; } p_has_draw = 1; defer_collect_reads(&g_defer, pg, tex, img); g_defer.p_dw |= ds.dw; g_defer.p_st |= ds.st && ((S.st_wmask[0] | S.st_wmask[1]) != 0) && (ds.fail_f != NVN_STENCIL_OP_KEEP || ds.zfail_f != NVN_STENCIL_OP_KEEP || ds.zpass_f != NVN_STENCIL_OP_KEEP || ds.fail_b != NVN_STENCIL_OP_KEEP || ds.zfail_b != NVN_STENCIL_OP_KEEP || ds.zpass_b != NVN_STENCIL_OP_KEEP); if (g_defer.p_dw) { g_defer.reason = 8; goto fail; } break;
            case OP_RESET_COUNTER: g_defer.reset_between = 1; break;
            case OP_COPY_T2T: { const p_copy_t2t *c = a;   // Texture copy: allowed only if neither end touches P's attachments and the destination is not a texture P samples
                uint64_t ss = c->src ? c->src->serial : 0, ds_ = c->dst ? c->dst->serial : 0; int bad = !ss || !ds_;
                for (int k = 0; k < g_defer.p_rt.n && !bad; k++) if (g_defer.p_rt.color[k] && (g_defer.p_rt.color[k]->serial == ss || g_defer.p_rt.color[k]->serial == ds_)) bad = 1;
                if (g_defer.p_rt.depth && (g_defer.p_rt.depth->serial == ss || g_defer.p_rt.depth->serial == ds_)) bad = 1;
                for (int k = 0; k < g_defer.nreads && !bad; k++) if (h_ser(g_defer.p_reads[k]) == ds_) bad = 1;
                if (g_defer.nreads > 512) bad = 1;
                if (bad) { g_defer.reason = 3; g_defer.reason_op = h.op; goto fail; } break; }
            case OP_REPORT_COUNTER: case OP_UPDATE_UB: case OP_COPY_B2B: case OP_CLEAR_BUF: case OP_COPY_B2T: case OP_COPY_T2B: case OP_CLEAR_TEX: case OP_DISPATCH: g_defer.reason = 3; g_defer.reason_op = h.op; goto fail;
            case OP_CLEAR_COLOR: { const p_clear_color *c = a; if (!p_end) { g_defer.reason = 16; goto fail; }   // A clear inside P's continuation segment: do not defer
                if (c->index >= 0 && c->index < scan_rt.n && (defer_rt_hits_reads(&scan_rt, c->index) || (scan_rt.color[c->index] && defer_rt_hits_reads(&g_defer.p_rt, c->index)))) { g_defer.reason = 5; goto fail; } break; }
            case OP_CLEAR_DS: if (!p_end) { g_defer.reason = 16; goto fail; } break;
            case OP_SET_RT: { const p_set_rt *r = a;
                if (!p_end && rt_same(r, &g_defer.p_rt)) break;   // The same target set once more (the game often sends it repeatedly): still a continuation segment of P, and its draws are deferred together
                if (!p_end) p_end = p; int shared = 0; for (int i = 0; i < g_defer.p_rt.n; i++) for (int j = 0; j < r->n; j++) if (g_defer.p_rt.color[i] && r->color[j] && sub_eq(&g_defer.p_rt, i, r, j)) shared = 1;
                if (shared) {   // An empty render pass (targets set, then changed again without drawing or clearing) touches no attachment: it is neither taken as Q nor counted as a conflict
                    const uint8_t *e = p + h.size; int ec = ci, empty = 0;
                    for (int g2 = 0; g2 < 200000; g2++) { op_hdr he; memcpy(&he, e, sizeof he);
                        if (he.op == OP_JUMP) { memcpy(&e, e + 8, 8); continue; }
                        if (he.op == OP_END) { const uint8_t *nx = defer_cont(&ec); if (!nx) break; e = nx; continue; }
                        if (he.op == OP_SET_RT) { empty = 1; break; }
                        if (he.op == OP_DRAW || he.op == OP_DRAW_IDX || he.op == OP_CLEAR_COLOR || he.op == OP_CLEAR_DS || he.op == OP_CALL || he.op == OP_DISPATCH || he.op == OP_CLEAR_TEX || he.op == OP_COPY_T2T || he.op == OP_COPY_B2T || he.op == OP_COPY_T2B) break;
                        e += he.size; }
                    if (empty) break; }
                if (!shared) break; if (!p_has_draw || !defer_q_superset(r)) { g_defer.reason = 6; goto fail; } q = r; qop = p; goto q_found; }
            default: break;
        }
        if (h.op == OP_SET_RT) { scan_rt = *(const p_set_rt *)a; srt_hit = -1; }
        p += h.size;
    }
fail:
    if (g_defer.reason && passdump()) PD("G %llu 不成立 reason=%d op=%d", g_stat_pass + 1, g_defer.reason, g_defer.reason_op); memset(&g_defer, 0, sizeof g_defer); return;
q_found:
    p = qop; op_hdr qh; memcpy(&qh, p, sizeof qh); p += qh.size; int q_body = 0, q_clears = 0;   // Binding tracking continues from the main scan's state (does not go back to S at the start of P)
    for (int guard = 0; guard < 200000; guard++) { op_hdr h; memcpy(&h, p, sizeof h); const void *a = p + 8; if (h.op == OP_JUMP) { memcpy(&p, a, 8); continue; }
        // The whole Q segment (up to the next SetRenderTargets / end of list): no draw may sample P's color attachments through a view covering P's level -- after deferral P's result is still in the tile and has not landed in main memory, so sampling would read old content
        if (h.op == OP_DRAW || h.op == OP_DRAW_IDX) { if (defer_prog_hits(pg, tex, img, &g_defer.p_rt, 1, 1)) { g_defer.reason = 14; goto fail; } q_body = 1; p += h.size; continue; }
        if (q_body && (h.op == OP_SET_RT || h.op == OP_END)) break;
        if (!q_body && h.op == OP_END) { const uint8_t *nx = defer_cont(&ci); if (!nx) { g_defer.reason = 9; goto fail; } p = nx; continue; }
        if (h.op == OP_CALL) { g_defer.reason = 9; g_defer.reason_op = h.op; goto fail; }
        // After Q's SetRenderTargets and before its first draw and any clear, another SetRenderTargets arrives (the game sets targets repeatedly, the previous one being an empty render pass): if the new targets still cover P take it as Q, otherwise do not defer
        if (!q_body && h.op == OP_SET_RT && !q_clears) { const p_set_rt *r2 = a; if (!defer_q_superset(r2)) { g_defer.reason = 15; goto fail; } q = r2; qop = p; p += h.size; continue; }
        if (h.op == OP_CLEAR_COLOR || h.op == OP_CLEAR_DS) q_clears = 1;
        if (!q_body && (h.op == OP_END || h.op == OP_SET_RT || h.op == OP_UPDATE_UB || h.op == OP_COPY_B2B || h.op == OP_CLEAR_BUF || h.op == OP_DISPATCH)) { g_defer.reason = 9; g_defer.reason_op = h.op; goto fail; }
        if (h.op == OP_BIND_PROGRAM) pg = ((const p_bind_program *)a)->prog; else if (h.op == OP_BIND_TEX || h.op == OP_BIND_IMG) { const p_bind_handles *b = a; bh_fresh(b); int lim = h.op == OP_BIND_IMG ? 32 : 128; if (b->stage >= 0 && b->stage < 6 && b->first >= 0 && b->first + b->n <= lim) for (int k = 0; k < b->n; k++) { if (h.op == OP_BIND_IMG) img[b->stage][b->first + k] = b->h[k]; else tex[b->stage][b->first + k] = b->h[k]; } }
        else if (h.op == OP_CLEAR_COLOR) { const p_clear_color *c = a; if (c->index >= 0 && c->index < q->n && q->color[c->index]) for (int i = 0; i < g_defer.p_rt.n; i++) if (g_defer.p_rt.color[i] && sub_eq(q, c->index, &g_defer.p_rt, i)) { g_defer.reason = 10; goto fail; } }
        p += h.size;
    }
    if (!q || !defer_q_superset(q) || g_defer.n >= 64) { g_defer.reason = 11; goto fail; }
    for (int i = 0; i < 4; i++) if (&g_defer_slots[i] != &g_defer && g_defer_slots[i].active && g_defer_slots[i].q_op == qop && !defer_same_q_depth(&g_defer_slots[i], q)) { g_defer.reason = 17; goto fail; }
    g_defer.q_rt = *q; if (!q->depth && g_defer.p_rt.depth) { g_defer.q_rt.depth = g_defer.p_rt.depth; g_defer.q_rt.dview = g_defer.p_rt.dview; g_defer.q_rt.has_dview = g_defer.p_rt.has_dview; } { int na = 0; for (int i = 0; i < 4; i++) { const defer_t *d = &g_defer_slots[i]; if (d != &g_defer && d->active && d->at_q) na += d->n; }
      if (na + 1 > 64) { g_defer.reason = 20; goto fail; } }
    g_defer.q_op = qop; g_defer.p_end_op = p_end; g_defer.in_p = 1; g_defer.active = 1; g_defer_nact++; g_defer_pend = p_end; g_defer_n++; g_defer.born = g_defer_n;
    // Chained: this render pass is exactly the Q of upstream groups (they are at_q, waiting to be replayed when this render pass opens), and this render pass is itself deferred as a whole -- the upstream groups are merged into the new group in creation order:
    // the target space of an upstream group = this render pass's targets = the new group's P targets, and the snapshot's mapping is composed as pm' = upstream lmap o pm; the upstream clears are already in R and are carried along with the new group below
    for (int gi, after = 0; (gi = defer_next(1, after)) >= 0;) { defer_t *d = &g_defer_slots[gi]; after = d->born; if (d == &g_defer) continue;
        for (int k = 0; k < 8; k++) if (d->lmap[k] != 0xff && d->lmap[k] != k) { g_defer.reason = 18; goto fail; }   // Only merge in upstream groups with an identity mapping (see defer_q_superset)
        for (int i = 0; i < d->n; i++) { defer_snap z = d->snap[i]; uint8_t pm[8]; for (int k = 0; k < 8; k++) pm[k] = d->lmap[k] == 0xff || d->lmap[k] >= 8 ? 0xff : z.pm[d->lmap[k]];
            memcpy(z.pm, pm, 8); g_defer.snap[g_defer.n++] = z; }
        g_defer.reset_between |= d->reset_between; d->n = 0; defer_release_one(d); } if (!defer_take(dw, di)) { defer_release_one(&g_defer); return; }
    memset(R.clear_c, 0, sizeof R.clear_c); R.clear_d = R.clear_s = 0; PD("G %llu 成立 组 %d P→Q draws=%d", g_stat_pass + 1, slot, draws);
}
// Pre-screen of the necessary condition for merging (binary search on the submit index, zero scanning on failure): merging requires that "the first draw of the later pass (the next SET_RT with a different target that has not been hoisted)
// samples some color attachment of the earlier pass", i.e. there is an SDX_READ event for that attachment at that draw's sequence number. If not satisfied there is certainly no merge, saving the per-operation scan below;
// only if satisfied does it go through the original planning (the decision is exactly the same as without the pre-screen). When the index is unavailable, plan as before.
static int fz_prefilter(void) {
    if (!g_sdx_ok || !g_exec_seq || g_sdx_op[g_exec_seq - 1] != g_exec_cur) return 1;
    uint32_t P = g_exec_seq - 1, lo = 0, hi = g_rtx_n;
    while (lo < hi) { uint32_t m = (lo + hi) / 2; if (g_rtx_seq[m] <= P) lo = m + 1; else hi = m; }
    for (; lo < g_rtx_n; lo++) { const uint8_t *op = g_sdx_op[g_rtx_seq[lo]]; if (g_hoist_n && hoist_skipped(op)) continue; if (!rt_same(&R.rt, (const p_set_rt *)(op + 8))) break; }
    if (lo >= g_rtx_n) return 0;
    uint32_t r2 = g_rtx_seq[lo], a = 0, b = g_drx_n;
    while (a < b) { uint32_t m = (a + b) / 2; if (g_drx_seq[m] <= r2) a = m + 1; else b = m; }
    if (a >= g_drx_n) return 0;
    uint32_t fd = g_drx_seq[a];
    for (uint32_t k = lo + 1; k < g_rtx_n && g_rtx_seq[k] < fd; k++) if (!(g_hoist_n && hoist_skipped(g_sdx_op[g_rtx_seq[k]]))) return 0;   // The target changes again before the first draw: step two of the original planning gives up as well
    for (int j = 0; j < R.rt.n && j < 8; j++) { if (!R.rt.color[j]) continue; int bk = sdx_bucket(R.rt.color[j]->serial, 0); if (bk < 0) continue;
        uint32_t e0 = g_sdx_bst[bk], e1 = g_sdx_bst[bk + 1];
        while (e0 < e1) { uint32_t m = (e0 + e1) / 2; if (g_sdx[m].seq < fd) e0 = m + 1; else e1 = m; }
        for (; e0 < g_sdx_bst[bk + 1] && g_sdx[e0].seq == fd; e0++) if (g_sdx[e0].kind == SDX_READ) return 1; }
    return 0;
}
static void fuse_try(void) {
    if (g_fz.active || !g_exec_next || g_hoist_n + 12 > HOIST_MAX || !R.rt.n) return;
    if (!fz_prefilter()) return;
    // One: from the earlier pass to the next SetRenderTargets, only draws and state are allowed
    const uint8_t *p = NULL, *r2op = NULL, *reuse[8]; int nreuse = 0; const p_set_rt *r2 = NULL; p_scissor sc = R.sc; int has_sc = R.has_sc;
    fz_cursor cur = {.p = g_exec_next}; op_hdr h; const void *a;
    for (int guard = 0; guard < 200000 && !r2; guard++) {
        if (!fz_next(&cur, &p, &h, &a)) return;
        switch (h.op) {
            case OP_COPY_T2T: case OP_COPY_B2T: case OP_COPY_T2B: case OP_CLEAR_TEX: case OP_DISPATCH:
            case OP_CLEAR_COLOR: case OP_CLEAR_DS: case OP_UPDATE_UB: case OP_COPY_B2B: case OP_CLEAR_BUF: return;
            case OP_SCISSOR: sc = *(const p_scissor *)a; has_sc = 1; break;
            case OP_SET_RT: {
                int pend = R.clear_d || R.clear_s; for (int i = 0; i < 8; i++) pend |= R.clear_c[i];
                if (rt_same(&R.rt, a)) {
                    if (g_enc_mode == 1 || pend || !R.has_rt || !tbdr_on() || nreuse >= 8) return;
                    reuse[nreuse++] = p; continue;
                }
                r2 = a; r2op = p; continue;
            }
            default: break;
        }
    }
    if (!r2 || rt_same(&R.rt, r2)) return;
    // Two: from the start of the later pass to its first draw, only clears (not clearing shared attachments) and state are allowed
    const uint8_t *cl[8]; int ncl = 0; uint8_t clr[8] = {0}; MTLClearColor cc[8]; int cd_f = 0, cs_f = 0; double cdv = 0; uint32_t csv = 0;
    if (!fz_next(&cur, &p, &h, &a)) return;   // Continue from the cursor of step one (already past the later pass's SET_RT, keeping the cross-handle / return stack position)
    for (int guard = 0; guard < 200000; guard++) {
        if (h.op == OP_DRAW || h.op == OP_DRAW_IDX) break;
        switch (h.op) {
            case OP_COPY_T2T: case OP_COPY_B2T: case OP_COPY_T2B: case OP_CLEAR_TEX: case OP_DISPATCH: case OP_SET_RT: case OP_UPDATE_UB: case OP_COPY_B2B: case OP_CLEAR_BUF: return;
            case OP_SCISSOR: sc = *(const p_scissor *)a; has_sc = 1; break;
            case OP_CLEAR_COLOR: { const p_clear_color *c = a; if (ncl >= 8 || c->index < 0 || c->index >= r2->n || !r2->color[c->index] || c->mask != 0xf) return;
                const tex_t *t = r2->color[c->index];
                if (has_sc && !(sc.x <= 0 && sc.y <= 0 && sc.x + sc.w >= (int)t->d.width && sc.y + sc.h >= (int)t->d.height)) return;
                clr[c->index] = 1; cc[c->index] = MTLClearColorMake(c->rgba[0], c->rgba[1], c->rgba[2], c->rgba[3]); cl[ncl++] = p; break; }
            case OP_CLEAR_DS: { const p_clear_ds *c = a; if (ncl >= 8 || !r2->depth) return;
                if (c->depth_mask) { cd_f = 1; cdv = c->depth; } if (c->stencil_mask) { cs_f = 1; csv = (uint32_t)c->stencil; } cl[ncl++] = p; break; }
            default: break;
        }
        if (!fz_next(&cur, &p, &h, &a)) return;
    }
    // Three: union of attachments (the earlier pass's attachments keep their original indices, those unique to the later pass are appended), consistent sizes, compatible depth
    p_set_rt U = R.rt; int n = R.rt.n; int8_t p2map[8]; memset(p2map, -1, sizeof p2map);
    for (int k = 0; k < r2->n && k < 8; k++) { if (!r2->color[k]) continue;
        int j = -1; for (int m = 0; m < R.rt.n; m++) if (R.rt.color[m] && sub_eq(&R.rt, m, r2, k)) { j = m; break; }
        if (j < 0) { if (n >= 8) return; j = n++; U.color[j] = r2->color[k]; U.cview[j] = r2->cview[k]; U.has_cview[j] = r2->has_cview[k]; }
        else if (clr[k]) return;   // The start of the later pass wants to clear an attachment the earlier pass is drawing to: after merging this clear has nowhere to go
        p2map[k] = (int8_t)j; }
    U.n = n;
    { uint64_t s0; int l0, x0; rt_sub(&U, 0, &s0, &l0, &x0); if (!U.color[0]) return;
      uint32_t w0 = MAX(U.color[0]->d.width >> l0, 1u), h0 = MAX(U.color[0]->d.height >> l0, 1u);
      for (int j = 1; j < n; j++) { uint64_t s_; int l_, x_; rt_sub(&U, j, &s_, &l_, &x_); if (!U.color[j] || MAX(U.color[j]->d.width >> l_, 1u) != w0 || MAX(U.color[j]->d.height >> l_, 1u) != h0) return; } }
    int depth_from_r2 = 0;
    if (R.rt.depth && r2->depth && !sub_eq(&R.rt, 8, r2, 8)) return;
    if (!R.rt.depth && r2->depth) { U.depth = r2->depth; U.dview = r2->dview; U.has_dview = r2->has_dview; depth_from_r2 = 1; }
    if ((cd_f || cs_f) && !depth_from_r2) return;   // The depth is already in use in the earlier pass: the depth clear at the start of the later pass has nowhere to go
    // Four: draws of the earlier pass must not sample attachments brought in by the later pass; when the depth comes from the later pass, depth-stencil testing must not be enabled
    static uint64_t tex[6][128], img[6][32]; memcpy(tex, S.tex, sizeof tex); memcpy(img, S.img, sizeof img);
    const prog_t *pg = S.prog; ds_st ds = S.ds;
    cur = (fz_cursor){.p = g_exec_next};
    for (;;) {   // The first loop iteration checks the draw that opens the render pass (current state), then checks the draws in the range one by one
        if (depth_from_r2 && (ds.dt || ds.dw || ds.st)) return;
        for (int j = R.rt.n; j < n; j++) if (prog_samples_ser(pg, tex, img, U.color[j]->serial)) return;
        if (depth_from_r2 && prog_samples_ser(pg, tex, img, U.depth->serial)) return;
        for (;;) {
            const uint8_t *q; op_hdr hq; const void *aq;
            if (!fz_next(&cur, &q, &hq, &aq)) return;
            if (q == r2op) goto planned;
            if (hq.op == OP_BIND_PROGRAM) pg = ((const p_bind_program *)aq)->prog;
            else if (hq.op == OP_BIND_STATE) { const p_bind_state *b = aq; if (b->kind == 5) memcpy(&ds, b->bytes, sizeof ds); }
            else if (hq.op == OP_BIND_TEX || hq.op == OP_BIND_IMG) { const p_bind_handles *b = aq; bh_fresh(b); int lim = hq.op == OP_BIND_IMG ? 32 : 128;
                if (b->stage >= 0 && b->stage < 6 && b->first >= 0 && b->first + b->n <= lim) for (int k = 0; k < b->n; k++) { if (hq.op == OP_BIND_IMG) img[b->stage][b->first + k] = b->h[k]; else tex[b->stage][b->first + k] = b->h[k]; } }
            else if (hq.op == OP_DRAW || hq.op == OP_DRAW_IDX) break;
        }
    }
planned:
    {   // Five: pre-screen the first draw of the later pass (static conditions), to avoid attaching the union of attachments for nothing: track the program and texture bindings up to that draw
        const prog_t *pg2 = S.prog; static uint64_t t2[6][128], i2[6][32]; memcpy(t2, S.tex, sizeof t2); memcpy(i2, S.img, sizeof i2);
        cur = (fz_cursor){.p = g_exec_next};
        for (;;) { const uint8_t *q; op_hdr hq; const void *aq; if (!fz_next(&cur, &q, &hq, &aq)) return; if (q == p) break;
            if (hq.op == OP_BIND_PROGRAM) pg2 = ((const p_bind_program *)aq)->prog;
            else if (hq.op == OP_BIND_TEX || hq.op == OP_BIND_IMG) { const p_bind_handles *b = aq; bh_fresh(b); int lim = hq.op == OP_BIND_IMG ? 32 : 128;
                if (b->stage >= 0 && b->stage < 6 && b->first >= 0 && b->first + b->n <= lim) for (int k = 0; k < b->n; k++) { if (hq.op == OP_BIND_IMG) i2[b->stage][b->first + k] = b->h[k]; else t2[b->stage][b->first + k] = b->h[k]; } }
        }
        NvProg *pr2 = pg2 ? prog_get(pg2) : nil; NvShader *vs2 = pr2 ? pr2->sh[0] : nil, *fs2 = pr2 ? pr2->sh[1] : nil;
        if (!vs2 || !fs2) return;
        int touches = 0;
        for (int st = 0; st < 5; st++) { __unsafe_unretained NvShader *sh = pr2->sh[st]; if (!sh || sh->stage >= 6) continue;
            for (int i = 0; i < sh->rf.nb; i++) { const nvn_rbind *bb = &sh->rf.b[i]; uint64_t h;
                if (bb->kind == NVN_RB_IMG) h = bb->reg < 32 ? i2[sh->stage][bb->reg] : 0; else if (bb->kind == NVN_RB_TEX || bb->kind == NVN_RB_TBUF) h = bb->reg < 128 ? t2[sh->stage][bb->reg] : 0; else continue;
                uint64_t ser = h ? h_ser(h) : 0; if (!ser) continue;
                int j = -1; for (int m = 0; m < R.rt.n; m++) if (R.rt.color[m] && R.rt.color[m]->serial == ser) { j = m; break; }
                if (j < 0) continue;
                if (sh != fs2 || bb->kind != NVN_RB_TEX || (fs2->rf.fc_mixed[bb->reg >> 6] >> (bb->reg & 63) & 1)) return;
                int any = 0;
                for (int c = 0; c < fs2->rf.nfc; c++) if (fs2->rf.fc_reg[c] == bb->reg) { int l = fs2->rf.fc_locn[c];
                    if (l >= 32 || (vs2->rf.uv_attr[l] < 0 && vs2->rf.uv_aff_attr[l] < 0)) return; any = 1; }
                if (!any) return;
                touches = 1; } }
        if (!touches) return;   // The later pass does not read the earlier pass's attachments: merging has no benefit
        int ident = 1; for (int c = 0; c < 8; c++) if (fs2->rf.color_mask >> c & 1) { if (p2map[c] < 0) return; if (p2map[c] != c) ident = 0; }
        if (!ident && !fs2->rf.nfo) return;
    }
    memset(&g_fz, 0, sizeof g_fz);
    g_fz.active = 1; g_fz.phase = 1; g_fz.n1 = R.rt.n; g_fz.nreuse = nreuse; g_fz.rt1 = R.rt; g_fz.rt2 = *r2; g_fz.rt2_op = r2op; memcpy(g_fz.reuse, reuse, sizeof reuse); memcpy(g_fz.p2map, p2map, 8);
    for (int j = 0; j < 8; j++) g_fz.lmap1[j] = j < R.rt.n ? (uint8_t)j : 0xff;
    fz_probe_cov2(r2, p, p2map, clr, g_fz.n1, n);
    g_fz.skip[g_fz.nskip++] = r2op; for (int i = 0; i < ncl; i++) g_fz.skip[g_fz.nskip++] = cl[i];
    for (int i = 0; i < g_fz.nskip; i++) g_hoist_skip[g_hoist_n++] = g_fz.skip[i];
    R.rt = U;
    memcpy(g_fz.clr2, clr, 8); memcpy(g_fz.cc2, cc, sizeof cc); g_fz.cd2f = (uint8_t)cd_f; g_fz.cs2f = (uint8_t)cs_f; g_fz.cd2 = cdv; g_fz.cs2 = csv;
    for (int k = 0; k < 8; k++) if (clr[k]) { R.clear_c[p2map[k]] = 1; R.cc[p2map[k]] = cc[k]; }
    if (cd_f) { R.clear_d = 1; R.cd = cdv; } if (cs_f) { R.clear_s = 1; R.cs = csv; }
    if (!g_fz_n++) MLOG("可读 tile 通道合并：首例 通道 %llu（物理附件 %d 个）", g_stat_pass + 1, n);
    PD("F %llu 合并下一个通道（可读 tile）：物理附件 %d 个，前一个通道 %d 个", g_stat_pass + 1, n, R.rt.n == n ? g_fz.n1 : g_fz.n1);
}
static id<MTLFunction> bf_fn(NvShader *fs, const int *fx, uint8_t fbmask, const uint8_t *lmap) {   // The readable tile version is specialized by constants (cached)
    os_unfair_lock_lock(&g_fz_fn_lock);
    if (!g_bf_fn_cache) { g_bf_fn_cache = [NSMutableDictionary dictionary]; g_bf_lib_cache = [NSMutableDictionary dictionary]; }
    size_t kl = strlen(fs->key); NSString *base = [NSString stringWithFormat:@"%.*s", (int)(kl > 2 ? kl - 2 : kl), fs->key];
    id<MTLLibrary> lib = g_bf_lib_cache[base];
    if (!lib) {
        NSError *e = nil; NSString *path = [NSString stringWithFormat:@"%@/%@_bf.metallib", g_msl_dir, base];
        lib = lib_from_file(path, &e);
        if (!lib) MLOG("【红】可读 tile 版本载入失败 %s：%s", path.UTF8String, e.localizedDescription.UTF8String);
        else g_bf_lib_cache[base] = lib;
    }
    if (!lib) { os_unfair_lock_unlock(&g_fz_fn_lock); return nil; }
    NSMutableString *k = [NSMutableString stringWithFormat:@"%02x", fbmask];
    for (int i = 0; i < fs->rf.nfc; i++) [k appendFormat:@",%d", fx[i]];
    for (int j = 0; j < 8; j++) [k appendFormat:@"/%02x", lmap[j]];
    NSString *cacheKey = [NSString stringWithFormat:@"%@|%@", base, k];
    id<MTLFunction> f = g_bf_fn_cache[cacheKey];
    if (f) { os_unfair_lock_unlock(&g_fz_fn_lock); return f; }
    MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new];
    for (int i = 0; i < fs->rf.nfc; i++) { int v = fx[i]; [cv setConstantValue:&v type:MTLDataTypeInt atIndex:(NSUInteger)i]; }
    for (int q = 0; q < 8; q++) { bool b = fbmask >> q & 1; [cv setConstantValue:&b type:MTLDataTypeBool atIndex:(NSUInteger)(32 + q)]; }
    for (int j = 0; j < fs->rf.nfo; j++) for (int q = 0; q < 8; q++) { bool b = lmap[q] == fs->rf.fo_color[j]; [cv setConstantValue:&b type:MTLDataTypeBool atIndex:(NSUInteger)(64 + 8 * j + q)]; }
    NSError *e = nil; f = [lib newFunctionWithName:@"main0" constantValues:cv error:&e];
    if (!f) { MLOG("【红】可读 tile 版本特化失败 %s（%s）：%s", fs->key, k.UTF8String, e.localizedDescription.UTF8String); os_unfair_lock_unlock(&g_fz_fn_lock); return nil; }
    g_bf_fn_cache[cacheKey] = f; os_unfair_lock_unlock(&g_fz_fn_lock); return f;
}
static int ubo_float(int slot, uint16_t ref, float *out) {   // Component cmp of vector vec of the uniform slot (read on the CPU side; not read if it falls within a buffer write pending execution in this submit)
    int sl = ref >> 8, vec = ref >> 2 & 63, cmp = ref & 3;
    if (slot < 0 || slot >= 6 || sl < 0 || sl >= 16) return 0;
    const p_bind_buf *u = &S.ubo[slot][sl]; uint64_t off = (uint64_t)vec * 16 + (uint64_t)cmp * 4;
    if (!u->addr || off + 4 > u->size || pend_hits(u->addr + off, 4)) return 0;
    void *mb; uint64_t mo; const uint8_t *base;
    if (!mtl_addr(u->addr + off, 4, &mb, &mo) || !(base = [(__bridge id<MTLBuffer>)mb contents])) return 0;
    memcpy(out, base + mo, 4); return 1;
}
static int zskip_fact_for(const char *key, zskip_fact *out) {
    if (!key) return 0;
    for (int i = 0; i < g_zskip_nf; i++) if (!strncmp(key, g_zskip_facts[i].key, strlen(g_zskip_facts[i].key))) { *out = g_zskip_facts[i]; return 1; }   // Runtime keys carry the _b suffix
    return 0;
}
static int zskip_view_zero(int tid) {
    if (!tid || tid >= g_ntreg || !g_treg[tid].ok) return 0;
    treg_t *r = &g_treg[tid]; const tex_t *t = &r->t; int lv0 = r->has_v && r->v.set_levels ? r->v.base_level : 0;
    int nl = r->has_v && r->v.set_levels && r->v.nlevels > 0 ? r->v.nlevels : 1;
    int sl0 = r->has_v && r->v.set_layers ? r->v.min_layer : 0;
    int ns = r->has_v && r->v.set_layers && r->v.nlayers > 0 ? r->v.nlayers : 1;
    for (int l = 0; l < nl; l++) for (int s = 0; s < ns; s++) if (!zskip_has(t->serial, lv0 + l, sl0 + s)) return 0;
    return 1;
}
// While an occlusion query is counting, a skipped draw should have contributed passed samples: it only does not matter if "there is no report before the next reset".
// Looking forward in replay order from after this draw: sample count reset encountered first -> does not matter (return 1); sample count report encountered first, or the end cannot be seen (submit ends) -> matters (return 0)
static int zskip_vis_dead(void) {
    if (!g_vis_counting) return 1;
    const uint8_t *p = g_exec_next; int ci = 0; if (!p) return 0;
    for (int guard = 0; guard < 200000; guard++) { op_hdr h; memcpy(&h, p, sizeof h); const void *a = p + 8;
        if (h.op == OP_JUMP) { memcpy(&p, a, 8); continue; }
        if (h.op == OP_END) { p = defer_cont(&ci); if (!p) return 0; continue; }
        if (h.op == OP_CALL) return 0;
        if (h.op == OP_RESET_COUNTER || h.op == OP_REPORT_COUNTER) { uint64_t t; memcpy(&t, a, 8); if (t == NVN_COUNTER_TYPE_SAMPLES_PASSED) return h.op == OP_RESET_COUNTER; }
        p += h.size; }
    return 0;
}
static int zskip_can_draw(NvProg *pr) {
    // Skip conditions (all must hold): the fragment key is in the fact table, the view bound to every texture slot declared by its reflection covers only subresources known to be all zero, the coefficient is finite,
    // no depth write and no stencil, not counting an occlusion query, and every color attachment to be written is already 0 (a candidate cleared entirely to 0 in this render pass, or known all zero) -- writing 0 over 0.
    // Reasons for not holding (the N line of PASSDUMP): 1 depth / stencil 2 occlusion query counting with a report before the reset 3 coefficient unreadable or non-finite 4 image slot 5 sampled view not known all zero 6 no target 7 target is not 0
    if (!zskip_on() || !pr) return 0;
    NvShader *fs = nil; for (int st = 0; st < 5; st++) if (pr->sh[st] && pr->sh[st]->stage == 1) fs = pr->sh[st];
    zskip_fact f; float coeff; int why = 0;
    if (!fs || !zskip_fact_for(fs->key, &f)) return 0;
    if (S.ds.dw || S.ds.st) why = 1;
    else if (!zskip_vis_dead()) why = 2;
    else if (f.ubo_slot >= 0 && (!ubo_float(1, (uint16_t)((f.ubo_slot << 8) | (f.vec << 2) | f.comp), &coeff) || !isfinite(coeff))) why = 3;   // ubo_slot < 0: this shader has no coefficient
    for (int i = 0; !why && i < fs->rf.nb; i++) { nvn_rbind *b = &fs->rf.b[i];
        if (b->kind == NVN_RB_IMG) { why = 4; break; }
        if (b->kind != NVN_RB_TEX && b->kind != NVN_RB_TBUF) continue;
        int tid = b->reg < 128 ? (int)(S.tex[1][b->reg] & 0xfffff) : 0; if (!zskip_view_zero(tid)) why = 5; }
    if (!why && (!R.has_rt || !R.rt.n)) why = 6;
    for (int i = 0; !why && i < R.rt.n && i < 8; i++) { if (!R.rt.color[i]) continue;
        int pending = g_enc ? g_zskip_cand_ok[i] : zskip_zero_clear(i);   // Render pass already open: look at the candidates noted when the render pass was opened (not yet drawn to); not open: look at the pending clears
        if (!pending && (g_enc || R.clear_c[i] || !zskip_target_zero(R.rt.color[i], &R.rt.cview[i], R.rt.has_cview[i]))) why = 7; }
    if (why) PD("N %llu 零源不跳过 reason=%d 集合 %d", g_stat_pass + 1, why, g_zskip_n);
    return !why;
}
static int uv_proof(NvShader *vs, int locn, const p_draw *dw, const p_draw_idx *di) {   // The interpolated output locn at every pixel center is exactly this pixel's texel center: u = (x + 1) / 2, v = (1 - y) / 2, w = 1, viewport and scissor cover the whole block
    if (locn < 0 || locn >= 32 || vs->rf.pos_attr < 0) return 0;
    int direct = vs->rf.uv_attr[locn] >= 0, aff = vs->rf.uv_aff_attr[locn] >= 0;
    if (!direct && !aff) return 0;
    float sc[4] = {1, 0, 1, 0};   // Pass-through as-is = scale 1, offset 0
    if (!direct) for (int q = 0; q < 4; q++) if (!ubo_float(0, vs->rf.uv_aff[locn][q], &sc[q])) return 0;
    NSUInteger W = g_pass.tw, H = g_pass.th;
    if (S.has_vp && !(S.vp[0][0] == 0 && S.vp[0][1] == 0 && S.vp[0][2] == (float)W && S.vp[0][3] == (float)H)) return 0;
    if (R.has_sc && !(R.sc.x <= 0 && R.sc.y <= 0 && R.sc.x + R.sc.w >= (int)W && R.sc.y + R.sc.h >= (int)H)) return 0;
    float P[64][4], Q[64][4]; int n = dw ? dw->count : di->count;
    if (attr_read(vs->rf.pos_attr, dw, di, P) || attr_read(direct ? vs->rf.uv_attr[locn] : vs->rf.uv_aff_attr[locn], dw, di, Q)) return 0;
    for (int k = 0; k < n; k++) { float w = vs->rf.pos_w1 ? 1.0f : P[k][3];
        float u = direct ? Q[k][0] : fmaf(Q[k][0], sc[0], sc[1]), v = direct ? Q[k][1] : fmaf(Q[k][1], sc[2], sc[3]);   // The same fma as the shader
        if (w != 1.0f || u != (P[k][0] + 1.0f) * 0.5f || v != (1.0f - P[k][1]) * 0.5f) { PD("F %llu uv 证明：顶点 %d 位置 %g,%g uv %g,%g（缩放 / 偏移 %g,%g,%g,%g）", g_stat_pass, k, P[k][0], P[k][1], u, v, sc[0], sc[1], sc[2], sc[3]); return 0; } }
    return 1;
}
// One draw of the later pass: verify and produce the specialization. Returns 0 = cannot be proven (the caller ends the merged render pass and reopens with the original targets)
static int fz_p2_draw(NvProg *pr, const p_draw *dw, const p_draw_idx *di, uint8_t lmap[8], __strong id<MTLFunction> *fn) {
    NvShader *vs = pr->sh[0], *fs = pr->sh[1]; *fn = nil;
    if (!vs || !fs) { PD("F %llu 证不出：无顶点 / 片元着色器", g_stat_pass); return 0; }
    if (!g_fz.rt2.depth && R.rt.depth && (S.ds.dt || S.ds.dw || S.ds.st)) { PD("F %llu 证不出：深度来自前一个通道而本次绘制开了深度模板测试", g_stat_pass); return 0; }
    int fx[32]; for (int i = 0; i < 32; i++) fx[i] = -1; uint8_t fbmask = 0;
    for (int st = 0; st < 5; st++) { __unsafe_unretained NvShader *sh = pr->sh[st]; if (!sh || sh->stage >= 6) continue;
        for (int i = 0; i < sh->rf.nb; i++) { const nvn_rbind *b = &sh->rf.b[i]; uint64_t h;
            if (b->kind == NVN_RB_IMG) h = b->reg < 32 ? S.img[sh->stage][b->reg] : 0; else if (b->kind == NVN_RB_TEX || b->kind == NVN_RB_TBUF) h = b->reg < 128 ? S.tex[sh->stage][b->reg] : 0; else continue;
            uint64_t ser = h ? h_ser(h) : 0; if (!ser) continue;
            int j = -1; for (int m = 0; m < g_fz.n1; m++) if (R.rt.color[m] && R.rt.color[m]->serial == ser) { j = m; break; }
            if (j < 0) continue;   // What is sampled is not an attachment of the earlier pass
            if (sh != fs || b->kind != NVN_RB_TEX) { PD("F %llu 证不出：非片元阶段或非贴图槽读前一个通道的附件", g_stat_pass); return 0; }
            if (fs->rf.fc_mixed[b->reg >> 6] >> (b->reg & 63) & 1) { PD("F %llu 证不出：槽 %d 混用", g_stat_pass, b->reg); return 0; }
            uint64_t s_; int lv, sl; rt_sub(&R.rt, j, &s_, &lv, &sl);
            if (lv || sl || MAX(R.rt.color[j]->d.width, 1u) != g_pass.tw || MAX(R.rt.color[j]->d.height, 1u) != g_pass.th) { PD("F %llu 证不出：附件不是第 0 级第 0 层或尺寸不符", g_stat_pass); return 0; }
            {   // The sampled view must also be exactly the attachment itself (previously only the attachment being level 0 layer 0 was checked, not the view bound by the shader -- when the same texture is sampled through a view of level k / layer k / a changed format,
                // the proof would still hold, the sampling would be changed to read on-tile level 0 / the original format, and the picture would be wrong). The view's starting level / layer must be 0 and its format must equal the attachment's actual format, otherwise do not merge
                int tid_ = (int)(h & 0xfffff); void *tv_ = NULL; if (tid_) tex_fast_rid(tid_, &tv_); __unsafe_unretained id<MTLTexture> tv = (__bridge id<MTLTexture>)tv_;
                uint32_t apf_ = tv.parentTexture ? (uint32_t)tv.parentTexture.pixelFormat : (uint32_t)tv.pixelFormat; uint8_t swz_[4];
                if (R.rt.has_cview[j] && R.rt.cview[j].set_format && !(nvn_fmt_to_mtl(R.rt.cview[j].format, &apf_, swz_) && apf_)) apf_ = 0;   // The attachment is attached through a format-changing view: take that format; if unrecognized, do not merge
                MTLPixelFormat apf = (MTLPixelFormat)apf_;
                if (!tv || (tv.parentTexture && (tv.parentRelativeLevel || tv.parentRelativeSlice)) || tv.pixelFormat != apf) {
                    static int lg; if (lg++ < 8) LOG("[nvn-mtl] 片上读合并：槽 %d 采样视图（起始级 %lu 层 %lu 格式 %lu）与附件（第 0 级第 0 层 格式 %lu）不符，不合并（帧 %llu 通道 %llu）", b->reg,
                        tv && tv.parentTexture ? (unsigned long)tv.parentRelativeLevel : 0ul, tv && tv.parentTexture ? (unsigned long)tv.parentRelativeSlice : 0ul, (unsigned long)tv.pixelFormat, (unsigned long)apf,
                        (unsigned long long)atomic_load(&g_cur_frame), g_stat_pass);
                    PD("F %llu 证不出：槽 %d 采样视图不是附件那一张（级 / 层 / 格式不符）", g_stat_pass, b->reg); return 0; } }
            int any = 0;
            for (int c = 0; c < fs->rf.nfc; c++) if (fs->rf.fc_reg[c] == b->reg) {
                if (!uv_proof(vs, fs->rf.fc_locn[c], dw, di)) { PD("F %llu 证不出：槽 %d 的采样坐标不是本像素（插值输出 %d）", g_stat_pass, b->reg, fs->rf.fc_locn[c]); return 0; }
                fx[c] = j; fbmask |= (uint8_t)(1u << j); any = 1; }
            if (!any) { PD("F %llu 证不出：槽 %d 的采样不在可改表里", g_stat_pass, b->reg); return 0; } } }
    memset(lmap, 0xff, 8);
    for (int c = 0; c < 8; c++) if (fs->rf.color_mask >> c & 1) { if (g_fz.p2map[c] < 0) { PD("F %llu 证不出：输出颜色 %d 无对应附件", g_stat_pass, c); return 0; } lmap[g_fz.p2map[c]] = (uint8_t)c; }
    int ident = !fbmask; for (int c = 0; c < 8; c++) if ((fs->rf.color_mask >> c & 1) && g_fz.p2map[c] != c) ident = 0;
    if (ident) { for (int q = 0; q < 8; q++) if (lmap[q] != 0xff) g_fz.wrote[q] = 1; if (fz_check_on()) fz_snap_take(dw, di); return 1; }   // Does not read the tile and the output is not shifted: use the original version (the write mask still follows the mapping)
    if (!fs->rf.nfo) { bf_want_note(fs->key); PD("F %llu 证不出：没有可读 tile 版本（输出须移位）", g_stat_pass); return 0; }
    *fn = bf_fn(fs, fx, fbmask, lmap);
    if (!*fn) return 0;
    memset(&g_fz_desc, 0, sizeof g_fz_desc); g_fz_desc.kind = 1; g_fz_desc.fbmask = fbmask; for (int i = 0; i < 32; i++) g_fz_desc.fx[i] = (int8_t)fx[i];
    for (int q = 0; q < 8; q++) if (lmap[q] != 0xff) g_fz.wrote[q] = 1;
    if (fz_check_on()) fz_snap_take(dw, di);
    g_fz_draws++; return 1;
}
static void bloomq_reset(void) { memset(&g_bloomq, 0, sizeof g_bloomq); }
static int bloomq_enabled(void) {
    if (g_bloom_q < 0) { const char *e = getenv("DM_NVN_BLOOM_Q"); g_bloom_q = !e || strcmp(e, "0"); }
    return g_bloom_q;
}
static int bloomq_target(uint64_t *ser, int *lv, int *sl) {
    if (!R.has_rt || R.rt.n != 1 || !R.rt.color[0]) return 0;
    rt_sub(&R.rt, 0, ser, lv, sl); return *ser != 0;
}
static int bloomq_make_pipeline(MTLPixelFormat fmt) {
    if (g_bloomq_pso && g_bloomq_fmt == fmt) return 1;
    static const char *src =
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "struct V { float4 p [[position]]; float2 uv; };\n"
        "vertex V vs(uint id [[vertex_id]]) { float2 uv=float2((id<<1)&2,id&2); V o; o.p=float4(uv*float2(2,-2)+float2(-1,1),0,1); o.uv=uv; return o; }\n"
        "struct P { float threshold; float coeff_g; float coeff_b; float dm; float sh; };\n"
        "fragment float4 fs(V in [[stage_in]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]], constant P& p [[buffer(0)]]) { float2 d=float2(1.0/p.dm,1.0/p.sh); float3 c=0; for (int y=-1;y<=1;y+=2) for (int x=-1;x<=1;x+=2) { float3 v=t.sample(s,in.uv+float2(x,y)*d).rgb; float l=v.r*0.2126+v.g*p.coeff_g+v.b*p.coeff_b; if(l>=p.threshold) c+=v/(1.0+l); } return float4(c*0.25,1); }\n";
    NSError *e = nil; id<MTLLibrary> lib = [g_dev newLibraryWithSource:@(src) options:nil error:&e];
    if (!lib) { MLOG("【红】泛光 1/4 着色器编译失败：%s", e.localizedDescription.UTF8String); return 0; }
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new]; pd.vertexFunction = [lib newFunctionWithName:@"vs"]; pd.fragmentFunction = [lib newFunctionWithName:@"fs"]; pd.colorAttachments[0].pixelFormat = fmt;
    g_bloomq_pso = [g_dev newRenderPipelineStateWithDescriptor:pd error:&e];
    if (!g_bloomq_pso) { MLOG("【红】泛光 1/4 管线失败：%s", e.localizedDescription.UTF8String); return 0; }
    g_bloomq_fmt = fmt;
    MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new]; sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterLinear; sd.sAddressMode = sd.tAddressMode = MTLSamplerAddressModeClampToEdge; sd.supportArgumentBuffers = YES;
    g_bloomq_samp = [g_dev newSamplerStateWithDescriptor:sd]; return g_bloomq_samp != nil;
}
static int bloomq_capture(void) {
    uint64_t ser; int lv, sl; if (!bloomq_target(&ser, &lv, &sl) || !g_bloomq.s_tex) return 0;
    id<MTLTexture> base = tex_mtl(R.rt.color[0]), src = tex_mtl(g_bloomq.s_tex); if (!base || !src) return 0;
    id<MTLTexture> dst = base; if (lv || sl) { static id<MTLTexture> vb; static int vl, vs; if (vb != base || vl != lv || vs != sl) { g_bloomq_view = [base newTextureViewWithPixelFormat:base.pixelFormat textureType:base.textureType levels:NSMakeRange(lv, 1) slices:NSMakeRange(sl, 1)]; if (g_bloomq_view) res_add(g_bloomq_view); vb = base; vl = lv; vs = sl; } dst = g_bloomq_view; }
    if (!dst || !bloomq_make_pipeline(dst.pixelFormat)) return 0;
    rp_end();
    MTL4RenderPassDescriptor *rp = [MTL4RenderPassDescriptor new]; rp.colorAttachments[0].texture = dst; rp.colorAttachments[0].loadAction = MTLLoadActionDontCare; rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTL4RenderCommandEncoder> en = [g_cb renderCommandEncoderWithDescriptor:rp]; if (!en) return 0;
    [en barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice]; [en setArgumentTable:g_bloomq_args atStages:MTLRenderStageFragment];
    [g_bloomq_args setTexture:src.gpuResourceID atIndex:0]; [g_bloomq_args setSamplerState:g_bloomq_samp.gpuResourceID atIndex:0];
    float *p = g_bloomq_const[g_bloomq_i].contents; p[0] = g_bloomq.threshold; p[1] = g_bloomq.coeff_g; p[2] = g_bloomq.coeff_b; p[3] = src.width; p[4] = src.height; [g_bloomq_args setAddress:g_bloomq_const[g_bloomq_i].gpuAddress atIndex:0];
    [en setViewport:(MTLViewport){0,0,dst.width,dst.height,0,1}]; [en setRenderPipelineState:g_bloomq_pso]; [en drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3]; m4_end(en);
    g_bloomq_i = (g_bloomq_i + 1) % 3; args_dirty_all(); g_pso_dirty = g_dss_dirty = 1; return 1;
}
// ---------- Drag-follow image for shop items: items on a vendor's shelf have no "picked up" state, so during a drag the replay layer takes "the draw that draws this item's icon", translates it to the finger at the end of the same pass and draws it once more ----------
// In the UI pass each rectangle is one 4-vertex draw (20-byte vertices: position 3 floats + texture coordinates 2 floats; the position goes to clip coordinates through the matrix in vertex constant 0).
// The front end (front_ios.m) gives the point where the finger pressed, the shelf grid and the center of the drag-follow image (normalized, top-left origin); here, in the UI pass, find the rectangle that "covers this point, has all four edges on grid lines, and has the smallest area" = that item
// (identifying by the point alone gets it wrong: about 0.1 seconds after the press the game draws a one-cell-sized rectangle centered on the press point, not on grid lines -- on a real device it was twice taken for the item)
// (does not depend on the game's selected state: selection has to wait about 0.1 seconds for the controller cursor to land on the item, and a quick drag that starts on press cannot wait for that). The same rectangle has two draws, cell background and icon:
// the cell background uses a common texture (many rectangles in this pass use it), while each item has its own icon texture -- take the draw with the texture "used the fewest times in this pass". The recognized rectangle is reported back to the front end (used to cover the original shelf position with a dark color).
// Vertical position and height of the UI rectangle in the real picture (both as a fraction of the picture height: y = 0 touches the top, h = 1 is the whole picture); registered by the front end and read fresh on every draw (changes when switching between docked / handheld mode).
// The game draws the UI into the upper h portion of the UI texture (the viewport is the whole texture, and the projection matrix carries the scaling).
static void (*g_ui_rect_fn)(double *y, double *h);
void mtl_set_ui_rect_fn(void (*fn)(double *y, double *h)) { g_ui_rect_fn = fn; }
// Safe-area inset for the HUD: which draws are side-anchored HUD elements is decided by WHERE each draw lands, not by its shader
// or its vertex count (measured: this game's HUD shares its fragment shader with a large class of other screen-space draws, so a
// shader-key test identifies the class but not the HUD). Each draw's position attribute is read, pushed through that draw's own
// projection matrix, and the NDC bounds decide: a draw touching one edge of the picture without spanning the middle band is a
// side-anchored element and is shifted inward; anything spanning the middle (damage vignette, transition curtain, dimming,
// full-screen map, world) is left alone. Same approach the archived tree used (archive/main-78d5dd6 dm_uiscreen.c + ui_draw_ndc).
#define UI_SAFE_SPAN_EPS 0.004f     // "touches the edge": NDC tolerance (~3 px at 1568 wide)
#define UI_SAFE_MID_EPS  0.10f      // reaching within this distance of the centre in NDC counts as spanning
#define UI_SAFE_MAX_VERTS 512       // read at most this many vertices per draw (HUD elements are small; a big batch is not one)
#define UI_SAFE_RANGE 1.35f         // a vertex attribute whose transformed x / y leaves this box is not position data (screen-space
                                    // art lies inside the clip volume; the small slack allows a widget sliding in at the edge)
static double g_ui_xsafe_l, g_ui_xsafe_r;   // insets in normalized device x (1.0 = half the picture width), from the front end
static int g_ui_xsafe_auto, g_ui_xsafe_log; // left inset follows the canvas / log every candidate (diagnostics)
static float g_ui_safe_mid = UI_SAFE_MID_EPS;   // DM_UI_SAFEMID overrides it
static const char *g_ui_safe_only;              // DM_UI_SAFEONLY=<vs-key>: shift only draws of this vertex shader (experiment)
void mtl_set_ui_safe(double left, double right) { g_ui_xsafe_l = left > 0 && left < 1 ? left : 0; g_ui_xsafe_r = right > 0 && right < 1 ? right : 0; }
// DM_UI_SAFEL / DM_UI_SAFER (clip units) set the same insets without a front end; DM_UI_SAFEAUTO=1 derives the left inset from the
// canvas ("put the UI box where a centred 16:9 picture would have it"), which is the case a full-panel phone actually needs.
static void ui_safe_env(void) {
    static int read; if (read) return; read = 1;
    const char *e = getenv("DM_UI_SAFEL"), *f = getenv("DM_UI_SAFER"), *a = getenv("DM_UI_SAFEAUTO"), *g = getenv("DM_UI_SAFELOG");
    if (e && *e) g_ui_xsafe_l = atof(e); if (f && *f) g_ui_xsafe_r = atof(f);
    g_ui_xsafe_auto = a && *a && *a != '0'; g_ui_xsafe_log = g && *g && *g != '0';
    { const char *m = getenv("DM_UI_SAFEMID"); if (m && *m) g_ui_safe_mid = (float)atof(m); }
    { const char *o = getenv("DM_UI_SAFEONLY"); if (o && *o) { g_ui_safe_only = o; MLOG("安全区内缩：只作用于顶点着色器 %s", o); } }
    if (g_ui_xsafe_l || g_ui_xsafe_r || g_ui_xsafe_auto || g_ui_xsafe_log)
        MLOG("安全区内缩：左 %.4f 右 %.4f%s%s", g_ui_xsafe_l, g_ui_xsafe_r, g_ui_xsafe_auto ? "，左值按画布自动算" : "", g_ui_xsafe_log ? "，记录候选绘制" : "");
}
// NDC bounds of one draw: 1 = measured, <0 = could not (position attribute unreadable, matrix not a plain affine one, too many verts).
// *zflat reports whether every vertex has the same NDC z, which is what a 2D screen-space quad looks like and a 3D one does not.
static int g_ui_ndc_attr, g_ui_ndc_blk;   // which attribute / which 4x4 block the bounds were measured from (diagnostics)
static int ui_ndc_attr(int a, const p_draw *dw, const p_draw_idx *di, const float m[16], float *x0, float *x1, float *y0, float *y1, int *zflat) {
    int n = dw ? dw->count : (di ? di->count : 0);
    if (n < 1 || n > UI_SAFE_MAX_VERTS) return -4;
    float P[64][4];
    int chunk = n > 64 ? 64 : n;
    float a0 = 9, a1 = -9, b0 = 9, b1 = -9, c0 = 9, c1 = -9;
    for (int base = 0; base < n; base += chunk) {
        p_draw dwc;
        const p_draw *pw = dw; const p_draw_idx *pi = di;
        if (n > 64) { if (!dw) return -6; dwc = *dw; dwc.count = chunk; dwc.first = dw->first + base; pw = &dwc; }
        if (attr_read(a, pw, pi, P)) return -5;
        for (int k = 0; k < chunk; k++) {
            float w = m[3] * P[k][0] + m[7] * P[k][1] + m[15]; if (w == 0) continue;
            float nx = (m[0] * P[k][0] + m[4] * P[k][1] + m[12]) / w, ny = (m[1] * P[k][0] + m[5] * P[k][1] + m[13]) / w;
            float nz = (m[2] * P[k][0] + m[6] * P[k][1] + m[14]) / w;
            if (!(nx > -UI_SAFE_RANGE && nx < UI_SAFE_RANGE && ny > -UI_SAFE_RANGE && ny < UI_SAFE_RANGE)) return -7;   // not position data
            if (nx < a0) a0 = nx; if (nx > a1) a1 = nx; if (ny < b0) b0 = ny; if (ny > b1) b1 = ny;
            if (nz < c0) c0 = nz; if (nz > c1) c1 = nz;
        }
    }
    if (a1 < -8) return -7;
    if (a1 - a0 < 0.002f && b1 - b0 < 0.002f) return -9;   // a point: this attribute is not position data under this block
    *x0 = a0; *x1 = a1; *y0 = b0; *y1 = b1;
    if (zflat) *zflat = c1 - c0 < 1e-4f;
    return 1;
}
static int ui_draw_ndc(const p_draw *dw, const p_draw_idx *di, float *x0, float *x1, float *y0, float *y1, int *zflat) {
    __unsafe_unretained NvProg *pr = S.prog ? prog_get(S.prog) : nil; NvShader *vs = pr ? pr->sh[0] : nil;
    if (!vs) return -1;
    const p_bind_buf *u = &S.ubo[0][0]; void *mb; uint64_t mo;
    if (!u->addr || u->size < 64 || u->size > 4096 || pend_hits(u->addr, 64) || !mtl_addr(u->addr, u->size, &mb, &mo)) return -2;
    const uint8_t *base = [(__bridge id<MTLBuffer>)mb contents];
    if (!base) return -2;
    // Shader reflection names the position attribute only when the vertex stage passes it through untouched; this game's UI
    // shaders transform it with two 4x4 blocks in cb0 (picked per vertex by the sign of attribute 1's z), so the reflection field
    // is empty here (-1) and the block that applies is not known up front. Probe the draw's attributes under both blocks and take
    // the first combination whose vertices land inside the clip volume and are flat in depth: that is screen-space art.
    int rc = -8;
    for (int a = (vs->rf.pos_attr >= 0 ? vs->rf.pos_attr : 0), hi = (vs->rf.pos_attr >= 0 ? vs->rf.pos_attr + 1 : S.nattr < 8 ? S.nattr : 8); a < hi; a++) {
        for (int bo = 0; bo < 256; bo += 64) {
            if (bo + 64 > (int)u->size) break;
            float m[16]; memcpy(m, base + mo + bo, 64);
            if (m[3] != 0 || m[7] != 0 || m[11] != 0 || m[15] == 0) { if (rc < -3) rc = -3; continue; }   // not an affine transform
            int r = ui_ndc_attr(a, dw, di, m, x0, x1, y0, y1, zflat);
            if (r == 1) { g_ui_ndc_attr = a; g_ui_ndc_blk = bo / 64; return 1; }
            if (r > rc) rc = r;
        }
    }
    return rc;
}
// +1 = move right (left-side HUD), -1 = move left (right-side HUD), 0 = leave alone.
// The left test is "lies wholly in the left half", not "touches the left edge": this game anchors its HUD to the UI box's
// top-left corner, so the left column starts a dozen pixels inside the picture and the edge test would catch only the bars.
static int ui_safe_dir(float x0, float x1) {
    if (x1 < -g_ui_safe_mid) return +1;   // entirely left of the middle band -> a left-anchored element
    if (x0 > g_ui_safe_mid) return -1;    // entirely right of it -> a right-anchored element
    return 0;                             // reaches the middle: full-width art (dimming, curtain, vignette, map)
}
static double ui_hfrac(void) { double y = 0, h = 1; if (g_ui_rect_fn) g_ui_rect_fn(&y, &h); return h > 0 && h < 1 ? h : 1; }
static _Atomic int g_ghost_on; static _Atomic uint64_t g_ghost_pt, g_ghost_at, g_ghost_found, g_ghost_grid;
void mtl_ui_ghost(int on, float px, float py, const float grid[4], float cx, float cy) {
    if (on) { uint64_t gv = 0; for (int i = 0; i < 4; i++) gv |= (uint64_t)(uint16_t)(fminf(fmaxf(grid[i], 0), 1) * 65535 + .5f) << (16 * i); atomic_store(&g_ghost_grid, gv);
              atomic_store(&g_ghost_pt, (uint64_t)(uint32_t)lroundf(fminf(fmaxf(px, 0), 1) * 65535) | (uint64_t)(uint32_t)lroundf(fminf(fmaxf(py, 0), 1) * 65535) << 32);
              atomic_store(&g_ghost_at, (uint64_t)(uint32_t)(int32_t)lroundf(cx * 65535) | (uint64_t)(uint32_t)(int32_t)lroundf(cy * 65535) << 32); }
    else atomic_store(&g_ghost_found, 0);
    atomic_store(&g_ghost_on, on);
}
int mtl_ui_ghost_rect(float r[4]) { uint64_t v = atomic_load(&g_ghost_found); if (!v) return 0; for (int i = 0; i < 4; i++) r[i] = (float)((v >> (16 * i)) & 0xffff) / 65535.f; return 1; }
static struct { uint64_t pass; int n, nt, busy; float area, r[4]; struct { void *S_; p_draw dw; uint64_t tex; } c[4]; uint64_t tid[128]; uint16_t tn[128]; } g_gh;
static uint64_t ghost_tex(void) {   // The texture actually sampled by this draw's fragment shader (slot taken from shader reflection; the other slots in S.tex are left over from earlier draws)
    __unsafe_unretained NvProg *pr = S.prog ? prog_get(S.prog) : nil; NvShader *f = pr ? pr->sh[1] : nil; if (!f) return 0;
    for (int i = 0; i < f->rf.nb; i++) if ((f->rf.b[i].kind == NVN_RB_TEX || f->rf.b[i].kind == NVN_RB_SAMP) && f->rf.b[i].reg < 128 && S.tex[1][f->rf.b[i].reg]) return S.tex[1][f->rf.b[i].reg];
    return 0; }
// Whether the current UI draw is "untextured and exactly covering the whole UI rectangle" (the UI rectangle occupies the upper hf of the texture)
static int ui_cover_all(double hf) {
    if (S.vbuf[0].size != 80 || ghost_tex()) return 0;
    const p_bind_buf *u = &S.ubo[0][0]; void *vb, *mb; uint64_t vo, mo; const uint8_t *vp_, *mp_;
    if (!u->addr || u->size < 64 || pend_hits(u->addr, 64) || pend_hits(S.vbuf[0].address, 80)) return 0;
    if (!mtl_addr(S.vbuf[0].address, 80, &vb, &vo) || !(vp_ = [(__bridge id<MTLBuffer>)vb contents]) || !mtl_addr(u->addr, 64, &mb, &mo) || !(mp_ = [(__bridge id<MTLBuffer>)mb contents])) return 0;
    float m[16], x0 = 9, y0 = 9, x1 = -9, y1 = -9; memcpy(m, mp_ + mo, 64);
    for (int i = 0; i < 4; i++) { float f[2]; memcpy(f, vp_ + vo + 20 * i, 8); float nx = (m[0] * f[0] + m[4] * f[1] + m[12] + 1) / 2, ny = (m[1] * f[0] + m[5] * f[1] + m[13] + 1) / 2;
        x0 = fminf(x0, nx); x1 = fmaxf(x1, nx); y0 = fminf(y0, ny); y1 = fmaxf(y1, ny); }
    return x0 < 0.002f && x1 > 0.998f && y0 < 0.002f && fabsf(y1 - (float)hf) < 0.002f;
}
static void ghost_note(const p_draw *dw) {
    if (g_gh.busy || dw->count != 4 || S.nattr != 2 || S.stream[0].stride != 20 || S.vbuf[0].size != 80) return;
    if (g_gh.pass != g_stat_pass) { for (int i = 0; i < g_gh.n; i++) free(g_gh.c[i].S_); g_gh.n = g_gh.nt = 0; g_gh.pass = g_stat_pass; }
    uint64_t t = ghost_tex(); int k = 0; for (; k < g_gh.nt; k++) if (g_gh.tid[k] == t) break;
    if (k == g_gh.nt && g_gh.nt < 128) { g_gh.tid[k] = t; g_gh.tn[k] = 0; g_gh.nt++; } if (k < g_gh.nt) g_gh.tn[k]++;
    const p_bind_buf *u = &S.ubo[0][0]; void *vb, *mb; uint64_t vo, mo; const uint8_t *vp_, *mp_;
    if (!u->addr || u->size < 64 || pend_hits(u->addr, 64) || pend_hits(S.vbuf[0].address, 80)) return;
    if (!mtl_addr(S.vbuf[0].address, 80, &vb, &vo) || !(vp_ = [(__bridge id<MTLBuffer>)vb contents]) || !mtl_addr(u->addr, 64, &mb, &mo) || !(mp_ = [(__bridge id<MTLBuffer>)mb contents])) return;
    float m[16], x0 = 9, y0 = 9, x1 = -9, y1 = -9; memcpy(m, mp_ + mo, 64);
    for (int i = 0; i < 4; i++) { float f[2]; memcpy(f, vp_ + vo + 20 * i, 8); float nx = (m[0] * f[0] + m[4] * f[1] + m[12] + 1) / 2, ny = (m[1] * f[0] + m[5] * f[1] + m[13] + 1) / 2;
        x0 = fminf(x0, nx); x1 = fmaxf(x1, nx); y0 = fminf(y0, ny); y1 = fmaxf(y1, ny); }
    { float hf = (float)ui_hfrac(); y0 /= hf; y1 /= hf; }   // The points given by the front end are normalized coordinates within the UI rectangle; when the UI occupies only the upper hf portion of the texture, convert to the same convention (unchanged when hf = 1)
    uint64_t pv = atomic_load(&g_ghost_pt); float qx = (float)(uint32_t)pv / 65535.f, qy = (float)(pv >> 32) / 65535.f;
    if (qx < x0 || qx > x1 || qy < y0 || qy > y1) return;
    uint64_t gv = atomic_load(&g_ghost_grid); float gx = (float)(gv & 0xffff) / 65535.f, gy = (float)((gv >> 16) & 0xffff) / 65535.f, gw = (float)((gv >> 32) & 0xffff) / 65535.f, gh = (float)(gv >> 48) / 65535.f;
    if (gw <= 0 || gh <= 0) return;
    float w = x1 - x0, h = y1 - y0, area = w * h, c0 = (x0 - gx) / gw, r0 = (y0 - gy) / gh, cn = w / gw, rn = h / gh;   // Which cell the top-left corner is in, and how many cells it occupies
    if (fabsf(c0 - roundf(c0)) > 0.15f || fabsf(r0 - roundf(r0)) > 0.15f || fabsf(cn - roundf(cn)) > 0.15f || fabsf(rn - roundf(rn)) > 0.15f || cn < 0.85f || rn < 0.85f || cn > 2.15f || rn > 4.15f) return;   // One item is at most 2x4 cells
    if (g_gh.n && area > g_gh.area * 1.03f) return;                       // A smaller one already exists
    if (g_gh.n && area < g_gh.area * 0.97f) { for (int i = 0; i < g_gh.n; i++) free(g_gh.c[i].S_); g_gh.n = 0; }   // Found a smaller one: the earlier ones are invalidated
    if (g_gh.n == 4) { free(g_gh.c[0].S_); memmove(&g_gh.c[0], &g_gh.c[1], 3 * sizeof g_gh.c[0]); g_gh.n = 3; }
    g_gh.area = area; g_gh.r[0] = x0; g_gh.r[1] = y0; g_gh.r[2] = x1; g_gh.r[3] = y1;
    typeof(g_gh.c[0]) *c = &g_gh.c[g_gh.n++]; c->S_ = malloc(sizeof S); memcpy(c->S_, &S, sizeof S); c->dw = *dw; c->tex = t;
}
static void ghost_flush(void) {   // Before the render pass ends: translate the recognized icon draw to the finger and draw it once more (moves the viewport, does not change vertices; the scissor is opened up)
    if (g_gh.busy || !g_gh.n) return;
    if (g_gh.pass != g_stat_pass || !atomic_load(&g_ghost_on)) { for (int i = 0; i < g_gh.n; i++) free(g_gh.c[i].S_); g_gh.n = 0; return; }
    int best = -1, bn = 1 << 30; for (int i = 0; i < g_gh.n; i++) { int cnt = 0; for (int k = 0; k < g_gh.nt; k++) if (g_gh.tid[k] == g_gh.c[i].tex) cnt = g_gh.tn[k]; if (cnt < bn) { bn = cnt; best = i; } }
    typeof(g_gh.c[0]) *c = &g_gh.c[best]; uint64_t at = atomic_load(&g_ghost_at); float ax = (float)(int32_t)(uint32_t)at / 65535.f, ay = (float)(int32_t)(uint32_t)(at >> 32) / 65535.f;
    float W = (float)g_pass.tw, H = (float)g_pass.th, ccx = (g_gh.r[0] + g_gh.r[2]) / 2, ccy = (g_gh.r[1] + g_gh.r[3]) / 2;
    { uint64_t v = 0; for (int i = 0; i < 4; i++) v |= (uint64_t)(uint16_t)(fminf(fmaxf(g_gh.r[i], 0), 1) * 65535 + .5f) << (16 * i); atomic_store(&g_ghost_found, v ? v : 1); }
    g_gh.busy = 1; typeof(S) save; memcpy(&save, &S, sizeof S); p_scissor sc = R.sc; int has_sc = R.has_sc;
    memcpy(&S, c->S_, sizeof S); if (!S.has_vp) { S.vp[0][0] = S.vp[0][1] = 0; S.vp[0][2] = W; S.vp[0][3] = H; S.has_vp = 1; }
    S.vp[0][0] += (ax - ccx) * W; S.vp[0][1] += (ay - ccy) * H * (float)ui_hfrac(); R.has_sc = 0;
    args_dirty_all(); g_pso_dirty = g_dss_dirty = 1; draw(&c->dw, NULL);
    { static int once; if (!once++) MLOG("拖货跟手图：首例 通道 %llu，候选 %d 个，取贴图本趟用到 %d 次的那次", g_stat_pass, g_gh.n, bn); }
    memcpy(&S, &save, sizeof S); args_dirty_all(); R.sc = sc; R.has_sc = has_sc; g_pso_dirty = g_dss_dirty = 1; g_gh.busy = 0;
    for (int i = 0; i < g_gh.n; i++) free(g_gh.c[i].S_); g_gh.n = 0;
}
// Diagnostic DM_NVN_KEYDUMP=<fragment shader key prefix>[:<count, default 6>[:<frame interval>]]: watch only this class of draws (for real devices; no frame number to guess). The first N matching draws force PASSDUMP's per-binding lines on (K / B / d:
// textures with format / view / channel swizzle, samplers with LOD clamps), plus [kd] lines: vertex attribute formats, each vertex stream (first 4 vertices of per-vertex streams, all instances of per-instance streams) and every value of each uniform buffer of both stages.
// Run the same switch once on the Mac and once on the device and compare item by item to see which input differs on the device (Super Mario 3D World + Bowser's Fury's magenta squares in 1-1: particle 5e4de7561c670fee, iPhone only, not reproducible on the Mac)
static void kd_floats(const char *what, int reg, uint64_t addr, uint64_t size, uint64_t cap) {
    void *mb; uint64_t mo; const uint8_t *base;
    if (!addr || !size) { fprintf(stderr, "[kd]   %s r%d 未绑\n", what, reg); return; }
    if (!mtl_addr(addr, 1, &mb, &mo) || !(base = [(__bridge id<MTLBuffer>)mb contents])) { fprintf(stderr, "[kd]   %s r%d addr=%#llx 不在内存池\n", what, reg, (unsigned long long)addr); return; }
    uint64_t n = MIN(size, cap) / 4; const float *v = (const float *)(base + mo);
    fprintf(stderr, "[kd]   %s r%d addr=%#llx size=%llu%s\n", what, reg, (unsigned long long)addr, (unsigned long long)size, pend_hits(addr, size) ? "（本提交里还有待执行的写）" : "");
    for (uint64_t i = 0; i < n; i += 16) {
        char line[1024]; size_t o = 0;
        for (uint64_t k = i; k < n && k < i + 16 && o < sizeof line - 24; k++) o += snprintf(line + o, sizeof line - o, " %.6g", v[k]);
        fprintf(stderr, "[kd]     +%#06llx%s\n", (unsigned long long)(i * 4), line);
    }
}
static int kd_draw(NvProg *pr, const p_draw *dw, const p_draw_idx *di) {
    static char key[24]; static int left = -1, every; static uint64_t last = ~0ull;
    if (left < 0) { left = 0; const char *e = getenv("DM_NVN_KEYDUMP");
        if (e && *e) { size_t n = strcspn(e, ":"); if (n > 23) n = 23; memcpy(key, e, n); key[n] = 0; const char *c = strchr(e, ':'); left = c ? atoi(c + 1) : 6; every = c && strchr(c + 1, ':') ? atoi(strchr(c + 1, ':') + 1) : 0;
            MLOG("DM_NVN_KEYDUMP：片元着色器 %s 的 %d 次绘制逐项记录%s（[kd] / [pass] 行）", key, left, every ? "，每隔这么多帧最多记一次" : ""); } }
    if (left <= 0 || !pr->sh[1] || strncmp(pr->sh[1]->key, key, strlen(key))) return 0;
    uint64_t fr = atomic_load(&g_cur_frame);
    if (every > 0 && last != ~0ull && fr < last + (uint64_t)every) return 0;   // at most once every `every` frames: samples spread over the session, no need to guess the frame of the scene
    last = fr; left--;
    int count = dw ? dw->count : di->count, inst = dw ? dw->instances : di->instances, first = dw ? dw->first : 0, binst = dw ? dw->base_instance : di->base_instance;
    fprintf(stderr, "[kd] F%llu 通道 %llu vs=%s fs=%s 图元 %d 顶点 %d 实例 %d 起点 %d 基顶点 %d 基实例 %d 目标 %s\n", (unsigned long long)atomic_load(&g_cur_frame), g_stat_pass,
            pr->sh[0] ? pr->sh[0]->key : "-", pr->sh[1]->key, dw ? dw->prim : di->prim, count, inst, first, di ? di->base_vertex : 0, binst, R.rt.n && R.rt.color[0] ? "有" : "无");
    for (int i = 0; i < S.nattr && i < 16; i++) fprintf(stderr, "[kd]   属性 %d 格式 %#x 偏移 %u 流 %u\n", i, (unsigned)S.attr[i].fmt, (unsigned)S.attr[i].offset, (unsigned)S.attr[i].stream);
    for (int s_ = 0; s_ < S.nstream && s_ < 16; s_++) {
        uint32_t st = S.stream[s_].stride, dv = S.stream[s_].divisor;
        fprintf(stderr, "[kd]   流 %d 步长 %u 除数 %u 缓冲 %#llx 大小 %llu\n", s_, st, dv, (unsigned long long)S.vbuf[s_].address, (unsigned long long)S.vbuf[s_].size);
        if (!S.vbuf[s_].address || !st) continue;
        uint64_t from = (uint64_t)(dv ? binst : (di ? di->base_vertex : first)) * st, n = (uint64_t)st * (dv ? (uint64_t)MIN(inst, 8) : 4);
        if (from < S.vbuf[s_].size) kd_floats("流", s_, S.vbuf[s_].address + from, MIN(n, S.vbuf[s_].size - from), 512);
    }
    for (int stg = 0; stg < 2; stg++) { NvShader *sh = pr->sh[stg]; if (!sh) continue;
        for (int i = 0; i < sh->rf.nb; i++) if (sh->rf.b[i].kind == NVN_RB_CBUF && sh->rf.b[i].reg < 16) {
            const p_bind_buf *u = &S.ubo[stg][sh->rf.b[i].reg]; kd_floats(stg ? "片元 uniform" : "顶点 uniform", sh->rf.b[i].reg, u->addr, u->size, 4096); } }
    return 1;
}
// Diagnostic DM_NVN_MAGSCAN=<fragment key>:<scan every N frames, default 30>:<frames per phase, default 600>: on-device self-check for the magenta squares (Super Mario 3D World + Bowser's Fury 1-1, iPhone only, not reproducible on the Mac).
// In a scan frame, as soon as each render pass ends, a compute kernel reads its color attachments (read goes through the same format decode as the game's shader samples) and gathers per 16x16 block: the magenta pixel count (R and B both >= 0.8 of the largest channel, G <= 0.1 of it),
// the non-finite pixel count and each channel's min / max. An attachment with a block holding one single magenta value throughout (exactly what the squares in the iPhone screenshot are, aligned to the 16-pixel grid) or with non-finite values gets a [mag] line (index of the pass within the frame, distance from the pass holding this particle, attachment serial / format / size, block position and value).
// Four phases rotate: phase 0 draws as usual, phase 1 drops every draw of this fragment shader, phases 2 / 3 keep the geometry and replace the fragment output with a constant (grey / bright magenta). Comparing the phases tells whether this draw causes it
// and whether its fragment output (texture reads, value range) matters; the first pass where the magenta blocks show tells which step goes wrong
static char g_ms_key[24]; static int g_ms_every, g_ms_plen, g_ms_init;
static int ms_on(void);
static const char *const k_ms_phase[4] = {"照常画", "丢掉该着色器", "该着色器改出常量灰", "该着色器改出常量高亮品红"};
static int ms_phase_now(void) { return (int)(atomic_load(&g_cur_frame) / (uint64_t)g_ms_plen % 4); }
// Phases 2 / 3: this shader's draws keep their geometry and the fragment stage is replaced by a constant output (grey 0.5 / bright magenta 1500,2,1500). The pipeline cache key carries the replacement function, kept apart from the original pipeline; these pipelines are not recorded in the prewarm list
static id<MTLFunction> ms_fs_override(NvShader *fs) {
    if (!fs || !ms_on() || strncmp(fs->key, g_ms_key, strlen(g_ms_key))) return nil;
    int ph = ms_phase_now(); if (ph < 2) return nil;
    static id<MTLFunction> fn[2]; static int failed;
    if (!fn[0] && !failed) { NSError *e = nil; id<MTLLibrary> lib = [g_dev newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal;\n"
            "fragment float4 msg() { return float4(0.5, 0.5, 0.5, 1.0); }\nfragment float4 msm() { return float4(1500.0, 2.0, 1500.0, 0.6); }\n" options:nil error:&e];
        fn[0] = [lib newFunctionWithName:@"msg"]; fn[1] = [lib newFunctionWithName:@"msm"];
        if (!fn[0] || !fn[1]) { failed = 1; MLOG("【红】DM_NVN_MAGSCAN：常量片元编不出 %s", e.localizedDescription.UTF8String); } }
    return failed ? nil : fn[ph - 2];
}
static int ms_on(void) {
    if (!g_ms_init) { g_ms_init = 1; const char *e = getenv("DM_NVN_MAGSCAN");
        if (e && *e) { size_t n = strcspn(e, ":"); if (n > 23) n = 23; memcpy(g_ms_key, e, n); g_ms_key[n] = 0;
            const char *c = strchr(e, ':'); g_ms_every = c ? atoi(c + 1) : 30; c = c ? strchr(c + 1, ':') : NULL; g_ms_plen = c ? atoi(c + 1) : 600;
            if (g_ms_every < 1) g_ms_every = 30; if (g_ms_plen < g_ms_every) g_ms_plen = 600;
            MLOG("DM_NVN_MAGSCAN：片元着色器 %s，每 %d 帧扫一次各通道颜色附件里的品红块；每 %d 帧换一相（相 0 照常画 / 1 丢掉这个着色器的绘制 / 2 它改出常量灰 / 3 它改出常量高亮品红）（[mag] 行）", g_ms_key, g_ms_every, g_ms_plen); } }
    return g_ms_key[0] != 0;
}
static uint64_t g_ms_ppass = ~0ull, g_ms_pframe = ~0ull, g_ms_drawn, g_ms_dropped;
// returns 1 = this phase drops the draw
static int ms_draw(NvProg *pr) {
    if (!ms_on() || !pr->sh[1] || strncmp(pr->sh[1]->key, g_ms_key, strlen(g_ms_key))) return 0;   // a deferred draw returns before this when recorded and only arrives here when replayed: every draw passes exactly once
    uint64_t f = atomic_load(&g_cur_frame);
    if (f / (uint64_t)g_ms_plen % 4 == 1) { g_ms_dropped++; return 1; }
    g_ms_drawn++; if (g_ms_pframe != f) { g_ms_pframe = f; g_ms_ppass = g_stat_pass; }   // the pass where this particle first draws this frame
    return 0;
}
static const char k_ms_msl[] =
"#include <metal_stdlib>\nusing namespace metal;\n"
"kernel void ms(texture2d<float, access::read> t [[texture(0)]], device uint *o [[buffer(0)]], uint2 g [[thread_position_in_grid]], uint2 tg [[threadgroup_position_in_grid]],\n"
"               uint li [[thread_index_in_threadgroup]], uint si [[simdgroup_index_in_threadgroup]], uint sl [[thread_index_in_simdgroup]]) {\n"
"  threadgroup uint sh[8][8]; bool in = g.x < o[0] && g.y < o[1]; float4 c = in ? t.read(g) : float4(0); float m = max(c.r, max(c.g, c.b));\n"
"  bool bad = in && (isnan(c.r) || isnan(c.g) || isnan(c.b) || isinf(c.r) || isinf(c.g) || isinf(c.b));\n"
"  bool mag = in && !bad && m > 0.0 && c.r >= 0.8 * m && c.b >= 0.8 * m && c.g <= 0.1 * m;\n"
"  uint r = as_type<uint>(max(c.r, 0.0)), gg = as_type<uint>(max(c.g, 0.0)), b = as_type<uint>(max(c.b, 0.0));\n"
"  uint v[8] = {simd_sum(mag ? 1u : 0u), simd_sum(bad ? 1u : 0u), simd_min(in ? r : ~0u), simd_max(in ? r : 0u), simd_min(in ? gg : ~0u), simd_max(in ? gg : 0u), simd_min(in ? b : ~0u), simd_max(in ? b : 0u)};\n"
"  if (sl == 0) for (int k = 0; k < 8; k++) sh[si][k] = v[k];\n"
"  threadgroup_barrier(mem_flags::mem_threadgroup); if (li) return;\n"
"  for (int s = 1; s < 8; s++) { v[0] += sh[s][0]; v[1] += sh[s][1]; for (int k = 2; k < 8; k += 2) { v[k] = min(v[k], sh[s][k]); v[k + 1] = max(v[k + 1], sh[s][k + 1]); } }\n"
"  device uint *q = o + 4 + 4 * (tg.y * o[2] + tg.x); q[0] = v[0] | v[1] << 16 | (v[2] == v[3] && v[4] == v[5] && v[6] == v[7] ? 1u << 31 : 0u); q[1] = v[3]; q[2] = v[5]; q[3] = v[7]; }\n";
#define MS_BUF (16u << 20)
#define MS_MAXE 256
typedef struct { uint64_t pass_idx, ser; int att, pdist, fmt; uint32_t w, h, off; } ms_ent;
typedef struct { __strong id<MTLBuffer> b; uint64_t frame, fence; uint32_t used; int n, busy, full; ms_ent e[MS_MAXE]; } ms_scan;
static ms_scan g_ms_s[2]; static int g_ms_cur = -1;   // which buffer this scan frame writes (-1 = not scanning this frame)
static uint64_t g_ms_curf = ~0ull, g_ms_first;
static __strong id<MTLTexture> g_ms_att[8]; static NSUInteger g_ms_lv[8], g_ms_sl[8]; static uint64_t g_ms_aser[8]; static int g_ms_na; static uint64_t g_ms_apass;
static int g_ms_seq = -1, g_ms_sc, g_ms_hitf, g_ms_maxblk, g_ms_pscan; static uint64_t g_ms_f0, g_ms_f1, g_ms_pmag;   // the phase being tallied (frame / frames per phase) and its totals
static void ms_poll(void) {
    for (int k = 0; k < 2; k++) { ms_scan *s = &g_ms_s[k]; if (!s->busy || g_ms_cur == k || s->fence > g_done) continue;
        int seq = (int)(s->frame / (uint64_t)g_ms_plen);
        if (seq != g_ms_seq) {
            if (g_ms_seq >= 0) MLOG("[mag] 相 %d（%s）帧 %llu–%llu：扫 %d 帧，有均一品红块（16×16 整块同一个品红值）的 %d 帧，单帧最多 %d 块；粒子所在通道的附件里品红像素 每帧均 %.1f（%d 帧）；该着色器至今画 %llu 次、丢 %llu 次",
                                    g_ms_seq % 4, k_ms_phase[g_ms_seq % 4], (unsigned long long)g_ms_f0, (unsigned long long)g_ms_f1, g_ms_sc, g_ms_hitf, g_ms_maxblk, g_ms_pscan ? (double)g_ms_pmag / g_ms_pscan : 0.0, g_ms_pscan,
                                    (unsigned long long)g_ms_drawn, (unsigned long long)g_ms_dropped);
            g_ms_seq = seq; g_ms_sc = g_ms_hitf = g_ms_maxblk = g_ms_pscan = 0; g_ms_pmag = 0; g_ms_f0 = s->frame; }
        g_ms_sc++; g_ms_f1 = s->frame; int fblk = 0, pseen = 0;
        for (int i = 0; i < s->n; i++) { const ms_ent *e = &s->e[i]; const uint32_t *o = (const uint32_t *)((const uint8_t *)s->b.contents + e->off);
            uint32_t tw = (e->w + 15) / 16, th = (e->h + 15) / 16; uint64_t bad = 0, mag = 0; int ublk = 0, mblk = 0, nl = 0; char line[1400]; size_t lo = 0;
            for (uint32_t t = 0; t < tw * th; t++) { const uint32_t *q = o + 4 + 4 * t; uint32_t nmag = q[0] & 0xffff, nbad = q[0] >> 16 & 0x7fff; bad += nbad; mag += nmag;
                uint32_t npx = MIN(16u, e->w - t % tw * 16) * MIN(16u, e->h - t / tw * 16);
                int uni = q[0] >> 31;   // the whole block holds one value
                int allm = nmag == npx;
                if (allm) mblk++; if (allm && uni && npx >= 64) ublk++;
                if (((allm && uni && npx >= 64) || nbad) && nl < 8 && lo < sizeof line - 120) { float r, g, b; memcpy(&r, &q[1], 4); memcpy(&g, &q[2], 4); memcpy(&b, &q[3], 4); nl++;
                    lo += snprintf(line + lo, sizeof line - lo, " [%u,%u%s 最大值 %g %g %g 非有限 %u]", t % tw * 16, t / tw * 16, uni ? " 均一" : "", r, g, b, nbad); } }
            fblk += ublk; if (e->pdist == 0 && e->att == 0) { g_ms_pmag += mag; pseen = 1; }
            if (ublk || bad) MLOG("[mag] F%llu 相 %llu 通道 %llu（粒子通道%+d）附件 %d ser=%llu 格式 %d %ux%u：均一品红块 %d（全品红块 %d），非有限 %llu 像素%s", (unsigned long long)s->frame, (unsigned long long)(s->frame / (uint64_t)g_ms_plen % 4),
                                  (unsigned long long)e->pass_idx, e->pdist, e->att, (unsigned long long)e->ser, e->fmt, e->w, e->h, ublk, mblk, (unsigned long long)bad, line);
        }
        if (fblk) g_ms_hitf++; if (fblk > g_ms_maxblk) g_ms_maxblk = fblk; g_ms_pscan += pseen;
        if (s->full) MLOG("[mag] F%llu 扫描缓冲满，本帧后面的通道没扫", (unsigned long long)s->frame);
        s->busy = 0; s->n = 0; s->used = 0; s->full = 0; }
}
void ms_note(MTL4RenderPassDescriptor *pd, NSUInteger nc) {
    g_ms_na = 0; if (!ms_on()) return;
    uint64_t f = atomic_load(&g_cur_frame);
    if (f != g_ms_curf) {   // a new frame: the previous scan frame's counts are processed once its submission completes
        g_ms_curf = f; g_ms_first = g_stat_pass + 1; if (g_ms_cur >= 0) g_ms_cur = -1;
        if (f % (uint64_t)g_ms_every == 0) { for (int k = 0; k < 2; k++) if (!g_ms_s[k].busy) { g_ms_cur = k; break; }
            if (g_ms_cur >= 0) { ms_scan *s = &g_ms_s[g_ms_cur];
                if (!s->b) { s->b = [g_dev newBufferWithLength:MS_BUF options:MTLResourceStorageModeShared]; if (!s->b) { g_ms_cur = -1; return; } res_add(s->b); }
                s->busy = 1; s->frame = f; s->n = 0; s->used = 0; s->full = 0; s->fence = 0; } } }
    ms_poll();
    if (g_ms_cur < 0) return;
    g_ms_apass = g_stat_pass + 1;
    for (NSUInteger i = 0; i < nc && i < 8; i++) { id<MTLTexture> t = pd.colorAttachments[i].texture; if (!t) continue;
        g_ms_att[g_ms_na] = t; g_ms_lv[g_ms_na] = pd.colorAttachments[i].level; g_ms_sl[g_ms_na] = pd.colorAttachments[i].slice; g_ms_aser[g_ms_na] = R.rt.color[i] ? R.rt.color[i]->serial : 0; g_ms_na++; }
}
void ms_after(void) {
    if (!g_ms_na) return;
    int na = g_ms_na; g_ms_na = 0;
    if (g_ms_cur < 0 || g_stat_pass != g_ms_apass) { for (int i = 0; i < na; i++) g_ms_att[i] = nil; return; }
    static int failed; static id<MTLComputePipelineState> ps; static id<MTL4ArgumentTable> at; static int logged_att;
    if (failed) return;
    if (!ps) { NSError *e = nil; id<MTLLibrary> lib = [g_dev newLibraryWithSource:@(k_ms_msl) options:nil error:&e]; id<MTLFunction> fn = lib ? [lib newFunctionWithName:@"ms"] : nil;
        ps = fn ? [g_dev newComputePipelineStateWithFunction:fn error:&e] : nil;
        MTL4ArgumentTableDescriptor *ad = [MTL4ArgumentTableDescriptor new]; ad.maxTextureBindCount = 1; ad.maxBufferBindCount = 1; at = [g_dev newArgumentTableWithDescriptor:ad error:&e];
        if (!ps || !at) { failed = 1; MLOG("【红】DM_NVN_MAGSCAN：计算管线建不出 %s", e.localizedDescription.UTF8String); return; } }
    ms_scan *s = &g_ms_s[g_ms_cur]; int pdist = g_ms_pframe == s->frame && g_ms_ppass != ~0ull ? (int)((int64_t)g_stat_pass - (int64_t)g_ms_ppass) : 9999;
    for (int i = 0; i < na; i++) { id<MTLTexture> t = g_ms_att[i]; g_ms_att[i] = nil;
        MTLPixelFormat pf = t.pixelFormat;
        int ok = pf == MTLPixelFormatRGBA8Unorm || pf == MTLPixelFormatRGBA8Unorm_sRGB || pf == MTLPixelFormatBGRA8Unorm || pf == MTLPixelFormatBGRA8Unorm_sRGB || pf == MTLPixelFormatRGB10A2Unorm
              || pf == MTLPixelFormatRG11B10Float || pf == MTLPixelFormatRGBA16Float || pf == MTLPixelFormatRGBA16Unorm || pf == MTLPixelFormatRGBA32Float;
        if (!ok || !(t.usage & MTLTextureUsageShaderRead) || t.sampleCount > 1 || (t.width >> g_ms_lv[i]) < 64 || t.storageMode == MTLStorageModeMemoryless || (t.textureType != MTLTextureType2D && t.textureType != MTLTextureType2DArray)) continue;
        if (pdist == 0 && !logged_att++) MLOG("[mag] 粒子所在通道的附件 %d：ser=%llu 格式 %lu %lux%lu 级 %lu 层 %lu 存储 %lu 用途 %#lx 压缩类型 %ld 允许 GPU 优化排布 %d", i, (unsigned long long)g_ms_aser[i], (unsigned long)pf,
                                             (unsigned long)t.width, (unsigned long)t.height, (unsigned long)g_ms_lv[i], (unsigned long)g_ms_sl[i], (unsigned long)t.storageMode, (unsigned long)t.usage, (long)t.compressionType, (int)t.allowGPUOptimizedContents);
        uint32_t w = (uint32_t)MAX(t.width >> g_ms_lv[i], 1ul), h = (uint32_t)MAX(t.height >> g_ms_lv[i], 1ul), tw = (w + 15) / 16, th = (h + 15) / 16, need = 16 + tw * th * 16;
        if (s->n >= MS_MAXE || s->used + need > MS_BUF) { s->full = 1; continue; }
        id<MTLTexture> v = t;
        if (t.textureType != MTLTextureType2D || g_ms_lv[i] || g_ms_sl[i]) v = [t newTextureViewWithPixelFormat:pf textureType:MTLTextureType2D levels:NSMakeRange(g_ms_lv[i], 1) slices:NSMakeRange(g_ms_sl[i], 1)];
        if (!v) continue;
        uint32_t *o = (uint32_t *)((uint8_t *)s->b.contents + s->used); o[0] = w; o[1] = h; o[2] = tw; o[3] = 0;
        ms_ent *e = &s->e[s->n++]; e->pass_idx = g_stat_pass - g_ms_first; e->ser = g_ms_aser[i]; e->att = i; e->pdist = pdist; e->fmt = (int)pf; e->w = w; e->h = h; e->off = s->used; s->used += need;
        if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
        id<MTL4ComputeCommandEncoder> ce = (id<MTL4ComputeCommandEncoder>)enc_c_begin(NULL); [ce setArgumentTable:at];
        [at setTexture:v.gpuResourceID atIndex:0]; [at setAddress:s->b.gpuAddress + e->off atIndex:0];
        enc_c_pso(ce, ps); enc_c_dispatch(ce, MTLSizeMake(tw, th, 1), MTLSizeMake(16, 16, 1)); enc_c_end(ce);
        args_dirty_all(); s->fence = g_fence + 1;
        if (v != t) defer_obj_unref(v);
    }
}
// Same-pass feedback sampling: a texture view this draw's fragment stage samples is exactly the level and layer of a color attachment of the current render pass (Super Mario 3D World + Bowser's Fury:
// 66a71a3081b91b9a writes the tone-mapped 1920x1080 RGB10A2 image as an attachment while sampling it as a texture). On the Switch (Maxwell, immediate-mode) the read returns the content from before this draw;
// on Metal, writing and sampling the same attachment within one render pass is undefined: the Mac happens to read correctly, while the iPhone (A20 Pro) now and then reads whole 16x16 blocks as (1,0,1) and FXAA carries them into the picture - the magenta squares in 1-1
// (measured with DM_NVN_MAGSCAN: the squares first appear on this attachment, and still appear with the particle dropped).
// The fix: before this draw opens its render pass, end the current render pass (the attachment lands in memory), copy that level and layer into a snapshot texture in the replay stream, make this draw sample the snapshot; later draws reopen the render pass as usual (loading the attachment).
// Only single-level 2D views that are exactly the attachment's level and layer are handled; other shapes are reported once and left as they were. Done only for the fragment shaders the game profile's feedback_snap lists (see fb_want)
// Which fragment shaders get it: the game profile's feedback_snap (key prefixes, space-separated); DM_NVN_FB_SNAP=0 none, all every shader, <keys,...> another list. The cost of doing all:
// Breath of the Wild has over a thousand draws a frame sampling the G-buffer of their own pass (frame time 16.6 -> 36.6 ms) and Mario Wonder's submit time went 1.4 -> 4.0 ms, so only shaders seen to break are listed
static char g_fb_keys[16][20]; static int g_fb_nk = -1, g_fb_all;
static int fb_want(const char *key) {
    if (g_fb_nk < 0) { g_fb_nk = 0; const char *e = getenv("DM_NVN_FB_SNAP"), *src = e && *e ? e : DM_GAME_FEEDBACK_SNAP;
        if (!strcmp(src, "all")) g_fb_all = 1;
        else if (strcmp(src, "0")) for (const char *q = src; *q && g_fb_nk < 16;) { size_t n = strcspn(q, ", "); if (n) { if (n > 16) n = 16; memcpy(g_fb_keys[g_fb_nk], q, n); g_fb_keys[g_fb_nk++][n] = 0; } q += n; q += strspn(q, ", "); }
        if (g_fb_all || g_fb_nk) MLOG("同通道反馈采样快照：%s%s", g_fb_all ? "全部片元着色器" : src, e && *e ? "（DM_NVN_FB_SNAP）" : "（游戏档案 feedback_snap）"); }
    if (g_fb_all) return 1;
    for (int i = 0; i < g_fb_nk; i++) if (!strncmp(key, g_fb_keys[i], strlen(g_fb_keys[i]))) return 1;
    return 0;
}
static id<MTLTexture> fb_snap_tex(MTLPixelFormat pf, NSUInteger w, NSUInteger h) {   // one per (format, size); full queue barriers separate consecutive snapshots, so reuse within a frame does not conflict
    static __strong id<MTLTexture> pool[4]; static int next;
    for (int i = 0; i < 4; i++) if (pool[i] && pool[i].pixelFormat == pf && pool[i].width == w && pool[i].height == h) return pool[i];
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pf width:w height:h mipmapped:NO];
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView; d.storageMode = MTLStorageModePrivate;
    id<MTLTexture> t = [g_dev newTextureWithDescriptor:d]; if (!t) return nil; res_add(t);
    if (pool[next]) defer_obj_release(pool[next]); pool[next] = t; next = (next + 1) & 3;
    return t;
}
// Called before the draw opens its render pass: if a snapshot is needed, end the current render pass, copy the snapshot and register the substitution (used by this draw's argument encoding)
static void fb_prepare(NvProg *pr) {
    g_draw_seq++; g_fb_n = 0;
    if (g_fb_rebind) { args_dirty_all(); g_fb_rebind = 0; }   // the previous draw had a substituted binding: re-encode with the original bindings this time
    if (!pr || !pr->sh[1] || !R.rt.n || g_fz.active || g_defer_replaying || !fb_want(pr->sh[1]->key)) return;
    NvShader *fs = pr->sh[1];
    for (int i = 0; i < fs->rf.nb && g_fb_n < 4; i++) { const nvn_rbind *b = &fs->rf.b[i]; if (b->kind != NVN_RB_TEX || b->reg >= 128) continue;
        int tid = (int)(S.tex[1][b->reg] & 0xfffff); if (!tid) continue;
        pthread_mutex_lock(&g_rmu); uint64_t ser = tid < g_ntreg && g_treg[tid].ok ? g_treg[tid].t.serial : 0; pthread_mutex_unlock(&g_rmu);
        if (!ser) continue;
        for (int k = 0; k < R.rt.n && k < 8 && g_fb_n < 4; k++) { if (!R.rt.color[k] || R.rt.color[k]->serial != ser) continue;
            uint64_t aser; int lv, sl; rt_sub(&R.rt, k, &aser, &lv, &sl);
            pthread_mutex_lock(&g_rmu); void *tv = NULL; tex_fast_rid(tid, &tv); id<MTLTexture> t = (__bridge id<MTLTexture>)tv; pthread_mutex_unlock(&g_rmu);
            if (!t) continue;
            int dup = 0; for (int q = 0; q < g_fb_n; q++) dup |= g_fb_src[q] == t; if (dup) continue;
            id<MTLTexture> base = t.parentTexture ? t.parentTexture : t; NSUInteger tl = t.parentTexture ? t.parentRelativeLevel : 0, ts = t.parentTexture ? t.parentRelativeSlice : 0;
            if (tl + t.mipmapLevelCount <= (NSUInteger)lv || tl > (NSUInteger)lv) continue;   // the sampled levels do not include the attachment's level (mip generation etc.): not feedback
            if (t.textureType != MTLTextureType2D || t.mipmapLevelCount != 1 || t.sampleCount != 1 || ts != (NSUInteger)sl || b->dtt != MTLTextureType2D || base.parentTexture) {
                RED_ONCE(80, "同通道反馈采样：采样视图不是附件那一级那一层的单级二维视图（类型 %lu 级 %lu+%lu 层 %lu，附件 级 %d 层 %d，着色器 %s），照旧", (unsigned long)t.textureType, (unsigned long)tl, (unsigned long)t.mipmapLevelCount, (unsigned long)ts, lv, sl, fs->key);
                continue; }
            NSUInteger w = MAX(base.width >> lv, 1ul), h = MAX(base.height >> lv, 1ul);
            id<MTLTexture> snap = fb_snap_tex(base.pixelFormat, w, h); if (!snap) continue;
            if (g_enc) rp_end();
            id<MTL4ComputeCommandEncoder> e = blit(); enc_b_copy_texture(e, base, (NSUInteger)sl, (NSUInteger)lv, MTLOriginMake(0, 0, 0), MTLSizeMake(w, h, 1), snap, 0, 0, MTLOriginMake(0, 0, 0));
            MTLTextureSwizzleChannels sw = t.swizzle;
            int same = t.pixelFormat == snap.pixelFormat && sw.red == MTLTextureSwizzleRed && sw.green == MTLTextureSwizzleGreen && sw.blue == MTLTextureSwizzleBlue && sw.alpha == MTLTextureSwizzleAlpha;
            id<MTLTexture> use = same ? snap : [snap newTextureViewWithPixelFormat:t.pixelFormat textureType:MTLTextureType2D levels:NSMakeRange(0, 1) slices:NSMakeRange(0, 1) swizzle:sw];
            if (!use) { RED_ONCE(81, "同通道反馈采样：快照视图建不出（格式 %lu），照旧", (unsigned long)t.pixelFormat); continue; }
            if (g_fb_use[g_fb_n] && g_fb_use[g_fb_n] != use) defer_obj_unref(g_fb_use[g_fb_n]);
            g_fb_src[g_fb_n] = t; g_fb_use[g_fb_n] = use; g_fb_rid[g_fb_n] = use.gpuResourceID._impl; g_fb_n++;
            g_fb_count++;
            if (g_fb_count == 1 || g_fb_count == 10 || g_fb_count == 1000 || g_fb_count % 100000 == 0)
                MLOG("同通道反馈采样：快照第 %llu 次（帧 %llu 通道 %llu，着色器 %s 槽 %d 采样附件 %d ser=%llu %lux%lu 格式 %lu）", (unsigned long long)g_fb_count, (unsigned long long)atomic_load(&g_cur_frame), g_stat_pass + !g_enc,
                     fs->key, b->reg, k, (unsigned long long)ser, (unsigned long)w, (unsigned long)h, (unsigned long)base.pixelFormat);
            PD("N %llu 同通道反馈采样：附件 %d ser=%llu 级 %d 层 %d 拷进快照，本次绘制改采样快照", g_stat_pass + !g_enc, k, (unsigned long long)ser, lv, sl);
        } }
    if (g_fb_n) { g_fb_seq = g_draw_seq; args_dirty_all(); g_fb_rebind = 1; }
}
// Whether this draw needs more from the inline updates deferred in the open pass than the uniform shadows give (see mtl_sub_update): a uniform binding they cover only in part, a bindless
// handle in such a part, or their data read as vertices, indices or a storage buffer. Then the pass ends first and the updates land before the draw, as before
static int ubd_conflict(NvProg *pr, const p_draw *dw, const p_draw_idx *di) {
    for (int s_ = 0; s_ < 16; s_++) if (S.vbuf[s_].address && ubd_hits(S.vbuf[s_].address, S.vbuf[s_].address + S.vbuf[s_].size)) return 1;
    if (di && di->ibuf && ubd_hits(di->ibuf, di->ibuf + (uint64_t)di->count * 4)) return 1;
    for (int st = 0; pr && st < 2; st++) { __unsafe_unretained NvShader *sh = pr->sh[st]; if (!sh) continue;   // the stages encode_args binds (vertex, fragment)
        for (int i = 0; i < sh->rf.nb; i++) { const nvn_rbind *b = &sh->rf.b[i]; if (b->reg >= 16) continue;
            const p_bind_buf *u = (b->kind == NVN_RB_SBUFP ? &S.ssbo[st][b->reg] : &S.ubo[st][b->reg]);   // the same table choice as encode_args_ (UBOP is a uniform buffer)
            if (!u->addr) continue;
            if (b->kind == NVN_RB_CBUF) { if (ubd_hits(u->addr, u->addr + u->size) && !ubd_covers(u->addr, u->addr + u->size)) return 1; }
            else if (b->kind == NVN_RB_TEXU || b->kind == NVN_RB_SAMPU) { uint64_t h = u->addr + b->off; if (ubd_hits(h, h + 4) && !ubd_covers(h, h + 4)) return 1; }
            else if (b->kind == NVN_RB_SBUFP || b->kind == NVN_RB_UBOP) { if (ubd_hits(u->addr, u->addr + (u->size ? u->size : 1))) return 1; } } }
    return 0;
}
static void draw(const p_draw *dw, const p_draw_idx *di) {
    if (!S.prog) { RED_ONCE(18, "绘制前没有绑定程序"); PD("Z %llu 绘制跳过：没有绑定程序", g_stat_pass + !g_enc); return; }
    __unsafe_unretained NvProg *pr = prog_get(S.prog);
    if (g_nubd && g_enc && ubd_conflict(pr, dw, di)) { g_ubd_conflicts++; rp_end(); }   // land the deferred inline updates before this draw (see ubd_conflict)
    if (g_ruler_truth == 0) {   // Ruler negative control (DM_NVN_RULER_NEG=<fragment key>): in the mode under test drop this shader's draws, and the ruler must report non-equivalence
        static const char *neg; static int ng = -1; if (ng < 0) { neg = getenv("DM_NVN_RULER_NEG"); ng = neg && *neg; }
        if (ng && pr) for (int st = 0; st < 5; st++) { __unsafe_unretained NvShader *sh = pr->sh[st]; if (sh && sh->stage == 1 && !strncmp(sh->key, neg, strlen(neg))) { static int once; if (!once++) MLOG("DM_NVN_RULER_NEG：被测模式丢弃着色器 %s 的绘制", sh->key); return; } }
    }
    if (zskip_can_draw(pr)) {   // Zero-source skip: the output is always 0, writing 0 over 0, so this draw changes no result (see zskip_can_draw)
        g_zskip_count++; { static int once; if (!once++) MLOG("零源跳过：首例 通道 %llu", g_stat_pass + 1); }
        PD("N %llu 零源跳过", g_stat_pass + 1); return; }
    if (!g_defer_replaying) {
        if (g_defer.active && g_defer.in_p && !g_defer.at_q) { if (defer_take(dw, di)) return; }
    }
    // Draws that cannot draw anything (both faces culled, empty scissor) return before the render pass is opened: otherwise the render pass is already open and on TBDR the whole attachment is loaded and stored for nothing
    // (Mac character select page: 3 render passes are opened only because of such draws, all loaded, zero draws)
    // Diagnostic switch for measuring an upper bound DM_NVN_SKIP_FS=<fragment key>[,<fragment key>...] (none by default): drops all draws of these fragment shaders (the picture will be wrong; only used in real-device ABA to measure "how much GPU time this class of draws is worth in total")
    { static char sk[16][20]; static int nsk = -1;
      if (nsk < 0) { nsk = 0; for (const char *q = getenv("DM_NVN_SKIP_FS"); q && *q && nsk < 16; q = strchr(q, ',') ? strchr(q, ',') + 1 : q + strlen(q)) { size_t n = strcspn(q, ","); if (n > 16) n = 16; memcpy(sk[nsk], q, n); sk[nsk++][n] = 0; }
                     if (nsk) MLOG("DM_NVN_SKIP_FS：丢弃 %d 个片元着色器的绘制（诊断，画面不对）", nsk); }
      if (g_ub_n > 0 && pr->sh[1] && !g_defer_replaying && ub_skip(pr->sh[1]->key)) return;   // In-process alternating measurement: this phase drops this class
      if (nsk && pr->sh[1] && !g_defer_replaying) for (int i = 0; i < nsk; i++) if (!strncmp(pr->sh[1]->key, sk[i], strlen(sk[i]))) return;
      if (ms_draw(pr)) return; }
    // Similar diagnostic DM_NVN_SKIP_SMALL=<width>: drop all draws whose target width (by the attached level) is <= this value -- measures how much "dropping tiny render passes in bulk" is worth under the parallel pipeline (fixed per-pass overhead)
    { static int small = -1; if (small < 0) { const char *e = getenv("DM_NVN_SKIP_SMALL"); small = e ? atoi(e) : 0; if (small) MLOG("DM_NVN_SKIP_SMALL：丢弃目标宽度 ≤ %d 的绘制（诊断，画面不对）", small); }
      if (small > 0 && R.has_rt && !g_defer_replaying) { int k = R.rt.n && R.rt.color[0] ? 0 : R.rt.depth ? 8 : -1;
        if (k >= 0) { uint64_t s_; int lv, sl; rt_sub(&R.rt, k, &s_, &lv, &sl); const tex_t *t = k == 8 ? R.rt.depth : R.rt.color[0]; if ((int)MAX(t->d.width >> lv, 1u) <= small) return; } } }
    // Experimental switch DM_NVN_NOFXAA=1 (off by default; keep or drop decided by the visual ruler + looking at the picture on a real machine): skips the FXAA full-screen draw, and the later sharpening / upscale reads FXAA's input instead (the tone mapping result)
    { static int nofxaa = -1; static uint64_t fx_from, fx_to; if (nofxaa < 0) { const char *e = getenv("DM_NVN_NOFXAA"); nofxaa = e && !strcmp(e, "1"); if (nofxaa) MLOG("DM_NVN_NOFXAA=1：跳过 FXAA"); }
      if (nofxaa && tbdr_on() && !g_defer_replaying && pr->sh[1]) { NvShader *f = pr->sh[1];
        if (!strncmp(f->key, "ebcfcd1fa0a77dbc", 16) && !g_enc && R.rt.n == 1 && R.rt.color[0] && !R.clear_c[0]) {
            uint64_t h = 0; for (int i = 0; i < f->rf.nb && !h; i++) if (f->rf.b[i].kind == NVN_RB_TEX && f->rf.b[i].reg < 128) h = S.tex[1][f->rf.b[i].reg];
            if (h) { fx_from = R.rt.color[0]->serial; fx_to = h; PD("N %llu 跳过 FXAA：ser=%llu 的读者改读 ser=%llu", g_stat_pass, (unsigned long long)fx_from, (unsigned long long)h_ser(h)); return; } }
        else if (fx_from && !strncmp(f->key, "52aac1f2306b8fc5", 16)) { for (int r = 0; r < 128; r++) if (S.tex[1][r] && h_ser(S.tex[1][r]) == fx_from) S.tex[1][r] = fx_to; fx_from = 0; } } }
    { static uint64_t bloomq_frame; if (bloomq_frame != atomic_load(&g_cur_frame)) { bloomq_frame = atomic_load(&g_cur_frame); bloomq_reset(); }
      if (bloomq_enabled() && tbdr_on() && !g_defer_replaying && pr->sh[1]) {
        NvShader *f = pr->sh[1]; uint64_t tgt = 0; int lv, sl;
        if (!strncmp(f->key, "2938a5b71b8e21b2", 16)) {
            if (g_bloomq.state) bloomq_reset();
            if (bloomq_target(&tgt, &lv, &sl) && R.rt.color[0]) {
                uint64_t h = 0; const tex_t *st = NULL; int cbuf = 0;
                for (int i = 0; i < f->rf.nb; i++) { nvn_rbind *b = &f->rf.b[i];
                    if (b->kind == NVN_RB_TEX && b->reg < 128 && !h) { h = S.tex[1][b->reg]; int tid = (int)(h & 0xfffff); if (tid > 0 && tid < g_ntreg && g_treg[tid].ok) st = &g_treg[tid].t; }
                    if (b->kind == NVN_RB_SAMPT && b->reg < 128) g_bloomq.samp_h = S.tex[1][b->reg] >> 20;
                    if (b->kind == NVN_RB_SAMP && b->reg < 128) g_bloomq.samp_h = S.samp[1][b->reg] >> 20;
                    if (b->kind == NVN_RB_CBUF && !cbuf) cbuf = ubo_float(1, (uint16_t)(b->reg << 8), &g_bloomq.threshold);   // c7[0].x
                }
                if (h && st && cbuf && f->rf.c1_len >= sizeof(float) * 2) {
                    float c1[2]; memcpy(c1, f->rf.c1, sizeof c1); g_bloomq.s_tex = st; g_bloomq.s_h = h; g_bloomq.s_ser = st->serial; g_bloomq.b0_ser = tgt; g_bloomq.coeff_g = c1[0]; g_bloomq.coeff_b = c1[1]; g_bloomq.state = 1;
                    { static int once; if (!once++) MLOG("泛光 1/4 起点：生效（S ser=%llu B0 ser=%llu）", (unsigned long long)g_bloomq.s_ser, (unsigned long long)g_bloomq.b0_ser); } PD("A %llu 泛光 1/4 起点 S=%llu B0=%llu", g_stat_pass, (unsigned long long)g_bloomq.s_ser, (unsigned long long)g_bloomq.b0_ser); R.clear_c[0] = 0;   // The whole B0 level is no longer needed: drop the pending clear too, otherwise a clear-only render pass would be opened just for it
                    return;
                }
            }
        } else if (g_bloomq.state == 1 && !strncmp(f->key, "019ce2f82edbda84", 16)) {
            uint64_t src = 0; for (int i = 0; i < f->rf.nb; i++) if (f->rf.b[i].kind == NVN_RB_TEX && f->rf.b[i].reg < 128) { src = h_ser(S.tex[1][f->rf.b[i].reg]); break; }
            if (src == g_bloomq.b0_ser && bloomq_target(&tgt, &lv, &sl) && bloomq_capture()) { R.clear_c[0] = 0; g_bloomq.b1_ser = tgt; g_bloomq.state = 2; PD("B %llu 泛光 1/4 原生截取 B0=%llu B1=%llu", g_stat_pass, (unsigned long long)g_bloomq.b0_ser, (unsigned long long)g_bloomq.b1_ser); return; }
            RED_ONCE(62, "泛光 1/4 起点：截取已跳过但原生截取没画成，本帧缺泛光"); bloomq_reset();
        } else if (g_bloomq.state == 2 && !strncmp(f->key, "019ce2f82edbda84", 16)) {
            for (int i = 0; i < f->rf.nb; i++) if (f->rf.b[i].kind == NVN_RB_TEX && f->rf.b[i].reg < 128 && h_ser(S.tex[1][f->rf.b[i].reg]) == g_bloomq.b1_ser) { g_bloomq.b1_h = S.tex[1][f->rf.b[i].reg]; break; }
        } else if (g_bloomq.state == 2 && !strncmp(f->key, "cc264583f2000c81", 16)) {
            if (bloomq_target(&tgt, &lv, &sl) && tgt == g_bloomq.b0_ser) { g_bloomq.state = 3; PD("C %llu 泛光 1/4 跳过 B0 上采样", g_stat_pass); return; }
        } else if (g_bloomq.state == 3 && !strncmp(f->key, "2e2d6700a2569795", 16)) {
            if (g_bloomq.b1_h) { for (int r = 0; r < 128; r++) if (S.tex[1][r] && h_ser(S.tex[1][r]) == g_bloomq.b0_ser) S.tex[1][r] = g_bloomq.b1_h; PD("D %llu 泛光 1/4 叠加改读 B1", g_stat_pass); g_bloomq_last_b0 = g_bloomq.b0_ser; g_bloomq_last_b1h = g_bloomq.b1_h; g_bloomq_last_b1s = g_bloomq.b1_ser; bloomq_reset(); }
            else { RED_ONCE(63, "泛光 1/4 状态缺少 B1 句柄"); bloomq_reset(); }
        } else if (!g_bloomq.state && g_bloomq_last_b1h && h_ser(g_bloomq_last_b1h) == g_bloomq_last_b1s && !strncmp(f->key, "2e2d6700a2569795", 16)) {   // The B1 handle still points to the same texture (not recognized after a rebuild on resolution change)
            // The bloom overlay is reached without the bloom chain having run this frame (the extrapolated frame of runtime extrapolation skipped it: outside the region the input 469 is the previous frame's LDR, so it cannot run): redirect the read to B1 as last time, overlaying the previous frame's bloom.
            // Without the redirect it would read B0 -- with bloom 1/4 on, B0's upsample is always skipped and never written, so what is read is uninitialized video memory (on M4 Pro the units in the region turn entirely green; elsewhere it is mostly 0, i.e. missing bloom)
            int hit = 0; for (int r = 0; r < 128; r++) if (S.tex[1][r] && h_ser(S.tex[1][r]) == g_bloomq_last_b0) { S.tex[1][r] = g_bloomq_last_b1h; hit = 1; }
            if (hit) { PD("D %llu 泛光链没跑：叠加沿用上一帧 B1", g_stat_pass); static uint64_t n; if (n++ % 600 == 0) MLOG("泛光 1/4：泛光链没跑的帧叠加沿用上一帧 B1（第 %llu 次）", (unsigned long long)n); }
        } else if (g_bloomq.state) { RED_ONCE(61, "泛光 1/4 起点：链中途出现别的绘制，本帧泛光不完整"); bloomq_reset(); }
      }
    }
    if (S.poly.cull == NVN_FACE_FRONT_AND_BACK) { PD("Z %llu 绘制跳过（开通道前）：两面全剔", g_stat_pass + !g_enc); return; }
    // The continued render pass may have been opened with fewer depth / stencil attachments (open modes 2 / 3 / 4): if the plane this draw needs is not attached, split before it (on reopen, attach per the new lookahead)
    if (g_rt_merged && g_enc_mode != 0 && (((S.ds.dt || S.ds.dw) && !g_pass.hd) || (S.ds.st && !g_pass.hs))) {
        static int once2; if (!once2++) MLOG("同目标续用：续用通道内绘制要的深度 / 模板平面没挂，在此切开（首例 通道 %llu）", g_stat_pass);
        PD("M %llu 续用通道内绘制要的深度 / 模板平面没挂，在此切开", g_stat_pass); rp_end(); }
    if (g_rt_range_active && (!rt_range_pr_ro(pr) || !S.ds.dt || S.ds.dw || S.ds.st)) rp_end();   // The external write visibility at the original encoder boundary is not removed
    if (g_rt_merged && draw_samples_rt(pr)) { static int once; if (!once++) MLOG("同目标续用：续用通道内绘制要采样附件，在此切开（首例 通道 %llu）", g_stat_pass);
        PD("M %llu 续用通道内绘制要采样附件，在此切开", g_stat_pass); rp_end(); }
    static id<MTLFunction> fz_fn;   // Specialized fragment function for this draw of the later pass in a merged render pass (nil = use the original version)
    if (g_fz.active && g_enc && g_fz.phase == 2 && !fz_p2_draw(pr, dw, di, g_fz.lmap2, &fz_fn)) {   // Cannot be proven: end the merged render pass (the earlier pass's result is stored as usual) and reopen with the later pass's original targets
        static int once3; if (!once3++) MLOG("可读 tile 通道合并：后一个通道的绘制证不出，退回分开（首例 通道 %llu）", g_stat_pass);
        g_fz_fallback++; rp_end(); }
    fb_prepare(pr);   // same-pass feedback sampling: copy the snapshot first (see fb_prepare)
    if (R.has_sc && !g_enc) {
        NSUInteger tw0 = 0, th0 = 0; rt_dims(&tw0, &th0);
        long x0 = R.sc.x < 0 ? 0 : R.sc.x, y0 = R.sc.y < 0 ? 0 : R.sc.y, x1 = (long)R.sc.x + R.sc.w, y1 = (long)R.sc.y + R.sc.h;
        if (x1 > (long)tw0) x1 = tw0; if (y1 > (long)th0) y1 = th0;
        if (x1 <= x0 || y1 <= y0) { PD("Z %llu 绘制跳过（开通道前）：剪裁为空（%d,%d %dx%d，目标 %lux%lu）", g_stat_pass + 1, R.sc.x, R.sc.y, R.sc.w, R.sc.h, (unsigned long)tw0, (unsigned long)th0); return; }
    }
#ifdef NVN_RT_RANGE_TEST
    if (!g_enc) rr_begin();
#endif
    int defer_q_draw = !g_enc && !g_defer_replaying && g_defer_natq > 0;   // The draw that opens Q's render pass: after the render pass is opened as usual through hoist_try / fuse_try, replay P first and then draw this one
    if (g_frz_state) goto frz_plain;   // Full frame / frozen frame of world freeze: no hoisting, no deferral, no merging (the pass-skipping state machine must see every set-target)
    if (!g_enc && tbdr_on()) hoist_try();   // First hoist and execute the pure-clear render passes sandwiched between this render pass and its continuation (before rp_begin_ex, not occupying this render pass)
    // The deferral decision comes after hoist_try: hoisted pure-clear render passes are skipped in the lookahead, and only then can P's same-target continuation segment be recognized
    if (!g_enc && tbdr_on() && !defer_has_in_p() && !g_defer_replaying && !g_fz.active && !g_fz_checking && g_exec_next) { defer_try(dw, di); if (g_defer_cur->active && g_defer_cur->in_p) return; }
    if (!g_enc && tbdr_on() && !g_fz_checking) fuse_try();   // Readable tile render pass merge is on by default; when the structure / same-pixel proof does not hold it still falls back to a normal render pass, and it is not re-entered during the CHECK redraw
frz_plain:;
    int need = g_enc || !tbdr_on() ? 3 : pass_ds_need(), mode = defer_q_draw ? 0 : need == 0 ? 2 : need == 2 ? 3 : need == 1 ? 4 : 0;   // 2 attach neither, 3 attach stencil only, 4 attach depth only
    int skip_d = mode != 0;
    if (!g_enc) { g_fd_dw = dw; g_fd_di = di; }
    id<MTLRenderCommandEncoder> en = rp_begin_ex(mode);
    g_fd_dw = NULL; g_fd_di = NULL;
    if (!en) { PD("Z %llu 绘制跳过：没开成渲染通道", g_stat_pass); return; }   // Z lines: why a draw was dropped at these early returns
    if (defer_q_draw) defer_replay();
    if (skip_d) { g_stat_ds_omit++; if (mode == 3) g_stat_s_only++; else if (mode == 4) g_stat_d_only++; }
    // The pipeline depends only on (program, render pass targets, blend / color / write mask / multisample state, vertex layout): if these did not change, reuse the previous one
    static id<MTLRenderPipelineState> cur_ps; static NvProg *cur_pr; static uint64_t cur_ps_pass;
    if (g_fz.active && !g_defer_replaying) { g_pso_lmap = g_fz.phase == 1 ? g_fz.lmap1 : g_fz.lmap2; g_pso_fsfn = g_fz.phase == 1 ? nil : fz_fn; g_pso_fz = g_fz.phase == 1 ? (pso_fz_desc){0} : g_fz_desc; g_pso_dirty = 1; }   // Readable tile merge: fetch the pipeline by the physical -> logical mapping and the specialization
    if (g_pso_dirty || pr != cur_pr || cur_ps_pass != g_pass.serial) { cur_ps = pso_get(pr, nil); cur_pr = pr; cur_ps_pass = g_pass.serial; g_pso_dirty = !cur_ps; }
    if (g_pso_lmap) { g_pso_lmap = NULL; g_pso_fsfn = nil; memset(&g_pso_fz, 0, sizeof g_pso_fz); g_pso_dirty = 1; }   // The next draw fetches again according to its own situation
    __unsafe_unretained id<MTLRenderPipelineState> ps = cur_ps;
    if (!ps) { PD("Z %llu 绘制跳过：没有渲染管线（顶点 %s 片元 %s）", g_stat_pass, pr->sh[0] ? pr->sh[0]->key : "-", pr->sh[1] ? pr->sh[1]->key : "-"); return; }
    if (ps != g_enc_pso) { enc_r_pso(en, ps); g_enc_pso = ps; }
    static id<MTLDepthStencilState> cur_ds; static uint64_t cur_ds_pass;
    if (g_dss_dirty || cur_ds_pass != g_pass.serial) { cur_ds = dss_get(g_pass.hd, g_pass.hs); cur_ds_pass = g_pass.serial; g_dss_dirty = 0; }
    if (cur_ds != g_enc_dss) { enc_r_dss(en, cur_ds); g_enc_dss = cur_ds; }
    if (g_rt_ovr && g_pass.hs) enc_r_stencil_ref(en, g_rt_ovr == 3 ? 0x40 : 0, 0); else enc_r_stencil_ref(en, S.st_ref[0], S.st_ref[1]);
    // Viewport / scissor (clamped within the target size)
    NSUInteger tw = g_pass.tw, th = g_pass.th;
    // When the UI rectangle does not touch the top edge of the picture (iPad true full screen: the UI is laid out at 16:9 at the bottom of the picture, see dm/host/dm_uiscreen.c), UI draws are shifted down as a whole: a vertical offset is added to the projection matrix (ui_mat_patch), and the scissor moves down with it.
    // A UI draw = 4 vertices, 2 attributes, 20-byte vertices, in a pass of its own and composited at end of frame (measured: startup loading, character select, character creation, entering-game loading, in-game, open panel, and full-screen map never mix in other draws).
    // When the matrix cannot be read this time (there are still unlanded writes in this submit), fall back to moving the viewport: same position, except that widgets lifted to the top get clipped by the viewport's top edge this one time.
    // A scissor starting from the top edge (y <= 0) does not have its top edge moved down: when the host lifts top-anchored widgets (minimap, area name etc.) back to the real top of the picture, they are drawn above the UI rectangle and must not be clipped.
    // Exception: the untextured block that exactly covers the whole UI rectangle (the dimming when a panel opens, the black curtain of a transition) is changed to cover the whole picture -- the viewport is scaled vertically by 1/h and not moved down, otherwise the 3D outside the UI rectangle is left with a light/dark dividing line.
    // When nobody registered the entry or the UI rectangle is the whole picture (iPhone, Mac default), vo = so = 0, vs = 1, the matrix is not changed, and what follows is word-for-word the same as before.
    long vo = 0, so = 0; double vs = 1, moff = 0, mxoff = 0;
    // Is this a screen-space UI draw? The game profile's signature (fragment shader key of the UI pass) wins when it has one;
    // the old 4-vertex / 2-attribute / 20-byte test stays for titles where the UI rectangle is a separate pass.
    static const char *uifs = DM_GAME_UI_FS_KEY; static int uifs_on = -1; if (uifs_on < 0) uifs_on = uifs && *uifs;
    int ui_draw = uifs_on ? (pr->sh[1] && !strncmp(pr->sh[1]->key, uifs, strlen(uifs)))
                          : (dw && dw->count == 4 && S.nattr == 2 && S.stream[0].stride == 20);
    if (g_ui_rect_fn && ui_draw) { double uy = 0, uh = 1; g_ui_rect_fn(&uy, &uh);
        if (uh > 0 && uh < 1) { if (ui_cover_all(uh)) vs = 1 / uh; else if (uy > 0) { so = lround(uy * th); if (ui_mat_ok()) moff = 2 * uy; else { vo = so; static int n; if (n++ < 4) MLOG("界面下移：投影矩阵这次读不到，退回挪视口（通道 %llu）", g_stat_pass); } } } }
    // DM_UI_VYOFF / DM_UI_HXOFF / DM_UI_VPXOFF (clip units, or pixels for VPXOFF) are diagnostics: they force one offset through one
    // of the two shift paths to tell "the offset never reaches the pixels" apart from "these are not the draws I am looking at".
    { static int got = -1; static double vy; if (got < 0) { const char *e = getenv("DM_UI_VYOFF"); got = e && *e; vy = got ? atof(e) : 0; if (got) MLOG("DM_UI_VYOFF：UI 竖直偏移强制为 %.4f（裁剪空间）", vy); }
      if (got && ui_draw && ui_mat_ok()) moff = vy; }
    // Hug-the-right shift: same projection-matrix channel as the vertical offset, but the draw is chosen by where it lands (ui_draw_ndc).
    // Every draw that is flat in depth and does not reach the middle band moves right; anything spanning the middle (dimming, curtain,
    // vignette, letterbox, full-screen map) stays put, and world geometry never qualifies (its perspective matrix has m[3].w != 0).
    { ui_safe_env();
      double xs = g_ui_xsafe_l;   // explicit override, in clip units (1.0 = half the picture width)
      if (xs == 0 && g_ui_xsafe_auto) { double extra = tw - (16.0 / 9.0) * th; xs = extra > 0 ? 2 * extra / tw : 0; }
      if (g_ui_safe_only && (!pr->sh[0] || strncmp(pr->sh[0]->key, g_ui_safe_only, strlen(g_ui_safe_only)))) xs = 0;
      if (xs != 0) { float nx0, nx1, ny0, ny1; int zf = 0;
        int rc = ui_draw_ndc(dw, di, &nx0, &nx1, &ny0, &ny1, &zf);
        if (rc == 1) { int dir = zf ? ui_safe_dir(nx0, nx1) : 0; if (dir > 0) mxoff = xs; }
        if (g_ui_xsafe_log) {   // DM_UI_SAFELOG: per shader pair per pass, whether the bounds could be measured, and what moved
            static struct { char k[40]; int ok, no, mv; } cls[48]; static int nc; static uint64_t hp;
            char ck[40]; snprintf(ck, sizeof ck, "%.16s/%.16s", pr->sh[0] ? pr->sh[0]->key : "-", pr->sh[1] ? pr->sh[1]->key : "-");
            int i = 0; for (; i < nc; i++) if (!strcmp(cls[i].k, ck)) break;
            if (i == nc && nc < 48) { snprintf(cls[nc].k, sizeof cls[nc].k, "%s", ck); cls[nc].ok = cls[nc].no = cls[nc].mv = 0; nc++; }
            if (i < nc) { if (rc != 1) cls[i].no++; else { cls[i].ok++; if (mxoff != 0) cls[i].mv++; } }
            if (hp != g_stat_pass) {
                if (hp) for (int q = 0; q < nc; q++) if (cls[q].ok + cls[q].no)
                    MLOG("内缩判定 通道 %llu %s 可测 %d 不可测 %d 右移 %d", (unsigned long long)hp, cls[q].k, cls[q].ok, cls[q].no, cls[q].mv);
                memset(cls, 0, sizeof cls); nc = 0; hp = g_stat_pass; } }
        static int once; if (mxoff != 0 && !once++) MLOG("界面右移：画布 %lux%lu，每帧右移 %.4f（裁剪空间）＝ %.0f 像素（首例通道 %llu，x[%+.3f %+.3f]）",
            (unsigned long)tw, (unsigned long)th, xs, xs * tw / 2, g_stat_pass, nx0, nx1);
      } }
    if (moff != g_ui_moff || mxoff != g_ui_mxoff) { g_ui_moff = moff; g_ui_mxoff = mxoff; g_args_dirty[0] = 1; }
    { MTLViewport v = S.has_vp ? (MTLViewport){S.vp[0][0], S.vp[0][1] + vo, S.vp[0][2], S.vp[0][3] * vs, S.dr[0], S.dr[1]} : (MTLViewport){0, vo, tw, th * vs, 0, 1};
      { // DM_UI_VPXOFF=<pixels>: move the UI passes' viewport instead of their matrix -- the control that tells "the clip-space shift
        // does not reach the pixels" apart from "these draws are not the ones that draw what I am looking at"
        static int got = -1; static double vx; if (got < 0) { const char *e = getenv("DM_UI_VPXOFF"); got = e && *e; vx = got ? atof(e) : 0; if (got) MLOG("DM_UI_VPXOFF：UI 视口水平移动 %.0f 像素", vx); }
        if (got && ui_draw) v.originX += vx; }
      if (atomic_load_explicit(&g_origin_ll, memory_order_relaxed)) { v.originY += v.height; v.height = -v.height; }   // lower-left origin (see g_origin_ll)
      enc_r_viewport(en, v); }
    MTLScissorRect sc = {0, 0, tw, th};
    if (R.has_sc) {
        long x0 = R.sc.x < 0 ? 0 : R.sc.x, y0 = R.sc.y <= 0 ? 0 : R.sc.y + so, x1 = (long)R.sc.x + R.sc.w, y1 = vs != 1 ? (long)th : (long)R.sc.y + R.sc.h + so;
        if (x1 > (long)tw) x1 = tw; if (y1 > (long)th) y1 = th;
        if (x1 <= x0 || y1 <= y0) { PD("Z %llu 绘制跳过：剪裁为空（%d,%d %dx%d，目标 %lux%lu）", g_stat_pass, R.sc.x, R.sc.y, R.sc.w, R.sc.h, (unsigned long)tw, (unsigned long)th); return; }
        sc = (MTLScissorRect){(NSUInteger)x0, (NSUInteger)y0, (NSUInteger)(x1 - x0), (NSUInteger)(y1 - y0)};
    }
    enc_r_scissor(en, sc);
    if (S.poly.cull == NVN_FACE_FRONT_AND_BACK) { PD("Z %llu 绘制跳过：两面全剔", g_stat_pass); return; }
    enc_r_cull(en, S.poly.cull == NVN_FACE_FRONT ? MTLCullModeFront : S.poly.cull == NVN_FACE_BACK ? MTLCullModeBack : MTLCullModeNone);
    enc_r_winding(en, (S.poly.front == NVN_FRONT_FACE_CCW) != atomic_load_explicit(&g_origin_ll, memory_order_relaxed) ? MTLWindingCounterClockwise : MTLWindingClockwise);   // the y flip of lower-left mode reverses the winding
    enc_r_fill(en, S.poly.mode == NVN_POLYGON_MODE_LINE ? MTLTriangleFillModeLines : MTLTriangleFillModeFill);
    if (S.poly.offset_en & 4) enc_r_depth_bias(en, S.po.units, S.po.factor, S.po.clamp); else enc_r_depth_bias(en, 0, 0, 0);
    enc_r_blend_color(en, S.blend_color[0], S.blend_color[1], S.blend_color[2], S.blend_color[3]);
    for (int s_ = 0; s_ < 16; s_++) {
        if (!S.vbuf[s_].address) continue;
        void *mb; uint64_t mo;
        uint64_t gpu;
        if (mtl_gpu_addr(S.vbuf[s_].address, 1, &mb, &mo, &gpu)) { enc_r_vbuf(en, VB_SLOT(s_), gpu); use_rec("buf", S.vbuf[s_].address, S.vbuf[s_].size, 0, NULL); }
    }
    static id<MTLBuffer> zero; if (!zero) { zero = [g_dev newBufferWithLength:64 options:MTLResourceStorageModeShared]; res_add(zero); }
    static uint64_t zero_gpu; if (!zero_gpu) zero_gpu = zero.gpuAddress;
    enc_r_vbuf(en, 30, zero_gpu);
    g_kd_force = kd_draw(pr, dw, di);
    if (pr->sh[0]) encode_args(en, pr->sh[0], 0, 1);
    if (pr->sh[1]) encode_args(en, pr->sh[1], 1, 0);
    // Parameter binding captures the actual resources under the registry lock; if the version changed between the eligibility check and the binding, split and retry before any GPU draw.
    if (g_rt_range_active && (g_pass.reg_gen != g_reg_gen || g_pass.ser_gen != g_ser_gen || g_pass.range_dirty != nvn_dirty_now())) { rp_end(); draw(dw, di); return; }
    int prim = dw ? dw->prim : di->prim;
    MTLPrimitiveType mp;
    switch (prim) {
        case 0: mp = MTLPrimitiveTypePoint; break; case 1: mp = MTLPrimitiveTypeLine; break; case 3: mp = MTLPrimitiveTypeLineStrip; break;
        case 4: case 6: case 7: mp = MTLPrimitiveTypeTriangle; break; case 5: mp = MTLPrimitiveTypeTriangleStrip; break;   // 6 triangle fan, 7 quads: Metal has neither, expanded into triangle lists
        default: RED_ONCE(19, "图元类型 %d 未实现", prim); return;
    }
    if (g_vis_counting) {   // Occlusion query counting: a new slot for this render pass (also switch to a new slot after a report / reset)
        if (!g_enc_has_vis) RED_ONCE(34, "遮挡查询计数开始时所在通道没挂可见性缓冲（首次清零发生在通道中途），该通道的采样不计");
        else {
            if (g_vis_cur < 0 || g_vis_cur_pass != g_pass.serial) {
                if (g_vis_n >= VIS_SLOTS) RED_ONCE(35, "可见性槽用尽");
                else { g_vis_cur = (int)g_vis_n++; g_vis_cur_pass = g_pass.serial;
                    if (g_nvs == g_vscap) { g_vscap = g_vscap ? g_vscap * 2 : 64; g_vs = realloc(g_vs, g_vscap * sizeof *g_vs); }
                    g_vs[g_nvs++] = (vis_slot){(void *)CFBridgingRetain(g_vis), (uint32_t)g_vis_cur}; }
            }
            if (g_vis_cur >= 0 && g_enc_vis != g_vis_cur + 1) { enc_r_visibility(en, MTLVisibilityResultModeCounting, (NSUInteger)g_vis_cur * 8); g_enc_vis = g_vis_cur + 1; }
        }
    }
    g_stat_draw++;
    { static int tg; if (!tg) { tg = 1; const char *e = getenv("DM_TEST_GHOST"); float v[8];   // Test switch DM_TEST_GHOST=px,py,gx,gy,gw,gh,cx,cy (normalized; press point, grid, center of the drag-follow image): turns on the drag-follow image directly without going through the front end (for Mac host screenshot verification)
        if (e && sscanf(e, "%f,%f,%f,%f,%f,%f,%f,%f", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7]) == 8) mtl_ui_ghost(1, v[0], v[1], &v[2], v[6], v[7]); } }
    if (dw && atomic_load(&g_ghost_on)) ghost_note(dw);
    if (enc_label_on() && !g_el_draws++ && pr->sh[1]) { memcpy(g_el_key, pr->sh[1]->key, 8); g_el_key[8] = 0; }
    if (passdump()) {   // Characteristics of a render pass's first draw (to judge whether the load can be omitted: full screen, no blending, writes all channels, does not depend on depth); with DM_NVN_PASSDUMP_ALL every draw is recorded
        static uint64_t last_pass = ~0ull; static int all = -1; if (all < 0) all = getenv("DM_NVN_PASSDUMP_ALL") != NULL;
        if (all || g_kd_force) PD("d %llu bv=%d bi=%d first=%d vs=%s fs=%s vc=%d inst=%d blend=%#x b0=%#llx dt=%d dw=%d df=%d cull=%d st=%d sf=%d,%d sop=%d,%d,%d/%d,%d,%d smask=%#x,%#x/%#x,%#x sref=%#x,%#x ib=%#llx it=%d cmask=%#x b1=%#llx b2=%#llx b3=%#llx", g_stat_pass, di ? di->base_vertex : 0, di ? di->base_instance : 0, dw ? dw->first : 0, pr->sh[0] ? pr->sh[0]->key : "-", pr->sh[1] ? pr->sh[1]->key : "-",
                    dw ? dw->count : di->count, dw ? dw->instances : di->instances, S.color.blend_en, (unsigned long long)*(uint64_t *)&S.blend[0], (int)S.ds.dt, (int)S.ds.dw, (int)S.ds.df, (int)S.poly.cull,
                    (int)S.ds.st, (int)S.ds.sf_f, (int)S.ds.sf_b, (int)S.ds.fail_f, (int)S.ds.zfail_f, (int)S.ds.zpass_f,
                    (int)S.ds.fail_b, (int)S.ds.zfail_b, (int)S.ds.zpass_b, S.st_vmask[0], S.st_vmask[1], S.st_wmask[0], S.st_wmask[1], S.st_ref[0], S.st_ref[1],
                    di ? (unsigned long long)di->ibuf : 0ull, di ? di->itype : 0, S.chmask.mask, (unsigned long long)*(uint64_t *)&S.blend[1], (unsigned long long)*(uint64_t *)&S.blend[2], (unsigned long long)*(uint64_t *)&S.blend[3]);
        if (all && pr->sh[0]) {   // Content fingerprint of each constant buffer of the vertex stage (to judge whether a draw is completely identical across frames: used by the shadow cache etc.); when unreadable (a write pending execution in this submit) record P
            char ub[400]; size_t o = 0; NvShader *v = pr->sh[0];
            for (int i = 0; i < v->rf.nb && o < sizeof ub - 40; i++) if (v->rf.b[i].kind == NVN_RB_CBUF && v->rf.b[i].reg < 16) {
                const p_bind_buf *u = &S.ubo[0][v->rf.b[i].reg]; void *mb; uint64_t mo; const uint8_t *base; uint64_t h = 1469598103934665603ull;
                if (!u->addr || !u->size) { o += snprintf(ub + o, sizeof ub - o, " r%d=-", v->rf.b[i].reg); continue; }
                if (pend_hits(u->addr, u->size) || !mtl_addr(u->addr, u->size, &mb, &mo) || !(base = [(__bridge id<MTLBuffer>)mb contents])) { o += snprintf(ub + o, sizeof ub - o, " r%d=P/%llu", v->rf.b[i].reg, (unsigned long long)u->size); continue; }
                uint64_t h64 = 0; for (uint64_t q = 0; q < u->size; q++) { h = (h ^ base[mo + q]) * 1099511628211ull; if (q == 63) h64 = h; }   // h64: the first 64 bytes (world matrix vec0–3 of the object buffer)
                { static uint64_t hp = ~0ull; static int hn; if (hp != g_stat_pass) { hp = g_stat_pass; hn = 0; }   // For the first 6 draws of each render pass, additionally print buffers of <= 608 bytes in hex (to compare byte by byte which fields changed across frames)
                  if (hn < 12 && u->size <= 608 && v->rf.b[i].reg == 3) { hn++; char hx[1300]; for (uint64_t q = 0; q < u->size; q++) snprintf(hx + 2 * q, 3, "%02x", base[mo + q]); PD("x %llu r%d %s", g_stat_pass, v->rf.b[i].reg, hx); } }
                o += snprintf(ub + o, sizeof ub - o, " r%d=%016llx/%llu/%016llx", v->rf.b[i].reg, (unsigned long long)h, (unsigned long long)u->size, (unsigned long long)h64); }
            PD("u %llu%s", g_stat_pass, ub); }
        if (all) { char vb[1600]; size_t o = 0; o += snprintf(vb + o, sizeof vb - o, "nattr=%d", S.nattr);
            for (int i = 0; i < S.nattr && i < 16 && o < sizeof vb - 100; i++) { int st = S.attr[i].stream; if (!S.attr[i].fmt) continue;   // only attributes with a format
                o += snprintf(vb + o, sizeof vb - o, " a%d{fmt=%#x off=%d s=%d stride=%u div=%u vb=%#llx/%llu}", i, S.attr[i].fmt, S.attr[i].offset, st,
                              st < 16 ? S.stream[st].stride : 0, st < 16 ? S.stream[st].divisor : 0, st < 16 ? (unsigned long long)S.vbuf[st].address : 0ull, st < 16 ? (unsigned long long)S.vbuf[st].size : 0ull); }
            PD("v %llu %s", g_stat_pass, vb);
            // DM_NVN_CBDUMP + DM_NVN_CBDUMP_KEY=vertex shader key: also save that draw's vertex streams and indices (F<frame>_p<pass>_vb<stream>_<address>.bin / _ib_<address>.bin)
            const char *cbd = getenv("DM_NVN_CBDUMP"), *cbk = getenv("DM_NVN_CBDUMP_KEY");
            if (cbd && cbk && pr->sh[0] && !strcmp(pr->sh[0]->key, cbk)) {
                char fp[1024];
                for (int st = 0; st < 16; st++) if (S.vbuf[st].address && S.vbuf[st].size) {
                    snprintf(fp, sizeof fp, "%s/F%llu_p%llu_vb%d_%llx.bin", cbd, (unsigned long long)atomic_load(&g_cur_frame), g_stat_pass, st, (unsigned long long)S.vbuf[st].address);
                    FILE *f = fopen(fp, "wb"); if (f) { fwrite((const void *)S.vbuf[st].address, 1, S.vbuf[st].size, f); fclose(f); } }
                if (di && di->ibuf) {
                    snprintf(fp, sizeof fp, "%s/F%llu_p%llu_ib%d_%llx.bin", cbd, (unsigned long long)atomic_load(&g_cur_frame), g_stat_pass, di->itype, (unsigned long long)di->ibuf);
                    FILE *f = fopen(fp, "wb"); if (f) { fwrite((const void *)di->ibuf, 1, (size_t)di->count << (di->itype == 2 ? 2 : di->itype), f); fclose(f); } }
            } }
        if (last_pass != g_stat_pass) {
            last_pass = g_stat_pass;
            PD("D %llu prim=%d vc=%d inst=%d blend=%#x wm=%#x dt=%d dw=%d df=%d st=%d sc=%lu,%lu,%lu,%lu tgt=%lux%lu vp=%.0f,%.0f,%.0f,%.0f", g_stat_pass, prim,
               dw ? dw->count : di->count, dw ? dw->instances : di->instances, S.color.blend_en, S.chmask.mask, (int)S.ds.dt, (int)S.ds.dw, (int)S.ds.df, (int)S.ds.st,
               (unsigned long)sc.x, (unsigned long)sc.y, (unsigned long)sc.width, (unsigned long)sc.height, (unsigned long)tw, (unsigned long)th,
               S.has_vp ? S.vp[0][0] : 0, S.has_vp ? S.vp[0][1] : 0, S.has_vp ? S.vp[0][2] : (float)tw, S.has_vp ? S.vp[0][3] : (float)th);
        }
    }
    g_kd_force = 0;
    if (rt_range_on() && !rt_range_pr_ro(pr)) g_pass.range_external=1;
    rt_range_note_depth();
#ifdef NVN_RT_RANGE_TEST
    rr_snap(dw,di);
#endif
    udc_note_draw(pr);
    // A draw with 0 primitives (the game does send count=0) draws no pixels, so skip encoding; Metal requires the count to be non-zero, otherwise the validation layer aborts on the spot (measured on the camp / wilderness / combat pages)
    if (dw && prim == 7) {   // Quads: shared indices 0,1,2,0,2,3
        if (!g_quad_ib) {
            uint32_t n = 65536; uint32_t *ix = malloc(n * 6 * 4);
            for (uint32_t q = 0; q < n; q++) { uint32_t b = q * 4; uint32_t v[6] = {b, b + 1, b + 2, b, b + 2, b + 3}; memcpy(ix + q * 6, v, 24); }
            g_quad_ib = [g_dev newBufferWithBytes:ix length:n * 24 options:MTLResourceStorageModeShared]; g_quad_gpu = g_quad_ib.gpuAddress; res_add(g_quad_ib); free(ix);
        }
        if (dw->count >= 4) enc_r_draw_indexed(en, mp, (dw->count / 4) * 6, MTLIndexTypeUInt32, g_quad_gpu, g_quad_ib.length, dw->instances > 0 ? dw->instances : 1, dw->first, dw->base_instance);
    } else if (dw && prim == 6) {   // triangle fans (13 Sentinels): shared indices (0, i, i+1), first as the base vertex
        if (!g_fan_ib) {
            uint32_t n = 65536; uint32_t *ix = malloc(n * 12);
            for (uint32_t i = 0; i < n; i++) { uint32_t v[3] = {0, i + 1, i + 2}; memcpy(ix + i * 3, v, 12); }
            g_fan_ib = [g_dev newBufferWithBytes:ix length:n * 12 options:MTLResourceStorageModeShared]; g_fan_gpu = g_fan_ib.gpuAddress; res_add(g_fan_ib); free(ix);
        }
        if (dw->count >= 3) enc_r_draw_indexed(en, mp, (dw->count - 2) * 3, MTLIndexTypeUInt32, g_fan_gpu, g_fan_ib.length, dw->instances > 0 ? dw->instances : 1, dw->first, dw->base_instance);
    } else if (dw) {
        if (dw->count && (dw->instances > 1 || dw->base_instance)) enc_r_draw_instanced(en, mp, dw->first, dw->count, dw->instances > 0 ? dw->instances : 1, dw->base_instance);
        else if (dw->count) enc_r_draw(en, mp, dw->first, dw->count);
    } else {
        if (prim == 6 || prim == 7) {
                                        // indexed triangle fans / quads (13 Sentinels): at replay the indices are read from guest memory and expanded into a 32-bit triangle list in this frame's arena (as for 8-bit indices)
            int isz = di->itype == 0 ? 1 : di->itype == 1 ? 2 : 4;
            uint32_t nt = prim == 7 ? (uint32_t)(di->count / 4) * 2 : di->count >= 3 ? (uint32_t)di->count - 2 : 0;
            if (!nt) return;
            void *mb; uint64_t mo; const uint8_t *src;
            if (!mtl_addr(di->ibuf, (uint64_t)di->count * isz, &mb, &mo) || !(src = [(__bridge id<MTLBuffer>)mb contents])) { RED_ONCE(22, "索引缓冲不在内存池里"); return; }
            if (pend_hits(di->ibuf, (uint64_t)di->count * isz)) RED_ONCE(73, "三角扇 / 四边形索引缓冲本提交里有 GPU 侧写入：按当前内容展开");
            src += mo;
            #define IX(i) (isz == 1 ? (uint32_t)src[i] : isz == 2 ? (uint32_t)((const uint16_t *)src)[i] : ((const uint32_t *)src)[i])
            id<MTLBuffer> ab; size_t off = arena_alloc(((size_t)nt * 12 + 15) & ~(size_t)15, &ab);
            uint32_t *d = (uint32_t *)((uint8_t *)ab.contents + off);
            for (uint32_t t = 0; t < nt; t++) {
                if (prim == 6) { d[3 * t] = IX(0); d[3 * t + 1] = IX(t + 1); d[3 * t + 2] = IX(t + 2); }
                else { uint32_t q = (t >> 1) * 4, h = t & 1; d[3 * t] = IX(q); d[3 * t + 1] = IX(q + 1 + h); d[3 * t + 2] = IX(q + 2 + h); }
            }
            #undef IX
            use_rec("buf", di->ibuf, (uint64_t)di->count * isz, 0, NULL);
            enc_r_draw_indexed(en, mp, nt * 3, MTLIndexTypeUInt32, g_cur_arena->gpu + off, (NSUInteger)nt * 12, di->instances ? di->instances : 1, di->base_vertex, di->base_instance);
            if (g_rt_range_pending) { PD("j rt_level_range pass=%llu vc=%u", g_stat_pass, (unsigned)di->count); g_rt_range_pending = 0; }
            return;
        }
        if (di->itype == 0) {
                                // 8-bit indices: Metal only has 16 / 32-bit. At replay they are read from guest memory and widened to 16-bit in this frame's arena (SMBW's full-screen passes and the final copy into the window all use 8-bit indices; without this the whole screen is black)
            if (!di->count) return;
            void *mb; uint64_t mo; const uint8_t *src;
            if (!mtl_addr(di->ibuf, di->count, &mb, &mo) || !(src = [(__bridge id<MTLBuffer>)mb contents])) { RED_ONCE(22, "索引缓冲不在内存池里"); return; }
            if (pend_hits(di->ibuf, di->count)) RED_ONCE(48, "8 位索引缓冲本提交里有 GPU 侧写入：按当前内容展开");
            id<MTLBuffer> ab; size_t off = arena_alloc(((size_t)di->count * 2 + 15) & ~(size_t)15, &ab);
            uint16_t *d = (uint16_t *)((uint8_t *)ab.contents + off); src += mo;
            for (int32_t i = 0; i < di->count; i++) d[i] = src[i];
            use_rec("buf", di->ibuf, (uint64_t)di->count, 0, NULL);
            enc_r_draw_indexed(en, mp, di->count, MTLIndexTypeUInt16, g_cur_arena->gpu + off, (NSUInteger)di->count * 2, di->instances ? di->instances : 1, di->base_vertex, di->base_instance);
            if (g_rt_range_pending) { PD("j rt_level_range pass=%llu vc=%u", g_stat_pass, (unsigned)di->count); g_rt_range_pending = 0; }
            return;
        }
        void *mb; uint64_t mo;
        uint64_t gpu;
        if (!mtl_gpu_addr(di->ibuf, 1, &mb, &mo, &gpu)) { RED_ONCE(22, "索引缓冲不在内存池里"); return; }
        use_rec("buf", di->ibuf, (uint64_t)di->count * (di->itype == 1 ? 2 : 4), 0, NULL);
        if (di->count) enc_r_draw_indexed(en, mp, di->count, di->itype == 1 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32, gpu, ((__bridge id<MTLBuffer>)mb).length - mo,
                di->instances ? di->instances : 1, di->base_vertex, di->base_instance);
    }
    if (g_rt_range_pending) { PD("j rt_level_range pass=%llu vc=%u", g_stat_pass, (unsigned)(dw ? dw->count : di->count)); g_rt_range_pending=0; }
}
static nvn_hmap g_cps = {.klen = sizeof(void *)};
static void warm_claim(void) {
    if (!atomic_exchange_explicit(&g_warm_ready, 0, memory_order_acq_rel)) return;
    os_unfair_lock_lock(&g_warm_lock);
    NSArray<WarmItem *> *items = [g_warm_items copy];
    [g_warm_items removeAllObjects];
    os_unfair_lock_unlock(&g_warm_lock);
    for (WarmItem *item in items) {
        if (item->kind == 0) {
            if (!g_shcache) g_shcache = [NSMutableDictionary dictionary];
            if (!g_shcache[item->key]) { g_shcache[item->key] = item->shader; g_warm_claim_shaders++; }
        } else if (item->kind == 1) {
            if (!g_shcache) g_shcache = [NSMutableDictionary dictionary];
            NvShader *vs = g_shcache[item->vs_key], *fs = item->fs_key.length ? g_shcache[item->fs_key] : nil;
            if (!vs) { vs = item->vs; g_shcache[item->vs_key] = vs; }
            if (item->fs_key.length && !fs) { fs = item->fs; g_shcache[item->fs_key] = fs; }
            item->pso.vs = (__bridge void *)vs; item->pso.fs = (__bridge void *)fs;
            if (!hm_get_id(&g_pso, &item->pso)) { hm_put_id(&g_pso, &item->pso, item->object); g_warm_claim_rps++; }
        } else {
            NvShader *cs = g_shcache[item->key];
            if (!cs) { cs = item->shader; if (!g_shcache) g_shcache = [NSMutableDictionary dictionary]; g_shcache[item->key] = cs; }
            const void *key = (__bridge const void *)cs;
            if (!hm_get_id(&g_cps, &key)) { hm_put_id(&g_cps, &key, item->object); g_warm_claim_cps++; }
        }
    }
}
// ---------- 3D picture brightening ----------
// Tone mapping (fragment a1b1aab66933fa8f) = PQ-encode the composited HDR x0.008, then look up a 32^3 color lookup table (recomputed per region every frame by compute shader 552f45c348a61b86, read only by tone mapping).
// Multiplying HDR by exposure k first is equivalent to replacing the table with L'(u) = L(PQ(k * PQ^-1(u))): after the dispatch in which the game finishes computing the table, make a copy and rewrite it in place accordingly (one small 32^3 dispatch); the table is recomputed every frame, so it does not get brighter with each multiplication.
// Only the 3D world is touched: UI, text and the health orbs are composited after tone mapping; highlights are compressed by PQ and do not blur into a sheet of white as a final whole-screen multiply would. The normal / readable tile merge / freeze paths all just read this table, so it takes effect in one place.
// k is given every frame by the host's dm_bright_k() (dm/host/dm_bright.c: 1 out of game, 4 for maze-type areas, 4 where the original game's picture does not follow day and night, the rest derived from the original game's ambient brightness, with smooth transitions); no rewrite when k ~= 1.
__attribute__((weak)) float dm_bright_k(void) { return 1.0f; }   // The host's dm_bright.c provides the strong definition; no brightening when graphics unit tests do not link the host
static const char k_lut_msl[] =
"#include <metal_stdlib>\nusing namespace metal;\n"
"constant float m1 = 0.1593017578125, m2 = 78.84375, c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;\n"   // Bit-identical to the PQ constants in the tone mapping shader
"static float3 pq(float3 v) { float3 p = pow(abs(v), m1); return pow((c1 + c2 * p) / (1.0 + c3 * p), m2); }\n"
"static float3 pq_inv(float3 u) { float3 q = pow(u, 1.0 / m2); return pow(max(q - c1, 0.0) / max(c2 - c3 * q, 1e-6), 1.0 / m1); }\n"
"kernel void lutk(texture3d<float> src [[texture(0)]], texture3d<float, access::write> dst [[texture(1)]], constant float &k [[buffer(0)]], uint3 g [[thread_position_in_grid]]) {\n"
"  uint3 n = uint3(dst.get_width(), dst.get_height(), dst.get_depth()); if (any(g >= n)) return;\n"
"  constexpr sampler ls(filter::linear, address::clamp_to_edge, coord::normalized);\n"
"  float3 u = (float3(g) + 0.5) / float3(n); float3 u2 = clamp(pq(k * pq_inv(u)), 0.0, 1.0);\n"
"  float4 b = src.sample(ls, u2), o = src.read(g);\n"   // b: output after brightening by exposure k; o: this cell in the original game
"  float s = smoothstep(0.65, 1.0, dot(b.rgb, float3(0.2126, 0.7152, 0.0722)));\n"   // Highlight protection: after brightening, the closer to white, the more it is pulled back to the original value (snow and skylight do not blur into a sheet of white); in dark areas s = 0 and brightening proceeds as usual
"  dst.write(float4(max(mix(b.rgb, o.rgb, s), o.rgb), b.a), g); }\n";
static void bright_lut(NvShader *cs) {
    static int failed; static id<MTLComputePipelineState> ps; static id<MTL4ArgumentTable> at; static id<MTLTexture> tmp; static id<MTLBuffer> kb[3]; static int ki;
    if (failed || strncmp(cs->key, "552f45c348a61b86", 16)) return;
    float k = dm_bright_k(); if (!(k > 1.005f)) return;   // k ~= 1 (out of game / noon): the table stays as in the original game
    if (k > 4.0f) k = 4.0f;
    int tid = 0; for (int i = 0; i < cs->rf.nb && !tid; i++) if (cs->rf.b[i].kind == NVN_RB_IMG && cs->rf.b[i].reg < 32) tid = (int)(S.img[5][cs->rf.b[i].reg] & 0xfffff);
    pthread_mutex_lock(&g_rmu); id<MTLTexture> lut = tid ? tex_by_id(tid) : nil; pthread_mutex_unlock(&g_rmu);
    if (!lut || lut.textureType != MTLTextureType3D || !(lut.usage & MTLTextureUsageShaderWrite)) { static int n; if (!n++) MLOG("【红】3D 画面加亮：调色表认不出（贴图号 %d），不加亮", tid); return; }
    if (!ps) { NSError *e = nil; id<MTLLibrary> lib = [g_dev newLibraryWithSource:@(k_lut_msl) options:nil error:&e]; id<MTLFunction> f = lib ? [lib newFunctionWithName:@"lutk"] : nil;
        ps = f ? [g_dev newComputePipelineStateWithFunction:f error:&e] : nil;
        MTL4ArgumentTableDescriptor *ad = [MTL4ArgumentTableDescriptor new]; ad.maxTextureBindCount = 2; ad.maxBufferBindCount = 1; at = [g_dev newArgumentTableWithDescriptor:ad error:&e];
        for (int i = 0; i < 3; i++) { kb[i] = [g_dev newBufferWithLength:16 options:MTLResourceStorageModeShared]; if (kb[i]) res_add(kb[i]); }
        if (!ps || !at || !kb[0] || !kb[1] || !kb[2]) { failed = 1; MLOG("【红】3D 画面加亮：计算管线建不出 %s（退回原版亮度）", e.localizedDescription.UTF8String); return; }
        MLOG("3D 画面加亮：建好（调色表 %lux%lux%lu，曝光每帧由 dm_bright_k 给）", (unsigned long)lut.width, (unsigned long)lut.height, (unsigned long)lut.depth); }
    if (!tmp || tmp.width != lut.width || tmp.height != lut.height || tmp.depth != lut.depth || tmp.pixelFormat != lut.pixelFormat) {
        MTLTextureDescriptor *d = [MTLTextureDescriptor new]; d.textureType = MTLTextureType3D; d.pixelFormat = lut.pixelFormat; d.width = lut.width; d.height = lut.height; d.depth = lut.depth;
        d.usage = MTLTextureUsageShaderRead; d.storageMode = MTLStorageModePrivate; if (tmp) defer_obj_release(tmp); tmp = [g_dev newTextureWithDescriptor:d]; if (!tmp) { failed = 1; return; } res_add(tmp); }
    id<MTLBuffer> b0 = kb[ki]; ki = (ki + 1) % 3; *(float *)b0.contents = k;   // Three copies in rotation: in-flight frames are still reading the previous copy
    id<MTL4ComputeCommandEncoder> b = blit(); enc_b_copy_texture(b, lut, 0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(lut.width, lut.height, lut.depth), tmp, 0, 0, MTLOriginMake(0, 0, 0)); enc_b_end(g_blit); g_blit = nil;
    id<MTL4ComputeCommandEncoder> ce = (id<MTL4ComputeCommandEncoder>)enc_c_begin(NULL); [ce setArgumentTable:at]; [at setTexture:tmp.gpuResourceID atIndex:0]; [at setTexture:lut.gpuResourceID atIndex:1]; [at setAddress:b0.gpuAddress atIndex:0];
    enc_c_pso(ce, ps); enc_c_dispatch(ce, MTLSizeMake((lut.width + 3) / 4, (lut.height + 3) / 4, (lut.depth + 3) / 4), MTLSizeMake(4, 4, 4)); enc_c_end(ce);
    args_dirty_all();
}
static void dispatch(const p_dispatch *dp) {
    zskip_clear();
    if (!S.prog) { RED_ONCE(27, "派发前没有绑定程序"); return; }
    NvShader *cs = prog_get(S.prog)->sh[5];
    if (!cs) { RED_ONCE(28, "派发的程序没有计算着色器（或未映射）"); return; }
    const void *k = (__bridge const void *)cs;
    id<MTLComputePipelineState> ps = hm_get_id(&g_cps, &k);
    if (!ps) {
        NSError *e = nil; uint64_t bt = bp_now(); ps = [g_dev newComputePipelineStateWithFunction:cs->fn error:&e];
        g_bp_pso_n++; g_bp_pso_ns += bp_now() - bt;
        if (!ps) { MLOG("【红】建计算管线失败：%s", e.localizedDescription.UTF8String); return; }
        pso_record("C\t%s\n", cs->key);
        if (cs->key[0]) pso_user_record("C\t%s\n", cs->key);
        hm_put_id(&g_cps, &k, ps);
    }
    if (cs->builtin) {   // Built-in compute kernel: n = uniform buffer 0, src = storage buffer 0, dst = storage buffer 1
        const p_bind_buf *u = &S.ubo[5][0], *a0 = &S.ssbo[5][0], *a1 = &S.ssbo[5][1];
        void *ub, *sb, *db; uint64_t uo, so, dof;
        if (!mtl_addr(u->addr, 4, &ub, &uo) || !mtl_addr(a0->addr, 8, &sb, &so) || !mtl_addr(a1->addr, 8, &db, &dof)) { RED_ONCE(30, "内嵌计算核 %s 的缓冲没绑全", cs->builtin); return; }
        if (!cs->logged++) {
            module_t *m = module_of((void *)dp->caller);
            MLOG("内嵌计算核 %s 首次派发：调用点 %s+%#llx，线程组 %dx%dx%d，元素数（CPU 侧当前值）%u", cs->builtin, m ? m->name : "?",
                 m ? (unsigned long long)(dp->caller - (uint64_t)m->base) : (unsigned long long)dp->caller, dp->x, dp->y, dp->z, *(uint32_t *)u->addr);
        }
        rp_end(); if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
        id<MTLComputeCommandEncoder> ce = pg_compute();
        enc_c_pso(ce, ps);
        enc_c_args(ce, ((__bridge id<MTLBuffer>)ub).gpuAddress + uo, 0); enc_c_args(ce, ((__bridge id<MTLBuffer>)sb).gpuAddress + so, 1); enc_c_args(ce, ((__bridge id<MTLBuffer>)db).gpuAddress + dof, 2);   // Built-in compute kernel: cold path, take the addresses directly
        enc_c_dispatch(ce, MTLSizeMake(dp->x, dp->y, dp->z), MTLSizeMake(32, 1, 1));
        enc_c_end(ce);
        return;
    }
    const uint16_t *tg = cs->rf.tg;
    if (!tg[0]) { RED_ONCE(29, "计算着色器缺线程组大小反射"); return; }
    rp_end(); if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
    id<MTLComputeCommandEncoder> ce = pg_compute();
    enc_c_pso(ce, ps);
    encode_args(ce, cs, 5, 0);
    enc_c_dispatch(ce, MTLSizeMake(dp->x, dp->y, dp->z), MTLSizeMake(tg[0], tg[1], tg[2]));
    enc_c_end(ce);
    bright_lut(cs);   // 3D picture brightening: rewrite after the game finishes computing the color lookup table
}
// GPU: texture -> buffer (row stride taken from SetCopyRowStride, layer stride from SetCopyImageStride, 0 = tightly packed)
static void copy_t2b(const p_copy_b2t *c) {
    rp_flush();   // 读之前先把未落地的纯 clear 落成真写入，和 copy_t2t 一致
    if (c->tex) xv_check(c->tex->serial, c->has_view && c->view.set_levels ? c->view.base_level : 0, c->has_view && c->view.set_layers ? c->view.min_layer : 0, "贴图读缓冲");
    id<MTLTexture> t = tex_mtl(c->tex);
    if (!t) return;
    int bw, bh, bpb = nvn_fmt_bpb(c->tex->d.format, &bw, &bh);
    unsigned level = c->has_view && c->view.set_levels ? c->view.base_level : 0, layer0 = c->has_view && c->view.set_layers ? c->view.min_layer : 0;
    size_t row = S.copy_stride ? (size_t)S.copy_stride : (size_t)((c->r.width + bw - 1) / bw) * bpb;
    size_t rows = (c->r.height + bh - 1) / bh, img = S.copy_img_stride ? (size_t)S.copy_img_stride : row * rows;   // per layer: SetCopyImageStride, else tightly packed
    int is3d = c->tex->d.target == 2, nz = c->r.depth ? c->r.depth : 1;
    void *mb; uint64_t mo;
    if (!mtl_addr(c->src, img * nz, &mb, &mo)) { RED_ONCE(31, "贴图→缓冲拷贝的目标不在内存池里"); return; }
    id<MTL4ComputeCommandEncoder> b = blit();
    if (is3d)
        enc_b_copy_texture_buffer(b, t, 0, level, MTLOriginMake(c->r.xoffset, c->r.yoffset, c->r.zoffset), MTLSizeMake(c->r.width, c->r.height, nz), (__bridge id<MTLBuffer>)mb, mo, row, img);
    else
        for (int z = 0; z < nz; z++)
            enc_b_copy_texture_buffer(b, t, layer0 + c->r.zoffset + z, level, MTLOriginMake(c->r.xoffset, c->r.yoffset, 0), MTLSizeMake(c->r.width, c->r.height, 1), (__bridge id<MTLBuffer>)mb, mo + z * img, row, img);
}
// CPU texture read (ReadTexels): for packaged textures the pool memory is the ground truth, de-layout and read directly; the rest are read back from the MTLTexture by blit (the caller has already waited for the GPU per NVN rules)
void mtl_read_texels(const tex_t *t, unsigned level, unsigned layer0, const NVNcopyRegion *r, uint8_t *dst, size_t row, size_t img) {
    xv_check(t->serial, level, layer0, "CPU读回");
    PD("U cpu_read ser=%llu lv%u sl%u xyz=%d,%d,%d size=%d,%d,%d dst=%p row=%lu img=%lu", t->serial, level, layer0, r->xoffset, r->yoffset, r->zoffset, r->width, r->height, r->depth, dst, (unsigned long)row, (unsigned long)img);
    int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh);
    size_t wb = (size_t)((r->width + bw - 1) / bw) * bpb, hb = (r->height + bh - 1) / bh, x0 = (size_t)(r->xoffset / bw) * bpb, y0 = r->yoffset / bh;
    int nz = r->depth ? r->depth : 1, is3d = t->d.target == 2;
    if (t->has_layout && t->mem && !rc_src_reclaimed((uintptr_t)t->mem, t->L.size)) {   // T5: a reclaimed source is read back from the private copy
        uint32_t lw = MAX(t->d.width >> level, 1u), lh = MAX(t->d.height >> level, 1u), lz = is3d ? MAX(t->d.depth >> level, 1u) : 1;
        size_t lp = (size_t)((lw + bw - 1) / bw) * bpb, lr = (lh + bh - 1) / bh;
        uint8_t *lin = malloc(lp * lr * lz);
        for (int z = 0; z < nz; z++) {
            unsigned layer = is3d ? 0 : layer0 + r->zoffset + z;
            if ((is3d && z == 0) || !is3d) if (nvn_deswizzle(&t->d, level, layer, t->mem, lin, lp)) { RED_ONCE(32, "ReadTexels 解排布失败"); free(lin); return; }
            const uint8_t *src = lin + (is3d ? (size_t)(r->zoffset + z) * lp * lr : 0);
            for (size_t y = 0; y < hb; y++) memcpy(dst + z * img + y * row, src + (y0 + y) * lp + x0, wb);
        }
        free(lin);
        return;
    }
    id<MTLTexture> mt = tex_mtl(t);
    if (!mt) { memset(dst, 0, img * nz); return; }
    mtl_finish_gpu();
    id<MTLBuffer> buf = [g_dev newBufferWithLength:wb * hb * nz options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = q_cb(); id<MTLBlitCommandEncoder> b = enc_b_begin(cb);
    for (int z = 0; z < nz; z++)
        [b copyFromTexture:mt sourceSlice:is3d ? 0 : layer0 + r->zoffset + z sourceLevel:level sourceOrigin:MTLOriginMake(r->xoffset, r->yoffset, is3d ? r->zoffset + z : 0)
                sourceSize:MTLSizeMake(r->width, r->height, 1) toBuffer:buf destinationOffset:z * wb * hb destinationBytesPerRow:wb destinationBytesPerImage:wb * hb];
    enc_b_end(b); [cb commit]; [cb waitUntilCompleted];
    for (int z = 0; z < nz; z++) for (size_t y = 0; y < hb; y++) memcpy(dst + z * img + y * row, (uint8_t *)buf.contents + z * wb * hb + y * wb, wb);
}
static void copy_b2t(const p_copy_b2t *c) {
    if (c->tex) zskip_del(c->tex->serial);
    rc_write_use(c->tex ? c->tex->serial : 0);   // T5: buffer->texture destination
    id<MTLTexture> t = tex_mtl(c->tex);
    PD("C ser=%llu", c->tex->serial);
    if (!t) return;
    int bw, bh, bpb = nvn_fmt_bpb(c->tex->d.format, &bw, &bh);
    unsigned level = c->has_view && c->view.set_levels ? c->view.base_level : 0, layer0 = c->has_view && c->view.set_layers ? c->view.min_layer : 0;
    size_t row = S.copy_stride ? (size_t)S.copy_stride : (size_t)((c->r.width + bw - 1) / bw) * bpb;
    size_t rows = (c->r.height + bh - 1) / bh, img = S.copy_img_stride ? (size_t)S.copy_img_stride : row * rows;   // per layer: SetCopyImageStride, else tightly packed
    int is3d = c->tex->d.target == 2, nz = c->r.depth ? c->r.depth : 1;
    void *mb; uint64_t mo;
    if (!mtl_addr(c->src, img * nz, &mb, &mo)) { RED_ONCE(23, "缓冲→贴图拷贝的源不在内存池里"); return; }
    uint32_t upf; int ubpp;
    if (bc_unpack(c->tex->d.format, &upf, &ubpp)) {   // No BC: the compressed blocks in the source are decompressed on the CPU into a temporary buffer, then copied pixel-wise into the decompressed-format texture
        const uint8_t *src = [(__bridge id<MTLBuffer>)mb contents];
        if (!src) { RED_ONCE(46, "无 BC 解压：缓冲→贴图拷贝的源缓冲 CPU 读不到"); return; }
        size_t up = (size_t)c->r.width * ubpp, uimg = up * c->r.height;
        id<MTLBuffer> tb = [g_dev newBufferWithLength:uimg * nz ?: 16 options:MTLResourceStorageModeShared];
        for (int z = 0; z < nz; z++) nvn_bc_decode(c->tex->d.format, src + mo + z * img, row, c->r.width, c->r.height, (uint8_t *)tb.contents + z * uimg, up);
        res_add(tb);
        id<MTL4ComputeCommandEncoder> b = blit();
        if (is3d) enc_b_copy_buffer_texture(b, tb, 0, up, uimg, MTLSizeMake(c->r.width, c->r.height, nz), t, 0, level, MTLOriginMake(c->r.xoffset, c->r.yoffset, c->r.zoffset));
        else for (int z = 0; z < nz; z++) enc_b_copy_buffer_texture(b, tb, z * uimg, up, uimg, MTLSizeMake(c->r.width, c->r.height, 1), t, layer0 + c->r.zoffset + z, level, MTLOriginMake(c->r.xoffset, c->r.yoffset, 0));
        defer_obj_release(tb);   // Release after this submit has finished running (and remove from residency)
        return;
    }
    id<MTL4ComputeCommandEncoder> b = blit();
    if (is3d)
        enc_b_copy_buffer_texture(b, (__bridge id<MTLBuffer>)mb, mo, row, img, MTLSizeMake(c->r.width, c->r.height, nz), t, 0, level, MTLOriginMake(c->r.xoffset, c->r.yoffset, c->r.zoffset));
    else
        for (int z = 0; z < nz; z++)
            enc_b_copy_buffer_texture(b, (__bridge id<MTLBuffer>)mb, mo + z * img, row, img, MTLSizeMake(c->r.width, c->r.height, 1), t, layer0 + c->r.zoffset + z, level, MTLOriginMake(c->r.xoffset, c->r.yoffset, 0));
}

// No BC + Legacy Device Pack: the game copies textures from files (already ASTC 4x4) whole into a BC3 texture array it creates itself (created as RGBA8 when there is no BC). On Switch the two formats have the same block size and
// the copy moves raw blocks; here it is changed to a compute kernel that decodes pixel by pixel and writes them in, with sRGB sources first read out as linear values and then encoded back to sRGB values, so the result = the values read in the destination format after moving the blocks as-is
static const char k_a2u_msl[] =
"#include <metal_stdlib>\nusing namespace metal;\n"
"static float3 enc(float3 c) { return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, c * 12.92, c <= 0.0031308); }\n"
"kernel void a2u(texture2d<float> src [[texture(0)]], texture2d<float, access::write> dst [[texture(1)]], constant uint4 *p [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {\n"
"  if (g.x >= p[1].x || g.y >= p[1].y) return; float4 v = src.read(g + p[0].xy); if (p[1].z) v.rgb = enc(v.rgb); dst.write(v, g + p[0].zw); }\n";
static int copy_t2t_astc(id<MTLTexture> s, id<MTLTexture> d, unsigned sl, unsigned ss, unsigned dl, unsigned ds, int sx, int sy, int dx, int dy, int w, int h) {
    static int failed; static id<MTLComputePipelineState> ps; static id<MTL4ArgumentTable> at;
    if (failed) return 0;
    if (!ps) { NSError *e = nil; id<MTLLibrary> lib = [g_dev newLibraryWithSource:@(k_a2u_msl) options:nil error:&e]; id<MTLFunction> f = lib ? [lib newFunctionWithName:@"a2u"] : nil;
        ps = f ? [g_dev newComputePipelineStateWithFunction:f error:&e] : nil;
        MTL4ArgumentTableDescriptor *ad = [MTL4ArgumentTableDescriptor new]; ad.maxTextureBindCount = 2; ad.maxBufferBindCount = 1; at = [g_dev newArgumentTableWithDescriptor:ad error:&e];
        if (!ps || !at) { failed = 1; MLOG("【红】无 BC：ASTC → 解压格式拷贝的计算管线建不出 %s", e.localizedDescription.UTF8String); return 0; } }
    MTLTextureType tt = MTLTextureType2D;
    id<MTLTexture> sv = [s newTextureViewWithPixelFormat:s.pixelFormat textureType:tt levels:NSMakeRange(sl, 1) slices:NSMakeRange(ss, 1)];
    id<MTLTexture> dv = [d newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm textureType:tt levels:NSMakeRange(dl, 1) slices:NSMakeRange(ds, 1)];
    id<MTLBuffer> pb = [g_dev newBufferWithLength:32 options:MTLResourceStorageModeShared];
    if (!sv || !dv || !pb) { RED_ONCE(47, "无 BC：ASTC → 解压格式拷贝建不出视图 / 参数块"); return 0; }
    uint32_t *p = pb.contents; p[0] = sx; p[1] = sy; p[2] = dx; p[3] = dy; p[4] = w; p[5] = h; p[6] = s.pixelFormat == MTLPixelFormatASTC_4x4_sRGB; p[7] = 0;
    res_add(pb);
    rp_end(); if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
    id<MTL4ComputeCommandEncoder> ce = (id<MTL4ComputeCommandEncoder>)enc_c_begin(NULL); [ce setArgumentTable:at];
    [at setTexture:sv.gpuResourceID atIndex:0]; [at setTexture:dv.gpuResourceID atIndex:1]; [at setAddress:pb.gpuAddress atIndex:0];
    enc_c_pso(ce, ps); enc_c_dispatch(ce, MTLSizeMake((w + 7) / 8, (h + 7) / 8, 1), MTLSizeMake(8, 8, 1)); enc_c_end(ce);
    args_dirty_all();
    defer_obj_unref(sv); defer_obj_unref(dv); defer_obj_release(pb);   // Release after this submit has finished running (views are not made resident separately; the parameter block is removed from residency)
    return 1;
}
// Texture copies a blit cannot do -- a format change, a scaled region or a mirrored one (MIRROR_X / MIRROR_Y) (Link's Awakening converts its 1408x792 RG11B10F scene to RGBA16F every frame; Super Mario 3D World scales 1920x1080 to 1280x720
// when leaving the title): one render pass per layer that draws the source region into the destination region, nearest or (LINEAR_FILTER) bilinear. Float, unorm and sRGB formats only, 2D / 2D array,
// destinations that can be render targets; anything else returns 0 and keeps the red line. Depth (Depth32Float / Depth32Float_Stencil8 into the same format) is drawn by writing depth from the fragment shader,
// always nearest; stencil is not carried over (Link's Awakening halves its 1408x792 depth-stencil buffer to 704x396)
static int fmt_int_or_ds(MTLPixelFormat f) {
    switch (f) {
        case MTLPixelFormatR8Uint: case MTLPixelFormatR8Sint: case MTLPixelFormatR16Uint: case MTLPixelFormatR16Sint: case MTLPixelFormatRG8Uint: case MTLPixelFormatRG8Sint:
        case MTLPixelFormatR32Uint: case MTLPixelFormatR32Sint: case MTLPixelFormatRG16Uint: case MTLPixelFormatRG16Sint: case MTLPixelFormatRGBA8Uint: case MTLPixelFormatRGBA8Sint:
        case MTLPixelFormatRGB10A2Uint: case MTLPixelFormatRG32Uint: case MTLPixelFormatRG32Sint: case MTLPixelFormatRGBA16Uint: case MTLPixelFormatRGBA16Sint:
        case MTLPixelFormatRGBA32Uint: case MTLPixelFormatRGBA32Sint: case MTLPixelFormatDepth16Unorm: case MTLPixelFormatDepth32Float: case MTLPixelFormatStencil8:
        case MTLPixelFormatDepth32Float_Stencil8: case MTLPixelFormatX32_Stencil8: return 1;
        default: return 0;
    }
}
static int copy_t2t_draw(id<MTLTexture> s, id<MTLTexture> d, unsigned sl, unsigned ss, unsigned dl, unsigned ds, const NVNcopyRegion *sr, const NVNcopyRegion *dr, int linear, int mirror) {
    static NSMutableDictionary<NSNumber *, id<MTLRenderPipelineState>> *pso; static id<MTLFunction> vf, ff, fd; static id<MTLSamplerState> samp[2]; static id<MTL4ArgumentTable> at; static id<MTLDepthStencilState> dss; static int failed;
    if (failed || sr->width <= 0 || sr->height <= 0 || dr->width <= 0 || dr->height <= 0) return 0;
    int depth = (s.pixelFormat == MTLPixelFormatDepth32Float || s.pixelFormat == MTLPixelFormatDepth32Float_Stencil8) && d.pixelFormat == s.pixelFormat, stencil = depth && d.pixelFormat == MTLPixelFormatDepth32Float_Stencil8;
    if (!depth && (fmt_int_or_ds(s.pixelFormat) || fmt_int_or_ds(d.pixelFormat))) return 0;
    if (mtl_fmt_block(d.pixelFormat) || !(d.usage & MTLTextureUsageRenderTarget) || s.sampleCount > 1 || d.sampleCount > 1) return 0;
    if (!pso) {
        static const char *src =
            "#include <metal_stdlib>\nusing namespace metal;\n"
            "struct V { float4 p [[position]]; };\n"
            "vertex V vs(uint id [[vertex_id]]) { float2 uv = float2((id << 1) & 2, id & 2); V o; o.p = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1); return o; }\n"
            // p[0] = source origin (texels) + source texels per destination texel, p[1] = destination origin + source level size
            "fragment float4 fs(V in [[stage_in]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]], constant float4 *p [[buffer(0)]]) {\n"
            "  float2 st = p[0].xy + (in.p.xy - p[1].xy) * p[0].zw; return t.sample(s, st / p[1].zw, level(0)); }\n"
            "struct D { float d [[depth(any)]]; };\n"
            "fragment D fsd(V in [[stage_in]], depth2d<float> t [[texture(0)]], sampler s [[sampler(0)]], constant float4 *p [[buffer(0)]]) {\n"
            "  float2 st = p[0].xy + (in.p.xy - p[1].xy) * p[0].zw; D o; o.d = t.sample(s, st / p[1].zw, level(0)); return o; }\n";
        NSError *e = nil; id<MTLLibrary> lib = [g_dev newLibraryWithSource:@(src) options:nil error:&e];
        vf = [lib newFunctionWithName:@"vs"]; ff = [lib newFunctionWithName:@"fs"]; fd = [lib newFunctionWithName:@"fsd"];
        MTLDepthStencilDescriptor *dd = [MTLDepthStencilDescriptor new]; dd.depthCompareFunction = MTLCompareFunctionAlways; dd.depthWriteEnabled = YES; dss = [g_dev newDepthStencilStateWithDescriptor:dd];
        MTL4ArgumentTableDescriptor *ad = [MTL4ArgumentTableDescriptor new]; ad.maxTextureBindCount = 1; ad.maxSamplerStateBindCount = 1; ad.maxBufferBindCount = 1; at = [g_dev newArgumentTableWithDescriptor:ad error:&e];
        for (int i = 0; i < 2; i++) { MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new]; sd.minFilter = sd.magFilter = i ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
            sd.sAddressMode = sd.tAddressMode = MTLSamplerAddressModeClampToEdge; sd.supportArgumentBuffers = YES; samp[i] = [g_dev newSamplerStateWithDescriptor:sd]; }
        if (!vf || !ff || !fd || !dss || !at || !samp[0] || !samp[1]) { failed = 1; MLOG("【红】换格式 / 缩放贴图拷贝的管线建不出：%s", e.localizedDescription.UTF8String); return 0; }
        pso = [NSMutableDictionary dictionary];
    }
    id<MTLRenderPipelineState> ps = pso[@(d.pixelFormat)];
    if (!ps) {
        MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new]; pd.vertexFunction = vf; pd.fragmentFunction = depth ? fd : ff;
        if (depth) { pd.depthAttachmentPixelFormat = d.pixelFormat; if (stencil) pd.stencilAttachmentPixelFormat = d.pixelFormat; } else pd.colorAttachments[0].pixelFormat = d.pixelFormat;
        NSError *e = nil; ps = [g_dev newRenderPipelineStateWithDescriptor:pd error:&e];
        if (!ps) { RED_ONCE(83, "换格式 / 缩放贴图拷贝：目标格式 %lu 的管线建不出：%s", (unsigned long)d.pixelFormat, e.localizedDescription.UTF8String); return 0; }
        pso[@(d.pixelFormat)] = ps;
    }
    unsigned sw = MAX(s.width >> sl, 1ul), sh = MAX(s.height >> sl, 1ul), dw = MAX(d.width >> dl, 1ul), dh = MAX(d.height >> dl, 1ul);
    if ((unsigned)(dr->xoffset + dr->width) > dw || (unsigned)(dr->yoffset + dr->height) > dh) return 0;
    int nz = sr->depth ? sr->depth : 1;
    rp_end(); if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
    for (int z = 0; z < nz; z++) {
        id<MTLTexture> sv = [s newTextureViewWithPixelFormat:s.pixelFormat textureType:MTLTextureType2D levels:NSMakeRange(sl, 1) slices:NSMakeRange(ss + sr->zoffset + z, 1)];
        id<MTLBuffer> pb = [g_dev newBufferWithLength:32 options:MTLResourceStorageModeShared];
        if (!sv || !pb) { RED_ONCE(84, "换格式 / 缩放贴图拷贝：建不出源视图 / 参数块"); return z > 0; }
        float *p = pb.contents; p[0] = sr->xoffset; p[1] = sr->yoffset; p[2] = (float)sr->width / dr->width; p[3] = (float)sr->height / dr->height;
        if (mirror & NVN_COPY_FLAGS_MIRROR_X_BIT) { p[0] += sr->width; p[2] = -p[2]; }   // mirrored: walk the source region from its far edge
        if (mirror & NVN_COPY_FLAGS_MIRROR_Y_BIT) { p[1] += sr->height; p[3] = -p[3]; }
        p[4] = dr->xoffset; p[5] = dr->yoffset; p[6] = sw; p[7] = sh;
        res_add(pb);
        MTL4RenderPassDescriptor *rp = [MTL4RenderPassDescriptor new];
        MTLRenderPassAttachmentDescriptor *a = depth ? (MTLRenderPassAttachmentDescriptor *)rp.depthAttachment : (MTLRenderPassAttachmentDescriptor *)rp.colorAttachments[0];
        a.texture = d; a.level = dl; a.slice = ds + dr->zoffset + z; a.loadAction = MTLLoadActionLoad; a.storeAction = MTLStoreActionStore;
        if (stencil) { MTLRenderPassStencilAttachmentDescriptor *st = rp.stencilAttachment; st.texture = d; st.level = dl; st.slice = a.slice; st.loadAction = MTLLoadActionLoad; st.storeAction = MTLStoreActionStore; }
        ser_gate(); id<MTL4RenderCommandEncoder> en = enc_r_begin(rp);
        if (depth) [en setDepthStencilState:dss];
        [en setArgumentTable:at atStages:MTLRenderStageFragment];
        [at setTexture:sv.gpuResourceID atIndex:0]; [at setSamplerState:samp[!!linear && !depth].gpuResourceID atIndex:0]; [at setAddress:pb.gpuAddress atIndex:0];
        [en setViewport:(MTLViewport){dr->xoffset, dr->yoffset, dr->width, dr->height, 0, 1}];
        [en setScissorRect:(MTLScissorRect){(NSUInteger)dr->xoffset, (NSUInteger)dr->yoffset, (NSUInteger)dr->width, (NSUInteger)dr->height}];
        [en setRenderPipelineState:ps]; [en drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3]; enc_r_end(en);
        defer_obj_unref(sv); defer_obj_release(pb);   // released after this submission has run
    }
    args_dirty_all(); g_pso_dirty = g_dss_dirty = 1;
    { static int once[2]; if (!once[depth]++) MLOG("换格式 / 缩放贴图拷贝：首例 %dx%d 格式 %lu → %dx%d 格式 %lu%s%s", sr->width, sr->height, (unsigned long)s.pixelFormat, dr->width, dr->height, (unsigned long)d.pixelFormat,
                                                    linear && !depth ? "（双线性）" : "", stencil ? "（只拷深度，模板不拷）" : ""); }
    return 1;
}
// GPU: texture -> texture (previously mislisted as a no-op, 3 times per frame on the character select page). Same size, not mirrored, same format -> blit layer by layer; the other semantics (scaling with filtering, mirroring, format change) cannot be expressed by blit: report a failure, to be implemented
static void copy_t2t(const p_copy_t2t *c) {
    if (c->src) xv_check(c->src->serial, c->has_sv && c->sv.set_levels ? c->sv.base_level : 0, c->has_sv && c->sv.set_layers ? c->sv.min_layer : 0, "贴图拷贝源");
    rp_flush();   // Land the pending pure-clear render pass first (zero-source candidates only enter the set when the render pass ends); the blit() below would end the render pass anyway
    if (c->dst) zskip_del(c->dst->serial);
    if (c->src && c->dst && c->sr.xoffset == 0 && c->sr.yoffset == 0 && c->dr.xoffset == 0 && c->dr.yoffset == 0
        && c->sr.width == (int)c->src->d.width && c->sr.height == (int)c->src->d.height && c->dr.width == (int)c->dst->d.width && c->dr.height == (int)c->dst->d.height) {
        int sl = c->has_sv && c->sv.set_levels ? c->sv.base_level : 0, dl = c->has_dv && c->dv.set_levels ? c->dv.base_level : 0;
        int ss = c->has_sv && c->sv.set_layers ? c->sv.min_layer : 0, ds = c->has_dv && c->dv.set_layers ? c->dv.min_layer : 0;
        if (zskip_has(c->src->serial, sl, ss)) zskip_add(c->dst->serial, dl, ds);
    }
    if (c->dst) rc_write_use(c->dst->serial);   // T5: copy destination, the content can change after the upload
    id<MTLTexture> s = tex_mtl(c->src), d = tex_mtl(c->dst);
    PD("T src=%llu dst=%llu %dx%dx%d→%dx%dx%d flags=%#x", c->src->serial, c->dst->serial, c->sr.width, c->sr.height, c->sr.depth, c->dr.width, c->dr.height, c->dr.depth, c->flags);
    if (!s || !d) return;
    unsigned sl = c->has_sv && c->sv.set_levels ? c->sv.base_level : 0, dl = c->has_dv && c->dv.set_levels ? c->dv.base_level : 0;
    unsigned sz0 = c->has_sv && c->sv.set_layers ? c->sv.min_layer : 0, dz0 = c->has_dv && c->dv.set_layers ? c->dv.min_layer : 0;
    if (c->flags & DM_COPY_RESOLVE) {   // Downsample: multisample resolve into the single-sample destination (depth takes sample 0); with equal sample counts it is a plain whole copy
        if (s.sampleCount == d.sampleCount && s.pixelFormat == d.pixelFormat && s.width == d.width && s.height == d.height) {
            enc_b_copy_texture(blit(), s, sz0, sl, MTLOriginMake(0, 0, 0), MTLSizeMake(s.width, s.height, 1), d, dz0, dl, MTLOriginMake(0, 0, 0));
            return;
        }
        if (s.sampleCount <= 1 || d.sampleCount != 1 || s.pixelFormat != d.pixelFormat || s.width != d.width || s.height != d.height) {
            RED_ONCE(72, "Downsample 未实现的形态：%lux%lu %lu 样本 格式 %lu → %lux%lu %lu 样本 格式 %lu", (unsigned long)s.width, (unsigned long)s.height, (unsigned long)s.sampleCount,
                     (unsigned long)s.pixelFormat, (unsigned long)d.width, (unsigned long)d.height, (unsigned long)d.sampleCount, (unsigned long)d.pixelFormat);
            return;
        }
        rp_end(); if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
        MTL4RenderPassDescriptor *rp = [MTL4RenderPassDescriptor new];
        MTLPixelFormat pf = s.pixelFormat;
        int depth = pf == MTLPixelFormatDepth32Float || pf == MTLPixelFormatDepth16Unorm || pf == MTLPixelFormatDepth32Float_Stencil8;
        MTLRenderPassAttachmentDescriptor *a = depth ? rp.depthAttachment : rp.colorAttachments[0];
        a.texture = s; a.level = sl; a.slice = sz0; a.resolveTexture = d; a.resolveLevel = dl; a.resolveSlice = dz0;
        a.loadAction = MTLLoadActionLoad; a.storeAction = MTLStoreActionStoreAndMultisampleResolve;
        if (depth) rp.depthAttachment.depthResolveFilter = MTLMultisampleDepthResolveFilterSample0;
        ser_gate(); enc_r_end(enc_r_begin(rp));
        return;
    }
    PD("t src=%llu lv%u sl%u xyz=%d,%d,%d dst=%llu lv%u sl%u xyz=%d,%d,%d size=%d,%d,%d", c->src->serial, sl, sz0, c->sr.xoffset, c->sr.yoffset, c->sr.zoffset,
       c->dst->serial, dl, dz0, c->dr.xoffset, c->dr.yoffset, c->dr.zoffset, c->sr.width, c->sr.height, c->sr.depth);
    int s3 = c->src->d.target == 2, d3 = c->dst->d.target == 2, nz = c->sr.depth ? c->sr.depth : 1;
    if (no_bc() && (s.pixelFormat == MTLPixelFormatASTC_4x4_LDR || s.pixelFormat == MTLPixelFormatASTC_4x4_sRGB) && (d.pixelFormat == MTLPixelFormatRGBA8Unorm || d.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB)
        && !s3 && !d3 && c->sr.width == c->dr.width && c->sr.height == c->dr.height && (d.usage & MTLTextureUsageShaderWrite)) {   // No BC: ASTC source -> decompressed-format destination (see copy_t2t_astc)
        int ok = 1;
        for (int z = 0; z < nz && ok; z++) ok = copy_t2t_astc(s, d, sl, sz0 + c->sr.zoffset + z, dl, dz0 + c->dr.zoffset + z, c->sr.xoffset, c->sr.yoffset, c->dr.xoffset, c->dr.yoffset, c->sr.width, c->sr.height);
        if (ok) return;
    }
    if ((c->sr.width != c->dr.width || c->sr.height != c->dr.height || s.pixelFormat != d.pixelFormat || (c->flags & (NVN_COPY_FLAGS_MIRROR_X_BIT | NVN_COPY_FLAGS_MIRROR_Y_BIT))) && !s3 && !d3
        && (c->sr.depth ? c->sr.depth : 1) == (c->dr.depth ? c->dr.depth : 1)
        && !(c->flags & ~(NVN_COPY_FLAGS_ENGINE_2D_BIT | NVN_COPY_FLAGS_LINEAR_FILTER_BIT | NVN_COPY_FLAGS_MIRROR_X_BIT | NVN_COPY_FLAGS_MIRROR_Y_BIT))
        && copy_t2t_draw(s, d, sl, sz0, dl, dz0, &c->sr, &c->dr, !!(c->flags & NVN_COPY_FLAGS_LINEAR_FILTER_BIT), c->flags)) return;   // format change / scaling / mirroring: draw it (see copy_t2t_draw)
    if (c->sr.width != c->dr.width || c->sr.height != c->dr.height || (c->sr.depth ? c->sr.depth : 1) != (c->dr.depth ? c->dr.depth : 1) || (c->flags & ~NVN_COPY_FLAGS_ENGINE_2D_BIT & ~NVN_COPY_FLAGS_LINEAR_FILTER_BIT)
        || s.pixelFormat != d.pixelFormat || s3 != d3) {
        RED_ONCE(33, "贴图→贴图拷贝未实现的形态：%dx%dx%d→%dx%dx%d 标志 %#x 格式 %lu→%lu 3D %d→%d", c->sr.width, c->sr.height, c->sr.depth, c->dr.width, c->dr.height, c->dr.depth,
                 c->flags, (unsigned long)s.pixelFormat, (unsigned long)d.pixelFormat, s3, d3);
        return;
    }
    id<MTL4ComputeCommandEncoder> b = blit();
    if (s3)
        enc_b_copy_texture(b, s, 0, sl, MTLOriginMake(c->sr.xoffset, c->sr.yoffset, c->sr.zoffset), MTLSizeMake(c->sr.width, c->sr.height, nz), d, 0, dl, MTLOriginMake(c->dr.xoffset, c->dr.yoffset, c->dr.zoffset));
    else
        for (int z = 0; z < nz; z++)
            enc_b_copy_texture(b, s, sz0 + c->sr.zoffset + z, sl, MTLOriginMake(c->sr.xoffset, c->sr.yoffset, 0), MTLSizeMake(c->sr.width, c->sr.height, 1), d, dz0 + c->dr.zoffset + z, dl, MTLOriginMake(c->dr.xoffset, c->dr.yoffset, 0));
    // Diagnostic DM_NVN_T2T_TRACE=WxH:file: after each copy into an RGBA16F texture of that size, copy the result once more into a trace buffer on the GPU timeline (frame numbers kept on the CPU side); after 240 copies write the file
    { static int on = -1; static uint32_t tw, th; static const char *tf; static id<MTLBuffer> tb; static int tn; static uint64_t tfr[240], from;
      if (on < 0) { const char *e = getenv("DM_NVN_T2T_TRACE"), *fe = getenv("DM_NVN_T2T_FROM"); on = e && sscanf(e, "%ux%u", &tw, &th) == 2 && (tf = strchr(e, ':')) && *++tf; from = fe ? strtoull(fe, NULL, 10) : 0; }
      if (on > 0 && tn < 240 && atomic_load(&g_cur_frame) >= from && d.width == tw && d.height == th && d.pixelFormat == MTLPixelFormatRGBA16Float && !s3) {
          if (!tb) { tb = [g_dev newBufferWithLength:(size_t)240 * tw * th * 8 options:MTLResourceStorageModeShared]; res_add(tb); }
          enc_b_copy_texture_buffer(blit(), d, dz0, dl, MTLOriginMake(0, 0, 0), MTLSizeMake(tw, th, 1), tb, (size_t)tn * tw * th * 8, (size_t)tw * 8, (size_t)tw * th * 8);
          tfr[tn++] = atomic_load(&g_cur_frame);
          if (tn == 240) {
              rp_end(); if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
              mtl_finish_gpu();
              FILE *f = fopen(tf, "wb");
              if (f) { fprintf(f, "%u %u %d\n", tw, th, tn); fwrite(tfr, 8, 240, f); fwrite(tb.contents, 1, (size_t)240 * tw * th * 8, f); fclose(f); }
              MLOG("[t2t-trace] %d 次 → %s（帧 %llu–%llu）", tn, tf, (unsigned long long)tfr[0], (unsigned long long)tfr[239]);
          }
      }
    }
}

// GPU: clear a texture region with the given value (previously mislisted as a no-op, called 5 times at initialization; the initial value of a Metal private texture is undefined, and the game relies on the cleared initial value).
// Whole level and whole face, all four channels written -> open one clear-only render pass per layer / depth slice (the hardware converts the clear value by the texture format, including integer and depth);
// view with changed format: clear on a view of that format; partial region, partial channels, compressed / packaged textures (no render target usage): report a failure, to be implemented
static void clear_tex(const p_clear_tex *c) {
    if (c->tex) zskip_del(c->tex->serial);
    rc_write_use(c->tex ? c->tex->serial : 0);   // T5
    id<MTLTexture> t = tex_mtl(c->tex);
    PD("X ser=%llu 视图格式 %#x 区域 %d,%d,%d %dx%dx%d 掩码 %#x %s %g,%g,%g,%g", c->tex->serial, c->has_view && c->view.set_format ? c->view.format : -1, c->r.xoffset, c->r.yoffset, c->r.zoffset, c->r.width, c->r.height, c->r.depth, c->mask,
       c->is_uint ? "ui" : "f", c->is_uint ? (double)c->c.u[0] : c->c.f[0], c->is_uint ? (double)c->c.u[1] : c->c.f[1], c->is_uint ? (double)c->c.u[2] : c->c.f[2], c->is_uint ? (double)c->c.u[3] : c->c.f[3]);
    if (!t) return;
    unsigned lv = c->has_view && c->view.set_levels ? c->view.base_level : 0, l0 = c->has_view && c->view.set_layers ? c->view.min_layer : 0;
    NSUInteger w = MAX(t.width >> lv, 1u), h = MAX(t.height >> lv, 1u);
    int is3d = t.textureType == MTLTextureType3D, nz = c->r.depth ? c->r.depth : 1;
    if (c->has_view && c->view.set_format) {   // View with changed format: the clear value is interpreted in the view format (all 5 of the game's calls are the ui version with an integer view), and the clear is done on a view of the same format
        uint32_t vpf = 0; uint8_t swz[4];
        if (!nvn_fmt_to_mtl(c->view.format, &vpf, swz) || !vpf) { RED_ONCE(42, "清贴图视图格式 %#x 无对应", c->view.format); return; }
        if (tex_need_pfv(t.pixelFormat, (MTLPixelFormat)vpf) && !(t.usage & MTLTextureUsagePixelFormatView)) t = tex_upgrade_pfv(c->tex, t, (MTLPixelFormat)vpf);
        if (tex_need_pfv(t.pixelFormat, (MTLPixelFormat)vpf)) {
            id<MTLTexture> v = [t newTextureViewWithPixelFormat:(MTLPixelFormat)vpf];
            if (!v) { RED_ONCE(43, "清贴图建不出视图：格式 %lu → %u", (unsigned long)t.pixelFormat, vpf); return; }
            t = v;
        }
    }
    MTLPixelFormat pf = t.pixelFormat;
    int depth = pf == MTLPixelFormatDepth32Float || pf == MTLPixelFormatDepth16Unorm || pf == MTLPixelFormatDepth32Float_Stencil8, stencil = pf == MTLPixelFormatStencil8 || pf == MTLPixelFormatDepth32Float_Stencil8;
    if (c->r.xoffset || c->r.yoffset || (NSUInteger)c->r.width != w || (NSUInteger)c->r.height != h || (c->mask & 0xf) != 0xf
        || !(t.usage & MTLTextureUsageRenderTarget)) {
        RED_ONCE(44, "清贴图未实现的形态：区域 %d,%d %dx%d（本级 %lux%lu）掩码 %#x 可作渲染目标 %d", c->r.xoffset, c->r.yoffset, c->r.width, c->r.height,
                 (unsigned long)w, (unsigned long)h, c->mask, !!(t.usage & MTLTextureUsageRenderTarget));
        return;
    }
    rp_end(); if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
    for (int z = 0; z < nz; z++) {
        MTL4RenderPassDescriptor *rp = [MTL4RenderPassDescriptor new];
        MTLRenderPassAttachmentDescriptor *a = depth ? rp.depthAttachment : stencil ? rp.stencilAttachment : rp.colorAttachments[0];
        a.texture = t; a.level = lv; a.loadAction = MTLLoadActionClear; a.storeAction = MTLStoreActionStore;
        if (is3d) a.depthPlane = c->r.zoffset + z; else a.slice = l0 + c->r.zoffset + z;
        if (depth) { rp.depthAttachment.clearDepth = c->c.f[0];
            if (stencil) { rp.stencilAttachment.texture = t; rp.stencilAttachment.level = lv; rp.stencilAttachment.slice = a.slice; rp.stencilAttachment.loadAction = MTLLoadActionClear;
                           rp.stencilAttachment.storeAction = MTLStoreActionStore; rp.stencilAttachment.clearStencil = c->c.u[1]; } }
        else if (stencil) rp.stencilAttachment.clearStencil = c->c.u[0];
        else rp.colorAttachments[0].clearColor = c->is_uint ? MTLClearColorMake(c->c.u[0], c->c.u[1], c->c.u[2], c->c.u[3]) : MTLClearColorMake(c->c.f[0], c->c.f[1], c->c.f[2], c->c.f[3]);
        ser_gate(); enc_r_end(enc_r_begin(rp));
    }
}

// Context description of one submit (mtl_sub_commit takes one per submit and prints it into the failure line when the command buffer errors): for the GPU memory fault on the first frame entering Travincal, the failure line had only the sequence number and error code, and it was impossible to tell what was in that submit
static const char *cctx_snap(void) {
    static char b[320]; static uint64_t p_pass, p_up, p_reup, p_ser; uint64_t up = 0, reup = 0; mtl_pkg_stats(&up, &reup);
    pthread_mutex_lock(&g_deferred_mu); unsigned long nd = (unsigned long)g_deferred.count; pthread_mutex_unlock(&g_deferred_mu);
    snprintf(b, sizeof b, "帧 %llu；自上次提交：通道 +%llu、贴图首传 +%llu、重传 +%llu、序号表变动 +%llu；延迟释放队列 %lu 个；世界冻结状态 %d",
             (unsigned long long)atomic_load(&g_cur_frame), (unsigned long long)(g_stat_pass - p_pass), (unsigned long long)(up - p_up), (unsigned long long)(reup - p_reup),
             (unsigned long long)(g_ser_gen - p_ser), nd, g_frz_state);
    p_pass = g_stat_pass; p_up = up; p_reup = reup; p_ser = g_ser_gen; return b;
}
// ---------- Do not draw the world while the world is frozen (the whole game can be frozen while a panel is open) ----------
// The "world frozen" status bit g_frz_on is set by mtl_world_frozen() (the host's freeze handshake calls it; in tests DM_NVN_TEST_FRZ=1 can stand in, or it can be switched in-process via @set:g_frz_on of DM_NVN_UB).
// This code does not judge by itself whether a panel is open. Structure of a frame (3.3.94012 in-game, per-pass dump): UI pass -> [UI stencil pass] -> about 40 passes of shadows / depth / lighting / transparency / post-processing -> final compositing pass (writes the window texture).
// The first frame after the bit is set = full frame: draw everything as usual, dropping only the draws of the UI stencil pass (stencil all 0 = no pixels blocked by the panel) -- post-processing normally does not write pixels blocked by the panel, so a shrinking panel would expose old content;
// from the second frame on = frozen frame: from the first render pass without a color attachment up to before the render pass that writes the window texture, set-target / clear / draw / texture-to-texture copy / compute are all not executed, the scene result stays as the full frame's copy, and the UI pass and the final compositing pass are drawn as usual.
// Frame boundaries are identified by render pass: the first set-target without the window texture after the window texture has been written = start of a new frame, and the status bit is latched here. Unrecognized structure (no render pass without color in a frame) = skip nothing.
// State constants / buffer updates / events / counters / texture uploads execute as usual (the UI pass needs them, the game is waiting).
int g_frz_on = -1;
static int g_frz_phase = 2, g_frz_drop, g_frz_seen_ds; static uint64_t g_frz_full_n, g_frz_skip_n;
void mtl_world_frozen(int on) { g_frz_on = on != 0; }
static int frz_wanted(void) { if (g_frz_on < 0) { const char *e = getenv("DM_NVN_TEST_FRZ"); g_frz_on = e && atoi(e); if (g_frz_on) MLOG("DM_NVN_TEST_FRZ：世界冻结状态位由测试开关顶替（常开）"); } tbdr_on(); return g_frz_on > 0 && tbdr_on() && g_ruler_truth <= 0; }
// ---------- Runtime extrapolation (power-saving rendering method; no switch for the player; it caused a stutter defect in released builds, now off by default, to be re-enabled once there is evidence of no stutter on A17) ----------
// 3D every other frame: a fully drawn frame draws as usual and, before the attachments are repurposed, copies the opaque HDR (468), the main depth-stencil, the post-tone-mapping 468 and the post-FXAA 469;
// extrapolated frame: the UI stencil pass runs as usual; shadow map / depth prepass / main pass keep only the draws of moving units (skinned); before the first skinned draw of the depth prepass run one extrapolation pass (nvn_rtex.inc: by cb3 +464 -> +336
// move the previous frame's HDR and depth to this frame, treat old unit positions 0x10 / UI cutout 0x2 as holes to fill, and export stencil 0x80 for holes and pixels newly exposed at the screen edge); translucents are drawn as usual and write effect bit 0x40;
// compositing / add bloom / add fog / tone mapping / FXAA really run only in the 0xD0 region, and outside the region the previous frame's LDR result is copied in with a whole-image shift (orthographic camera, rounded); full-screen items such as the bloom chain are skipped; sharpening upscale and UI compositing run as usual.
// Fallback to the original rendering method: version file render_safe = true (dm_update.m), the extrapolation pipeline cannot be built (whole run), test DM_RT_AB=0. When a panel is open and the world is frozen, yield to the freeze path.
// On a camera jump (orientation change or translation over 64 pixels) and for 4 frames before and after switching the 3D render scale, whole frames are fully drawn. Unit shadows on the ground keep using the previous fully drawn frame.
static int rt_is_skin(void) {   // Moving units = the 10 whose vertex shader reads constants by bone index (the key table is taken from the Mac shader cache b-424a072084e6, identified by the dynamic vp_c5 index in the source)
    __unsafe_unretained NvProg *pr = S.prog ? prog_get(S.prog) : nil; if (!pr || !pr->sh[0]) return 0;
    static const char *k[] = { "0fa4eb7559f1637a", "1d97d33ab66abcba", "3deaf68fd65b5298", "4dffa355f2fbcdaf", "54c93fa22abff9ee",
                               "64dd7d022f3fae39", "67e2ebc28c830f44", "add86ee8e20f89b9", "b3ae9be5cc670f35", "fbf6dc1873a449e9" };
    for (unsigned i = 0; i < sizeof k / sizeof *k; i++) if (!strncmp(pr->sh[0]->key, k[i], 16)) return 1;
    return 0;
}
int g_dm_rt_ab = 0;   // Default is the original rendering method (the new method stutters in-game on A17 iPhone 15 Pro and ghosts characters on the character select page, so off for now); DM_RT_AB=1 turns on the new method for testing. The version file render_safe can only set it to 0, no effect
static struct { int env, frame, filt, first, hdrp, mainp_seen, skipall, oit_seen, tm_seen, fx_state, comp_seen; uint64_t ldr_ser; float vpc[16], vpp[16]; id<MTLTexture> ldr, pl468, pl469; int ui_seen, main_seen, copied, ex_done, shadow, have_prev, have_ldr, sdx, sdy, cam_checked, force_full, disabled, was_on; uint64_t n_jump; uint64_t ds_ser, hdr_ser, n_ex, n_full, n_noex;
    id<MTLTexture> ds, hdr, pds, phdr; struct { uint64_t key; int skin, other; } sh[16]; int nsh, cur_sh; } g_rt = { .env = -1 };
static int rt_want(void);
static void rt_toggle_poll(void) { static int tg = -1; if (tg < 0) { const char *e = getenv("DM_RT_AB_TOGGLE"); tg = e ? atoi(e) : 0; (void)rt_want(); }   // On first call also read DM_RT_AB (on a real device the world-freeze bit is set to 0 very early, so frz_op is not necessarily called)   // test: switch back and forth between A / B every tg seconds by wall clock
    if (tg > 0) { int v = (int)((time(NULL) / tg) & 1); if (v != g_dm_rt_ab) { g_dm_rt_ab = v; MLOG("DM_RT_AB_TOGGLE：画法切到 %s", v ? "B" : "A"); } } }
void mtl_rt_force_full(int n) { g_rt.force_full = n; }   // Called by the host: before an overall picture change such as switching the 3D render scale, the following n frames are all fully drawn
// ---------- Power-saving mode (30 frames) ----------
// On: the display link object changes to 30 Hz (frame rate range 30–30, a ProMotion screen lowers its refresh rate accordingly; the frame rate hint in front_ios.m is lowered together), the game presents one frame per display-link tick; runtime extrapolation off (really draws every frame).
// Game logic already runs at 25 ticks / second and unit positions are interpolated by time, so there are no repeated frames at 30 frames (measured on a real device: coefficient of variation of per-frame camera displacement 0.292 at 30 frames, 0.296 at 60 frames).
// The switch is on the 「自定义」(Custom) page (dm_custom.c eco_fps, off by default, takes effect on apply); the test switch DM_FPS=30/60 pins it.
__attribute__((weak)) int dm_fps_setting(void) { return 60; }   // The host's dm_custom.c provides the strong definition; always 60 when graphics unit tests do not link the host
// When the game itself asks for 30 fps (pacing by the game's present interval, interval 2, hle_nvn.c) the display link runs at 30 Hz too: games whose logic counts frames, such as Engage, are not pulled up to 60 fps by the display link on iOS
__attribute__((weak)) int nvn_game_fps(void) { return 60; }   // hle_nvn.c provides the strong definition
int mtl_fps_target(void) { static int env = -2; if (env == -2) { const char *e = getenv("DM_FPS"); env = e ? atoi(e) : -1; if (env != 30 && env != 60) env = -1; else MLOG("DM_FPS=%d：帧数由测试开关钉住", env); }
    return env > 0 ? env : dm_fps_setting() == 30 || nvn_game_fps() == 30 ? 30 : 60; }
// Whenever the extrapolation enabled state changes (power-saving mode toggled, world freeze takes over / hands back): first let the "no store across frames" ledger relearn (during the trial period it stores as usual), and the extrapolation really switches one frame later --
// guaranteeing the last submit before the switch is a full store. Otherwise attachments cleared first every frame at 30 frames are learned as no-store, and the first extrapolated frame after switching back to 60 frames reads old content and flashes for one frame (measured on a test build; only reverted after the xv self-check reported a failure)
static void xv_relearn(const char *why) { g_xv_n = 0; g_xv_ntrial = 0; g_xv_ntouch = 0; MLOG("跨帧不写回：%s，账本清空重学（试用期内照常写回）", why); }
static int rt_want_raw(void);
static int rt_want(void) {
    static int eff = -1; static uint64_t pend; int want = rt_want_raw(); uint64_t f = atomic_load(&g_cur_frame);
    if (eff < 0) eff = want;
    if (want == eff) { pend = 0; return eff; }
    if (!pend) { pend = f + 1; xv_relearn(want ? "外推将启用" : "外推将停用"); return eff; }
    if (f > pend) { eff = want; pend = 0; }
    return eff;
}
static int rt_want_raw(void) { static int forced; if (g_rt.env < 0) { const char *e = getenv("DM_RT_AB"); g_rt.env = e ? atoi(e) : 0; forced = e != NULL; if (e) MLOG("DM_RT_AB=%d：画法由环境变量钉住（测试用，自定义页的开关不再起作用）", g_rt.env); }
    if (forced) g_dm_rt_ab = g_rt.env;
    return g_dm_rt_ab == 1 && !g_rt.disabled && g_frz_on <= 0 && tbdr_on() && g_ruler_truth <= 0 && mtl_fps_target() == 60; }   // When a panel is open and the world is frozen, yield to the freeze path (saves more); power-saving mode (30 frames) really draws every frame
#include "nvn_rtex.inc"
static uint64_t rt_key(const p_set_rt *r) { uint64_t s_; int lv, sl; rt_sub(r, 8, &s_, &lv, &sl); return s_ * 64 + (uint64_t)lv * 8 + (uint64_t)sl; }
static int rt_is_shadow(const p_set_rt *r) { return !r->n && r->depth && r->depth->d.width == r->depth->d.height; }
static id<MTLTexture> rt_cp(id<MTLTexture> src, id<MTLTexture> dst0) {
    id<MTLTexture> dst = dst0;
    if (!dst || dst.width != src.width || dst.height != src.height || dst.pixelFormat != src.pixelFormat) {
        MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:src.pixelFormat width:src.width height:src.height mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView; d.storageMode = MTLStorageModePrivate;
        if (dst0) defer_obj_release(dst0);   // Size change: the old copy may still be read by an in-flight command buffer (MTL4 command buffers do not retain resources), so release it deferred
        dst = [g_dev newTextureWithDescriptor:d]; if (!dst) return nil; res_add(dst); }
    id<MTL4ComputeCommandEncoder> b = blit(); enc_b_copy_texture(b, src, 0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(src.width, src.height, 1), dst, 0, 0, MTLOriginMake(0, 0, 0)); return dst; }
static void rt_blit_shift(id<MTLTexture> src, id<MTLTexture> dst, int dx, int dy) {   // Integer-offset copy: dst(x + dx, y + dy) = src(x, y); the strip at the screen edge that cannot be moved is left for the region recompute (the extrapolation pass has already marked it 0x80)
    if (!src || !dst || src.width != dst.width || src.height != dst.height) return;
    int W = (int)src.width, H = (int)src.height, w = W - abs(dx), h = H - abs(dy); if (w <= 0 || h <= 0) return;
    id<MTL4ComputeCommandEncoder> b = blit(); enc_b_copy_texture(b, src, 0, 0, MTLOriginMake(MAX(0, -dx), MAX(0, -dy), 0), MTLSizeMake(w, h, 1), dst, 0, 0, MTLOriginMake(MAX(0, dx), MAX(0, dy), 0)); }
static void rt_copy_prev(void) {   // Fully drawn frame: copy the opaque HDR and the main depth-stencil once each (in order within the GPU, in this command buffer)
    if (g_rt.copied || !g_rt.main_seen || !g_rt.hdr || !g_rt.ds) return; g_rt.copied = 1;   // Do not copy when the main pass was not recognized this frame (foreground / loading screen): the textures recognized last time may already have been released by the game and not be in the residency set
    if (!rx_setup(g_rt.hdr.pixelFormat, g_rt.ds.pixelFormat)) { if (!g_rt.disabled) { g_rt.disabled = 1; MLOG("【红】运行时外推：外推管线在这台设备上建不出，整局退回原画法"); } return; }   // Cannot be built (old GPU does not support stencil export etc.): fall back for the whole run
    g_rt.phdr = rt_cp(g_rt.hdr, g_rt.phdr); g_rt.pds = rt_cp(g_rt.ds, g_rt.pds); g_rt.have_prev = g_rt.phdr && g_rt.pds; }
static void rt_frame_begin(void) { g_rt.filt = g_rt.ui_seen = g_rt.main_seen = g_rt.copied = g_rt.ex_done = g_rt.mainp_seen = 0; g_rt.cur_sh = -1;
    g_rt.oit_seen = g_rt.tm_seen = g_rt.fx_state = g_rt.comp_seen = g_rt.skipall = g_rt.cam_checked = 0; g_rt_ovr = 0; g_dss_dirty = 1;
    if (g_frz_state == 0) { g_rt.have_prev = g_rt.have_ldr = 0; for (int i = 0; i < g_rt.nsh; i++) g_rt.sh[i].skin = g_rt.sh[i].other = 0; } }   // Fully drawn frame: the next frame may extrapolate only if this frame's copies succeeded
static void rt_set_rt(const p_set_rt *r, int win) {   // Every set-target (both fully drawn and extrapolated frames go through here): identify roles, find the copy points in fully drawn frames, and in extrapolated frames decide whether this render pass keeps only skinned draws
    g_rt.cur_sh = -1; g_rt.shadow = 0; g_rt.first = 1; g_rt.hdrp = 0;
    if (win) { if (g_frz_state == 0) { rt_copy_prev(); if (g_rt.fx_state == 2 && g_rt.ldr) { g_rt.pl469 = rt_cp(g_rt.ldr, g_rt.pl469); g_rt.have_ldr = g_rt.pl469 != nil; g_rt.fx_state = 3; } }
        g_rt.filt = 0; g_rt.skipall = 0; if (g_rt_ovr) { g_rt_ovr = 0; g_dss_dirty = 1; } return; }
    id<MTLTexture> c0 = r->n > 0 && r->color[0] ? tex_mtl(r->color[0]) : nil, dt = r->depth ? tex_mtl(r->depth) : nil;
    int hdr_here = 0; for (int i = 0; i < r->n && i < 8; i++) if (r->color[i] && g_rt.hdr_ser && r->color[i]->serial == g_rt.hdr_ser) hdr_here = 1;
    if (g_frz_state == 0) {
        if (!g_rt.main_seen && r->n == 1 && c0 && c0.pixelFormat == MTLPixelFormatRG11B10Float && dt && dt.pixelFormat == MTLPixelFormatDepth32Float_Stencil8 && dt.width >= 500 && dt.width != dt.height) {
            { static uint64_t lh, ld; if (lh != r->color[0]->serial || ld != r->depth->serial) { lh = r->color[0]->serial; ld = r->depth->serial; MLOG("运行时外推：认出主通道 不透明 HDR 序号 %llu %lux%lu、主深度模板序号 %llu %lux%lu（通道 %llu）", (unsigned long long)lh, (unsigned long)c0.width, (unsigned long)c0.height, (unsigned long long)ld, (unsigned long)dt.width, (unsigned long)dt.height, g_stat_pass + 1); } }
            g_rt.main_seen = 1; g_rt.hdr_ser = r->color[0]->serial; g_rt.ds_ser = r->depth->serial; g_rt.hdr = c0; g_rt.ds = dt; }
        else if (g_rt.main_seen && hdr_here && !g_rt.copied) rt_copy_prev();   // The first set-target after the main pass that attaches the opaque HDR again (the translucent pass; 468 occupies slot 0 but is not written): before it, copy the HDR and the main depth-stencil
        else if (g_rt.main_seen && hdr_here && r->n == 1 && r->depth && r->depth->serial == g_rt.ds_ser) g_rt.fx_state = 1;   // Only 468 + main depth attached = tone mapping (repurposes 468 as the LDR output)
        else if (g_rt.fx_state == 1) {   // The first render pass after tone mapping = FXAA (reads 468, writes 469): before it, copy the post-tone-mapping 468 (LDR) and note down 469
            if (r->n >= 1 && r->color[0] && c0) { g_rt.ldr_ser = r->color[0]->serial; g_rt.ldr = c0; g_rt.pl468 = rt_cp(g_rt.hdr, g_rt.pl468); g_rt.fx_state = 2; } else g_rt.fx_state = 0; }
        else if (g_rt.fx_state == 2 && g_rt.ldr) { g_rt.pl469 = rt_cp(g_rt.ldr, g_rt.pl469); g_rt.have_ldr = g_rt.pl468 && g_rt.pl469; g_rt.fx_state = 3; }   // Before the first render pass after FXAA: copy the post-FXAA 469
    }
    if (rt_is_shadow(r)) { uint64_t k = rt_key(r); int i; for (i = 0; i < g_rt.nsh && g_rt.sh[i].key != k; i++) {}
        if (i == g_rt.nsh && i < 16) { g_rt.sh[i].key = k; g_rt.sh[i].skin = g_rt.sh[i].other = 0; g_rt.nsh++; } g_rt.cur_sh = i < 16 ? i : -1; g_rt.shadow = 1; }
    int ds_here = r->depth && g_rt.ds_ser && r->depth->serial == g_rt.ds_ser;
    if (ds_here && !r->n && !g_rt.ui_seen) { g_rt.ui_seen = 1; g_rt.filt = 0; return; }   // The first render pass in this frame with the main depth and no color = the UI stencil pass: run as usual
    g_rt.hdrp = hdr_here && ds_here && r->n == 1 && !g_rt.mainp_seen; if (g_rt.hdrp) g_rt.mainp_seen = 1;   // The first render pass in this frame with "only opaque HDR + main depth attached" = the main pass
    g_rt.filt = g_rt.shadow || (ds_here && !r->n) || g_rt.hdrp;
    if (g_frz_state == 2 && g_rt.mainp_seen && !g_rt.hdrp) {   // Extrapolated frame, after the main pass: post-processing runs only in the unit / effect region (plan B)
        int c0ldr = r->n >= 1 && r->color[0] && g_rt.ldr_ser && r->color[0]->serial == g_rt.ldr_ser, ovr = 0; g_rt.skipall = 0;
        if (g_rt.fx_state == 2) {}   // After FXAA (sharpening upscale + vignette, UI compositing): run everything
        else if (g_rt.fx_state == 1) g_rt.fx_state = 2;
        else if (!g_rt.oit_seen && ds_here && r->n >= 2) { g_rt.oit_seen = 1; ovr = 3; }   // Translucent (weighted blended, two targets): draw as usual, writing effect bit 0x40 where it passes
        else if (!g_rt.comp_seen) { if (ds_here && c0ldr) { g_rt.comp_seen = 1; ovr = 1; rt_blit_shift(g_rt.pl469, g_rt.ldr, g_rt.sdx, g_rt.sdy); } else g_rt.skipall = 1; }   // Before compositing: the previous frame's post-FXAA 469 is copied in with an integer shift (then recomputed in the region)   // the other passes before compositing (mip chain, 511 refraction layer etc.) are skipped
        else if (ds_here && c0ldr && g_rt.tm_seen) {   // The FXAA pass itself (recomputed in the region; the 469 outside the region was already shift-copied before compositing)
            g_rt.fx_state = 1; ovr = 1; }
        else if (ds_here) { if (hdr_here && r->n == 1) { g_rt.tm_seen = 1; rt_blit_shift(g_rt.pl468, g_rt.hdr, g_rt.sdx, g_rt.sdy); } ovr = 1; }   // Before tone mapping: the previous frame's post-tone-mapping 468 (LDR) is copied in with an integer shift   // add bloom, add fog, tone mapping: run only in the region
        else g_rt.skipall = 1;   // Those without the main depth attached, such as the bloom chain: skip (the region keeps using the previous frame's bloom)
        if (ovr != g_rt_ovr) { g_rt_ovr = ovr; g_dss_dirty = 1; }
    }
}
static int rt_op(unsigned op) {   // Non-set-target operations in an extrapolated frame: returns 1 = not executed
    if (g_frz_state == 2 && g_rt_ovr == 1 && op == OP_CLEAR_COLOR) return 1;   // The region post-processing passes do not clear: outside the region is the previous frame's result copied in with a shift
    if (g_rt.skipall && (op == OP_DRAW || op == OP_DRAW_IDX || op == OP_CLEAR_COLOR || op == OP_CLEAR_DS || op == OP_COPY_T2T || op == OP_CLEAR_TEX || op == OP_DISPATCH)) return 1;
    if (op == OP_DRAW || op == OP_DRAW_IDX) {
        if (g_frz_state == 2 && !g_rt.cam_checked) {   // First draw of an extrapolated frame (UI stencil pass, before shadows / prepass): check this frame's camera, and on a jump switch the whole frame to fully drawn
            g_rt.cam_checked = 1; const p_bind_buf *u = &S.ubo[0][3]; void *mb; uint64_t mo; const uint8_t *base; int ok = 0;
            if (u->addr && u->size >= 528 && mtl_addr(u->addr, u->size, &mb, &mo) && (base = [(__bridge id<MTLBuffer>)mb contents])) {
                const float *cur = (const float *)(base + mo + 336), *prev = (const float *)(base + mo + 464); int dx, dy; float md = 0;
                for (int r = 0; r < 3; r++) for (int c = 0; c < 4; c++) md = MAX(md, fabsf(cur[r * 4 + c] - prev[r * 4 + c]));   // The first three rows (orientation / scale) should be unchanged: an orthographic camera only translates
                rt_shift_px(prev, cur, g_rt.hdr.width, g_rt.hdr.height, &dx, &dy);
                ok = md < 1e-4f && abs(dx) <= 64 && abs(dy) <= 64; if (!ok) { static int n; if (n++ < 20) MLOG("运行时外推：镜头跳变（平移 %d,%d 像素、朝向差 %g），本帧改全画", dx, dy, md); } }
            if (!ok) { g_frz_state = 0; g_rt.n_jump++; rt_frame_begin(); g_rt.cam_checked = 1; return 0; }
        }
        if (g_frz_state == 0) { if (g_rt.cur_sh >= 0) { if (rt_is_skin()) g_rt.sh[g_rt.cur_sh].skin++; else g_rt.sh[g_rt.cur_sh].other++; } return 0; }
        if (!g_rt.filt) return 0;
        if (!rt_is_skin()) return 1;
        if (!g_rt.ex_done && !g_rt.shadow) {   // Before the first skinned draw of the depth prepass: the extrapolation pass (reads the cb3 bound by this draw: +336 this frame's, +464 the previous frame's view-projection)
            g_rt.ex_done = 1; const p_bind_buf *u = &S.ubo[0][3]; void *mb; uint64_t mo; const uint8_t *base;
            if (u->addr && u->size >= 528 && mtl_addr(u->addr, u->size, &mb, &mo) && (base = [(__bridge id<MTLBuffer>)mb contents])) {
                float cur[16], prev[16]; memcpy(cur, base + mo + 336, 64); memcpy(prev, base + mo + 464, 64); memcpy(g_rt.vpc, cur, 64); memcpy(g_rt.vpp, prev, 64); rt_shift_px(prev, cur, g_rt.hdr.width, g_rt.hdr.height, &g_rt.sdx, &g_rt.sdy);
                rp_end(); if (rt_ex_encode(g_rt.hdr, g_rt.ds, g_rt.phdr, g_rt.pds, prev, cur, 0x10, 0x2)) g_rt.n_ex++; else g_rt.n_noex++;
            } else g_rt.n_noex++;
        }
        return 0;
    }
    if (g_frz_state == 2 && g_rt.filt) {
        if (op == OP_CLEAR_COLOR || op == OP_CLEAR_DS) {
            if (g_rt.shadow && g_rt.cur_sh >= 0 && !g_rt.sh[g_rt.cur_sh].other) return 0;   // A shadow level that had only skinned draws in the previous fully drawn frame (character-only level): clear and draw as usual
            return 1; }
    }
    return 0;
}
static int frz_op(unsigned op, const void *a) {   // Returns 1 = this operation is not executed
    if (op == OP_SET_RT) {
        const p_set_rt *r = a; int win = 0;
        for (int i = 0; i < r->n; i++) { uint64_t s_; int lv, sl; rt_sub(r, i, &s_, &lv, &sl); if (s_ && xv_present(s_)) win = 1; }
        if (g_frz_phase == 2 && !win && rt_want()) {   // Runtime extrapolation: new frame, fully drawn frames and extrapolated frames alternate
            int was = g_frz_state; g_frz_phase = 0; g_rt.was_on = 1;
            { static int dbg = -1, n; if (dbg < 0) dbg = getenv("DM_RT_DBG") != NULL; if (dbg && n < 4000 && g_rt.ds_ser) { n++; MLOG("[rtdbg] 帧界 通道 %llu 上一段状态 %d 认出主通道 %d 拷了 %d 外推 %d fx %d ldr %d hoist %d fz %d defer %d/%d", g_stat_pass, was, g_rt.main_seen, g_rt.copied, g_rt.ex_done, g_rt.fx_state, g_rt.have_ldr, g_hoist_n, g_fz.active, defer_has_active(), g_defer_replaying); } }
            if (g_rt.force_full > 0) g_rt.force_full--;
            g_frz_state = !was && g_rt.have_prev && g_rt.have_ldr && !g_rt.force_full && !g_hoist_n && !g_fz.active && !defer_has_active() && !g_defer_replaying ? 2 : 0;
            if (g_frz_state == 2) g_rt.frame++; else g_rt.n_full++;
            if (g_frz_state != was && (g_rt.frame + g_rt.n_full) % 600 < 2) MLOG("运行时外推：全画帧 %llu、外推帧 %d（外推趟编了 %llu 次、没编 %llu 次；镜头跳变改全画 %llu 次）", g_rt.n_full, g_rt.frame, g_rt.n_ex, g_rt.n_noex, g_rt.n_jump);
            rt_frame_begin();
        }
        if (g_frz_phase != 2 && rt_want()) { if (win) g_frz_phase = 2; rt_set_rt(r, win); return g_rt.skipall; }
        if (g_frz_phase == 2 && win && rt_want()) { rt_set_rt(r, win); return 0; }
        { int on = rt_want(); if (g_rt.was_on && !on) {   // The new rendering method was just turned off (version file fallback / panel freeze takes over / fallback for the whole run): reset state to zero and hand back to the original path
              g_frz_state = 0; g_frz_phase = 2; g_rt.have_prev = g_rt.have_ldr = 0; if (g_rt_ovr) { g_rt_ovr = 0; g_dss_dirty = 1; } g_rt.skipall = 0;
              MLOG("运行时外推：新画法停用（画法 %d、世界冻结 %d、整局退回 %d）", g_dm_rt_ab, g_frz_on, g_rt.disabled); }
          g_rt.was_on = on; }
        if (g_frz_phase == 2 && !win) {   // New frame
            int on = frz_wanted(), was = g_frz_state; g_frz_phase = 0; g_frz_seen_ds = 0;
            { static uint64_t fr, p0, k0; if (was == 2 && (++fr == 4 || fr % 600 == 0)) MLOG("世界冻结：第 %llu 个冻结帧开了 %llu 个通道、跳过 %llu 条设目标", fr, g_stat_pass - p0, g_frz_skip_n - k0); if (was != 2) fr = 0; p0 = g_stat_pass; k0 = g_frz_skip_n; }
            if (!on) g_frz_state = 0;
            else if (was) g_frz_state = 2;
            else if (!g_hoist_n && !g_fz.active && !defer_has_active() && !g_defer_replaying) g_frz_state = 1;   // The premises / merges / deferrals left by the previous frame have not finished yet: enter next frame
            if (g_frz_state != was) MLOG("世界冻结：通道 %llu 起 %s（完整帧累计 %llu，已跳过通道 %llu）", g_stat_pass + 1, g_frz_state == 1 ? "完整帧（不带界面模板）" : g_frz_state == 2 ? "冻结帧（只画界面与末尾合成）" : "恢复正常", g_frz_full_n += g_frz_state == 1, g_frz_skip_n);
        }
        g_frz_drop = 0;
        if (win) { g_frz_phase = 2; return 0; }
        if (!g_frz_state || g_frz_phase == 2) return 0;
        if (g_frz_phase == 0 && !r->n && r->depth && !g_frz_seen_ds) {   // The first render pass without color in this frame
            g_frz_seen_ds = 1;
            if (g_frz_state == 2) g_frz_phase = 1;
            else { uint64_t s_; int lv, sl; rt_sub(r, 8, &s_, &lv, &sl); static int neg = -1; if (neg < 0) neg = getenv("DM_NVN_TEST_FRZ_NEG") != NULL;   // Negative control: the full frame is drawn with the UI stencil as usual, and old pixels must be exposed when the panel shrinks
                   g_frz_drop = !neg && MAX(r->depth->d.width >> lv, 1u) != MAX(r->depth->d.height >> lv, 1u); }   // Full frame: the one that is not square (shadow map) is the UI stencil pass
        }
        if (g_frz_phase == 1) { g_frz_skip_n++; return 1; }
        return 0;
    }
    if (rt_want() && g_frz_phase != 2) return rt_op(op);
    if (g_frz_drop) return op == OP_DRAW || op == OP_DRAW_IDX;
    if (g_frz_state == 2 && g_frz_phase == 1) return op == OP_CLEAR_COLOR || op == OP_CLEAR_DS || op == OP_DRAW || op == OP_DRAW_IDX || op == OP_COPY_T2T || op == OP_CLEAR_TEX || op == OP_DISPATCH;
    return 0;
}

// Hand a submission to the GPU in segments (DM_NVN_CUT=draw count, 0 = off; default 64): when the game holds back the next frame's submit until this frame's GPU work is done (The Legend of Zelda: Breath of the Wild: the main thread's nvnSyncWait on the fence of the frame just submitted),
// the GPU sits idle for the milliseconds the host spends translating inside the submit (in the shrine each frame is GPU 13 ms + translation 7 ms back to back). Each time this many draws have accumulated, the command buffer encoded so far is committed between two passes: the GPU runs the earlier passes while the CPU translates the later ones.
// Cut only when no encoder is open and no pass is deferred / merged / hoisted; commit this batch's uploads first (later draws use them). The fence signal and completion work stay on the last segment only: the queue executes in commit order and every encoder starts with a queue barrier
static int cut_draws(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_NVN_CUT"); v = e ? atoi(e) : 64; MLOG("分段提交：每 %d 次绘制在通道之间先交一段（DM_NVN_CUT，0 关）", v); } return v; }
static uint64_t g_cut_draw0, g_cut_sub, g_cuts;
static void sub_cut(void) {
    int n = cut_draws();
    if (n <= 0 || !g_cb || g_enc) return;
    if (g_cut_sub != g_sub_serial) { g_cut_sub = g_sub_serial; g_cut_draw0 = g_stat_draw; }   // count from the start of each submission
    if (g_stat_draw - g_cut_draw0 < (uint64_t)n || g_hoist_n || g_defer_nact || g_defer_pend || g_fz.active || g_rt_merged) return;
    if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
    up_flush();
    up_wait();
    uint64_t f = atomic_load(&g_cur_frame); gpu_tag_begin(f);
    MTL4CommitOptions *o = [MTL4CommitOptions new];
    [o addFeedbackHandler:^(id<MTL4CommitFeedback> cb) {
        gpu_tag_done(f, cb.GPUStartTime, cb.GPUEndTime);
        if (cb.error) { atomic_store(&g_last_commit_error, 1); static int k; if (k++ < 8) MLOG("【红】分段命令缓冲执行出错：%s", cb.error.description.UTF8String); }
    }];
    res_commit();
    [g_cb endCommandBuffer];
    id<MTL4CommandBuffer> buffers[] = {g_cb};
    [g_q4 commit:buffers count:1 options:o];
    g_cb = [g_dev newCommandBuffer];
    [g_cb beginCommandBufferWithAllocator:g_alloc4[g_alloc_cur]];   // one allocator encodes several command buffers in sequence; it is reset only once its fence (the last segment) has passed, when a later submission starts
    g_cut_draw0 = g_stat_draw;
    if (!g_cuts++) MLOG("分段提交：首段在帧 %llu 通道 %llu 交出", (unsigned long long)f, (unsigned long long)g_stat_pass);
}

void mtl_exec(const uint8_t *p) {
    static int inited; if (!inited) { inited = 1; state_reset(); }
    static int dump_all = -1; if (dump_all < 0) dump_all = getenv("DM_NVN_PASSDUMP_ALL") != NULL;
    int trace = dump_all && passdump();   // Normal mode does not check the frame range per operation; the full command position is only used for the designated diagnostic frames
    for (;;) {
        op_hdr h; memcpy(&h, p, sizeof h);
        g_exec_cur = p;   // Scan start point of the store lookahead: the operation that ends the render pass (inclusive)
        if (p == g_defer_pend) { for (int gi = 0; gi < 4; gi++) if (g_defer_slots[gi].active && g_defer_slots[gi].p_end_op == p) g_defer_slots[gi].in_p = 0; g_defer_pend = NULL; }
        sdx_seq(p);
        if (h.op == OP_SET_RT) rt_toggle_poll();
        if ((g_frz_on || g_frz_state || g_dm_rt_ab == 1) && frz_op(h.op, p + 8)) { p += h.size; continue; }
        if (trace) PD("o cmd=%p pass=%llu op=%u size=%u", p, g_stat_pass, (unsigned)h.op, (unsigned)h.size);
        // Diagnostic DM_NVN_ALLOCPROF=1: growth of the GPU allocated size is charged to the previous operation's type (rough: allocations on other threads are counted too); the memory report line appends the top 4
        if (g_allocprof < 0) g_allocprof = getenv("DM_NVN_ALLOCPROF") != NULL;
        if (g_allocprof) { static uint64_t last; static int prev = -1; uint64_t now = g_dev.currentAllocatedSize; if (prev >= 0 && prev < 64 && now > last) g_ap_bytes[prev] += now - last; last = now; prev = h.op; }
        if (g_hoist_n && hoist_take(p)) { if (g_fz.active && p == g_fz.rt2_op) { g_fz.phase = 2; g_rt_merged = 0; PD("F %llu 合并通道进入后一个通道的绘制", g_stat_pass); } p += h.size; continue; }   // A pure-clear render pass already executed early / the start of a later pass already merged in   // Entering the later pass: the same-target continuation of the earlier pass ends here; the later pass reading the earlier pass's attachment is proven to be same-pixel by fz_p2_draw, and is no longer split by the continuation rule
        const void *a = p + 8;
        switch (h.op) {
            case OP_END: g_exec_cur = NULL;
                if (g_hoist_n && !g_ret_n && !g_nrest) {
                    if (g_fz.active) fz_unskip();
                    if (g_hoist_n) { MLOG("【红】纯清除通道前提：提交结束时还有 %d 条已前提的操作没走到，清空（不应发生）", g_hoist_n); g_hoist_n = 0; }
                }
                return;   // An end-of-pass triggered outside a list (present / CPU readback etc.) must not scan expired command memory: a null start point means store everything
            case OP_BEGIN: break;
            case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); p = t; continue; }
            case OP_CALL: { uint64_t t; memcpy(&t, a, 8);
                if (g_ret_n < 16) g_ret[g_ret_n] = p + h.size; g_ret_n++;   // Return stack for lookahead: after a sub-list ends, continue looking at the caller
                mtl_exec((const uint8_t *)t); g_ret_n--; break; }
            case OP_SIGNAL_EVENT: { volatile uint32_t *val; uint32_t m, v; memcpy(&val, a, 8); memcpy(&m, (const uint8_t *)a + 8, 4); memcpy(&v, (const uint8_t *)a + 12, 4);
                PD("Y signal_event cmd=%p dst=%p mode=%u value=%u", p, val, m, v); mtl_sub_event(val, m, v); break; }
            case OP_REPORT_COUNTER: { uint64_t t, ad; memcpy(&t, a, 8); memcpy(&ad, (const uint8_t *)a + 8, 8); g_cnt_report[t & 31]++;
                PD("Q report type=%llu cmd=%p dst=%#llx", (unsigned long long)t, p, (unsigned long long)ad);
                if (t == NVN_COUNTER_TYPE_SAMPLES_PASSED) { post((post_op){3, NULL, 0, 0, ad, (uint32_t)t, vis_snapshot()}); g_vis_cur = -1; }
                else mtl_sub_counter(ad, (uint32_t)t);
                break; }
            case OP_RESET_COUNTER: { uint64_t t; memcpy(&t, a, 8); g_cnt_reset[t & 31]++; PD("Q reset type=%llu cmd=%p", (unsigned long long)t, p);
                if (t == NVN_COUNTER_TYPE_SAMPLES_PASSED) {
#ifdef NVN_RT_RANGE_TEST
                    rr_reset();
#endif
                    g_vis_on = 1; g_vis_counting = 1; vis_clear(); }
                break; }
            case OP_CLEAR_BUF: { const p_clear_buf *cb = a; mtl_sub_fill(cb->addr, cb->size, cb->value); break; }
            case OP_COPY_B2B: { uint64_t s_, d, n; memcpy(&s_, a, 8); memcpy(&d, (const uint8_t *)a + 8, 8); memcpy(&n, (const uint8_t *)a + 16, 8);
                PD("Y copy_b2b cmd=%p src=%#llx dst=%#llx bytes=%llu", p, (unsigned long long)s_, (unsigned long long)d, (unsigned long long)n); mtl_sub_copy(s_, d, n); break; }
            case OP_UPDATE_UB: { uint64_t d, n; memcpy(&d, a, 8); memcpy(&n, (const uint8_t *)a + 8, 8);
                PD("Y update_ub cmd=%p src=%#llx dst=%#llx bytes=%llu f=%g,%g,%g,%g", p, (unsigned long long)((const uint8_t *)a + 16), (unsigned long long)d, (unsigned long long)n,
                   n >= 4 ? ((const float *)((const uint8_t *)a + 16))[0] : 0.f, n >= 8 ? ((const float *)((const uint8_t *)a + 16))[1] : 0.f, n >= 12 ? ((const float *)((const uint8_t *)a + 16))[2] : 0.f, n >= 16 ? ((const float *)((const uint8_t *)a + 16))[3] : 0.f); mtl_sub_update((const uint8_t *)a + 16, d, n); break; }
            case OP_SET_RT: {
                const p_set_rt *r = a;
                if (defer_begin_q(p)) break;
                int pend = R.clear_d || R.clear_s; for (int i = 0; i < 8; i++) pend |= R.clear_c[i];
                if (g_enc && !pend && R.has_rt && tbdr_on() && rt_range_same(&R.rt,r)) {
                    g_rt_merged = g_rt_range_active = g_rt_range_pending = 1; R.rt = *r; break; }
                if (g_enc && g_fz.active && g_fz.phase == 1 && fz_reuse_op(p) && !pend && g_enc_mode != 1 && R.has_rt && tbdr_on() && rt_same(&g_fz.rt1, r)) {
                    g_rt_merged = 1; if (!g_rt_merge_n++) MLOG("同目标续用：首例 通道 %llu", g_stat_pass); PD("M %llu 同目标续用通道", g_stat_pass); break; }
                // Do not continue a render pass opened only for a clear; one with fewer depth / stencil attachments can be continued (check the planes before drawing)
                if (g_enc && g_enc_mode != 1 && !pend && R.has_rt && tbdr_on() && rt_same(&R.rt, r)) { g_rt_merged = 1; if (!g_rt_merge_n++) MLOG("同目标续用：首例 通道 %llu", g_stat_pass); PD("M %llu 同目标续用通道", g_stat_pass); R.rt = *r; break; }
                rp_flush(); sub_cut(); R.rt = *r; R.has_rt = 1; break;
            }
            case OP_CLEAR_COLOR: {
                const p_clear_color *c = a;
                // A scissored clear to 0 outside a pass with all channels (SMBW clears pyramid levels as sub-rectangles of 480×272 and other targets): copy from the zero buffer into the scissor rectangle in the replay stream
                if (!g_enc && c->mask == 0xf && !c->rgba[0] && !c->rgba[1] && !c->rgba[2] && !c->rgba[3] && R.has_sc && c->index >= 0 && c->index < R.rt.n && R.rt.color[c->index]
                    && !scissor_full(R.rt.color[c->index], &R.rt.cview[c->index], R.rt.has_cview[c->index]) && clear_rect_zero(c->index)) break;
                if (g_rt_merged) rp_end();   // A clear arrived in a continued render pass: end it here, and the clear is attached to the start of the next render pass as usual
                if (c->index < 0 || c->index >= R.rt.n || !R.rt.color[c->index]) { RED_ONCE(5, "清色目标 %d 无效", c->index); break; }
                if (g_enc || c->mask != 0xf || !scissor_full(R.rt.color[c->index], &R.rt.cview[c->index], R.rt.has_cview[c->index])) {
                    RED_ONCE(6, "通道中途 / 部分掩码 / 部分剪裁的清色未实现（首例 掩码 %#x）", c->mask);
                    static uint64_t n; if (getenv("DM_NVN_CLEARLOG") && n++ < 40) MLOG("清色跳过 帧%llu 通道%llu 目标%d ser=%llu %ux%u 中途=%d 掩码=%#x 剪裁=%d,%d,%d,%d 色=%g,%g,%g,%g", (unsigned long long)atomic_load(&g_cur_frame), g_stat_pass, c->index,
                        (unsigned long long)R.rt.color[c->index]->serial, R.rt.color[c->index]->d.width, R.rt.color[c->index]->d.height, g_enc != nil, c->mask, R.sc.x, R.sc.y, R.sc.w, R.sc.h, c->rgba[0], c->rgba[1], c->rgba[2], c->rgba[3]);
                    break; }
                R.clear_c[c->index] = 1; R.cc[c->index] = MTLClearColorMake(c->rgba[0], c->rgba[1], c->rgba[2], c->rgba[3]);
                break;
            }
            case OP_CLEAR_DS: {
                const p_clear_ds *c = a;
                if (g_rt_merged) rp_end();
                // Depth / stencil clear in the middle of a pass: end the current pass and attach the clear to the start of the next one (color attachments load as usual, as a continued pass does above when a clear arrives). It used to be skipped, and later draws depth-tested against stale depth.
                // When the scissor does not cover the depth target, Metal's load-action clear cannot express a partial clear; still skipped
                if (g_enc) {
                    if (R.rt.depth && !scissor_full(R.rt.depth, &R.rt.dview, R.rt.has_dview)) { RED_ONCE(7, "通道中途的剪裁深度模板清除未实现"); break; }
                    { static int once; if (!once++) MLOG("通道中途的深度模板清除：结束本通道，清除挂到下一个通道开头（首例 通道 %llu）", g_stat_pass); }
                    rp_end();
                }
                if (c->depth_mask) { R.clear_d = 1; R.cd = c->depth; }
                if (c->stencil_mask) { R.clear_s = 1; R.cs = (uint32_t)c->stencil; }
                break;
            }
            case OP_DRAW: g_exec_next = p + h.size; g_op_count[25]++; draw(a, NULL); g_exec_next = NULL; break;   // Lookahead start point
            case OP_DRAW_IDX: g_exec_next = p + h.size; g_op_count[26]++; draw(NULL, a); g_exec_next = NULL; break;
            // Do not mark dirty if the state did not change (the game often re-sends the same program / state block / vertex format before each draw; the dirty flag only decides whether to re-fetch the pipeline / depth-stencil object, and equal values mean the same object)
            case OP_SCISSOR: case OP_BIND_PROGRAM: case OP_BIND_STATE: case OP_BIND_VATTR: case OP_BIND_VSTREAM: case OP_BIND_VBUF:
            case OP_BIND_UBO: case OP_BIND_SSBO: case OP_BIND_TEX: case OP_BIND_SAMP: case OP_BIND_IMG: case OP_STENCIL:
            case OP_POLY_OFFSET: case OP_VIEWPORTS: case OP_DEPTH_RANGE: case OP_BLEND_COLOR: case OP_COPY_STRIDE:
                state_op(h.op, a); break;
            case OP_COPY_B2T: copy_b2t(a); break;
            case OP_COPY_T2B: copy_t2b(a); break;
            case OP_COPY_T2T: copy_t2t(a); break;
            case OP_CLEAR_TEX: clear_tex(a); break;
            case OP_DISPATCH: dispatch(a); break;
            case OP_BARRIER: g_op_count[28]++; case OP_SET_TEX_POOL: case OP_SET_SAMP_POOL: break;
            default: break;
        }
        p += h.size;
    }
}

// ---------- Present: draw the swapchain texture to the CAMetalLayer; optional screenshot ----------
static id<MTLRenderPipelineState> g_blit_pso; static MTLPixelFormat g_blit_fmt;
static id<MTLSamplerState> g_blit_samp;
// Ambient light in the bars (same design as Touchable's: 64x36 light map, smoothed across frames, blurred along the edge, seam
// feather). The bars take the picture's own light instead of flat black, so the boundary reads as a glow rather than a cut.
// Cost is fixed and tiny: one 64x36 pass per frame plus the bar pixels only (two scissored full-screen triangles); the picture is
// never touched and nothing changes on the game side.
//   DM_NVN_AMBIENT_BARS=<0..1>   how far the light reaches into the bar (0 = flat black, the previous behaviour)
static id<MTLRenderPipelineState> g_amb_pso, g_amb_map_pso; static MTLPixelFormat g_amb_fmt;
static id<MTLSamplerState> g_amb_samp;
static id<MTLTexture> g_amb_map;        // AMB_MAP_W x AMB_MAP_H light map, RGBA16Float, blended across frames
static id<MTLBuffer> g_present_amb[3];  // the bar pass's constants (one per in-flight present)
static id<MTLBuffer> g_present_map[3];  // the light-map pass's constants
static float g_bar_ambient = -1;        // < 0 = not read yet
#define BAR_AMBIENT_DEFAULT 0.55f
#define AMB_MAP_W 32
#define AMB_MAP_H 18
#define AMB_MAP_SIZE "32x18"
#define AMB_TAU_S 0.06                  // seconds: the bars catch up with the picture in about 3 time constants; a one-frame flash keeps ~1/4
static uint64_t g_present_n;
uint64_t mtl_present_count(void) { return g_present_n; }   // Actual present count (heat attribution DM_HEAT computes the present frame rate)
void mtl_res_snapshot(mtl_res_count *out) {
    memset(out, 0, sizeof *out);
    out->pso = g_pso.n;
    out->cps = g_cps.n;
    out->dss = g_dss.n;
    out->shaders = (uint32_t)(g_shcache ? g_shcache.count : 0);
    os_unfair_lock_lock(&g_c1_lock);
    out->c1_bufs = (uint32_t)(g_c1_banks.count + g_c1_exclusive); out->c1_alloc = g_c1_alloc_bytes; out->c1_data = g_c1_data_bytes;
    os_unfair_lock_unlock(&g_c1_lock);
    pthread_mutex_lock(&g_smu);
    out->live_tex = g_by_serial.n;
    pthread_mutex_unlock(&g_smu);
    pthread_mutex_lock(&g_mu);
    out->pools = (uint32_t)g_npool;
    out->pool_adds = g_pool_adds;
    pthread_mutex_unlock(&g_mu);
    pthread_mutex_lock(&g_rmu);
    out->view_slots = (uint32_t)g_ntreg;
    for (int i = 0; i < g_ntreg; i++) if (g_treg[i].view) out->view_nonnull++;
    for (int i = 0; i < g_nsreg; i++) if (g_sview[i]) out->sview_nonnull++;
    pthread_mutex_unlock(&g_rmu);
}
// Resource counts (read by long-run stability L3): the host calls it per game tick / event (every 250 ticks, loading screen dismissed, exit), not sampled by wall clock / present count.
// Line format: [nvn] 资源计数 拍 <拍号> <触发>：<名> <数值> … ([nvn] resource counts tick <tick number> <trigger>: <name> <value> ...) (names contain no spaces, single-space separated); new fields are appended to the end of this line
void mtl_res_counts(uint32_t tick, const char *why) {
    mtl_res_count c;
    mtl_res_snapshot(&c);
    int dl, du; uint64_t da; nvn_dirty_slot_stats(&dl, &du, &da);
    LOG("[nvn] 资源计数 拍 %u %s：渲染管线 %u 计算管线 %u 着色器 %u 深度模板 %u view非空 %u view槽 %u 存活贴图 %u 内存池存活 %u 内存池累计 %llu 采样器view非空 %u 脏页区间存活 %d 脏页区间累计 %llu c1缓冲 %u c1字节 %llu c1实际 %llu",
        tick, why, c.pso, c.cps, c.shaders, c.dss, c.view_nonnull, c.view_slots, c.live_tex,
        c.pools, (unsigned long long)c.pool_adds, c.sview_nonnull, dl, (unsigned long long)da, c.c1_bufs, (unsigned long long)c.c1_alloc, (unsigned long long)c.c1_data);
}
// Screenshot on request (used by DM_NVN_SHOT_UI): dm_gfx_shot_at(k, path) = take a screenshot at the kth frame after the current present frame; due requests are checked at every frame's present
#define SHOTQ_MAX 8
static struct { uint64_t frame; char path[1024]; } g_shotq[SHOTQ_MAX]; static int g_nshotq;
static pthread_mutex_t g_shotq_mu = PTHREAD_MUTEX_INITIALIZER;
static int present_encode(id<MTLCommandBuffer> cb, id<MTLTexture> src, id<MTLTexture> dst, const int crop[4]);
static int present_encode_m4(id<MTL4CommandBuffer> cb, id<MTLTexture> src, id<MTLTexture> dst, const int crop[4], int uv_i);
// Game picture needed by the whole-window screenshot (front end window_shot; no longer does drawViewHierarchy on a Metal layer with the display link attached): at the next present, the present thread draws the same frame
// with the same encoding as the present (present_encode) into an offscreen texture of the same size as the swapchain, and after the GPU finishes hands out a CGImage in the completion callback (the callback does not block the present). Guarded by g_shotq_mu
static void (^g_grab_cb)(CGImageRef img);
void mtl_grab_next_present(void (^cb)(CGImageRef img)) { pthread_mutex_lock(&g_shotq_mu); g_grab_cb = [cb copy]; pthread_mutex_unlock(&g_shotq_mu); }
static void grab_present(id<MTLTexture> src, const int crop[4], CGSize sz) {
    pthread_mutex_lock(&g_shotq_mu); void (^cb)(CGImageRef) = g_grab_cb; g_grab_cb = nil; pthread_mutex_unlock(&g_shotq_mu);
    if (!cb) return;
    size_t w = (size_t)sz.width, h = (size_t)sz.height, row = w * 4;
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:w height:h mipmapped:NO];
    d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead; d.storageMode = MTLStorageModePrivate;
    id<MTLTexture> out = w && h ? [g_dev newTextureWithDescriptor:d] : nil; id<MTLBuffer> buf = out ? [g_dev newBufferWithLength:row * h options:MTLResourceStorageModeShared] : nil;
    id<MTLCommandBuffer> cb2 = buf ? q_cb() : nil;
    if (!cb2 || !present_encode(cb2, src, out, crop)) { MLOG("【红】整窗截图：游戏画面离屏编码失败（%zux%zu）", w, h); cb(NULL); return; }
    id<MTLBlitCommandEncoder> b = [cb2 blitCommandEncoder];
    [b copyFromTexture:out sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(w, h, 1) toBuffer:buf destinationOffset:0 destinationBytesPerRow:row destinationBytesPerImage:row * h];
    [b endEncoding];
    [cb2 addCompletedHandler:^(id<MTLCommandBuffer> c) { (void)c;
        CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
        CGContextRef ctx = CGBitmapContextCreate(buf.contents, w, h, 8, row, cs, kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Little);
        CGImageRef img = ctx ? CGBitmapContextCreateImage(ctx) : NULL; cb(img);
        CGImageRelease(img); CGContextRelease(ctx); CGColorSpaceRelease(cs); }];
    [cb2 commit];
}
void dm_gfx_shot_at(long k, const char *path) {
    pthread_mutex_lock(&g_shotq_mu);
    if (g_nshotq < SHOTQ_MAX) { g_shotq[g_nshotq].frame = g_present_n + (uint64_t)(k > 0 ? k : 0); strlcpy(g_shotq[g_nshotq].path, path, sizeof g_shotq[0].path); g_nshotq++; }
    else MLOG("【红】截图请求超过 %d 个，丢弃 %s", SHOTQ_MAX, path);
    pthread_mutex_unlock(&g_shotq_mu);
}
// DM_NVN_SHOT_OUT=WIDTHxHEIGHT (for picture quality metrics): the present-frame screenshot instead captures the result "upscaled to this resolution by present_encode", i.e. the picture the player sees on screen; screen=0 (texture dump) is unaffected
static void shot(id<MTLTexture> src, const char *path, int screen, const int crop[4]) {
    static int out_w = -1, out_h;
    if (out_w < 0) {
        out_w = 0;
        const char *e = getenv("DM_NVN_SHOT_OUT"); int w = 0, h = 0; char sep = 0, extra = 0;
        if (e && sscanf(e, "%d%c%d%c", &w, &sep, &h, &extra) == 3 && sep == 'x' && w > 0 && h > 0) { out_w = w; out_h = h; }
    }
    id<MTLTexture> read = src;
    if (out_w && screen) {
        MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:out_w height:out_h mipmapped:NO];
        d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead; d.storageMode = MTLStorageModePrivate;
        id<MTLTexture> out = [g_dev newTextureWithDescriptor:d];
        if (!out) { MLOG("【红】截图输出贴图建不出 %dx%d", out_w, out_h); return; }
        id<MTLCommandBuffer> cb = q_cb();
        if (!present_encode(cb, src, out, crop)) return;
        [cb commit]; [cb waitUntilCompleted];
        read = out;
    }
    if (read.pixelFormat != MTLPixelFormatRGBA8Unorm && read.pixelFormat != MTLPixelFormatRGBA8Unorm_sRGB &&
        read.pixelFormat != MTLPixelFormatBGRA8Unorm && read.pixelFormat != MTLPixelFormatBGRA8Unorm_sRGB) { MLOG("【红】截图不支持格式 %lu", (unsigned long)read.pixelFormat); return; }
    size_t w = read.width, h = read.height, row = w * 4;
    id<MTLBuffer> buf = [g_dev newBufferWithLength:row * h options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = q_cb();
    id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
    [b copyFromTexture:read sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(w, h, 1)
              toBuffer:buf destinationOffset:0 destinationBytesPerRow:row destinationBytesPerImage:row * h];
    [b endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (screen && read == src && atomic_load_explicit(&g_origin_ll, memory_order_relaxed)) {   // lower-left origin: the window image is stored bottom-up (see g_origin_ll)
        uint8_t *p = buf.contents, *tmp = malloc(row);
        if (tmp) { for (size_t y = 0; y < h / 2; y++) { memcpy(tmp, p + y * row, row); memcpy(p + y * row, p + (h - 1 - y) * row, row); memcpy(p + (h - 1 - y) * row, tmp, row); } free(tmp); } }
    int bgra = read.pixelFormat == MTLPixelFormatBGRA8Unorm || read.pixelFormat == MTLPixelFormatBGRA8Unorm_sRGB;
    CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    CGBitmapInfo bi = bgra ? (kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Little) : (CGBitmapInfo)kCGImageAlphaNoneSkipLast;
    CGContextRef ctx = CGBitmapContextCreate(buf.contents, w, h, 8, row, cs, bi);
    CGImageRef img = ctx ? CGBitmapContextCreateImage(ctx) : NULL;
    CFURLRef url = CFURLCreateFromFileSystemRepresentation(NULL, (const UInt8 *)path, strlen(path), false);
    CGImageDestinationRef dst = url ? CGImageDestinationCreateWithURL(url, (__bridge CFStringRef)UTTypePNG.identifier, 1, NULL) : NULL;
    // If any step yields null (directory does not exist / not writable, bitmap context cannot be created), report a failure and return; previously it went straight to CFRelease(NULL) and crashed in CFRelease.cold (hit during a 1952-wide TEXDUMP)
    int ok = 0;
    if (!ctx || !img || !dst) MLOG("【红】截图 %s 失败：%s", path, !ctx ? "位图上下文建不出" : !img ? "取图失败" : "写不出（目录不存在或不可写）");
    else { CGImageDestinationAddImage(dst, img, NULL); ok = CGImageDestinationFinalize(dst); }
    if (dst) CFRelease(dst); if (url) CFRelease(url); CGImageRelease(img); CGContextRelease(ctx); CGColorSpaceRelease(cs);
    dm_shot_written(path, ok);   // Let the host write the 「冻结拍」(frozen tick) line first, then write the 「成功」(success) line: the screenshot script stops the host as soon as it sees 「成功」(success), and with the order reversed the log would lack the 「冻结拍」(frozen tick) line under full load (false failure in the gate check's require_log)
    MLOG("截图 %zux%zu 格式 %lu → %s %s", w, h, (unsigned long)read.pixelFormat, path, ok ? "成功" : "失败");
    if (g_ruler_truth && getenv("DM_NVN_RULER") && !strcmp(getenv("DM_NVN_RULER"), "1") && ok) { static int shots; if (++shots == 2) g_ruler_pending = 1; }
    if (g_defer_ab >= 0 && getenv("DM_NVN_DEFER_AB") && !strcmp(getenv("DM_NVN_DEFER_AB"), "1") && ok) { static int shots; if (++shots == 2) { g_defer_ab_on = 1; MLOG("DM_NVN_DEFER_AB：第二张截图写盘完成，打开推迟合并通道"); } }
    if (g_udc_ab >= 0 && getenv("DM_NVN_UDC_AB") && !strcmp(getenv("DM_NVN_UDC_AB"), "1") && ok) {
        static int shots;
        if (++shots == 2) { g_udc_ab_on = 1; MLOG("DM_NVN_UDC_AB：第二张截图写盘完成，打开 DM_NVN_UNWRITTEN_DC"); }
    }
}
// ---------- DM_NVN_TEXDUMP=frame number:path (%d in the path is replaced with "index*100+mip level"): save each color render target used by that frame as an image ----------
// Diagnostic: before the first render pass of the next frame is opened, wait for the GPU to finish, then draw each one (the mip level used by that frame) by full-screen sampling into RGBA8 and save as PNG (float formats are clamped to [0,1])
static long g_td_frame = -2; static const char *g_td_path; static NSMutableArray *g_td_list; static int g_td_done;
static void td_dump(void);
// DM_NVN_TEXDUMP=size=WxH:path instead picks the frame by content: the first frame that opens a render pass whose first color attachment level is WxH (for one-off work whose frame differs between runs)
static unsigned g_td_w, g_td_h;
void td_note(MTL4RenderPassDescriptor *pd, NSUInteger nc) {
    if (g_td_frame == -2) { const char *e = getenv("DM_NVN_TEXDUMP"), *c = e ? strchr(e, ':') : NULL; g_td_frame = c ? atol(e) : -1; g_td_path = c ? c + 1 : NULL; g_td_list = [NSMutableArray array];
        if (c && !strncmp(e, "size=", 5) && sscanf(e + 5, "%ux%u", &g_td_w, &g_td_h) == 2) g_td_frame = LONG_MAX; }
    if (g_td_frame < 0 || g_td_done) return;
    long f = (long)atomic_load(&g_cur_frame);
    if (g_td_frame == LONG_MAX) { id<MTLTexture> t0 = nc ? pd.colorAttachments[0].texture : nil; NSUInteger lv = pd.colorAttachments[0].level;
        if (t0 && MAX(t0.width >> lv, 1ul) == g_td_w && MAX(t0.height >> lv, 1ul) == g_td_h) { g_td_frame = f; MLOG("[texdump] 首个 %ux%u 渲染目标在帧 %ld：转储这一帧", g_td_w, g_td_h, f); } }
    if (f > g_td_frame) { td_dump(); return; }
    if (f != g_td_frame) return;
    for (NSUInteger i = 0; i < nc && i < 8; i++) {
        id<MTLTexture> t = pd.colorAttachments[i].texture; if (!t) continue;
        uint64_t ser = i < (NSUInteger)R.rt.n && R.rt.color[i] ? R.rt.color[i]->serial : 0;
        NSUInteger lv = pd.colorAttachments[i].level, sl = t.textureType == MTLTextureType2DArray ? pd.colorAttachments[i].slice : 0;
        BOOL seen = NO; for (NSArray *e in g_td_list) if ([e[1] unsignedLongLongValue] == ser && [e[2] unsignedIntegerValue] == lv && [e[3] unsignedIntegerValue] == sl) seen = YES;
        if (!seen) [g_td_list addObject:@[t, @(ser), @(lv), @(sl)]];
    }
}
static void td_dump(void) {
    g_td_done = 1;
    rp_end(); if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
    mtl_finish_gpu();
    static id<MTLRenderPipelineState> ps; static id<MTLSamplerState> sp;
    if (!ps) {
        static const char *src_msl =
            "#include <metal_stdlib>\nusing namespace metal;\n"
            "struct V { float4 p [[position]]; float2 uv; };\n"
            "vertex V vs(uint id [[vertex_id]]) { float2 uv = float2((id << 1) & 2, id & 2); V o; o.p = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1); o.uv = uv; return o; }\n"
            "fragment float4 fs(V in [[stage_in]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]]) { float4 c = t.sample(s, in.uv); return float4(c.rgb, 1); }\n";
        NSError *e = nil; id<MTLLibrary> lib = [g_dev newLibraryWithSource:@(src_msl) options:nil error:&e];
        MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
        d.vertexFunction = [lib newFunctionWithName:@"vs"]; d.fragmentFunction = [lib newFunctionWithName:@"fs"];
        d.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
        ps = [g_dev newRenderPipelineStateWithDescriptor:d error:&e];
        MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new]; sp = [g_dev newSamplerStateWithDescriptor:sd];
        if (!ps) { MLOG("【红】TEXDUMP 管线失败 %s", e.localizedDescription.UTF8String); return; }
    }
    MLOG("[texdump] 帧 %ld 共 %lu 张颜色目标", g_td_frame, (unsigned long)g_td_list.count);
    // DM_NVN_TEXRAW=index,...: save raw bytes by texture index (including 3D; file: %d in the path is replaced with the index, suffix .raw, first-line text header: width height depth format bytes-per-row)
    for (const char *q = getenv("DM_NVN_TEXRAW"); q && *q; q = strchr(q, ',') ? strchr(q, ',') + 1 : q + strlen(q)) {
        if (!strncmp(q, "info", 4)) {   // DM_NVN_TEXRAW=info: the textures remembered by DM_NVN_TEXINFO
            static char buf[256]; size_t o = 0; for (int i = 0; i < g_texinfo_n; i++) o += snprintf(buf + o, sizeof buf - o, "%s%llu", i ? "," : "", (unsigned long long)g_texinfo_ser[i]);
            if (o) { static char env[300]; snprintf(env, sizeof env, "%s", buf); setenv("DM_NVN_TEXRAW", env, 1); q = getenv("DM_NVN_TEXRAW"); MLOG("[texraw] info → %s", q); } else continue;
        }
        uint64_t ser = (uint64_t)atoll(q); id o = serial_get(ser);
        id<MTLTexture> t = [o isKindOfClass:[NvPkg class]] ? ((NvPkg *)o)->tex : o;
        if (!t) { MLOG("[texraw] ser=%llu 不在序号表", ser); continue; }
        int bw, bh; NSUInteger w = t.width, h = t.height, dz = t.depth;
        size_t bpp = t.pixelFormat == MTLPixelFormatRGBA16Float ? 8 : t.pixelFormat == MTLPixelFormatRGBA32Float ? 16 : 4; (void)bw; (void)bh;
        size_t row = w * bpp, img = row * h;
        id<MTLBuffer> buf = [g_dev newBufferWithLength:img * dz options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = q_cb(); id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
        [b copyFromTexture:t sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(w, h, dz) toBuffer:buf destinationOffset:0 destinationBytesPerRow:row destinationBytesPerImage:img];
        [b endEncoding]; [cb commit]; [cb waitUntilCompleted];
        char path[1024]; snprintf(path, sizeof path, g_td_path, (int)ser); strlcat(path, ".raw", sizeof path);
        FILE *f = fopen(path, "wb"); if (f) { fprintf(f, "%lu %lu %lu %lu %zu\n", (unsigned long)w, (unsigned long)h, (unsigned long)dz, (unsigned long)t.pixelFormat, row); fwrite(buf.contents, 1, img * dz, f); fclose(f); }
        MLOG("[texraw] ser=%llu %lux%lux%lu 格式 %lu → %s", ser, (unsigned long)w, (unsigned long)h, (unsigned long)dz, (unsigned long)t.pixelFormat, path);
    }
    for (NSArray *e in g_td_list) {
        id<MTLTexture> t = e[0]; uint64_t ser = [e[1] unsignedLongLongValue]; NSUInteger lv = [e[2] unsignedIntegerValue], sl = [e[3] unsignedIntegerValue];
        // 2D arrays are saved by the attached layer (file index ser*100 + layer*10 + level; layer 0 is named like 2D)
        if (t.textureType != MTLTextureType2D && t.textureType != MTLTextureType2DArray) { MLOG("[texdump] ser=%llu 类型 %lu 跳过", ser, (unsigned long)t.textureType); continue; }
        if (getenv("DM_NVN_SNAPRAW")) {   // Also save the raw bytes (converting an HDR target to an 8-bit image truncates; first-line text header: width height format bytes-per-row)
            NSUInteger rw = MAX(t.width >> lv, 1u), rh = MAX(t.height >> lv, 1u);
            size_t bpp = t.pixelFormat == MTLPixelFormatRGBA16Float ? 8 : t.pixelFormat == MTLPixelFormatRGBA32Float ? 16 : t.pixelFormat == MTLPixelFormatR16Float || t.pixelFormat == MTLPixelFormatR16Unorm ? 2 : 4;
            id<MTLBuffer> rb = [g_dev newBufferWithLength:rw * rh * bpp options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> rc = q_cb(); id<MTLBlitCommandEncoder> bb = [rc blitCommandEncoder];
            [bb copyFromTexture:t sourceSlice:0 sourceLevel:lv sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(rw, rh, 1) toBuffer:rb destinationOffset:0 destinationBytesPerRow:rw * bpp destinationBytesPerImage:rw * rh * bpp];
            [bb endEncoding]; [rc commit]; [rc waitUntilCompleted];
            char rp_[1024]; snprintf(rp_, sizeof rp_, g_td_path, (int)(ser * 100 + lv)); strlcat(rp_, ".raw", sizeof rp_);
            FILE *f = fopen(rp_, "wb"); if (f) { fprintf(f, "%lu %lu %lu %zu\n", (unsigned long)rw, (unsigned long)rh, (unsigned long)t.pixelFormat, rw * bpp); fwrite(rb.contents, 1, rw * rh * bpp, f); fclose(f); }
        }
        id<MTLTexture> v = [t newTextureViewWithPixelFormat:t.pixelFormat textureType:MTLTextureType2D levels:NSMakeRange(lv, 1) slices:NSMakeRange(sl, 1)];
        NSUInteger w = MAX(t.width >> lv, 1u), h = MAX(t.height >> lv, 1u);
        MTLTextureDescriptor *od = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:w height:h mipmapped:NO];
        od.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead; od.storageMode = MTLStorageModePrivate;
        id<MTLTexture> o = [g_dev newTextureWithDescriptor:od];
        id<MTLCommandBuffer> cb = q_cb();
        MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = o; rp.colorAttachments[0].loadAction = MTLLoadActionDontCare; rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> en = [cb renderCommandEncoderWithDescriptor:rp];
        [en setRenderPipelineState:ps]; [en setFragmentTexture:v atIndex:0]; [en setFragmentSamplerState:sp atIndex:0];
        [en drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3]; [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
        char path[1024]; snprintf(path, sizeof path, g_td_path, (int)(ser * 100 + sl * 10 + lv));
        MLOG("[texdump] ser=%llu lv=%lu 层 %lu 格式 %lu %lux%lu", ser, (unsigned long)lv, (unsigned long)sl, (unsigned long)t.pixelFormat, (unsigned long)w, (unsigned long)h);
        shot(o, path, 0, NULL);
    }
    [g_td_list removeAllObjects];
}
// DM_NVN_TEXSNAP=in-frame index,... (together with the frame number and path of DM_NVN_TEXDUMP): as soon as these render passes end, copy their color attachments and save them together with the end-of-frame dump
// (the same target is often cleared and reused within a frame, so the end-of-frame state cannot reflect the output of intermediate passes); file name index = in-frame index*100000 + texture index*100 + mip level
static int g_ts_idx[64], g_ts_n = -1, g_ts_on; static uint64_t g_ts_first, g_ts_curf = ~0ull;
static __strong id<MTLTexture> g_ts_att[8]; static uint64_t g_ts_ser[8]; static NSUInteger g_ts_lv[8]; static int g_ts_na;
void ts_note(MTL4RenderPassDescriptor *pd, NSUInteger nc) {
    if (g_ts_n < 0) { g_ts_n = 0; const char *e = getenv("DM_NVN_TEXSNAP");
        for (const char *q = e; q && *q && g_ts_n < 64; q = strchr(q, ',') ? strchr(q, ',') + 1 : q + strlen(q)) g_ts_idx[g_ts_n++] = atoi(q); }
    g_ts_on = 0; if (!g_ts_n || g_td_frame < 0 || g_td_done) return;
    uint64_t f = atomic_load(&g_cur_frame); if (f != g_ts_curf) { g_ts_curf = f; g_ts_first = g_stat_pass; }
    if ((long)f != g_td_frame) return;
    int idx = (int)(g_stat_pass - g_ts_first); for (int i = 0; i < g_ts_n; i++) if (g_ts_idx[i] == idx) g_ts_on = idx + 1;
    g_ts_na = 0; if (!g_ts_on) return;
    for (NSUInteger i = 0; i < nc && i < 8; i++) { id<MTLTexture> t = pd.colorAttachments[i].texture; if (!t) continue;
        g_ts_att[g_ts_na] = t; g_ts_ser[g_ts_na] = R.rt.color[i] ? R.rt.color[i]->serial : 0; g_ts_lv[g_ts_na] = pd.colorAttachments[i].level; g_ts_na++; }
}
void ts_after(void) {
    if (!g_ts_on) return;
    id<MTLCommandBuffer> dcb = q_cb();
    for (int i = 0; i < g_ts_na; i++) {
        id<MTLTexture> t = g_ts_att[i]; if (t.textureType != MTLTextureType2D && t.textureType != MTLTextureType2DArray) continue;   // 2D arrays take slice 0
        MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:t.pixelFormat width:t.width height:t.height mipmapped:t.mipmapLevelCount > 1];
        d.usage = MTLTextureUsageShaderRead; d.storageMode = MTLStorageModePrivate; d.mipmapLevelCount = t.mipmapLevelCount;
        id<MTLTexture> c = [g_dev newTextureWithDescriptor:d];
        id<MTLBlitCommandEncoder> b = [dcb blitCommandEncoder];
        [b copyFromTexture:t sourceSlice:0 sourceLevel:g_ts_lv[i] toTexture:c destinationSlice:0 destinationLevel:g_ts_lv[i] sliceCount:1 levelCount:1]; [b endEncoding];
        [g_td_list addObject:@[c, @((uint64_t)(g_ts_on - 1) * 1000 + g_ts_ser[i]), @(g_ts_lv[i])]];
    }
    g_ts_on = 0;
    [dcb commit];
}
// Ring of actual presentation times: present sequence number -> presentedTime (seconds; -1 = dropped by the system, not presented)
#define PRING 1024
typedef struct { uint64_t n; double t; } pres_ent;
static pres_ent g_pres[PRING];
// Get the actual presentation time of the nth present (returns 0 if not ready, -1 if dropped)
double mtl_presented_time(uint64_t n) { pres_ent e = g_pres[n % PRING]; return e.n == n ? e.t : 0; }
static _Atomic uint64_t g_drawable_ns;   // Accumulated wall clock waiting for nextDrawable in present (timing instrumentation)
uint64_t mtl_drawable_ns_take(void) { return atomic_exchange(&g_drawable_ns, 0); }
// Maximum refresh rate (Hz) of the screen when there is a window; returns 0 without a window. Mac uses CoreGraphics display modes (thread-safe C interface; the present configuration is read on the game thread)
int mtl_display_hz(void) {
    if (!front_metal_layer()) return 0;
#if TARGET_OS_OSX
    CGDisplayModeRef m = CGDisplayCopyDisplayMode(CGMainDisplayID());
    double hz = m ? CGDisplayModeGetRefreshRate(m) : 0; if (m) CGDisplayModeRelease(m);
    return (int)(hz + 0.5);
#else
    Class c = NSClassFromString(@"UIScreen"); id s = c ? [c valueForKey:@"mainScreen"] : nil;
    return s ? [[s valueForKey:@"maximumFramesPerSecond"] intValue] : 0;
#endif
}
static _Atomic uint64_t g_present_stall_ns;   // For unit tests: an extra pause before each present on the present thread (simulating a drawable that stays unavailable)
static CAMetalLayer *g_present_layer_resident;
void mtl_present_test_stall(uint64_t ns) { g_present_stall_ns = ns; }
static void present_pipeline(MTLPixelFormat format) {
    if (!g_blit_pso || g_blit_fmt != format) {
        static const char *src_msl =
            "#include <metal_stdlib>\nusing namespace metal;\n"
            "struct V { float4 p [[position]]; float2 uv; };\n"
            "vertex V vs(uint id [[vertex_id]]) { float2 uv = float2((id << 1) & 2, id & 2); V o; o.p = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1); o.uv = uv; return o; }\n"
            "struct Crop { float4 rect; float4 edge; };\n"
            "fragment float4 fs(V in [[stage_in]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]], constant Crop& c [[buffer(0)]]) { return t.sample(s, clamp(in.uv * c.rect.zw + c.rect.xy, c.edge.xy, c.edge.zw)); }\n";
        NSError *e = nil;
        id<MTLLibrary> lib = [g_dev newLibraryWithSource:@(src_msl) options:nil error:&e];
        if (!lib) MDIE("出图着色器编译失败 %s", e.localizedDescription.UTF8String);
        MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
        pd.vertexFunction = [lib newFunctionWithName:@"vs"]; pd.fragmentFunction = [lib newFunctionWithName:@"fs"];
        pd.colorAttachments[0].pixelFormat = format;
        g_blit_pso = [g_dev newRenderPipelineStateWithDescriptor:pd error:&e];
        if (!g_blit_pso) MDIE("出图管线失败 %s", e.localizedDescription.UTF8String);
        g_blit_fmt = format;
        MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new]; sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterLinear; sd.supportArgumentBuffers = YES;
        g_blit_samp = [g_dev newSamplerStateWithDescriptor:sd];
    }
}
// Ambient light in the bars, three steps (the design is Touchable's -- see docs/research/touchable.md -- reimplemented on this
// tree's present path):
//  1. Light map: the picture is split into 64 x 36 cells and 8 x 8 samples of each cell are averaged in linear light. The present
//     path blends the new values into the previous map with the time constant AMB_TAU_S, so the bars follow the picture within
//     about three time constants and a one-frame flash keeps only about a quarter of its strength.
//  2. Bars: each bar pixel blurs the light map along the picture edge with a Gaussian whose width grows with the distance from the
//     picture; over-bright light is compressed without a hue change, brightness falls off with distance in linear light and reaches
//     zero smoothly at the screen edge.
//  3. Seam: the part of the bar next to the picture (4% of the picture edge, at most half the bar) starts from the outermost
//     light-map cell (strength 0.75, not compressed) and turns into the light, so the picture edge is not a hard step.
// Only bar pixels run (two scissored full-screen triangles); no pixel of the picture is touched. Cost per frame: one 64x36 pass,
// plus one bilinear-blurred fragment per bar pixel -- a fixed, tiny amount that does not grow with the picture's content.
static void ambient_pipeline(MTLPixelFormat format) {
    if (g_amb_pso && g_amb_fmt == format) return;
    static const char *src_msl =
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "struct V { float4 p [[position]]; };\n"
        "struct A { float4 rect; float4 uv; float4 edge; float4 params; float4 light; };\n"
        "constant float3 lum = float3(0.2126, 0.7152, 0.0722);\n"
        "vertex V vs(uint id [[vertex_id]]) { float2 q = float2((id << 1) & 2, id & 2); V o; o.p = float4(q * float2(2, -2) + float2(-1, 1), 0, 1); return o; }\n"
        "fragment float4 mapfs(V in [[stage_in]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]], constant A& a [[buffer(0)]]) {\n"
        "  float2 cell = 1.0 / a.light.yz;\n"
        "  float2 p0 = floor(in.p.xy) * cell;\n"
        "  float3 sum = float3(0.0);\n"
        "  for (int j = 0; j < 8; ++j) for (int i = 0; i < 8; ++i) {\n"
        "    float2 q = p0 + (float2(i, j) + 0.5) * (cell / 8.0);\n"
        "    sum += pow(max(t.sample(s, clamp(q * a.uv.zw + a.uv.xy, a.edge.xy, a.edge.zw)).rgb, float3(0.0)), float3(2.2));\n"
        "  }\n"
        "  return float4(sum / 64.0, a.light.x);\n"
        "}\n"
        "// One side's glow: 'horiz' picks the top/bottom bars (blur runs along x), otherwise the left/right ones.\n"
        "// The colour at d = 0 is the picture's outermost cell EXACTLY: no extra gain, no compression, no second mix. Every\n"
        "// gradient in here decays outward from that one point, which is what makes the seam invisible:\n"
        "//   colour(0) = edge colour   ->   falloff(0) = 1   ->   falloff decreases monotonically   ->   0 at the screen edge.\n"
        "static float3 glow_side(texture2d<float> m, sampler s, constant A& a, bool horiz, bool first, float t, float d, float bar, float texels) {\n"
        "  float x = saturate(d / max(bar, 1.0));\n"
        "  // The edge colour itself: the outermost cell, blurred only ALONG the edge (its detail is high-frequency, and a smooth\n"
        "  // bar is what the eye expects). Never sampled from outside the picture, so the black bar can never leak in.\n"
        "  float outer = 0.5 / texels; outer = first ? outer : 1.0 - outer;\n"
        "  float se = (1.0 + 3.0 * x) / texels;\n"
        "  float3 e = float3(0.0); float ew = 0.0;\n"
        "  for (int k = -3; k <= 3; ++k) {\n"
        "    float w = exp(-0.5 * float(k * k) / 2.25);\n"
        "    float along = clamp(t + float(k) * se, 0.0, 1.0);\n"
        "    e += w * m.sample(s, horiz ? float2(along, outer) : float2(outer, along)).rgb; ew += w;\n"
        "  }\n"
        "  float3 c = e / ew;\n"
        "  // Farther out, the colour takes in a wider stretch of the picture (still from inside it), which keeps texture in the bar\n"
        "  // without ever reaching the black.\n"
        "  float sigma_t = 1.2 + 4.0 * x;\n"
        "  float depth = (0.6 + 2.4 * x) / texels;\n"
        "  float across = first ? depth : 1.0 - depth;\n"
        "  float3 sum = float3(0.0); float wsum = 0.0;\n"
        "  for (int k = -6; k <= 6; ++k) {\n"
        "    float w = exp(-0.5 * float(k * k) / (sigma_t * sigma_t));\n"
        "    float along = clamp(t + float(k) / texels, 0.0, 1.0);\n"
        "    sum += w * m.sample(s, horiz ? float2(along, across) : float2(across, along)).rgb; wsum += w;\n"
        "  }\n"
        "  float st2 = sigma_t * 3.0;\n"
        "  for (int k = -3; k <= 3; ++k) {\n"
        "    float w = exp(-0.5 * float(k * k) / (st2 * st2)) * 0.5;\n"
        "    float along = clamp(t + float(k) * 2.0 / texels, 0.0, 1.0);\n"
        "    sum += w * m.sample(s, horiz ? float2(along, across) : float2(across, along)).rgb; wsum += w;\n"
        "  }\n"
        "  float3 deep = sum / wsum;\n"
        "  // Where the light comes from, how far out we are: the edge colour is the anchor, the wider average takes over with x.\n"
        "  float3 src = mix(c, max(mix(float3(dot(deep, lum)), deep, 1.15), float3(0.0)), smoothstep(0.0, 0.35, x));\n"
        "  // The decay is the ONLY thing that changes the colour. At d = 0 it is exactly 1, so bar and picture agree at the seam.\n"
        "  // The gain sets how far the light reaches (the decay length), never the colour at the seam.\n"
        "  float reach = max(a.params.w, 0.01);\n"
        "  float fall = exp(-2.2 * x / reach) * (1.0 - smoothstep(0.55, 1.0, x));\n"
        "  return src * fall;\n"
        "}\n"
        "fragment float4 barfs(V in [[stage_in]], texture2d<float> m [[texture(0)]], sampler s [[sampler(0)]], constant A& a [[buffer(0)]]) {\n"
        "  float2 size = a.params.xy, px = in.p.xy;\n"
        "  float4 r = a.rect;\n"
        "  // Outside the picture only: nothing is drawn over the game, so the picture's own pixels are untouched by definition.\n"
        "  if (px.x >= r.x && px.x <= r.x + r.z && px.y >= r.y && px.y <= r.y + r.w) discard_fragment();\n"
        "  float dxo = max(max(r.x - px.x, px.x - (r.x + r.z)), 0.0);\n"
        "  float dyo = max(max(r.y - px.y, px.y - (r.y + r.w)), 0.0);\n"
        "  // Both axes always contribute, weighted by the distance: one axis alone left a hard 45-degree line through the corners.\n"
        "  float wsum = dxo + dyo; if (wsum <= 0.0) wsum = 1.0;\n"
        "  float wx = dxo / wsum, wy = dyo / wsum;\n"
        "  float3 c = float3(0.0);\n"
        "  if (wx > 0.0) {   // left / right bars: the blur runs along y\n"
        "    bool first = px.x < r.x;\n"
        "    float d = first ? r.x - px.x : px.x - r.x - r.z;\n"
        "    float bar = first ? r.x : size.x - r.x - r.z;\n"
        "    c += wx * glow_side(m, s, a, false, first, (px.y - r.y) / r.w, d, bar, a.light.y);\n"
        "  }\n"
        "  if (wy > 0.0) {   // top / bottom bars: the blur runs along x\n"
        "    bool first = px.y < r.y;\n"
        "    float d = first ? r.y - px.y : px.y - r.y - r.w;\n"
        "    float bar = first ? r.y : size.y - r.y - r.w;\n"
        "    c += wy * glow_side(m, s, a, true, first, (px.x - r.x) / r.z, d, bar, a.light.z);\n"
        "  }\n"
        "  c = pow(max(c, float3(0.0)), float3(1.0 / 2.2));\n"
        "  float h = fract(52.9829189 * fract(dot(px, float2(0.06711056, 0.00583715))));\n"
        "  return float4(saturate(c + (h - 0.5) / 255.0), 1.0);\n"
        "}\n";
    NSError *e = nil; g_amb_pso = nil; g_amb_fmt = format;
    id<MTLLibrary> lib = [g_dev newLibraryWithSource:@(src_msl) options:nil error:&e];
    if (!lib) { MLOG("【红】黑边光效着色器编译失败 %s（留白保持纯黑）", e.localizedDescription.UTF8String); return; }
    if (!g_amb_samp) { MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new]; sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterLinear;
        sd.sAddressMode = sd.tAddressMode = MTLSamplerAddressModeClampToEdge; sd.supportArgumentBuffers = YES; g_amb_samp = [g_dev newSamplerStateWithDescriptor:sd]; }
    if (!g_amb_map_pso) {   // the light map is RGBA16Float and blended across frames
        MTLRenderPipelineDescriptor *md = [MTLRenderPipelineDescriptor new];
        md.vertexFunction = [lib newFunctionWithName:@"vs"]; md.fragmentFunction = [lib newFunctionWithName:@"mapfs"];
        MTLRenderPipelineColorAttachmentDescriptor *c = md.colorAttachments[0];
        c.pixelFormat = MTLPixelFormatRGBA16Float; c.blendingEnabled = YES;
        c.sourceRGBBlendFactor = MTLBlendFactorSourceAlpha; c.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        c.sourceAlphaBlendFactor = MTLBlendFactorOne; c.destinationAlphaBlendFactor = MTLBlendFactorZero;
        g_amb_map_pso = [g_dev newRenderPipelineStateWithDescriptor:md error:&e];
        if (!g_amb_map_pso) MLOG("【红】黑边光照图管线失败 %s（留白保持纯黑）", e.localizedDescription.UTF8String);
    }
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction = [lib newFunctionWithName:@"vs"]; pd.fragmentFunction = [lib newFunctionWithName:@"barfs"];
    MTLRenderPipelineColorAttachmentDescriptor *bc = pd.colorAttachments[0];
    bc.pixelFormat = format;
    // Added onto the frame, not written over it: the bars are an overlay and must never replace what is already there (writing over it
    // made the whole picture look different with the effect on, which the A/B pixel diff caught).
    bc.blendingEnabled = YES;
    bc.rgbBlendOperation = MTLBlendOperationAdd; bc.alphaBlendOperation = MTLBlendOperationAdd;
    bc.sourceRGBBlendFactor = MTLBlendFactorOne; bc.destinationRGBBlendFactor = MTLBlendFactorOne;
    bc.sourceAlphaBlendFactor = MTLBlendFactorOne; bc.destinationAlphaBlendFactor = MTLBlendFactorZero;
    g_amb_pso = [g_dev newRenderPipelineStateWithDescriptor:pd error:&e];
    if (!g_amb_pso) { MLOG("【红】黑边光效管线失败 %s（留白保持纯黑）", e.localizedDescription.UTF8String); return; }
}
// DM_AMBIENT_AB=<frames>: diagnostic, alternate the effect on and off every N presents in the same process, so the two halves see
// the same scene and the GPU ring can be differenced frame by frame. Never a multiple of 1/fps once the phase length is chosen so
// (frames here, so it is exact). 0 = off (normal behaviour).
static int ambient_ab(void) {
    static int n = -2;
    if (n == -2) { const char *e = getenv("DM_AMBIENT_AB"); n = e && *e ? atoi(e) : 0; if (n > 0) MLOG("黑边光效 A/B：每 %d 帧交替开关（诊断用）", n); }
    return n;
}
static float ambient_gain(void) {
    if (g_bar_ambient < 0) {
        // The value is the light's REACH: how far into the bar it stays visible, as a fraction of the bar's thickness.
        // It deliberately does not scale the colour -- at the picture's edge the brightness is always the edge's own.
        const char *e = getenv("DM_NVN_AMBIENT_BARS"); float v = BAR_AMBIENT_DEFAULT;
        if (e && *e) v = (float)atof(e);
        if (!(v >= 0)) v = 0; if (v > 1) v = 1;
        g_bar_ambient = v;
        if (v > 0) MLOG("黑边光效：扩散距离 %.2f（DM_NVN_AMBIENT_BARS，0 = 纯黑；画面边缘亮度始终等于画面本身）", v);
    }
    int ab = ambient_ab();
    if (ab > 0) {   // alternate per present, and log which side each present belonged to so the frame times can be split afterwards
        static int ph = -1; int now = (int)(g_present_n / ab) % 2;
        if (now != ph) { ph = now; MLOG("黑边光效 A/B：%llu 帧起 %s", (unsigned long long)g_present_n, now ? "关" : "开"); }
        if (now) return 0;
    }
    return g_bar_ambient;
}
static id<MTLTexture> ambient_map(void) {
    if (g_amb_map) return g_amb_map;
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:AMB_MAP_W height:AMB_MAP_H mipmapped:NO];
    d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead; d.storageMode = MTLStorageModePrivate;
    g_amb_map = [g_dev newTextureWithDescriptor:d]; if (g_amb_map) res_add(g_amb_map);
    if (g_amb_map) MLOG("黑边光照图：%dx%d，跨帧平滑时间常数 %.2f 秒", AMB_MAP_W, AMB_MAP_H, AMB_TAU_S);
    return g_amb_map;
}
// The bar rectangles: top/bottom or left/right, only one of the two kinds ever occurs; sides thinner than 2 pixels are skipped.
static int ambient_bars_n(MTLViewport vp, NSUInteger dw, NSUInteger dh, MTLScissorRect out[4], int inner) {
    (void)inner;   // the glow never runs over the picture: it only fills the bars, so nothing is ever drawn on the game
    int n = 0;
    NSUInteger x0 = (NSUInteger)floor(vp.originX), x1 = (NSUInteger)ceil(vp.originX + vp.width);
    NSUInteger y0 = (NSUInteger)floor(vp.originY), y1 = (NSUInteger)ceil(vp.originY + vp.height);
    if (x1 > dw) x1 = dw; if (y1 > dh) y1 = dh;
    if (inner) {   // extend each bar by the spill distance into the picture, so the inner ramp is not clipped
        NSUInteger k = (NSUInteger)(0.030 * (vp.width < vp.height ? vp.width : vp.height));
        x0 = vp.originX > k ? x0 + k : 0; x1 = x1 + k < dw ? x1 + k : dw;
        y0 = vp.originY > k ? y0 + k : 0; y1 = y1 + k < dh ? y1 + k : dh;
    }
    if (x0 >= 2) out[n++] = (MTLScissorRect){0, 0, x0, dh};                    // left
    if (x1 < dw && dw - x1 >= 2) out[n++] = (MTLScissorRect){x1, 0, dw - x1, dh};   // right
    if (y0 >= 2) out[n++] = (MTLScissorRect){0, 0, dw, y0};                    // top
    if (y1 < dh && dh - y1 >= 2) out[n++] = (MTLScissorRect){0, y1, dw, dh - y1};   // bottom
    return n;
}
static int ambient_bars(MTLViewport vp, NSUInteger dw, NSUInteger dh, MTLScissorRect out[4]) { return ambient_bars_n(vp, dw, dh, out, 0); }
// 12 floats: rect (picture in output pixels), uv (source rect), edge (its texel-centre clamp), params (dw, dh, bars along y, gain), light (alpha, map w, map h).
static void ambient_const(float *o, double dw, double dh, MTLViewport vp, const float uv[8], float gain, float alpha) {
    memset(o, 0, sizeof(float) * 20);
    o[0] = (float)vp.originX; o[1] = (float)vp.originY; o[2] = (float)vp.width; o[3] = (float)vp.height;
    memcpy(o + 4, uv, sizeof(float) * 4); memcpy(o + 8, uv + 4, sizeof(float) * 4);
    o[12] = (float)dw; o[13] = (float)dh; o[14] = vp.originY > 0.5 ? 0.0f : 1.0f; o[15] = gain;
    o[16] = alpha; o[17] = AMB_MAP_W; o[18] = AMB_MAP_H; o[19] = 0;
}
// Blend weight of the present path from the interval between two presents; the first frame or a pause over 0.5 s replaces the map.
static float ambient_alpha(void) {
    static double last;
    double now = CACurrentMediaTime(), dt = now - last; last = now;
    if (!(dt > 0.0) || dt > 0.5) return 1.0f;
    return (float)(1.0 - exp(-dt / AMB_TAU_S));
}
// Whether this frame needs the ambient pass at all (the picture fills the frame = no bars, or the effect is off).
static int ambient_wanted(double dw, double dh, MTLViewport vp, MTLPixelFormat format) {
    if (ambient_gain() <= 0) return 0;
    MTLScissorRect bars[4];
    if (!ambient_bars(vp, (NSUInteger)dw, (NSUInteger)dh, bars)) return 0;
    ambient_pipeline(format);
    return g_amb_pso != nil && ambient_map() != nil;
}
// DM_UI_BAR_DEMO=<0..1>: shrink the picture by this fraction of the frame for one run, to see the effect where a real device has
// bars (the Mac's own drawable is always the canvas's aspect, so otherwise there is nothing to look at). Diagnostic only.
static void ambient_demo(double dw, double dh, MTLViewport *vp) {
    static int got = -1; static double f;
    if (got < 0) { const char *e = getenv("DM_UI_BAR_DEMO"); got = e && *e; f = got ? atof(e) : 0;
        if (f > 0) MLOG("留白演示：画面按 %.2f 缩小（只用于看黑边光效，诊断用）", f); }
    if (f <= 0) return;
    vp->originX = floor(dw * f); vp->originY = floor(dh * f * 0.6);
    vp->width = round(dw * (1 - 2 * f)); vp->height = round(dh * (1 - 1.2 * f));
}
// Present samples only the valid source rectangle declared by NVN; kept separate from the raw swapchain readback screenshot. Edges are clamped to the texel centers of the valid area, so bilinear does not read the black texels outside.
// Present samples the raw bytes stored in the swapchain texture (the Switch's scan-out sends the stored bytes as they are): an sRGB source is replaced by a non-sRGB view of the same storage,
// otherwise sampling decodes to linear first, the BGRA8Unorm layer does not encode again, and the whole picture comes out dark. Other formats are returned as they are (Mario's swapchain is RGBA8 and does not come here).
// View cache of 4 entries: swapchain textures live long and are few; a view holds its parent, so an equal parent pointer means the same object. A replaced view is released after the submissions that used it complete (Metal 4 does not retain resources)
static id<MTLTexture> present_raw_src(id<MTLTexture> src) {
    MTLPixelFormat pf = src.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB ? MTLPixelFormatRGBA8Unorm : src.pixelFormat == MTLPixelFormatBGRA8Unorm_sRGB ? MTLPixelFormatBGRA8Unorm : MTLPixelFormatInvalid;
    if (pf == MTLPixelFormatInvalid) return src;
    static __strong id<MTLTexture> views[4]; static int next; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&mu);
    id<MTLTexture> v = nil;
    for (int i = 0; i < 4 && !v; i++) if (views[i] && views[i].parentTexture == src) v = views[i];
    if (!v && (v = [src newTextureViewWithPixelFormat:pf])) { if (views[next]) defer_obj_unref(views[next]); views[next] = v; next = (next + 1) & 3; }
    pthread_mutex_unlock(&mu);
    if (!v) { RED_ONCE(69, "出图：sRGB 交换链贴图的非 sRGB 视图建不出（格式 %lu），按解码后的值出图", (unsigned long)src.pixelFormat); return src; }
    return v;
}
static int present_encode(id<MTLCommandBuffer> cb, id<MTLTexture> src, id<MTLTexture> dst, const int crop[4]) {
    NSUInteger cx = 0, cy = 0, cw = src.width, ch = src.height;
    if (crop && (crop[2] || crop[3])) {
        if (crop[0] < 0 || crop[1] < 0 || crop[2] <= 0 || crop[3] <= 0 ||
            (uint64_t)crop[0] + crop[2] > src.width || (uint64_t)crop[1] + crop[3] > src.height) {
            MLOG("【红】出图裁切越界：%d,%d %dx%d / 原纹理 %lux%lu", crop[0], crop[1], crop[2], crop[3], (unsigned long)src.width, (unsigned long)src.height); return 0;
        }
        cx = crop[0]; cy = crop[1]; cw = crop[2]; ch = crop[3];
    }
    static uint64_t last_crop[8]; uint64_t meta[8] = {src.width, src.height, cx, cy, cw, ch, dst.width, dst.height};
    if (memcmp(meta, last_crop, sizeof meta)) { memcpy(last_crop, meta, sizeof meta);
        MLOG("出图裁切：原纹理 %lux%lu，有效 %lu,%lu %lux%lu → 目标 %lux%lu", (unsigned long)src.width, (unsigned long)src.height,
             (unsigned long)cx, (unsigned long)cy, (unsigned long)cw, (unsigned long)ch, (unsigned long)dst.width, (unsigned long)dst.height); }
    present_pipeline(dst.pixelFormat);
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    // Keep the game picture's aspect ratio and center it, leaving black bars in the rest (no stretching when the screen ratio differs from the game's, e.g. iPhone about 19.5:9 versus the game's 16:9);
    // when the ratio difference is < 1% (the remainder from the game width being a multiple of 8 in the full-screen setup, e.g. 1952x900 versus 2868x1320 differs by 0.17%) fill directly, leaving no few-pixel border
    double dw = dst.width, dh = dst.height, s = fmin(dw / cw, dh / ch);
    MTLViewport vp = {floor((dw - cw * s) / 2), floor((dh - ch * s) / 2), round(cw * s), round(ch * s), 0, 1};
    if (fabs((dw / dh) / ((double)cw / ch) - 1) < 0.01) vp = (MTLViewport){0, 0, dw, dh, 0, 1};
    ambient_demo(dw, dh, &vp);
    int full = vp.originX == 0 && vp.originY == 0 && vp.width == dw && vp.height == dh;
    static double last_vp[4];   // Log one line when the viewport changes (used by the real-device screen-level acceptance screen_check.py to check the aspect ratio)
    if (memcmp(last_vp, &vp, sizeof last_vp)) { memcpy(last_vp, &vp, sizeof last_vp);
        MLOG("出图视口 %.0f,%.0f %.0fx%.0f / 画布 %.0fx%.0f（游戏画面 %lux%lu）", vp.originX, vp.originY, vp.width, vp.height, dw, dh, (unsigned long)src.width, (unsigned long)src.height); }
    int amb = !full && ambient_wanted(dw, dh, vp, dst.pixelFormat);
    float uv[8] = {(float)cx / src.width, (float)cy / src.height, (float)cw / src.width, (float)ch / src.height,
                   (cx + 0.5f) / src.width, (cy + 0.5f) / src.height, (cx + cw - 0.5f) / src.width, (cy + ch - 0.5f) / src.height};
    if (atomic_load_explicit(&g_origin_ll, memory_order_relaxed)) { uv[1] += uv[3]; uv[3] = -uv[3]; }   // lower-left origin: the window image is stored bottom-up
    id<MTLTexture> map = amb ? ambient_map() : nil;
    // Light map first (its own pass): the picture averaged into 64x36 cells and blended into the previous map
    if (amb) {
        static float mapc[20]; static int mi;
        ambient_const(mapc, dw, dh, vp, uv, ambient_gain(), ambient_alpha());
        MTLRenderPassDescriptor *rd = [MTLRenderPassDescriptor renderPassDescriptor];
        rd.colorAttachments[0].texture = map; rd.colorAttachments[0].storeAction = MTLStoreActionStore;
        rd.colorAttachments[0].loadAction = mapc[16] >= 1.0f ? MTLLoadActionClear : MTLLoadActionLoad;   // never-written content can be NaN
        rd.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
        id<MTLRenderCommandEncoder> ed = [cb renderCommandEncoderWithDescriptor:rd];
        if (ed && g_amb_map_pso) {
            [ed setViewport:(MTLViewport){0, 0, AMB_MAP_W, AMB_MAP_H, 0, 1}];
            [ed setRenderPipelineState:g_amb_map_pso]; [ed setFragmentTexture:present_raw_src(src) atIndex:0]; [ed setFragmentSamplerState:g_amb_samp atIndex:0];
            [ed setFragmentBytes:mapc length:sizeof mapc atIndex:0];
            [ed drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
            [ed endEncoding];
        } else if (ed) [ed endEncoding]; else amb = 0;
        mi = (mi + 1) % 3;
    }
    rp.colorAttachments[0].texture = dst; rp.colorAttachments[0].loadAction = full ? MTLLoadActionDontCare : MTLLoadActionClear;
    rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> en = [cb renderCommandEncoderWithDescriptor:rp];
    if (amb) {   // the bars only: up to four scissored full-screen triangles (before the picture, so the edge is clean)
        MTLScissorRect bars[4]; int nb = ambient_bars(vp, dst.width, dst.height, bars);
        static float barc[20]; ambient_const(barc, dw, dh, vp, uv, ambient_gain(), 1.0f);
        [en setViewport:(MTLViewport){0, 0, dw, dh, 0, 1}];
        [en setRenderPipelineState:g_amb_pso]; [en setFragmentTexture:map atIndex:0]; [en setFragmentSamplerState:g_amb_samp atIndex:0];
        [en setFragmentBytes:barc length:sizeof barc atIndex:0];
        for (int i = 0; i < nb; i++) { [en setScissorRect:bars[i]]; [en drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3]; }
        [en setScissorRect:(MTLScissorRect){0, 0, (NSUInteger)dw, (NSUInteger)dh}];
        [en setViewport:vp];
    }
    [en setViewport:vp];
    [en setRenderPipelineState:g_blit_pso]; [en setFragmentTexture:present_raw_src(src) atIndex:0]; [en setFragmentSamplerState:g_blit_samp atIndex:0];
    [en setFragmentBytes:uv length:sizeof uv atIndex:0];
    [en drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [en endEncoding];
    return 1;
}
static int present_encode_m4(id<MTL4CommandBuffer> cb, id<MTLTexture> src, id<MTLTexture> dst, const int crop[4], int uv_i) {
    NSUInteger cx = 0, cy = 0, cw = src.width, ch = src.height;
    if (crop && (crop[2] || crop[3])) {
        if (crop[0] < 0 || crop[1] < 0 || crop[2] <= 0 || crop[3] <= 0 ||
            (uint64_t)crop[0] + crop[2] > src.width || (uint64_t)crop[1] + crop[3] > src.height) {
            MLOG("【红】出图裁切越界：%d,%d %dx%d / 原纹理 %lux%lu", crop[0], crop[1], crop[2], crop[3], (unsigned long)src.width, (unsigned long)src.height); return 0;
        }
        cx = crop[0]; cy = crop[1]; cw = crop[2]; ch = crop[3];
    }
    present_pipeline(dst.pixelFormat);
    double dw = dst.width, dh = dst.height, s = fmin(dw / cw, dh / ch);
    MTLViewport vp = {floor((dw - cw * s) / 2), floor((dh - ch * s) / 2), round(cw * s), round(ch * s), 0, 1};
    if (fabs((dw / dh) / ((double)cw / ch) - 1) < 0.01) vp = (MTLViewport){0, 0, dw, dh, 0, 1};
    // The picture's rectangle in the output frame: the front end keeps its on-screen controls out of the bars when the aspect leaves
    // any (a 16:9 picture on a 19.5:9 panel leaves a bar at either side). Reported only when it changes, so the common case costs
    // two compares on the present thread.
    { static double last_px = -1, last_pw = -1;
      if (vp.originX != last_px || vp.width != last_pw) {
          last_px = vp.originX; last_pw = vp.width;
          front_set_picture_rect(vp.originX, vp.width);
          MLOG("画面矩形：左边留白 %.0f 像素、画面宽 %.0f 像素（出图 %lux%lu）", vp.originX, vp.width, dst.width, dst.height);
      } }
    ambient_demo(dw, dh, &vp);
    int full = vp.originX == 0 && vp.originY == 0 && vp.width == dw && vp.height == dh;
    int amb = !full && ambient_wanted(dw, dh, vp, dst.pixelFormat);
    float *uv = g_present_uv[uv_i].contents;
    uv[0] = (float)cx / src.width; uv[1] = (float)cy / src.height; uv[2] = (float)cw / src.width; uv[3] = (float)ch / src.height;
    uv[4] = (cx + 0.5f) / src.width; uv[5] = (cy + 0.5f) / src.height; uv[6] = (cx + cw - 0.5f) / src.width; uv[7] = (cy + ch - 0.5f) / src.height;
    if (atomic_load_explicit(&g_origin_ll, memory_order_relaxed)) { uv[1] += uv[3]; uv[3] = -uv[3]; }   // lower-left origin: the window image is stored bottom-up
    id<MTLTexture> map = amb ? ambient_map() : nil;
    float alpha = 1.0f;
    if (amb) {   // light map first (its own pass): the picture averaged into 64x36 cells and blended into the previous map
        if (!g_present_map[uv_i]) { g_present_map[uv_i] = [g_dev newBufferWithLength:sizeof(float) * 20 options:MTLResourceStorageModeShared]; res_add(g_present_map[uv_i]); }
        float uvz[8]; memcpy(uvz, uv, sizeof uvz);
        if (atomic_load_explicit(&g_origin_ll, memory_order_relaxed)) { uvz[1] = (float)cy / src.height; uvz[3] = (float)ch / src.height; }   // the light map is a plain picture sample: no flip
        alpha = ambient_alpha();
        ambient_const(g_present_map[uv_i].contents, dw, dh, vp, uvz, ambient_gain(), alpha);
        MTL4RenderPassDescriptor *rd = [MTL4RenderPassDescriptor new];
        rd.colorAttachments[0].texture = map; rd.colorAttachments[0].storeAction = MTLStoreActionStore;
        rd.colorAttachments[0].loadAction = alpha >= 1.0f ? MTLLoadActionClear : MTLLoadActionLoad;   // never-written content can be NaN
        rd.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
        id<MTL4RenderCommandEncoder> ed = [cb renderCommandEncoderWithDescriptor:rd];
        if (ed && g_amb_map_pso) {
            [ed barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice];
            [ed setArgumentTable:g_present_args atStages:MTLRenderStageFragment];
            [g_present_args setTexture:present_raw_src(src).gpuResourceID atIndex:0];
            [g_present_args setSamplerState:g_amb_samp.gpuResourceID atIndex:0];
            [g_present_args setAddress:g_present_map[uv_i].gpuAddress atIndex:0];
            [ed setViewport:(MTLViewport){0, 0, AMB_MAP_W, AMB_MAP_H, 0, 1}];
            [ed setRenderPipelineState:g_amb_map_pso];
            [ed drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
            m4_end(ed);
        } else { if (ed) m4_end(ed); amb = 0; }
    }
    MTL4RenderPassDescriptor *rp = [MTL4RenderPassDescriptor new];
    rp.colorAttachments[0].texture = dst;
    rp.colorAttachments[0].loadAction = full ? MTLLoadActionDontCare : MTLLoadActionClear;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
    id<MTL4RenderCommandEncoder> en = [cb renderCommandEncoderWithDescriptor:rp];
    if (!en) { MLOG("【红】M4 出图渲染编码器创建失败"); return 0; }
    [en barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice];
    [en setArgumentTable:g_present_args atStages:MTLRenderStageFragment];
    [g_present_args setTexture:present_raw_src(src).gpuResourceID atIndex:0];
    [g_present_args setSamplerState:g_blit_samp.gpuResourceID atIndex:0];
    [g_present_args setAddress:g_present_uv[uv_i].gpuAddress atIndex:0];
    if (amb) {   // the bars only: two scissored full-screen triangles, the picture below is untouched
        if (!g_present_amb[uv_i]) { g_present_amb[uv_i] = [g_dev newBufferWithLength:sizeof(float) * 20 options:MTLResourceStorageModeShared]; res_add(g_present_amb[uv_i]); }
        float uvz[8]; memcpy(uvz, uv, sizeof uvz);
        ambient_const(g_present_amb[uv_i].contents, dw, dh, vp, uvz, ambient_gain(), 1.0f);
        MTLScissorRect bars[4]; int nb = ambient_bars(vp, dst.width, dst.height, bars);
        [en setViewport:(MTLViewport){0, 0, dw, dh, 0, 1}];
        [en setRenderPipelineState:g_amb_pso];
        [g_present_args setTexture:map.gpuResourceID atIndex:0];
        [g_present_args setSamplerState:g_amb_samp.gpuResourceID atIndex:0];
        [g_present_args setAddress:g_present_amb[uv_i].gpuAddress atIndex:0];
        for (int i = 0; i < nb; i++) { [en setScissorRect:bars[i]]; [en drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3]; }
        [en setScissorRect:(MTLScissorRect){0, 0, dst.width, dst.height}];
        [g_present_args setTexture:present_raw_src(src).gpuResourceID atIndex:0];
        [g_present_args setSamplerState:g_blit_samp.gpuResourceID atIndex:0];
        [g_present_args setAddress:g_present_uv[uv_i].gpuAddress atIndex:0];
    }
    [en setViewport:vp]; [en setRenderPipelineState:g_blit_pso];
    [en drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    m4_end(en);
    return 1;
}
static double g_link_it_tin;   // Enqueue time of the frame the display-link callback is presenting (read and written only on the display-link thread)
static int present_now(const tex_t *t, double period, uint64_t frame, uint64_t ticket, const int crop[4], id<CAMetalDrawable> supplied, double target_ts) {
    if (g_present_stall_ns) { struct timespec ts = {0, (long)g_present_stall_ns}; nanosleep(&ts, NULL); }
    g_present_n++;
    if (g_use != USE_OFF && g_use != USE_DONE) use_tick();
    if (bootprof() && (g_present_n == 1 || g_present_n % 120 == 0)) {
        if (g_present_n == 1) g_bp_first_present = bp_now() - g_bp_t0;
        MLOG("[bootprof] t=%.3f s 出图第 %llu 次：metallib 载入 %llu 个 / %.1f ms；管线创建 %llu 个 / %.1f ms（均在提交线程同步）；预热领用 着色器 %llu / 渲染管线 %llu / 计算管线 %llu；"
             "贴图上传 %.1f MB / %llu 张首传 + %llu 次重传 / %.1f ms；首帧出图 t=%.3f s", (bp_now() - g_bp_t0) / 1e9, g_present_n,
             g_bp_lib_n, g_bp_lib_ns / 1e6, g_bp_pso_n, g_bp_pso_ns / 1e6, g_warm_claim_shaders, g_warm_claim_rps, g_warm_claim_cps,
             g_bp_up_bytes / 1048576.0, g_pkg_uploads, g_pkg_reuploads,
             g_up_total_ns / 1e6, g_bp_first_present / 1e9);
    }
    PD("X ser=%llu", t->serial);
    id<MTLTexture> src = tex_mtl(t);
    if (!src) { RED_ONCE(8, "出图贴图建不出 MTLTexture"); return 0; }
    // 在本次解析中新建的贴图（窗口贴图"先 present 后绘制"那种）只进了驻留集，而集合只有一次提交才 commit：
    // 显示命令缓冲会采样到集合外的贴图 → GPU 页错误，之后 macOS 无视整个进程的每一条命令缓冲，窗口恒粉
    // （无窗口的跑测从不 present，所以看不出来）。无新成员时 res_commit 是空操作
    res_commit();
    // DM_NVN_REGLUM=x0,y0,x1,y1:file (normalized coordinates): read back this region on every present and record "present sequence number, present time ns, region mean luminance, subsampled grayscale thumbnail (hex)" (diagnostic: verify that animation speed is independent of frame rate).
    // A separate command buffer is queued after this frame's rendering (the same queue executes in submit order); does not wait for the GPU and does not disturb frame pacing
    static int rl = -1; static float rr[4]; static FILE *rlf; static pthread_mutex_t rlmu = PTHREAD_MUTEX_INITIALIZER;
    if (rl < 0) { const char *e = getenv("DM_NVN_REGLUM"), *c = e ? strchr(e, ':') : NULL;
        rl = c && sscanf(e, "%f,%f,%f,%f", &rr[0], &rr[1], &rr[2], &rr[3]) == 4 && (rlf = fopen(c + 1, "w")) != NULL;
        if (e && !rl) MLOG("【红】DM_NVN_REGLUM 参数或文件无效：%s", e); }
    if (rl && (src.pixelFormat == MTLPixelFormatRGBA8Unorm || src.pixelFormat == MTLPixelFormatBGRA8Unorm || src.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB || src.pixelFormat == MTLPixelFormatBGRA8Unorm_sRGB)) {
        NSUInteger x0 = (NSUInteger)(rr[0] * src.width), y0 = (NSUInteger)(rr[1] * src.height), w = (NSUInteger)(rr[2] * src.width) - x0, h = (NSUInteger)(rr[3] * src.height) - y0;
        id<MTLBuffer> rb = [g_dev newBufferWithLength:w * h * 4 options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = q_cb(); id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
        [b copyFromTexture:src sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(x0, y0, 0) sourceSize:MTLSizeMake(w, h, 1) toBuffer:rb destinationOffset:0 destinationBytesPerRow:w * 4 destinationBytesPerImage:w * h * 4];
        [b endEncoding];
        unsigned long long n = g_present_n, tns = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        [cb addCompletedHandler:^(id<MTLCommandBuffer> c) { (void)c; const uint8_t *q = rb.contents; uint64_t sum = 0;
            uint8_t g[4096]; NSUInteger ng = 0;   // Subsampled grayscale thumbnail (every 4th row / column, at most 4096 points), recorded along with the mean, for per-frame image correlation analysis
            for (NSUInteger y = 0; y < h; y += 4) for (NSUInteger x = 0; x < w && ng < sizeof g; x += 4) { const uint8_t *px = q + 4 * (y * w + x); g[ng++] = (uint8_t)((px[0] + px[1] + px[2]) / 3); }
            for (NSUInteger k = 0; k < w * h; k++) sum += q[4 * k] + q[4 * k + 1] + q[4 * k + 2];
            pthread_mutex_lock(&rlmu); fprintf(rlf, "%llu\t%llu\t%.4f\t", n, tns, sum / (3.0 * w * h));
            for (NSUInteger k = 0; k < ng; k++) fprintf(rlf, "%02x", g[k]);
            fputc('\n', rlf); fflush(rlf); pthread_mutex_unlock(&rlmu); }];
        [cb commit];
    }
    // DM_NVN_SHOT=frame number[,frame number...]:path (%d in the path is replaced with the frame number)
    static long shots[4096]; static int nshot = -1; static const char *shot_path;   // up to 4096 frames (a burst over a frame range, for scoring frame by frame)
    if (nshot < 0) {
        nshot = 0; const char *e = getenv("DM_NVN_SHOT"), *c = e ? strchr(e, ':') : NULL;
        if (c) { shot_path = c + 1; for (const char *q = e; q < c && nshot < 4096; q = strchr(q, ',') ? strchr(q, ',') + 1 : c) shots[nshot++] = atol(q); }
    }
    for (int i = 0; i < nshot; i++) if ((long)g_present_n == shots[i]) { char path[1024]; snprintf(path, sizeof path, shot_path, (int)g_present_n); mtl_finish_gpu(); shot(src, path, 1, crop); }
    pthread_mutex_lock(&g_shotq_mu);
    for (int i = 0; i < g_nshotq;) if (g_present_n >= g_shotq[i].frame) { mtl_finish_gpu(); shot(src, g_shotq[i].path, 1, crop); g_shotq[i] = g_shotq[--g_nshotq]; } else i++;
    pthread_mutex_unlock(&g_shotq_mu);

    CAMetalLayer *layer = (__bridge CAMetalLayer *)front_metal_layer();
    if (!layer) return 0;
    if (g_present_layer_resident != layer) { [g_q4 addResidencySet:layer.residencySet]; g_present_layer_resident = layer; }
#if TARGET_OS_OSX
    if (!layer.displaySyncEnabled) layer.displaySyncEnabled = YES;   // Windowed mode follows the display's vsync to prevent tearing
#endif
    present_pipeline(layer.pixelFormat);   // Same as the original path: the first frame compiles the pipeline before taking the drawable
    if (g_grab_cb) grab_present(src, crop, layer.drawableSize);   // Game picture needed by the whole-window screenshot (peek without the lock first; the real take locks inside the function)
    uint64_t dt0 = now_ns();
    id<CAMetalDrawable> dr = supplied ? supplied : [layer nextDrawable];
    if (!supplied) g_drawable_ns += now_ns() - dt0;
    if (!dr) return 0;
    int alloc_i = (int)(ticket % 3);
    if (g_present_alloc_fence[alloc_i]) {
        pthread_mutex_lock(&g_pq_mu);
        while (g_pq_done < g_present_alloc_fence[alloc_i]) pthread_cond_wait(&g_pq_cv, &g_pq_mu);
        pthread_mutex_unlock(&g_pq_mu);
    }
    [g_present_alloc4[alloc_i] reset];
    [g_q4 waitForDrawable:dr];
    id<MTL4CommandBuffer> cb = g_present_cb[alloc_i];
    [cb beginCommandBufferWithAllocator:g_present_alloc4[alloc_i]];
    if (!present_encode_m4(cb, src, dr.texture, crop, alloc_i)) { [cb endCommandBuffer]; [g_q4 signalDrawable:dr]; return 0; }
    [cb endCommandBuffer];
    MTL4CommitOptions *o = [MTL4CommitOptions new];
    [o addFeedbackHandler:^(id<MTL4CommitFeedback> f) {
        if (f.error) MLOG("【红】上屏命令缓冲错误：%s", f.error.description.UTF8String);
        pthread_mutex_lock(&g_pq_mu); if (ticket > g_pq_done) g_pq_done = ticket; g_pq_inflight--; pthread_cond_broadcast(&g_pq_cv); pthread_mutex_unlock(&g_pq_mu);
    }];
    id<MTL4CommandBuffer> buffers[] = {cb};
    pthread_mutex_lock(&g_pq_mu); g_pq_inflight++; pthread_mutex_unlock(&g_pq_mu);
    [g_q4 commit:buffers count:1 options:o];
    if (target_ts > 0) {
        static double leads[300], e2e[300];   // e2e: game enqueue -> system-scheduled present (including the time waiting in the queue for the display-link callback), the full span of input latency that our side can read
        static uint64_t lead_n;
        leads[lead_n % 300] = (target_ts - CACurrentMediaTime()) * 1000.0; e2e[lead_n % 300] = g_link_it_tin > 0 ? (target_ts - g_link_it_tin) * 1000.0 : 0;
        lead_n++;
        if (lead_n % 300 == 0) {
            double a[300]; memcpy(a, leads, sizeof a);
            for (int i = 1; i < 300; i++) { double v = a[i]; int j = i - 1; while (j >= 0 && a[j] > v) { a[j + 1] = a[j]; j--; } a[j + 1] = v; }
            double b[300]; memcpy(b, e2e, sizeof b);
            for (int i = 1; i < 300; i++) { double v = b[i]; int j = i - 1; while (j >= 0 && b[j] > v) { b[j + 1] = b[j]; j--; } b[j + 1] = v; }
            MLOG("显示节拍 至 %llu：节拍 300、出图 %llu、空节拍 %llu（其中按出图间隔留拍 %llu）、合并丢弃旧帧 %llu、实测 %d Hz；提交领先目标上屏 p50 %.3f ms / 最小 %.3f ms；游戏入队到目标上屏 p50 %.3f ms / 最小 %.3f / 最大 %.3f ms",
                 (unsigned long long)g_link_tick_n, (unsigned long long)g_present_n, (unsigned long long)(g_link_tick_n - g_present_n), (unsigned long long)g_link_held,
                 (unsigned long long)g_link_dropped, g_link_hz_meas, a[149], a[0], b[149], b[0], b[299]);
        }
    }
    [g_q4 signalDrawable:dr];
    g_present_alloc_fence[alloc_i] = ticket;
#if TARGET_OS_SIMULATOR   // The iOS simulator SDK has no presentation-time callback and no minimum display duration (used only for picture acceptance, display-link ticks are not counted)
    (void)period; [dr present];
#else
    uint64_t pn = ticket;   // Record the present time by the game's present sequence number (ticket number): hle's present statistics are taken by "the game's nth present"; in the display link path the actual present count != the game's present count (old frames get merged and dropped)
    [dr addPresentedHandler:^(id<MTLDrawable> d) {   // Actual presentation time (the time this frame is really put on screen, CACurrentMediaTime base)
        CFTimeInterval tt = d.presentedTime;
        g_pres[pn % PRING] = (pres_ent){pn, tt > 0 ? tt : -1};
    }];
    // Each frame is displayed for at least one present period: when the GPU completion time jitters around a 120 Hz refresh boundary, the paired "25 ms + 8.33 ms" stutter no longer appears
    if (target_ts > 0) [dr present];
    else if (period > 0) [dr presentAfterMinimumDuration:period]; else [dr present];
#endif
    (void)frame;
    return 1;
}
// ---------- Present thread ----------
// NVN semantics: QueuePresentTexture only queues the present and returns immediately (the game often calls it while holding a lock; iOS real device: synchronously waiting for nextDrawable inside it
// takes about 30 ms, the lock is held about 90% of the time, and the loading thread starves). Taking the drawable, blit and present are all done on this thread in enqueue order.
// Ticket number = which enqueue it is (starting from 1); g_pq_done = the ticket number that has been processed (blit committed to the GPU queue).
// Swapchain texture return: before AcquireTexture gets a texture it waits for the ticket of that texture's previous present to be processed (mtl_present_wait), after which the commands the game draws into it
// are necessarily ordered after that blit on the GPU queue.
#define PQ 8
typedef struct { tex_t t; double period; uint64_t frame; int crop[4]; double t_in; } pq_item;   // t_in: the time the game handed this frame into the present queue (CACurrentMediaTime)
static pq_item g_pq[PQ];
static uint64_t g_pq_in;
static int present_now(const tex_t *, double, uint64_t, uint64_t, const int[4], id<CAMetalDrawable>, double);

@interface NvMetalDisplayLinkDelegate : NSObject <CAMetalDisplayLinkDelegate>
@end

// Fallback for the display link going silent while in the foreground (two smoke-test hangs before merge: the display link stopped calling back while the app was in the foreground, nobody takes the queued frames, and the game waits forever for its previous present to be processed before "acquiring the next swapchain texture"):
// when mtl_present_wait has waited longer than STALL_NS and the display link has not called back for that long, drop the waiting frame and earlier queued frames as when entering the background (accounted as dropped frames) and let the game keep running; when the display link comes back things return to normal automatically.
// One stall logs one hint line, and one line on recovery (neither is a failure); only when the same stall lasts longer than STALL_RED_NS is it a failure. All under g_pq_mu
#define STALL_NS 100000000ull
#define STALL_RED_NS 2000000000ull
static uint64_t g_stall_t0, g_stall_drop, g_stall_n; static int g_stall_red, g_stall_logged;
// The display link watchdog only recognizes the "active" state: when the App loses active status (pulling down Control Center / notifications, incoming call, system popup) and in the background the system does not call back anyway, and the picture is covered too -- the frame-drop fallback stays as before,
// but it is not counted as stuck and no failure is reported; the stall is measured from the later of "the most recent callback / the most recent activation" (A16 logs: 8–17 seconds inactive and 17 seconds in the background were both reported as 「前台卡住」(stuck in foreground))
__attribute__((weak)) int dm_focus_now(void) { return 1; }   // The host's hle_oe.c provides the strong definition; when graphics unit tests do not link the host it is treated as always active
__attribute__((weak)) uint64_t dm_focus_in_since(void) { return 0; }
@implementation NvMetalDisplayLinkDelegate
- (void)metalDisplayLink:(CAMetalDisplayLink *)link needsUpdate:(CAMetalDisplayLinkUpdate *)update {
    (void)link;
    pthread_mutex_lock(&g_pq_mu);
    if (g_stall_t0) { uint64_t since = dm_focus_in_since(), t = now_ns();
        MLOG("显示节拍恢复：这段停了 %.0f ms（其中激活中 %.0f ms），兜底丢了 %llu 帧（第 %llu 段）", (t - g_stall_t0) / 1e6, (t - (g_stall_t0 > since ? g_stall_t0 : since)) / 1e6, (unsigned long long)g_stall_drop, (unsigned long long)g_stall_n); g_stall_t0 = 0; }
    pthread_mutex_unlock(&g_pq_mu);
    pthread_mutex_lock(&g_link_mu);
    g_link_tick_ns = now_ns();
    g_link_tick_n++;
    // The link's actual rate. The window is 30 ticks (half a second at 60 Hz). A rate is accepted after 3 windows agree with it within 10 Hz:
    // the window's own measurement jitters by a Hz or two (59, 60, 61), so requiring the exact value to repeat would mean the rate is never accepted,
    // and 10 Hz is far finer than the difference the per-frame step cares about (60 vs 120).
    {   uint64_t t = g_link_tick_ns;
        if (!g_link_hz_win_t0) { g_link_hz_win_t0 = t; g_link_hz_win_n = 0; }
        g_link_hz_win_n++;
        uint64_t dt = t - g_link_hz_win_t0;
        if (g_link_hz_win_n >= 30 && dt >= 30000000ull) {   // at least 30 ticks and 30 ms, so the rate is worth something
            int hz = (int)((g_link_hz_win_n - 1) * 1000000000ull / dt);   // n-1 intervals between n ticks
            if (!g_link_hz_meas) {   // nothing accepted yet: the first window is the reference, the second and third have to agree
                if (g_link_hz_cand_n == 0 || hz > g_link_hz_cand + 10 || hz + 10 < g_link_hz_cand) { g_link_hz_cand = hz; g_link_hz_cand_n = 1; }
                else g_link_hz_cand_n++;
                if (g_link_hz_cand_n >= 3) { g_link_hz_meas = hz; g_link_hz_log = t; MLOG("显示节拍：实测 %.0f Hz（请求 %d Hz）", (double)hz, g_dm_present_link == 2 ? 120 : 60); }
            } else if (hz > g_link_hz_meas + 10 || hz + 10 < g_link_hz_meas) {   // a different rate: it has to hold for 3 windows before it is taken
                if (g_link_hz_cand_n == 0 || hz > g_link_hz_cand + 10 || hz + 10 < g_link_hz_cand) { g_link_hz_cand = hz; g_link_hz_cand_n = 1; }
                else g_link_hz_cand_n++;
                if (g_link_hz_cand_n >= 3) { g_link_hz_meas = hz; g_link_hz_cand_n = 0; g_link_hz_log = t; MLOG("显示节拍：实测 %.0f Hz（请求 %d Hz）", (double)hz, g_dm_present_link == 2 ? 120 : 60); }
            } else { g_link_hz_cand_n = 0; }   // the same rate as the accepted one: nothing to do
            g_link_hz_win_t0 = t; g_link_hz_win_n = 0;
        }
    }
    pthread_cond_broadcast(&g_link_cv);
    pthread_mutex_unlock(&g_link_mu);
    pthread_mutex_lock(&g_pq_mu);
    if (!g_dm_present_link || g_pq_read == g_pq_in) { pthread_mutex_unlock(&g_pq_mu); return; }
    // The previous frame's present interval was >= 2 (period > 25 ms): keep it on screen for its whole period before taking the next one, as the Switch flips only every N vertical syncs. This also holds while the link runs at 60 Hz (Mario 1.2.1 sets 2 for single frames, see nvn_game_fps in hle_nvn.c);
    //   before, every tick took the newest frame, so that frame, carrying the extra time step the game advanced for it, was shown for only 16.7 ms and the game looked like it jumped ahead
    if (g_link_hold_until > 0 && update.targetPresentationTimestamp < g_link_hold_until - 0.002) { g_link_held++; pthread_mutex_unlock(&g_pq_mu); return; }
    uint64_t first = g_pq_read + 1, ticket = g_pq_in;
    // 按序、每 tick 一帧、最多落后最新一帧：等着的有两帧就先出旧的、新的下一 tick；三帧以上，中间的直接并入丢弃。
    // 每 tick 取最新的做法，在同一刷新里落进两帧时（60 fps 游戏一帧 16–17.5 ms 就会这样）永远丢掉前一帧，
    // 下一帧于是要挂满 33 ms（黑暗之魂重制版，PR #31：14.7–18.7 ms 内 present 98.2 → 99.98%，入队到显示约多 2 ms）。
    // DM_PRESENT_FIFO=0 回到取最新
    static int fifo = -1; if (fifo < 0) { const char *e = getenv("DM_PRESENT_FIFO"); fifo = !(e && !strcmp(e, "0")); MLOG("显示节拍：%s", fifo ? "按序出帧，最多落后一帧" : "每 tick 取最新帧（DM_PRESENT_FIFO=0）"); }
    if (fifo && ticket > first) ticket = ticket - first >= 2 ? g_pq_in - 1 : first;
    pq_item it = g_pq[(ticket - 1) % PQ]; g_link_it_tin = it.t_in;
    g_pq_read = ticket;
    if (ticket > first) { g_link_dropped += ticket - first; for (uint64_t k = first; k < ticket; k++) g_pres[k % PRING] = (pres_ent){k, -1}; }   // Old frames merged and dropped are recorded as "not presented", so the present statistics honestly count them as dropped frames
    if (ticket > 1 && g_pq_done < ticket - 1) g_pq_done = ticket - 1;
    pthread_cond_broadcast(&g_pq_cv);
    pthread_mutex_unlock(&g_pq_mu);
    int submitted;
    @autoreleasepool { submitted = present_now(&it.t, it.period, it.frame, ticket, it.crop, update.drawable, update.targetPresentationTimestamp); }
    g_link_hold_until = submitted && it.period > 0.025 ? update.targetPresentationTimestamp + it.period : 0;
    if (!submitted) {
        pthread_mutex_lock(&g_pq_mu);
        if (ticket > g_pq_done) g_pq_done = ticket;
        pthread_cond_broadcast(&g_pq_cv);
        pthread_mutex_unlock(&g_pq_mu);
    }
}
@end

static CAMetalDisplayLink *g_link; static CFRunLoopRef g_link_rl; static int g_link_started;
static int g_link_owns_layer, g_classic_busy;   // Under g_pq_mu: nextDrawable throws while the layer has the display link object attached, so the present thread that takes its own drawable and the display link object are mutually exclusive
static int link_on(void) { return g_dm_present_link && g_link != nil; }
static CAFrameRateRange link_range(int mode) { return mode == 3 ? CAFrameRateRangeMake(30, 30, 30) : mode == 2 ? CAFrameRateRangeMake(60, 120, 120) : CAFrameRateRangeMake(60, 60, 60); }
static int g_link_mode = 1;   // Current mode of the display link object: 1 / 2 same as g_dm_present_link, 3 = power-saving mode 30 Hz
static void *link_main(void *arg) {   // Display link thread: started when switching to display-link mode, exits when switching back to 0
    (void)arg;
    pthread_setname_np("nvn-present-link");
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    CAMetalLayer *layer = (__bridge CAMetalLayer *)front_metal_layer();
    pthread_mutex_lock(&g_pq_mu); g_link_owns_layer = 1; while (g_classic_busy) pthread_cond_wait(&g_pq_cv, &g_pq_mu); pthread_mutex_unlock(&g_pq_mu);   // Wait for the present thread to finish handing over the frame it holds
    NvMetalDisplayLinkDelegate *delegate = [NvMetalDisplayLinkDelegate new];
    CAMetalDisplayLink *link = [[CAMetalDisplayLink alloc] initWithMetalLayer:layer];
    link.preferredFrameRateRange = link_range(g_link_mode);
    link.preferredFrameLatency = g_present_latency;
    link.delegate = delegate;
    [link addToRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
    g_link_rl = CFRunLoopGetCurrent(); g_link = link;
    CFRunLoopRun();   // When switching back to mode 0 the display link object is invalidated and this thread exits (nextDrawable cannot get a drawable while the layer still has the display link object attached, so pausing alone is not enough)
    g_link_started = 0;
    return NULL;
}
static void link_sync(void) {   // Called every frame by the game's present thread: aligns the display link object's state to g_dm_present_link and the power-saving mode (can change at run time)
    static int applied; int want = g_dm_present_link;
    if (want < 0 || want > 2) { g_dm_present_link = applied == 3 ? 1 : applied; return; }
    if (want && mtl_fps_target() == 30) want = 3;   // Power-saving mode: display link at 30 Hz
    if (want == applied) return;
    g_link_mode = want;
    if (!g_link_started) {
        if (!want) { applied = 0; return; }
        if (!front_metal_layer()) return;
        g_link_started = 1; pthread_t th; if (pthread_create(&th, NULL, link_main, NULL)) MDIE("显示节拍线程起不来"); pthread_detach(th);
        return;   // The display link object is created on a new thread; align again next frame
    }
    CAMetalDisplayLink *link = g_link; if (!link) return;
    if (want) link.preferredFrameRateRange = link_range(want);
    else { CFRunLoopRef rl = g_link_rl; g_link = nil; CFRunLoopPerformBlock(rl, kCFRunLoopDefaultMode, ^{ [link invalidate]; pthread_mutex_lock(&g_pq_mu); g_link_owns_layer = 0; pthread_cond_broadcast(&g_pq_cv); pthread_mutex_unlock(&g_pq_mu); CFRunLoopStop(rl); }); CFRunLoopWakeUp(rl); }
    MLOG("出图节拍模式 %d → %d（帧 %llu）", applied, want, (unsigned long long)atomic_load(&g_cur_frame));
    applied = want;
    pthread_mutex_lock(&g_pq_mu); pthread_cond_broadcast(&g_pq_cv); pthread_mutex_unlock(&g_pq_mu);
}
// Before going to the background (mtl_drain): when the display link drives presents, frames queued but not yet presented can only be taken by the display-link callback, and in the background the system no longer calls back --
// waiting as-is for "the present queue to empty" would wait forever, the enter-background callback would not return, and the system would kill the process (the process disappears within 22 seconds after switching to another App).
// These frames will not be seen anyway: discard them directly (accounted as dropped frames) and only wait for the one present already submitted to the GPU to finish.
static void present_drop_for_suspend(void) {
    pthread_mutex_lock(&g_pq_mu);
    if (g_link_owns_layer) {
        uint64_t n = g_pq_in - g_pq_read;
        for (uint64_t k = g_pq_read + 1; k <= g_pq_in; k++) g_pres[k % PRING] = (pres_ent){k, -1};
        g_link_dropped += n; g_pq_read = g_pq_in;
        while (g_pq_inflight) pthread_cond_wait(&g_pq_cv, &g_pq_mu);
        if (g_pq_done < g_pq_in) g_pq_done = g_pq_in;
        pthread_cond_broadcast(&g_pq_cv);
        if (n) MLOG("进后台：丢弃排队中未上屏的 %llu 帧（显示节拍在后台不回调）", (unsigned long long)n);
    }
    pthread_mutex_unlock(&g_pq_mu);
}
static void *present_main(void *arg) {
    (void)arg;
    pthread_setname_np("nvn-present");
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    qos_class_t qc = QOS_CLASS_UNSPECIFIED; int rp = 0; pthread_get_qos_class_np(pthread_self(), &qc, &rp); atomic_store(&g_present_qos, (uint32_t)qc);
    for (;;) {
        pthread_mutex_lock(&g_pq_mu);
        while (g_pq_read == g_pq_in || g_link_owns_layer) pthread_cond_wait(&g_pq_cv, &g_pq_mu);   // In display-link mode the display-link callback takes them; an empty queue is judged by "taken"; "done" has to wait for the feedback of the present submit and lags behind taken
        pq_item it = g_pq[g_pq_read % PQ];
        uint64_t ticket = ++g_pq_read; g_classic_busy = 1;
        pthread_mutex_unlock(&g_pq_mu);
        int submitted;
        @autoreleasepool { submitted = present_now(&it.t, it.period, it.frame, ticket, it.crop, nil, 0); }
        pthread_mutex_lock(&g_pq_mu); g_classic_busy = 0; pthread_cond_broadcast(&g_pq_cv); pthread_mutex_unlock(&g_pq_mu);
        if (!submitted) {
            pthread_mutex_lock(&g_pq_mu);
            if (ticket > g_pq_done) g_pq_done = ticket;
            pthread_cond_broadcast(&g_pq_cv);
            pthread_mutex_unlock(&g_pq_mu);
        }
    }
    return NULL;
}
uint32_t mtl_present_qos(void) { return atomic_load(&g_present_qos); }
static void present_start(void) { pthread_t th; if (pthread_create(&th, NULL, present_main, NULL)) MDIE("出图线程起不来"); pthread_detach(th); }
static _Atomic int g_present_discard;
void mtl_link_pause(int on);
// The short period when the front end goes to the background and waits for the game to save: presents are dropped directly (the display link does not call back in the background, enqueueing would block the game thread; nobody sees the picture anyway).
// When turned on, also clear the frames already queued: the system first stops the display-link callback and only afterwards notifies about going to the background, and in that gap the game has already filled the swapchain and is stuck at "before acquiring the next texture, wait for its previous present to be processed"
// (hle_nvn.c nvnWindowAcquireTexture -> mtl_present_wait). Without clearing, the game cannot advance a single tick during those 2.5 seconds of waiting for the save, and the save is not written (the same pattern three times in player logs: 3 frames queued, 0 submits)
void mtl_present_discard(int on) { atomic_store(&g_present_discard, on != 0); if (on) present_drop_for_suspend(); }
uint64_t mtl_present(const tex_t *t, double period, const int crop[4]) {
    if (atomic_load(&g_present_discard)) return 0;   // Ticket number 0 = no pending present; the swapchain texture acquire does not wait
    if (t) { xv_check(t->serial, 0, 0, "出图"); if (g_xv_npresent < 128) g_xv_present[g_xv_npresent++] = t->serial; }
    static pthread_once_t once = PTHREAD_ONCE_INIT; pthread_once(&once, present_start);
    link_sync();
    {   // Test switch DM_TEST_LINK_STALL=start seconds,duration seconds: counting from the first present, pause the display link at that point (imitating the display link not calling back while in the foreground) and release it when the duration is up. For verifying the fallback; not set in the product
        static double t0, at = -1, dur; static int st; double now = CACurrentMediaTime();
        if (at < 0) { at = 0; const char *e = getenv("DM_TEST_LINK_STALL"); if (e && sscanf(e, "%lf,%lf", &at, &dur) == 2 && at > 0 && dur > 0) { t0 = now; st = 1; } else at = 0; }
        if (st == 1 && link_on() && now - t0 >= at) { st = 2; mtl_link_pause(1); MLOG("DM_TEST_LINK_STALL：暂停显示节拍 %.1f 秒（帧 %llu）", dur, (unsigned long long)atomic_load(&g_cur_frame)); }
        else if (st == 2 && now - t0 >= at + dur) { st = 3; mtl_link_pause(0); MLOG("DM_TEST_LINK_STALL：放开显示节拍（帧 %llu）", (unsigned long long)atomic_load(&g_cur_frame)); }
    }
    pthread_mutex_lock(&g_pq_mu);
    while (g_pq_in - g_pq_done >= PQ) pthread_cond_wait(&g_pq_cv, &g_pq_mu);   // Wait only when the queue is full; normally AcquireTexture blocks first
    pq_item *it = &g_pq[g_pq_in % PQ]; *it = (pq_item){*t, period, atomic_load(&g_cur_frame), {0}, CACurrentMediaTime()};
    if (crop) memcpy(it->crop, crop, sizeof it->crop);   // The current frame's state is enqueued with the ticket number, rather than reading the mutable window late on the present thread
    uint64_t tk = ++g_pq_in;
    pthread_cond_broadcast(&g_pq_cv);
    pthread_mutex_unlock(&g_pq_mu);
    return tk;
}
// Do not wait during discard: the frame enqueued only after the flag was set (the present thread had just passed the flag check) has nobody to take it, and waiting for it would stall again; it is left for the following mtl_drain to clear
// Do not wait indefinitely in display-link mode: see the note at STALL_NS. Only frames "not yet taken by the display link" are dropped; the one already on its way to the screen is waited for until handed to the GPU (same dropping as when entering the background)
void mtl_present_wait(uint64_t tk) {
    pthread_mutex_lock(&g_pq_mu);
    while (g_pq_done < tk && !atomic_load(&g_present_discard)) {
        if (!g_link_owns_layer) { pthread_cond_wait(&g_pq_cv, &g_pq_mu); continue; }
        if (g_pq_read >= tk) { pthread_cond_wait(&g_pq_cv, &g_pq_mu); continue; }   // This one has already been taken by the display link and is being presented: wait for it to be handed to the GPU (the GPU completion callback will wake us)
        struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); ts.tv_nsec += (long)STALL_NS; if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
        // Already determined to be in a stall (when the display link comes back the callback first clears g_stall_t0, then records the display-link tick time): no longer wait 100 ms in vain every frame, drop directly, and the game proceeds by its own display-link wait timeout (about two ticks)
        if (!g_stall_t0 && (pthread_cond_timedwait(&g_pq_cv, &g_pq_mu, &ts) != ETIMEDOUT || g_pq_done >= tk)) continue;
        uint64_t now = now_ns(), last = __atomic_load_n(&g_link_tick_ns, __ATOMIC_RELAXED);
        if (g_pq_read >= tk) continue;
        if (now - last < STALL_NS) { pthread_cond_timedwait(&g_pq_cv, &g_pq_mu, &ts); continue; }   // The display link is still running, it is just not this one's turn yet: keep waiting (bounded, no busy spinning)
        uint64_t n = tk - g_pq_read;
        for (uint64_t k = g_pq_read + 1; k <= tk; k++) g_pres[k % PRING] = (pres_ent){k, -1};
        g_link_dropped += n; g_pq_read = tk;
        while (g_pq_inflight) pthread_cond_wait(&g_pq_cv, &g_pq_mu);
        if (g_pq_done < tk) g_pq_done = tk;
        pthread_cond_broadcast(&g_pq_cv);
        if (!g_stall_t0) { g_stall_t0 = last ? last : now; g_stall_drop = 0; g_stall_red = 0; g_stall_logged = 0; g_stall_n++; }
        g_stall_drop += n;
        int active = dm_focus_now() == 1; uint64_t since = dm_focus_in_since(), from = g_stall_t0 > since ? g_stall_t0 : since;
        if (active && !g_stall_logged && now - from >= STALL_NS) { g_stall_logged = 1;
            MLOG("显示节拍 %.0f ms 没回调（激活中）：丢掉排队未上屏的 %llu 帧，游戏继续（第 %llu 段；节拍回来自动照常）", (now - from) / 1e6, (unsigned long long)n, (unsigned long long)g_stall_n); }
        if (active && !g_stall_red && now - from >= STALL_RED_NS) { g_stall_red = 1; MLOG("【红】显示节拍激活中连续 %.1f 秒没回调，出图全靠兜底丢帧，画面不动", (now - from) / 1e9); }
    }
    pthread_mutex_unlock(&g_pq_mu);
}
// Wait until all queued presents have been processed (used by mtl_finish = nvnQueueFinish etc.): goes through mtl_present_wait, so there is still a fallback when the display link stops (real device: the game called QueueFinish once while the display link was stopped,
// and this used to wait here without a timeout; nobody took the 3 queued frames and the game stalled; those two runs on Mac did not hit this call)
void mtl_present_flush(void) { pthread_mutex_lock(&g_pq_mu); uint64_t h = g_pq_in; pthread_mutex_unlock(&g_pq_mu); mtl_present_wait(h); }
void mtl_stats(uint64_t *passes, uint64_t *draws) { *passes = g_stat_pass; *draws = g_stat_draw; }
void mtl_args_stats(uint64_t *enc, uint64_t *reuse) { *enc = g_stat_args_enc; *reuse = g_stat_args_reuse; }
void mtl_use_stats(uint64_t *call, uint64_t *skip) { *call = 0; *skip = 0; }
uint64_t mtl_ds_omit(void) { return g_stat_ds_omit; }
void mtl_bw(uint64_t *load, uint64_t *store, uint64_t *full) { *load = g_bw_load; *store = g_bw_store; *full = g_pass_full_rw; }
uint64_t mtl_udc_saved_bytes(void) { return g_udc_saved_bytes; }
int mtl_has_window(void) { return front_metal_layer() != NULL; }
int mtl_link_active(void) { return link_on(); }
void mtl_link_pause(int on) { CAMetalDisplayLink *l = g_link; if (l) l.paused = on != 0; }   // For testing (DM_TEST_BG): imitates the system stopping the display-link callback in the background
int mtl_link_wait_tick(uint64_t timeout_ns) {
    if (!mtl_link_active()) return 0;
    // Wait for the display link only when running faster than the display: if a display-link tick has already passed since the previous frame started work (this frame is already behind), start immediately instead of waiting in vain for the next display-link tick --
    // otherwise when the frame time is slightly more than one display-link period it would "miss the tick -> wait another whole tick", locking the frame rate straight to half (measured on Mac under windowed load: runs of dropped frames)
    static uint64_t started_tick;   // Display-link tick number when the previous frame started work (only the game's present thread calls this function)
    // How many display-link ticks one frame takes. Mode 2 asks for a 120 Hz link and one frame takes 2 ticks (the game still outputs 60 frames), but the system may hand out only 60 Hz:
    // then waiting for 2 ticks needs 33.3 ms, which is exactly the caller's timeout, so every frame would wait the full timeout and the frame rate would fall to 30 fps with a 33 ms / 0 ms stutter.
    // The step therefore follows the MEASURED link rate, not the requested mode: a link that measures 60 Hz gets step 1.
    uint64_t step = g_dm_present_link == 2 ? 2 : 1;
    if (g_dm_present_link == 2 && g_link_hz_measured() < 100) step = 1;   // the link is not really running at 120 Hz
    pthread_mutex_lock(&g_link_mu);
    if (g_link_tick_n >= started_tick + step) { started_tick = g_link_tick_n; pthread_mutex_unlock(&g_link_mu); return 1; }
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += timeout_ns / 1000000000ull; ts.tv_nsec += timeout_ns % 1000000000ull;
    if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
    while (g_link_tick_n < started_tick + step) if (pthread_cond_timedwait(&g_link_cv, &g_link_mu, &ts) == ETIMEDOUT) { pthread_mutex_unlock(&g_link_mu); return 0; }
    started_tick = g_link_tick_n;
    pthread_mutex_unlock(&g_link_mu); return 1;
}

static id<CAMetalDrawable> g_prefetched_drawable; // Used for compositing straight onto the drawable
id<CAMetalDrawable> mtl_present_take_prefetched(void) { return g_prefetched_drawable; }

#ifdef NVN_RT_RANGE_TEST
#include "rt_range_check.inc"
#endif

// Called by the watchdog (no progress): writes the state of all blocking wait points on the graphics side to the log -- who is waiting for whom is visible at a glance (after switching to the Metal 4 submit model there was once a case of "ran to 900 frames then stopped, no failure line")
void mtl_wait_state_dump(void) {
    MLOG("等待点状态：主栅栏 已提交 %llu / 已完成 %llu；主分配器栅栏 [%llu %llu %llu]；上传 未完成 %d、分配器占用 [%d %d %d]、序号 %u、有未冲刷批次 %d；出图队列 入 %llu / 完成 %llu；主命令缓冲在编 %d、提交线程已设 %d",
         (unsigned long long)g_fence, (unsigned long long)g_done, (unsigned long long)g_alloc_fence[0], (unsigned long long)g_alloc_fence[1], (unsigned long long)g_alloc_fence[2],
         atomic_load(&g_up_pending), atomic_load(&g_up_alloc_busy[0]), atomic_load(&g_up_alloc_busy[1]), atomic_load(&g_up_alloc_busy[2]), g_up_seq, g_up_cb != nil,
         (unsigned long long)g_pq_in, (unsigned long long)g_pq_done, g_cb != nil, atomic_load(&g_sub_thread_set));
}
void mtl_alloc_stats(uint64_t *waits, uint64_t *grown, int *n) { *waits = g_alloc_waits; *grown = g_alloc_grown; *n = g_alloc_n; }   // Main submit allocator pool: how many times it waited, how many times it grew, how many exist now
#pragma clang attribute pop
