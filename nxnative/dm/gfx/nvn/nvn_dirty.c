// Write-protect dirty-page tracking for guest memory: pool pages of packaged textures (CPU writes, GPU reads) are set read-only after upload, and as soon as the game writes it faults -> mark dirty, unprotect;
// before use, check the write generation of the covered pages and re-upload if newer than at upload time. NVN semantics: a texture descriptor points to pool memory, and the GPU reads the current content every time
// (measured: the game writes streamed assets into texture storage that is still being sampled, without re-registering).
// Signal handler constraint: the page-fault path uses only atomic operations and mprotect, no allocation, no locks.
#include "nvn_dirty.h"
#include "../../host/dm.h"
#include <mach-o/dyld.h>
#include <mach-o/getsect.h>
#include <stdatomic.h>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>
#include <mach/mach_time.h>
#include <sched.h>

// Order of page state and mprotect (from d2r-mobile ef9b1ff13): the side changing protection first sets the state to BUSY and sets the final state only after mprotect has landed; other threads wait when they see BUSY.
//   set read-only: NONE -> BUSY -> mprotect(read-only) -> PROT; unprotect: PROT -> BUSY -> mprotect(writable) -> NONE.
// Invariant: a page whose state is NONE is always writable, one whose state is PROT is always read-only. Setting read-only used to be "record PROT, then mprotect"; in between, another thread's unprotect-before-write
// (the movie decoder thread writing a frame into texture storage, a disk read into a guest buffer) could set the state back to NONE, then the mprotect landed => the page read-only with state NONE: a game thread writing it got only
// "re-execute" from the fault handler and spun forever; a host system call reading into it saw NONE in prep_write, skipped, and failed with EFAULT (reproduced by t_dirty_race).
// BUSY lasts at most one mprotect (plus a few atomic writes when setting read-only). The wait is capped at BUSY_WAIT_MS and reports red beyond it: in the fault handler it then returns "not our page" so the crash handler reports truthfully
enum { PG_NONE = 0, PG_PROT = 1, PG_BUSY = 2 };
enum { BUSY_WAIT_MS = 2000 };
// Tracked ranges = the memory pools (registered by MemoryPoolInitialize). After a pool is destroyed its live=0 slot is reused; the signal handler scans without locks.
#define MAX_RANGE 4096   // was 256: with more memory pools than that the excess pools are untracked and their textures are treated as dirty and re-uploaded on every use (for Diablo's measured peak see "memory pools" in the memory readout line; Zelda has about 300 and hit this). The scan only covers slots in use, so a larger capacity costs nothing
// Ranges are rounded to host pages (16 KB) but guest memory pools are only 4 KB aligned: adjacent pools share their first / last pages, so one address can fall in several registered ranges.
// So: "is this texture tracked" looks for a range that holds its whole span; a fault / host pre-write unprotects and stamps every range covering that page; pool destruction identifies its range by the pool's exact start mem.
// (Previously the first range containing the address was always taken: two SMBW ASTC 6x5 textures whose pool's first page is shared with a small preceding pool were judged untracked and re-uploaded on every use, about 6 MB per frame)
typedef struct { uintptr_t base, end, mem; _Atomic uint8_t *state; _Atomic uint64_t *gen; size_t cap; _Atomic int live; } range_t;
static range_t g_r[MAX_RANGE];
static _Atomic int g_nr;
static uintptr_t g_pg, g_shift;
static _Atomic uint64_t g_now = 1;
static _Atomic uint64_t g_faults, g_preps;
static _Atomic uint64_t g_adds_total;
static _Atomic int g_live, g_live_peak;   // number of registered pools and peak (for the memory readout line)
void nvn_dirty_stat(int *live, int *peak, int *cap) { *live = atomic_load(&g_live); *peak = atomic_load(&g_live_peak); *cap = MAX_RANGE; }
static _Atomic int g_overflow_logged;
static pthread_mutex_t g_slot_mu = PTHREAD_MUTEX_INITIALIZER;   // Pool creation / destruction (guest threads, possibly concurrent) pick slots under a mutex; the page-fault handling path does not touch it
static _Atomic uint64_t g_mp_calls, g_mp_pages, g_mp_ticks;   // This module's mprotect: call count, pages covered, time spent (mach timebase, safe in the page-fault path too)
static void mp(uintptr_t a, size_t len, int prot) {
    uint64_t t = mach_absolute_time(); mprotect((void *)a, len, prot);
    atomic_fetch_add(&g_mp_ticks, mach_absolute_time() - t); atomic_fetch_add(&g_mp_calls, 1); atomic_fetch_add(&g_mp_pages, len >> g_shift);
}

static mach_timebase_info_data_t g_tb;   // obtained when the first pool is registered (the fault path only multiplies and divides)
static _Atomic int g_busy_red;
// DM_NVN_DIRTYWRITER=1: log the writer of each write fault (which code touches the pool pages). The signal handler may only use atomics and mprotect --
// this path only reads the context and logs, and only for the first few faults and then every 1000th.
static _Atomic int g_writer_diag, g_writer_n;
// DM_NVN_DIRTYWRITER=1: count the host's own writes into pool pages, by path (which one keeps the pages dirty)
uint64_t dm_dbg_wr_packaged, dm_dbg_wr_texels, dm_dbg_wr_fs;
void nvn_dirty_writer_counts(uint64_t *packaged, uint64_t *texels, uint64_t *fs) { *packaged = dm_dbg_wr_packaged; *texels = dm_dbg_wr_texels; *fs = dm_dbg_wr_fs; }
// The dyld image that holds an address: the host's own code is in the app binary, but a fault can also land in a shared-cache library
// (the app binary's text ends below 0x101000000, so a host address above that is in the cache). The name is copied into a static buffer:
// the signal handler must not allocate.
static char g_writer_img[160];
static const char *image_of(uintptr_t a) {
    for (uint32_t i = 0; i < _dyld_image_count(); i++) {
        const struct mach_header_64 *mh = (const struct mach_header_64 *)_dyld_get_image_header(i);
        if (!mh) continue;
        uintptr_t base = (uintptr_t)mh;
        // The image's text extent: the __TEXT segment only. Using the largest segment would also cover __DATA and the page zero, which would
        // wrongly claim addresses that belong to a later image.
        uintptr_t end = base;
        const char *p = (const char *)mh + sizeof(struct mach_header_64);
        for (uint32_t c = 0; c < mh->ncmds; c++) {
            const struct load_command *lc = (const struct load_command *)p;
            if (lc->cmd == LC_SEGMENT_64) { const struct segment_command_64 *sg = (const struct segment_command_64 *)lc; if (strncmp(sg->segname, "__TEXT", 6) == 0 && sg->vmaddr + sg->vmsize > end) end = sg->vmaddr + sg->vmsize; }
            p += lc->cmdsize;
        }
        if (a >= base && a < end) { const char *nm = _dyld_get_image_name(i); snprintf(g_writer_img, sizeof g_writer_img, "%s", nm ? nm : "?"); return g_writer_img; }
    }
    return "（不在任何镜像里）";
}
// Wait for this page to leave BUSY. Returns 1 = it has left; 0 = waited past the limit (reported red once). Usable in a signal handler: only atomic reads, mach_absolute_time and sched_yield
static int busy_wait(range_t *r, uintptr_t p) {
    uint64_t t0 = 0;
    for (unsigned n = 1; atomic_load(&r->state[p]) == PG_BUSY; n++) {
        if (n & 0x3ff) continue;
        uint64_t now = mach_absolute_time(); if (!t0) t0 = now;
        if ((now - t0) * g_tb.numer / g_tb.denom > (uint64_t)BUSY_WAIT_MS * 1000000ull) {
            if (!atomic_exchange(&g_busy_red, 1)) LOG("【红】脏页跟踪：页 %p 停在「忙」超过 %d 毫秒（改保护的线程没走完），不再等", (void *)(r->base + (p << g_shift)), BUSY_WAIT_MS);
            return 0; }
        sched_yield();
    }
    return 1;
}
static int live_at(int i) { return atomic_load_explicit(&g_r[i].live, memory_order_acquire); }
// The registered range that holds all of [a, a+len) (used for a texture's tracking / protection / dirty check)
static range_t *find_span(uintptr_t a, size_t len) {
    int n = atomic_load(&g_nr);
    for (int i = 0; i < n; i++) if (live_at(i) && a >= g_r[i].base && a + len <= g_r[i].end) return &g_r[i];
    return NULL;
}
static void unprotect_page(range_t *r, uintptr_t p) {   // The caller has already changed the state from PROT to BUSY
    mp(r->base + (p << g_shift), g_pg, PROT_READ | PROT_WRITE);
    atomic_store(&r->gen[p], atomic_fetch_add(&g_now, 1) + 1);
    atomic_store(&r->state[p], PG_NONE);
}
// Unprotect one page of a range: if protected, unprotect and stamp it; if another thread is changing its protection, wait for it to finish and look again -- if the other side was setting read-only, the page is PROT after the wait and still has to be unprotected
// (an unprotect-before-write never reaches the fault handler, so it must be done here). Returns 1 = this page is managed by this module; 0 = it is not, or the wait on BUSY timed out
static int release_page(range_t *r, uintptr_t p, _Atomic uint64_t *count) {
    for (;;) {
        uint8_t st = PG_PROT;
        if (atomic_compare_exchange_strong(&r->state[p], &st, PG_BUSY)) { unprotect_page(r, p); if (count) atomic_fetch_add(count, 1); return 1; }
        if (st != PG_BUSY) return atomic_load(&r->gen[p]) != 0;   // already NONE: just unprotected by another thread (the fault happened before the unprotect) -> re-execute; pages never protected are not ours
        if (!busy_wait(r, p)) return 0;
    }
}
static int on_fault(void *addr, int is_write, void *uc) {
    if (!is_write) return 0;
    uintptr_t a = (uintptr_t)addr; int ours = 0, n = atomic_load(&g_nr);
    for (int i = 0; i < n; i++)   // Every range sharing the page must be released: releasing only one leaves the other range's state at "protected", and later writes no longer fault and are missed
        if (live_at(i) && a >= g_r[i].base && a < g_r[i].end) ours |= release_page(&g_r[i], (a - g_r[i].base) >> g_shift, &g_faults);
    // DM_NVN_DIRTYWRITER=1: log the writer of each fault (the PC and the link register of the faulting context, as module+offset).
    // Answers "who keeps writing these pool pages": the game's own code, a host thread (the movie decoder, a disk read), or the host's own upload staging.
    if (ours && g_writer_diag) {
        ucontext_t *c = (ucontext_t *)uc;
        uintptr_t pc = c ? (uintptr_t)c->uc_mcontext->__ss.__pc : 0, lr = c ? (uintptr_t)c->uc_mcontext->__ss.__lr : 0;
        module_t *mp_ = pc ? module_of((void *)pc) : NULL, *ml = lr ? module_of((void *)lr) : NULL;
        // Only the first few faults and then every 1000th: the count can reach 15000 in a run
        uint64_t k = atomic_fetch_add(&g_writer_n, 1);
        if (k < 8 || (k % 1000) == 0) {
            char tn[32] = "?"; pthread_getname_np(pthread_self(), tn, sizeof tn);
            LOG("脏页写入 #%llu 地址 %p 线程「%s」pc=%s+%#llx lr=%s+%#llx",
                (unsigned long long)k, (void *)a, tn,
                mp_ ? mp_->name : image_of(pc), mp_ ? (unsigned long long)(pc - (uintptr_t)mp_->base) : (unsigned long long)pc,
                ml ? ml->name : image_of(lr), ml ? (unsigned long long)(lr - (uintptr_t)ml->base) : (unsigned long long)lr);
            // The host's own code: a short stack scan from the frame pointer, so the function that writes the pool pages can be named
            if (!mp_ && c) {
                uintptr_t sp = (uintptr_t)c->uc_mcontext->__ss.__sp, fp = (uintptr_t)c->uc_mcontext->__ss.__fp;
                LOG("脏页写入 #%llu 宿主栈扫描 fp=%#llx sp=%#llx:", (unsigned long long)k, (unsigned long long)fp, (unsigned long long)sp);
                for (int i = 0; i < 24 && fp && !(fp & 7); i++) {
                    uintptr_t ret = ((uintptr_t *)fp)[1];
                    module_t *m = module_of((void *)ret);
                    LOG("  #%d 返回地址 %p %s", i, (void *)ret, m ? m->name : "（宿主）");
                    uintptr_t next = ((uintptr_t *)fp)[0];
                    if (next <= fp || next - fp > (1u << 20)) break;
                    fp = next;
                }
            }
        }
    }
    return ours;
}
static void prep_write(void *addr, size_t len) {
    uintptr_t a = (uintptr_t)addr, e = a + len; int n = atomic_load(&g_nr);
    if (!len) return;
    for (int i = 0; i < n; i++) {
        range_t *r = &g_r[i];
        if (!live_at(i) || e <= r->base || a >= r->end) continue;
        uintptr_t lo = a > r->base ? a : r->base, hi = e < r->end ? e : r->end;
        for (uintptr_t p = (lo - r->base) >> g_shift, q = (hi - 1 - r->base) >> g_shift; p <= q; p++) release_page(r, p, &g_preps);
    }
}

void nvn_dirty_add_range(const void *mem, size_t size) {
    static int inited;
    if (!inited) { inited = 1; g_pg = (uintptr_t)getpagesize(); g_shift = __builtin_ctzl(g_pg); mach_timebase_info(&g_tb); dm_fault_hook_register(on_fault); dm_guest_write_prep_register(prep_write);
        g_writer_diag = getenv("DM_NVN_DIRTYWRITER") != NULL; }
    atomic_fetch_add(&g_adds_total, 1);
    pthread_mutex_lock(&g_slot_mu);
    int n = atomic_load(&g_nr);
    int slot = -1;
    for (int i = 0; i < n; i++) if (!atomic_load_explicit(&g_r[i].live, memory_order_acquire)) { slot = i; break; }
    if (slot < 0) {
        if (n >= MAX_RANGE) {
            int expected = 0;
            if (atomic_compare_exchange_strong(&g_overflow_logged, &expected, 1)) LOG("【红】脏页跟踪区间已满（%d），池 %p 不跟踪", MAX_RANGE, mem);
            pthread_mutex_unlock(&g_slot_mu); return;
        }
        slot = n;
    }
    { int lv = atomic_fetch_add(&g_live, 1) + 1, pk = atomic_load(&g_live_peak); while (lv > pk && !atomic_compare_exchange_weak(&g_live_peak, &pk, lv)) {} }
    range_t *r = &g_r[slot];
    uintptr_t base = (uintptr_t)mem & ~(g_pg - 1), end = ((uintptr_t)mem + size + g_pg - 1) & ~(g_pg - 1);
    size_t np = (end - base) >> g_shift;
    _Atomic uint8_t *state = r->state;
    _Atomic uint64_t *gen = r->gen;
    if (r->cap < np) {
        state = calloc(np, 1); gen = calloc(np, sizeof *gen);
        if (!state || !gen) die("脏页表分配失败（%zu 页）", np);
        // The old array is not freed: the signal handler may still have a very brief stale reader; leak = 9 bytes per page.
        r->cap = np;
    } else {
        memset((void *)state, 0, np); memset((void *)gen, 0, np * sizeof *gen);
    }
    r->base = base; r->end = end; r->mem = (uintptr_t)mem; r->state = state; r->gen = gen;
    if (slot == n) atomic_store(&g_nr, n + 1);
    atomic_store_explicit(&r->live, 1, memory_order_release);
    pthread_mutex_unlock(&g_slot_mu);
}
// Pool destruction: first restore still-protected pages to writable (the pool memory may be reused for something else once returned to the game), then mark invalid
void nvn_dirty_remove_range(const void *mem) {
    pthread_mutex_lock(&g_slot_mu);
    range_t *r = NULL; int n = atomic_load(&g_nr);
    for (int i = 0; i < n && !r; i++) if (live_at(i) && g_r[i].mem == (uintptr_t)mem) r = &g_r[i];   // Match by the pool's exact start: an adjacent pool sharing a page may have the same rounded base
    if (r) { prep_write((void *)r->base, r->end - r->base); atomic_store_explicit(&r->live, 0, memory_order_release); atomic_fetch_sub(&g_live, 1); }
    pthread_mutex_unlock(&g_slot_mu);
}
int nvn_dirty_tracked(const void *addr, size_t len) {
    return find_span((uintptr_t)addr, len) != NULL;
}
// Set read-only: mark the pages BUSY first and mark them PROT only after mprotect has landed (for the order and the invariant see the state notes at the top of the file)
void nvn_dirty_protect(const void *addr, size_t len) {
    range_t *r = len ? find_span((uintptr_t)addr, len) : NULL;
    if (!r) return;
    uintptr_t a = (uintptr_t)addr, p = (a - r->base) >> g_shift, q = (a + len - 1 - r->base) >> g_shift;
    // Consecutive pages are merged into one mprotect (per-page calls reach thousands per frame on the first upload of large textures; character select page startup frame 5: 4888 calls).
    // Pages already PROT / BUSY break the consecutive run: do not protect again, and do not touch pages another thread is changing
    uintptr_t run = 0; int open = 0;
    for (uintptr_t i = p; i <= q + 1; i++) {
        uint8_t st = PG_NONE;
        int took = i <= q && atomic_compare_exchange_strong(&r->state[i], &st, PG_BUSY);
        if (took && !open) { run = i; open = 1; }
        if (!took && open) { mp(r->base + (run << g_shift), (i - run) << g_shift, PROT_READ); for (uintptr_t k = run; k < i; k++) atomic_store(&r->state[k], PG_PROT); open = 0; }
    }
}
uint64_t nvn_dirty_now(void) { return atomic_load(&g_now); }
int nvn_dirty_since(const void *addr, size_t len, uint64_t stamp) {
    if (!len) return 0;
    range_t *r = find_span((uintptr_t)addr, len);
    if (!r) return 1;   // When untracked, conservatively treat as dirty: better to upload more than to show stale content
    uintptr_t a = (uintptr_t)addr, p = (a - r->base) >> g_shift, q = (a + len - 1 - r->base) >> g_shift;
    for (uintptr_t i = p; i <= q; i++) if (atomic_load(&r->gen[i]) > stamp) return 1;   // The generation of the last write before upload is <= the stamp; any later write must be > the stamp
    return 0;
}
// The content of the pages that were written after the stamp.
// A page that was written is not necessarily a page whose content changed: the game re-writes the same bytes into the same pool pages
// (the audio banks and the GPK assets it loads on demand land there over and over). Measured on the device: the content of the two hot
// textures never changed, yet they were re-uploaded 4096 and 8192 times -- 1.4 GB/s of pointless upload traffic, one mprotect per fault.
// The comparison is per page and only over the pages that were actually written, so it stays cheap: the measured hot texture has ONE dirty
// page out of 1024 (8 KB of 16 MB).
// The caller supplies the hash of each page as it was at upload time (page_hashes[i] = the hash of the page at the time of the upload) and gets
// the current hash of every page back (cur_hashes[i]); the caller compares. The page hash is a plain FNV-1a over the page's bytes.
#define NVN_DIRTY_MAX_PAGES 4096
static uint64_t nvn_page_hash(const uint8_t *p, size_t n) { uint64_t h = 1469598103934665603ull; for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 1099511628211ull; return h; }
// Hash the pages of [addr, addr+len) that were written after the stamp into out[] (the page index within the range). Returns the number of pages hashed.
int nvn_dirty_content_hashes(const void *addr, size_t len, uint64_t stamp, uint64_t *out, int max_pages, uint64_t *first_off, uint64_t *last_end) {
    if (first_off) *first_off = 0;
    if (last_end) *last_end = 0;
    if (!len) return 0;
    range_t *r = find_span((uintptr_t)addr, len);
    if (!r) return -1;
    uintptr_t a = (uintptr_t)addr, p = (a - r->base) >> g_shift, q = (a + len - 1 - r->base) >> g_shift;
    int n = 0, dirty = 0;
    for (uintptr_t i = p; i <= q; i++) {
        if (atomic_load(&r->gen[i]) <= stamp) continue;   // not written since the upload
        uintptr_t lo = r->base + (i << g_shift), hi = lo + g_pg;
        if (lo < a) lo = a;
        if (hi > a + len) hi = a + len;
        // The index of the page within the range (so the caller can look up the hash it took at upload time) and its current hash.
        if (n < max_pages) { out[2 * n] = (uint64_t)(i - p); out[2 * n + 1] = nvn_page_hash((const uint8_t *)lo, hi - lo);
            if (first_off && (!dirty || lo - a < *first_off)) *first_off = lo - a; if (last_end && (!dirty || hi - a > *last_end)) *last_end = hi - a; }
        n++; dirty = 1;
    }
    return n;
}
// The hash of every page of the range, in page order (the caller compares the dirty ones against these).
int nvn_dirty_all_hashes(const void *addr, size_t len, uint64_t *out, int max_pages) {
    if (!len) return 0;
    range_t *r = find_span((uintptr_t)addr, len);
    if (!r) return -1;
    uintptr_t a = (uintptr_t)addr, p = (a - r->base) >> g_shift, q = (a + len - 1 - r->base) >> g_shift;
    int n = 0;
    for (uintptr_t i = p; i <= q; i++) {
        if (n >= max_pages) break;
        uintptr_t lo = r->base + (i << g_shift), hi = lo + g_pg;
        if (lo < a) lo = a;
        if (hi > a + len) hi = a + len;
        out[n++] = nvn_page_hash((const uint8_t *)lo, hi - lo);
    }
    return n;
}
int nvn_dirty_span_since(const void *addr, size_t len, uint64_t stamp, uint64_t *pages, uint64_t *first_off, uint64_t *last_end) {
    if (pages) *pages = 0;
    if (first_off) *first_off = 0;
    if (last_end) *last_end = 0;
    if (!len) return 0;
    range_t *r = find_span((uintptr_t)addr, len);
    if (!r) return -1;
    uintptr_t a = (uintptr_t)addr, p = (a - r->base) >> g_shift, q = (a + len - 1 - r->base) >> g_shift;
    int dirty = 0;
    for (uintptr_t i = p; i <= q; i++) if (atomic_load(&r->gen[i]) > stamp) {
        uintptr_t lo = r->base + (i << g_shift), hi = lo + g_pg;
        if (lo < a) lo = a;
        if (hi > a + len) hi = a + len;
        uint64_t bo = lo - a, be = hi - a;
        if (pages) (*pages)++;
        if (first_off && (!dirty || bo < *first_off)) *first_off = bo;
        if (last_end && (!dirty || be > *last_end)) *last_end = be;
        dirty = 1;
    }
    return dirty;
}
void nvn_dirty_slot_stats(int *live, int *used_slots, uint64_t *adds_total) {
    int n = atomic_load(&g_nr), active = 0;
    for (int i = 0; i < n; i++) active += atomic_load_explicit(&g_r[i].live, memory_order_acquire) != 0;
    *live = active; *used_slots = n; *adds_total = atomic_load(&g_adds_total);
}
void nvn_dirty_stats(uint64_t *faults, uint64_t *preps) { *faults = atomic_load(&g_faults); *preps = atomic_load(&g_preps); }
void nvn_dirty_mp_stats(uint64_t *calls, uint64_t *pages, uint64_t *ns) {
    static mach_timebase_info_data_t tb; if (!tb.denom) mach_timebase_info(&tb);
    *calls = atomic_load(&g_mp_calls); *pages = atomic_load(&g_mp_pages); *ns = atomic_load(&g_mp_ticks) * tb.numer / tb.denom;
}
