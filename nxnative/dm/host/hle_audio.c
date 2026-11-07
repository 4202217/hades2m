// nn::audio AudioOut: the game (FMOD nx output layer) appends PCM buffers block by block; once played the system puts them back on the "released" queue and signals the event.
// Host: real audio output = CoreAudio AudioUnit (Mac DefaultOutput / iOS RemoteIO); silent sink = consumes buffers in real time (no sound, pacing matches the real device).
//       iOS defaults to real audio output (the product must have sound), tests that want silence set DM_AUDIO=0 explicitly; Mac defaults to the silent sink (replay / gate checks are silent), DM_AUDIO=1 for real audio output.
// The contents of AudioOut / AudioOutBuffer / AudioOutParameter are read and written only through the interfaces below; the host defines its own internal layout and only writes the front part of the structs.
#include "hle.h"
#include <AudioToolbox/AudioToolbox.h>
#include <TargetConditionals.h>
#include <dispatch/dispatch.h>
#include <stdatomic.h>
#include <unistd.h>

enum { FMT_PCM16 = 2, STATE_STARTED = 0, STATE_STOPPED = 1, RATE = 48000 };
enum { BK_NONE, BK_AU, BK_NULL };   // 现在谁在消费这个输出的队列（aout.backend）

typedef struct { void *ptr; uint64_t cap, size; } gbuf;     // The first 24 bytes written into the guest AudioOutBuffer
#define QMAX 64
#define RELMAX QMAX                    // Released max = number of in-flight buffers (the game must take a buffer back before it can append again), same limit as pending
typedef struct {
    pthread_mutex_t m;
    void *event;                       // Guest SystemEvent: signaled whenever a buffer is released
    int ch;
    _Atomic int state;
    gbuf *q[QMAX]; _Atomic uint32_t q_head, q_tail;       // Pending: written by the game side, read by the output side
    gbuf *rel[RELMAX]; _Atomic uint32_t rel_head, rel_tail; // Released: written by the output side, read by the game side
    uint64_t pos;                      // Bytes already played of the current block
    AudioUnit au;
    int backend;                       // BK_*：现在谁在消费这个输出的队列。Start / Stop / 切后台 / 看门狗只改这个字段（都在 g_aos_lock 下），不再靠 a->au 加 g_suspended 拼凑状态
    pthread_t null_th; _Atomic int null_run;
    dispatch_semaphore_t notify_sem;
    pthread_t notify_th; _Atomic int notify_run, notify_pending, notify_armed;
    int id;                            // open sequence number (#N in the log; tells simultaneous outputs apart, e.g. Engage's Wwise and Unity's built-in FMOD)
    int nlog;                          // Start / Stop / Close lines logged so far (at most 32 per output)
    _Atomic uint64_t under_n, under_b; // underruns: times the pending queue was empty when the callback needed data / zero-filled bytes
    uint64_t st_calls, st_qsum, st_u0, st_ub0; uint32_t st_fmin, st_fmax;   // sound card callback statistics (read and written only on the callback thread)
    _Atomic uint64_t cb_n;             // 本输出的设备回调次数（看门狗据此判断设备还在播）
    uint64_t wd_n, wd_t;               // 看门狗：上次看到的回调次数与那一刻的时刻（在 g_aos_lock 下）
    int stalled, retry_ms; uint64_t retry_at;   // 设备不回调（或根本起不来）：静音接收端先顶上，看门狗在 retry_at 重试 AudioUnit，间隔 1→2→4→8 s
} aout;

// Registry of open outputs: suspended / resumed all at once by the front-end lifecycle (going to background).
// g_aos_lock 同时把守所有后端切换（Start / Stop / 切后台 / 看门狗）：以前 Start 在锁外先标记已播、再挑后端，
// 中间来一次切后台就会让静音接收端和 AudioUnit 一起消费同一条队列（游戏那边听起来就是声音时有时无）
#define AOMAX 8
static aout *g_aos[AOMAX];
static pthread_mutex_t g_aos_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_suspended;
static aout *ao(const void *g) { aout *a = objmap_get(g); if (!a) die("AudioOut %p 不是宿主建的", g); return a; }
// the call site as "module+offset"; "宿主" (host) when the host itself calls (in-process audio renderer)
static const char *site(void *lr, char *buf, size_t n) {
    module_t *m = lr ? module_of(lr) : NULL;
    if (m) snprintf(buf, n, "%s+%#llx", m->name, (unsigned long long)((uint8_t *)lr - m->base)); else snprintf(buf, n, "宿主");
    return buf;
}
// Start / Stop / Close: at most 32 lines per output; middleware calls them only at init, suspend and resume, more means it keeps toggling
static void au_note(aout *a, const char *what, void *lr) {
    char b[64];
    if (a->nlog < 32) { a->nlog++; LOG("audio #%d %s（调用点 %s）%s", a->id, what, site(lr, b, sizeof b), a->nlog == 32 ? "；此后不再记" : ""); }
}

static void notify_release(aout *a) {
    atomic_store_explicit(&a->notify_pending, 1, memory_order_release);
    if (atomic_exchange_explicit(&a->notify_armed, 1, memory_order_acq_rel) == 0)
        dispatch_semaphore_signal(a->notify_sem);
}

static void *event_notify(void *p) {
    aout *a = p;
    for (;;) {
        dispatch_semaphore_wait(a->notify_sem, DISPATCH_TIME_FOREVER);
        if (atomic_exchange_explicit(&a->notify_pending, 0, memory_order_acq_rel) && a->event)
            hle_signal_system_event(a->event);
        atomic_store_explicit(&a->notify_armed, 0, memory_order_release);
        if (!atomic_load_explicit(&a->notify_run, memory_order_acquire)) break;
        if (atomic_load_explicit(&a->notify_pending, memory_order_acquire) &&
            atomic_exchange_explicit(&a->notify_armed, 1, memory_order_acq_rel) == 0)
            dispatch_semaphore_signal(a->notify_sem);
    }
    return NULL;
}

// Take n bytes from the pending queue and hand them to the output side; once a whole block has played, move it to released and notify the ordinary thread.
static void pull(aout *a, uint8_t *dst, uint64_t n) {
    int released = 0;
    uint32_t qh = atomic_load_explicit(&a->q_head, memory_order_relaxed);
    while (n) {
        uint32_t qt = atomic_load_explicit(&a->q_tail, memory_order_acquire);
        if (qh == qt || atomic_load_explicit(&a->state, memory_order_acquire) != STATE_STARTED) {
            if (qh == qt && atomic_load_explicit(&a->state, memory_order_acquire) == STATE_STARTED) { atomic_fetch_add(&a->under_n, 1); atomic_fetch_add(&a->under_b, n); }
            memset(dst, 0, n); break; }
        gbuf *b = a->q[qh & (QMAX - 1)];
        uint64_t k = b->size - a->pos; if (k > n) k = n;
        memcpy(dst, (uint8_t *)b->ptr + a->pos, k);
        dst += k; n -= k; a->pos += k;
        if (a->pos >= b->size) {
            uint32_t rw = atomic_load_explicit(&a->rel_tail, memory_order_relaxed);
            uint32_t rr = atomic_load_explicit(&a->rel_head, memory_order_acquire);
            if (rw - rr == RELMAX) die("AudioOut 已释放队列溢出");
            a->rel[rw & (RELMAX - 1)] = b;
            atomic_store_explicit(&a->rel_tail, rw + 1, memory_order_release);
            qh++;
            atomic_store_explicit(&a->q_head, qh, memory_order_release);
            a->pos = 0; released = 1;
        }
    }
    if (released) notify_release(a);
}

static _Atomic uint64_t g_render_calls;   // Cumulative sound card callbacks (lifecycle verification: must keep growing after resume)
uint64_t dm_audio_render_calls(void) { return atomic_load_explicit(&g_render_calls, memory_order_relaxed); }
static OSStatus au_render(void *ctx, AudioUnitRenderActionFlags *f, const AudioTimeStamp *ts, UInt32 bus, UInt32 frames, AudioBufferList *bl) {
    (void)f; (void)ts; (void)bus; (void)frames;
    aout *a = ctx;
    atomic_fetch_add_explicit(&g_render_calls, 1, memory_order_relaxed); atomic_fetch_add_explicit(&a->cb_n, 1, memory_order_relaxed);
    {
        // Every 2000 callbacks of each output, log one line: frames requested per call, pending blocks at callback start, underruns (check this line on a new device; docs/agents/03-bringup.md "Crackling on iOS only").
        // Kept per output: with two outputs open, combined numbers mislead (one fully fed and one fed by nobody look like "every other callback underruns")
        uint32_t qd = atomic_load(&a->q_tail) - atomic_load(&a->q_head);
        if (!a->st_calls || frames < a->st_fmin) a->st_fmin = frames; if (frames > a->st_fmax) a->st_fmax = frames; a->st_qsum += qd;
        if (++a->st_calls % 2000 == 0) { uint64_t un = atomic_load(&a->under_n), ubb = atomic_load(&a->under_b);
            LOG("audio #%d 声卡回调 %llu 次：每次 %u–%u 帧，回调时待播均 %.1f 块；这 2000 次里欠载 %llu 次、补零 %.1f ms", a->id, (unsigned long long)a->st_calls, a->st_fmin, a->st_fmax, a->st_qsum / 2000.0,
                (unsigned long long)(un - a->st_u0), (ubb - a->st_ub0) / (2.0 * a->ch) / (RATE / 1000.0)); a->st_fmin = ~0u; a->st_fmax = 0; a->st_qsum = 0; a->st_u0 = un; a->st_ub0 = ubb; } }
    pull(a, bl->mBuffers[0].mData, bl->mBuffers[0].mDataByteSize);
    { static int mute = -1; if (mute < 0) { const char *e = getenv("DM_AUDIO_MUTE"); mute = e && atoi(e); }   // diagnostic: the whole real-output path (AudioUnit callback pacing, background take-over) but with the output silenced (lifecycle tests on the Mac without noise)
      if (mute) memset(bl->mBuffers[0].mData, 0, bl->mBuffers[0].mDataByteSize); }
    return noErr;
}
// 建 AudioUnit：返回状态而不是直接退出，因为媒体服务被重置后（iOS）旧的 AudioUnit 全部作废，得能重新建一个（dm_audio_reset）
static OSStatus au_create(aout *a) {
    AudioComponentDescription d = {.componentType = kAudioUnitType_Output,
#if TARGET_OS_IPHONE
                                   .componentSubType = kAudioUnitSubType_RemoteIO,
#else
                                   .componentSubType = kAudioUnitSubType_DefaultOutput,
#endif
                                   .componentManufacturer = kAudioUnitManufacturer_Apple};
    AudioComponent c = AudioComponentFindNext(NULL, &d);
    a->au = NULL; if (!c || AudioComponentInstanceNew(c, &a->au)) { a->au = NULL; return -1; }
    AudioStreamBasicDescription f = {.mSampleRate = RATE, .mFormatID = kAudioFormatLinearPCM,
                                     .mFormatFlags = kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked,
                                     .mBytesPerPacket = 2u * a->ch, .mFramesPerPacket = 1, .mBytesPerFrame = 2u * a->ch,
                                     .mChannelsPerFrame = (UInt32)a->ch, .mBitsPerChannel = 16};
    AURenderCallbackStruct cb = {au_render, a};
#if !TARGET_OS_IPHONE
    { const char *e = getenv("DM_DBG_AU_FRAMES"); if (e) {   // diagnostic: on the Mac, make the sound card callback ask N frames each time (1024 = the iOS default, reproduces the crackling)
        UInt32 fr = (UInt32)atoi(e); OSStatus st = AudioUnitSetProperty(a->au, kAudioDevicePropertyBufferFrameSize, kAudioUnitScope_Global, 0, &fr, sizeof fr);
        LOG("audio 声卡缓冲帧数设为 %u：状态 %d", fr, (int)st); } }
#endif
    OSStatus st = AudioUnitSetProperty(a->au, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &f, sizeof f);
    if (!st) st = AudioUnitSetProperty(a->au, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0, &cb, sizeof cb);
    if (!st) st = AudioUnitInitialize(a->au);
    if (st) { AudioComponentInstanceDispose(a->au); a->au = NULL; }   // 半成品要扔掉：留着会让后面的启停用一个没初始化过的单元
    return st;
}
// OpenAudioOut 跑在游戏的线程上，而建 AudioUnit 要吃约 24 KB 栈（Mac 实测：组件查找、实例、Initialize），有的游戏线程剩不了这么多。
// 所以放到宿主线程上跑，每个 AudioOut 一次。不用 HLE_FORCE_BIG：那会把 open 钩子记的调用点（__builtin_return_address(0)）换成跳板
static void *au_create_th(void *a) { return (void *)(intptr_t)au_create(a); }
static void au_open(aout *a) {
    pthread_t t; void *r = NULL;
    if (pthread_create(&t, NULL, au_create_th, a) || pthread_join(t, &r)) die("AudioUnit 建立：起线程失败");
    OSStatus st = (OSStatus)(intptr_t)r; if (st) die("AudioUnit 设置失败（状态 %d）", (int)st);
}
// Silent sink: consumes in real time at 48 kHz every 5 ms. When more than 100 ms behind (iOS suspended the whole process in the background, SIGSTOP on the Mac), restart the count from now and do not make up the missed time --
// making it up would pull hundreds of times in one go (seconds' worth), releasing all of the game's buffers and logging hundreds of underruns; a suspended console does not play that stretch either
static void *null_sink(void *p) {
    aout *a = p; uint8_t scratch[RATE / 200 * 2 * 8];
    uint64_t bytes = (uint64_t)RATE / 200 * 2 * a->ch, t = now_ns();
    while (atomic_load_explicit(&a->null_run, memory_order_acquire)) {
        t += 5000000; uint64_t now = now_ns();
        if (t > now) usleep((useconds_t)((t - now) / 1000)); else if (now - t > 100000000) t = now;
        pull(a, scratch, bytes);
    }
    return NULL;
}
// 后端切换；调用者持有 g_aos_lock。pull() 不拿锁，所以在锁里 join 静音接收端不会死锁
static void null_start(aout *a) {
    if (a->backend == BK_NULL) return;   // 已经在跑接收端了，别起第二个（两个线程抢同一条队列）
    atomic_store_explicit(&a->null_run, 1, memory_order_release);
    pthread_create(&a->null_th, NULL, null_sink, a);
    a->backend = BK_NULL;
}
static void null_stop(aout *a) {
    if (a->backend != BK_NULL) return;   // 没有接收端在跑（别去 join 一个没起过的线程）
    atomic_store_explicit(&a->null_run, 0, memory_order_release);
    pthread_join(a->null_th, NULL);
    a->backend = BK_NONE;
}
static void backend_stop(aout *a) {
    if (a->backend == BK_AU) { AudioOutputUnitStop(a->au); a->backend = BK_NONE; }
    else if (a->backend == BK_NULL) null_stop(a);
}
// 起 AudioUnit（调用者持有 g_aos_lock）：设备不让起（会话被电话占着，或正在被重置）时不退出，改由静音接收端顶上，看门狗按 1、2、4、8 s 重试
static void au_go(aout *a, uint64_t now) {
    OSStatus st = a->au ? AudioOutputUnitStart(a->au) : -1;
    a->wd_n = atomic_load_explicit(&a->cb_n, memory_order_relaxed); a->wd_t = now;
    if (!st) { a->backend = BK_AU; a->stalled = 0; return; }
    null_start(a);
    if (!a->stalled) { a->stalled = 1; a->retry_ms = 1000; LOG("【红】audio #%d：AudioUnit 起不来（状态 %d）：先由静音接收端顶上，稍后重试", a->id, (int)st); }
    else if (a->retry_ms < 8000) a->retry_ms *= 2;
    a->retry_at = now + (uint64_t)a->retry_ms * 1000000;
}
// 看门狗（只管真出声的输出）：App 在前台时，已经启动的 AudioUnit 必须一直有回调。iOS 在别的音频会话接管（Siri、闹铃、来电）
// 或媒体服务重置时，会不通知 AudioUnit 就停掉回调：游戏的缓冲再也回不来，等缓冲的游戏音频（Wwise 的纯轮询输出、FMOD）就一直没声音。
// 200 ms 没有回调就换成静音接收端（游戏的节奏不变），AudioUnit 再按 1、2、4、8 s 重试；看门狗自己被拖醒（整个进程被挂起）时只重新计时
enum { WD_TICK_MS = 50, WD_STALL_MS = 200 };
static void *au_watchdog(void *p) {
    (void)p; uint64_t last = now_ns();
    for (;;) {
        usleep(WD_TICK_MS * 1000);
        pthread_mutex_lock(&g_aos_lock);
        uint64_t t = now_ns(), gap = t - last; last = t;   // 时间在锁内读：au_go 也在锁内记 wd_t，锁外读到的时间会早于后台返回刚重启的 AudioUnit 的戳，无符号相减回绕成"已停摆几十亿 ms"，把活着的 AudioUnit 拆掉重开
        for (int i = 0; i < AOMAX; i++) {
            aout *a = g_aos[i]; if (!a || !a->au) continue;
            uint64_t n = atomic_load_explicit(&a->cb_n, memory_order_relaxed);
            if (a->backend != BK_AU || g_suspended || gap > 4ull * WD_TICK_MS * 1000000 || n != a->wd_n) {
                if (a->backend == BK_AU && n != a->wd_n) a->retry_ms = 0;   // 回调还在来：下次再停摆时退避又从 1 s 开始
                a->wd_n = n; a->wd_t = t;
            } else if (t - a->wd_t > WD_STALL_MS * 1000000ull) {
                LOG("【红】audio #%d：播放中设备 %.0f ms 没回调：先由静音接收端顶上，稍后重试 AudioUnit", a->id, (t - a->wd_t) / 1e6);
                AudioOutputUnitStop(a->au); a->backend = BK_NONE; null_start(a);
                a->stalled = 1; a->retry_ms = a->retry_ms ? (a->retry_ms < 8000 ? a->retry_ms * 2 : 8000) : 1000; a->retry_at = t + (uint64_t)a->retry_ms * 1000000;
            }
            if (a->stalled && a->backend == BK_NULL && !g_suspended && t >= a->retry_at) {
                null_stop(a); au_go(a, t);
                if (a->backend == BK_AU) LOG("audio #%d：AudioUnit 重新起来了", a->id);
            }
        }
        pthread_mutex_unlock(&g_aos_lock);
    }
    return NULL;
}
static void au_watchdog_start(void) { pthread_t t; if (!pthread_create(&t, NULL, au_watchdog, NULL)) pthread_detach(t); }
static int real_audio(void) { const char *e = getenv("DM_AUDIO"); return e ? atoi(e) != 0 : TARGET_OS_IPHONE; }

// AudioOutParameter = { int32 sampleRate; uint16 channelCount; uint16 _r; ... }, 0 = system default (48 kHz / 2 channels)
HLE_FORCE("_ZN2nn5audio27InitializeAudioOutParameterEPNS0_17AudioOutParameterE", "音频输出参数默认值", void, h_au_InitParam, uint8_t *p) { memset(p, 0, 8); }
// AudioOutInfo = { char name[256] }
HLE_FORCE("_ZN2nn5audio13ListAudioOutsEPNS0_12AudioOutInfoEi", "音频输出设备列表；宿主 1 个", int, h_au_List, char *info, int n) {
    if (n < 1) return 0;
    memset(info, 0, 256); strcpy(info, "DeviceOut"); return 1;
}
static Result au_open_at(void *g, void *ev, const char *name, const uint8_t *param, void *lr) {
    static _Atomic int nopen;
    int ch = *(const uint16_t *)(param + 4); int rate = *(const int32_t *)param;
    if (!ch) ch = 2;
    if (rate && rate != RATE) die("AudioOut 采样率 %d 不支持（只做 48 kHz）", rate);
    aout *a = calloc(1, sizeof *a);
    pthread_mutex_init(&a->m, NULL); a->event = ev; a->ch = ch;
    atomic_init(&a->state, STATE_STOPPED);
    atomic_init(&a->q_head, 0); atomic_init(&a->q_tail, 0);
    atomic_init(&a->rel_head, 0); atomic_init(&a->rel_tail, 0);
    atomic_init(&a->null_run, 0); atomic_init(&a->notify_run, 1);
    atomic_init(&a->notify_pending, 0); atomic_init(&a->notify_armed, 0);
    a->notify_sem = dispatch_semaphore_create(0); a->id = atomic_fetch_add(&nopen, 1);
    pthread_create(&a->notify_th, NULL, event_notify, a);
    if (ev) hle_bind_system_event(ev, 1);   // Buffer release event: bind a kernel event on the guest thread (auto-clear), afterwards signaled by the audio thread
    objmap_put(g, a);
    pthread_mutex_lock(&g_aos_lock);
    int slot = -1; for (int i = 0; i < AOMAX; i++) if (!g_aos[i]) { slot = i; break; }
    if (slot < 0) die("AudioOut 同时打开超过 %d 个", AOMAX);
    g_aos[slot] = a;
    pthread_mutex_unlock(&g_aos_lock);
    char sb[64];
    LOG("audio #%d OpenAudioOut %s 声道 %d %s%s（调用点 %s）", a->id, name ? name : "(默认)", ch, real_audio() ? "真出声" : "静音接收端", ev ? "" : "，无释放事件（游戏轮询）", site(lr, sb, sizeof sb));
    if (real_audio()) { static pthread_once_t wd = PTHREAD_ONCE_INIT; au_open(a); pthread_once(&wd, au_watchdog_start); }   // 看门狗只起一次，盯着所有真出声的输出
    return RES_OK;
}
HLE_FORCE("_ZN2nn5audio12OpenAudioOutEPNS0_8AudioOutEPNS_2os11SystemEventEPKcRKNS0_17AudioOutParameterE", "连 audout 服务；宿主 AudioUnit / 静音接收端",
          Result, h_au_Open, void *g, void *ev, const char *name, const uint8_t *param) { return au_open_at(g, ev, name, param, __builtin_return_address(0)); }
// The two default-device overloads (Engage, SDK 12): as OpenAudioOut with the default device name; the overload without an event binds no release event (the game polls GetReleasedAudioOutBuffer)
HLE_FORCE("_ZN2nn5audio19OpenDefaultAudioOutEPNS0_8AudioOutEPNS_2os11SystemEventERKNS0_17AudioOutParameterE", "连 audout 服务（缺省设备）；同 OpenAudioOut",
          Result, h_au_OpenDef, void *g, void *ev, const uint8_t *param) { return au_open_at(g, ev, NULL, param, __builtin_return_address(0)); }
HLE_FORCE("_ZN2nn5audio19OpenDefaultAudioOutEPNS0_8AudioOutERKNS0_17AudioOutParameterE", "连 audout 服务（缺省设备、无事件）；同 OpenAudioOut",
          Result, h_au_OpenDef0, void *g, const uint8_t *param) { return au_open_at(g, NULL, NULL, param, __builtin_return_address(0)); }
static Result au_start_at(void *g, void *lr) {
    aout *a = ao(g);
    pthread_mutex_lock(&g_aos_lock);
    int was = atomic_exchange_explicit(&a->state, STATE_STARTED, memory_order_acq_rel);
    au_note(a, was == STATE_STARTED ? "Start（已在播）" : g_suspended ? "Start（App 在后台：先由静音接收端接手）" : "Start", lr);
    if (was != STATE_STARTED) {
        if (a->au && !g_suspended) { a->stalled = 0; au_go(a, now_ns()); }
        else null_start(a);
    }
    pthread_mutex_unlock(&g_aos_lock);
    return RES_OK;
}
HLE_FORCE("_ZN2nn5audio13StartAudioOutEPNS0_8AudioOutE", "开始播放", Result, h_au_Start, void *g) { return au_start_at(g, __builtin_return_address(0)); }
static void au_stop_at(void *g, void *lr) {   // lr NULL: the stop inside Close, not logged separately
    aout *a = ao(g);
    pthread_mutex_lock(&g_aos_lock);
    int was = atomic_exchange_explicit(&a->state, STATE_STOPPED, memory_order_acq_rel);
    if (lr) au_note(a, was == STATE_STARTED ? "Stop" : "Stop（本来就停着）", lr);
    backend_stop(a);   // AudioUnit，或者静音接收端（暂停时接管过来的那个）
    pthread_mutex_unlock(&g_aos_lock);
}
HLE_FORCE("_ZN2nn5audio12StopAudioOutEPNS0_8AudioOutE", "停止播放", void, h_au_Stop, void *g) { au_stop_at(g, __builtin_return_address(0)); }
HLE_FORCE("_ZN2nn5audio13CloseAudioOutEPNS0_8AudioOutE", "关闭音频输出", void, h_au_Close, void *g) {
    au_note(ao(g), "Close", __builtin_return_address(0));
    au_stop_at(g, NULL);
    aout *a = objmap_del(g);
    atomic_store_explicit(&a->notify_run, 0, memory_order_release);
    dispatch_semaphore_signal(a->notify_sem);
    pthread_join(a->notify_th, NULL);
    pthread_mutex_lock(&g_aos_lock);
    for (int i = 0; i < AOMAX; i++) if (g_aos[i] == a) g_aos[i] = NULL;
    pthread_mutex_unlock(&g_aos_lock);
    if (a->au) { AudioUnitUninitialize(a->au); AudioComponentInstanceDispose(a->au); }
    dispatch_release(a->notify_sem);
    pthread_mutex_destroy(&a->m); free(a);
}
HLE_FORCE("_ZN2nn5audio16GetAudioOutStateEPKNS0_8AudioOutE", "播放状态", int, h_au_State, const void *g) { return atomic_load_explicit(&ao(g)->state, memory_order_acquire); }
HLE_FORCE("_ZN2nn5audio21GetAudioOutSampleRateEPKNS0_8AudioOutE", "采样率", int, h_au_Rate, const void *g) { (void)ao(g); return RATE; }
HLE_FORCE("_ZN2nn5audio23GetAudioOutChannelCountEPKNS0_8AudioOutE", "声道数", int, h_au_Ch, const void *g) { return ao(g)->ch; }
HLE_FORCE("_ZN2nn5audio15GetAudioOutNameEPKNS0_8AudioOutE", "设备名；宿主只有缺省设备", const char *, h_au_Name, const void *g) { (void)g; return "DeviceOut"; }
HLE_FORCE("_ZN2nn5audio23GetAudioOutSampleFormatEPKNS0_8AudioOutE", "采样格式；只做 PCM16", int, h_au_Fmt, const void *g) { (void)ao(g); return FMT_PCM16; }

HLE_FORCE("_ZN2nn5audio21SetAudioOutBufferInfoEPNS0_14AudioOutBufferEPvmm", "缓冲描述；写进客体结构体前 24 字节", void, h_au_SetBuf, gbuf *b, void *p, uint64_t cap, uint64_t size) {
    b->ptr = p; b->cap = cap; b->size = size;
}
HLE_FORCE("_ZN2nn5audio28GetAudioOutBufferDataPointerEPKNS0_14AudioOutBufferE", "缓冲数据指针", void *, h_au_BufPtr, const gbuf *b) { return b->ptr; }
HLE_FORCE("_ZN2nn5audio25GetAudioOutBufferDataSizeEPKNS0_14AudioOutBufferE", "缓冲数据大小", uint64_t, h_au_BufSize, const gbuf *b) { return b->size; }
HLE_FORCE("_ZN2nn5audio20AppendAudioOutBufferEPNS0_8AudioOutEPNS0_14AudioOutBufferE", "追加待播缓冲", unsigned, h_au_Append, void *g, gbuf *b) {
    aout *a = ao(g);
    // Diagnostic switch DM_AUDIO_DUMP=<file>: every PCM block the game appends is written out as is (16-bit, 48 kHz, channel count per the OpenAudioOut log, usually 6). Used to judge "did a given sound play or not":
    // cross-correlate the original game's audio against it (this is how the tier-one drop sound was verified). The file is opened only when the variable is set; when unset this costs only one extra cached check
    { static FILE *df; static int on = -1; if (on < 0) { const char *e = getenv("DM_AUDIO_DUMP"); on = e != NULL; if (on && !(df = fopen(e, "wb"))) LOG("【红】DM_AUDIO_DUMP 打不开 %s", e); }
      if (df && b && b->ptr && b->size) fwrite(b->ptr, 1, (size_t)b->size, df); }
    pthread_mutex_lock(&a->m);
    uint32_t qt = atomic_load_explicit(&a->q_tail, memory_order_relaxed);
    uint32_t qh = atomic_load_explicit(&a->q_head, memory_order_acquire);
    int ok = qt - qh < QMAX;
    if (ok) { a->q[qt & (QMAX - 1)] = b; atomic_store_explicit(&a->q_tail, qt + 1, memory_order_release); }
    pthread_mutex_unlock(&a->m);
    return ok;      // bool: false when the queue is full (the real device has a limit too)
}
HLE_FORCE("_ZN2nn5audio25GetReleasedAudioOutBufferEPNS0_8AudioOutE", "取已播完的缓冲；没有则 NULL", gbuf *, h_au_Released, void *g) {
    aout *a = ao(g); gbuf *b = NULL;
    pthread_mutex_lock(&a->m);
    uint32_t rr = atomic_load_explicit(&a->rel_head, memory_order_relaxed);
    uint32_t rw = atomic_load_explicit(&a->rel_tail, memory_order_acquire);
    if (rr != rw) { b = a->rel[rr & (RELMAX - 1)]; atomic_store_explicit(&a->rel_head, rr + 1, memory_order_release); }
    pthread_mutex_unlock(&a->m);
    return b;
}

// Host-internal PCM output (used by the in-process audio renderer, see hle_audren.c): the same AudioOut machinery (real output / silent sink / background take-over), keyed by a host-allocated dummy object
void *dm_pcmout_open(int ch) {
    void *key = calloc(1, 64); uint8_t param[8] = {0}; *(uint16_t *)(param + 4) = (uint16_t)ch;
    if (au_open_at(key, NULL, "AudioRenderer", param, NULL)) die("宿主 PCM 输出打不开");
    return key;
}
void dm_pcmout_start(void *k) { au_start_at(k, NULL); }
int dm_pcmout_append(void *k, void *b) { return (int)h_au_Append(k, b); }
void *dm_pcmout_released(void *k) { return h_au_Released(k); }

// Front-end lifecycle / audio session interruption: suspend real audio output, with the silent sink taking over real-time consumption meanwhile (pacing unbroken); switch back to the AudioUnit on resume.
// The front end calls with 1 on going to background / interruption began, and with 0 on returning to foreground / interruption ended (shouldResume).
void dm_audio_suspend(int on) {
    pthread_mutex_lock(&g_aos_lock);
    if (g_suspended != on) {
        g_suspended = on; int failed = 0;
        for (int i = 0; i < AOMAX; i++) {
            aout *a = g_aos[i];
            if (!a || !a->au || atomic_load_explicit(&a->state, memory_order_acquire) != STATE_STARTED) continue;
            // While suspended the silent sink consumes in real time instead: otherwise buffers are no longer released, the FMOD output thread waits forever and the game's audio pacing stalls (same for an incoming call interruption)
            if (on) { if (a->backend == BK_AU) { AudioOutputUnitStop(a->au); null_start(a); } }
            else if (a->backend == BK_NULL) {
                // 会话可能还被别人占着（回到游戏但电话还没挂）：不是致命错误，静音接收端继续顶着，重试交给看门狗
                null_stop(a); a->stalled = 0; au_go(a, now_ns()); failed |= a->backend != BK_AU; }
        }
        LOG("audio %s", on ? "暂停（切后台 / 被打断）" : failed ? "已恢复，但 AudioUnit 还起不来（继续重试）" : "恢复（回前台 / 打断结束）");
    }
    pthread_mutex_unlock(&g_aos_lock);
}
// 媒体服务被重置了（iOS）：所有 AudioUnit 都作废。重新建一个，原来在播的接着播（前端已经先把音频会话重新设好了）
void dm_audio_reset(void) {
    pthread_mutex_lock(&g_aos_lock);
    for (int i = 0; i < AOMAX; i++) {
        aout *a = g_aos[i]; if (!a || !a->au) continue;
        int playing = a->backend == BK_AU;
        if (playing) { AudioOutputUnitStop(a->au); a->backend = BK_NONE; }
        AudioUnitUninitialize(a->au); AudioComponentInstanceDispose(a->au);
        OSStatus st = au_create(a);
        if (st) LOG("【红】audio #%d：媒体服务重置后建不出 AudioUnit（状态 %d）：这个输出保持静音", a->id, (int)st);
        if (playing) { if (a->au && !g_suspended) { a->stalled = 0; au_go(a, now_ns()); } else null_start(a); }
    }
    LOG("音频输出在媒体服务重置后重新建好");
    pthread_mutex_unlock(&g_aos_lock);
}
