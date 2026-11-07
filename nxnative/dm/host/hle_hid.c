// nn::hid (controller): on the real device shared memory is read through the hid service; the host takes the current state written by the front end (Mac keyboard / controller, iOS virtual gamepad / controller).
// Generic, game-independent version: one standard controller. DM_OPMODE=0 (handheld) → handheld controller (Npad id 0x20); otherwise → 1P full-key controller (id 0).
// Vibration is provided on iOS through Core Haptics (hle_haptics.c); six-axis, mouse and keyboard are not
// provided by the host (device count 0 / not connected); the controller applet reports "1 player, current controller selected".
#include "hle.h"
#include "hle_haptics.h"
#include "dm_input.h"
#include <stdatomic.h>
#include <unistd.h>
#include <strings.h>
#include <pthread.h>
#include <time.h>
#include <math.h>

enum { NPAD_NO1 = 0, NPAD_HANDHELD = 0x20 };
enum { STYLE_FULLKEY = 1u << 0, STYLE_HANDHELD = 1u << 1, STYLE_JOYDUAL = 1u << 2, STYLE_JOYLEFT = 1u << 3, STYLE_JOYRIGHT = 1u << 4 };
enum { ATTR_CONNECTED = 1u << 0, ATTR_WIRED = 1u << 1, ATTR_LEFT_CONN = 1u << 2, ATTR_LEFT_WIRED = 1u << 3, ATTR_RIGHT_CONN = 1u << 4, ATTR_RIGHT_WIRED = 1u << 5 };
// Stick direction pseudo-buttons (NpadButton 16..23): derived from the stick values, threshold about half travel
enum { STICK_THRESH = 0x4000 };

// nn::hid::NpadFullKeyState / Handheld / JoyDual / JoyLeft / JoyRight share one layout (0x28)
typedef struct { int64_t sampling; uint64_t buttons; int32_t lx, ly, rx, ry; uint32_t attributes, _pad; } NpadState;
_Static_assert(sizeof(NpadState) == 0x28, "NpadState");

// Front-end publishing: the keyboard / touch path (dm_input_set) and the physical controller path (dm_pad_set) are ORed; press edges are latched until the game's next read so short presses are not lost
static _Atomic uint64_t g_btn, g_axes, g_latch, g_pad_btn, g_pad_axes, g_pad_latch;
static _Atomic int64_t g_sampling;
static uint64_t pack(int32_t lx, int32_t ly, int32_t rx, int32_t ry) {
    return (uint64_t)(uint16_t)lx | (uint64_t)(uint16_t)ly << 16 | (uint64_t)(uint16_t)rx << 32 | (uint64_t)(uint16_t)ry << 48;
}
void dm_input_set(uint64_t buttons, int32_t lx, int32_t ly, int32_t rx, int32_t ry) {
    uint64_t old = atomic_exchange(&g_btn, buttons); atomic_fetch_or(&g_latch, buttons & ~old); atomic_store(&g_axes, pack(lx, ly, rx, ry));
}
void dm_pad_set(uint64_t buttons, int32_t lx, int32_t ly, int32_t rx, int32_t ry) {
    uint64_t old = atomic_exchange(&g_pad_btn, buttons); atomic_fetch_or(&g_pad_latch, buttons & ~old); atomic_store(&g_pad_axes, pack(lx, ly, rx, ry));
}
uint64_t dm_input_samples(void) { return (uint64_t)atomic_load(&g_sampling); }
// DM_KEYS scripted input (for tests): "seconds:key[+key][~hold seconds][*count/interval seconds],...", e.g. "8:A,12:A,15:Down~0.5,16:A+B,40:A*80/2" (A every 2 s from 40 s on, 80 times). Key names A B X Y LS RS L R ZL ZR Plus Minus Left Up Right Down, left stick LLeft LUp LRight LDown;
//   held 0.15 s by default, at most 1024 steps (counted after repeats are expanded). Goes through the controller path (dm_pad_set), ORed with keyboard / touch, neither overrides the other. Times are counted from host start.
//   An f before the time means a presented frame number ("f7600:A", and the repeat interval counts frames too): when the machine is busy seconds drift from the game's progress, frame numbers do not (used for the Zelda BotW opening route)
typedef struct { double at, hold; uint64_t btn; int32_t lx, ly, rx, ry; int frame; } key_step;
#define KEY_STEPS 1024
static key_step g_steps[KEY_STEPS]; static int g_nsteps;
static void *keys_main(void *arg) {
    (void)arg; struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < g_nsteps; i++) {
        for (;;) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); double el = (t.tv_sec - t0.tv_sec) + (t.tv_nsec - t0.tv_nsec) / 1e9;
                   if (g_steps[i].frame ? (double)dm_guest_frame() >= g_steps[i].at : el >= g_steps[i].at) break; usleep(5000); }
        LOG("DM_KEYS：%s%.2f 按 %#llx（按住 %.2f s）", g_steps[i].frame ? "帧 " : "", g_steps[i].at, (unsigned long long)g_steps[i].btn, g_steps[i].hold);
        dm_pad_set(g_steps[i].btn, g_steps[i].lx, g_steps[i].ly, g_steps[i].rx, g_steps[i].ry); usleep((useconds_t)(g_steps[i].hold * 1e6)); dm_pad_set(0, 0, 0, 0, 0);
    }
    return NULL;
}
void hid_keys_init(void) {
    const char *e = getenv("DM_KEYS"); if (!e || !*e) return;
    static const char *names[16] = {"A", "B", "X", "Y", "LS", "RS", "L", "R", "ZL", "ZR", "Plus", "Minus", "Left", "Up", "Right", "Down"};
    char buf[4096]; snprintf(buf, sizeof buf, "%s", e);
    for (char *tok = strtok(buf, ","); tok && g_nsteps < KEY_STEPS; tok = strtok(NULL, ",")) {
        char *c = strchr(tok, ':'); if (!c) { LOG("【红】DM_KEYS 步骤缺「秒:」：%s", tok); continue; }
        *c = 0; int fr = (tok[0] == 'f' || tok[0] == 'F'); key_step st = {atof(tok + fr), 0.15, 0}; st.frame = fr;
        int rep = 1; double every = 0;
        char *r = strchr(c + 1, '*'); if (r) { *r = 0; rep = atoi(r + 1); char *sl = strchr(r + 1, '/'); every = sl ? atof(sl + 1) : 1; if (rep < 1) rep = 1; }
        char *h = strchr(c + 1, '~'); if (h) { *h = 0; st.hold = atof(h + 1); }
        for (char *k = c + 1; k && *k;) { char *plus = strchr(k, '+'); if (plus) *plus = 0;
            int found = 0; for (int b = 0; b < 16; b++) if (!strcasecmp(k, names[b])) { st.btn |= 1ull << b; found = 1; }
            // stick fully pushed: LLeft LUp LRight LDown / RLeft RUp RRight RDown (left by default; the right stick exists for cursor-style input, e.g. aiming)
            static const char *const sticks[8] = {"LLeft", "LUp", "LRight", "LDown", "RLeft", "RUp", "RRight", "RDown"};
            static const int sx[8] = {-32767, 0, 32767, 0, -32767, 0, 32767, 0}, sy[8] = {0, 32767, 0, -32767, 0, 32767, 0, -32767};
            for (int d = 0; d < 8; d++) if (!strcasecmp(k, sticks[d])) {
                if (sx[d]) { if (d < 4) st.lx = sx[d]; else st.rx = sx[d]; }
                if (sy[d]) { if (d < 4) st.ly = sy[d]; else st.ry = sy[d]; }
                found = 1; }
            if (!found) LOG("【红】DM_KEYS 不认识的键名：%s", k);
            k = plus ? plus + 1 : NULL; }
        for (int i = 0; i < rep && g_nsteps < KEY_STEPS; i++) { g_steps[g_nsteps] = st; g_steps[g_nsteps++].at = st.at + i * every; }
    }
    pthread_t th; if (g_nsteps && !pthread_create(&th, NULL, keys_main, NULL)) { pthread_detach(th); LOG("DM_KEYS：%d 步", g_nsteps); }
}

static uint32_t connected_id(void) { return dm_opmode() == 0 ? NPAD_HANDHELD : NPAD_NO1; }
uint32_t hle_hid_connected_npad(void) { return connected_id(); }   // nfp's NFC device sits on this controller (hle_sdk16.c)
static uint32_t style_of(uint32_t id) { return id != connected_id() ? 0 : id == NPAD_HANDHELD ? STYLE_HANDHELD : STYLE_FULLKEY; }
static int32_t ax(uint64_t v, int i) { return (int16_t)(v >> (16 * i)); }
static int32_t pick(int32_t a, int32_t b) { return (a < 0 ? -a : a) >= (b < 0 ? -b : b) ? a : b; }   // whichever of the two paths is deflected more

static void cursor_trace(void);
static void dm_tap_pump(void);
static void fill(NpadState *s, uint32_t id, uint32_t style) {
    memset(s, 0, sizeof *s); cursor_trace();
    dm_tap_pump();   // live tap-to-select: reads the screen stack straight from the module
    s->sampling = atomic_fetch_add(&g_sampling, 1) + 1;
    if (!(style_of(id) & style)) return;   // no controller of this id / style: attributes = 0
    uint64_t b = atomic_load(&g_btn) | atomic_exchange(&g_latch, 0) | atomic_load(&g_pad_btn) | atomic_exchange(&g_pad_latch, 0);
    uint64_t k = atomic_load(&g_axes), p = atomic_load(&g_pad_axes);
    s->lx = pick(ax(k, 0), ax(p, 0)); s->ly = pick(ax(k, 1), ax(p, 1)); s->rx = pick(ax(k, 2), ax(p, 2)); s->ry = pick(ax(k, 3), ax(p, 3));
    b &= 0xffffull;
    if (s->lx < -STICK_THRESH) b |= 1u << 16;
    if (s->ly > STICK_THRESH) b |= 1u << 17;
    if (s->lx > STICK_THRESH) b |= 1u << 18;
    if (s->ly < -STICK_THRESH) b |= 1u << 19;
    if (s->rx < -STICK_THRESH) b |= 1u << 20;
    if (s->ry > STICK_THRESH) b |= 1u << 21;
    if (s->rx > STICK_THRESH) b |= 1u << 22;
    if (s->ry < -STICK_THRESH) b |= 1u << 23;
    s->buttons = b;
    s->attributes = ATTR_CONNECTED | (id == NPAD_HANDHELD ? ATTR_WIRED | ATTR_LEFT_CONN | ATTR_LEFT_WIRED | ATTR_RIGHT_CONN | ATTR_RIGHT_WIRED : ATTR_LEFT_CONN | ATTR_RIGHT_CONN);
}
// GetNpadStates(out, n, id): returns the number written, newest first. The host keeps no history: only 1 current state (callers handle the returned count)
static int states(NpadState *out, int n, uint32_t id, uint32_t style) { if (n <= 0) return 0; fill(out, id, style); return 1; }

// DM_CURTRACE diagnostic (read-only): the touch-menus work located the game's cursor subsystem in the Hades II build
// (docs/research/touch-menus-feasibility-20261024.md). The singleton cached at main+0xd39b80 is the input manager whose
// object holds the cursor state; the offsets below come from disassembling the GamepadCursorControl / MouseTether* readers.
// The probe dumps the candidate fields twice a second so a DM_KEYS stick hold can be matched against what moves; it
// never writes. Addresses belong to build 8ce1ac48… only (module_t.build_id gate, the repo's fixed-address rule).
#define CUR_BUILD "8ce1ac48e9fa74479051f3520e632df4"
// The diff-dump regions: the input manager cached at main+0xd39b80 and the singleton behind main+0x6bd6e8 (whose byte the
// hover path checks). Only changed 8-byte words are printed, so a stick phase shows up as a moving pair immediately.
static void cur_diff(const char *tag, uint64_t obj, int len, uint64_t *prev) {
    if (!obj) return;
    for (int i = 0; i < len; i += 8) {
        uint64_t v = *(uint64_t *)(uintptr_t)(obj + i);
        if (v != prev[i / 8]) {
            prev[i / 8] = v;
            float fx, fy; memcpy(&fx, &v, 4); memcpy(&fy, (char *)&v + 4, 4);
            LOG("  %s+%03x: %#018llx  f(%.2f, %.2f)  i(%d, %d)", tag, i, (unsigned long long)v, fx, fy,
                (int32_t)v, (int32_t)(v >> 32));
        }
    }
}
// Find the component vector on a screen object: scan begin/end pointer pairs and keep the one whose
// elements look like GUIComponents (element[0] = vtable inside the module, typeinfo name "N3sgg…").
static int cur_components(uint8_t *b, uint64_t top, uint64_t *out_beg, uint64_t *out_cnt) {
    if (!top) return 0;
    uintptr_t base = (uintptr_t)b; uint64_t size = *(uint64_t *)(base + 0); (void)size;
    module_t *m = 0; for (int i = 0; i < g_nmod; i++) if (!strcmp(g_mod[i].name, "main")) m = &g_mod[i];
    if (!m) return 0;
    int bestoff = -1; uint64_t bestb = 0, bestn = 0;
    for (int q = 0; q + 8 < 0x1000; q += 8) {
        uint64_t pb = *(uint64_t *)(uintptr_t)(top + q), pe = *(uint64_t *)(uintptr_t)(top + q + 8);
        if (pb < 0x7000000000ull || pb >= 0x7100000000ull || pe <= pb || pe >= 0x7100000000ull) continue;
        uint64_t cnt = (pe - pb) / 8; if (cnt < 1 || cnt > 200) continue;
        uint64_t good = 0, lim = cnt < 8 ? cnt : 8;
        for (uint64_t i = 0; i < lim; i++) {
            uint64_t el = *(uint64_t *)(uintptr_t)(pb + i * 8);
            if (el < 0x7000000000ull || el >= 0x7100000000ull || (el & 7)) continue;
            uint64_t vt = *(uint64_t *)(uintptr_t)el;
            if (vt <= base || vt - base >= m->size - 16) continue;
            uint64_t ti = *(uint64_t *)(uintptr_t)(vt - 8);
            if (ti <= base || ti - base >= m->size - 16) continue;
            uint64_t nm = *(uint64_t *)(uintptr_t)(ti + 8);
            if (nm <= base || nm - base >= m->size) continue;
            const char *q2 = (const char *)(uintptr_t)nm;
            if (q2[0] == 'N' && q2[1] == '3' && !strncmp(q2 + 2, "sgg", 3)) good++;
        }
        if (good > bestn || (good == bestn && cnt > bestn && bestoff < 0)) { if (good >= bestn) { bestn = good; bestb = pb; bestoff = q; } }
        if (bestn >= 3 && (uint64_t)bestoff == (uint64_t)q) { /* keep scanning a bit */ }
    }
    if (bestoff < 0) return 0;
    *out_beg = bestb; *out_cnt = bestn ? *(uint64_t *)(uintptr_t)(bestb - (bestb ? 0 : 0)) : 0;
    // recompute count from the pair at bestoff
    uint64_t pe = *(uint64_t *)(uintptr_t)(top + bestoff + 8);
    *out_cnt = (pe - bestb) / 8;
    return bestoff + 1;   // offset + 1 so 0 stays "not found"
}

// Tap-to-select shared state. A "tap" arrives either from the env script (DM_CURTAP, prototype
// journeys) or live from a front end (dm_tap_set, front.h — the Mac mouse, later the iPhone touch).
// dm_tap_tick runs the closed loop: walk the top screen's buttons, pick the one nearest the tap,
// inject Up/Down edges through the controller path until the game's own selection lands there.
static float g_tap_x, g_tap_y;          // design space (1920x1080)
static uint64_t g_tap_arm;              // inject nothing before this guest frame
static int g_tap_live, g_nav_dir, g_nav_wait, g_nav_done, g_nav_tgt = -1;
static uint64_t g_nav_next; static uint64_t g_tap_top;
static unsigned g_tap_seq, g_nav_seq;   // one tap = one navigation: the pump steers only while it has
                                        // NOT yet consumed the newest tap's seq (no self re-arm — re-arming
                                        // from the stale tap yanked the selection around the settings screen)
void dm_tap_set(float nx, float ny) {   // normalized, origin top-left (the dm_touch convention)
    // PROTOTYPE, OFF by default: the tap loop is a workaround (the real hover subsystem is out of reach on
    // this build, see the research doc), and an unexplained model-disappearance appeared with it live once
    // (control run without taps was clean; the lifecycle fix landed but was never re-verified in a dungeon).
    // Opt in with DM_TAP=1.
    { static int on = -1; if (on < 0) { const char *e = getenv("DM_TAP"); on = e && *e && strcmp(e, "0"); }
      if (!on) return; }
    g_tap_x = nx * 1920.0f; g_tap_y = ny * 1080.0f;
    g_tap_live = 1; g_nav_tgt = -1; g_nav_done = 0; g_nav_wait = 0; g_tap_top = 0;
    g_tap_seq++;
    static int once; if (!once) { once = 1; LOG("触屏选单：首次点按 (%.0f, %.0f) 设计空间", g_tap_x, g_tap_y); }
}
static void dm_tap_tick(uint64_t f, uint8_t *b, uint64_t beg, uint64_t n, module_t *main_) {
    static uint64_t lasttick;
    if (!g_tap_live || f < g_tap_arm || n == 0) return;
    if (lasttick && f - lasttick < 8) return;   // ~7.5 nav steps/s
    lasttick = f;
    uint64_t top = *(uint64_t *)(uintptr_t)(beg + (n - 1) * 8);
    uint64_t cb = 0, cnt = 0;
    if (!top || !cur_components(b, top, &cb, &cnt)) { g_nav_tgt = -1; g_nav_seq = g_tap_seq; return; }   // top screen changed/gone: disarm
    if (top != g_tap_top) { g_tap_top = top; g_nav_tgt = -1; g_nav_seq = g_tap_seq; }   // a different screen came up: disarm
    int nb = 0, tgt = -1, cur = -1;
    uint64_t sel = *(uint64_t *)(uintptr_t)(top + 0xc0);
    float bd = 1e30f;
    uintptr_t base = (uintptr_t)main_->base;
    for (uint64_t i = 0; i < cnt && nb < 64; i++) {
        uint64_t c = *(uint64_t *)(uintptr_t)(cb + i * 8);
        if (c < 0x7000000000ull || c >= 0x7100000000ull || (c & 7)) continue;
        uint64_t vt = *(uint64_t *)(uintptr_t)c; char cls[120] = "?";
        if (vt > base && vt - base < main_->size - 16) {
            uint64_t ti = *(uint64_t *)(uintptr_t)(vt - 8);
            if (ti > base && ti - base < main_->size - 16) {
                uint64_t nm = *(uint64_t *)(uintptr_t)(ti + 8);
                if (nm > base && nm - base < main_->size) { const char *q2 = (const char *)(uintptr_t)nm; size_t k3 = 0;
                    while (k3 + 1 < sizeof cls && q2[k3] >= 32 && q2[k3] < 127) { cls[k3] = q2[k3]; k3++; } cls[k3] = 0; }
            }
        }
        if (!strstr(cls, "Button")) continue;
        if (c == sel) cur = nb;
        float cx = *(float *)(uintptr_t)(c + 0x10), cy = *(float *)(uintptr_t)(c + 0x14);
        float dx = cx - g_tap_x, dy = cy - g_tap_y, d = dx * dx + dy * dy;
        if (d < bd) { bd = d; tgt = nb; }
        nb++;
    }
    if (!nb || tgt < 0) { g_nav_tgt = -1; g_nav_seq = g_tap_seq; return; }
    if (cur < 0) { g_nav_tgt = -1; g_nav_seq = g_tap_seq; return; }   // selection left the button list (menu closed): disarm, never steer in gameplay
    if (cur == tgt) { g_nav_tgt = -1; g_nav_seq = g_tap_seq; if (!g_nav_done) { g_nav_done = 1; LOG("触屏选单：已选中第 %d 项（A 确认）", tgt); } return; }
    if (g_nav_tgt < 0) {                       // arm once per tap; never re-arm from a consumed tap
        if (g_nav_seq == g_tap_seq) return;
        g_nav_tgt = tgt; g_nav_seq = g_tap_seq;
    }
    tgt = g_nav_tgt;
    if (f < g_nav_next) return;
    unsigned dir = tgt > cur ? (1u << 15) : (1u << 13);   // Down / Up
    if (g_nav_wait) { dm_pad_set(0, 0, 0, 0, 0); g_nav_wait = 0; g_nav_next = f + 3; }
    else { dm_pad_set(dir, 0, 0, 0, 0); g_nav_wait = 1; g_nav_next = f + 9; g_nav_dir = dir;
        { static uint64_t logged_top; if (logged_top != top) { logged_top = top;
            uintptr_t base2 = (uintptr_t)b; uint64_t vt = *(uint64_t *)(uintptr_t)top; char cls2[120] = "?";
            if (vt > base2 && vt - base2 < main_->size - 16) { uint64_t ti = *(uint64_t *)(uintptr_t)(vt - 8);
                if (ti > base2 && ti - base2 < main_->size - 16) { uint64_t nm = *(uint64_t *)(uintptr_t)(ti + 8);
                    if (nm > base2 && nm - base2 < main_->size) { const char *q3 = (const char *)(uintptr_t)nm; size_t k4 = 0;
                        while (k4 + 1 < sizeof cls2 && q3[k4] >= 32 && q3[k4] < 127) { cls2[k4] = q3[k4]; k4++; } cls2[k4] = 0; } } }
            LOG("触屏选单：注入 %s 于 %s（帧 %llu）", (dir & (1u << 15)) ? "Down" : "Up", cls2, (unsigned long long)f); } } }
}

// Called from fill() (every controller read): runs the tap loop at ~7.5 steps/s when a tap is live.
static void dm_tap_pump(void) {
    static uint64_t lasttick; static module_t *main_;
    if (!g_tap_live) { for (int i = 0; i < g_nmod; i++) if (!strcmp(g_mod[i].name, "main")) main_ = &g_mod[i]; return; }
    if (!main_) { for (int i = 0; i < g_nmod; i++) if (!strcmp(g_mod[i].name, "main")) main_ = &g_mod[i]; if (!main_) return; }
    if (!main_->base) return;
    uint64_t f = dm_guest_frame();
    if (lasttick && f - lasttick < 8) return;
    lasttick = f;
    uint64_t mgr = *(uint64_t *)(main_->base + 0xd39b80);
    uint64_t cont = mgr ? *(uint64_t *)(uintptr_t)(mgr + 0x698) : 0;
    uint64_t beg = cont ? *(uint64_t *)(uintptr_t)(cont + 0x48) : 0;
    uint64_t end = cont ? *(uint64_t *)(uintptr_t)(cont + 0x50) : 0;
    uint64_t n = (end > beg && end - beg <= 256) ? (end - beg) / 8 : 0;
    dm_tap_tick(f, main_->base, beg, n, main_);
}

static void cursor_trace(void) {
    static int on = -1; static module_t *main_; static int64_t last = -1;
    static uint64_t p_mgr[0x600 / 8], p_sm[0x400 / 8], p_ms[0x400 / 8];
    if (on < 0) { const char *e = getenv("DM_CURTRACE"); on = e && *e && strcmp(e, "0");
        for (int i = 0; i < g_nmod; i++) if (!strcmp(g_mod[i].name, "main")) main_ = &g_mod[i];
        if (on && (!main_ || strcmp(main_->build_id, CUR_BUILD))) { LOG("DM_CURTRACE：main 构建 %s 没有对应地址，不读", main_ && main_->build_id[0] ? main_->build_id : "（无）"); on = 0; } }
    if (!on || !main_) return;
    uint64_t f = dm_guest_frame(); if (f < 30) return;
    if (last >= 0 && (int64_t)f - last < 30) return;   // ~ twice a second
    last = (int64_t)f;
    uint8_t *b = main_->base;
    uint64_t sc = *(uint64_t *)(b + 0x6be3e0);
    LOG("帧 %llu 差分：游标控制 %d 用鼠标 %d 比例对 (%.3f, %.3f)", (unsigned long long)f,
        *(uint8_t *)(b + 0xd3c3dc), *(uint8_t *)(b + 0xd3c3ec),
        sc ? *(float *)(uintptr_t)sc : -1.0f, sc ? *(float *)(uintptr_t)(sc + 4) : -1.0f);
    cur_diff("mgr", *(uint64_t *)(b + 0xd39b80), 0x600, p_mgr);
    cur_diff("sm", *(uint64_t *)(b + 0x6bd6e8), 0x400, p_sm);
    // Follow the manager's own pointers: the input devices hang off it, and the cursor state sits in the device
    // (the tether smoother's object: sample ring at +0x378, pairs at +0x4c / +0x84 / +0x130).
    static uint64_t p_dev[8][0x400 / 8]; static uint64_t last_ptr[8]; static int ndev;
    uint64_t mgr = *(uint64_t *)(b + 0xd39b80);
    if (mgr) {
        int k = 0;
        for (int i = 0; i < 0x600 && k < 8; i += 8) {
            uint64_t p = *(uint64_t *)(uintptr_t)(mgr + i);
            if (p < 0x7000000000ull || p >= 0x7100000000ull || (p & 7)) continue;   // guest AS pointers only
            int dup = 0; for (int j = 0; j < k; j++) if (last_ptr[j] == p) dup = 1;
            if (!dup && p != mgr) { last_ptr[k] = p; k++; }
        }
        if (k > ndev) ndev = k;
        for (int j = 0; j < ndev; j++) { char tag[8]; snprintf(tag, sizeof tag, "d%d", j); cur_diff(tag, last_ptr[j], 0x400, p_dev[j]); }
        // Identify each followed pointer by its vtable's RTTI: that names the input subsystem objects.
        { static int64_t lastid = -1; static uint64_t lastn; static uint64_t seen[8];
          if (lastid < 0 || lastn != (uint64_t)ndev || (int64_t)f - lastid >= 600) {
              lastid = (int64_t)f; lastn = (uint64_t)ndev;
              uintptr_t base = (uintptr_t)main_->base;
              for (int j = 0; j < ndev; j++) {
                  uint64_t el = last_ptr[j]; if (el == seen[j]) continue; seen[j] = el;
                  uint64_t vt = *(uint64_t *)(uintptr_t)el; char cls[160] = "?";
                  if (vt > base && vt - base < main_->size - 16) {
                      uint64_t ti = *(uint64_t *)(uintptr_t)(vt - 8);
                      if (ti > base && ti - base < main_->size - 16) {
                          uint64_t nm = *(uint64_t *)(uintptr_t)(ti + 8);
                          if (nm > base && nm - base < main_->size) {
                              const char *q = (const char *)(uintptr_t)nm; size_t k = 0;
                              while (k + 1 < sizeof cls && q[k] >= 32 && q[k] < 127) { cls[k] = q[k]; k++; } cls[k] = 0;
                          }
                      }
                  }
                  LOG("mgr 指针 d%d = %#llx 类 %s（+0x250 对 %.1f,%.1f）", j, (unsigned long long)el, cls,
                      *(float *)(uintptr_t)(el + 0x250), *(float *)(uintptr_t)(el + 0x254));
              }
          } }
    }
    // Follow the mouse singleton chain the tether getter uses: [main+0x6bd0c0] -> holder -> [holder] = instance.
    uint64_t ms = 0, holder = *(uint64_t *)(b + 0x6bd0c0);
    if (holder) ms = *(uint64_t *)(uintptr_t)holder;
    cur_diff("ms", ms, 0x400, p_ms);
    // DM_CURMOUSE=<x>,<y>: write the pointer position the hover dispatcher uses
    // (0x3c7df0 -> 0x3c63a0 takes int [[0x6be400]+0xbc] / [[0x6be400]+0xc4]). Probe-grade:
    // it makes the game think a mouse sits at that canvas point, so the menu can be hovered.
    {
        static int set; static float mx, my; static float pts[8][2]; static int npts;
        if (!set) { set = 1; const char *e = getenv("DM_CURMOUSE");
            if (e && *e) {   // "x,y" or "x1,y1;x2,y2;…": with several points the probe cycles every 120 frames
                char buf[256]; snprintf(buf, sizeof buf, "%s", e);
                for (char *tok = strtok(buf, ";"); tok && npts < 8; tok = strtok(NULL, ";")) {
                    const char *c = strchr(tok, ','); if (!c) continue;
                    pts[npts][0] = atof(tok); pts[npts][1] = atof(c + 1); npts++;
                }
                LOG("DM_CURMOUSE：%d 个指针位置", npts);
            } }
        if (npts) { int k = npts > 1 ? (int)((f / 120) % (uint64_t)npts) : 0; mx = pts[k][0]; my = pts[k][1]; }
        if (mx > 0 || my > 0) {
            // The hover path reads the mouse point only when UseMouse is on (0x3c63f0: cbz skips the whole
            // mouse branch); GamepadCursorControl makes the screen accept pointer input.
            dm_guest_write_prepare(b + 0xd3c3ec, 1); *(uint8_t *)(b + 0xd3c3ec) = 1;
            dm_guest_write_prepare(b + 0xd3c3dc, 1); *(uint8_t *)(b + 0xd3c3dc) = 1;
            uint64_t mo = *(uint64_t *)(b + 0x6be400);
            if (mo) { dm_guest_write_prepare((void *)(uintptr_t)(mo + 0xbc), 4); *(float *)(uintptr_t)(mo + 0xbc) = mx;
                      dm_guest_write_prepare((void *)(uintptr_t)(mo + 0xc4), 4); *(float *)(uintptr_t)(mo + 0xc4) = my; }
        }
    }
    // The hover dispatcher (0x3c7df0 tails into 0x3c63a0) takes its point from [[0x6be400]+0xbc] / +0xc4, as ints.
    {
        static int64_t lastm = -1;
        if (lastm < 0 || (int64_t)f - lastm >= 60) {
            lastm = (int64_t)f;
            uint64_t mo = *(uint64_t *)(b + 0x6be400);
            LOG("帧 %llu 鼠标源：[0x6be400]=%#llx +0xbc=%.2f +0xc4=%.2f", (unsigned long long)f,
                (unsigned long long)mo,
                mo ? *(float *)(uintptr_t)(mo + 0xbc) : -1.0f,
                mo ? *(float *)(uintptr_t)(mo + 0xc4) : -1.0f);
            if (mo && getenv("DM_CURDUMP")) {   // the fields the hover dispatcher touches on its own state object
                LOG("  该对象：+0x14=(%.2f,%.2f) +0x1c=(%.2f,%.2f) +0x24=%d +0x28=%d +0x260=%d +0x261=%d +0x262=%d +0x274=%.2f +0x28c=%.2f",
                    *(float *)(uintptr_t)(mo + 0x14), *(float *)(uintptr_t)(mo + 0x18),
                    *(float *)(uintptr_t)(mo + 0x1c), *(float *)(uintptr_t)(mo + 0x20),
                    *(int32_t *)(uintptr_t)(mo + 0x24), *(uint8_t *)(uintptr_t)(mo + 0x28),
                    *(uint8_t *)(uintptr_t)(mo + 0x260), *(uint8_t *)(uintptr_t)(mo + 0x261),
                    *(uint8_t *)(uintptr_t)(mo + 0x262),
                    *(float *)(uintptr_t)(mo + 0x274), *(float *)(uintptr_t)(mo + 0x28c));
            }
        }
    }
    // DM_CURARR (read-only): dump the input manager's device list, the array the hover gate walks
    // ([mgr+0x698] -> container -> vector at +0x48). For each element: address, vtable class name,
    // and the three bytes the gate tests (+0x262 / +0x263 / int at +0x3c). This is what decides
    // "mouse present" on this build; it runs once every 120 frames so the log stays readable.
    {
        static int on = -1; static int64_t lastarr = -1;
        if (on < 0) {   // any of the screen/component probes turns this block on
            const char *e = getenv("DM_CURARR"), *e2 = getenv("DM_CURCOMP"), *e3 = getenv("DM_CURPICK"), *e4 = getenv("DM_CURTAP");
            on = (e && *e && strcmp(e, "0")) || (e2 && *e2 && strcmp(e2, "0")) ||
                 (e3 && *e3 && strcmp(e3, "0")) || (e4 && *e4 && strcmp(e4, "0"));
        }
        if (on && (lastarr < 0 || (int64_t)f - lastarr >= 120)) {
            lastarr = (int64_t)f;
            uint64_t mgr2 = *(uint64_t *)(b + 0xd39b80), cont = mgr2 ? *(uint64_t *)(uintptr_t)(mgr2 + 0x698) : 0;
            uint64_t beg = cont ? *(uint64_t *)(uintptr_t)(cont + 0x48) : 0;
            uint64_t end = cont ? *(uint64_t *)(uintptr_t)(cont + 0x50) : 0;
            uint64_t n = (end > beg) ? (end - beg) / 8 : 0;
            if (n > 16) n = 16;
            // Also diff every screen object in that list: a live cursor has to move some field here.
            { static uint64_t ps[16][0x800 / 8];
              for (uint64_t i = 0; i < n; i++) {
                  uint64_t el = *(uint64_t *)(uintptr_t)(beg + i * 8);
                  char tag[16]; snprintf(tag, sizeof tag, "scr%llu", (unsigned long long)i);
                  cur_diff(tag, el, 0x800, ps[i]);
              } }
            // DM_CURCOMP: dump the top screen's component list (screen+0x48 begin/end) with each element's
            // class name and the position-ish fields, to see whether a generic hit-test is possible.
            if (getenv("DM_CURCOMP") && n) {
                uint64_t top = *(uint64_t *)(uintptr_t)(beg + (n - 1) * 8);
                // The component vector is not at a fixed offset: find begin/end pairs of guest pointers in the
                // first 0x1000 bytes where end > begin and the element count is small.
                uint64_t cab = 0, cae = 0; int coff = -1;
                if (top) for (int q = 0; q + 8 < 0x1000; q += 8) {
                    uint64_t pb = *(uint64_t *)(uintptr_t)(top + q), pe = *(uint64_t *)(uintptr_t)(top + q + 8);
                    if (pb < 0x7000000000ull || pb >= 0x7100000000ull || pe <= pb || pe >= 0x7100000000ull) continue;
                    uint64_t cnt = (pe - pb) / 8;
                    if (cnt >= 1 && cnt <= 200) { cab = pb; cae = pe; coff = q; break; }
                }
                uint64_t cn = (cae > cab) ? (cae - cab) / 8 : 0; if (cn > 24) cn = 24;
                uintptr_t base = (uintptr_t)main_->base;
                LOG("  屏幕 %#llx 组件向量偏移 %#x 共 %llu 个（%#llx..%#llx）", (unsigned long long)top, coff,
                    (unsigned long long)cn, (unsigned long long)cab, (unsigned long long)cae);
                // Which screen field points at one of those components? That is the "selected" slot.
                if (top) for (int q = 0; q + 8 <= 0x400; q += 8) {
                    uint64_t v2 = *(uint64_t *)(uintptr_t)(top + q);
                    for (uint64_t ci = 0; ci < cn; ci++)
                        if (v2 && v2 == *(uint64_t *)(uintptr_t)(cab + ci * 8)) {
                            LOG("    屏幕+%#x -> 组件 c%llu（疑似当前选中）", q, (unsigned long long)ci);
                            break;
                        }
                }
                for (uint64_t ci = 0; ci < cn; ci++) {
                    uint64_t c = *(uint64_t *)(uintptr_t)(cab + ci * 8);
                    if (c < 0x7000000000ull || c >= 0x7100000000ull || (c & 7)) continue;
                    uint64_t vt = *(uint64_t *)(uintptr_t)c; char cls[120] = "?";
                    if (vt > base && vt - base < main_->size - 16) {
                        uint64_t ti = *(uint64_t *)(uintptr_t)(vt - 8);
                        if (ti > base && ti - base < main_->size - 16) {
                            uint64_t nm = *(uint64_t *)(uintptr_t)(ti + 8);
                            if (nm > base && nm - base < main_->size) { const char *q = (const char *)(uintptr_t)nm; size_t k2 = 0;
                                while (k2 + 1 < sizeof cls && q[k2] >= 32 && q[k2] < 127) { cls[k2] = q[k2]; k2++; } cls[k2] = 0; }
                        }
                    }
                    // print the offsets whose float lands in a plausible design-space coordinate range,
                    // so the X/Y fields can be matched against the .sjson values
                    char hits[400]; int ho = 0; hits[0] = 0;
                    for (int o = 0x10; o < 0x300; o += 4) {
                        float v = *(float *)(uintptr_t)(c + o);
                        if (v > 1.0f && v < 2000.0f)
                            ho += snprintf(hits + ho, sizeof hits - ho, " %#x=%.0f", o, v);
                        if (ho > 330) break;
                    }
                    LOG("   c%llu %#llx %s %s", (unsigned long long)ci, (unsigned long long)c, cls, hits);
                }
            }
            
// DM_CURPICK=<x>,<y> (design space, 1920x1080): pick the button nearest that point on the top
            // screen and make it the selected component (screen+0xc0 / +0x1b0). Prototype for tap-to-select:
            // every screen (including Lua-built in-game ones) exposes its live component list, so this needs
            // no per-menu table. Confirm by pressing A afterwards.
            {
                static int set; static float px, py;
                if (!set) { set = 1; const char *e = getenv("DM_CURPICK");
                    if (e && *e) { const char *c = strchr(e, ','); if (c) { px = atof(e); py = atof(c + 1);
                        LOG("DM_CURPICK：目标点 (%.0f, %.0f) 设计空间", px, py); } } }
                if (px > 0 && n) {
                    uint64_t top = *(uint64_t *)(uintptr_t)(beg + (n - 1) * 8);
                    if (top) {
                        uint64_t cb = 0, cnt2 = 0;
                        cur_components(b, top, &cb, &cnt2);
                        uint64_t best = 0; float bd = 1e30f;
                        uintptr_t base = (uintptr_t)main_->base;
                        for (uint64_t ci = 0; ci < cnt2 && ci < 200; ci++) {
                            uint64_t c = *(uint64_t *)(uintptr_t)(cb + ci * 8);
                            if (c < 0x7000000000ull || c >= 0x7100000000ull || (c & 7)) continue;
                            uint64_t vt = *(uint64_t *)(uintptr_t)c; char cls[120] = "?";
                            if (vt > base && vt - base < main_->size - 16) {
                                uint64_t ti = *(uint64_t *)(uintptr_t)(vt - 8);
                                if (ti > base && ti - base < main_->size - 16) {
                                    uint64_t nm = *(uint64_t *)(uintptr_t)(ti + 8);
                                    if (nm > base && nm - base < main_->size) { const char *q2 = (const char *)(uintptr_t)nm; size_t k3 = 0;
                                        while (k3 + 1 < sizeof cls && q2[k3] >= 32 && q2[k3] < 127) { cls[k3] = q2[k3]; k3++; } cls[k3] = 0; }
                                }
                            }
                            if (!strstr(cls, "Button")) continue;
                            float cx = *(float *)(uintptr_t)(c + 0x10), cy = *(float *)(uintptr_t)(c + 0x14);
                            float dx = cx - px, dy = cy - py, d = dx * dx + dy * dy;
                            if (d < bd) { bd = d; best = c; }
                        }
                        if (best) {
                            dm_guest_write_prepare((void *)(uintptr_t)(top + 0xc0), 8); *(uint64_t *)(uintptr_t)(top + 0xc0) = best;
                            dm_guest_write_prepare((void *)(uintptr_t)(top + 0x1b0), 8); *(uint64_t *)(uintptr_t)(top + 0x1b0) = best;
                            static uint64_t lastbest;
                            if (best != lastbest) { lastbest = best;
                                LOG("DM_CURPICK：选中 %#llx 中心 (%.0f, %.0f) 距目标 %.0f（组件 %llu 个）",
                                    (unsigned long long)best, *(float *)(uintptr_t)(best + 0x10),
                                    *(float *)(uintptr_t)(best + 0x14), sqrtf(bd), (unsigned long long)cnt2); }
                        } else { static int once; if (!once) { once = 1; LOG("DM_CURPICK：该屏没有 Button 组件（%llu 个组件）", (unsigned long long)cnt2); } }
                    }
                }
            }
            // DM_CURTAP=<x>,<y>[,<arm-frame>]: the scripted-journey form of the tap (same state machine the
            // front ends drive through dm_tap_set). The arm frame exists because a journey still walking menus
            // plus a stray Down plus a scripted A once opened the delete-save dialog (see the research doc).
            {
                static int set;
                if (!set) { set = 1; const char *e = getenv("DM_CURTAP");
                    if (e && *e) { const char *c = strchr(e, ',');
                        if (c) { g_tap_x = atof(e); g_tap_y = atof(c + 1);
                            const char *c2 = strchr(c + 1, ',');
                            g_tap_arm = c2 ? strtoull(c2 + 1, 0, 0) : 0xffffffffffffffffull;
                            g_tap_live = 1; g_nav_tgt = -1; g_nav_done = 0;
                            LOG("DM_CURTAP：目标 (%.0f, %.0f)，第 %llu 帧起允许注入", g_tap_x, g_tap_y, (unsigned long long)g_tap_arm); } } }
            }

            LOG("帧 %llu 设备表：mgr %#llx 容器 %#llx 条目 %llu（%#llx..%#llx）",
                (unsigned long long)f, (unsigned long long)mgr2, (unsigned long long)cont,
                (unsigned long long)n, (unsigned long long)beg, (unsigned long long)end);
            for (uint64_t i = 0; i < n; i++) {
                uint64_t el = *(uint64_t *)(uintptr_t)(beg + i * 8);
                if (!el) { LOG("  [%llu] 空", (unsigned long long)i); continue; }
                uint64_t vt = *(uint64_t *)(uintptr_t)el;
                // Runtime pointers are (module base + vaddr): convert back to read the class name.
                uintptr_t base = (uintptr_t)main_->base;
                uint64_t vt_off = vt - base, ti = 0, nm = 0;
                char cls[160] = "?";
                if (vt_off < main_->size) {
                    ti = *(uint64_t *)(uintptr_t)(vt - 8);              // typeinfo pointer
                    if (ti > base && ti - base < main_->size - 16) {
                        nm = *(uint64_t *)(uintptr_t)(ti + 8);           // name pointer
                        if (nm > base && nm - base < main_->size) {
                            const char *q = (const char *)(uintptr_t)nm;
                            size_t k = 0; while (k + 1 < sizeof cls && q[k] >= 32 && q[k] < 127) { cls[k] = q[k]; k++; } cls[k] = 0;
                        }
                    }
                }
                LOG("  [%llu] %#llx 类 %s  +0x262=%d +0x263=%d +0x3c=%d (+0x260=%d +0x261=%d)",
                    (unsigned long long)i, (unsigned long long)el, cls,
                    *(uint8_t *)(uintptr_t)(el + 0x262), *(uint8_t *)(uintptr_t)(el + 0x263),
                    *(int32_t *)(uintptr_t)(el + 0x3c), *(uint8_t *)(uintptr_t)(el + 0x260),
                    *(uint8_t *)(uintptr_t)(el + 0x261));
            }
        }
    }
}

HLE_FORCE("_ZN2nn3hid14InitializeNpadEv", "连 hid 服务；宿主从前端状态取输入", void, h_hid_InitNpad, void) {}
HLE_FORCE("_ZN2nn3hid22SetSupportedNpadIdTypeEPKjm", "支持的手柄号；宿主固定", void, h_hid_SetIds, const uint32_t *ids, uint64_t n) { (void)ids; (void)n; }
static uint32_t g_styles_supported = 0xffffffffu;
HLE_FORCE("_ZN2nn3hid24SetSupportedNpadStyleSetENS_4util10BitFlagSetILi32ENS0_12NpadStyleTagEEE", "支持的手柄样式；宿主记下，按它选报的样式", void, h_hid_SetStyles, uint32_t styles) {
    if (styles != g_styles_supported) LOG("hid 支持的手柄样式 %#x", styles); g_styles_supported = styles; }
HLE_FORCE("_ZN2nn3hid15GetNpadStyleSetERKj", "手柄样式查询；跟 DM_OPMODE 走", uint32_t, h_hid_GetStyle, const uint32_t *id) { return style_of(*id); }
HLE_FORCE("_ZN2nn3hid13GetNpadStatesEPNS0_16NpadFullKeyStateEiRKj", "全键手柄状态", int, h_hid_StatesFullKey, NpadState *s, int n, const uint32_t *id) { return states(s, n, *id, STYLE_FULLKEY); }
HLE_FORCE("_ZN2nn3hid13GetNpadStatesEPNS0_17NpadHandheldStateEiRKj", "掌机手柄状态", int, h_hid_StatesHandheld, NpadState *s, int n, const uint32_t *id) { return states(s, n, *id, STYLE_HANDHELD); }
HLE_FORCE("_ZN2nn3hid13GetNpadStatesEPNS0_16NpadJoyDualStateEiRKj", "双 Joy-Con 状态（宿主不接）", int, h_hid_StatesJoyDual, NpadState *s, int n, const uint32_t *id) { return states(s, n, *id, STYLE_JOYDUAL); }
HLE_FORCE("_ZN2nn3hid13GetNpadStatesEPNS0_16NpadJoyLeftStateEiRKj", "左 Joy-Con 状态（宿主不接）", int, h_hid_StatesJoyLeft, NpadState *s, int n, const uint32_t *id) { return states(s, n, *id, STYLE_JOYLEFT); }
HLE_FORCE("_ZN2nn3hid13GetNpadStatesEPNS0_17NpadJoyRightStateEiRKj", "右 Joy-Con 状态（宿主不接）", int, h_hid_StatesJoyRight, NpadState *s, int n, const uint32_t *id) { return states(s, n, *id, STYLE_JOYRIGHT); }
HLE_FORCE("_ZN2nn3hid14DisconnectNpadERKj", "断开手柄；宿主忽略", void, h_hid_Disconnect, const uint32_t *id) { (void)id; }
static uint32_t g_hold_type;
HLE_FORCE("_ZN2nn3hid18SetNpadJoyHoldTypeENS0_15NpadJoyHoldTypeE", "Joy-Con 握持方向；宿主只记下", void, h_hid_HoldType, int t) { g_hold_type = (uint32_t)t; }
HLE_FORCE("_ZN2nn3hid20GetNpadJoyAssignmentERKj", "Joy-Con 分配：双持", int, h_hid_GetJoyAssign, const uint32_t *id) { (void)id; return 0; }
HLE_FORCE("_ZN2nn3hid30SetNpadJoyAssignmentModeSingleERKj", "Joy-Con 单持；宿主忽略", void, h_hid_JoySingle, const uint32_t *id) { (void)id; }
HLE_FORCE("_ZN2nn3hid18SwapNpadAssignmentERKjS2_", "交换手柄号；宿主忽略", void, h_hid_Swap, const uint32_t *a, const uint32_t *b) { (void)a; (void)b; }
HLE_FORCE("_ZN2nn3hid21StartLrAssignmentModeEv", "L/R 分配模式；宿主忽略", void, h_hid_LrStart, void) {}
HLE_FORCE("_ZN2nn3hid20StopLrAssignmentModeEv", "L/R 分配模式；宿主忽略", void, h_hid_LrStop, void) {}
// Controller applet (system UI): report "1 player, current controller selected" directly. ControllerSupportResultInfo = { int playerCount; uint32_t selectedId; }
HLE_FORCE("_ZN2nn3hid21ShowControllerSupportEPNS0_27ControllerSupportResultInfoERKNS0_20ControllerSupportArgE", "手柄小程序；宿主直接报已选好",
          Result, h_hid_ShowCtrl, int32_t *info, const void *arg) { (void)arg; if (info) { info[0] = 1; info[1] = (int32_t)connected_id(); } return RES_OK; }
// Six-axis: handles are handed out, but no states can be read (0 entries)
HLE_FORCE("_ZN2nn3hid23GetSixAxisSensorHandlesEPNS0_19SixAxisSensorHandleEiRKjNS_4util10BitFlagSetILi32ENS0_12NpadStyleTagEEE", "六轴句柄",
          int, h_hid_SixHandles, uint32_t *h, int n, const uint32_t *id, uint32_t style) { (void)style; int k = n < 2 ? n : 2; for (int i = 0; i < k; i++) h[i] = (*id << 8) | (uint32_t)i; return k; }
HLE_FORCE("_ZN2nn3hid18StartSixAxisSensorERKNS0_19SixAxisSensorHandleE", "六轴；宿主不提供", void, h_hid_SixStart, const uint32_t *h) { (void)h; }
HLE_FORCE("_ZN2nn3hid22GetSixAxisSensorStatesEPNS0_18SixAxisSensorStateEiRKNS0_19SixAxisSensorHandleE", "六轴状态；宿主不提供（0 条）",
          int, h_hid_SixStates, void *out, int n, const uint32_t *h) { (void)out; (void)n; (void)h; return 0; }
// Vibration: routed to the haptics sink (hle_haptics.c). Without a sink (the Mac host, no DM_HAPTIC_FAKE)
// the host reports no devices, which is the behaviour every game saw before haptics existed.
HLE_FORCE("_ZN2nn3hid20IsVibrationPermittedEv", "震动许可：有触感接收端（iOS Core Haptics）时为真", unsigned, h_hid_VibPermit, void) { return (unsigned)dm_haptics_supported(); }
HLE_FORCE("_ZN2nn3hid25GetVibrationDeviceHandlesEPNS0_21VibrationDeviceHandleEiRKjNS_4util10BitFlagSetILi32ENS0_12NpadStyleTagEEE", "震动设备：左右两个马达，句柄 (npad<<8)|motor",
          int, h_hid_VibHandles, void *h, int n, const uint32_t *id, uint32_t style) {
    (void)style;
    if (!dm_haptics_supported() || n <= 0) return 0;
    int k = n < 2 ? n : 2; uint32_t *out = h;
    for (int i = 0; i < k; i++) out[i] = dm_haptics_handle((int)*id, i);
    static int logged;
    if (logged++ < 4) LOG("震动：游戏请求 %d 个设备句柄（npad %#x）→ %#x %#x", k, *id, out[0], k > 1 ? out[1] : 0);
    return k;
}
HLE_FORCE("_ZN2nn3hid25InitializeVibrationDeviceERKNS0_21VibrationDeviceHandleE", "震动：初始化一个马达通道", void, h_hid_InitVib, const uint32_t *h) { dm_haptics_init_device(*h); }
HLE_FORCE("_ZN2nn3hid18SendVibrationValueERKNS0_21VibrationDeviceHandleERKNS0_14VibrationValueE", "震动：把 Switch 震动值送给触感接收端", void, h_hid_SendVib, const uint32_t *h, const dm_vib_value *v) { dm_haptics_send(*h, v); }
// Values queued by the vibration graph (VibrationNode::Update) reach it from the sdk's internal FlushEnqueuedVibrationValues through the PLT, into the hid service proxy (sm abort). Metroid Dread
// sets handles on its VibrationTargets even when the host reports 0 devices, so values get queued; return success and the flush clears its count as usual
HLE_FORCE("_ZN2nn3hid6detail19SendVibrationValuesEPKNS0_21VibrationDeviceHandleEPKNS0_14VibrationValueEi", "震动（批量，震动图 flush 用）：逐通道转给触感接收端", Result, h_hid_SendVibs, const uint32_t *h, const dm_vib_value *v, int n) { dm_haptics_send_many(h, v, n); return RES_OK; }
// Mouse / keyboard: not connected
HLE_FORCE("_ZN2nn3hid15InitializeMouseEv", "鼠标；宿主不提供", void, h_hid_InitMouse, void) {}
HLE_FORCE("_ZN2nn3hid14GetMouseStatesEPNS0_10MouseStateEi", "鼠标状态：0 条", int, h_hid_MouseStates, void *out, int n) { (void)out; (void)n; return 0; }
// Single state (13 Sentinels calls it every frame; without the hook the sdk aborts on its uninitialized mouse resource): MouseState is 0x28 bytes {sampling number i64, x, y, dx, dy, wheel dx, dy, buttons, attributes}; attributes 0 = not connected
HLE_FORCE("_ZN2nn3hid13GetMouseStateEPNS0_10MouseStateE", "鼠标状态：未连接（MouseState 0x28 字节清零）", void, h_hid_MouseState, void *s) {
    memset(s, 0, 0x28); ((int64_t *)s)[0] = atomic_fetch_add(&g_sampling, 1) + 1;
}
HLE_FORCE("_ZN2nn3hid18InitializeKeyboardEv", "键盘；宿主不提供", void, h_hid_InitKbd, void) {}
HLE_FORCE("_ZN2nn3hid16GetKeyboardStateEPNS0_13KeyboardStateE", "键盘状态：未连接（KeyboardState 0x30 字节清零）", void, h_hid_KbdState, void *s) {
    memset(s, 0, 0x30); ((int64_t *)s)[0] = atomic_fetch_add(&g_sampling, 1) + 1;
}

// ---------- SDK 12 interfaces (Engage): single GetNpadState, mode queries / setters, touch screen, gestures, six-axis details, controller colors, etc. ----------
// In the original all of these read hid shared memory or go through the hid service; the host has no shared memory and answers from the same front-end state as above. Library functions that only modify the game's own buffers (vibration file parse / generate, ControllerSupportArg SetDefault / SetExplainText) keep running in the SDK
HLE_FORCE("_ZN2nn3hid12GetNpadStateEPNS0_16NpadFullKeyStateERKj", "全键手柄当前状态", void, h_hid_StateFullKey, NpadState *s, const uint32_t *id) { fill(s, *id, STYLE_FULLKEY); }
HLE_FORCE("_ZN2nn3hid12GetNpadStateEPNS0_17NpadHandheldStateERKj", "掌机手柄当前状态", void, h_hid_StateHandheld, NpadState *s, const uint32_t *id) { fill(s, *id, STYLE_HANDHELD); }
HLE_FORCE("_ZN2nn3hid12GetNpadStateEPNS0_16NpadJoyDualStateERKj", "双 Joy-Con 当前状态（宿主不接）", void, h_hid_StateJoyDual, NpadState *s, const uint32_t *id) { fill(s, *id, STYLE_JOYDUAL); }
HLE_FORCE("_ZN2nn3hid12GetNpadStateEPNS0_16NpadJoyLeftStateERKj", "左 Joy-Con 当前状态（宿主不接）", void, h_hid_StateJoyLeft, NpadState *s, const uint32_t *id) { fill(s, *id, STYLE_JOYLEFT); }
HLE_FORCE("_ZN2nn3hid12GetNpadStateEPNS0_17NpadJoyRightStateERKj", "右 Joy-Con 当前状态（宿主不接）", void, h_hid_StateJoyRight, NpadState *s, const uint32_t *id) { fill(s, *id, STYLE_JOYRIGHT); }

static uint32_t g_comm_mode, g_hh_mode;
HLE_FORCE("_ZN2nn3hid18GetNpadJoyHoldTypeEv", "Joy-Con 握持方向：回报设过的值", int, h_hid_GetHold, void) { return (int)g_hold_type; }
HLE_FORCE("_ZN2nn3hid24GetSupportedNpadStyleSetEv", "支持的手柄样式：全部", uint32_t, h_hid_GetStyles, void) { return g_styles_supported; }
HLE_FORCE("_ZN2nn3hid24GetNpadCommunicationModeEv", "无线通信模式：回报设过的值", int, h_hid_GetComm, void) { return (int)g_comm_mode; }
HLE_FORCE("_ZN2nn3hid24SetNpadCommunicationModeENS0_21NpadCommunicationModeE", "无线通信模式；宿主只记下", void, h_hid_SetComm, int m) { g_comm_mode = (uint32_t)m; }
HLE_FORCE("_ZN2nn3hid29GetNpadHandheldActivationModeEv", "掌机激活模式：回报设过的值", int, h_hid_GetHh, void) { return (int)g_hh_mode; }
HLE_FORCE("_ZN2nn3hid29SetNpadHandheldActivationModeENS0_26NpadHandheldActivationModeE", "掌机激活模式；宿主只记下", void, h_hid_SetHh, int m) { g_hh_mode = (uint32_t)m; }
HLE_FORCE("_ZN2nn3hid28SetNpadJoyAssignmentModeDualERKj", "Joy-Con 双持；宿主忽略", void, h_hid_JoyDual, const uint32_t *id) { (void)id; }
HLE_FORCE("_ZN2nn3hid30SetNpadJoyAssignmentModeSingleERKjNS0_17NpadJoyDeviceTypeE", "Joy-Con 单持（指定左右）；宿主忽略", void, h_hid_JoySingleT, const uint32_t *id, int t) { (void)id; (void)t; }
HLE_FORCE("_ZN2nn3hid30SetNpadJoyAssignmentModeSingleEPjRKjNS0_17NpadJoyDeviceTypeE", "Joy-Con 单持（取回另一半）；宿主忽略，没有另一半", int, h_hid_JoySingleOut, uint32_t *out, const uint32_t *id, int t) { (void)out; (void)id; (void)t; return 0; }
HLE_FORCE("_ZN2nn3hid23MergeSingleJoyAsDualJoyERKjS2_", "合并单 Joy-Con；宿主忽略，报成功", Result, h_hid_Merge, const uint32_t *a, const uint32_t *b) { (void)a; (void)b; return RES_OK; }
// Bound and signalled once for the controller that is connected (the hid service reports the controller already attached at start the same way). Link's Awakening reads no controller state
// until this event has fired: without it GetNpadState was never called and the title's "Press L+R" never saw a press
HLE_FORCE("_ZN2nn3hid27BindNpadStyleSetUpdateEventERKjPNS_2os15SystemEventTypeENS3_14EventClearModeE", "手柄样式变化事件：绑定，连着的手柄触发一次", void, h_hid_BindStyleEv,
          const uint32_t *id, void *ev, int clear) { hle_bind_system_event(ev, clear); if (*id == connected_id()) hle_signal_system_event(ev); }
HLE_FORCE("_ZN2nn3hid19GetPlayerLedPatternERKj", "玩家灯：1 号灯", uint8_t, h_hid_Led, const uint32_t *id) { (void)id; return 1; }
// NpadControllerColor = { main RGBA, button RGBA }: grey body, black buttons
static void ctrl_color(uint32_t *c) { c[0] = 0xff828282u; c[1] = 0xff0f0f0fu; }
HLE_FORCE("_ZN2nn3hid22GetNpadControllerColorEPNS0_19NpadControllerColorERKj", "手柄颜色", Result, h_hid_Color, uint32_t *c, const uint32_t *id) { (void)id; ctrl_color(c); return RES_OK; }
HLE_FORCE("_ZN2nn3hid22GetNpadControllerColorEPNS0_19NpadControllerColorES2_RKj", "左右 Joy-Con 颜色", Result, h_hid_Color2, uint32_t *l, uint32_t *r, const uint32_t *id) { (void)id; ctrl_color(l); ctrl_color(r); return RES_OK; }
HLE_FORCE("_ZN2nn3hid21ShowControllerSupportERKNS0_20ControllerSupportArgE", "手柄小程序（不要结果）；宿主直接报已选好", Result, h_hid_ShowCtrl0, const void *arg) { (void)arg; return RES_OK; }
HLE_FORCE("_ZN2nn3hid24ShowControllerStrapGuideEv", "腕带说明小程序；宿主不显示", Result, h_hid_Strap, void) { return RES_OK; }
HLE_FORCE("_ZN2nn3hid28ShowControllerFirmwareUpdateERKNS0_27ControllerFirmwareUpdateArgE", "手柄固件更新小程序；宿主不显示", Result, h_hid_FwUpd, const void *arg) { (void)arg; return RES_OK; }

// Touch screen: TouchScreenState<N> = { int64 sampling number; int32 count; int32 reserved; TouchState[N] }. Report 0 touch points for now (only the 16-byte header is written)
static void touch_head(void *s) { int64_t *p = s; p[0] = atomic_fetch_add(&g_sampling, 1) + 1; ((int32_t *)s)[2] = 0; ((int32_t *)s)[3] = 0; }
HLE_FORCE("_ZN2nn3hid21InitializeTouchScreenEv", "连 hid 触屏；宿主暂报无触点", void, h_hid_InitTouch, void) {}
#define TOUCH(N) \
    HLE_FORCE("_ZN2nn3hid19GetTouchScreenStateILm" #N "EEEvPNS0_16TouchScreenStateIXT_EEE", "触屏当前状态：0 个触点", void, h_hid_Touch1_##N, void *s) { touch_head(s); } \
    HLE_FORCE("_ZN2nn3hid20GetTouchScreenStatesILm" #N "EEEiPNS0_16TouchScreenStateIXT_EEEi", "触屏状态：1 条、0 个触点", int, h_hid_TouchN_##N, void *s, int n) { if (n <= 0) return 0; touch_head(s); return 1; }
TOUCH(1) TOUCH(2) TOUCH(3) TOUCH(4) TOUCH(5) TOUCH(6) TOUCH(7) TOUCH(8) TOUCH(9) TOUCH(10) TOUCH(11) TOUCH(12) TOUCH(13) TOUCH(14) TOUCH(15) TOUCH(16)
HLE_FORCE("_ZN2nn3hid17InitializeGestureEv", "手势；宿主不提供", void, h_hid_InitGesture, void) {}
HLE_FORCE("_ZN2nn3hid16GetGestureStatesEPNS0_12GestureStateEi", "手势状态：0 条", int, h_hid_Gestures, void *out, int n) { (void)out; (void)n; return 0; }

// Six-axis details: at rest, fusion off; SixAxisSensorState 0x60 bytes zeroed (attributes = 0: not connected)
HLE_FORCE("_ZN2nn3hid17StopSixAxisSensorERKNS0_19SixAxisSensorHandleE", "六轴；宿主不提供", void, h_hid_SixStop, const uint32_t *h) { (void)h; }
HLE_FORCE("_ZN2nn3hid21GetSixAxisSensorStateEPNS0_18SixAxisSensorStateERKNS0_19SixAxisSensorHandleE", "六轴当前状态：未连接", void, h_hid_SixState, void *s, const uint32_t *h) { (void)h; memset(s, 0, 0x60); }
HLE_FORCE("_ZN2nn3hid21IsSixAxisSensorAtRestERKNS0_19SixAxisSensorHandleE", "六轴静止：是", unsigned, h_hid_SixRest, const uint32_t *h) { (void)h; return 1; }
// The console's own six-axis sensor (SevenSixAxisSensor, the body's orientation in handheld mode; The Legend of Zelda: Breath of the Wild): takes a work buffer and is switched through the hid service. The host provides none: switching succeeds, 0 states
HLE_FORCE("_ZN2nn3hid28InitializeSevenSixAxisSensorEPvm", "机身六轴初始化；宿主不提供", Result, h_hid_SevenInit, void *buf, size_t n) { (void)buf; (void)n; return RES_OK; }
HLE_FORCE("_ZN2nn3hid26FinalizeSevenSixAxisSensorEv", "机身六轴", Result, h_hid_SevenFini, void) { return RES_OK; }
HLE_FORCE("_ZN2nn3hid23StartSevenSixAxisSensorEv", "机身六轴；宿主不提供", Result, h_hid_SevenStart, void) { return RES_OK; }
HLE_FORCE("_ZN2nn3hid22StopSevenSixAxisSensorEv", "机身六轴", Result, h_hid_SevenStop, void) { return RES_OK; }
HLE_FORCE("_ZN2nn3hid32ResetSevenSixAxisSensorTimestampEv", "机身六轴时间戳", void, h_hid_SevenReset, void) {}
HLE_FORCE("_ZN2nn3hid27GetSevenSixAxisSensorStatesEPNS0_23SevenSixAxisSensorStateEi", "机身六轴状态：0 条", int, h_hid_SevenStates, void *out, int n) { (void)out; (void)n; return 0; }
HLE_FORCE("_ZN2nn3hid25EnableSixAxisSensorFusionERKNS0_19SixAxisSensorHandleEb", "六轴融合；宿主忽略", void, h_hid_SixFusionOn, const uint32_t *h, uint32_t on) { (void)h; (void)on; }
HLE_FORCE("_ZN2nn3hid28IsSixAxisSensorFusionEnabledERKNS0_19SixAxisSensorHandleE", "六轴融合：未开", unsigned, h_hid_SixFusionQ, const uint32_t *h) { (void)h; return 0; }
HLE_FORCE("_ZN2nn3hid25GetGyroscopeZeroDriftModeERKNS0_19SixAxisSensorHandleE", "陀螺零漂模式：标准", int, h_hid_GyroGet, const uint32_t *h) { (void)h; return 1; }
HLE_FORCE("_ZN2nn3hid25SetGyroscopeZeroDriftModeERKNS0_19SixAxisSensorHandleERKNS0_22GyroscopeZeroDriftModeE", "陀螺零漂模式；宿主忽略", void, h_hid_GyroSet, const uint32_t *h, const int *m) { (void)h; (void)m; }
HLE_FORCE("_ZN2nn3hid32GetSixAxisSensorFusionParametersEPfS1_RKNS0_19SixAxisSensorHandleE", "六轴融合参数：0", void, h_hid_FusionGet, float *a, float *b, const uint32_t *h) { (void)h; *a = 0; *b = 0; }
HLE_FORCE("_ZN2nn3hid32SetSixAxisSensorFusionParametersERKNS0_19SixAxisSensorHandleEff", "六轴融合参数；宿主忽略", void, h_hid_FusionSet, const uint32_t *h, float a, float b) { (void)h; (void)a; (void)b; }
HLE_FORCE("_ZN2nn3hid34ResetSixAxisSensorFusionParametersERKNS0_19SixAxisSensorHandleE", "六轴融合参数；宿主忽略", void, h_hid_FusionReset, const uint32_t *h) { (void)h; }
HLE_FORCE("_ZN2nn3hid41IsFirmwareUpdateAvailableForSixAxisSensorERKNS0_19SixAxisSensorHandleE", "六轴固件更新：无", unsigned, h_hid_SixFw, const uint32_t *h) { (void)h; return 0; }
// Vibration device info / actual value: describe the two phone actuator channels, report the last accepted value
HLE_FORCE("_ZN2nn3hid22GetVibrationDeviceInfoEPNS0_19VibrationDeviceInfoERKNS0_21VibrationDeviceHandleE", "震动设备信息：线性马达，1 = 左 2 = 右", void, h_hid_VibInfo, uint32_t *info, const uint32_t *h) {
    int on = dm_haptics_supported();
    info[0] = on ? 1u : 0u;                                      // type: linear resonant actuator
    info[1] = on ? (uint32_t)(dm_haptics_device(*h) + 1) : 0u;   // position: 1 left, 2 right
}
HLE_FORCE("_ZN2nn3hid23GetActualVibrationValueEPNS0_14VibrationValueERKNS0_21VibrationDeviceHandleE", "实际震动值：最后一次被接受的值", void, h_hid_VibActual, dm_vib_value *v, const uint32_t *h) { *v = dm_haptics_actual(*h); }
