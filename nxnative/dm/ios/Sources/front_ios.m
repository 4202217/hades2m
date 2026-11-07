// iOS front end (the iOS implementation of front.h): a full-screen Metal view whose picture aspect follows the screen (front_game_res feeds the game a screen-aspect internal resolution, so the engine widens natively instead of leaving black bars) + a full-screen virtual gamepad layer (DmGamepad.m, controls drawn over the picture).
// The host runs on the big-stack thread started by main.m; front_run runs the guest directly on the calling thread.
#import <UIKit/UIKit.h>
#import <QuartzCore/CAMetalLayer.h>
#import <Metal/Metal.h>
#import <AVFAudio/AVFAudio.h>
#include <errno.h>
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include "front.h"
#include "hle.h"
#include "dm.h"

UIView *dm_gamepad_view(CGRect frame);
void dm_gamepad_set_bars(double x, double w);   // the picture's rectangle: the shoulder buttons move into the bars
void dm_gamepad_reset(void);
void front_ios_boot(void);
int front_ios_boot_ok(void);
void dm_haptics_ios_init(void);   // Haptics.m: Core Haptics engines for the Switch HD vibration stream

static CAMetalLayer *g_layer;
static void front_ios_audio_watch(void);
// The panel's native long side in pixels, cached on the main thread: front_game_res runs on guest threads (thread-local resolution slots
// are filled for every guest thread), and UIKit such as UIScreen.nativeBounds must not be touched off the main thread.
// 0 = not read yet; front_game_res falls back to 16:9 in that case (the value is read before the host starts, see viewDidAppear).
static _Atomic uint32_t g_screen_long_px, g_screen_short_px;
static void update_screen_aspect(void) {   // main thread only
    CGRect nb = UIScreen.mainScreen.nativeBounds;
    double w = fmax(nb.size.width, nb.size.height), h = fmin(nb.size.width, nb.size.height);
    atomic_store(&g_screen_long_px, (uint32_t)lround(w));
    atomic_store(&g_screen_short_px, (uint32_t)lround(h));
}

@interface DmGameView : UIView
@end
@implementation DmGameView
+ (Class)layerClass { return CAMetalLayer.class; }
- (instancetype)initWithFrame:(CGRect)f {
    if ((self = [super initWithFrame:f])) {
        CAMetalLayer *l = (CAMetalLayer *)self.layer;
        l.device = MTLCreateSystemDefaultDevice(); l.pixelFormat = MTLPixelFormatBGRA8Unorm;
        l.contentsScale = UIScreen.mainScreen.nativeScale; l.framebufferOnly = NO; l.opaque = YES;
        self.backgroundColor = UIColor.blackColor;
    }
    return self;
}
- (void)layoutSubviews {
    [super layoutSubviews];
    CAMetalLayer *l = (CAMetalLayer *)self.layer;
    l.drawableSize = CGSizeMake(self.bounds.size.width * l.contentsScale, self.bounds.size.height * l.contentsScale);
}
@end

@interface DmRootVC : UIViewController
@property (nonatomic, strong) DmGameView *game;
@property (nonatomic, strong) UIView *pad;
@end
@implementation DmRootVC
- (void)viewDidLoad {
    [super viewDidLoad];
    self.view.backgroundColor = UIColor.blackColor;
    if (!front_ios_boot_ok()) {   // No game files: show a hint and do not start the host
        UILabel *lb = [[UILabel alloc] initWithFrame:CGRectInset(self.view.bounds, 40, 40)];
        lb.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
        lb.numberOfLines = 0; lb.textAlignment = NSTextAlignmentCenter; lb.textColor = UIColor.whiteColor;
        lb.text = @"Game files not found.\nPut the RomFS folder into this App's Documents as \"romfs\" (Files app), then restart.\n\n没找到游戏文件：把 RomFS 文件夹以 romfs 为名放进本 App 的「文稿」后重开。";
        [self.view addSubview:lb]; return;
    }
    self.game = [[DmGameView alloc] initWithFrame:self.view.bounds];
    [self.view addSubview:self.game];
    g_layer = (CAMetalLayer *)self.game.layer;
    self.pad = dm_gamepad_view(self.view.bounds);
    self.pad.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
    [self.view addSubview:self.pad];
}
// The game picture's rectangle inside the output frame, in points: x = the left bar's width, w = the picture's width.
// Reported by the graphics backend after the aspect fit (on the present thread); the pad uses it to move the shoulder buttons
// into the bars when there are any. 0 = the picture fills the frame.
static volatile double g_pic_x, g_pic_w;
void front_set_picture_rect(double x, double w) {   // the host's interface (front.h): pixels; the pad works in points
    double scale = UIScreen.mainScreen.nativeScale;
    if (scale <= 0) scale = 1;
    x /= scale; w /= scale;
    if (g_pic_x == x && g_pic_w == w) return;
    g_pic_x = x; g_pic_w = w;
    dispatch_async(dispatch_get_main_queue(), ^{ dm_gamepad_set_bars(x, w); });
}
// The picture's rectangle is an aspect fit of the canvas front_game_res hands the game, so the front end can compute it here
// without waiting for the first presented frame. Without this the shoulder buttons are laid out at their old place for the first
// launch moment and jump into the bars as soon as present_encode_m4 reports the rectangle (a visible flash on every launch).
// Reported through the same dm_gamepad_set_bars the present thread uses; when the real rectangle arrives it is identical, so the
// pad does not move again.
static void update_expected_picture_rect(UIView *pad) {
    if (!pad) return;
    uint64_t r = front_game_res(1);   // docked and handheld differ by 4/5 only, so the aspect -- which is all that decides the bars -- is the same
    double cw = (double)(uint32_t)r, ch = (double)(r >> 32);
    CGRect b = pad.bounds;
    if (cw <= 0 || ch <= 0 || b.size.width <= 0 || b.size.height <= 0) return;
    double s = fmin(b.size.width / cw, b.size.height / ch);
    double x = floor((b.size.width - cw * s) / 2), w = cw * s;
    static double logged_x = -1;   // log every change: the launch goes through more than one view size, and the sequence is what shows there is no jump
    if (x != logged_x) { logged_x = x; fprintf(stderr, "[dm] 肩键留白（本地）：%.0f 点（画布 %gx%g，视图 %gx%g）\n", x, cw, ch, b.size.width, b.size.height); }
    dm_gamepad_set_bars(x, w);
}

- (void)viewDidLayoutSubviews {
    [super viewDidLayoutSubviews];
    update_screen_aspect();
    update_expected_picture_rect(self.pad);
    CGRect b = self.view.bounds; if (!self.game) return;
    // Full screen. The game renders the 16:9 canvas front_game_res hands it (1792x1008, the size the user picked) and the present
    // scales it up to the panel's height; on a wider panel (19.5:9) that leaves a bar at either side, which is what the pad's
    // shoulder rule uses. The view itself is still full screen; only the picture inside it is letterboxed.
    self.game.frame = b;
}
- (void)viewDidAppear:(BOOL)a { [super viewDidAppear:a]; update_screen_aspect(); dm_haptics_ios_init(); front_ios_audio_watch(); front_ios_boot(); UIApplication.sharedApplication.idleTimerDisabled = YES; }   // a game: no touch input required, the display must not auto-lock; haptics start before boot so the guest sees the vibration devices at hid init
// front_ios_audio_watch() runs just BEFORE front_ios_boot() (which is what sets the audio session up, front_ios.m): installing the interruption observers in
// viewDidLoad, i.e. even earlier, makes iOS pick a different I/O buffer (5.33 ms instead of 5.00 ms on the iPhone Air), the sound card then asks for 256 frames
// per callback instead of 240, and the game's 4 blocks × 5 ms no longer fit in one callback -- 153 underruns in the first 2000 callbacks (measured).
// Moving the observers from viewDidLoad to viewDidAppear (still before the session exists) brought the buffer back to 5.00 ms: 240 frames per callback, 0 underruns,
// 0.0 ms of zero padding. So the requirement is "not in viewDidLoad", not "after the session": the observers must not be installed any earlier than this.
- (BOOL)prefersHomeIndicatorAutoHidden { return YES; }
- (UIRectEdge)preferredScreenEdgesDeferringSystemGestures { return UIRectEdgeAll; }   // Edge gestures need two swipes, to avoid accidental exits
- (BOOL)prefersStatusBarHidden { return YES; }
- (UIInterfaceOrientationMask)supportedInterfaceOrientations { return UIInterfaceOrientationMaskLandscape; }
@end
UIViewController *front_ios_root(void) { return [DmRootVC new]; }

void *front_metal_layer(void) { return (__bridge void *)g_layer; }
const char *front_documents_dir(void) {   // the app's Documents folder, for the diagnostics that write a file (the residency dump)
    static char buf[1024];
    NSString *doc = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES)[0];
    if (!doc) return NULL;
    snprintf(buf, sizeof buf, "%s", doc.UTF8String);
    return buf;
}
// Internal render resolution, fed to the game through the rewriter's resolution parameter slot (patch_nso.py PARAMS, KRES0: the Hades build's
// render-device init now reads it instead of the original 1280x720 constants). The game renders at this size AND lays its UI out for it, so the
// resolution decides where the HUD lands.
//
// The canvas is 16:9 and the present blit centres it, leaving a black bar on each side. Why not widen the canvas to the panel's aspect
// instead (the world widens, Hor+, no bars)?
//   The game's HUD is laid out in its own unit space (a 16:9 picture is 3840 units wide) and scales by the picture height, centred on the
//   picture's centreline. On a 1568-wide canvas its left column ends up 10 px from the left edge, while the landscape safe area needs
//   186 px (62 pt at 3x) for the Dynamic Island and the rounded corners. The left HUD column then lies under the island.
//   With the 16:9 canvas the bar is wider than the safe area, so the HUD stays clear of it.
//
// The other candidate -- shifting the HUD inward with the projection matrix (the dm_uiscreen.c route) -- does not work on this build: it needs
// each draw's NDC bounding box, and every draw here is an indexed draw (the "dw->count==4 && nattr==2 && stride==20" test never matches) and
// 41 of the 56 shader JSONs have no pos_attr, so the reflection gives nothing.
//
// The canvas's size is the other half of the trade: the game's own work grows with it, and both the app CPU and the frame-time p95
// have to stay inside the gate. Measured on the iPhone Air (the game's own work = the app CPU minus the host's submit; the p95 is the
// gate's 300-frame window, and the limit is 18.04 ms):
//   1568x720  (the panel aspect)  the game 2.64 ms  the GPU  5.72 ms  the p95 17.65-17.88  -- the HUD is at the edge
//   1280x720  (16:9)              the game 4.24 ms  the GPU  7.87 ms  the p95 17.82-17.93
//   1408x792  (16:9)              the game 4.89 ms  the GPU  8.33 ms  the p95 17.63-18.30
//   1792x1008 (16:9)              the game 8.22 ms  the GPU 11.51 ms  the app CPU p95 13.02 -- the gate's app-CPU limit is exceeded
//   2240x1260 (16:9, panel short side)  the game 10.42 ms  the GPU 13.98 ms  the app CPU p95 16.51
// The canvas is 16:9 at 1008 lines (1792x1008), the resolution the target form was measured at, so the black bar takes the rest of the panel.
// The 16:9 canvas costs about twice as much per pixel as the wide one (the game's own work per pixel doubles), which is why the canvas
// size is a real choice and not just "as big as the panel".
//
// A panel narrower than 16:9 (iPad 4:3) would overflow, so the height is reduced until the picture fits.
// DM_GAME_RES=<width>x<height> overrides docked mode (handheld = 4/5 of it), matching the Mac front end, for A/B measurements.
// The player picks one of six exact 16:9 canvases in the app's Settings bundle ("Render resolution", dm_appsettings.m) and it
// arrives here as DM_RES_TIER: a lower tier is the same picture with less of the panel's work, which is what a slower phone needs.
//
// Fill screen (DM_FILL_SCREEN, Settings → "Fill the screen"; default on for a panel narrower than 16:9, off for every other panel).
// A 16:9 canvas on a 4:3 iPad leaves a bar above and below -- 182 px each at tier 5 -- so the picture is also scaled UP above 1:1.
// Filling instead gives the canvas the panel's own aspect at the panel's short side, and the bars are gone:
//   iPad Air 5 (2746x1908)  16:9: canvas 2240x1260 -> 2746x1545 on screen, 182 px bars
//                           fill: canvas 2488x1728 -> 2746x1908, 1:1, no bars (measured in the city: 60 fps held, frame p95 17.0 ms both ways;
//                                 the GPU p95 moved within 13.7-16.0 ms across runs, the spread being what each scene draws rather than the canvas)
// Two things force the shape of this: the game's HUD is scaled by min(W/1792, H/1008) and anchored to the canvas' TOP-LEFT, so a
// taller canvas lifts the HUD off the bottom edge (measured: the health bar moves from 126 px above the screen's bottom to 398 px;
// it also clears the virtual stick, which covers it at 16:9); and the engine refuses any canvas above 4096x2160 (dm/tools/patch_nso.py
// raises its output cap to exactly that), which is why the short side is capped at 2160 rather than taken from the panel.
// Off for a panel WIDER than 16:9 (every iPhone): filling there widens the canvas instead, which is the Hor+ case the HUD cannot follow
// (the left column ends up under the Dynamic Island) -- see docs/research/c1-fullpanel-20261022.md.
static int fill_screen_wanted(void) {
    static int v = -1;
    if (v < 0) {
        unsigned lg = atomic_load(&g_screen_long_px), sh = atomic_load(&g_screen_short_px);
        if (!lg || !sh) return 0;   // panel size not read yet: 16:9 until it is
        int narrow = (uint64_t)lg * 9u < (uint64_t)sh * 16u;   // the panel is narrower than 16:9 (iPad)
        const char *e = getenv("DM_FILL_SCREEN");
        v = (e && *e) ? (*e != '0') : narrow;
        fprintf(stderr, "[dm] 填满屏幕：%s（面板 %ux%u %s 16:9，环境%s）\n", v ? "开" : "关", lg, sh,
                narrow ? "窄于" : "不窄于", e && *e ? "指定" : "未设，按面板");
    }
    return v;
}
static uint64_t game_res_aspect(int docked) {
    // 16:9 at 1008 lines (1792x1008): the canvas the user picked. The present scales it up 1.25x to the panel's height.
    unsigned h = 1008u;
    unsigned w = (h * 16u / 9u + 3u) / 4u * 4u;   // 1792
    if (fill_screen_wanted()) {
        // The panel's own aspect, at the panel's short side (the engine's 2160 cap is above every panel's short side today).
        unsigned sh = atomic_load(&g_screen_short_px), lg = atomic_load(&g_screen_long_px);
        if (sh && lg && sh <= lg) {
            // Docked = the panel's aspect at the panel's short side, capped by the engine's 2160 (see the note above: the engine refuses a
            // docked value above it and silently falls back to a mismatched canvas, which renders black). Handheld is 4/5 of that, the same
            // ratio the rest of this function uses.  iPad Air 5 (2746x1908): docked 3108x2160 -> handheld 2488x1728, both the panel's aspect.
            unsigned dh = (unsigned)(((uint64_t)sh * 5u + 2u) / 4u) & ~1u;
            if (dh > 2160u) dh = 2160u;
            unsigned dw = (unsigned)(((uint64_t)dh * lg + sh / 2) / sh) & ~3u;
            if (docked) return (uint64_t)dh << 32 | dw;
            return (uint64_t)(dh * 4u / 5u) << 32 | ((dw * 4u / 5u + 3u) / 4u * 4u);
        }
    }
    front_res_tier_canvas(front_res_tier(), &w, &h);   // the player's Settings choice; no tier leaves the default above
    if (!w || !h) { h = 1008u; w = (h * 16u / 9u + 3u) / 4u * 4u; }
    unsigned long_px = atomic_load(&g_screen_long_px);
    if (long_px && w > long_px) { h = (long_px * 9u / 16u) / 2u * 2u; w = (h * 16u / 9u + 3u) / 4u * 4u; }
    if (docked) { h = h * 5u / 4u; w = w * 5u / 4u; }
    return (uint64_t)h << 32 | w;
}
uint64_t front_game_res(int docked) {
    unsigned w, h; const char *e = getenv("DM_GAME_RES");
    if (e && sscanf(e, "%ux%u", &w, &h) == 2 && w >= 320 && h >= 180) {   // diagnostic override: docked mode; handheld = 4/5
        if (w > 4096u) w = 4096u; if (h > 2160u) h = 2160u;
        if (docked) return (uint64_t)h << 32 | w;
        return (uint64_t)(h * 4 / 5) << 32 | ((w * 4 / 5 + 3) / 4 * 4);
    }
    int t = front_res_tier();
    if (t >= 0) {   // a Settings choice: log the first one only, the slot is asked for repeatedly
        static int logged = -1;
        if (logged != t) { logged = t; unsigned cw, ch; front_res_tier_canvas(t, &cw, &ch); uint64_t r = game_res_aspect(0);
            fprintf(stderr, "[dm] 设置：渲染分辨率档位 %d（%ux%u）→ 画布 %ux%u\n", t, cw, ch, (unsigned)(uint32_t)r, (unsigned)(r >> 32)); }
    }
    return game_res_aspect(docked);
}
void front_run(void (*guest)(void)) { guest(); }   // Already on main.m's big-stack thread

// Software keyboard: show a UIAlertController on the main thread, the calling thread waits for the result
int front_text_input(const char *title, const char *sub, const char *guide, const char *initial, int max_chars, char *out, int out_cap) {
    (void)max_chars;
    if (!g_layer) return -1;
    __block int r = 0; __block NSString *text = nil;
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    NSString *t = title && *title ? @(title) : @"", *m = [NSString stringWithFormat:@"%s%s%s", sub ? sub : "", sub && *sub && guide && *guide ? "\n" : "", guide ? guide : ""];
    NSString *init = initial ? @(initial) : @"", *ph = guide ? @(guide) : @"";
    dispatch_async(dispatch_get_main_queue(), ^{
        UIAlertController *ac = [UIAlertController alertControllerWithTitle:t message:m preferredStyle:UIAlertControllerStyleAlert];
        [ac addTextFieldWithConfigurationHandler:^(UITextField *f) { f.text = init; f.placeholder = ph; }];
        [ac addAction:[UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:^(UIAlertAction *x) { (void)x; r = 1; text = ac.textFields.firstObject.text; dispatch_semaphore_signal(done); }]];
        [ac addAction:[UIAlertAction actionWithTitle:@"Cancel" style:UIAlertActionStyleCancel handler:^(UIAlertAction *x) { (void)x; dispatch_semaphore_signal(done); }]];
        UIViewController *vc = nil;
        for (UIScene *s in UIApplication.sharedApplication.connectedScenes) if ([s isKindOfClass:UIWindowScene.class]) for (UIWindow *w in ((UIWindowScene *)s).windows) if (w.isKeyWindow) vc = w.rootViewController;
        if (!vc) { dispatch_semaphore_signal(done); return; }
        [vc presentViewController:ac animated:YES completion:nil];
    });
    // 输入框期间客体时钟照常走：冻结会在关框那一帧把积压时间一次追完（实测一次 4.7 s 长帧 + 唤醒风暴），与 hle_swkbd.c 同步去掉
    dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
    if (!r) return 0;
    snprintf(out, out_cap, "%s", text.UTF8String ? text.UTF8String : "");
    return 1;
}

// Save write failed (most likely the phone's storage is full): show a notice on the main thread without blocking the game (as d2r-mobile does). hle_fs.c calls this when a save transaction is refused or a mount has to fall back to read-only;
// the iOS front end used not to implement it, so there was only a log line and the player never learned that progress was not being saved
void front_save_failed(int err) {
    LOG("前端：存档写失败 errno=%d，提示玩家", err);
    dispatch_async(dispatch_get_main_queue(), ^{
        UIViewController *vc = nil;
        for (UIScene *s in UIApplication.sharedApplication.connectedScenes) if ([s isKindOfClass:UIWindowScene.class]) for (UIWindow *w in ((UIWindowScene *)s).windows) if (w.isKeyWindow) vc = w.rootViewController;
        while (vc.presentedViewController) vc = vc.presentedViewController;
        if (!vc) { LOG("【红】前端：存档失败提示没有窗口 errno=%d", err); return; }
        NSString *m = err == ENOSPC ? @"存储空间不足，进度没有存上。请清理空间后再存档。\nStorage is full, so progress was not saved. Free up space and save again."
                                    : [NSString stringWithFormat:@"存档写入失败（errno %d），进度没有存上。\nSaving failed (errno %d); progress was not saved.", err, err];
        UIAlertController *a = [UIAlertController alertControllerWithTitle:@"存档失败 / Save failed" message:m preferredStyle:UIAlertControllerStyleAlert];
        [a addAction:[UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:nil]];
        [vc presentViewController:a animated:YES completion:nil];
    });
}

// Audio session: the frames RemoteIO asks for in each callback follow the I/O buffer duration, about 21 ms by default on iOS (1024 frames at 48 kHz). The game keeps only 4 blocks × 5 ms = 20 ms in flight,
// so every callback came up 1.3 ms short and was zero-filled, about 47 gaps per second (crackling; reproducible on the Mac by forcing 1024-frame callbacks: 1999 underruns in 2000 callbacks). Ask for 5 ms and the system picks the nearest it supports;
// the category stays the default SoloAmbient (silent switch and interrupting other apps' audio behave as before). Set once at boot and again on return to the foreground (the system may reset it after an interruption)
static int g_session_off;   // 上一次激活失败（别的 App 或电话还占着会话）：等 App 回到前台再试
void front_ios_audio_session(void) {
    AVAudioSession *s = AVAudioSession.sharedInstance; NSError *e = nil;
    if (![s setCategory:AVAudioSessionCategorySoloAmbient error:&e]) LOG("【红】音频会话类别设不上：%s", e.localizedDescription.UTF8String);
    [s setPreferredSampleRate:48000 error:nil];
    if (![s setPreferredIOBufferDuration:0.005 error:&e]) LOG("【红】音频会话 I/O 缓冲时长设不上：%s", e.localizedDescription.UTF8String);
    g_session_off = ![s setActive:YES error:&e];
    if (g_session_off) LOG("【红】音频会话激活失败：%s", e.localizedDescription.UTF8String);
    LOG("音频会话：采样率 %.0f Hz，I/O 缓冲 %.2f ms（要 5 ms）", s.sampleRate, s.IOBufferDuration * 1000);
}

// App 在后台、或音频会话被别的 App 接管（Siri、闹铃、来电）时，真正的出声要停掉，由静音接收端维持游戏的节奏。
// 没有这个打断观察者，AudioUnit 还挂着「已启动」，而 iOS 早就不回调了：等缓冲的游戏音频就一直没声音，直到重开 App。
// 这一套都在主队列上跑，和场景回调一样
static int g_audio_bg, g_audio_intr;
static void front_ios_audio_apply(void) { dm_audio_suspend(g_audio_bg || g_audio_intr); }
static void front_ios_audio_watch(void) {
    NSNotificationCenter *nc = NSNotificationCenter.defaultCenter;
    [nc addObserverForName:AVAudioSessionInterruptionNotification object:nil queue:NSOperationQueue.mainQueue usingBlock:^(NSNotification *n) {
        NSUInteger type = [n.userInfo[AVAudioSessionInterruptionTypeKey] unsignedIntegerValue], reason = [n.userInfo[AVAudioSessionInterruptionReasonKey] unsignedIntegerValue];
        if (type == AVAudioSessionInterruptionTypeBegan) { LOG("音频会话：被打断（原因 %lu）", (unsigned long)reason); g_audio_intr = 1; }
        else {
            LOG("音频会话：打断结束"); g_audio_intr = 0;
            if (!g_audio_bg) front_ios_audio_session();
        }
        front_ios_audio_apply();
    }];
    [nc addObserverForName:AVAudioSessionMediaServicesWereResetNotification object:nil queue:NSOperationQueue.mainQueue usingBlock:^(NSNotification *n) {
        (void)n; LOG("音频会话：媒体服务被重置，重新设置会话和音频输出");
        front_ios_audio_session(); dm_audio_reset();
    }];
}

// Scene lifecycle (called by SceneDelegate in main.m): 0 resign active, 1 enter background, 2 return to foreground, 3 become active. On entering background, freeze the guest clock and suspend after in-flight GPU work completes (submitting GPU work in background gets the process killed)
void front_ios_focus_event(int ev) {
    if (!g_layer) return;
    switch (ev) {
        case 0: dm_gamepad_reset(); dm_focus_set(2); break;
        case 1: UIApplication.sharedApplication.idleTimerDisabled = NO; dm_focus_set(3); g_audio_bg = 1; front_ios_audio_apply(); guest_clock_freeze_for(1, 1); dm_gfx_suspend(1); break;
        case 2: UIApplication.sharedApplication.idleTimerDisabled = YES; dm_gfx_suspend(0); guest_clock_freeze_for(1, 0); front_ios_audio_session(); g_audio_bg = 0; front_ios_audio_apply(); dm_focus_set(2); break;
        case 3:   // 回到前台也顺带结束「没等到结束通知」的打断（不是每一次都有结束通知），并把之前激活失败的会话再试一次
            UIApplication.sharedApplication.idleTimerDisabled = YES;
            if (g_audio_intr) { LOG("音频会话：回到前台但没收到打断结束通知，按恢复处理"); g_audio_intr = 0; g_session_off = 1; }
            if (g_session_off) front_ios_audio_session();
            front_ios_audio_apply(); dm_focus_set(1); break;
    }
}
