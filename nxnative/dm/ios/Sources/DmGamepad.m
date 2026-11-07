// Virtual gamepad (Nintendo button positions) + physical controllers (GameController). Both are merged and handed to the host through dm_input_set (hle_hid.c, bit layout in dm_input.h).
//   Two layouts; one tap on the layout button at the top centre (labelled with the current layout, "2D" / "3D") switches them, and the choice is kept in NSUserDefaults (each game is its own app, so per game); the first launch uses the game profile's pad_layout (DM_GAME_PAD_FULL).
//   2D (simple): top corners ZL above L (one column), ZR above R likewise, round keys; − and + side by side at the top centre; the left stick in the bottom-left corner (its grab circle a soft disc with glass off and the glass face with it on, tilt measured from its centre, small dead zone);
//     the A (right) B (bottom) X (top) Y (left) diamond in the bottom-right corner, sliding a finger between keys switches keys. Each bottom corner holds a single control, so it sits close in (reaching halfway into the side safe-area inset).
//   3D (full, Pro Controller style): everything in 2D (left stick and diamond keep their place and size), plus a D-pad (four round keys with outward triangles, inward of and below the left stick, 8-way by angle), a right stick (inward of and below the diamond)
//     and L3 / R3 (small round keys in the top row, inward of ZL / ZR). Breath of the Wild crouches with L3 and uses the scope with R3; 3D World and Breath of the Wild turn the camera with the right stick.
//   Touch-down ownership: locked keys first (shoulders, − / +, the layout button, L3 / R3), then whichever of the diamond, the D-pad and the two sticks has the smallest distance / grab radius (the left stick's grab circle borders the D-pad).
//   Everything hugs the edges and corners to leave the picture room; controls sit over the picture. With liquid glass on (DM_PAD_GLASS,
//   remembered in NSUserDefaults, on by default) each key is a UIGlassEffect view in one UIGlassContainerEffect: the element IS the
//   button (no fill or ring under it), its label or SF Symbol sits on the glass, and a press lights a shape layer in the key's own
//   outline. A per-key scrim under the container keeps the label readable without dimming the picture. With glass off it falls back
//   to one CAShapeLayer per key and a fill swap on press, and a stick shows its grab circle as a soft disc (the glass face is what
//   shows that area otherwise). A stick only moves its knob; nothing redraws the whole screen.
//   The virtual gamepad (layout button included) hides while a physical controller is connected and comes back when it is removed. Physical controllers keep every button (D-pad and right stick included) and map by position: bottom = B, right = A, left = Y, top = X (same positions as on a Switch controller).
//   Position editing: a long press (0.6 s) on the layout button turns edit mode on. Every control -- shown then in both layouts' full set,
//   letters and triangles included -- gets a dashed outline around the group it moves with (the diamond, each stick, the D-pad, the two
//   shoulder columns, L3 / R3 and − / +) and can be dragged; offsets are kept in NSUserDefaults in u, so they survive a rotation or a
//   different phone. While editing the pad publishes no input. The layout key becomes a 完成 capsule and a 重置 capsule sits below it (one
//   size, one style): 完成 or a tap on empty space leaves the mode and saves, 重置 asks once and puts every group back.
//   Key taps give a light UIImpactFeedbackGenerator tap (DmPadHaptic, on by default, also gated by the Settings switch "Enable haptics"), on top of
//   whatever the game itself rumbles: a tap is the player's own finger, not the game's HD vibration, so it does not go through Haptics.m.
#import <UIKit/UIKit.h>
#import <GameController/GameController.h>
#import <CoreText/CoreText.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "dm_input.h"
#include "../../host/dm_padlayout.h"
#include "game_profile.h"

enum { B_A = 0, B_B, B_X, B_Y, B_LS, B_RS, B_L, B_R, B_ZL, B_ZR, B_PLUS, B_MINUS, B_LEFT, B_UP, B_RIGHT, B_DOWN, B_N };
enum { C_NONE = 0, C_STICK, C_DPAD, C_FACE, C_BTN, C_SWAP };   // which control a touch holds

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_touch_btn, g_pad_btn;
static int32_t g_touch_lx, g_touch_ly, g_touch_rx, g_touch_ry, g_pad_lx, g_pad_ly, g_pad_rx, g_pad_ry;
static int tilted(int32_t x, int32_t y) { return (int64_t)x * x + (int64_t)y * y > (int64_t)4000 * 4000; }
static void publish(void) {
    pthread_mutex_lock(&g_mu);
    int32_t lx = g_touch_lx, ly = g_touch_ly, rx = g_touch_rx, ry = g_touch_ry;
    if (tilted(g_pad_lx, g_pad_ly)) { lx = g_pad_lx; ly = g_pad_ly; }   // a tilted physical stick wins
    if (tilted(g_pad_rx, g_pad_ry)) { rx = g_pad_rx; ry = g_pad_ry; }
    dm_input_set(g_touch_btn | g_pad_btn, lx, ly, rx, ry);
    pthread_mutex_unlock(&g_mu);
}

// ---- Layout constants (in u; x measured from the outer edge, the right half mirrors) ----
static const CGFloat
    K_ZL_X = 44, K_ZL_Y = 40, K_ZL_R = 26,          // ZL / ZR: centre from the outer edge and the top, radius
    K_L_X = 44, K_L_Y = 108, K_L_R = 24,            // L / R: same column as ZL / ZR
    K_CORNER_IN = 0.5,                              // how far the bottom-corner controls reach into the side safe-area inset (in landscape there is only the rounded screen corner there; the Dynamic Island is mid-edge)
    K_FACE_X = 81, K_PAD_Y = 82,                    // diamond centre from the outer edge; diamond and stick centre from the bottom (B's lower edge about 5 above it)
    K_FACE_GAP = 50, K_FACE_R = 27,                 // diamond: key distance from the centre, key radius
    K_PILL_DX = 52, K_PILL_Y = 22, K_PILL_W = 50, K_PILL_H = 28,   // − / +: centre from the screen's midline (the middle goes to the layout button) and from the top, size
    K_SWAP_R = 17,                                  // layout button: top centre, level with − / +
    K_STICK_X = 81, K_STICK_R = 77, K_KNOB_R = 30,  // left stick: mirrors the diamond, the ring as large as the whole diamond (key distance + key radius), the knob a little larger than a diamond key
    K_STICK_GRAB = 1.4,                             // how near the ring centre a touch-down grabs a stick (times the ring radius)
    K_FULL = 0.8,                                   // full tilt at this fraction of the ring radius (the knob still travels to the ring edge)
    K_DEAD = 0.1,                                   // stick dead zone (times the ring radius)
    // controls only the 3D layout has
    K_LOW_X = 231, K_LOW_Y = 64,                    // D-pad / right stick centre from the outer edge and the bottom (inward of and below the left stick / the diamond, as on a Pro Controller)
    K_DPAD_R = 65, K_DPAD_BTN_R = 21, K_DPAD_BTN_D = 44,   // D-pad: overall radius (grab = D + R), each round key's radius and distance from the centre; neighbours clear each other by ~6u
    K_DPAD_GRAB = 1.3,                              // D-pad grab radius (times the outer edge)
    K_RSTICK_R = 60, K_RKNOB_R = 24,                // right stick: ring and knob radius
    K_CLICK_X = 102, K_CLICK_R = 17;                // L3 / R3: centre from the outer edge (level with ZL / ZR), radius
static const CGFloat K_GLASS_TINT = 0.35;           // liquid glass: black tint under each glass element, so the keys stay readable without dimming the picture (the HIG's 35% scrim, per element instead of full screen)
static const int k_btns[] = {B_ZL, B_L, B_ZR, B_R, B_MINUS, B_PLUS, B_LS, B_RS};   // keys locked at touch-down (L3 / R3 only in the 3D layout)
static NSString *const k_layout_key = @"DmPadLayout";   // NSUserDefaults: 1 = 3D, 0 = 2D
static NSString *const k_glass_key = @"DmPadGlass";     // NSUserDefaults: 1 = liquid glass backdrops, 0 = the plain fills (DM_PAD_GLASS overrides once and is remembered)
// Glass slots: one per key (0..B_N-1), then the layout button and the two stick faces. The stick face is the throw zone the
// knob moves in: a glass disc with no outline at all (the outline was what looked like a grey ring), so the area is visible
// without drawing a ring around the picture.
enum { G_SWAP = B_N, G_STICK0, G_STICK1, G_N };
static const CGFloat K_GLASS_TINT_STICK = 0.10;    // the stick face is the largest element: a lighter scrim than the keys
// Position-editing groups: what moves as one piece. The two shoulder columns carry ZL with L (and ZR with R) because they share a column
// and a player who moves one wants both; the layout button itself never moves (it is the way in and out of edit mode).
enum { P_FACE = 0, P_STICK0, P_STICK1, P_DPAD, P_SHL, P_SHR, P_LS, P_RS, P_MINUS, P_PLUS, P_N };
static NSString *const k_tap_key = @"DmPadHaptic";     // NSUserDefaults: 1 = a light tap for every key press (DM_PAD_HAPTIC overrides once and is remembered)
static NSString *const k_off_key = @"DmPadOffsets";    // NSUserDefaults: the player's drag offsets, 2 numbers per group, in u
// Edit mode's two buttons: one capsule size, one above the other in the middle of the top edge (完成 where the layout key sits,
// 重置 below it). The width clears the − / + pills, whose inner edges are K_PILL_DX - K_PILL_W / 2 = 27u from the middle.
static const CGFloat K_EDIT_BTN_W = 46, K_EDIT_BTN_H = 30, K_EDIT_BTN_Y2 = 58;   // 46 wide: the − / + pills start 27u from the middle, so the capsules clear them
static const CGFloat K_EDIT_HOLD = 0.6;            // seconds to hold the layout button before edit mode turns on
static const CGFloat K_EDIT_MOVE = 12;             // u a swap-button touch may travel before the long press is dropped (so a slide never opens edit mode)

@interface DmGamepad : UIView
@end
static __weak DmGamepad *g_view;

@implementation DmGamepad {
    NSMapTable<UITouch *, NSNumber *> *_owner;     // touch → control (with the key for C_BTN or the stick for C_STICK, << 8)
    UITouch *_stick[2]; CGPoint _stickNow[2];      // the touch holding each stick and where it is now (0 left, 1 right)
    uint64_t _faceBits, _dpadBits, _btnBits;
    NSMapTable<UITouch *, NSNumber *> *_faceOf;    // which keys each touch on the diamond currently holds
    CAShapeLayer *_key[B_N]; CATextLayer *_label[B_N];   // one layer per key
    CALayer *_glyph[B_N];                          // an SF Symbol drawn over a key (the D-pad's triangles, − / +)
    CAShapeLayer *_ring[2], *_knob[2];
    CAShapeLayer *_swap; CATextLayer *_swapLabel; BOOL _swapHeld;
    BOOL _sticksShown;                             // a stick was held at the last refresh
    BOOL _full;                                    // the 3D layout is current
    CGFloat _bar;                                  // the picture's left bar (points); 0 = the picture fills the frame
    BOOL _glass;                                   // liquid glass backdrops are on
    UIVisualEffectView *_glassHost;                 // the full-screen glass container, the view's first subview (below every shape layer)
    UIVisualEffectView *_glassEl[G_N];              // one glass element per slot (keys, layout button, stick faces)
    CAShapeLayer *_glassWash[G_N];                  // the pressed wash over each element, in the element's own outline
    CALayer *_glassDimLayer;                        // scrim host, drawn below the glass container
    CAShapeLayer *_glassDimShape[G_N];              // one scrim shape per element
    CGRect _glassFrame[G_N];                        // where the scrim shapes go (same geometry as the glass elements)
    BOOL _glassShown[G_N];                          // and which of them are on screen
    BOOL _glassTintDirty;                           // layout / switch changed: resync the scrim on the next pass
    // Position editing (a long press on the layout button) and the key-tap haptic
    CGPoint _off[P_N];                              // the player's drag offsets, in u (0, 0 = the built-in place)
    BOOL _edit;                                     // edit mode: drag the groups outlined in dashes; no input is published
    int _drag; UITouch *_dragTouch;                 // the group being dragged and the finger dragging it
    CGPoint _dragFrom, _dragBase;                   // where that finger went down, and the offset the drag started from
    CAShapeLayer *_editBox[P_N];                    // the dashed outlines, on screen only in edit mode
    CAShapeLayer *_resetBtn; CATextLayer *_resetLabel;   // "重置 / Reset", below the layout button, edit mode only
    UITouch *_swapTouch; CGPoint _swapFrom; int _holdGen;   // the touch on the layout button and the long press waiting on it
    UITouch *_resetTouch;                           // the touch on the edit-mode reset button (counts when released on it)
    UIImpactFeedbackGenerator *_tapper;             // a light tap per key press (DmPadHaptic)
    BOOL _tapOn;                                    // and whether it is enabled
    BOOL _tapSaved;                                 // the switch's value at the last read (to notice a change made in Settings)
}

static UIColor *k_idle, *k_on, *k_line, *k_ringFill, *k_text, *k_textOn;
static UIColor *k_textGlass, *k_glassOn;   // a glass key's label (no fill or ring under it) and its pressed wash
static CAShapeLayer *shape(CALayer *parent) {
    CAShapeLayer *l = [CAShapeLayer layer];
    l.fillColor = k_idle.CGColor; l.strokeColor = k_line.CGColor; l.lineWidth = 1.5;
    l.shadowColor = UIColor.blackColor.CGColor; l.shadowOpacity = 0.2; l.shadowRadius = 3; l.shadowOffset = CGSizeMake(0, 1);
    [parent addSublayer:l]; return l;
}
static CATextLayer *label(CALayer *parent, NSString *s) {
    CATextLayer *t = [CATextLayer layer];
    t.string = s; t.alignmentMode = kCAAlignmentCenter; t.foregroundColor = k_text.CGColor; t.contentsScale = UIScreen.mainScreen.scale;
    [parent addSublayer:t]; return t;
}
// An SF Symbol as a layer (Apple's own game-pad glyphs: arrowtriangle.*.fill, plus, minus). Drawn as a picture tinted with
// k_text / k_textOn, so a press changes it exactly like a label.
// A symbol as a solid-colour layer masked by the symbol's alpha: tinting is then just the layer's background colour, which is
// what a press has to change (imageWithTintColor: on a symbol image was not reliable -- the triangles came out black).
static CALayer *glyph(CALayer *parent, NSString *name) {
    UIImage *im = [UIImage systemImageNamed:name withConfiguration:
                   [UIImageSymbolConfiguration configurationWithPointSize:40 weight:UIImageSymbolWeightBold]];
    CALayer *l = [CALayer layer];
    l.backgroundColor = k_text.CGColor;
    CALayer *mask = [CALayer layer];
    mask.contents = (__bridge id)im.CGImage; mask.contentsGravity = kCAGravityResizeAspect;
    mask.contentsScale = UIScreen.mainScreen.scale;
    l.mask = mask;
    l.contentsScale = UIScreen.mainScreen.scale;
    [parent addSublayer:l]; return l;
}
static void placeGlyph(CALayer *g, CGPoint c, CGFloat size) {
    if (!g) return;
    g.bounds = CGRectMake(0, 0, size, size);
    g.position = c;
    g.mask.frame = CGRectMake(0, 0, size, size);
}
static void tintGlyph(CALayer *g, UIColor *col) { if (g) g.backgroundColor = col.CGColor; }
static BOOL fullOnly(int b) { return b == B_LS || b == B_RS || b >= B_LEFT; }   // keys shown only in the 3D layout
// Which position-editing group a key belongs to. Exhaustive over B_N: a key nobody names would not move with anything.
static int groupOf(int b) {
    if (b <= B_Y) return P_FACE;
    if (b == B_LS) return P_LS;
    if (b == B_RS) return P_RS;
    if (b == B_L || b == B_ZL) return P_SHL;
    if (b == B_R || b == B_ZR) return P_SHR;
    if (b == B_PLUS) return P_PLUS;
    if (b == B_MINUS) return P_MINUS;
    return P_DPAD;   // B_LEFT / B_UP / B_RIGHT / B_DOWN
}
// Edit mode shows every group in both layouts: the player arranges the pad once, and the 3D-only controls keep their place for
// when they switch to 3D. (Outside edit mode a group still follows its layout: D-pad, right stick, L3 / R3 are 3D only.)
static BOOL editShown(int g, BOOL full, BOOL edit) { return edit || full || (g != P_DPAD && g != P_STICK1 && g != P_LS && g != P_RS); }

- (instancetype)initWithFrame:(CGRect)f {
    if ((self = [super initWithFrame:f])) {
        self.multipleTouchEnabled = YES; self.opaque = NO; self.backgroundColor = UIColor.clearColor;
        _owner = [NSMapTable strongToStrongObjectsMapTable]; _faceOf = [NSMapTable strongToStrongObjectsMapTable];
        static dispatch_once_t once;
        dispatch_once(&once, ^{
            // each colour carries its own low opacity instead of lowering the whole view's alpha (that would add an off-screen pass for the full-screen layer)
            k_idle = [UIColor colorWithWhite:1 alpha:0.14]; k_on = [UIColor colorWithWhite:1 alpha:0.45]; k_line = [UIColor colorWithWhite:1 alpha:0.38];
            k_ringFill = [UIColor colorWithWhite:1 alpha:0.06];   // a stick's grab circle: with glass off it is the only thing showing where the finger may land
            k_text = [UIColor colorWithWhite:1 alpha:0.6]; k_textOn = [UIColor colorWithWhite:0 alpha:0.7];
            k_textGlass = [UIColor colorWithWhite:1 alpha:0.95]; k_glassOn = [UIColor colorWithWhite:1 alpha:0.28];
        });
        for (int i = 0; i < 2; i++) {
            // The grab circle is a soft disc, not an outline (a ring around the picture read as a grey hoop). With glass on the
            // stick's glass disc already fills that area, so it hides there; with glass off it is what shows where to put the finger.
            _ring[i] = shape(self.layer); _ring[i].fillColor = k_ringFill.CGColor; _ring[i].strokeColor = nil; _ring[i].shadowOpacity = 0; _ring[i].hidden = YES;
            _knob[i] = shape(self.layer);
            CAShapeLayer *in = [CAShapeLayer layer]; in.fillColor = nil; in.strokeColor = k_line.CGColor; in.lineWidth = 1.5;   // inner ring of the knob
            [_knob[i] addSublayer:in];
        }
        static NSString *const names[B_N] = {@"A", @"B", @"X", @"Y", @"L3", @"R3", @"L", @"R", @"ZL", @"ZR", @"", @""};
        // The D-pad keys and − / + carry an Apple game-pad symbol instead of a letter.
        static NSString *const glyphs[B_N] = {nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, @"plus", @"minus"};
        for (int b = 0; b < B_N; b++) {
            _key[b] = shape(self.layer);
            if (names[b].length) _label[b] = label(_key[b], names[b]);
            if (glyphs[b]) _glyph[b] = glyph(_key[b], glyphs[b]);
        }
        _glyph[B_UP] = glyph(_key[B_UP], @"arrowtriangle.up.fill"); _glyph[B_DOWN] = glyph(_key[B_DOWN], @"arrowtriangle.down.fill");
        _glyph[B_LEFT] = glyph(_key[B_LEFT], @"arrowtriangle.left.fill"); _glyph[B_RIGHT] = glyph(_key[B_RIGHT], @"arrowtriangle.right.fill");
        _swap = shape(self.layer); _swapLabel = label(_swap, @"");
        _resetBtn = shape(self.layer); _resetLabel = label(_resetBtn, @"重置");
        _resetBtn.hidden = YES; _resetLabel.hidden = YES;
        NSNumber *saved = [NSUserDefaults.standardUserDefaults objectForKey:k_layout_key];
        _full = saved ? saved.boolValue : DM_GAME_PAD_FULL;
        fprintf(stderr, "[dm] 手柄布局：存档 %s（档案默认 %d）→ %s\n", saved ? (saved.boolValue ? "3D" : "2D") : "(无)", DM_GAME_PAD_FULL, _full ? "3D" : "2D");
        [self readGlassSetting];
        [self buildGlass];
        [self loadOffsets];
        [self readTapSetting];
        // DM_PAD_EDIT=1 opens edit mode at launch: there is no way to inject a long press from a script, so a screenshot run needs
        // this to see the editor. Diagnostics only; it persists nothing and the next launch without it is back to normal.
        if (getenv("DM_PAD_EDIT")) { fprintf(stderr, "[dm] 手柄编辑：环境 DM_PAD_EDIT=%s，启动即进入\n", getenv("DM_PAD_EDIT")); _edit = 1; _swapTouch = nil; }
        [self applyLayout];
        NSNotificationCenter *nc = NSNotificationCenter.defaultCenter;
        [nc addObserver:self selector:@selector(padsChanged) name:GCControllerDidConnectNotification object:nil];
        [nc addObserver:self selector:@selector(padsChanged) name:GCControllerDidDisconnectNotification object:nil];
        // The Settings app writes DmPadHaptic in another process while this one is suspended: re-read on the way back.
        [nc addObserver:self selector:@selector(readTapSetting) name:UIApplicationDidBecomeActiveNotification object:nil];
        [nc addObserver:self selector:@selector(readTapSetting) name:UIApplicationWillEnterForegroundNotification object:nil];
        [self padsChanged];
    }
    return self;
}
// Liquid glass switch: DM_PAD_GLASS=0 / 1 decides once and is remembered in NSUserDefaults (a sideloaded app cannot be given
// environment variables from outside, so the env.txt that main.m parses is the only way in); with neither, glass is on.
- (void)readGlassSetting {
    const char *e = getenv("DM_PAD_GLASS");
    NSNumber *saved = [NSUserDefaults.standardUserDefaults objectForKey:k_glass_key];
    _glassTintDirty = YES;
    if (e) {
        _glass = strcmp(e, "0") != 0;
        [NSUserDefaults.standardUserDefaults setObject:@(_glass) forKey:k_glass_key];
    } else _glass = saved ? saved.boolValue : YES;
    fprintf(stderr, "[dm] 手柄玻璃：%s（存档 %s，环境 %s）→ %s\n", _glass ? "开" : "关",
            saved ? (saved.boolValue ? "开" : "关") : "(无)", e ? e : "(无)", _glass ? "开" : "关");
}
// ---- Key-tap haptics ----
// A light tap for every key that goes down. It is the player's own finger, not the game's HD rumble, so it uses UIKit's impact
// generator rather than Haptics.m: that file owns the Core Haptics player for the game's vibration stream, and one shared player
// would let the game's zero cut a tap short (and a tap's parameter update would sit in the game's next rumble).
// DM_PAD_HAPTIC=0 / 1 decides once and is remembered, like the glass switch; the Settings bundle's DmPadHaptic is the switch, and
// "Enable haptics" (DM_HAPTIC_OFF) turns every kind of feedback off at once, which is what that switch promises.
- (void)readTapSetting {
    const char *e = getenv("DM_PAD_HAPTIC");
    NSUserDefaults *d = NSUserDefaults.standardUserDefaults;
    NSNumber *saved = [d objectForKey:k_tap_key];
    // The Settings app is a different process: it writes this key while the game sits in the background, so the value has to be
    // re-read when the app comes back. A live change here means the player just flipped the switch; it is logged either way
    // because "I turned it off and it still taps" is exactly the report this line has to answer.
    BOOL before = _tapOn;
    if (e) {
        _tapOn = strcmp(e, "0") != 0;
        [d setObject:@(_tapOn) forKey:k_tap_key];
    } else _tapOn = saved ? saved.boolValue : (_tapSaved ? _tapSaved : YES);
    // "Enable haptics": the env variable is the boot-time answer (an env file pins it), and otherwise the Settings switch itself is
    // read here -- the host caches DM_HAPTIC_OFF at its first use, so a flag toggled in Settings while the game runs would
    // otherwise not reach the tap at all.
    // DM_HAPTIC_OFF is a flag, not a 0/1 value (hle_haptics.c: "set" turns haptics off, whatever it is set to), so any non-empty
    // value means off. Without it the Settings switch "Enable haptics" decides.
    const char *off = getenv("DM_HAPTIC_OFF");
    BOOL all = (off && *off) ? NO : ([d objectForKey:@"DmHaptics"] ? [d boolForKey:@"DmHaptics"] : YES);
    if (!all) _tapOn = NO;
    if (_tapSaved && before != _tapOn) {   // a later read that changed: the player just flipped one of the two switches
        fprintf(stderr, "[dm] 按键震动：已改为 %s（按键开关 %s，总开关 %s）\n", _tapOn ? "开" : "关",
                saved ? (saved.boolValue ? "开" : "关") : "(无)", all ? "开" : "关");
    } else if (!_tapSaved) {
        fprintf(stderr, "[dm] 按键震动：%s（存档 %s，环境 %s，总开关 %s）\n", _tapOn ? "开" : "关",
                saved ? (saved.boolValue ? "开" : "关") : "(无)", e ? e : "(无)", all ? "开" : "关");
    }
    _tapSaved = YES;
    // Warm the engine once so the first tap does not pay for its start-up. Built on the first read and kept afterwards: a switch
    // turned off and on again reuses it, and building one costs a haptic engine start.
    if (_tapOn && !_tapper) [self makeTapper];
    [_tapper prepare];
}
- (void)makeTapper { _tapper = [UIImpactFeedbackGenerator new]; }
- (void)tapKey {
    if (!_tapOn || !_tapper) return;
    [_tapper impactOccurredWithIntensity:0.6];   // lighter than the default: a key press, not a notification
    [_tapper prepare];
}

// ---- Drag offsets ----
// One (x, y) pair per group in u, so a rotation, a different phone or a different picture size keeps the arrangement the player
// built. Written on every drag end; read once at init.
- (void)loadOffsets {
    NSArray *a = nil;
    id raw = [NSUserDefaults.standardUserDefaults objectForKey:k_off_key];
    // Written as an NSArray, but a value that arrives from a plist by hand can be any array-like; take whatever is there.
    if ([raw isKindOfClass:NSArray.class]) a = raw;
    else if ([raw isKindOfClass:NSData.class]) a = [NSPropertyListSerialization propertyListWithData:raw options:NSPropertyListImmutable format:NULL error:NULL];
    if (![a isKindOfClass:NSArray.class]) a = nil;
    fprintf(stderr, "[dm] 手柄位置：存档 %s（%lu 项）\n", a ? "有" : "(无)", (unsigned long)a.count);
    if (a.count == 2 * P_N) for (int g = 0; g < P_N; g++) {
        _off[g] = CGPointMake([a[2 * g] doubleValue], [a[2 * g + 1] doubleValue]);
        if (!isfinite(_off[g].x)) _off[g].x = 0;
        if (!isfinite(_off[g].y)) _off[g].y = 0;
    }
    // Only log the groups that actually moved: the log line is how a bug report says whether the drag did anything.
    NSMutableString *moved = [NSMutableString string];
    for (int g = 0; g < P_N; g++) if (_off[g].x || _off[g].y) [moved appendFormat:@" %d:(%.1f,%.1f)", g, _off[g].x, _off[g].y];
    fprintf(stderr, "[dm] 手柄位置：%s\n", moved.length ? moved.UTF8String : "全部默认");
}
- (void)saveOffsets {
    NSMutableArray *a = [NSMutableArray arrayWithCapacity:2 * P_N];
    for (int g = 0; g < P_N; g++) { [a addObject:@(_off[g].x)]; [a addObject:@(_off[g].y)]; }
    [NSUserDefaults.standardUserDefaults setObject:a forKey:k_off_key];
}

// ---- Edit mode ----
// The outline of a group: the area its finger can grab, drawn as a dashed rounded rect. Centres come from the same getters the
// drawing and the hit tests use, so an outline always sits exactly where the controls are.
- (CGRect)editRect:(int)g {
    CGFloat u = self.u;
    switch (g) {
        case P_FACE: { CGPoint c = [self faceGroupCenter]; CGFloat r = self.faceGap + self.faceR; return CGRectMake(c.x - r, c.y - r, 2 * r, 2 * r); }
        case P_STICK0: case P_STICK1: { int i = g == P_STICK1; CGPoint c = [self stickHome:i]; CGFloat r = [self stickR:i]; return CGRectMake(c.x - r, c.y - r, 2 * r, 2 * r); }
        case P_DPAD: { CGPoint c = self.dpadCenter; CGFloat r = (K_DPAD_BTN_D + K_DPAD_BTN_R) * u; return CGRectMake(c.x - r, c.y - r, 2 * r, 2 * r); }
        case P_SHL: case P_SHR: {   // one column: ZL above L
            BOOL right = g == P_SHR; CGPoint z = [self dropCenter:right ? B_ZR : B_ZL], l = [self dropCenter:right ? B_R : B_L];
            CGFloat r = fmax([self dropR:right ? B_ZR : B_ZL], [self dropR:right ? B_R : B_L]);
            return CGRectMake(z.x - r, z.y - r, 2 * r, (l.y - z.y) + 2 * r);
        }
        case P_LS: case P_RS: { BOOL right = g == P_RS; CGPoint c = [self clickCenter:right ? B_RS : B_LS]; CGFloat r = K_CLICK_R * u; return CGRectMake(c.x - r, c.y - r, 2 * r, 2 * r); }
        default: return [self pillRect:g == P_PLUS ? B_PLUS : B_MINUS];
    }
}
- (int)editGroupAt:(CGPoint)p {   // topmost first; the groups do not overlap at their default places
    for (int g = P_N - 1; g >= 0; g--) if (editShown(g, _full, _edit) && CGRectContainsPoint([self editRect:g], p)) return g;
    return -1;
}
- (void)layoutEdit {
    for (int g = 0; g < P_N; g++) {
        CAShapeLayer *l = _editBox[g];
        BOOL show = _edit && editShown(g, _full, _edit);
        if (!l) {
            if (!show) continue;
            l = [CAShapeLayer layer];
            l.fillColor = [UIColor colorWithWhite:1 alpha:0.10].CGColor;
            l.strokeColor = [UIColor colorWithWhite:1 alpha:0.85].CGColor; l.lineWidth = 2;
            l.lineDashPattern = @[@6, @4];
            l.shadowOpacity = 0;
            [self.layer addSublayer:l]; _editBox[g] = l;
        }
        l.hidden = !show;
        if (!show) continue;
        CGRect r = [self editRect:g];
        l.frame = r;
        CGPathRef p = CGPathCreateWithRoundedRect(CGRectMake(0, 0, r.size.width, r.size.height), 12, 12, NULL);
        l.path = p; CFRelease(p);
    }
}
- (void)setEdit:(BOOL)edit {
    if (edit == _edit) return;
    [self reset];   // no key stays held across the switch, and the finger that opened the editor no longer owns anything
    _edit = edit; _swapTouch = nil; _holdGen++;
    fprintf(stderr, "[dm] 手柄编辑：%s\n", edit ? "开" : "关");
    if (!edit) [self saveOffsets];
    [self applyLayout]; [self setNeedsLayout];
}
- (void)holdFired {   // the layout button was held long enough
    if (!_edit) [self setEdit:YES];   // inside edit mode the long press does nothing more: the reset has its own button
}
- (void)armHold {
    int gen = ++_holdGen;
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(K_EDIT_HOLD * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
        if (gen != self->_holdGen || !self->_swapTouch) return;
        self->_swapTouch = nil;
        [self holdFired];
    });
}
- (void)askReset {
    UIViewController *vc = nil;
    for (UIScene *s in UIApplication.sharedApplication.connectedScenes) if ([s isKindOfClass:UIWindowScene.class])
        for (UIWindow *w in ((UIWindowScene *)s).windows) if (w.isKeyWindow) vc = w.rootViewController;
    while (vc.presentedViewController) vc = vc.presentedViewController;
    if (!vc) {   // no window to ask in: reset rather than leave the player with no way back to the default places
        fprintf(stderr, "[dm] 恢复默认位置：没有窗口，直接恢复\n");
        memset(_off, 0, sizeof _off); [self saveOffsets]; [self setNeedsLayout]; return;
    }
    UIAlertController *a = [UIAlertController alertControllerWithTitle:@"恢复默认位置 / Reset positions"
                                                              message:@"所有按键回到默认位置。\nEvery control goes back to its default place."
                                                       preferredStyle:UIAlertControllerStyleAlert];
    [a addAction:[UIAlertAction actionWithTitle:@"恢复 / Reset" style:UIAlertActionStyleDestructive handler:^(UIAlertAction *x) { (void)x;
        memset(self->_off, 0, sizeof self->_off); [self saveOffsets]; [self setNeedsLayout]; }]];
    [a addAction:[UIAlertAction actionWithTitle:@"取消 / Cancel" style:UIAlertActionStyleCancel handler:nil]];
    [vc presentViewController:a animated:YES completion:nil];
}

// One full-screen UIGlassContainerEffect, with one nested UIGlassEffect view per round / capsule key inside its contentView:
// the container renders them all into a single combined effect (UIGlassEffect.h), which is what keeps the cost to one pass
// instead of one blur per key. The views sit below every shape layer (insertSubview:atIndex:0) and take no touches, so the
// self-computed hit tests and the fill swap of a press are untouched. No parent of an effect view may have alpha < 1 or a mask
// (UIVisualEffectView.h), which this view does not have.
- (void)buildGlass {
    if (!_glass) return;
    UIGlassContainerEffect *ce = [UIGlassContainerEffect new];
    ce.spacing = 0;   // elements keep their own shape instead of merging into one blob
    _glassHost = [[UIVisualEffectView alloc] initWithEffect:ce];
    _glassHost.frame = self.bounds; _glassHost.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
    _glassHost.userInteractionEnabled = NO; _glassHost.contentView.userInteractionEnabled = NO;
    _glassHost.overrideUserInterfaceStyle = UIUserInterfaceStyleDark;
    [self insertSubview:_glassHost atIndex:0];
    for (int i = 0; i < G_N; i++) {
        UIVisualEffectView *v = [self glassElement];
        [_glassHost.contentView addSubview:v];
        _glassEl[i] = v;
        // The wash goes into the element's contentView, which the container draws ABOVE the glass: a press lightens the key
        // without touching the material. It is a shape layer so it takes the key's outline (circle, capsule), not its frame.
        CAShapeLayer *w = [CAShapeLayer layer];
        w.fillColor = k_glassOn.CGColor; w.hidden = YES;
        [v.contentView.layer addSublayer:w];
        _glassWash[i] = w;
    }
}
// One glass element: a nested effect view inside the container's contentView. It must take no touches (UIKit would otherwise
// hand them to it and the pad's own touchesBegan: would never run), and its parent must keep alpha 1 and no mask
// (UIVisualEffectView.h), which this view does. A square frame becomes a circle, the pill a capsule.
- (UIVisualEffectView *)glassElement {
    UIVisualEffectView *v = [[UIVisualEffectView alloc] initWithEffect:[UIGlassEffect effectWithStyle:UIGlassEffectStyleClear]];
    v.userInteractionEnabled = NO; v.contentView.userInteractionEnabled = NO;
    v.cornerConfiguration = [UICornerConfiguration capsuleConfiguration];
    return v;
}
// Backdrop frames follow the same geometry the shape paths use, in layoutSubviews, and take exactly the drawn size: the hit
// tests are larger (1.25 / 1.35 / 1.6), but a backdrop on the hit footprint would show as a dark halo around the drawn shape.
// The D-pad is four circles; a stick's glass is its throw-zone disc (the knob stays a plain shape on top).
- (void)layoutGlass {
    if (!_glassHost) return;
    CGFloat u = self.u;
    UICornerConfiguration *circle = [UICornerConfiguration capsuleConfiguration];   // square frame → circle, pill → capsule
    for (int i = 0; i < G_N; i++) {
        CGRect f = CGRectZero; UICornerConfiguration *cc = circle; BOOL shown = NO;
        if (i <= B_Y) {   // A / B / X / Y
            CGFloat r = self.faceR; CGPoint c = [self facePos:i];
            f = CGRectMake(c.x - r, c.y - r, 2 * r, 2 * r); shown = YES;
        } else if (i == B_ZL || i == B_L || i == B_ZR || i == B_R) {   // shoulders: a circle (was a teardrop)
            CGFloat r = [self dropR:i]; CGPoint c = [self dropCenter:i];
            f = CGRectMake(c.x - r, c.y - r, 2 * r, 2 * r); shown = YES;
        } else if (i == B_LS || i == B_RS) {
            CGFloat r = K_CLICK_R * u; CGPoint c = [self clickCenter:i];
            f = CGRectMake(c.x - r, c.y - r, 2 * r, 2 * r); shown = YES;
        } else if (i == B_PLUS || i == B_MINUS) {
            f = [self pillRect:i]; shown = YES;
        } else if (i >= B_LEFT && i <= B_DOWN) {   // D-pad: four round keys
            CGFloat r = [self dpadKeyR]; CGPoint c = [self dpadKeyPos:i];
            f = CGRectMake(c.x - r, c.y - r, 2 * r, 2 * r); shown = YES;
        } else if (i == G_SWAP) {
            CGFloat r = K_SWAP_R * u; CGPoint c = self.swapCenter;
            f = CGRectMake(c.x - r, c.y - r, 2 * r, 2 * r); shown = YES;
        } else {   // G_STICK0 / G_STICK1: the stick's throw zone, a glass disc under the knob
            int k = i == G_STICK1; CGFloat r = [self stickR:k]; CGPoint c = [self stickHome:k];
            f = CGRectMake(c.x - r, c.y - r, 2 * r, 2 * r); shown = YES;
        }
        if (!shown || !_glass) continue;
        _glassEl[i].frame = _glassFrame[i] = f; _glassEl[i].cornerConfiguration = cc;
        // The pressed wash has to follow the element's shape: a contentView does not clip to the corner configuration, so a plain
        // backgroundColor came out as a square over a round key. A shape layer of the same outline does.
        CGRect lb = CGRectMake(0, 0, f.size.width, f.size.height);
        CGPathRef wp = CGPathCreateWithRoundedRect(lb, MIN(lb.size.width, lb.size.height) / 2,
                                                   MIN(lb.size.width, lb.size.height) / 2, NULL);
        _glassWash[i].frame = lb; _glassWash[i].path = wp; CFRelease(wp);
    }
    _glassTintDirty = YES;
}
// Scrim: what the glass element samples is darkened by a same-shaped shape layer drawn under the container, so the key is
// readable while the picture around it is untouched. (A tint inside the element would sit above the glass — the container draws
// its elements behind contentView — and dim nothing.) Shape layers rather than a bitmap keep it in step with the shoulders
// moving into a bar, the 2D / 3D switch and a rotation, all of which end in layoutSubviews + applyLayout anyway.
- (void)updateGlassDim {
    CALayer *host = _glassDimLayer;
    if (!_glass) { host.hidden = YES; return; }
    if (!_glassTintDirty && host) return;
    _glassTintDirty = NO;
    if (!host) { host = [CALayer layer]; [self.layer insertSublayer:host below:_glassHost.layer]; _glassDimLayer = host; }
    host.hidden = NO; host.frame = self.bounds;
    for (int i = 0; i < G_N; i++) {
        CAShapeLayer *l = _glassDimShape[i];
        if (!_glassShown[i] || _glassFrame[i].size.width <= 0) { if (l) l.hidden = YES; continue; }
        if (!l) {
            l = [CAShapeLayer layer]; l.strokeColor = nil;
            [host addSublayer:l]; _glassDimShape[i] = l;
        }
        CGFloat tint = (i == G_STICK0 || i == G_STICK1) ? K_GLASS_TINT_STICK : K_GLASS_TINT;
        l.fillColor = [UIColor colorWithWhite:0 alpha:tint].CGColor;
        l.hidden = NO; l.frame = _glassFrame[i];
        // The scrim copies the element's own corner configuration; every one of them is a rounded rect, and a capsule about
        // both axes (its own minimum side) is the most it can be.
        CGRect b0 = CGRectMake(0, 0, _glassFrame[i].size.width, _glassFrame[i].size.height);
        CGFloat r = MIN(b0.size.width, b0.size.height) / 2;
        CGPathRef p = CGPathCreateWithRoundedRect(b0, r, r, NULL);
        l.path = p; CFRelease(p);
    }
}
- (void)applyLayout {   // show / hide the controls for the current layout
    [CATransaction begin]; [CATransaction setDisableActions:YES];
    // Edit mode ignores the layout: every key is shown (and outlined) so the pad is arranged once, for 2D and 3D alike. The keys
    // stay drawn as their plain translucent selves (refresh swaps the glass fills out) so it is clear what each dashed group holds.
    // The key layers must not be hidden here: a hidden layer hides its sublayers, and a key's letter lives inside it.
    for (int b = 0; b < B_N; b++) { _key[b].hidden = _edit ? NO : (fullOnly(b) && !_full); }   // every key, letters and all, so the player sees what they move
    _knob[1].hidden = _edit ? NO : !_full;
    _resetBtn.hidden = !_edit; _resetLabel.hidden = !_edit;
    // The stick discs are the glass-off stand-in for the glass stick face: glass on would draw the area twice, so they hide then.
    _ring[0].hidden = _edit || _glass; _ring[1].hidden = _edit || _glass || (!_full && !_edit);
    _glassHost.hidden = !_glass || _edit;
    for (int i = 0; i < G_N; i++) {
        BOOL hid = !_glass || _edit;   // in edit mode the outlines are what shows; the glass would fight the dashed boxes
        if (!_edit) {
            if (i < B_N) hid = hid || (fullOnly(i) && !_full);
            if (i == G_STICK1) hid = hid || !_full;   // the right stick exists only in 3D
        }
        _glassEl[i].hidden = hid; _glassShown[i] = !hid;
    }
    [CATransaction commit];
    _glassTintDirty = YES; [self updateGlassDim];
}
- (void)setFull:(BOOL)full {   // switch layouts: release every touch first (no key stays held), save, lay out again
    if (full == _full) return;
    [self reset];
    _full = full;
    [NSUserDefaults.standardUserDefaults setObject:@(full) forKey:k_layout_key];
    [self applyLayout]; [self setNeedsLayout];
}

// ---- Layout (points; computed inside the safe area) ----
- (CGRect)safe { return UIEdgeInsetsInsetRect(self.bounds, self.safeAreaInsets); }
- (CGRect)corner { UIEdgeInsets i = self.safeAreaInsets; i.left *= K_CORNER_IN; i.right *= K_CORNER_IN; return UIEdgeInsetsInsetRect(self.bounds, i); }   // the rectangle the bottom-corner controls use
- (CGFloat)u { return MIN(self.bounds.size.height / 430.0, 1.3); }   // 1 = 430 points tall (iPhone Pro Max landscape)
static CGPoint side(CGRect s, int right, CGFloat dx, CGFloat y) { return CGPointMake(right ? CGRectGetMaxX(s) - dx : CGRectGetMinX(s) + dx, y); }
- (CGFloat)bar { return _bar; }   // the picture's left bar (points); the right bar is the same width
// The drag offset: a whole group moves by one vector, so a key's offset-free place plus this vector is where it draws and hits. The two
// shoulder columns keep their Y (a column is a top-edge control and the picture's bar decides its X), so their offset is X only.
- (CGPoint)shift:(int)group {
    if (group == P_SHL || group == P_SHR) return CGPointMake(_off[group].x * self.u, 0);
    return CGPointMake(_off[group].x * self.u, _off[group].y * self.u);
}
static CGPoint shifted(CGPoint p, CGPoint d) { return CGPointMake(p.x + d.x, p.y + d.y); }
static CGRect shiftRect(CGRect r, CGPoint d) { return CGRectOffset(r, d.x, d.y); }
// The setter has to be a method: _bar lives in the @implementation's ivar block, which is private to the class, and the front end
// reaches the view through a C function (dm_gamepad_set_bars) that cannot touch it.
// Sub-point changes are ignored: the front end computes the bar in points before the first frame (82.67) and the present thread
// reports it in pixels rounded down (248 px / 3 = 82.67 -> the same value), so a difference smaller than a point is rounding noise,
// and reacting to it would re-lay out the pad once for nothing.
- (void)setBarPoints:(CGFloat)bar { if (fabs(_bar - bar) < 1.0) return; _bar = bar; [self setNeedsLayout]; }
- (CGPoint)faceCenter { CGRect s = self.corner; CGFloat u = self.u; return side(s, 1, K_FACE_X * u, CGRectGetMaxY(s) - K_PAD_Y * u); }   // the built-in place, offset-free
// Where the diamond is now: the built-in place plus the player's drag. The touch-down test has to use this one -- with the
// offset-free centre, a diamond the player moved was only grabbable around its old place, so the keys stopped responding.
- (CGPoint)faceGroupCenter { return shifted(self.faceCenter, [self shift:P_FACE]); }
- (CGFloat)faceGap { return K_FACE_GAP * self.u; }
- (CGFloat)faceR { return K_FACE_R * self.u; }
// Every getter below ends in "plus its group's drag offset", so the drawn path, the glass frame, the scrim and the hit test all move
// together: there is one place a control's position comes from, and the editor only changes that offset.
- (CGPoint)facePos:(int)b { CGPoint c = self.faceCenter; CGFloat g = self.faceGap;
    CGPoint p;
    switch (b) { case B_A: p = CGPointMake(c.x + g, c.y); break; case B_B: p = CGPointMake(c.x, c.y + g); break;
                 case B_X: p = CGPointMake(c.x, c.y - g); break; default: p = CGPointMake(c.x - g, c.y); }
    return shifted(p, [self shift:P_FACE]); }
- (CGPoint)stickHome:(int)i { CGRect s = self.corner; CGFloat u = self.u;
    CGPoint p = i ? side(s, 1, K_LOW_X * u, CGRectGetMaxY(s) - K_LOW_Y * u) : side(s, 0, K_STICK_X * u, CGRectGetMaxY(s) - K_PAD_Y * u);
    return shifted(p, [self shift:i ? P_STICK1 : P_STICK0]); }
- (CGFloat)stickR:(int)i { return (i ? K_RSTICK_R : K_STICK_R) * self.u; }
- (CGPoint)dpadCenter { CGRect s = self.corner; CGFloat u = self.u;
    return shifted(side(s, 0, K_LOW_X * u, CGRectGetMaxY(s) - K_LOW_Y * u), [self shift:P_DPAD]); }
- (CGFloat)dpadR { return K_DPAD_R * self.u; }
// The D-pad is four round keys around its centre (Switch style), each carrying a triangle pointing outward. The hit test stays
// angle based (dpadAt:, 8 directions), so sliding between two keys still presses both.
- (CGPoint)dpadKeyPos:(int)b { CGPoint c = self.dpadCenter; CGFloat d = K_DPAD_BTN_D * self.u;
    switch (b) { case B_UP: return CGPointMake(c.x, c.y - d); case B_DOWN: return CGPointMake(c.x, c.y + d);
                 case B_LEFT: return CGPointMake(c.x - d, c.y); default: return CGPointMake(c.x + d, c.y); } }
- (CGFloat)dpadKeyR { return K_DPAD_BTN_R * self.u; }
static BOOL isDrop(int b) { return b == B_ZL || b == B_L || b == B_ZR || b == B_R; }
static BOOL isClick(int b) { return b == B_LS || b == B_RS; }
// The shoulder column (ZL / L on the left, ZR / R on the right). It sits at K_ZL_X from the safe area's edge, which on a
// 19.5:9 phone puts it at 106 points -- over the picture when the picture starts at 124. When there is a bar (a 16:9 picture on a
// 19.5:9 panel) the column moves into it, centred in the bar; but only as far as the bar's middle, and never further out than it
// was: a bar narrower than the button keeps the button where it was, and no bar (the picture fills the frame) changes nothing.
- (CGFloat)shoulderX:(BOOL)right { CGRect s = self.safe; CGFloat u = self.u;
    return dm_pad_shoulder_x(CGRectGetMinX(s), CGRectGetMaxX(s), CGRectGetMaxX(self.bounds), self.bar,
                             K_ZL_X * u, K_ZL_R * u, right);
}
- (CGPoint)dropCenter:(int)b { CGRect s = self.safe; CGFloat u = self.u; BOOL z = b == B_ZL || b == B_ZR;
    CGPoint p = CGPointMake([self shoulderX:b == B_ZR || b == B_R], CGRectGetMinY(s) + (z ? K_ZL_Y : K_L_Y) * u);
    return shifted(p, [self shift:b == B_ZR || b == B_R ? P_SHR : P_SHL]); }
- (CGFloat)dropR:(int)b { return (b == B_ZL || b == B_ZR ? K_ZL_R : K_L_R) * self.u; }
// L3 / R3. Without a bar they sit at K_CLICK_X from the safe edge, as before. With a bar they keep a fixed distance INWARD of
// the shoulder column: centring them in the bar on their own would put them exactly on top of ZL / ZR (both columns are
// narrower than the bar, so the rule returns the same x for each), which made the two labels overlap.
- (CGPoint)clickCenter:(int)b { CGRect s = self.safe; CGFloat u = self.u; BOOL right = b == B_RS;
    CGFloat x;
    if (self.bar > 0) x = [self shoulderX:right] + (right ? -1 : 1) * (K_CLICK_X - K_ZL_X) * u;
    else x = dm_pad_shoulder_x(CGRectGetMinX(s), CGRectGetMaxX(s), CGRectGetMaxX(self.bounds), self.bar,
                               K_CLICK_X * u, K_CLICK_R * u, right);
    return shifted(CGPointMake(x, CGRectGetMinY(s) + K_ZL_Y * u), [self shift:right ? P_RS : P_LS]); }   // its own group: L3 stays put when the shoulder column moves
- (CGRect)pillRect:(int)b { CGRect s = self.safe; CGFloat u = self.u, w = K_PILL_W * u, h = K_PILL_H * u;
    CGPoint c = CGPointMake(CGRectGetMidX(s) + (b == B_PLUS ? K_PILL_DX : -K_PILL_DX) * u, CGRectGetMinY(s) + K_PILL_Y * u);   // − on the left, + on the right
    CGRect r = CGRectMake(c.x - w / 2, c.y - h / 2, w, h);
    return shiftRect(r, [self shift:b == B_PLUS ? P_PLUS : P_MINUS]); }
- (CGPoint)swapCenter { CGRect s = self.safe; return CGPointMake(CGRectGetMidX(s), CGRectGetMinY(s) + K_PILL_Y * self.u); }
// Edit mode's capsules: one rect rule, y = the capsule centre's distance from the safe area's top edge.
- (CGRect)editButtonRect:(CGFloat)y { CGRect s = self.safe; CGFloat u = self.u, w = K_EDIT_BTN_W * u, h = K_EDIT_BTN_H * u;
    CGPoint c = CGPointMake(CGRectGetMidX(s), CGRectGetMinY(s) + y * u);
    return CGRectMake(c.x - w / 2, c.y - h / 2, w, h); }
- (CGRect)resetRect { return [self editButtonRect:K_EDIT_BTN_Y2]; }
- (CGRect)swapRect { return [self editButtonRect:K_PILL_Y]; }   // 完成 takes the layout key's own place, as a capsule of the reset button's size
- (BOOL)inReset:(CGPoint)p { return _edit && CGRectContainsPoint(CGRectInset([self resetRect], -6 * self.u, -6 * self.u), p); }
- (BOOL)inSwap:(CGPoint)p {
    CGPoint c = self.swapCenter; CGFloat r = K_SWAP_R * self.u * 1.6, dx = p.x - c.x, dy = p.y - c.y;
    if (hypot(dx, dy) <= r) return YES;   // the round 2D / 3D key
    return _edit && CGRectContainsPoint(CGRectInset([self swapRect], -4 * self.u, -4 * self.u), p);   // or the 完成 capsule it becomes
}

static CGFloat dist(CGPoint a, CGPoint b) { return hypot(a.x - b.x, a.y - b.y); }
- (int)btnAt:(CGPoint)p {   // locked keys (tested on touch-down only, then locked)
    CGFloat u = self.u;
    for (int i = 0; i < (int)(sizeof k_btns / sizeof *k_btns); i++) { int b = k_btns[i];
        if (fullOnly(b) && !_full) continue;
        BOOL hit = isDrop(b) ? dist(p, [self dropCenter:b]) <= [self dropR:b] * 1.35
                 : isClick(b) ? dist(p, [self clickCenter:b]) <= K_CLICK_R * u * 1.35
                 : CGRectContainsPoint(CGRectInset([self pillRect:b], -8 * u, -8 * u), p);
        if (hit) return b; }
    return -1;
}
- (uint64_t)faceAt:(CGPoint)p {   // keys under the finger in the diamond (between two keys both count)
    uint64_t m = 0; CGFloat r = self.faceR * 1.25;
    int nearest = -1; CGFloat nd = CGFLOAT_MAX;
    for (int b = B_A; b <= B_Y; b++) {
        CGFloat d = dist(p, [self facePos:b]);
        if (d <= r) m |= 1ull << b;
        if (d < nd) { nd = d; nearest = b; }
    }
    // No key's circle reached the finger: it landed in one of the diamond's holes -- the 16 pt dead disc at the middle, or the
    // sliver between two adjacent circles (the circles' radii stop 1.6 pt short of the point where two neighbours meet). A press
    // there used to do nothing at all, which the thumb feels as "X sometimes does not work" (X is the far key when reaching up from
    // the bottom of the screen, so its presses drift toward the middle). Inside the diamond's outer reach, give the press to the
    // nearest key; its centre is further away only for a point that is already closer to another key.
    if (!m && nearest >= 0 && nd <= self.faceGap + self.faceR) m = 1ull << nearest;
    return m;
}
- (uint64_t)dpadAt:(CGPoint)p {   // 8 directions: 45° each, diagonals press two keys; a small centre circle counts as none
    CGPoint c = self.dpadCenter; CGFloat dx = p.x - c.x, dy = p.y - c.y;
    if (hypot(dx, dy) < self.dpadR * 0.25) return 0;
    double a = atan2(-dy, dx) * 180 / M_PI; uint64_t m = 0;
    if (a > -67.5 && a < 67.5) m |= 1ull << B_RIGHT;
    if (a > 22.5 && a < 157.5) m |= 1ull << B_UP;
    if (a > 112.5 || a < -112.5) m |= 1ull << B_LEFT;
    if (a < -22.5 && a > -157.5) m |= 1ull << B_DOWN;
    return m;
}
- (CGPoint)knobOffset:(int)i {   // knob offset from the ring centre (clamped to the ring)
    if (!_stick[i]) return CGPointZero;
    CGPoint h = [self stickHome:i]; CGFloat dx = _stickNow[i].x - h.x, dy = _stickNow[i].y - h.y, d = hypot(dx, dy), R = [self stickR:i];
    if (d > R) { dx *= R / d; dy *= R / d; }
    return CGPointMake(dx, dy);
}

// ---- Touch ----
- (void)recompute {
    uint64_t face = 0; for (UITouch *t in _faceOf) face |= [_faceOf objectForKey:t].unsignedLongLongValue;
    uint64_t btn = 0, dpad = 0; BOOL swap = NO;
    for (UITouch *t in _owner) { uint64_t v = [_owner objectForKey:t].unsignedLongLongValue; int c = (int)(v & 0xff);
        if (c == C_BTN) btn |= 1ull << (v >> 8);
        else if (c == C_DPAD) dpad |= [self dpadAt:[t locationInView:self]];
        else if (c == C_SWAP) swap = YES; }
    int32_t ax[4] = {0};
    for (int i = 0; i < 2; i++) {
        CGPoint o = [self knobOffset:i]; CGFloat R = [self stickR:i], d = hypot(o.x, o.y), dz = R * K_DEAD;
        if (d <= dz) continue;
        CGFloat m = fmin((d - dz) / (R * K_FULL - dz), 1) / d;   // outside the dead zone: 0 rising to full at the full-tilt radius
        ax[i * 2] = (int32_t)lrint(o.x * m * 32767); ax[i * 2 + 1] = (int32_t)lrint(-o.y * m * 32767);   // host: up is positive; rounded so full tilt is exactly 32767
    }
    // A key that was not down and now is: one light tap (sliding between keys taps again, holding one does not). The layout button
    // is not a game key, so it does not tap.
    uint64_t now = face | dpad | btn, fresh = now & ~(_faceBits | _dpadBits | _btnBits);
    if (fresh && !_edit) [self tapKey];
    BOOL changed = face != _faceBits || dpad != _dpadBits || btn != _btnBits || swap != _swapHeld;
    _faceBits = face; _dpadBits = dpad; _btnBits = btn; _swapHeld = swap;
    // In edit mode nothing reaches the game: a drag must not swing a sword, and a tap on a key must not fire it.
    pthread_mutex_lock(&g_mu);
    if (_edit) { g_touch_btn = 0; g_touch_lx = g_touch_ly = g_touch_rx = g_touch_ry = 0; }
    else { g_touch_btn = face | dpad | btn; g_touch_lx = ax[0]; g_touch_ly = ax[1]; g_touch_rx = ax[2]; g_touch_ry = ax[3]; }
    pthread_mutex_unlock(&g_mu);
    publish();
    BOOL sticks = _stick[0] || _stick[1];
    if (changed || sticks || _sticksShown) [self refresh];   // also refresh on the stick release so the knob returns to the centre
    _sticksShown = sticks;
}
- (void)touchesBegan:(NSSet<UITouch *> *)ts withEvent:(UIEvent *)e {
    for (UITouch *t in ts) {
        CGPoint p = [t locationInView:self];
        if (_edit) {   // edit mode owns every touch: the layout button works, a group starts a drag, anything else leaves the mode
            if ([self inReset:p]) { _resetTouch = t; continue; }
            if ([self inSwap:p]) { _swapTouch = t; _swapFrom = p; [self armHold]; }
            else { int g = [self editGroupAt:p]; if (g >= 0) { _drag = g; _dragTouch = t; _dragFrom = p; _dragBase = _off[g]; } }
            continue;
        }
        int hit = [self btnAt:p];
        if (hit >= 0) { [_owner setObject:@(C_BTN | (uint64_t)hit << 8) forKey:t]; continue; }
        if ([self inSwap:p]) { [_owner setObject:@(C_SWAP) forKey:t]; _swapTouch = t; _swapFrom = p; [self armHold]; continue; }
        // diamond, D-pad, the two sticks: take the one with the smallest distance / grab radius, if it is <= 1
        CGFloat best = 1; int kind = C_NONE, idx = 0;
        CGFloat f = dist(p, [self faceGroupCenter]) / (self.faceGap + self.faceR * 1.6); if (f <= best) { best = f; kind = C_FACE; }
        if (_full) { CGFloat d = dist(p, self.dpadCenter) / (self.dpadR * K_DPAD_GRAB); if (d < best) { best = d; kind = C_DPAD; } }
        for (int i = 0; i < (_full ? 2 : 1); i++) if (!_stick[i]) {
            CGFloat s = dist(p, [self stickHome:i]) / ([self stickR:i] * K_STICK_GRAB); if (s < best) { best = s; kind = C_STICK; idx = i; } }
        if (kind == C_FACE) { [_owner setObject:@(C_FACE) forKey:t]; [_faceOf setObject:@([self faceAt:p]) forKey:t]; }
        else if (kind == C_DPAD) [_owner setObject:@(C_DPAD) forKey:t];
        else if (kind == C_STICK) { [_owner setObject:@(C_STICK | (uint64_t)idx << 8) forKey:t]; _stick[idx] = t; _stickNow[idx] = p; }
    }
    [self recompute];
}
- (void)touchesMoved:(NSSet<UITouch *> *)ts withEvent:(UIEvent *)e {
    for (UITouch *t in ts) {
        CGPoint p = [t locationInView:self];
        if (t == _swapTouch && dist(p, _swapFrom) > K_EDIT_MOVE * self.u) { _swapTouch = nil; _holdGen++; }   // a slide, not a hold
        if (_edit) {
            if (t == _dragTouch) {   // move the dragged group by the finger's travel; clamp so its outline stays on screen
                CGPoint d = CGPointMake((p.x - _dragFrom.x) / self.u, (p.y - _dragFrom.y) / self.u);
                BOOL xOnly = _drag == P_SHL || _drag == P_SHR;   // a shoulder column keeps its top-edge height
                // Clamp the GROUP's own rect, taken once with its offset cleared -- not the rect at the current offset, which
                // already has this frame's movement in it and would feed back into the next one.
                CGPoint keep = _off[_drag]; _off[_drag] = CGPointZero;
                CGRect base = [self editRect:_drag];
                _off[_drag] = keep;
                CGFloat x = _dragBase.x + d.x, y = xOnly ? 0 : _dragBase.y + d.y;
                CGRect b = self.bounds, r = shiftRect(base, CGPointMake(x * self.u, y * self.u));
                if (CGRectGetMinX(r) < 0) x -= CGRectGetMinX(r) / self.u;
                if (CGRectGetMaxX(r) > b.size.width) x -= (CGRectGetMaxX(r) - b.size.width) / self.u;
                if (!xOnly) {
                    if (CGRectGetMinY(r) < 0) y -= CGRectGetMinY(r) / self.u;
                    if (CGRectGetMaxY(r) > b.size.height) y -= (CGRectGetMaxY(r) - b.size.height) / self.u;
                }
                _off[_drag] = CGPointMake(x, y);
                [self setNeedsLayout];
            }
            continue;
        }
        uint64_t v = [_owner objectForKey:t].unsignedLongLongValue; int c = (int)(v & 0xff);
        if (c == C_FACE) [_faceOf setObject:@([self faceAt:p]) forKey:t];
        else if (c == C_STICK) _stickNow[v >> 8] = p;
    }
    if (!_edit) [self recompute];
}
- (void)endTouches:(NSSet<UITouch *> *)ts tapped:(BOOL)ended {
    BOOL swap = NO, leave = NO;
    for (UITouch *t in ts) {
        if (_edit) {
            BOOL onSwap = [self inSwap:[t locationInView:self]];
            // The wait for the long press ends with the touch: a short tap on the layout button inside edit mode leaves the mode,
            // and so does a tap on empty space (a finished drag does not).
            if (t == _swapTouch) { _swapTouch = nil; _holdGen++; if (ended && onSwap) leave = YES; }
            if (t == _resetTouch) {
                _resetTouch = nil;
                if (ended && [self inReset:[t locationInView:self]]) { [self askReset]; continue; }   // the prompt, not a reset on the spot
                continue;
            }
            if (t == _dragTouch) { _dragTouch = nil; [self saveOffsets]; }
            else if (ended && !onSwap && [self editGroupAt:[t locationInView:self]] < 0) leave = YES;
            [_owner removeObjectForKey:t]; [_faceOf removeObjectForKey:t];
            continue;
        }
        BOOL onSwap = [self inSwap:[t locationInView:self]];
        if (t == _swapTouch) { _swapTouch = nil; _holdGen++; }   // the wait is over whether or not it was a hold
        if (ended && onSwap && ([_owner objectForKey:t].unsignedLongLongValue & 0xff) == C_SWAP) swap = YES;   // layout button: counts only when released on the button
        [_owner removeObjectForKey:t]; [_faceOf removeObjectForKey:t]; for (int i = 0; i < 2; i++) if (t == _stick[i]) _stick[i] = nil;
    }
    if (leave) { [self setEdit:NO]; return; }   // setEdit resets and republishes
    [self recompute];
    if (swap) [self setFull:!_full];
}
- (void)touchesEnded:(NSSet<UITouch *> *)ts withEvent:(UIEvent *)e { [self endTouches:ts tapped:YES]; }
- (void)touchesCancelled:(NSSet<UITouch *> *)ts withEvent:(UIEvent *)e { [self endTouches:ts tapped:NO]; }
- (void)reset { [_owner removeAllObjects]; [_faceOf removeAllObjects]; _stick[0] = _stick[1] = nil;
    _swapTouch = nil; _resetTouch = nil; _dragTouch = nil; _drag = -1; _holdGen++; [self recompute]; }

// ---- Shapes ----
static CGPathRef circlePath(CGPoint c, CGFloat r) { return CGPathCreateWithEllipseInRect(CGRectMake(c.x - r, c.y - r, 2 * r, 2 * r), NULL); }
static void setPath(CAShapeLayer *l, CGPathRef p) { l.path = p; l.shadowPath = p; CGPathRelease(p); }
static void placeLabel(CATextLayer *t, CGPoint c, CGFloat fs) {
    CTFontRef f = CTFontCreateUIFontForLanguage(kCTFontUIFontEmphasizedSystem, fs, NULL);
    t.font = f; t.fontSize = fs;
    CGFloat h = CTFontGetAscent(f) + CTFontGetDescent(f) + CTFontGetLeading(f), w = fs * 3;
    t.frame = CGRectMake(c.x - w / 2, c.y - h / 2, w, h);
    CFRelease(f);
}

- (void)layoutSubviews {
    [super layoutSubviews];
    CGFloat u = self.u;
    [self layoutGlass];   // frames only, no drawing: the glass views are plain UIViews
    [CATransaction begin]; [CATransaction setDisableActions:YES];
    // Every key keeps its letter or symbol in both modes; the layout button says 完成 while editing.
    static NSString *const names[B_N] = {@"A", @"B", @"X", @"Y", @"L3", @"R3", @"L", @"R", @"ZL", @"ZR", @"", @""};
    for (int b = 0; b < B_N; b++) if (_label[b]) _label[b].string = names[b];
    _swapLabel.string = _edit ? @"完成" : (_full ? @"3D" : @"2D");
    for (int b = B_A; b <= B_Y; b++) { CGPoint c = [self facePos:b]; setPath(_key[b], circlePath(c, self.faceR)); placeLabel(_label[b], c, 20 * u); }
    for (int i = 0; i < (int)(sizeof k_btns / sizeof *k_btns); i++) { int b = k_btns[i];
        if (isDrop(b)) {
            CGPoint c = [self dropCenter:b];
            setPath(_key[b], circlePath(c, [self dropR:b])); placeLabel(_label[b], c, 16 * u);   // a circle, like the face keys (the glass can only be a rounded rect)
        } else if (isClick(b)) {
            CGPoint c = [self clickCenter:b]; setPath(_key[b], circlePath(c, K_CLICK_R * u)); placeLabel(_label[b], c, 13 * u);
        } else {
            CGRect rc = [self pillRect:b];
            setPath(_key[b], CGPathCreateWithRoundedRect(rc, rc.size.height / 2, rc.size.height / 2, NULL));
            CGPoint pc = CGPointMake(CGRectGetMidX(rc), CGRectGetMidY(rc));
            if (_glyph[b]) placeGlyph(_glyph[b], pc, 18 * u); else placeLabel(_label[b], pc, 18 * u);
        }
    }
    // D-pad: four round keys (Switch style), the direction shown by a triangle pointing outward on each.
    { CGFloat r = [self dpadKeyR], glyph = 15 * u;   // the triangle sits well inside the 21u circle
      for (int i = 0; i < 4; i++) { int b = (int[]){B_UP, B_RIGHT, B_DOWN, B_LEFT}[i];
          CGPoint c = [self dpadKeyPos:b]; setPath(_key[b], circlePath(c, r)); placeGlyph(_glyph[b], c, glyph); } }
    // Layout and reset buttons: round 2D / 3D key normally; in edit mode two capsules of one size, 完成 over 重置.
    CGPoint sc = self.swapCenter;
    if (_edit) { CGRect sr = [self swapRect];
        setPath(_swap, CGPathCreateWithRoundedRect(sr, sr.size.height / 2, sr.size.height / 2, NULL));
        placeLabel(_swapLabel, CGPointMake(CGRectGetMidX(sr), CGRectGetMidY(sr)), 14 * u);
    } else { setPath(_swap, circlePath(sc, K_SWAP_R * u)); placeLabel(_swapLabel, sc, 13 * u); }
    {   CGRect rr = [self resetRect];
        setPath(_resetBtn, CGPathCreateWithRoundedRect(rr, rr.size.height / 2, rr.size.height / 2, NULL));
        placeLabel(_resetLabel, CGPointMake(CGRectGetMidX(rr), CGRectGetMidY(rr)), 14 * u);
    }
    for (int i = 0; i < 2; i++) {
        setPath(_ring[i], circlePath([self stickHome:i], [self stickR:i]));   // the grab circle's own path (glass layout draws its own disc)
        CGFloat k = (i ? K_RKNOB_R : K_KNOB_R) * u; _knob[i].bounds = CGRectMake(0, 0, 2 * k, 2 * k);
        setPath(_knob[i], circlePath(CGPointMake(k, k), k));
        CAShapeLayer *in = (CAShapeLayer *)_knob[i].sublayers.firstObject; CGPathRef ip = circlePath(CGPointMake(k, k), k * 0.62);
        in.frame = _knob[i].bounds; in.path = ip; CGPathRelease(ip);
    }
    [CATransaction commit];
    [self layoutEdit];   // dashed outlines, on screen only in edit mode
    [self refresh];
    [self updateGlassDim];   // after the frames are set; it no-ops unless the layout or the switch changed
}
- (void)safeAreaInsetsDidChange { [super safeAreaInsetsDidChange]; [self setNeedsLayout]; }
- (void)refresh {   // pressed state → fill, sticks → knob position
    uint64_t all = _faceBits | _dpadBits | _btnBits;
    [CATransaction begin]; [CATransaction setDisableActions:YES];
    for (int b = 0; b < B_N; b++) { BOOL on = (all >> b) & 1;
        UIColor *txt = _glass ? k_textGlass : (on ? k_textOn : k_text);
        if (_edit) {   // edit mode: the plain translucent key again (the glass is off), so every key is visible and can be told apart
            _key[b].fillColor = k_idle.CGColor; _key[b].strokeColor = k_line.CGColor; _key[b].shadowOpacity = 0;
            txt = k_text;
        } else if (_glass) {   // the glass element IS the button: no fill or ring under it, the label sits on the glass
            _key[b].fillColor = nil; _key[b].strokeColor = nil; _key[b].shadowOpacity = 0;
            _glassWash[b].hidden = !on;   // pressed: a light wash in the key's own outline, above the glass
        } else {
            _key[b].fillColor = (on ? k_on : k_idle).CGColor; _key[b].strokeColor = k_line.CGColor; _key[b].shadowOpacity = 0.2;
        }
        _label[b].foregroundColor = txt.CGColor;
        if (_glyph[b]) tintGlyph(_glyph[b], txt);
    }
    if (_edit) {   // edit mode: the pad is in its glass-off look, layout and reset buttons in the same one
        _swap.fillColor = k_idle.CGColor; _swap.strokeColor = k_line.CGColor; _swap.shadowOpacity = 0;
        _swapLabel.foregroundColor = k_text.CGColor;
        _glassWash[G_SWAP].hidden = YES;
        _resetBtn.fillColor = k_idle.CGColor; _resetBtn.strokeColor = k_line.CGColor; _resetBtn.shadowOpacity = 0;
        _resetLabel.foregroundColor = k_text.CGColor;
    } else if (_glass) {
        _swap.fillColor = nil; _swap.strokeColor = nil; _swap.shadowOpacity = 0;
        _swapLabel.foregroundColor = k_textGlass.CGColor;
        _glassWash[G_SWAP].hidden = !_swapHeld;
    } else {
        _swap.fillColor = (_swapHeld ? k_on : k_idle).CGColor; _swap.strokeColor = k_line.CGColor; _swap.shadowOpacity = 0.2;
        _swapLabel.foregroundColor = (_swapHeld ? k_textOn : k_text).CGColor;
    }
    for (int i = 0; i < 2; i++) {
        CGPoint h = [self stickHome:i], o = [self knobOffset:i];
        _knob[i].position = CGPointMake(h.x + o.x, h.y + o.y); _knob[i].fillColor = (_stick[i] ? k_on : k_idle).CGColor;
    }
    [CATransaction commit];
}

// ---- Physical controllers ----
static int32_t axis(float v) { return (int32_t)(fmaxf(-1.f, fminf(1.f, v)) * 32767); }
static void read_pad(GCExtendedGamepad *g) {
    uint64_t m = 0;
    #define K(btn, bit) if (btn.isPressed) m |= 1ull << bit
    K(g.buttonB, B_A); K(g.buttonA, B_B); K(g.buttonY, B_X); K(g.buttonX, B_Y);   // by position: right A, bottom B, top X, left Y
    K(g.leftShoulder, B_L); K(g.rightShoulder, B_R); K(g.leftTrigger, B_ZL); K(g.rightTrigger, B_ZR);
    K(g.buttonMenu, B_PLUS); if (g.buttonOptions) K(g.buttonOptions, B_MINUS);
    if (g.leftThumbstickButton) K(g.leftThumbstickButton, B_LS); if (g.rightThumbstickButton) K(g.rightThumbstickButton, B_RS);
    K(g.dpad.left, B_LEFT); K(g.dpad.up, B_UP); K(g.dpad.right, B_RIGHT); K(g.dpad.down, B_DOWN);
    #undef K
    pthread_mutex_lock(&g_mu);
    g_pad_btn = m; g_pad_lx = axis(g.leftThumbstick.xAxis.value); g_pad_ly = axis(g.leftThumbstick.yAxis.value);
    g_pad_rx = axis(g.rightThumbstick.xAxis.value); g_pad_ry = axis(g.rightThumbstick.yAxis.value);
    pthread_mutex_unlock(&g_mu);
    publish();
}
- (void)padsChanged {
    BOOL any = NO;
    for (GCController *c in GCController.controllers) if (c.extendedGamepad) {
        any = YES; c.extendedGamepad.valueChangedHandler = ^(GCExtendedGamepad *g, GCControllerElement *el) { (void)el; read_pad(g); };
        read_pad(c.extendedGamepad);
    }
    if (!any) { pthread_mutex_lock(&g_mu); g_pad_btn = 0; g_pad_lx = g_pad_ly = g_pad_rx = g_pad_ry = 0; pthread_mutex_unlock(&g_mu); publish(); }
    dispatch_async(dispatch_get_main_queue(), ^{ self.hidden = any; if (any) [self reset]; });
}
@end

UIView *dm_gamepad_view(CGRect frame) { DmGamepad *v = [[DmGamepad alloc] initWithFrame:frame]; g_view = v; return v; }
// The game picture's rectangle inside the output frame, in points: x = the left bar's width, w = the picture's width.
// The shoulder buttons move into the bar when there is one. Called on the main thread.
void dm_gamepad_set_bars(double x, double w) {
    DmGamepad *v = g_view; (void)w; if (!v) return;   // w is the picture's width; the pad only needs the bar
    [v setBarPoints:(CGFloat)x];
}
void dm_gamepad_reset(void) {   // On resign active: release everything (a key held while switching away and released in background gets no "release")
    dispatch_async(dispatch_get_main_queue(), ^{ [g_view reset]; });
    pthread_mutex_lock(&g_mu); g_pad_btn = 0; g_pad_lx = g_pad_ly = g_pad_rx = g_pad_ry = 0; pthread_mutex_unlock(&g_mu); publish();
}
