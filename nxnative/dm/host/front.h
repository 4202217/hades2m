// Host front end (Mac: front_mac.m; iOS: dm/ios separately implements the same-named interface).
#pragma once
// Run the guest: with a window the guest runs on its own thread and the main thread belongs to the system UI; without a window it runs directly on the current thread.
void front_run(void (*guest)(void));
// The CAMetalLayer the graphics backend presents frames to (NULL when there is no window)
void *front_metal_layer(void);
const char *front_documents_dir(void);   // the front end's Documents folder (the iOS app's; NULL elsewhere), for the diagnostics that write a file
// Game output resolution (the rewriter's PARAMS resolution parameter slot): docked = 1 docked mode / 0 handheld mode, returns (u64)height << 32 | width.
// The original game's constants are 1600x900 / 1280x720 (16:9); front ends with a different screen aspect ratio widen at the same height, and the game natively renders at the new width and re-lays out the UI (width ≤ 4096).
#include <stdint.h>
uint64_t front_game_res(int docked);
// The player's resolution tier (the iOS Settings bundle's "Render resolution", dm_appsettings.m): 0 = 1280x720, 1 = 1408x792,
// 2 = 1536x864, 3 = 1600x900, 4 = 1792x1008 (the iPhone default), 5 = 2240x1260, or -1 when unset / not a tier.
// Every tier is an exact 16:9 canvas on purpose: the game's HUD anchors to the canvas' top-left and is never shifted, so a wider
// canvas would put the left column under the Dynamic Island. A front end keeps 16:9 and scales the canvas down on a narrower panel.
// A tier names the canvas outright, so a Mac run and an iPhone run at the same tier render the same picture; -1 means each front
// end's own default (Mac 1280x720 handheld for its gate comparisons, iPhone 1792x1008). DM_GAME_RES overrides all of this.
#include <stdlib.h>
static inline int front_res_tier(void) {
    static int v = -2;
    if (v == -2) { const char *e = getenv("DM_RES_TIER"); int n = e && *e ? atoi(e) : -1; v = (n >= 0 && n <= 5) ? n : -1; }
    return v;
}
// The canvas a tier states, in pixels; (0, 0) when there is no tier.
static inline void front_res_tier_canvas(int tier, unsigned *w, unsigned *h) {
    static const unsigned short t[6][2] = {{1280, 720}, {1408, 792}, {1536, 864}, {1600, 900}, {1792, 1008}, {2240, 1260}};
    *w = (tier >= 0 && tier <= 5) ? t[tier][0] : 0;
    *h = (tier >= 0 && tier <= 5) ? t[tier][1] : 0;
}
// The game picture's rectangle inside the output frame, in PIXELS, origin at the top left: x = the width of the left bar,
// w = the picture's width. The graphics backend reports it after the aspect fit, so a front end can keep its on-screen controls
// out of the picture when the aspect leaves bars (a 16:9 picture on a 19.5:9 panel leaves a bar at either side).
// x = 0 means the picture fills the frame (a 19.5:9 picture on a 19.5:9 panel). Called from the present thread; the iOS front end
// hops to the main thread. A front end that does not implement it gets the weak default (no bars).
void front_set_picture_rect(double x, double w);
// Text input box (for the software keyboard swkbd): the main thread shows a native input box and the calling thread blocks until the user confirms / cancels.
// All text is UTF-8; max_chars is the maximum character count (0 = unlimited, only a hint for the front end; truncation is done uniformly by the caller hle_swkbd.c in UTF-16 units).
// Confirm returns 1 and writes the result into out (including the terminating 0); cancel returns 0;
// with no window (front_metal_layer() == NULL) nothing is shown and -1 is returned.
int front_text_input(const char *title, const char *sub, const char *guide, const char *initial, int max_chars, char *out, int out_cap);

// Early start of host loading (main.c): reading the NSO / relocation / forced takeover / prefetch start are moved to a background thread, in parallel with the front end (UIKit launch chain, Metal view);
// the dm_main / main called afterwards waits for it to finish. For call timing and preconditions see the comments in main.c (set DM_ROMFS and the other environment variables and the cwd first; arguments identical item by item to those of the later main).
void dm_host_preload_start(int argc, char **argv);

// Touchscreen (implemented by the host's hle_touch.c; called by the iOS / Mac front end): whenever the set of touch points changes (down / move / up / cancel) the whole set is pushed once; n = 0 = all lifted.
// id = finger slot number 0–15 (the lowest free slot on down, unchanged until up; used directly as finger_id); x / y = normalized coordinates [0,1) on the output frame, origin at top left.
// The host converts to the panel coordinates the game reads, 1280x720 (x·1280, y·720, the way main 0x1b65260 reads them), and latches short taps (a down and up between two reads is reported at least once). May be called from any thread.
typedef struct { uint32_t id; float x, y; } dm_touch;
void dm_touch_set(const dm_touch *t, int n);
// Tap-to-select (menus): the front end reports a tap at normalized coordinates (origin top-left, the
// dm_touch convention); the host navigates the game's own menu selection to the nearest button. The
// game has no mouse/touch source on this build (the PC keyboard/mouse handler never runs), so taps
// are fulfilled by closed-loop direction injection — see docs/research/touch-menus-feasibility-20261024.md.
void dm_tap_set(float nx, float ny);

// Save not written (called by the host's hle_fs.c; the iOS front end provides a strong implementation of the same name that shows the native prompt 「存储空间不足，进度未保存」(Not enough storage. Progress was not saved); the default weak implementation only logs).
// Timing: when a save write fails (disk full etc.) within one of the game's save transactions (between two Commits), so that this Commit is refused and the previously committed save is unchanged, it is called once per refused transaction;
// the next transaction that succeeds commits as usual, and the front end need not do any recovery. err = the first failing errno (ENOSPC = 28 means out of space). Called on the game thread; the front end switches to the main thread itself and must not block
void front_save_failed(int err);
