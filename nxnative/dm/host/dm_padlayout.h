// The on-screen pad's shoulder-column rule: where the ZL / L / ZR / R buttons go when the game picture leaves bars.
// Shared by the iOS pad (DmGamepad.m) and the host test (t_pad_bars.c), so the rule is testable on its own.
//
// The buttons sit at a fixed distance from the safe area's edge, which on a 19.5:9 phone puts them at 106 points -- over the
// picture when the picture starts at 124 (a 16:9 picture on that panel leaves a bar at either side). When there is a bar the
// column moves into it, centred; but never further out than it was, and never off the screen edge.
//
// All the lengths are in points. "right" selects the right-hand column.
#pragma once
#include <stdint.h>

// The x of the shoulder column's centre.
//   safe_min / safe_max: the safe area's horizontal bounds; bounds_max: the output frame's right edge
//   bar: the picture's bar on this side (0 = the picture fills the frame)
//   off: the column's distance from the safe area's edge (K_ZL_X for the shoulders, K_CLICK_X for L3 / R3)
//   r: the button's radius (the edge floor keeps it on the screen)
static inline double dm_pad_shoulder_x(double safe_min, double safe_max, double bounds_max, double bar,
                                        double off, double r, int right) {
    double base = right ? safe_max - off : safe_min + off;   // where the button was before there were bars
    if (bar <= 0) return base;                               // no bar: the picture fills the frame, nothing moves
    if (bar < 2 * r) return base;                            // the bar is too narrow to hold the button: it stays where it was
    // The bar's middle, clamped so the whole button stays inside the bar: the left edge of the button at r, the right at bar - r.
    // On the left the bar is at the frame's left edge, on the right at its right edge.
    double lo = right ? bounds_max - bar + r : r;
    double hi = right ? bounds_max - r : bar - r;
    double mid = right ? bounds_max - bar / 2 : bar / 2;
    if (mid < lo) mid = lo;
    if (mid > hi) mid = hi;
    return mid;
}
