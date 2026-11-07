// The on-screen pad's shoulder-column rule (dm_padlayout.h): the ZL / L / ZR / R buttons move into the picture's bar when
// there is one, and stay where they were when there is not.
#include "../../host/dm_padlayout.h"
#include <stdio.h>
#include <math.h>

static int fails = 0;
static void check(const char *what, double got, double want, double eps) {
    if (fabs(got - want) > eps) { printf("  FAIL %s: got %.2f, want %.2f\n", what, got, want); fails++; }
}

// The iPhone Air in landscape: 1368 x 630 points, the side safe-area inset 62, the shoulder column 44 from the safe edge,
// the button radius 26. A 16:9 picture on that panel leaves a bar of 124 points.
#define SAFE_MIN 62.0
#define SAFE_MAX (1368.0 - 62.0)
#define BOUNDS_MAX 1368.0
#define ZL_OFF 44.0
#define ZL_R 26.0

int main(void) {
    // No bar: the buttons stay exactly where they were (106 points from the edge on the left, 1262 on the right).
    check("no bar, left", dm_pad_shoulder_x(SAFE_MIN, SAFE_MAX, BOUNDS_MAX, 0, ZL_OFF, ZL_R, 0), 106.0, 0.01);
    check("no bar, right", dm_pad_shoulder_x(SAFE_MIN, SAFE_MAX, BOUNDS_MAX, 0, ZL_OFF, ZL_R, 1), 1262.0, 0.01);

    // A 124-point bar: the button sits in the middle of the bar and the circle is entirely inside it.
    double l = dm_pad_shoulder_x(SAFE_MIN, SAFE_MAX, BOUNDS_MAX, 124, ZL_OFF, ZL_R, 0);
    check("a 124 bar, left", l, 62.0, 0.01);
    if (l - ZL_R < 0) { printf("  FAIL the button is off the screen's left edge\n"); fails++; }
    if (l + ZL_R > 124) { printf("  FAIL the button is over the picture (left)\n"); fails++; }
    double r = dm_pad_shoulder_x(SAFE_MIN, SAFE_MAX, BOUNDS_MAX, 124, ZL_OFF, ZL_R, 1);
    check("a 124 bar, right", r, 1306.0, 0.01);
    if (r + ZL_R > BOUNDS_MAX) { printf("  FAIL the button is off the screen's right edge\n"); fails++; }
    if (r - ZL_R < BOUNDS_MAX - 124) { printf("  FAIL the button is over the picture (right)\n"); fails++; }

    // Every bar width: the button stays on the screen, and it is in the bar whenever the bar can hold it.
    for (double bar = 0; bar <= 400; bar += 0.5) {
        double x = dm_pad_shoulder_x(SAFE_MIN, SAFE_MAX, BOUNDS_MAX, bar, ZL_OFF, ZL_R, 0);
        if (x - ZL_R < -0.01) { printf("  FAIL a %.1f bar: the left button is off the screen (x=%.1f)\n", bar, x); fails++; break; }
        if (x + ZL_R > bar + 0.01 && bar >= 2 * ZL_R) { printf("  FAIL a %.1f bar: the left button is over the picture (x=%.1f)\n", bar, x); fails++; break; }
        double y = dm_pad_shoulder_x(SAFE_MIN, SAFE_MAX, BOUNDS_MAX, bar, ZL_OFF, ZL_R, 1);
        if (y + ZL_R > BOUNDS_MAX + 0.01) { printf("  FAIL a %.1f bar: the right button is off the screen (x=%.1f)\n", bar, y); fails++; break; }
        if (y - ZL_R < BOUNDS_MAX - bar - 0.01 && bar >= 2 * ZL_R) { printf("  FAIL a %.1f bar: the right button is over the picture (x=%.1f)\n", bar, y); fails++; break; }
    }

    // A bar too narrow to hold the button leaves it exactly where it was.
    for (double bar = 0; bar < 2 * ZL_R; bar += 0.5) {
        double x = dm_pad_shoulder_x(SAFE_MIN, SAFE_MAX, BOUNDS_MAX, bar, ZL_OFF, ZL_R, 0);
        check("a narrow bar leaves the left button", x, 106.0, 0.01);
        double y = dm_pad_shoulder_x(SAFE_MIN, SAFE_MAX, BOUNDS_MAX, bar, ZL_OFF, ZL_R, 1);
        check("a narrow bar leaves the right button", y, 1262.0, 0.01);
    }

    // The L3 / R3 column (further in, 102 from the safe edge) follows the same rule.
    double l3 = dm_pad_shoulder_x(SAFE_MIN, SAFE_MAX, BOUNDS_MAX, 124, 102, 17, 0);
    check("a 124 bar, L3", l3, 62.0, 0.01);
    if (l3 - 17 < 0 || l3 + 17 > 124) { printf("  FAIL the L3 button is not inside the bar\n"); fails++; }

    printf("PAD_BARS cases=checked failed=%d\n", fails);
    return fails ? 1 : 0;
}
