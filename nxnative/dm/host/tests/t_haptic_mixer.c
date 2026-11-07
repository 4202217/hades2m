// Regression test for the haptic mixer rework, driven by the shapes the review raised.
#include "../hle_haptics.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>

// The mixer is the part of the haptics path that decides how hard the phone vibrates, and its
// rules are not obvious from the code: the two Switch motors add up, and the pair is sampled
// only once both have been written, so a level that no motor was ever at never reaches the phone.
//
// dm_guest_frame is a host symbol the mixer uses for the trace; this test has no guest.
uint64_t dm_guest_frame(void) { return 0; }

static int failures;
static void check(const char *name, int ok, const char *detail) {
    printf("%-52s %s %s\n", name, ok ? "PASS" : "FAIL", detail ? detail : "");
    if (!ok) ++failures;
}

static dm_vib_value vib(float al, float fl, float ah, float fh) {
    dm_vib_value v = {al, fl, ah, fh}; return v;
}
static void send(int dev, dm_vib_value v) {
    dm_haptics_send(dm_haptics_handle(0, dev), &v);
}
static int near(float a, float b, float eps) { return fabsf(a - b) < eps; }

int main(void) {
    setenv("DM_HAPTIC_FAKE", "1", 1);
    dm_haptics_configure(1);

    // --- 1. both motors: the level must be the sum, not the stronger one ---
    dm_haptics_send(dm_haptics_handle(0, 1), &(dm_vib_value){0.119f, 160, 0.119f, 320});
    dm_haptics_send(dm_haptics_handle(0, 0), &(dm_vib_value){0.0f, 160, 0.0f, 320});
    dm_haptics_send(dm_haptics_handle(0, 1), &(dm_vib_value){0.119f, 160, 0.119f, 320});
    float one_motor = dm_haptics_output().intensity;
    dm_haptics_send(dm_haptics_handle(0, 0), &(dm_vib_value){0.105f, 160, 0.105f, 320});
    dm_haptics_send(dm_haptics_handle(0, 1), &(dm_vib_value){0.105f, 160, 0.105f, 320});
    float both = dm_haptics_output().intensity;
    // sqrt(2*0.119^2)*3 = 0.5049, sqrt(2*(0.105^2+0.105^2))*3 = 0.6300
    check("one motor at 0.119 -> 0.5049", near(one_motor, 0.5049f, 0.002f), "");
    check("both motors at 0.105 -> 0.6300", near(both, 0.6300f, 0.002f), "");
    check("both motors beat the stronger-motor rule", both > one_motor, "");

    // --- 2. the game's own shape, replayed ---
    // The game writes the two motors as a pair, one microsecond apart, with the same value.
    // Phase one: only motor 1 is on, at 0.119. Phase two: both are on, at 0.105.
    dm_haptics_suspend(DM_HAPTICS_INACTIVE, 1);
    dm_haptics_suspend(DM_HAPTICS_INACTIVE, 0);
    float peak = 0;
    for (int i = 0; i < 10; i++) {
        dm_haptics_send(dm_haptics_handle(0, 0), &(dm_vib_value){0.0f, 160, 0.0f, 320});
        dm_haptics_send(dm_haptics_handle(0, 1), &(dm_vib_value){0.119f, 160, 0.119f, 320});
        float v = dm_haptics_output().intensity;
        if (v > peak) peak = v;
    }
    check("motor 1 alone at 0.119 -> 0.5049", near(dm_haptics_output().intensity, 0.5049f, 0.002f), "");
    for (int i = 0; i < 10; i++) {
        dm_haptics_send(dm_haptics_handle(0, 0), &(dm_vib_value){0.105f, 160, 0.105f, 320});
        dm_haptics_send(dm_haptics_handle(0, 1), &(dm_vib_value){0.105f, 160, 0.105f, 320});
        float v = dm_haptics_output().intensity;
        if (v > peak) peak = v;
    }
    check("both motors at 0.105 -> 0.6300", near(dm_haptics_output().intensity, 0.6300f, 0.002f), "");
    check("no phantom level above 0.6300", peak <= 0.6301f, "");

    // --- 3. a single-motor rumble still produces haptics ---
    dm_haptics_suspend(DM_HAPTICS_INACTIVE, 1);
    dm_haptics_suspend(DM_HAPTICS_INACTIVE, 0);
    for (int i = 0; i < 10; i++)
        dm_haptics_send(dm_haptics_handle(0, 1), &(dm_vib_value){0.119f, 160, 0.119f, 320});
    float single = dm_haptics_output().intensity;
    check("single-motor rumble is audible", single > 0.1f, "");

    // --- 3b. a motor that stops being written must not freeze the rumble ---
    // The game writes both motors as a pair, but it may skip one. If the mixer kept waiting for
    // the missing motor, the pair would freeze and the rumble would never end.
    dm_haptics_suspend(DM_HAPTICS_INACTIVE, 1);
    dm_haptics_suspend(DM_HAPTICS_INACTIVE, 0);
    for (int i = 0; i < 5; i++) {
        dm_haptics_send(dm_haptics_handle(0, 0), &(dm_vib_value){0.119f, 160, 0.119f, 320});
        dm_haptics_send(dm_haptics_handle(0, 1), &(dm_vib_value){0.119f, 160, 0.119f, 320});
    }
    float both_on = dm_haptics_output().intensity;
    check("both motors at 0.119 -> 0.7140", near(both_on, 0.7140f, 0.002f), "");
    // Motor 0 stops being written. Motor 1 goes to zero and stays there.
    for (int i = 0; i < 40; i++) {
        dm_haptics_send(dm_haptics_handle(0, 1), &(dm_vib_value){0.0f, 160, 0.0f, 320});
        usleep(2000);
    }
    float ended = dm_haptics_output().intensity;
    check("a stopped motor does not freeze the rumble", ended == 0.0f, "");

    // --- 4. a suspend must not leave a rumble behind ---
    dm_haptics_suspend(DM_HAPTICS_INACTIVE, 1);
    float suspended = dm_haptics_output().intensity;
    dm_haptics_suspend(DM_HAPTICS_INACTIVE, 0);
    float resumed = dm_haptics_output().intensity;
    check("suspend silences the rumble", suspended == 0.0f, "");
    check("resume does not fire a phantom rumble", resumed == 0.0f, "");

    printf("\n%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
