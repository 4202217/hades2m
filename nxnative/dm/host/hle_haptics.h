// Switch HD vibration -> phone haptics, platform neutral host side.
//
// The guest talks to nn::hid. A game that supports rumble asks for vibration device handles,
// initializes them and then streams a VibrationValue per device every frame (or through the
// SDK's vibration graph). hle_hid.c forwards those calls here. This file keeps the two channels,
// sanitizes the values and mixes them into the single actuator an iPhone has: intensity from the
// combined band energy and sharpness from the amplitude weighted log frequency. The platform
// front end (dm/ios/Sources/Haptics.m) owns CoreHaptics and is reached through the weak hook
// dm_haptics_platform_apply, so the Mac host links without haptics and the iOS app gets them.
//
// The mixer is adapted from Touchable's ss_haptics.c (MIT, Copyright (c) 2026 mzhao), which is
// itself derived from d2r-mobile; see docs/research/touchable.md.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// nn::hid::VibrationValue is four floats, 16 bytes, device side channel order preserved.
typedef struct {
    float amplitude_low, frequency_low, amplitude_high, frequency_high;
} dm_vib_value;

// What the platform sink receives.
//
// intensity and sharpness are the two axes an iPhone actuator has: how hard, and how bright.
// The Switch gives two motors, each with a low band and a high band, so the mixer maps them onto
// those two axes instead of throwing one away: the energy of both motors sets the intensity, and
// the share of that energy in the high band sets the sharpness.
//
// duration_ms says how long the rumble runs. It rides on the zero that ends the rumble, so the
// sink can tell a real rumble from a stray tick and can hold the short ones to a floor.
//
// epoch advances whenever the queued values are cleared (suspend / device reset) so the sink can
// drop stale work.
typedef struct {
    float intensity, sharpness;
    float duration_ms;   // how long the rumble runs, in milliseconds
    uint64_t epoch;
} dm_haptic_output;

// Suspend reasons, ORed: the app is inactive or the audio session was interrupted.
enum { DM_HAPTICS_INACTIVE = 1u, DM_HAPTICS_INTERRUPTED = 2u };

// A handle is (npad id << 8) | device index, the layout Nintendo uses: 0 / 1 for the two motors
// of the first controller, 0x2000 / 0x2001 for the handheld controller.
static inline uint32_t dm_haptics_handle(int npad_id, int device) {
    return ((uint32_t)npad_id << 8) | (uint32_t)(device & 1);
}
static inline int dm_haptics_device(uint32_t handle) { return (int)(handle & 1u); }

// Platform: 1 when a sink exists (iOS CoreHaptics); 0 on the Mac host unless DM_HAPTIC_FAKE=1.
// DM_HAPTIC_OFF=1 disables haptics even when the platform supports them.
void dm_haptics_set_sink(void (*sink)(dm_haptic_output));   // Haptics.m registers the Core Haptics bridge
void dm_haptics_configure(int supported);
int dm_haptics_supported(void);
void dm_haptics_suspend(unsigned reason, int suspended);

// nn::hid service implementations (called from hle_hid.c)
void dm_haptics_init_device(uint32_t handle);
void dm_haptics_send(uint32_t handle, const dm_vib_value *value);
void dm_haptics_send_many(const uint32_t *handles, const dm_vib_value *values, int n);
dm_vib_value dm_haptics_actual(uint32_t handle);
void dm_haptics_stop(void);
dm_haptic_output dm_haptics_output(void);   // current mixed output, for replay after an engine restart

#ifdef __cplusplus
}
#endif
