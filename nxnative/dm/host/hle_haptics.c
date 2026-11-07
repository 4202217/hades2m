#include "hle_haptics.h"
#include <math.h>
#include <stdatomic.h>
#include <pthread.h>
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

uint64_t dm_guest_frame(void);   // hle_os.c / test stubs; only used for diagnostics

static uint64_t trace_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

// The iOS front end registers its Core Haptics sink through dm_haptics_set_sink (Haptics.m). The Mac
// host has none, so the pointer stays null and the mixer runs without output (DM_HAPTIC_FAKE=1 logs it).
static void (*g_sink)(dm_haptic_output);

// Optional trace of every vibration value the game sends, for measuring the pulse shape (how long
// each rumble lasts, which amplitudes and frequencies it uses). Purely a diagnostic: it never
// changes the mixing or the output. DM_HAPTIC_TRACE=<path> writes a TSV:
//   t_ns  frame  device  amp_low  freq_low  amp_high  freq_high
// One line per SendVibrationValue, in the order the game sent them.
//
// The game calls this from guest threads at 200 Hz, so the guest side only copies the value into a
// ring buffer; a dedicated writer thread does the file I/O. Writing to the file from the guest
// thread crashed the host (fprintf is not safe there, and two guest threads race on the stream).
#define TRACE_RING 4096
typedef struct { uint64_t t_ns, frame; int dev; float al, fl, ah, fh; } trace_rec;
// DM_HAPTIC_TRACE_ALL=1 records every non-zero tick instead of only the changes. The changes
// alone hide a rumble that reuses the amplitude of the previous one.
static int g_trace_all;
static trace_rec g_ring[TRACE_RING];
static _Atomic uint64_t g_ring_head, g_ring_tail;
static pthread_t g_trace_thread;
static int g_trace_on;
static char g_trace_path[512];

static void trace_start(void);
static void rumble_multiplier_refresh(void);

static void *trace_writer(void *arg) {
    (void)arg;
    FILE *f = fopen(g_trace_path, "w");
    if (!f) { fprintf(stderr, "[dm] 震动探针：无法写入 %s\n", g_trace_path); return NULL; }
    fprintf(f, "# t_ns\tframe\tdevice\tamp_low\tfreq_low\tamp_high\tfreq_high\n");
    fprintf(stderr, "[dm] 震动探针：写入 %s\n", g_trace_path);
    uint64_t last = 0;   // the writer's own cursor into the ring
    for (;;) {
        uint64_t head = atomic_load(&g_ring_head);
        int wrote = 0;
        while (last != head) {
            trace_rec *r = &g_ring[last % TRACE_RING];
            fprintf(f, "%llu\t%llu\t%d\t%.6f\t%.1f\t%.6f\t%.1f\n",
                    (unsigned long long)r->t_ns, (unsigned long long)r->frame, r->dev,
                    r->al, r->fl, r->ah, r->fh);
            last++;
            wrote = 1;
        }
        if (wrote) fflush(f);
        atomic_store(&g_ring_tail, last);
        usleep(2000);
    }
    return NULL;
}

static void trace_start(void) {
    const char *path = getenv("DM_HAPTIC_TRACE");
    rumble_multiplier_refresh();
    const char *all = getenv("DM_HAPTIC_TRACE_ALL");
    g_trace_all = (all && *all && atoi(all) != 0);
    if (!path || !*path) return;
    snprintf(g_trace_path, sizeof g_trace_path, "%s", path);
    // pthread_once runs this on the calling thread, which is a guest thread: no I/O here. The
    // writer thread opens the file and reports.
    g_trace_on = 1;
    pthread_create(&g_trace_thread, NULL, trace_writer, NULL);
    pthread_detach(g_trace_thread);
}

static void trace_start(void);

// Called from the guest thread: copy the value into the ring and return. No I/O here.
static void trace_record(int dev, const dm_vib_value *v) {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, trace_start);
    if (!g_trace_on) return;
    // Only the values that matter for the envelope: a non-zero sample, or the zero that ends a
    // run of non-zero samples. The game ticks both devices at 200 Hz with zeros in between, and
    // recording all of it buries the signal (and floods the file).
    static _Atomic int prev_nonzero[2];
    int nz = (v->amplitude_low > 0 || v->amplitude_high > 0);
    int prev = atomic_exchange(&prev_nonzero[dev], nz);
    dm_vib_value zero = {0, 160, 0, 320};
    if (!nz && prev) {
        v = &zero;   // the zero that closes a burst: record it, so the end is in the file
    } else if (nz == prev && !g_trace_all) {
        // Only the value changes are interesting, unless the caller asked for every tick.
        return;
    }
    // Reserve the slot with a compare-and-swap, so two guest threads cannot write the same
    // record. The head and tail counters are 64 bit, so the difference stays correct past 2^32
    // sends.
    uint64_t head = atomic_load(&g_ring_head);
    for (;;) {
        uint64_t tail = atomic_load(&g_ring_tail);
        if (head - tail >= TRACE_RING) return;   // the writer fell behind: drop
        if (atomic_compare_exchange_weak(&g_ring_head, &head, head + 1)) break;
    }
    trace_rec *r = &g_ring[head % TRACE_RING];
    r->t_ns = trace_now_ns();
    r->frame = dm_guest_frame();
    r->dev = dev;
    r->al = v->amplitude_low; r->fl = v->frequency_low;
    r->ah = v->amplitude_high; r->fh = v->frequency_high;
}

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static const dm_vib_value k_idle = {0, 160, 0, 320};
static dm_vib_value g_value[2] = {{0, 160, 0, 320}, {0, 160, 0, 320}};
static dm_haptic_output g_output;
// Envelope tracking: the game streams a rumble as a run of values, and the shape of that run is
// what the player feels. The mixer remembers when the run started so the sink can draw the onset
// and the release instead of holding one flat level.
static uint64_t g_burst_start_ns;   // when the current run of non-zero values began
static uint64_t g_last_nonzero_ns;  // when it last carried energy
static int g_burst_active;          // a run of non-zero values is in progress
static dm_vib_value g_settled[2] = {{0, 160, 0, 320}, {0, 160, 0, 320}};  // the pair both motors agreed on
static unsigned g_settle_written;    // which motors have been written since the last settle
static uint64_t g_settle_start_ns;    // when the current round started
// How long the mixer waits for the other motor before it settles the pair anyway. The game
// writes a pair of motor values microseconds apart, so this only matters when a motor stops
// being written.
#define SETTLE_GRACE_NS 30000000ull   // 30 ms
static unsigned g_motor_used;        // which motors have carried energy in this run
static float g_burst_last_ms;       // how long the run that just ended lasted
static uint64_t g_epoch = 1, g_requests, g_nonzero, g_outputs, g_stops;
static int g_platform_supported, g_suspended;
static int g_env_fake = -1, g_env_off = -1;
// Output gain: Switch rumble amplitudes are subtle (0.06 for a normal hit at the in-game 100%),
// and the Taptic Engine feels much weaker than a Joy-Con at the same intensity.
//
// The in-game strength setting is applied by the game itself, before nn::hid is called: the game
// stores GeneralRumbleMultiplier in the profile (Profile1.sjson) and multiplies the amplitude it
// sends (measured: 0.030 at 50%, 0.060 at 100%, 0.119 at 200%). At 0% the game sends nothing at
// all, so the switch works without any host help.
//
// The strength therefore follows the game setting: the host multiplies the incoming amplitude by
// this gain and does NOT cancel the game's multiplier, so the in-game slider is what the player
// feels. With the default gain of 3 the output is 0.126 at 50%, 0.252 at 100% and 0.505 at 200%.
//
// DM_HAPTIC_GAIN overrides the gain (0 = the raw Touchable/d2r mapping, 1 = the amplitude as the
// game sent it).
#define DM_HAPTIC_GAIN_DEFAULT 3.0f
static float g_gain = -1;
static float gain(void) {
    if (g_gain < 0) {
        const char *e = getenv("DM_HAPTIC_GAIN");
        float v = (e && *e) ? (float)atof(e) : DM_HAPTIC_GAIN_DEFAULT;
        if (!isfinite(v)) v = 1;
        g_gain = fmaxf(0.0f, fminf(v, 10.0f));
    }
    return g_gain;
}

// The in-game strength setting, read from the save the game wrote: Profile<N>.sjson holds
// GeneralRumbleMultiplier (1.0 at 100%, 2.0 at 200%). The game multiplies the amplitude it sends by
// it, so the value that arrives at nn::hid is already scaled (measured 0.060 at 100%, 0.119 at 200%).
//
// The host reads it back only to report the setting in the log; the multiplier itself is already
// part of the amplitude the game sent, so the felt strength follows the in-game slider.
// The in-game strength setting, as the host reports it in the log.
//
// The save file is read by a background thread, never by the guest thread: the guest thread holds
// the mixer lock while it works, and doing file I/O there would block every other vibration call
// and could read a half-written file. (The save commit renames a finished tree into place, so a
// reader sees the old tree or the new one, never a mix.)
//
// The guest thread only ever reads the cached value, and only under the mixer lock, so there is
// no race and no path from the guest thread back into the file reading.
static float g_rumble_mult = 1.0f;   // the value the background thread published
static float g_rumble_mult_read = -1; // the file-reading path's own copy
static void rumble_multiplier_refresh(void);

static float rumble_multiplier(void) {
    // Only the background thread calls this, so it can use its own copy.
    if (g_rumble_mult_read > 0) return g_rumble_mult_read;
    g_rumble_mult_read = 1.0f;   // default until the save says otherwise
    const char *save = getenv("DM_SAVE");
    if (!save || !*save) return g_rumble_mult_read;
    // The active profile is named by save/activeProfile; the settings live next to it in Profile<N>.sjson.
    char path[1200];
    snprintf(path, sizeof path, "%s/activeProfile", save);
    FILE *f = fopen(path, "rb");
    char profile[160] = "";
    if (f) {
        // The file is "SGB1" + a 4 byte length + the profile name ("Profile1"), not plain text.
        char raw[128] = "";
        size_t n = fread(raw, 1, sizeof raw - 1, f);
        (void)n;
        fclose(f);
        const char *hit = strstr(raw, "Profile");
        if (hit) {
            size_t k = 0;
            while (hit[k] && hit[k] != '\r' && hit[k] != '\n' && hit[k] != ' ' && k + 1 < sizeof profile) { profile[k] = hit[k]; k++; }
            profile[k] = 0;
        }
    }
    if (!profile[0]) snprintf(profile, sizeof profile, "Profile1");
    snprintf(path, sizeof path, "%s/%s.sjson", save, profile);
    f = fopen(path, "rb");
    if (!f) {
        // No activeProfile (or it names something else): the first Profile*.sjson that carries the
        // setting is good enough.
        for (int i = 1; i <= 8; i++) {
            snprintf(path, sizeof path, "%s/Profile%d.sjson", save, i);
            f = fopen(path, "rb");
            if (f) break;
        }
    }
    if (!f) return g_rumble_mult_read;
    char buf[4096];
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = 0;
    const char *key = strstr(buf, "GeneralRumbleMultiplier");
    if (key) {
        const char *eq = strchr(key, '=');
        if (eq) {
            float v = (float)atof(eq + 1);
            if (isfinite(v) && v > 0.05f && v <= 10.0f) g_rumble_mult_read = v;
        }
    }
    return g_rumble_mult_read;
}

// The whole read, on a background thread.
static void *rumble_multiplier_thread(void *arg) {
    (void)arg;
    for (;;) {
        g_rumble_mult_read = 0.0f;   // force the next read to look at the file again
        float v = rumble_multiplier();
        pthread_mutex_lock(&g_mu);
        g_rumble_mult = v;            // published under the lock, read by the guest under it too
        pthread_mutex_unlock(&g_mu);
        usleep(1000000);
    }
    return NULL;
}

static void rumble_multiplier_refresh(void) {
    static pthread_t th;
    static int started;
    if (started) return;
    started = 1;
    if (pthread_create(&th, NULL, rumble_multiplier_thread, NULL) == 0) pthread_detach(th);
}

static int env_flag(const char *name, int *cache) {
    if (*cache < 0) { const char *e = getenv(name); *cache = !!(e && *e && strcmp(e, "0")); }
    return *cache;
}
static int fake(void) { return env_flag("DM_HAPTIC_FAKE", &g_env_fake); }
static int disabled(void) { return env_flag("DM_HAPTIC_OFF", &g_env_off); }

static int supported_locked(void) { return !disabled() && (g_platform_supported || fake()); }
static int allowed_locked(void) { return supported_locked() && !g_suspended; }

static float bounded(float v, float lo, float hi, float fallback) {
    return !isfinite(v) ? fallback : fmaxf(lo, fminf(v, hi));
}
static dm_vib_value clean(dm_vib_value v) {
    v.amplitude_low = bounded(v.amplitude_low, 0, 1, 0);
    v.amplitude_high = bounded(v.amplitude_high, 0, 1, 0);
    v.frequency_low = bounded(v.frequency_low, 40, 1280, 160);
    v.frequency_high = bounded(v.frequency_high, 40, 1280, 320);
    return v;
}

// Envelope tracking. The game streams a rumble as a run of values at 200 Hz; the run's length is
// what the player feels. The mixer measures the run (when it started, how long it lasted) and
// hands that number to the sink.
//
// A run is a stretch of time where the rumble carries energy. It ends when the game sends a zero
// value. The measured Hades II rumble is 295 ms long: the first motor runs at full strength for
// 215 ms, then the second one joins at a lower strength for the last 80 ms.
static uint64_t now_ns_mono(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

// Called with the mixer lock held. Returns the burst's age in ms, or 0 when no rumble is running.
static float burst_age_ms_locked(void) {
    if (!g_burst_active) return 0.0f;
    uint64_t now = now_ns_mono();
    if (now <= g_burst_start_ns) return 0.0f;
    return (float)(now - g_burst_start_ns) / 1e6f;
}

// Follows the rumble across the values the game streams. Called with the mixer lock held, once
// per value, from the guest thread.
//
// The state is: idle, or running. A run starts with the first value that carries energy and ends
// with the first value that does not. While it runs, the mixer remembers the loudest level and
// when the level first dropped below it, which is where the onset ends.
static void track_burst_locked(void) {
    // The level of the rumble is the loudest thing either motor is doing. Watching only the
    // motor whose value just arrived would restart the run every time the other motor went
    // quiet, and the run would never be longer than the quietest motor's share of it.
    float level = 0;
    for (int d = 0; d < 2; d++) {
        level = fmaxf(level, fmaxf(g_settled[d].amplitude_low, g_settled[d].amplitude_high));
    }
    uint64_t now = now_ns_mono();
    if (level > 0) {
        if (!g_burst_active) {
            g_burst_active = 1;
            g_burst_start_ns = now;
        }
        g_last_nonzero_ns = now;
    } else if (g_burst_active) {
        // The run ended. Remember how long it was: the zero that ends it carries this length, so
        // the sink knows whether to hold the rumble or let it stop.
        g_burst_last_ms = (float)(now - g_burst_start_ns) / 1e6f;
        g_burst_active = 0;
        g_motor_used = 0;
        }
}

// The burst's shape, as two numbers the sink can draw.
//
// A Hades II rumble (measured) runs like this: the first motor starts alone at full strength, and
// 215 ms later the second motor joins at a lower strength, which lowers the level for the rest of
// the run. The loud stretch at the front is the onset; the rest is the body.
//
// The onset ends when the level drops. The mixer remembers the loudest level of the run and when
// it first dropped below it, so the onset length is known without waiting for the run to end.
// The Switch gives two motors, each with a low band and a high band. An iPhone has one actuator
// with two axes: intensity (how hard) and sharpness (how bright). The mixer maps the two motors
// onto those axes instead of dropping one of them:
//
//   intensity  = the energy of both motors, so the rumble is as strong as the game asked
//   sharpness  = how much of that energy sits in the high band, so a bright rumble stays bright
//
// The old rule (Touchable's: take the stronger band per band, then sum) discarded the second
// motor entirely whenever the first one was louder, and its sharpness came from the frequency
// alone. Hades II always sends 160 / 320 Hz, so that sharpness was a constant 0.5 and carried no
// information at all.
static dm_haptic_output mixed_locked(void) {
    dm_haptic_output o = {.epoch = g_epoch};
    if (!allowed_locked()) return o;
    // The game writes the two motors as a pair, microseconds apart, with the same value. The
    // mixer samples a settled pair rather than the raw values, so a level that neither motor was
    // at can never be produced.
    //
    // The mixer therefore works on a settled pair: the values both motors had at the last moment
    // the game had written to both of them. The pair settles one write after a change, which is
    // a few microseconds here.
    float low_power = 0, high_power = 0;
    for (int d = 0; d < 2; d++) {
        float lp = g_settled[d].amplitude_low * g_settled[d].amplitude_low;
        float hp = g_settled[d].amplitude_high * g_settled[d].amplitude_high;
        // Both motors drive the same actuator, so they add up.
        low_power += lp;
        high_power += hp;
    }
    float power = low_power + high_power;
    if (power > 0) {
        // sqrt(power) is the raw intensity; the gain compensates the Taptic. The game's own
        // multiplier is deliberately NOT divided out: the in-game strength slider is what the
        // player feels. Measured with the default gain of 3: 0.126 at 50%, 0.252 at 100%,
        // 0.505 at 200%.
        o.intensity = fminf(1, sqrtf(power) * gain());
        // Sharpness is the share of the energy in the high band, mapped to 0..1. A rumble that is
        // all low band is dull (0.35), a rumble that is all high band is bright (0.9).
        float share = high_power / power;
        o.sharpness = 0.35f + 0.55f * share;
        // How long the rumble has been running so far.
        o.duration_ms = burst_age_ms_locked();
    } else {
        // The rumble is over. The length of the run that just ended rides along on this zero, so
        // the sink can tell a real rumble from a stray tick without another message.
        //
        // The length of the run that just ended rides along on this zero.
        o.duration_ms = g_burst_last_ms;
    }
    return o;
}

static void emit_locked(int force) {
    dm_haptic_output o = mixed_locked();
    // The run length is only good for the zero that reports it. The idle ticks that follow must
    // not carry it, or the sink would treat every idle tick as the end of another rumble and
    // hold the actuator on.
    if (o.intensity == 0) g_burst_last_ms = 0;
    if (!force && fabsf(o.intensity - g_output.intensity) < 0.00001f &&
        fabsf(o.sharpness - g_output.sharpness) < 0.00001f) return;
    if (g_output.intensity > 0 && o.intensity == 0) ++g_stops;
    ++g_outputs;
    g_output = o;
    if (g_sink) {
        g_sink(o);   // enqueue only: the platform never blocks the guest thread
    } else if (fake() && o.intensity > 0) {
        static int shown;
        if (shown++ < 8 || shown % 500 == 0)
            fprintf(stderr, "[dm] 震动（无接收端）#%d I=%.3f S=%.3f 增益%.1f（%.0fHz/%.0fHz）\n", shown, o.intensity,
                    o.sharpness, gain(), g_value[0].frequency_low, g_value[1].frequency_high);
    }
}

static void clear_locked(void) {
    g_value[0] = g_value[1] = k_idle;
    // The mixer reads the settled pair, so that is what has to be cleared. Leaving it behind
    // would fire a phantom rumble the moment the app became active again.
    g_settled[0] = g_settled[1] = k_idle;
    g_settle_written = 0;
    g_motor_used = 0;
    g_burst_active = 0;
    g_burst_start_ns = 0;
    g_burst_last_ms = 0;
    ++g_epoch;
    emit_locked(1);
}

void dm_haptics_set_sink(void (*sink)(dm_haptic_output)) {
    pthread_mutex_lock(&g_mu);
    g_sink = sink;
    clear_locked();
    pthread_mutex_unlock(&g_mu);
}

void dm_haptics_configure(int supported) {
    pthread_mutex_lock(&g_mu);
    g_platform_supported = supported != 0;
    clear_locked();
    pthread_mutex_unlock(&g_mu);
}

int dm_haptics_supported(void) {
    pthread_mutex_lock(&g_mu);
    int r = supported_locked();
    pthread_mutex_unlock(&g_mu);
    return r;
}

void dm_haptics_suspend(unsigned reason, int suspended) {
    pthread_mutex_lock(&g_mu);
    unsigned before = (unsigned)g_suspended;
    if (suspended) g_suspended |= (int)reason; else g_suspended &= ~(int)reason;
    if (before != (unsigned)g_suspended) clear_locked();
    pthread_mutex_unlock(&g_mu);
}

void dm_haptics_init_device(uint32_t handle) {
    pthread_mutex_lock(&g_mu);
    int dev = dm_haptics_device(handle);
    if (dev == 0 || dev == 1) {
        g_value[dev] = k_idle;
        g_settled[dev] = k_idle;
        emit_locked(0);
    }
    pthread_mutex_unlock(&g_mu);
}

void dm_haptics_send(uint32_t handle, const dm_vib_value *value) {
    if (!value) return;
    pthread_mutex_lock(&g_mu);
    int dev = dm_haptics_device(handle);
    if (dev != 0 && dev != 1) { pthread_mutex_unlock(&g_mu); return; }
    ++g_requests;
    dm_vib_value v = clean(*value);
    // Trace the raw value the game sent (before clamping) so the measurement shows what the game
    // actually asks for. The copy is cheap; the file write happens on the trace thread.
    trace_record(dev, value);
    if (v.amplitude_low > 0 || v.amplitude_high > 0) {
        if (++g_nonzero == 1 || g_nonzero % 3000 == 0)
            fprintf(stderr, "[dm] 震动值 #%llu 帧 %llu %.3f/%.0fHz %.3f/%.0fHz（增益 %.1f，游戏倍率 %.2f）\n",
                    (unsigned long long)g_nonzero, (unsigned long long)dm_guest_frame(),
                    v.amplitude_low, v.frequency_low, v.amplitude_high, v.frequency_high, gain(), g_rumble_mult);
    }
    // Do not replay background signals after the app becomes active again.
    g_value[dev] = allowed_locked() ? v : k_idle;
    // The mixer works on a settled pair rather than the raw values, so a level that neither motor
    // was at can never be produced.
    //
    // A round is one write to each motor that is in use, and the pair settles at the end of a
    // round, so the two values it holds were written for the same moment. A motor counts as
    // "in use" once it has carried energy in this run, so a game that drives only one motor does
    // not wait for the other one forever. A motor that is not written in a round is not running:
    // the game stopped driving it, so its last value must not keep contributing.
    //
    // A motor counts as "in use" once it has carried energy in this run, so a game that drives
    // only one motor does not wait for the other one forever.
    if (v.amplitude_low > 0 || v.amplitude_high > 0) g_motor_used |= 1u << dev;
    unsigned needed = g_motor_used ? g_motor_used : (1u << dev);
    g_settle_written |= 1u << dev;
    // A motor that stops being written must not freeze the pair: the rumble would keep the
    // level it had and never end. So the pair settles as soon as the motors in use have been
    // written, or after a short grace period, whichever comes first.
    //
    // The grace period is a few game ticks: long enough for the other motor to be written (the
    // game writes a pair microseconds apart), short enough that a stopped motor cannot hold the
    // rumble on.
    if ((g_settle_written & needed) == needed || now_ns_mono() - g_settle_start_ns > SETTLE_GRACE_NS) {
        // A motor that was not written in this round is not running: the game stopped driving
        // it, so its last value must not keep contributing to the level. Without this a motor
        // that stops being written holds the rumble on forever.
        g_settled[0] = (g_settle_written & 1u) ? g_value[0] : k_idle;
        g_settled[1] = (g_settle_written & 2u) ? g_value[1] : k_idle;
        g_settle_written = 0;
        g_settle_start_ns = now_ns_mono();
    }
    track_burst_locked();
    emit_locked(0);
    pthread_mutex_unlock(&g_mu);
}

void dm_haptics_send_many(const uint32_t *handles, const dm_vib_value *values, int n) {
    if (!handles || !values || n <= 0) return;
    for (int i = 0; i < n; i++) dm_haptics_send(handles[i], &values[i]);
}

dm_vib_value dm_haptics_actual(uint32_t handle) {
    pthread_mutex_lock(&g_mu);
    int dev = dm_haptics_device(handle);
    dm_vib_value r = (allowed_locked() && (dev == 0 || dev == 1)) ? g_value[dev] : k_idle;
    pthread_mutex_unlock(&g_mu);
    return r;
}

void dm_haptics_stop(void) {
    pthread_mutex_lock(&g_mu);
    clear_locked();
    pthread_mutex_unlock(&g_mu);
}

dm_haptic_output dm_haptics_output(void) {
    pthread_mutex_lock(&g_mu);
    dm_haptic_output r = g_output;
    pthread_mutex_unlock(&g_mu);
    return r;
}
