// iPhone / iPad haptics sink for the Switch HD vibration stream.
//
// hle_haptics.c mixes the two controller motors into one intensity/sharpness pair and calls
// dm_haptics_platform_apply. This file owns the Core Haptics engines: one for the Taptic Engine and one
// for a connected physical controller (GameController exposes a haptics engine per controller). Both run
// on a serial queue; guest threads only enqueue a parameter set and never wait for haptic hardware.
//
// Adapted from Touchable's platform/ios/Haptics.m (MIT, Copyright (c) 2026 mzhao); see
// docs/research/touchable.md. The muted-start trick and the "controls live outside the looped pattern"
// rule come from there: without them a new 30 s loop re-applies its initial strength at every boundary
// and the first pulse lands as a full-strength tap.
#import <CoreHaptics/CoreHaptics.h>
#import <UIKit/UIKit.h>
#import <AVFAudio/AVFAudio.h>
#import <GameController/GameController.h>
#include "../../host/hle_haptics.h"
#include <stdarg.h>
#include <stdio.h>

@interface DmHaptics : NSObject
@property(nonatomic, readonly) BOOL supported;
+ (instancetype)shared;
- (void)apply:(dm_haptic_output)output;
@end

static DmHaptics *g_shared;

// How crisp the envelope is. The controls range from -1 (shortest) to 1 (longest), 0 is the
// event's own time. The game's rumbles are step shaped: they start at full strength and stop, so
// the rise and the fall should be as short as the Taptic allows.
static const float ENVELOPE_CRISP = -0.8f;

static void haptic_log(const char *format, ...) __attribute__((format(printf, 1, 2)));
static void haptic_log(const char *format, ...) {
    va_list args; va_start(args, format);
    fputs("[haptic] ", stderr); vfprintf(stderr, format, args); fputc('\n', stderr);
    va_end(args);
}

@implementation DmHaptics {
    dispatch_queue_t _queue;
    // Taptic Engine
    CHHapticEngine *_engine;
    id<CHHapticAdvancedPatternPlayer> _player;
    BOOL _engineRunning, _retryScheduled, _phoneSupported;
    NSUInteger _starts, _updates, _stops, _errors;
    uint64_t _pulseGen;   // silence watchdog: only the newest pulse may clear the player
    NSTimeInterval _t0;          // start of this session, for the timestamps in the log
    // Physical controller
    CHHapticEngine *_padEngine;
    id<CHHapticPatternPlayer> _padPlayer;
    __weak GCController *_padController;
    BOOL _padEngineRunning, _padRetryScheduled;
    NSUInteger _padStarts;
    NSArray<id<NSObject>> *_observers;
}

+ (instancetype)shared { return g_shared; }

- (instancetype)init {
    if ((self = [super init])) {
        _phoneSupported = CHHapticEngine.capabilitiesForHardware.supportsHaptics;
        _queue = dispatch_queue_create("dm haptics",
            dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0));
        NSMutableArray<id<NSObject>> *observers = [NSMutableArray array];
        NSNotificationCenter *center = NSNotificationCenter.defaultCenter;
        void (^suspend)(unsigned, BOOL) = ^(unsigned reason, BOOL on) { dm_haptics_suspend(reason, on ? 1 : 0); };
        [observers addObject:[center addObserverForName:UIApplicationWillResignActiveNotification object:nil queue:nil usingBlock:^(NSNotification *n) {
            (void)n; suspend(DM_HAPTICS_INACTIVE, YES);
        }]];
        [observers addObject:[center addObserverForName:UIApplicationDidEnterBackgroundNotification object:nil queue:nil usingBlock:^(NSNotification *n) {
            (void)n; suspend(DM_HAPTICS_INACTIVE, YES);
        }]];
        [observers addObject:[center addObserverForName:UIApplicationDidBecomeActiveNotification object:nil queue:nil usingBlock:^(NSNotification *n) {
            (void)n; suspend(DM_HAPTICS_INACTIVE, NO);
        }]];
        [observers addObject:[center addObserverForName:AVAudioSessionInterruptionNotification object:AVAudioSession.sharedInstance queue:nil usingBlock:^(NSNotification *n) {
            BOOL began = [n.userInfo[AVAudioSessionInterruptionTypeKey] unsignedIntegerValue] == AVAudioSessionInterruptionTypeBegan;
            suspend(DM_HAPTICS_INTERRUPTED, began);
        }]];
        for (NSString *name in @[GCControllerDidConnectNotification, GCControllerDidDisconnectNotification,
                                 GCControllerDidBecomeCurrentNotification, GCControllerDidStopBeingCurrentNotification]) {
            [observers addObject:[center addObserverForName:name object:nil queue:nil usingBlock:^(NSNotification *n) {
                (void)n;
                dispatch_async(self->_queue, ^{ [self refreshController]; });
            }]];
        }
        _observers = observers;
        _t0 = CACurrentMediaTime();
        [self publishSupport];
        dispatch_async(_queue, ^{ [self refreshController]; });
    }
    return self;
}

// The guest sees a vibration device only when something can actually play it. A controller that
// supports haptics counts even on an iPad whose own Taptic Engine is absent.
- (BOOL)controllerHapticsAvailable {
    GCController *current = GCController.current;
    if (current.extendedGamepad && current.haptics) return YES;
    for (GCController *controller in GCController.controllers)
        if (controller.extendedGamepad && controller.haptics) return YES;
    return NO;
}
- (void)publishSupport { dm_haptics_configure(self.supported ? 1 : 0); }
- (BOOL)supported { return _phoneSupported || [self controllerHapticsAvailable]; }

- (void)reportError:(NSError *)error operation:(const char *)operation {
    if (++_errors <= 8 || _errors % 64 == 0) haptic_log("%s failed: %s", operation, error.description.UTF8String);
}

- (NSArray<CHHapticDynamicParameter *> *)parameters:(dm_haptic_output)output {
    return @[
        [[CHHapticDynamicParameter alloc] initWithParameterID:CHHapticDynamicParameterIDHapticIntensityControl value:output.intensity relativeTime:0],
        /* The sharpness control ranges from -1 (dull) to 1 (bright) and adds to the event's own
         * sharpness, so the event sits at the neutral 0.5 and the control carries the offset. */
        [[CHHapticDynamicParameter alloc] initWithParameterID:CHHapticDynamicParameterIDHapticSharpnessControl value:output.sharpness - 0.5f relativeTime:0],
        /* The attack control shortens the rise of the event, and the release control shortens its
         * fall. The game's rumbles are step shaped: they start at full strength and stop dead, so
         * the Taptic should reach the level at once and let go at once instead of ramping into it
         * and trailing out of it. Both controls range from -1 (shortest) to 1 (longest), and 0 is
         * no change. */
        [[CHHapticDynamicParameter alloc] initWithParameterID:CHHapticDynamicParameterIDHapticAttackTimeControl value:ENVELOPE_CRISP relativeTime:0],
        [[CHHapticDynamicParameter alloc] initWithParameterID:CHHapticDynamicParameterIDHapticReleaseTimeControl value:ENVELOPE_CRISP relativeTime:0]
    ];
}

- (void)retryCurrentOutput {
    if (_retryScheduled) return;
    _retryScheduled = YES;
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_SEC), _queue, ^{
        self->_retryScheduled = NO;
        dm_haptic_output current = dm_haptics_output();
        if (self->_phoneSupported && current.intensity > 0) [self applyPhone:current];
    });
}

- (void)discardPlayer {
    if (!_player) return;
    NSError *error = nil;
    if (![_player stopAtTime:CHHapticTimeImmediate error:&error]) [self reportError:error operation:"stop"];
    _player = nil;
    if (++_stops <= 8) haptic_log("phone stop %lu", (unsigned long)_stops);
}

- (BOOL)startEngine {
    if (!_engine) {
        NSError *error = nil;
        _engine = [[CHHapticEngine alloc] initWithAudioSession:AVAudioSession.sharedInstance error:&error];
        if (!_engine) { [self reportError:error operation:"create engine"]; return NO; }
        _engine.playsHapticsOnly = YES;
        _engine.autoShutdownEnabled = NO;   // a game streams rumble in short pulses: keep the engine warm
        __weak DmHaptics *weakSelf = self;
        _engine.stoppedHandler = ^(CHHapticEngineStoppedReason reason) {
            DmHaptics *owner = weakSelf; if (!owner) return;
            dispatch_async(owner->_queue, ^{
                owner->_player = nil; owner->_engineRunning = NO;
                haptic_log("engine stopped reason=%ld", (long)reason);
                [owner retryCurrentOutput];
            });
        };
        _engine.resetHandler = ^{
            DmHaptics *owner = weakSelf; if (!owner) return;
            dispatch_async(owner->_queue, ^{
                owner->_player = nil; owner->_engineRunning = NO;
                haptic_log("engine reset");
                [owner retryCurrentOutput];
            });
        };
    }
    if (!_engineRunning) {
        NSError *error = nil;
        if (![_engine startAndReturnError:&error]) { [self reportError:error operation:"start engine"]; return NO; }
        _engineRunning = YES;
    }
    return YES;
}

- (void)applyPhone:(dm_haptic_output)output {
    dm_haptic_output current = dm_haptics_output();
    if (!_phoneSupported) return;
    // Epoch changes mean suspend / device reset: drop the running pulse. Values queued before the
    // change are not replayed afterwards.
    if (output.epoch != current.epoch) { [self discardPlayer]; return; }
    if (output.intensity == 0 && !_player) return;   // idle before the first pulse: nothing to do
    if (_retryScheduled && !_player) return;
    if (![self startEngine]) { if (output.intensity > 0) [self retryCurrentOutput]; return; }
    NSError *error = nil;
    NSArray<CHHapticDynamicParameter *> *parameters = [self parameters:output];
    if (!_player) {
        // One continuous event carries the whole rumble, and its level comes from the live
        // intensity control, so the same player covers any rumble.
        //
        // A transient for the onset was tried and dropped: the pattern is looped with a 30 s body,
        // so the loop length is 30 s and the transient would fire every 30 s instead of once per
        // rumble. Making it fire per rumble means rebuilding the pattern for every rumble, and
        // creating a player costs more than the rumble lasts.
        CHHapticEvent *body = [[CHHapticEvent alloc] initWithEventType:CHHapticEventTypeHapticContinuous parameters:@[
            [[CHHapticEventParameter alloc] initWithParameterID:CHHapticEventParameterIDHapticIntensity value:1],
            [[CHHapticEventParameter alloc] initWithParameterID:CHHapticEventParameterIDHapticSharpness value:0.5]
        ] relativeTime:0 duration:30];
        /* The controls stay outside the looped pattern, otherwise its initial strength returns at each loop. */
        CHHapticPattern *pattern = [[CHHapticPattern alloc] initWithEvents:@[body] parameters:@[] error:&error];
        if (!pattern) { [self reportError:error operation:"create pattern"]; [self retryCurrentOutput]; return; }
        _player = [_engine createAdvancedPlayerWithPattern:pattern error:&error];
        if (!_player) { [self reportError:error operation:"create player"]; [self retryCurrentOutput]; return; }
        _player.loopEnabled = YES;
        /* Start muted, set this pulse's controls, then unmute: no full-strength leading tap. */
        _player.isMuted = YES;
        if (![_player startAtTime:CHHapticTimeImmediate error:&error]) {
            [self reportError:error operation:"start player"]; _player = nil; _engineRunning = NO;
            [self retryCurrentOutput]; return;
        }
        if (![_player sendParameters:parameters atTime:CHHapticTimeImmediate error:&error]) {
            [self reportError:error operation:"initial controls"]; [self discardPlayer]; [self retryCurrentOutput]; return;
        }
        _player.isMuted = NO;
        if (++_starts <= 8) haptic_log("phone start %lu intensity=%.3f sharpness=%.3f", (unsigned long)_starts, output.intensity, output.sharpness);
    } else if (output.intensity == 0 && output.duration_ms < 120.0f) {
        // The host reports how long the rumble that just ended ran. A rumble shorter than 120 ms
        // (Hades sends some for 1-2 frames, 16-33 ms) is too short for the Taptic Engine to
        // ramp up, so hold it instead of muting.
        //
        // The hold ends by itself: a separate block mutes 120 ms after the rumble ended, so the
        // hold lasts the 120 ms floor and no more. A longer rumble ends right away, on the
        // game's own zero.
        [self armMuteAfter:0.12];
        return;
    } else if (![_player sendParameters:parameters atTime:CHHapticTimeImmediate error:&error]) {
        [self reportError:error operation:"update player"]; [self discardPlayer]; _engineRunning = NO;
        [self retryCurrentOutput]; return;
    }
    // Keep the player running between pulses: a game sends rumble in 1-3 frame bursts, and
    // creating/starting a player per burst is slower than the burst itself (the pulse was cut
    // before the Taptic Engine ramped). Silence is an intensity control of 0 instead.
    if (output.intensity > 0) [self armSilence];
    if (++_updates <= 8 || _updates % 1000 == 0)
        haptic_log("phone accepted %lu t=%.3f intensity=%.3f sharpness=%.3f dur=%.0fms",
                   (unsigned long)_updates, CACurrentMediaTime() - _t0, output.intensity, output.sharpness, output.duration_ms);
}

// If the game stops sending without a zero value, mute after a guard time. A newer pulse bumps
// _pulseGen, so only the newest scheduled block mutes.
//
// The guard has to be longer than the longest rumble the game sends, otherwise it cuts the rumble
// off in the middle. Hades II's measured rumbles run 295 ms, and the old 120 ms guard cut them
// short and then let them run on for another 120 ms, so every rumble was the wrong length. The
// game sends an explicit zero when a rumble ends, so the guard only has to cover the case where
// it stops without one.
- (void)armMuteAfter:(NSTimeInterval)delay {
    uint64_t gen = ++_pulseGen;
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(delay * NSEC_PER_SEC)), _queue, ^{
        if (gen != self->_pulseGen || !self->_player) return;
        NSError *error = nil;
        NSArray<CHHapticDynamicParameter *> *silence = @[
            [[CHHapticDynamicParameter alloc] initWithParameterID:CHHapticDynamicParameterIDHapticIntensityControl value:0 relativeTime:0]
        ];
        [self->_player sendParameters:silence atTime:CHHapticTimeImmediate error:&error];
    });
}

// If the game stops sending without a zero value, mute after a guard time. A newer pulse bumps
// _pulseGen, so only the newest scheduled block mutes.
//
// The guard has to be longer than the longest rumble the game sends, otherwise it cuts the rumble
// off in the middle. Hades II's measured rumbles run 295 ms, so the backstop is 500 ms. A rumble
// the game ends on its own stops on the game's own zero.
- (void)armSilence { [self armMuteAfter:0.5]; }

// ---- Physical controller (GameController hands out a Core Haptics engine per controller) ----

- (void)refreshController {
    GCController *current = GCController.current;
    if (!current.extendedGamepad || !current.haptics) {
        for (GCController *c in GCController.controllers)
            if (c.extendedGamepad && c.haptics) { current = c; break; }
    }
    if (current.extendedGamepad && current.haptics) { _padController = current; }
    else if (_padController && !_padController.haptics) { _padController = nil; }
    if (_padEngine) {
        if (_padEngineRunning) [_padEngine stopWithCompletionHandler:nil];
        _padEngine = nil; _padPlayer = nil; _padEngineRunning = NO;
    }
    [self publishSupport];
    if (_padController) haptic_log("controller haptics: %s", (_padController.vendorName ?: @"controller").UTF8String);
}

- (void)retryController {
    if (_padRetryScheduled) return;
    _padRetryScheduled = YES;
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_SEC), _queue, ^{
        self->_padRetryScheduled = NO;
        dm_haptic_output current = dm_haptics_output();
        if (current.intensity > 0) [self applyController:current];
    });
}

- (void)discardPadPlayer {
    if (!_padPlayer) return;
    NSError *error = nil;
    if (![_padPlayer stopAtTime:CHHapticTimeImmediate error:&error]) [self reportError:error operation:"stop controller player"];
    _padPlayer = nil;
}

- (BOOL)startPadEngine {
    if (!_padController) return NO;
    if (!_padEngine) {
        _padEngine = [_padController.haptics createEngineWithLocality:GCHapticsLocalityDefault];
        if (!_padEngine) { haptic_log("controller engine unavailable"); return NO; }
        _padEngine.autoShutdownEnabled = YES;
        __weak DmHaptics *weakSelf = self;
        _padEngine.stoppedHandler = ^(CHHapticEngineStoppedReason reason) {
            DmHaptics *owner = weakSelf; if (!owner) return;
            dispatch_async(owner->_queue, ^{
                owner->_padPlayer = nil; owner->_padEngineRunning = NO;
                haptic_log("controller engine stopped reason=%ld", (long)reason);
                [owner retryController];
            });
        };
        _padEngine.resetHandler = ^{
            DmHaptics *owner = weakSelf; if (!owner) return;
            dispatch_async(owner->_queue, ^{
                owner->_padPlayer = nil; owner->_padEngineRunning = NO;
                [owner retryController];
            });
        };
    }
    if (!_padEngineRunning) {
        NSError *error = nil;
        if (![_padEngine startAndReturnError:&error]) { [self reportError:error operation:"start controller engine"]; return NO; }
        _padEngineRunning = YES;
    }
    return YES;
}

- (void)applyController:(dm_haptic_output)output {
    dm_haptic_output current = dm_haptics_output();
    if (!_padController) { [self discardPadPlayer]; return; }
    if (output.intensity == 0) {
        // The same floor as the phone: a rumble shorter than 120 ms is too short for the
        // controller's motor to ramp up, so hold it. A longer rumble stops on the game's zero.
        if (output.duration_ms >= 120.0f || !_padPlayer) { [self discardPadPlayer]; return; }
        // The hold ends by itself after the 120 ms floor, so a short rumble cannot leave the
        // controller buzzing until the next one.
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.12 * NSEC_PER_SEC)), _queue, ^{
            [self discardPadPlayer];
        });
        return;
    }
    if (output.epoch != current.epoch) return;
    if (_padRetryScheduled && !_padPlayer) return;
    if (![self startPadEngine]) { [self retryController]; return; }
    NSError *error = nil;
    NSArray<CHHapticDynamicParameter *> *parameters = [self parameters:output];
    if (!_padPlayer) {
        CHHapticEvent *event = [[CHHapticEvent alloc] initWithEventType:CHHapticEventTypeHapticContinuous parameters:@[
            [[CHHapticEventParameter alloc] initWithParameterID:CHHapticEventParameterIDHapticIntensity value:0],
            [[CHHapticEventParameter alloc] initWithParameterID:CHHapticEventParameterIDHapticSharpness value:0.5]
        ] relativeTime:0 duration:3600];
        CHHapticPattern *pattern = [[CHHapticPattern alloc] initWithEvents:@[event] parameters:@[] error:&error];
        if (!pattern) { [self reportError:error operation:"create controller pattern"]; [self retryController]; return; }
        _padPlayer = [_padEngine createPlayerWithPattern:pattern error:&error];
        if (!_padPlayer) { [self reportError:error operation:"create controller player"]; [self retryController]; return; }
        if (![_padPlayer startAtTime:CHHapticTimeImmediate error:&error]) {
            [self reportError:error operation:"start controller player"]; _padPlayer = nil; _padEngineRunning = NO;
            [self retryController]; return;
        }
        if (++_padStarts <= 8) haptic_log("controller start %lu", (unsigned long)_padStarts);
    }
    if (![_padPlayer sendParameters:parameters atTime:CHHapticTimeImmediate error:&error]) {
        [self reportError:error operation:"update controller player"]; [self discardPadPlayer]; _padEngineRunning = NO;
        [self retryController];
    }
}

- (void)apply:(dm_haptic_output)output {
    dispatch_async(_queue, ^{ @autoreleasepool {
        [self applyPhone:output];
        [self applyController:output];
    } });
}

- (void)dealloc {
    for (id<NSObject> observer in _observers) [NSNotificationCenter.defaultCenter removeObserver:observer];
}
@end

static void dm_haptics_apply_c(dm_haptic_output output) { [g_shared apply:output]; }

// Called once from the iOS front end before the host starts: the guest must already see the vibration
// devices when it initializes nn::hid.
void dm_haptics_ios_init(void) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        if (CHHapticEngine.capabilitiesForHardware.supportsHaptics || GCController.current.haptics) {
            g_shared = [[DmHaptics alloc] init];
            dm_haptics_set_sink(dm_haptics_apply_c);
            haptic_log("available (Taptic Engine %s)", CHHapticEngine.capabilitiesForHardware.supportsHaptics ? "yes" : "no");
        } else {
            dm_haptics_configure(0);
            haptic_log("unavailable on this device");
        }
    });
}
