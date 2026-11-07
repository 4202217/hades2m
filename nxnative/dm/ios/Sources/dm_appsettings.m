// The app's Settings bundle (Settings.bundle/Root.plist) turned into the environment variables the host already reads.
//
// Settings → Apps → <the game's name> shows the controls; the values live in NSUserDefaults under the keys Root.plist names. This
// file maps each key to one variable and applies it with setenv(..., 0), so it never overrides a value that is already set:
// Documents/env.txt and the bundle's dm/env.txt are applied first (main.m) and win, which is what keeps a verification cycle
// from inheriting the player's choices (cycle.sh pushes a fresh env.txt every run).
//
// Every variable here is read once at startup by its consumer, so a change takes effect on the next launch — the same behaviour
// the Settings app itself has. Nothing here runs after boot; dm_appsettings_apply is called once from main.m's boot_env().
//
// Keys, all with a default that reproduces the behaviour from before this file existed (a player who never opens Settings gets
// exactly the old app):
//   DmResTier     → DM_RES_TIER        0-5, an exact 16:9 canvas (front.h front_res_tier_canvas; 4 = the front end's default)
//   DmFillScreen  → DM_FILL_SCREEN     the canvas takes the panel's own aspect instead of 16:9, so a 4:3 iPad shows no bars
//                                      (front_ios.m fill_screen_wanted; ignored on a panel wider than 16:9)
//   DmPadGlass    → DM_PAD_GLASS       already owned by DmGamepad.m, which remembers it in NSUserDefaults itself and treats the
//                                      variable as a one-off override; not applied here, or the player could never change it again
//   DmPadHaptic   → DM_PAD_HAPTIC      same owner and the same rule: the key-tap feedback is read by DmGamepad.m straight from
//                                      NSUserDefaults, so the Settings switch takes effect on the next launch without this file
//   DmHaptics     → DM_HAPTIC_OFF      inverted: the switch is "on", the variable is "off" (hle_haptics.c)
//   DmHapticGain  → DM_HAPTIC_GAIN     the output gain, 0-10 in whole steps (hle_haptics.c; 3 is the built-in default)
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Applies one variable unless it is already set. Returns 1 when it was applied, so the caller can log what really took effect
static int put(NSString *key, NSString *value, const char *name) {
    if (!key.length || !value.length) return 0;
    if (getenv(name)) {   // set by an env file: that wins, and the log says so rather than silently ignoring the switch
        fprintf(stderr, "boot: 设置 %s 未生效（%s 已由环境文件设为 %s）\n", key.UTF8String, name, getenv(name));
        return 0;
    }
    setenv(name, value.UTF8String, 1);
    fprintf(stderr, "boot: 设置 %s=%s → %s=%s\n", key.UTF8String, value.UTF8String, name, value.UTF8String);
    return 1;
}

static BOOL has(NSUserDefaults *d, NSString *key) { return [d objectForKey:key] != nil; }

void dm_appsettings_apply(void) {
    @autoreleasepool {
        NSUserDefaults *d = NSUserDefaults.standardUserDefaults;
        // Resolution tier. Out of range values are ignored (the front end would ignore them too, but saying so in the log helps
        // when a plist and this file drift apart).
        if (has(d, @"DmResTier")) {
            NSInteger t = [d integerForKey:@"DmResTier"];
            if (t >= 0 && t <= 5) put(@"DmResTier", [NSString stringWithFormat:@"%ld", (long)t], "DM_RES_TIER");
            else fprintf(stderr, "boot: 设置 DmResTier=%ld 超出 0–5，忽略\n", (long)t);
        }
        // Fill the screen: the canvas takes the panel's aspect (4:3 iPad, no bars) instead of the tier's 16:9. Only written when the
        // player has actually touched the switch: an untouched Settings switch has no stored value, and writing the "off" that
        // boolForKey reports would override the front end's own default (on for a panel narrower than 16:9, see fill_screen_wanted).
        if (has(d, @"DmFillScreen")) put(@"DmFillScreen", [d boolForKey:@"DmFillScreen"] ? @"1" : @"0", "DM_FILL_SCREEN");
        // Haptics on/off. The switch default is on, which is the old behaviour (DM_HAPTIC_OFF unset).
        BOOL on = has(d, @"DmHaptics") ? [d boolForKey:@"DmHaptics"] : YES;
        if (!on) put(@"DmHaptics", @"1", "DM_HAPTIC_OFF");
        // Strength. A switch off means no output at all, so the strength is irrelevant then and is not applied. The control is a
        // list of whole numbers (a slider showed no number); a value saved by the old slider is rounded once and written back, so
        // the list shows a selection instead of none.
        if (on && has(d, @"DmHapticGain")) {
            double g = [d doubleForKey:@"DmHapticGain"];
            long r = lround(g);
            if (r < 0) r = 0;
            if (r > 10) r = 10;
            if (g != (double)r) {
                [d setInteger:r forKey:@"DmHapticGain"];
                fprintf(stderr, "boot: 设置 DmHapticGain=%.2f 不是整数档，改为 %ld\n", g, r);
            }
            put(@"DmHapticGain", [NSString stringWithFormat:@"%ld", r], "DM_HAPTIC_GAIN");
        }
    }
}
