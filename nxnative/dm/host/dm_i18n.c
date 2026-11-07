// String lookup for the UI text we added ourselves (English version: the language follows the original game's setting, and everything other than
// Simplified Chinese is English; item notes / legacy translated names are not shown in English).
// Single source: dm/i18n/ui_text.tsv (key / Chinese / English / where used); dm/ios/tools/dm_i18n.py does two things with it:
//   - Appends it to the original string table ui.json (RomFS overlay): zhCN column = Chinese, the other 12 columns = English. Layouts write "@key", and
//   the host's dm_str goes through the original string lookup at main+0x7f5630;
//     on a language switch the original game clears and reloads the string tables itself (SetLanguage 0x317298, no longer patched out as of r34), so we
//     do not have to follow it.
//   - Generates dm_i18n_tab.h (compiled into this file): places the game cannot draw (UIKit alerts, the asset download page, shared file names,
//   bottom-bar glyphs) pick by dm_lang_zh().
// Language: once the game has set it (SetLanguage observation gate), that value wins; before that (asset download page, before entering the front end)
// read "Language" from Settings/Settings.json in the save data;
//   no settings file = first install, use the iOS system language (dm_sys_lang). Our text: Chinese for Simplified and Traditional Chinese, English for
//   everything else.
#include "dm_i18n.h"
#include "dm_i18n_tab.h"
#include <TargetConditionals.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Atomic int g_dm_lang = -1;

#if TARGET_OS_IPHONE
#include <CoreFoundation/CoreFoundation.h>
#endif
// Language tag (BCP 47, e.g. en-US, zh-Hans-CN, zh-Hant-TW, yue-Hant-HK) → the original game's language index. Only two outcomes: a language code that
// is Chinese (zh) or a Chinese variety
// (yue Cantonese, wuu Wu, hak Hakka, nan Min Nan, cmn Mandarin, gan Gan, hsn Xiang, lzh Literary Chinese) → 8 Simplified Chinese, regardless of script
// and region, because we have no Traditional Chinese text of our own
// and the original game's Traditional Chinese plus our Simplified text would put two scripts on one screen; everything else → 0 English, because our own
// text exists only in Chinese and English, and nobody has verified the original game's other languages such as German / Japanese paired with our English
// text. Empty tag = English
int dm_lang_from_tag(const char *t) {
    if (!t || !*t) return 0;
    char l[4] = { 0 }; int i = 0; while (i < 3 && t[i] && t[i] != '-' && t[i] != '_') { l[i] = (char)(t[i] | 0x20); i++; }
    static const char *const zh[] = { "zh", "yue", "wuu", "hak", "nan", "cmn", "gan", "hsn", "lzh" };
    for (unsigned k = 0; k < sizeof zh / sizeof *zh; k++) if (!strcmp(l, zh[k])) return 8;
    return 0;
}
// iOS system language tag → the tag the game's own system-language table recognises (the one nn::oe::GetDesiredLanguage is matched against; table at
// main+0x6b2ed8 in build 8ce1ac48, 18 entries, verified by booting with DM_LANG and reading which Game/Text/<lang>/ the game opened).
// The match is exact on the tag *after* the game canonicalises it, and that canonicalisation keeps the region for everything but Chinese: a system tag
// of ja-JP or de-DE finds no entry (the table holds the bare ja / de), while zh-Hans-CN / zh-Hant-TW do resolve (canonicalised to the script form). So
// this maps to the table's own spelling instead of passing the system tag through.
// NULL = no entry: the game falls back to English. That covers English itself, plain pt and nl (the table deliberately sends those to en), and — worth
// knowing — pl / tr / uk / el, which ship 182 text files each but have no entry in the game's own table (the Switch system language list has no such
// languages either). Those four can only be reached by writing the "Language" key of GlobalSettingsNX64.sjson, not through DM_LANG.
const char *dm_lang_tag_from_ios(const char *t) {
    if (!t || !*t) return NULL;
    char l[4] = { 0 }; int i = 0; while (i < 3 && t[i] && t[i] != '-' && t[i] != '_') { l[i] = (char)(t[i] | 0x20); i++; }
    if (!strcmp(l, "ja")) return "ja";
    if (!strcmp(l, "ko")) return "ko";
    if (!strcmp(l, "de")) return "de";
    if (!strcmp(l, "fr")) return "fr";
    if (!strcmp(l, "it")) return "it";
    if (!strcmp(l, "es")) return "es";
    if (!strcmp(l, "ru")) return "ru";
    if (!strcmp(l, "pt")) return !strncmp(t, "pt-BR", 5) || !strncmp(t, "pt_BR", 5) ? "pt-BR" : NULL;   // only Brazilian Portuguese ships text
    if (!strcmp(l, "zh")) {
        // Script wins over region when both are present (zh-Hant-HK → Traditional), the region decides when only it is (zh-HK → Traditional, zh-SG → Simplified)
        const char *p = t + 2; while (*p == '-' || *p == '_') p++;
        int hant = !strncmp(p, "Hant", 4) || !strncmp(p, "hant", 4);
        int hans = !strncmp(p, "Hans", 4) || !strncmp(p, "hans", 4);
        if (!hant && !hans) { if (!strncmp(p, "TW", 2) || !strncmp(p, "tw", 2) || !strncmp(p, "HK", 2) || !strncmp(p, "hk", 2) || !strncmp(p, "MO", 2) || !strncmp(p, "mo", 2)) hant = 1; }
        return hant ? "zh-TW" : "zh-CN";
    }
    return NULL;   // en*, nl, pl, tr, uk, el and anything else: the game's table has no entry → English
}
// Pick from a preferred-language list (NSLocale.preferredLanguages, most preferred first). Walk it in order and take the first entry the game can show,
// so ["pl-PL", "de-DE"] lands on German instead of English — the list is a preference order, not a single answer.
// English stops the walk and yields NULL (leave DM_LANG unset, the game's own default is English): a player whose English is preferred over Chinese
// must not be dragged into Chinese by a later entry.
const char *dm_lang_pick(const char *const *t, int n) {
    for (int i = 0; t && i < n; i++) {
        const char *g = dm_lang_tag_from_ios(t[i]); if (g) return g;
        if (t[i]) { char l[4] = { 0 }; int k = 0; while (k < 3 && t[i][k] && t[i][k] != '-' && t[i][k] != '_') { l[k] = (char)(t[i][k] | 0x20); k++; }
            if (!strcmp(l, "en")) return NULL; }
    }
    return NULL;
}
// Language on first install (no settings file yet): by the first entry of the iOS preferred languages (CFLocaleCopyPreferredLanguages, through dm_lang_from_tag: Chinese and Sinitic varieties → Simplified Chinese, everything else → English).
// It used to be always Simplified Chinese (back then the touch version had no way to change the language, and a Chinese player whose system was in English got the English version on a fresh install and could not change it back); the Custom page title row now has a language switch (dm_custom.c lang_en), so this is opened up.
// DM_LANG (host / test simulation, same as hle_oe.c) takes priority; a non-iOS host has no system language to ask and uses English, the default game language
int dm_sys_lang(void) {
    const char *e = getenv("DM_LANG"); if (e && *e) return dm_lang_from_tag(e);
#if TARGET_OS_IPHONE
    int r = 0; CFArrayRef a = CFLocaleCopyPreferredLanguages();
    if (a) { if (CFArrayGetCount(a) > 0) { char t[64]; if (CFStringGetCString((CFStringRef)CFArrayGetValueAtIndex(a, 0), t, sizeof t, kCFStringEncodingUTF8)) r = dm_lang_from_tag(t); } CFRelease(a); }
    return r;
#else
    return 0;
#endif
}

static int settings_lang(void) {   // "Language" in the settings file; file missing = first install, use the system language (hle_fs.c writes the same value when it first creates the settings file); file present but key missing = -1
    const char *sv = getenv("DM_SAVE"); if (!sv) return dm_sys_lang();
    char path[1200]; snprintf(path, sizeof path, "%s/Settings/Settings.json", sv);
    FILE *f = fopen(path, "rb"); if (!f) return dm_sys_lang();
    char buf[8192]; size_t n = fread(buf, 1, sizeof buf - 1, f); fclose(f); buf[n] = 0;
    const char *k = strstr(buf, "\"Language\""); if (!k) return -1;
    k = strchr(k + 10, ':'); return k ? atoi(k + 1) : -1;
}

int dm_lang_index(void) {
    int l = atomic_load(&g_dm_lang); if (l >= 0) return l;
    static _Atomic int file = -2; int f = atomic_load(&file); if (f == -2) { f = settings_lang(); atomic_store(&file, f); }
    return f < 0 ? 0 : f;   // settings file exists but has no language entry = the game default (the host reports en-US)
}
int dm_lang_zh(void) { int l = dm_lang_index(); return l == 8 || l == 9; }   // for Traditional Chinese (zhTW) our text also uses Simplified Chinese (Traditional Chinese players read Simplified more easily than English)

const char *dm_ui(const char *key) {
    for (unsigned i = 0; i < sizeof k_dm_i18n / sizeof *k_dm_i18n; i++) if (!strcmp(k_dm_i18n[i].k, key)) return dm_lang_zh() ? k_dm_i18n[i].zh : k_dm_i18n[i].en;
    return key;
}

const char *dm_str(uint8_t *b, const char *key) {
    if (!b) return dm_ui(key);
    struct { const char *k; uint64_t n; } s = { key, strlen(key) };
    const char *r = ((const char *(*)(void *))(b + 0x7f5630))(&s);
    return r && *r && strcmp(r, key) ? r : dm_ui(key);   // if the overlay lacks this key (should not happen), fall back to the compile-time table
}
