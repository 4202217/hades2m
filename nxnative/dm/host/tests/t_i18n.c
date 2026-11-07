// Language source of dm_i18n.c (for UIKit text): once the game has set the language (the SetLanguage observation gate writes g_dm_lang), that value
// wins; if not, read "Language" from Settings/Settings.json in the save data;
// no settings file = first install, use the system language (simulated in tests with DM_LANG). Our text: Chinese for zhCN and zhTW, English for
// everything else.
// Expected values come from the rule itself (8 zhCN / 9 zhTW are Chinese, everything else is English; for the language index table see the name table at
// main+0x20ae478 in diag.c), without running the game. Each case needs its own process (the file is read once and then cached), so this program runs one
// case per argv and the Makefile invokes them one by one.
#include "../dm_i18n.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
static int fails;
static void expect(const char *what, const char *got, const char *want) { int ok = !strcmp(got, want); if (!ok) fails++; printf("%s %s：期望「%s」→ 真结果「%s」\n", ok ? "✓" : "✗", what, want, got); }
int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "";
    char dir[512]; snprintf(dir, sizeof dir, "%s/i18n_%s", getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp", mode);
    char sd[600], f[700]; snprintf(sd, sizeof sd, "%s/Settings", dir); snprintf(f, sizeof f, "%s/Settings.json", sd);
    mkdir(dir, 0755); mkdir(sd, 0755); remove(f); setenv("DM_SAVE", dir, 1);
    if (!strcmp(mode, "en") || !strcmp(mode, "game")) { FILE *o = fopen(f, "w"); fputs("{\n    \"Item Name Display\": 1,\n    \"Language\": 0,\n    \"AutoMapMode\": 2\n}\n", o); fclose(o); }
    if (!strcmp(mode, "zh")) { FILE *o = fopen(f, "w"); fputs("{ \"Language\": 8 }", o); fclose(o); }
    if (!strcmp(mode, "en")) expect("设置文件 Language 0、游戏还没设", dm_ui("dmConfirm"), "OK");
    else if (!strcmp(mode, "zh")) expect("设置文件 Language 8", dm_ui("dmConfirm"), "确定");
    else if (!strcmp(mode, "none")) { unsetenv("DM_LANG"); expect("没有设置文件、没给系统语言（Mac 缺省英文）", dm_ui("dmConfirm"), "OK"); }
    else if (!strcmp(mode, "sysen")) { setenv("DM_LANG", "en-US", 1); expect("没有设置文件（首次安装）、系统语言 en-US", dm_ui("dmConfirm"), "OK"); }
    else if (!strcmp(mode, "systw")) { setenv("DM_LANG", "zh-Hant-TW", 1); expect("没有设置文件、系统语言 zh-Hant-TW（繁体用中文）", dm_ui("dmConfirm"), "确定"); }
    else if (!strcmp(mode, "tags")) {   // language tag → the original game's language index
        static const struct { const char *t; int want; } c[] = { { "en-US", 0 }, { "en-GB", 0 }, { "zh-Hans-CN", 8 }, { "zh-Hans", 8 }, { "zh", 8 }, { "zh-Hans-HK", 8 }, { "zh-Hans-TW", 8 }, { "zh-Hans-MO", 8 },
            { "zh-Hant-TW", 8 }, { "zh-Hant-HK", 8 }, { "zh-Hant", 8 }, { "zh-HK", 8 }, { "zh_TW", 8 }, { "yue-Hant-HK", 8 }, { "yue-Hans-CN", 8 }, { "yue", 8 }, { "wuu-Hans-CN", 8 },
            { "de-DE", 0 }, { "es-ES", 0 }, { "es-419", 0 }, { "fr-CA", 0 }, { "it-IT", 0 }, { "ko-KR", 0 }, { "pl-PL", 0 }, { "ru-RU", 0 }, { "ja-JP", 0 }, { "pt-BR", 0 }, { "nl-NL", 0 }, { "ar", 0 }, { "zu-ZA", 0 } };
        for (unsigned i = 0; i < sizeof c / sizeof *c; i++) { char w[8], g[8]; snprintf(w, sizeof w, "%d", c[i].want); snprintf(g, sizeof g, "%d", dm_lang_from_tag(c[i].t)); expect(c[i].t, g, w); } }
    else if (!strcmp(mode, "taggame")) {   // iOS system language tag → what to put in DM_LANG for the game's own table (main+0x6b2ed8, 18 entries)
        // The table holds the bare ja / ko / de / fr / it / es / ru and pt-BR, plus zh-CN / zh-TW / zh-Hans / zh-Hant
        static const struct { const char *in; const char *want; } c[] = {
            { "ja-JP", "ja" }, { "ko-KR", "ko" }, { "de-DE", "de" }, { "fr-FR", "fr" }, { "fr-CA", "fr" }, { "it-IT", "it" },
            { "es-ES", "es" }, { "es-419", "es" }, { "es-MX", "es" }, { "ru-RU", "ru" },
            { "zh-Hans-CN", "zh-CN" }, { "zh-Hans", "zh-CN" }, { "zh-Hans-SG", "zh-CN" }, { "zh-SG", "zh-CN" }, { "zh", "zh-CN" },
            { "zh-Hant-TW", "zh-TW" }, { "zh-Hant-HK", "zh-TW" }, { "zh-Hant", "zh-TW" }, { "zh-HK", "zh-TW" }, { "zh-TW", "zh-TW" }, { "zh-MO", "zh-TW" },
            { "zh_Hant_TW", "zh-TW" },   // iOS prefers hyphens, the underscore form is accepted too
            { "pt-BR", "pt-BR" }, { "pt-BR_x", "pt-BR" },
            // No entry in the game's table: the game falls back to English, and so do we (NULL = do not set DM_LANG at all)
            { "en-US", NULL }, { "en-GB", NULL }, { "pt-PT", NULL }, { "pt", NULL }, { "nl-NL", NULL },
            { "pl-PL", NULL }, { "tr-TR", NULL }, { "uk-UA", NULL }, { "el-GR", NULL },   // 182 text files ship for these four, but the game's own table has no entry
            { "ar-EG", NULL }, { "", NULL } };
        for (unsigned i = 0; i < sizeof c / sizeof *c; i++) {
            const char *g = dm_lang_tag_from_ios(c[i].in);
            char w[16]; snprintf(w, sizeof w, "%s", c[i].want ? c[i].want : "（不设）");
            expect(c[i].in, g ? g : "（不设）", w);
        } }
    else if (!strcmp(mode, "pick")) {   // the whole preferred-language list, in order: first entry the game can show; English stops the walk
        const struct { const char *const *in; int n; const char *want; } c[] = {
            { (const char *const[]){ "zh-Hans-CN", "en-US" }, 2, "zh-CN" },
            { (const char *const[]){ "pl-PL", "de-DE" }, 2, "de" },          // Polish ships text but the game's table has no entry: fall through to German
            { (const char *const[]){ "en-US", "zh-Hans-CN" }, 2, NULL },     // English preferred over Chinese: the game stays English
            { (const char *const[]){ "ja-JP" }, 1, "ja" },
            { (const char *const[]){ "en-GB" }, 1, NULL },
            { (const char *const[]){ }, 0, NULL },
        };
        for (unsigned i = 0; i < sizeof c / sizeof *c; i++) {
            const char *g = dm_lang_pick(c[i].in, c[i].n);
            char w[16], what[96] = { 0 }; for (int k = 0; k < c[i].n; k++) { strncat(what, c[i].in[k], sizeof what - strlen(what) - 2); if (k + 1 < c[i].n) strncat(what, ",", 1); }
            snprintf(w, sizeof w, "%s", c[i].want ? c[i].want : "（不设）");
            expect(what[0] ? what : "（空列表）", g ? g : "（不设）", w);
        } }
    else if (!strcmp(mode, "game")) {   // the settings file says English, then the game switches back to Simplified Chinese: the game wins; then switch to zhTW (9): everything other than Simplified Chinese is English
        expect("游戏没设前按设置文件", dm_ui("dmConfirm"), "OK");
        atomic_store(&g_dm_lang, 8); expect("游戏切到 zhCN", dm_ui("dmConfirm"), "确定");
        atomic_store(&g_dm_lang, 9); expect("游戏切到 zhTW（繁体用中文）", dm_ui("dmConfirm"), "确定");
        atomic_store(&g_dm_lang, 1); expect("游戏切到 deDE（按英文）", dm_ui("dmConfirm"), "OK");
        expect("表里没有的键原样返回", dm_ui("dmNoSuchKey"), "dmNoSuchKey"); }
    else { fprintf(stderr, "用法：t_i18n en|zh|none|sysen|systw|tags|taggame|pick|game\n"); return 2; }
    return fails ? 1 : 0;
}
