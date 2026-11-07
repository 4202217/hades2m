// The DM_NVN_HILITE diagnostic setup in nvn_mtl.m's pso_desc: the keys and the Metal library must be built exactly once, even when several threads build pipelines at the same time.
// The old lazy setup raced: the "has it been built yet" check and the build were interleaved with the key-matching loop, so two threads could both build the library and one replaced the library the other was using (Hades II, 6 keys) — the second thread then used a replaced library (objc_msgSend crash in pso_desc).
// This test mirrors that setup block (nvn_mtl.m, the "Diagnostic DM_NVN_HILITE" comment) and drives it from many threads at once. It checks two things:
//   1. the production source uses the once-only form (the fix is really in the file that ships);
//   2. the setup builds the library exactly once and every thread sees the same library (the property the fix buys).
// The build is deliberately slowed so that the old form is caught deterministically instead of rarely.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <pthread.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import <unistd.h>

static const char *HILITE_SRC =
    "#include <metal_stdlib>\nusing namespace metal;\nstruct O { float4 c0 [[color(0)]]; float4 c1 [[color(1)]]; float4 c2 [[color(2)]]; };\n"
    "fragment O hl0() { return {float4(1,0,1,1),float4(1,0,1,1),float4(1,0,1,1)}; } fragment O hl1() { return {float4(0,1,0,1),float4(0,1,0,1),float4(0,1,0,1)}; }\n"
    "fragment O hl2() { return {float4(0,1,1,1),float4(0,1,1,1),float4(0,1,1,1)}; } fragment O hl3() { return {float4(1,1,0,1),float4(1,1,0,1),float4(1,1,0,1)}; }\n"
    "fragment O hl4() { return {float4(0,0,1,1),float4(0,0,1,1),float4(0,0,1,1)}; } fragment O hl5() { return {float4(1,1,1,1),float4(1,1,1,1),float4(1,1,1,1)}; }\n";

enum { NTHREAD = 16 };

static id<MTLDevice> g_dev;
static char g_keys[6][24]; static int g_nk;
static id<MTLLibrary> g_hl;
static int g_builds;                                  // how many times the library was built
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static int g_building;                                // threads currently inside the build

// The build step, slowed so that a race is observable rather than rare: the thread waits until the other
// threads have also arrived (or until the wait times out). Under the once-only form only one thread is
// ever inside, so it waits out the timeout; under the lazy form every thread that sees "not built yet" enters.
static id<MTLLibrary> build_library_slowly(void) {
    pthread_mutex_lock(&g_mu); g_building++; pthread_mutex_unlock(&g_mu);
    for (int i = 0; i < 200; i++) {                 // up to ~200 ms: wait for the other threads
        pthread_mutex_lock(&g_mu); int b = g_building; pthread_mutex_unlock(&g_mu);
        if (b >= NTHREAD) break;
        usleep(1000);
    }
    NSError *err = nil;
    id<MTLLibrary> hl = [g_dev newLibraryWithSource:[NSString stringWithUTF8String:HILITE_SRC] options:nil error:&err];
    pthread_mutex_lock(&g_mu); g_builds++; g_building--; pthread_mutex_unlock(&g_mu);
    if (!hl) fprintf(stderr, "FAIL: the HILITE library did not compile: %s\n", err.localizedDescription.UTF8String);
    return hl;
}

// The setup under test: the once-only form (the fix).
static void setup_once(void) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        const char *e = getenv("DM_NVN_HILITE");
        for (const char *q = e; q && *q && g_nk < 6; q = strchr(q, ',') ? strchr(q, ',') + 1 : q + strlen(q)) {
            size_t n = strcspn(q, ","); if (n > 23) n = 23; memcpy(g_keys[g_nk], q, n); g_keys[g_nk++][n] = 0;
        }
        if (!g_nk) return;
        g_hl = build_library_slowly();
    });
}

// The same setup in the old lazy form (upstream before the fix), kept for the A/B: the check and the build are
// interleaved with the key-matching loop, so nothing serialises two threads that both see "not built yet".
static void setup_lazy(void) {
    static int nk = -1; static id<MTLLibrary> hl;
    if (nk < 0) { nk = 0; const char *e = getenv("DM_NVN_HILITE");
        for (const char *q = e; q && *q && nk < 6; q = strchr(q, ',') ? strchr(q, ',') + 1 : q + strlen(q)) { size_t n = strcspn(q, ","); if (n > 23) n = 23; memcpy(g_keys[nk], q, n); g_keys[nk++][n] = 0; } }
    for (int i = 0; i < nk; i++) if (!strncmp("deadbeefdeadbeef", g_keys[i], strlen(g_keys[i]))) {
        if (!hl) hl = build_library_slowly();
        break; }
    g_hl = hl;
}

static int g_lazy;                                    // which setup to use

// The thread body: what pso_desc does on every pipeline build — set up, then look the key up and take a function from the library.
static void *worker(void *arg) {
    (void)arg;
    for (int i = 0; i < 5; i++) {
        if (g_lazy) setup_lazy(); else setup_once();
        if (!g_hl) continue;
        for (int k = 0; k < g_nk; k++) if (!strncmp("deadbeefdeadbeef", g_keys[k], strlen(g_keys[k]))) {
            id<MTLFunction> f = [g_hl newFunctionWithName:[NSString stringWithFormat:@"hl%d", k]];
            if (!f) { fprintf(stderr, "FAIL: the function hl%d is missing from the library\n", k); exit(1); }
            break;
        }
    }
    return NULL;
}

// The production source must use the once-only form: the test would otherwise pass while the shipped code still races.
static int production_uses_once(void) {
    FILE *f = fopen("../nvn/nvn_mtl.m", "r");
    if (!f) { fprintf(stderr, "FAIL: cannot open ../nvn/nvn_mtl.m\n"); return 0; }
    char line[4096]; int in_hilite = 0, saw_once = 0, saw_lazy = 0;
    while (fgets(line, sizeof line, f)) {
        if (strstr(line, "DM_NVN_HILITE")) { in_hilite = 1; continue; }
        if (!in_hilite) continue;
        if (strstr(line, "dispatch_once")) saw_once = 1;
        if (strstr(line, "static int nk = -1")) saw_lazy = 1;
        if (strstr(line, "d.rasterSampleCount")) break;      // the block ends here
    }
    fclose(f);
    if (!saw_once) { fprintf(stderr, "FAIL: the production DM_NVN_HILITE setup does not use dispatch_once\n"); return 0; }
    if (saw_lazy) { fprintf(stderr, "FAIL: the production DM_NVN_HILITE setup still uses the lazy form\n"); return 0; }
    return 1;
}

static int run(int lazy, const char *what) {
    g_lazy = lazy; g_nk = 0; g_builds = 0; g_building = 0; g_hl = nil;
    pthread_t th[NTHREAD];
    for (int i = 0; i < NTHREAD; i++) if (pthread_create(&th[i], NULL, worker, NULL)) { fprintf(stderr, "FAIL: pthread_create\n"); exit(1); }
    for (int i = 0; i < NTHREAD; i++) pthread_join(th[i], NULL);

    int fail = 0;
    if (g_nk != 3) { fprintf(stderr, "FAIL: %d keys parsed, expected 3\n", g_nk); fail++; }
    if (g_builds != 1) { fprintf(stderr, "FAIL: the library was built %d times, expected exactly 1\n", g_builds); fail++; }
    if (!g_hl) { fprintf(stderr, "FAIL: no library\n"); fail++; }
    else for (int k = 0; k < g_nk; k++) {
        id<MTLFunction> f = [g_hl newFunctionWithName:[NSString stringWithFormat:@"hl%d", k]];
        if (!f) { fprintf(stderr, "FAIL: the function hl%d is missing\n", k); fail++; }
    }
    printf("%-28s library built %d time(s), %d key(s) → %s\n", what, g_builds, g_nk, fail ? "FAIL ✗" : "ok ✓");
    return fail;
}

int main(int argc, char **argv) {
    setenv("DM_NVN_HILITE", "deadbeefdeadbeef,cafebabecafebabe,0123456789abcdef", 1);
    g_dev = MTLCreateSystemDefaultDevice();
    if (!g_dev) { fprintf(stderr, "FAIL: no Metal device\n"); return 1; }

    int fail = 0;
    if (argc > 1 && !strcmp(argv[1], "--old")) {
        // The A/B: the old lazy form must fail this check, or the test has no teeth.
        int old_fail = run(1, "the old lazy form");
        if (!old_fail) { fprintf(stderr, "FAIL: the old lazy form passed — the test cannot tell the two apart\n"); fail++; }
        else printf("the old lazy form is caught by this check ✓\n");
        return fail != 0;
    }

    fail += run(0, "the once-only form");
    if (!production_uses_once()) fail++;
    printf("== %s ==\n", fail ? "FAIL" : "PASS");
    return fail != 0;
}
