// nn::ro (loading NRO modules at run time): on the real device it connects to the ldr:ro service. The host has no such service; running the SDK's Initialize as is reads a null session in InitializeWithPortName and crashes (Engage).
// Engage (Unity) imports 8 ro interfaces but its RomFS has no .nro / .nrr: at startup it only calls Initialize, and symbol lookups go through LookupSymbol.
// Takeover: Initialize is a no-op; LookupSymbol finds the definition by name in the loaded modules (main / subsdk* / sdk); loading / unloading / registering modules always fails and reports red once (there is nothing to load).
// Only success matters: the game checks IsSuccess only, so the failure code is a non-zero value under the ro module number 22.
#include "hle.h"

#define RO_FAIL ((Result)(22 | (1 << 9)))   // a failure under ro module 22

HLE_FORCE("_ZN2nn2ro10InitializeEv", "连 ldr:ro 服务；宿主没有，查符号直接查已装模块", Result, h_ro_Initialize, void) { LOG("init 接管：ro::Initialize（空操作）"); return 0; }

HLE_FORCE("_ZN2nn2ro12LookupSymbolEPmPKc", "按名字在已装模块里找定义", Result, h_ro_LookupSymbol, uintptr_t *out, const char *name) {
    void *p = name ? guest_sym(name) : NULL;
    static int n; if (n++ < 40) LOG("ro::LookupSymbol %s → %p", name ? name : "(null)", p);
    if (!p) { static int red; if (!red++) LOG("【红】ro::LookupSymbol 找不到 %s（首例）", name ? name : "(null)"); return RO_FAIL; }
    if (out) *out = (uintptr_t)p;
    return 0;
}

HLE_FORCE("_ZN2nn2ro18LookupModuleSymbolEPmPKNS0_6ModuleEPKc", "没有装过 NRO：按模块查一律找不到", Result, h_ro_LookupModuleSymbol, uintptr_t *out, const void *m, const char *name) {
    (void)out; (void)m; static int red; if (!red++) LOG("【红】ro::LookupModuleSymbol %s：没有装过 NRO", name ? name : "(null)"); return RO_FAIL;
}

HLE_FORCE("_ZN2nn2ro13GetBufferSizeEPmPKv", "不装 NRO", Result, h_ro_GetBufferSize, size_t *out, const void *image) {
    (void)image; if (out) *out = 0; static int red; if (!red++) LOG("【红】ro::GetBufferSize：宿主不装 NRO"); return RO_FAIL;
}

HLE_FORCE("_ZN2nn2ro10LoadModuleEPNS0_6ModuleEPKvPvmi", "不装 NRO", Result, h_ro_LoadModule, void *m, const void *image, void *buf, size_t size, int flag) {
    (void)m; (void)image; (void)buf; (void)flag; static int red; if (!red++) LOG("【红】ro::LoadModule：宿主不装 NRO（缓冲 %#zx）", size); return RO_FAIL;
}

// SDK 19 起 LoadModule 多一个 bool（在 bind 标志之后）：哈迪斯 II（SDK 20）的 main 只导入这个形式，用它加载自己的 Epic 在线服务 NRO。
// 旧钩子只按 “…Pvmi” 精确匹配，这个形式会漏过 → 落到 SDK 自己的实现去连 fsp-srv，可能 abort。签名以 SDK 为准（一个多余的 bool）。
// SDK 19 adds a bool after the bind flag: Hades II's main (SDK 20) imports only this form, to load its Epic Online Services NRO. The older hook matches “…Pvmi” exactly and misses this form, so the call reaches the SDK's own code, which opens fsp-srv and may abort.
HLE_FORCE("_ZN2nn2ro10LoadModuleEPNS0_6ModuleEPKvPvmib", "不装 NRO", Result, h_ro_LoadModule2, void *m, const void *image, void *buf, size_t size, int flag, int b) {
    (void)b; return h_ro_LoadModule(m, image, buf, size, flag);
}

// ro::Finalize：Initialize 没有真的连 ldr:ro（宿主没有这个服务），所以收尾也什么都不做；不接管的话它会去找那个并不存在的会话。
// ro::Finalize: Initialize opened no ldr:ro session (the host has no such service), so finalizing does nothing; without this takeover the SDK looks for a session that does not exist.
HLE_FORCE("_ZN2nn2ro8FinalizeEv", "Initialize 没有真的连 ldr:ro：收尾是空操作", void, h_ro_Finalize, void) { }

HLE_FORCE("_ZN2nn2ro12UnloadModuleEPNS0_6ModuleE", "不装 NRO", void, h_ro_UnloadModule, void *m) { (void)m; }

HLE_FORCE("_ZN2nn2ro18RegisterModuleInfoEPNS0_16RegistrationInfoEPKv", "不登记 NRR", Result, h_ro_RegisterModuleInfo, void *info, const void *nrr) {
    (void)info; (void)nrr; static int red; if (!red++) LOG("【红】ro::RegisterModuleInfo：宿主不登记 NRR"); return RO_FAIL;
}

HLE_FORCE("_ZN2nn2ro20UnregisterModuleInfoEPNS0_16RegistrationInfoE", "不登记 NRR", void, h_ro_UnregisterModuleInfo, void *info) { (void)info; }
