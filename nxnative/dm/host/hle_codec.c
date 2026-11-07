// nn::codec::HardwareOpusDecoder: the real device decodes in hardware via the hwopus service (the game's audio and cutscene tracks are all Opus; this is not an online feature).
// Host: the constructor instead constructs the SDK's software OpusDecoder in the same memory (the sdk bundles libopus, native ARM).
// Basis: both are implementations of nn::codec::IOpusDecoder, with the 10 vtable slots identical one by one in order and signature (destructor×2, GetWorkBufferSize, IsInitialized,
// Initialize, Finalize, GetSampleRate, GetChannelCount, Reset, DecodeInterleaved); the game allocates 0x40 bytes and the software object is 0x30 bytes.
#include "hle.h"

HLE_FORCE("_ZN2nn5codec19HardwareOpusDecoderC1Ei", "hwopus 服务；换成 SDK 软件 OpusDecoder（同一 IOpusDecoder 虚接口）",
          void, h_codec_HwOpusCtor, void *self, int option) {
    (void)option;
    static void (*soft_ctor)(void *);
    if (!soft_ctor && !(soft_ctor = guest_sym("_ZN2nn5codec11OpusDecoderC1Ev"))) die("缺 OpusDecoder 构造");
    soft_ctor(self);
}

// ---- The free-function form of the same API (FMOD in Hades II) ----
// FMOD asks for both the hardware and the software work-buffer size at codec creation and then drives one of the two.
// The hardware size query connects to the hwopus service (GetHardwareOpusDecoderManagerByHipc → hipc InitializeHipcServiceResolution), which the host has no service for, and the SDK aborts
// (sdk+0x2bdbb4, inside GetHardwareOpusDecoderWorkBufferSize).
// The host forwards the whole hardware family to the SDK's software decoder: same arguments, the hardware family's trailing option dropped (the software type fits the game's allocation, which is sized from the larger of the two).
// A small resolver caches each software entry point (they live in the sdk) so a per-buffer decode path does not search the symbol table every time.
static void *codec_sw(const char *name) {
    void *p = guest_sym(name);
    if (!p) die("缺软件 Opus 实现 %s", name);
    return p;
}

HLE_FORCE("_ZN2nn5codec36GetHardwareOpusDecoderWorkBufferSizeEii", "hwopus 服务；改用 SDK 软件 OpusDecoder 尺寸", size_t, h_codec_HwSize2, int rate, int ch) {
    static size_t (*f)(int, int); if (!f) f = codec_sw("_ZN2nn5codec28GetOpusDecoderWorkBufferSizeEii");
    return f(rate, ch);
}
HLE_FORCE("_ZN2nn5codec36GetHardwareOpusDecoderWorkBufferSizeEiii", "hwopus 服务；改用 SDK 软件 OpusDecoder 尺寸", size_t, h_codec_HwSize3, int rate, int ch, int opt) {
    (void)opt;
    static size_t (*f)(int, int); if (!f) f = codec_sw("_ZN2nn5codec28GetOpusDecoderWorkBufferSizeEii");
    return f(rate, ch);
}
HLE_FORCE("_ZN2nn5codec47GetHardwareOpusMultiStreamDecoderWorkBufferSizeEiiii", "hwopus 服务；改用 SDK 软件 OpusDecoder 尺寸", size_t, h_codec_HwMsSize4, int rate, int ch, int total, int coupled) {
    static size_t (*f)(int, int, int, int); if (!f) f = codec_sw("_ZN2nn5codec39GetOpusMultiStreamDecoderWorkBufferSizeEiiii");
    return f(rate, ch, total, coupled);
}
HLE_FORCE("_ZN2nn5codec47GetHardwareOpusMultiStreamDecoderWorkBufferSizeEiiiii", "hwopus 服务；改用 SDK 软件 OpusDecoder 尺寸", size_t, h_codec_HwMsSize5, int rate, int ch, int total, int coupled, int opt) {
    (void)opt;
    static size_t (*f)(int, int, int, int); if (!f) f = codec_sw("_ZN2nn5codec39GetOpusMultiStreamDecoderWorkBufferSizeEiiii");
    return f(rate, ch, total, coupled);
}
HLE_FORCE("_ZN2nn5codec29InitializeHardwareOpusDecoderEPNS0_23HardwareOpusDecoderTypeEiiPvmi", "hwopus 服务；改用 SDK 软件 OpusDecoder", Result, h_codec_HwInit, void *self, int rate, int ch, void *work, size_t workSize, int opt) {
    (void)opt;
    static Result (*f)(void *, int, int, void *, size_t); if (!f) f = codec_sw("_ZN2nn5codec21InitializeOpusDecoderEPNS0_15OpusDecoderTypeEiiPvm");
    return f(self, rate, ch, work, workSize);
}
HLE_FORCE("_ZN2nn5codec40InitializeHardwareOpusMultiStreamDecoderEPNS0_34HardwareOpusMultiStreamDecoderTypeEiiiiPKhPvmi", "hwopus 服务；改用 SDK 软件 OpusDecoder", Result, h_codec_HwMsInit, void *self, int rate, int ch, int total, int coupled, const unsigned char *mapping, void *work, size_t workSize, int opt) {
    (void)opt;
    static Result (*f)(void *, int, int, int, int, const unsigned char *, void *, size_t); if (!f) f = codec_sw("_ZN2nn5codec32InitializeOpusMultiStreamDecoderEPNS0_26OpusMultiStreamDecoderTypeEiiiiPKhPvm");
    return f(self, rate, ch, total, coupled, mapping, work, workSize);
}
HLE_FORCE("_ZN2nn5codec27FinalizeHardwareOpusDecoderEPNS0_23HardwareOpusDecoderTypeE", "hwopus 服务；改用 SDK 软件 OpusDecoder", void, h_codec_HwFinal, void *self) {
    static void (*f)(void *); if (!f) f = codec_sw("_ZN2nn5codec19FinalizeOpusDecoderEPNS0_15OpusDecoderTypeE");
    f(self);
}
HLE_FORCE("_ZN2nn5codec38FinalizeHardwareOpusMultiStreamDecoderEPNS0_34HardwareOpusMultiStreamDecoderTypeE", "hwopus 服务；改用 SDK 软件 OpusDecoder", void, h_codec_HwMsFinal, void *self) {
    static void (*f)(void *); if (!f) f = codec_sw("_ZN2nn5codec30FinalizeOpusMultiStreamDecoderEPNS0_26OpusMultiStreamDecoderTypeE");
    f(self);
}
HLE_FORCE("_ZN2nn5codec24ResetHardwareOpusDecoderEPNS0_23HardwareOpusDecoderTypeE", "hwopus 服务；改用 SDK 软件 OpusDecoder", void, h_codec_HwReset, void *self) {
    static void (*f)(void *); if (!f) f = codec_sw("_ZN2nn5codec16ResetOpusDecoderEPNS0_15OpusDecoderTypeE");
    f(self);
}
HLE_FORCE("_ZN2nn5codec35ResetHardwareOpusMultiStreamDecoderEPNS0_34HardwareOpusMultiStreamDecoderTypeE", "hwopus 服务；改用 SDK 软件 OpusDecoder", void, h_codec_HwMsReset, void *self) {
    static void (*f)(void *); if (!f) f = codec_sw("_ZN2nn5codec27ResetOpusMultiStreamDecoderEPNS0_26OpusMultiStreamDecoderTypeE");
    f(self);
}
HLE_FORCE("_ZN2nn5codec33DecodeOpusInterleavedWithHardwareEPNS0_23HardwareOpusDecoderTypeEPmPiPsmPKvm", "hwopus 服务；改用 SDK 软件 OpusDecoder", Result, h_codec_HwDecode, void *self, size_t *outBytes, int *outSamples, size_t *p, int arg, const void *in, size_t inSize) {
    static Result (*f)(void *, size_t *, int *, size_t *, int, const void *, size_t); if (!f) f = codec_sw("_ZN2nn5codec21DecodeOpusInterleavedEPNS0_15OpusDecoderTypeEPmPiPsmPKvm");
    return f(self, outBytes, outSamples, p, arg, in, inSize);
}
HLE_FORCE("_ZN2nn5codec44DecodeOpusMultiStreamInterleavedWithHardwareEPNS0_34HardwareOpusMultiStreamDecoderTypeEPmPiPsmPKvm", "hwopus 服务；改用 SDK 软件 OpusDecoder", Result, h_codec_HwMsDecode, void *self, size_t *outBytes, int *outSamples, size_t *p, int arg, const void *in, size_t inSize) {
    static Result (*f)(void *, size_t *, int *, size_t *, int, const void *, size_t); if (!f) f = codec_sw("_ZN2nn5codec32DecodeOpusMultiStreamInterleavedEPNS0_26OpusMultiStreamDecoderTypeEPmPiPsmPKvm");
    return f(self, outBytes, outSamples, p, arg, in, inSize);
}
HLE_FORCE("_ZN2nn5codec32GetHardwareOpusDecoderSampleRateEPKNS0_23HardwareOpusDecoderTypeE", "hwopus 服务；改用 SDK 软件 OpusDecoder", int, h_codec_HwSRate, const void *self) {
    static int (*f)(const void *); if (!f) f = codec_sw("_ZN2nn5codec24GetOpusDecoderSampleRateEPKNS0_15OpusDecoderTypeE");
    return f(self);
}
HLE_FORCE("_ZN2nn5codec34GetHardwareOpusDecoderChannelCountEPKNS0_23HardwareOpusDecoderTypeE", "hwopus 服务；改用 SDK 软件 OpusDecoder", int, h_codec_HwCh, const void *self) {
    static int (*f)(const void *); if (!f) f = codec_sw("_ZN2nn5codec26GetOpusDecoderChannelCountEPKNS0_15OpusDecoderTypeE");
    return f(self);
}
HLE_FORCE("_ZN2nn5codec43GetHardwareOpusMultiStreamDecoderSampleRateEPKNS0_34HardwareOpusMultiStreamDecoderTypeE", "hwopus 服务；改用 SDK 软件 OpusDecoder", int, h_codec_HwMsSRate, const void *self) {
    static int (*f)(const void *); if (!f) f = codec_sw("_ZN2nn5codec35GetOpusMultiStreamDecoderSampleRateEPKNS0_26OpusMultiStreamDecoderTypeE");
    return f(self);
}
HLE_FORCE("_ZN2nn5codec45GetHardwareOpusMultiStreamDecoderChannelCountEPKNS0_34HardwareOpusMultiStreamDecoderTypeE", "hwopus 服务；改用 SDK 软件 OpusDecoder", int, h_codec_HwMsCh, const void *self) {
    static int (*f)(const void *); if (!f) f = codec_sw("_ZN2nn5codec37GetOpusMultiStreamDecoderChannelCountEPKNS0_26OpusMultiStreamDecoderTypeE");
    return f(self);
}
HLE_FORCE("_ZN2nn5codec49GetHardwareOpusMultiStreamDecoderTotalStreamCountEPKNS0_34HardwareOpusMultiStreamDecoderTypeE", "hwopus 服务；改用 SDK 软件 OpusDecoder", int, h_codec_HwMsTotal, const void *self) {
    static int (*f)(const void *); if (!f) f = codec_sw("_ZN2nn5codec41GetOpusMultiStreamDecoderTotalStreamCountEPKNS0_26OpusMultiStreamDecoderTypeE");
    return f(self);
}
HLE_FORCE("_ZN2nn5codec50GetHardwareOpusMultiStreamDecoderStereoStreamCountEPKNS0_34HardwareOpusMultiStreamDecoderTypeE", "hwopus 服务；改用 SDK 软件 OpusDecoder", int, h_codec_HwMsStereo, const void *self) {
    static int (*f)(const void *); if (!f) f = codec_sw("_ZN2nn5codec42GetOpusMultiStreamDecoderStereoStreamCountEPKNS0_26OpusMultiStreamDecoderTypeE");
    return f(self);
}
