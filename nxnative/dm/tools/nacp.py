#!/usr/bin/env python3
# The game name from the NSP's control data (control.nacp, saved in the extraction folder by nx_extract): the iOS app name comes from it (shared by the packing script and build_ipa.sh).
# Usage: nacp.py <control.nacp> (prints the game name; prints an empty line and exits 1 when none can be read); from Python: import nacp; nacp.name(path)
import struct, sys, zlib
from pathlib import Path


def name(path):
    """The title area holds 0x300 bytes per language (name 0x200 + publisher 0x100); take the first non-empty one (0 = American English); a missing file or no sensible name returns an empty string"""
    p = Path(path)
    if not p.is_file(): return ''
    b = p.read_bytes(); t = b[:0x3000]
    try:   # newer SDKs compress the title area: a u16 size + raw deflate, inflating to 32 languages × 0x300 (Mario 1.2.1 does)
        z = zlib.decompress(b[2:2 + struct.unpack_from('<H', b)[0]], -15)
        if len(z) == 0x6000: t = z
    except zlib.error: pass   # not compressed
    for i in range(len(t) // 0x300):
        n = t[i * 0x300:i * 0x300 + 0x200].split(b'\0', 1)[0]
        if not n: continue
        try: n = n.decode('utf-8')
        except UnicodeDecodeError: return ''
        return n if n.isprintable() else ''
    return ''


if __name__ == '__main__':
    if len(sys.argv) != 2: sys.exit('用法 | usage: nacp.py <control.nacp>')
    n = name(sys.argv[1]); print(n); sys.exit(0 if n else 1)
