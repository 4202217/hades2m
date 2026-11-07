#!/bin/bash
# One-command IPA build: the NSP/NSZ files in games/ + keys/prod.keys → games/<game name>.ipa (unsigned).
# Install: hand the IPA to a self-signing tool (AltStore / Sideloadly / LiveContainer); they swap in
# your own certificate and bundle ID at install time.
# Usage:
#   ./make_ipa.sh                # defaults: NSP folder = games/, keys = keys/prod.keys
#   ./make_ipa.sh <extra args>   # passed through to nxnative/scripts/build_ipa_from_nsp.py (-h for all options)
# The first run takes about 10–35 minutes (mostly shader library compilation), later runs about 10
# minutes with the cache.
# Bring your own (never shipped with the repo): the game files (base game + update) and prod.keys;
# only use games you legitimately own.
set -eu

ROOT=$(cd "$(dirname "$0")" && pwd)
NSP_DIR=${NSP_DIR:-$ROOT/games}
KEYS=${KEYS:-$ROOT/keys/prod.keys}
PY=$ROOT/.venv/bin/python

say() { echo "[$(date +%T)] $*"; }
die() { say "FAIL: $*"; exit 1; }

# 1) System dependencies (print the install command for whatever is missing; the nxnative script
#    runs the same prerequisite check again on its own)
missing=""
for c in xcodegen spirv-cross zstd dotnet; do
  command -v "$c" >/dev/null 2>&1 || missing="$missing $c"
done
[ -d /Applications/Xcode.app ] || missing="$missing Xcode"
if [ -n "$missing" ]; then
  say "Missing dependencies:$missing"
  say "  brew install xcodegen spirv-cross zstd"
  say "  brew install --cask dotnet-sdk      # .NET SDK (builds the nx_extract extraction tool)"
  say "  Install Xcode from the App Store"
  exit 1
fi

[ -f "$KEYS" ] || die "no keys at $KEYS (put prod.keys in keys/, or override with KEYS=<path>)"
ls "$NSP_DIR"/*.nsp "$NSP_DIR"/*.nsz >/dev/null 2>&1 \
  || die "no game files in $NSP_DIR (put the base game and update NSP/NSZ in games/, or override with NSP_DIR=<path>)"

# Homebrew's dotnet is not on the default search path of the apphost executables that get built
# (setting PATH alone is not enough: nx_extract and friends fail with "missing runtime"): point
# DOTNET_ROOT at the runtime home.
if [ -z "${DOTNET_ROOT:-}" ]; then
  for d in /opt/homebrew/opt/dotnet/libexec /usr/local/opt/dotnet/libexec /usr/local/share/dotnet; do
    if [ -d "$d/shared/Microsoft.NETCore.App" ]; then export DOTNET_ROOT="$d"; break; fi
  done
fi

# 2) Python virtual environment (lz4/capstone/fonttools are required by the pipeline; nsz is added
#    when an NSZ file is present)
if [ ! -x "$PY" ]; then
  say "creating .venv and installing the Python dependencies"
  python3 -m venv "$ROOT/.venv"
fi
"$PY" -c 'import lz4, capstone, fontTools' 2>/dev/null || "$PY" -m pip install -q lz4 capstone fontTools

# 3) NSZ → NSP (build_ipa_from_nsp.py only picks up *.nsp)
shopt -s nullglob
for z in "$NSP_DIR"/*.nsz; do
  n="${z%.nsz}.nsp"
  if [ -e "$n" ]; then
    say "$(basename "$n") already exists, skipping $(basename "$z")"
  else
    "$PY" -c 'import nsz' 2>/dev/null || "$PY" -m pip install -q nsz
    say "decompressing $(basename "$z") → $(basename "$n")"
    "$ROOT/.venv/bin/nsz" -D "$z"
  fi
done
shopt -u nullglob

say "building the IPA (full output of every step in $NSP_DIR/*_build/logs/)"
exec "$PY" "$ROOT/nxnative/scripts/build_ipa_from_nsp.py" "$NSP_DIR" -k "$KEYS" "$@"
