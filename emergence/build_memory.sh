#!/usr/bin/env bash
# build_memory.sh — Compile Lumina's C memory accelerator
#
# Run from inside ~/nexus_agi/emergence/
# Works on Termux/ARM64 and Linux x86_64

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$DIR/lumina_vecdot.c"
OUT="$DIR/lumina_vecdot.so"

echo "  Building Lumina memory accelerator..."
echo "  Source : $SRC"
echo "  Output : $OUT"

# Detect compiler
if command -v gcc >/dev/null 2>&1; then
    CC=gcc
elif command -v clang >/dev/null 2>&1; then
    CC=clang
else
    echo "  ERROR: no C compiler found (install gcc: pkg install gcc)"
    exit 1
fi

echo "  Compiler: $CC"

# Detect architecture for optimal flags
ARCH=$(uname -m)
case "$ARCH" in
    aarch64|arm64)
        ARCH_FLAGS="-march=armv8-a+simd"
        echo "  Architecture: ARM64 (NEON SIMD enabled)"
        ;;
    armv7*)
        ARCH_FLAGS="-march=armv7-a -mfpu=neon -mfloat-abi=softfp"
        echo "  Architecture: ARM32 (NEON enabled)"
        ;;
    x86_64)
        ARCH_FLAGS="-march=native"
        echo "  Architecture: x86_64 (AVX auto-vectorization)"
        ;;
    *)
        ARCH_FLAGS=""
        echo "  Architecture: $ARCH (scalar fallback)"
        ;;
esac

# Compile
$CC \
    -O3 \
    $ARCH_FLAGS \
    -shared \
    -fPIC \
    -ffast-math \
    -o "$OUT" \
    "$SRC" \
    -lm

echo ""
echo "  ✓ Built: $OUT"
echo "  ✓ $(ls -lh "$OUT" | awk '{print $5}') shared library ready"
echo ""
echo "  Lumina's memory search is now C-accelerated."
