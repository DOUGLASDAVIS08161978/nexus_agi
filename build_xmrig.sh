#!/data/data/com.termux/files/usr/bin/bash
# build_xmrig.sh — Compile XMRig from source on Termux
# Takes ~10-15 minutes on a phone. Run once, then use nova_xmr_miner.py.

set -e

DEST="$HOME/nexus_agi/xmrig"
SRC="/tmp/xmrig_build"

echo ""
echo "══════════════════════════════════════════════════"
echo "  Building XMRig from source for Android/Termux"
echo "══════════════════════════════════════════════════"
echo ""

# ── Step 1: Dependencies ──────────────────────────────────────────────────────
echo "[1/4] Installing build dependencies..."
pkg install -y cmake clang make libuv openssl git

echo ""
echo "[2/4] Cloning XMRig source..."
rm -rf "$SRC"
git clone --depth 1 https://github.com/xmrig/xmrig "$SRC"
cd "$SRC"
mkdir build && cd build

echo ""
echo "[3/4] Configuring build (no GPU, no hwloc — CPU only)..."
cmake .. \
    -DWITH_OPENCL=OFF \
    -DWITH_CUDA=OFF \
    -DWITH_HWLOC=OFF \
    -DWITH_ASM=ON \
    -DWITH_RANDOMX=ON \
    -DCMAKE_BUILD_TYPE=Release

echo ""
echo "[4/4] Compiling... (this is the slow part — 10-15 min on a phone)"
make -j$(nproc)

echo ""
echo "Copying binary to ~/nexus_agi/xmrig..."
cp xmrig "$DEST"
chmod +x "$DEST"

echo ""
echo "══════════════════════════════════════════════════"
echo "  Done!  XMRig is ready at ~/nexus_agi/xmrig"
echo "  Now run:  python ~/nexus_agi/nova_xmr_miner.py"
echo "══════════════════════════════════════════════════"
echo ""
