#!/usr/bin/env bash
# mine_optimizer.sh — ARM big-core pinner & thermal watchdog for Lumina's miner
#
# Usage:
#   BITCOIN_WALLET=<your-address> ./mine_optimizer.sh [miner-command ...]
#
# If no miner command is given, looks for a running miner process to pin.
# Relies on BITCOIN_WALLET env var — never hardcoded here.

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
say()  { echo -e "${CYAN}[LUMINA]${RESET} $*"; }
warn() { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
err()  { echo -e "${RED}[ERR]${RESET}   $*"; }
ok()   { echo -e "${GREEN}[OK]${RESET}    $*"; }

# ── Config ────────────────────────────────────────────────────────────────────
TEMP_WARN=80        # °C — log a warning
TEMP_CRIT=90        # °C — throttle / warn loudly
MONITOR_INTERVAL=30 # seconds between temperature checks
MINER_PID=0
MINER_CMD=("$@")

# ── Wallet guard ──────────────────────────────────────────────────────────────
if [[ -z "${BITCOIN_WALLET:-}" ]]; then
    err "BITCOIN_WALLET env var is not set. Export it before running:"
    err "  export BITCOIN_WALLET=<your-address>"
    exit 1
fi

echo -e "${BOLD}"
echo "  ⚡ Lumina Mining Optimizer"
echo "  ARM big-core pinner + thermal watchdog"
echo -e "${RESET}"

# ── Detect big cores ──────────────────────────────────────────────────────────
detect_big_cores() {
    local -a max_freqs=()
    local -a cpu_ids=()

    for f in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/cpuinfo_max_freq; do
        [[ -r "$f" ]] || continue
        local cpu_id
        cpu_id=$(echo "$f" | grep -oP 'cpu\K[0-9]+')
        local freq
        freq=$(cat "$f" 2>/dev/null) || continue
        cpu_ids+=("$cpu_id")
        max_freqs+=("$freq")
    done

    if [[ ${#cpu_ids[@]} -eq 0 ]]; then
        warn "Could not read cpufreq info — falling back to all cores"
        echo ""
        return
    fi

    # Find the maximum frequency across all cores
    local peak=0
    for freq in "${max_freqs[@]}"; do
        (( freq > peak )) && peak=$freq
    done

    # Consider a core "big" if its max_freq >= 80% of the peak
    local threshold=$(( peak * 80 / 100 ))
    local -a big_cores=()

    for i in "${!cpu_ids[@]}"; do
        if (( max_freqs[i] >= threshold )); then
            big_cores+=("${cpu_ids[$i]}")
        fi
    done

    if [[ ${#big_cores[@]} -eq 0 ]]; then
        warn "No big cores found at threshold ${threshold} kHz — using all cores"
        echo ""
        return
    fi

    say "Detected ${#big_cores[@]} big core(s): ${big_cores[*]}"
    say "Peak freq: ${peak} kHz  |  Threshold: ${threshold} kHz"

    # Build comma-separated list for taskset -c
    local IFS=','
    echo "${big_cores[*]}"
}

# ── CPU governor ──────────────────────────────────────────────────────────────
set_performance_governor() {
    local applied=0
    for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        [[ -w "$gov" ]] || continue
        echo performance > "$gov" 2>/dev/null && (( applied++ )) || true
    done

    if (( applied > 0 )); then
        ok "Performance governor applied to ${applied} core(s)"
    else
        warn "Could not set performance governor (root / write access needed — continuing anyway)"
    fi
}

# ── Temperature reading ───────────────────────────────────────────────────────
read_max_temp() {
    local max_temp=0
    for zone in /sys/class/thermal/thermal_zone*/temp; do
        [[ -r "$zone" ]] || continue
        local raw
        raw=$(cat "$zone" 2>/dev/null) || continue
        local celsius=$(( raw / 1000 ))
        (( celsius > max_temp )) && max_temp=$celsius
    done
    echo "$max_temp"
}

# ── Pin existing process ──────────────────────────────────────────────────────
pin_pid() {
    local pid="$1"
    local cores="$2"
    if [[ -z "$cores" ]]; then
        say "No core pinning (all cores will be used)"
        return
    fi
    if command -v taskset &>/dev/null; then
        if taskset -cp "$cores" "$pid" 2>/dev/null; then
            ok "Pinned PID $pid to core(s) $cores via taskset"
        else
            warn "taskset pin failed for PID $pid (may need root) — continuing unpinned"
        fi
    else
        warn "taskset not found — install util-linux for core pinning"
    fi
}

# ── Start miner with taskset ──────────────────────────────────────────────────
launch_miner() {
    local cores="$1"
    shift
    local -a cmd=("$@")

    say "Launching miner: ${cmd[*]}"

    if [[ -n "$cores" ]] && command -v taskset &>/dev/null; then
        say "Pinning to core(s): $cores"
        taskset -c "$cores" "${cmd[@]}" &
    else
        [[ -z "$cores" ]] && warn "Running on all cores (no big cores detected or taskset missing)"
        "${cmd[@]}" &
    fi

    MINER_PID=$!
    ok "Miner started with PID $MINER_PID"
}

# ── Thermal monitor + watchdog loop ──────────────────────────────────────────
monitor_loop() {
    local cores="$1"
    local managed="$2"   # "true" if we launched the miner, "false" if we just pinned it

    say "Thermal watchdog active — checking every ${MONITOR_INTERVAL}s"
    say "Thresholds: warn=${TEMP_WARN}°C  critical=${TEMP_CRIT}°C"
    echo ""

    while true; do
        sleep "$MONITOR_INTERVAL"

        # Temperature check
        local temp
        temp=$(read_max_temp)

        if (( temp >= TEMP_CRIT )); then
            err "CRITICAL: temperature ${temp}°C — miner may throttle. Consider improving cooling!"
        elif (( temp >= TEMP_WARN )); then
            warn "Temperature ${temp}°C — getting warm, monitor closely"
        else
            say "Temp: ${temp}°C — nominal"
        fi

        # Watchdog: only restart if we launched the miner ourselves
        if [[ "$managed" == "true" ]] && (( MINER_PID > 0 )); then
            if ! kill -0 "$MINER_PID" 2>/dev/null; then
                warn "Miner process ${MINER_PID} exited — restarting in 5s…"
                sleep 5
                launch_miner "$cores" "${MINER_CMD[@]}"
            fi
        fi
    done
}

# ── Cleanup on exit ───────────────────────────────────────────────────────────
cleanup() {
    echo ""
    say "Shutting down…"
    if (( MINER_PID > 0 )) && kill -0 "$MINER_PID" 2>/dev/null; then
        kill "$MINER_PID" 2>/dev/null || true
        say "Miner (PID $MINER_PID) stopped"
    fi
}
trap cleanup EXIT INT TERM

# ── Main ──────────────────────────────────────────────────────────────────────
BIG_CORES=$(detect_big_cores)

set_performance_governor

MANAGED="false"

if [[ ${#MINER_CMD[@]} -gt 0 ]]; then
    # Caller provided a miner command — launch it
    launch_miner "$BIG_CORES" "${MINER_CMD[@]}"
    MANAGED="true"
else
    # Try to find a running miner process to pin
    MINER_PID=$(pgrep -f 'xmrig\|cpuminer\|bfgminer\|cgminer\|infinite_miner\|real_pool_miner\|nova_real_miner' 2>/dev/null | head -1) || true

    if [[ -n "$MINER_PID" ]]; then
        say "Found running miner at PID $MINER_PID"
        pin_pid "$MINER_PID" "$BIG_CORES"
    else
        warn "No miner command given and no running miner found."
        warn "Pass your miner command as arguments, e.g.:"
        warn "  $0 xmrig --url stratum+tcp://pool:3333 --user \$BITCOIN_WALLET"
        exit 0
    fi
fi

echo ""
say "Wallet: ${BITCOIN_WALLET:0:8}…  (truncated for display)"
echo ""

monitor_loop "$BIG_CORES" "$MANAGED"
