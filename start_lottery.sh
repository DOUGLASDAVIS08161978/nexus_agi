#!/bin/bash
# Nova Miner — Solo lottery pool (public-pool.io)
# Chance at full 3.125 BTC block reward — all CPU cores dedicated
#
# If phone throttles after a few minutes, try fewer threads to reduce heat:
#   MINING_THREADS=6 bash start_lottery.sh
cd "$(dirname "$0")"
python3 real_pool_miner.py
