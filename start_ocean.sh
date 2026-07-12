#!/bin/bash
# Nova Miner — Ocean.xyz proportional pool
# Earns small but steady Bitcoin payouts on every share
cd "$(dirname "$0")"
MINING_POOL_HOST=mine.ocean.xyz \
MINING_POOL_PORT=3334 \
MINING_THREADS=4 \
python3 real_pool_miner.py
