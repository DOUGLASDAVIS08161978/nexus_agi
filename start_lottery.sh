#!/bin/bash
# Nova Miner — Solo lottery pool (public-pool.io)
# Chance at full 3.125 BTC block reward — ALL 8 CORES DEDICATED
cd "$(dirname "$0")"
MINING_THREADS=8 python3 real_pool_miner.py
