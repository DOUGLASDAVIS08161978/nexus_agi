#!/bin/bash
# Nova Miner — Solo lottery pool (public-pool.io)
# Chance at full 3.125 BTC block reward
cd "$(dirname "$0")"
MINING_THREADS=4 python3 real_pool_miner.py
