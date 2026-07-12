#!/bin/bash
# Nova Miner — Solo lottery pool (public-pool.io)
# Chance at full 3.125 BTC block reward — all CPU cores dedicated
cd "$(dirname "$0")"
python3 real_pool_miner.py
