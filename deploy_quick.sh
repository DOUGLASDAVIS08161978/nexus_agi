#!/bin/bash
# Quick deploy with free public Polygon RPC (no API key needed)
export POLYGON_RPC_URL="https://polygon-rpc.com"
export PRIVATE_KEY="0eee6f45b0af8f5a6a24744a1a978346d5bd66b41c64dc30bd18a32e246515cd"
export INITIAL_WBTC_SUPPLY="100"
export INITIAL_BRIDGE_SUPPLY="1000"
node scripts/deploy_with_ethers.js
