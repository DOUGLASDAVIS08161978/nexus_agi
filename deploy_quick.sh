#!/bin/bash
# Quick deploy with Infura Polygon RPC
export POLYGON_RPC_URL="https://polygon-mainnet.infura.io/v3/5f5c1ddd0f2b469f83dc4b6a1cfa4057"
export PRIVATE_KEY="0eee6f45b0af8f5a6a24744a1a978346d5bd66b41c64dc30bd18a32e246515cd"
export INITIAL_WBTC_SUPPLY="100"
export INITIAL_BRIDGE_SUPPLY="1000"
node scripts/deploy_with_ethers.js
