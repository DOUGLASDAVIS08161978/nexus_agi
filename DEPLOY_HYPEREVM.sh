#!/bin/bash
# Deploy TBTC to HyperEVM - Multiple Methods Available

echo "🚀 HyperEVM TBTC Deployment Options"
echo "═══════════════════════════════════════════"
echo ""
echo "Choose deployment method:"
echo "1) Direct deployment (simple, single-chain)"
echo "2) CreateX deployment (deterministic, cross-chain compatible)"
echo ""
read -p "Select option (1 or 2): " -n 1 -r
echo ""
echo ""

if [[ $REPLY == "1" ]]; then
    echo "🚀 Direct Deployment"
    echo "Using pure curl + Node.js (no Python eth libraries)..."
    ./deploy_hyperevm_curl.sh
elif [[ $REPLY == "2" ]]; then
    echo "🚀 CreateX Deployment"
    echo "Deploying via CreateX for deterministic addresses..."
    ./deploy_tbtc_createx.sh
else
    echo "❌ Invalid option selected"
    exit 1
fi

