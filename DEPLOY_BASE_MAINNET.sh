#!/bin/bash
# Deploy WTBTC to Base Mainnet - Multiple Methods Available

echo "🚀 Base Mainnet WTBTC Deployment Options"
echo "═══════════════════════════════════════════"
echo ""
echo -e "\033[1;31m⚠️  WARNING: YOU ARE DEPLOYING TO MAINNET! ⚠️\033[0m"
echo -e "\033[1;31m   This will use REAL ETH for gas fees!\033[0m"
echo ""
echo "Choose deployment method:"
echo "1) Direct deployment (simple, single-chain)"
echo "2) CreateX deployment (deterministic, cross-chain compatible)"
echo ""
read -p "Select option (1 or 2): " -n 1 -r
echo ""
echo ""

if [[ $REPLY == "1" ]]; then
    echo "🚀 Direct Deployment to Base Mainnet"
    ./deploy_base_mainnet.sh
elif [[ $REPLY == "2" ]]; then
    echo "🚀 CreateX Deployment to Base Mainnet"
    ./deploy_base_mainnet_createx.sh
else
    echo "❌ Invalid option selected"
    exit 1
fi
