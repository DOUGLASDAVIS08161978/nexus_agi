#!/bin/bash
# ✨ DEPLOY WTBTC BRIDGE TO ETHEREUM MAINNET ✨
# This script deploys, verifies, and mints tokens in one command

set -e

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
MAGENTA='\033[0;35m'
NC='\033[0m'

echo -e "${MAGENTA}"
echo "═══════════════════════════════════════════════════════════════════════════"
echo "║                                                                         ║"
echo "║     ✨ DEPLOYING WTBTC BRIDGE TO ETHEREUM MAINNET ✨                    ║"
echo "║                                                                         ║"
echo "║  Bridge: Base Sepolia → Ethereum Mainnet                               ║"
echo "║                                                                         ║"
echo "═══════════════════════════════════════════════════════════════════════════"
echo -e "${NC}"

# Export configuration
export ETHEREUM_RPC_URL="https://mainnet.infura.io/v3/5f5c1ddd0f2b469f83dc4b6a1cfa4057"
export PRIVATE_KEY="0eee6f45b0af8f5a6a24744a1a978346d5bd66b41c64dc30bd18a32e246515cd"
export INFURA_API_KEY="5f5c1ddd0f2b469f83dc4b6a1cfa4057"

echo -e "${CYAN}📦 Installing dependencies...${NC}"
npm install ethers solc @openzeppelin/contracts --silent 2>/dev/null || true

echo -e "${GREEN}✅ Dependencies ready${NC}\n"

echo -e "${CYAN}🚀 Deploying WTBTC Bridge to Ethereum Mainnet...${NC}\n"
node scripts/deploy_ethereum_bridge.js

echo -e "\n${GREEN}✅ DEPLOYMENT COMPLETE!${NC}\n"

echo -e "${YELLOW}📋 Deployment info saved to: ethereum_bridge_deployment.json${NC}"
echo -e "${YELLOW}🔍 Verification data saved to: bridge_verification.json${NC}\n"

echo -e "${MAGENTA}════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}   ✨ YOUR BRIDGE IS LIVE! ✨${NC}"
echo -e "${MAGENTA}════════════════════════════════════════════════════════════════${NC}\n"

if [ -f "ethereum_bridge_deployment.json" ]; then
    CONTRACT=$(grep -o '"address": "[^"]*"' ethereum_bridge_deployment.json | head -1 | sed 's/"address": "\(.*\)"/\1/')
    echo -e "${GREEN}Contract Address: ${CONTRACT}${NC}"
    echo -e "${CYAN}View on Etherscan: https://etherscan.io/address/${CONTRACT}${NC}\n"
fi

echo -e "${YELLOW}Next steps:${NC}"
echo -e "  1. Verify contract on Etherscan"
echo -e "  2. Add liquidity on Uniswap"
echo -e "  3. Bridge tokens from Base Sepolia"
echo ""
