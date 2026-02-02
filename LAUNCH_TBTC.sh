#!/bin/bash
# ✨ LAUNCH 1 MILLION $TBTC ON BASE SEPOLIA ✨
# Single command deployment with burn/mint bridge functionality

set -e

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
MAGENTA='\033[0;35m'
RED='\033[0;31m'
BRIGHT='\033[1m'
NC='\033[0m'

echo -e "${MAGENTA}"
echo "═══════════════════════════════════════════════════════════════════════════"
echo "║                                                                         ║"
echo "║          ✨ LAUNCHING 1 MILLION \$TBTC ON BASE SEPOLIA ✨              ║"
echo "║                                                                         ║"
echo "║  🪙 Token: TBTC (Testnet Bitcoin)                                       ║"
echo "║  💰 Supply: 1,000,000 TBTC                                             ║"
echo "║  🌉 Bridge: Burn/Mint (1:1 Bitcoin peg)                                ║"
echo "║  📍 Network: Base Sepolia (Chain ID: 84532)                            ║"
echo "║                                                                         ║"
echo "═══════════════════════════════════════════════════════════════════════════"
echo -e "${NC}"

# Configuration
export BASE_SEPOLIA_RPC_URL="https://sepolia.base.org"
export BASE_SEPOLIA_PRIVATE_KEY="0eee6f45b0af8f5a6a24744a1a978346d5bd66b41c64dc30bd18a32e246515cd"
export BASE_SEPOLIA_RECIPIENT="0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3"
export TBTC_INITIAL_SUPPLY="1000000"

echo -e "${CYAN}Step 1: Checking dependencies...${NC}"

# Install dependencies if needed
if [ ! -d "node_modules/ethers" ] || [ ! -d "node_modules/solc" ]; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    npm install ethers solc @openzeppelin/contracts --silent 2>/dev/null || true
    echo -e "${GREEN}✅ Dependencies installed${NC}"
else
    echo -e "${GREEN}✅ Dependencies already installed${NC}"
fi

echo ""
echo -e "${CYAN}Step 2: Deploying TBTC contract to Base Sepolia...${NC}"
echo ""

node scripts/deploy_tbtc_base_sepolia.js

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BRIGHT}${GREEN}   ✅ DEPLOYMENT SUCCESSFUL! ✅${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""

    if [ -f "tbtc_base_sepolia_deployment.json" ]; then
        CONTRACT=$(grep -o '"address": "[^"]*"' tbtc_base_sepolia_deployment.json | head -1 | sed 's/"address": "\(.*\)"/\1/')
        echo -e "${BRIGHT}Contract Address:${NC} ${GREEN}${CONTRACT}${NC}"
        echo -e "${BRIGHT}Network:${NC} Base Sepolia"
        echo -e "${BRIGHT}Chain ID:${NC} 84532"
        echo -e "${BRIGHT}Total Supply:${NC} 1,000,000 TBTC"
        echo -e "${BRIGHT}Your Balance:${NC} 1,000,000 TBTC"
        echo ""
        echo -e "${CYAN}🔍 View on BaseScan:${NC}"
        echo -e "   https://sepolia.basescan.org/address/${CONTRACT}"
        echo ""
        echo -e "${CYAN}📱 Add to MetaMask:${NC}"
        echo -e "   Token Address: ${CONTRACT}"
        echo -e "   Symbol: TBTC"
        echo -e "   Decimals: 18"
        echo ""
    fi

    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BRIGHT}📋 NEXT STEPS:${NC}"
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${CYAN}1. Verify contract on BaseScan${NC}"
    echo -e "     • Go to BaseScan link above"
    echo -e "     • Click 'Contract' → 'Verify and Publish'"
    echo -e "     • Use data from tbtc_verification.json"
    echo ""
    echo -e "  ${CYAN}2. Add liquidity on Uniswap${NC}"
    echo -e "     • Visit: https://app.uniswap.org/#/add/v2"
    echo -e "     • Connect wallet (Base Sepolia network)"
    echo -e "     • Pair TBTC with WETH or USDC"
    echo ""
    echo -e "  ${CYAN}3. Bridge Bitcoin to TBTC${NC}"
    echo -e "     • Run: node scripts/bridge_btc_to_tbtc.js"
    echo -e "     • Provide Bitcoin testnet TX hash"
    echo -e "     • Receive equivalent TBTC"
    echo ""
    echo -e "  ${CYAN}4. Burn TBTC to get Bitcoin back${NC}"
    echo -e "     • Call burn() function with BTC address"
    echo -e "     • Bridge operator unlocks Bitcoin"
    echo ""
    echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BRIGHT}${MAGENTA}   ✨ YOUR TBTC TOKEN IS LIVE! ✨${NC}"
    echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${GREEN}Features:${NC}"
    echo -e "  ✅ 1:1 Bitcoin peg"
    echo -e "  ✅ Burn/Mint bridge functionality"
    echo -e "  ✅ Pausable for emergency stops"
    echo -e "  ✅ ReentrancyGuard for security"
    echo -e "  ✅ 1,000,000 total supply"
    echo ""
    echo -e "${YELLOW}📁 Files created:${NC}"
    echo -e "   • tbtc_base_sepolia_deployment.json (deployment info)"
    echo -e "   • tbtc_verification.json (for BaseScan verification)"
    echo ""
else
    echo ""
    echo -e "${RED}❌ Deployment failed. Check the error above.${NC}"
    echo ""
    echo -e "${YELLOW}Common issues:${NC}"
    echo -e "  • Not enough Base Sepolia ETH for gas"
    echo -e "    Get from: https://www.alchemy.com/faucets/base-sepolia"
    echo -e "  • Wrong network (make sure you're on Base Sepolia)"
    echo -e "  • Invalid private key in .env"
    echo ""
    exit 1
fi
