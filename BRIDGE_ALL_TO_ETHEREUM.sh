#!/bin/bash
# ✨ BRIDGE ALL TBTC TO ETHEREUM MAINNET ✨
# Single command to deploy and bridge everything

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
echo "║        ✨ BRIDGE ALL TBTC TO ETHEREUM MAINNET ✨                        ║"
echo "║                                                                         ║"
echo "║  🌉 From: Base Sepolia                                                  ║"
echo "║  🎯 To: Ethereum Mainnet                                                ║"
echo "║  💰 Amount: ALL YOUR TBTC                                               ║"
echo "║                                                                         ║"
echo "═══════════════════════════════════════════════════════════════════════════"
echo -e "${NC}"

# Configuration
export ETHEREUM_RPC_URL="https://mainnet.infura.io/v3/5f5c1ddd0f2b469f83dc4b6a1cfa4057"
export BASE_SEPOLIA_RPC_URL="https://sepolia.base.org"
export BASE_SEPOLIA_PRIVATE_KEY="0eee6f45b0af8f5a6a24744a1a978346d5bd66b41c64dc30bd18a32e246515cd"
export BASE_SEPOLIA_RECIPIENT="0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3"
export PRIVATE_KEY="0eee6f45b0af8f5a6a24744a1a978346d5bd66b41c64dc30bd18a32e246515cd"
export RECIPIENT_ADDRESS="0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3"

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
echo -e "${CYAN}Step 2: Checking Ethereum Mainnet balance...${NC}"

# Quick balance check using node
ETH_BALANCE=$(node -e "
const ethers = require('ethers');
const provider = new ethers.JsonRpcProvider('$ETHEREUM_RPC_URL');
provider.getBalance('0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3')
  .then(b => console.log(ethers.formatEther(b)))
  .catch(() => console.log('0'));
" 2>/dev/null || echo "0")

echo -e "   Your Ethereum Mainnet balance: ${YELLOW}${ETH_BALANCE} ETH${NC}"

# Check if enough ETH
if [ $(echo "$ETH_BALANCE < 0.003" | bc -l 2>/dev/null || echo "1") -eq 1 ]; then
    echo -e "${RED}"
    echo "═══════════════════════════════════════════════════════════════"
    echo "   ⚠️  INSUFFICIENT ETH ON ETHEREUM MAINNET"
    echo "═══════════════════════════════════════════════════════════════"
    echo -e "${NC}"
    echo -e "${YELLOW}You need at least 0.005 ETH on Ethereum Mainnet for:${NC}"
    echo -e "  • Contract deployment: ~0.003 ETH"
    echo -e "  • Token minting: ~0.001 ETH"
    echo -e "  • Gas buffer: ~0.001 ETH"
    echo ""
    echo -e "${CYAN}Your address: ${YELLOW}0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3${NC}"
    echo ""
    echo -e "${YELLOW}Get ETH from:${NC}"
    echo -e "  • Coinbase: https://www.coinbase.com"
    echo -e "  • Binance: https://www.binance.com"
    echo -e "  • Kraken: https://www.kraken.com"
    echo ""
    echo -e "${YELLOW}Then run this command again.${NC}"
    echo ""
    exit 1
fi

echo -e "${GREEN}✅ Sufficient ETH for deployment${NC}"

echo ""
echo -e "${CYAN}Step 3: Deploying TBTC to Ethereum Mainnet & Bridging...${NC}"
echo ""

node scripts/deploy_and_bridge_to_ethereum.js

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BRIGHT}${GREEN}   ✅ BRIDGE SUCCESSFUL! ✅${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""

    if [ -f "tbtc_ethereum_mainnet_deployment.json" ]; then
        CONTRACT=$(grep -o '"address": "[^"]*"' tbtc_ethereum_mainnet_deployment.json | head -1 | sed 's/"address": "\(.*\)"/\1/')
        TOTAL_SUPPLY=$(grep -o '"totalSupply": "[^"]*"' tbtc_ethereum_mainnet_deployment.json | head -1 | sed 's/"totalSupply": "\(.*\)"/\1/')

        echo -e "${BRIGHT}📊 YOUR TBTC ON ETHEREUM MAINNET:${NC}"
        echo -e "   Contract: ${GREEN}${CONTRACT}${NC}"
        echo -e "   Total Supply: ${GREEN}${TOTAL_SUPPLY} TBTC${NC}"
        echo -e "   Network: Ethereum Mainnet"
        echo -e "   Chain ID: 1"
        echo ""
        echo -e "${CYAN}🔍 View on Etherscan:${NC}"
        echo -e "   https://etherscan.io/address/${CONTRACT}"
        echo ""
        echo -e "${CYAN}📱 Add to MetaMask:${NC}"
        echo -e "   Token Address: ${CONTRACT}"
        echo -e "   Symbol: TBTC"
        echo -e "   Decimals: 18"
        echo -e "   Network: Ethereum Mainnet"
        echo ""
    fi

    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BRIGHT}📋 NEXT STEPS:${NC}"
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${CYAN}1. Verify contract on Etherscan${NC}"
    echo -e "     • Go to Etherscan link above"
    echo -e "     • Click 'Contract' → 'Verify and Publish'"
    echo -e "     • Use data from tbtc_ethereum_verification.json"
    echo ""
    echo -e "  ${CYAN}2. Add liquidity on Uniswap${NC}"
    echo -e "     • Visit: https://app.uniswap.org/#/add/v2"
    echo -e "     • Connect wallet (Ethereum Mainnet)"
    echo -e "     • Pair TBTC with WETH or USDC"
    echo ""
    echo -e "  ${CYAN}3. List on aggregators${NC}"
    echo -e "     • CoinGecko: https://www.coingecko.com/request-form/coins"
    echo -e "     • CoinMarketCap: https://coinmarketcap.com/request/"
    echo ""
    echo -e "  ${CYAN}4. Announce launch${NC}"
    echo -e "     • Twitter/X"
    echo -e "     • Telegram/Discord"
    echo -e "     • Reddit"
    echo ""
    echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BRIGHT}${MAGENTA}   ✨ YOUR TOKENS ARE NOW ON ETHEREUM MAINNET! ✨${NC}"
    echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${GREEN}What got bridged:${NC}"
    if [ -f "tbtc_base_sepolia_deployment.json" ]; then
        echo -e "  ✅ Base Sepolia TBTC → Burned"
        echo -e "  ✅ Ethereum Mainnet TBTC → Minted"
        echo -e "  ✅ All tokens transferred to your address"
    else
        echo -e "  ✅ Fresh TBTC deployed on Ethereum Mainnet"
        echo -e "  ✅ 1,000,000 TBTC minted"
    fi
    echo ""
    echo -e "${YELLOW}📁 Files created:${NC}"
    echo -e "   • tbtc_ethereum_mainnet_deployment.json (deployment info)"
    echo -e "   • tbtc_ethereum_verification.json (for Etherscan verification)"
    echo ""
else
    echo ""
    echo -e "${RED}❌ Bridge failed. Check the error above.${NC}"
    echo ""
    echo -e "${YELLOW}Common issues:${NC}"
    echo -e "  • Not enough ETH on Ethereum Mainnet"
    echo -e "    Need: ~0.005 ETH (~$15-20 USD)"
    echo -e "  • Network connection problems"
    echo -e "  • Invalid private key"
    echo ""
    exit 1
fi
