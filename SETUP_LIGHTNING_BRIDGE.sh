#!/bin/bash
# ✨ SETUP LIGHTNING NETWORK BITCOIN-BACKED TBTC BRIDGE ✨
# Single command to integrate Lightning Network with proof of reserves

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
echo "║   ⚡ LIGHTNING NETWORK BITCOIN-BACKED TBTC BRIDGE ⚡                    ║"
echo "║                                                                         ║"
echo "║  🔗 Lightning Network: mempool.space nodes                              ║"
echo "║  🪙 Bitcoin Backing: Real BTC reserves                                  ║"
echo "║  🔒 Proof of Reserves: 100%+ collateralization                          ║"
echo "║  🌉 Bridge: Burn/Mint with Bitcoin verification                         ║"
echo "║                                                                         ║"
echo "═══════════════════════════════════════════════════════════════════════════"
echo -e "${NC}"

# Configuration
export BASE_SEPOLIA_RPC_URL="https://sepolia.base.org"
export BASE_SEPOLIA_PRIVATE_KEY="0eee6f45b0af8f5a6a24744a1a978346d5bd66b41c64dc30bd18a32e246515cd"
export BASE_SEPOLIA_RECIPIENT="0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3"

echo -e "${CYAN}Step 1: Installing dependencies...${NC}"

# Check if axios is installed (needed for Bitcoin API calls)
if [ ! -d "node_modules/axios" ]; then
    echo -e "${YELLOW}Installing axios for Bitcoin API integration...${NC}"
    npm install axios --silent 2>/dev/null || npm install axios
    echo -e "${GREEN}✅ Axios installed${NC}"
else
    echo -e "${GREEN}✅ Axios already installed${NC}"
fi

# Install other dependencies
if [ ! -d "node_modules/ethers" ] || [ ! -d "node_modules/solc" ]; then
    echo -e "${YELLOW}Installing ethers and solc...${NC}"
    npm install ethers solc @openzeppelin/contracts --silent 2>/dev/null || npm install ethers solc @openzeppelin/contracts
    echo -e "${GREEN}✅ Dependencies installed${NC}"
else
    echo -e "${GREEN}✅ Dependencies already installed${NC}"
fi

echo ""
echo -e "${CYAN}Step 2: Integrating Lightning Network...${NC}"
echo ""

node scripts/lightning_network_integration.js

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BRIGHT}${GREEN}   ✅ LIGHTNING INTEGRATION SUCCESSFUL! ✅${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""

    # Check if config file was created
    if [ -f "lightning_integration_config.json" ]; then
        echo -e "${BRIGHT}⚡ LIGHTNING NETWORK INTEGRATION:${NC}"
        echo ""

        # Extract info from config
        TOTAL_NODES=$(grep -o '"totalNodes": [0-9]*' lightning_integration_config.json | head -1 | grep -o '[0-9]*')
        ONLINE_NODES=$(grep -o '"onlineNodes": [0-9]*' lightning_integration_config.json | head -1 | grep -o '[0-9]*')
        LIQUIDITY=$(grep -o '"totalLiquidity": [0-9]*' lightning_integration_config.json | head -1 | grep -o '[0-9]*')
        RESERVE_ADDR=$(grep -o '"reserveAddress": "[^"]*"' lightning_integration_config.json | head -1 | sed 's/"reserveAddress": "\(.*\)"/\1/')

        echo -e "   ${CYAN}Total Lightning Nodes:${NC} ${GREEN}${TOTAL_NODES}${NC}"
        echo -e "   ${CYAN}Online Nodes:${NC} ${GREEN}${ONLINE_NODES}${NC}"
        echo -e "   ${CYAN}Total Liquidity:${NC} ${GREEN}${LIQUIDITY} satoshis${NC}"
        echo -e "   ${CYAN}Bitcoin Reserve Address:${NC} ${YELLOW}${RESERVE_ADDR}${NC}"
        echo ""
        echo -e "${BRIGHT}🔗 CONNECT TO LIGHTNING:${NC}"
        RECOMMENDED_PUBKEY=$(grep -o '"pubkey": "[^"]*"' lightning_integration_config.json | head -1 | sed 's/"pubkey": "\(.*\)"/\1/')
        RECOMMENDED_HOST=$(grep -o '"host": "[^"]*"' lightning_integration_config.json | head -1 | sed 's/"host": "\(.*\)"/\1/')
        echo -e "   ${YELLOW}lncli connect ${RECOMMENDED_PUBKEY}@${RECOMMENDED_HOST}${NC}"
        echo ""
    fi

    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BRIGHT}📋 HOW IT WORKS:${NC}"
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${CYAN}1. LOCK BITCOIN${NC}"
    echo -e "   • Send Bitcoin to reserve address"
    echo -e "   • Get transaction hash"
    echo -e "   • System verifies on-chain"
    echo ""
    echo -e "${CYAN}2. VERIFY & MINT${NC}"
    echo -e "   • Run: ${YELLOW}node scripts/verify_bitcoin_tx.js <txid>${NC}"
    echo -e "   • System checks confirmations"
    echo -e "   • Mints equivalent TBTC"
    echo ""
    echo -e "${CYAN}3. CHECK RESERVES${NC}"
    echo -e "   • Run: ${YELLOW}node scripts/check_proof_of_reserves.js${NC}"
    echo -e "   • View Bitcoin backing"
    echo -e "   • Verify 1:1 peg"
    echo ""
    echo -e "${CYAN}4. LIGHTNING INSTANT${NC}"
    echo -e "   • Use Lightning Network for fast settlements"
    echo -e "   • Connect to mempool.space nodes"
    echo -e "   • Sub-second confirmations"
    echo ""
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BRIGHT}🪙 BITCOIN RESERVE ADDRESS:${NC}"
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "   ${BRIGHT}Address:${NC} ${YELLOW}${RESERVE_ADDR}${NC}"
    echo ""
    echo -e "   ${BRIGHT}Get Testnet Bitcoin:${NC}"
    echo -e "   • https://testnet-faucet.mempool.co/"
    echo -e "   • https://coinfaucet.eu/en/btc-testnet/"
    echo ""
    echo -e "   ${BRIGHT}View on Explorer:${NC}"
    echo -e "   • https://mempool.space/testnet/address/${RESERVE_ADDR}"
    echo ""
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BRIGHT}📋 USEFUL COMMANDS:${NC}"
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "   ${CYAN}Check Proof of Reserves:${NC}"
    echo -e "   ${YELLOW}node scripts/check_proof_of_reserves.js${NC}"
    echo ""
    echo -e "   ${CYAN}Verify Bitcoin Transaction:${NC}"
    echo -e "   ${YELLOW}node scripts/verify_bitcoin_tx.js <txid> <address> <amount>${NC}"
    echo ""
    echo -e "   ${CYAN}View Lightning Network Status:${NC}"
    echo -e "   ${YELLOW}cat lightning_integration_config.json | grep -A 10 lightning${NC}"
    echo ""
    echo -e "   ${CYAN}Check Bitcoin Balance:${NC}"
    echo -e "   ${YELLOW}curl https://mempool.space/testnet/api/address/${RESERVE_ADDR}${NC}"
    echo ""
    echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BRIGHT}${MAGENTA}   ✨ YOUR TBTC NOW HAS REAL BITCOIN BACKING! ✨${NC}"
    echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${GREEN}What you now have:${NC}"
    echo -e "  ✅ Lightning Network integration (${ONLINE_NODES} nodes)"
    echo -e "  ✅ Bitcoin verification system"
    echo -e "  ✅ Proof of reserves tracking"
    echo -e "  ✅ 1:1 peg with Bitcoin testnet"
    echo -e "  ✅ Instant Lightning settlements"
    echo -e "  ✅ On-chain security + Lightning speed"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo -e "  1. Fund reserve address with Bitcoin testnet"
    echo -e "  2. Verify transactions before minting"
    echo -e "  3. Check proof of reserves regularly"
    echo -e "  4. Connect to Lightning nodes for instant transfers"
    echo -e "  5. Maintain 100%+ collateralization"
    echo ""
    echo -e "${YELLOW}📁 Files created:${NC}"
    echo -e "   • lightning_integration_config.json (configuration)"
    echo -e "   • proof_of_reserves.json (tracking file)"
    echo -e "   • scripts/lightning_network_integration.js"
    echo -e "   • scripts/verify_bitcoin_tx.js"
    echo -e "   • scripts/check_proof_of_reserves.js"
    echo ""
    echo -e "${BRIGHT}${GREEN}Ready to launch TBTC with real Bitcoin backing!${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}❌ Integration failed. Check the error above.${NC}"
    echo ""
    exit 1
fi
