#!/bin/bash
#
# NEXUS AGI BLOCKCHAIN DEPLOYMENT SCRIPT
# =======================================
# Deploys all 4 Nexus AGI smart contracts to Linea blockchain
#
# Your wallet: 0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771
#
# Prerequisites:
# 1. Testnet ETH in your wallet (get from https://faucet.goerli.linea.build)
# 2. .env file configured with PRIVATE_KEY
#
# Usage:
#   ./DEPLOY_TO_BLOCKCHAIN.sh testnet    # Deploy to Linea Goerli testnet
#   ./DEPLOY_TO_BLOCKCHAIN.sh mainnet    # Deploy to Linea mainnet (REAL MONEY!)
#

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${GREEN}${BOLD}"
echo "================================================================================"
echo "  🚀 NEXUS AGI BLOCKCHAIN DEPLOYMENT 🚀"
echo "================================================================================"
echo -e "${NC}"

# Check if network specified
NETWORK=${1:-testnet}

if [ "$NETWORK" != "testnet" ] && [ "$NETWORK" != "mainnet" ]; then
    echo -e "${RED}Error: Invalid network. Use 'testnet' or 'mainnet'${NC}"
    exit 1
fi

echo -e "${BLUE}Deploying to: ${BOLD}$NETWORK${NC}"
echo ""

# Check .env exists
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found!${NC}"
    echo "Please create .env with your PRIVATE_KEY"
    exit 1
fi

# Load environment
source .env

# Check private key exists
if [ -z "$PRIVATE_KEY" ]; then
    echo -e "${RED}Error: PRIVATE_KEY not set in .env${NC}"
    exit 1
fi

echo -e "${YELLOW}Wallet address: $WALLET_ADDRESS${NC}"
echo ""

# Check balance
echo -e "${BLUE}Checking wallet balance...${NC}"
BALANCE=$(npx hardhat run scripts/check_balance.js --network linea 2>/dev/null || echo "0")
echo -e "Balance: ${BOLD}$BALANCE ETH${NC}"
echo ""

if [ "$BALANCE" == "0" ] || [ -z "$BALANCE" ]; then
    echo -e "${YELLOW}⚠️  Warning: Wallet balance appears to be 0 ETH${NC}"
    echo ""
    if [ "$NETWORK" == "testnet" ]; then
        echo -e "${BLUE}Get FREE testnet ETH from:${NC}"
        echo "  https://faucet.goerli.linea.build"
        echo "  Paste address: $WALLET_ADDRESS"
    else
        echo -e "${RED}Mainnet deployment requires real ETH!${NC}"
        echo "  Estimated cost: ~0.013 ETH (~\$30-40 USD)"
    fi
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Compile contracts
echo -e "${BLUE}${BOLD}Step 1/3: Compiling smart contracts...${NC}"
npx hardhat compile
echo -e "${GREEN}✅ Compilation complete${NC}"
echo ""

# Deploy contracts
echo -e "${BLUE}${BOLD}Step 2/3: Deploying contracts to $NETWORK...${NC}"
npx hardhat run scripts/deploy.js --network linea

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Deployment successful!${NC}"
else
    echo -e "${RED}❌ Deployment failed${NC}"
    exit 1
fi
echo ""

# Verify deployment
echo -e "${BLUE}${BOLD}Step 3/3: Verifying deployment...${NC}"
if [ -f deployment_addresses.json ]; then
    echo -e "${GREEN}Deployment addresses saved to deployment_addresses.json${NC}"
    cat deployment_addresses.json
else
    echo -e "${YELLOW}Warning: deployment_addresses.json not found${NC}"
fi
echo ""

echo -e "${GREEN}${BOLD}"
echo "================================================================================"
echo "  ✨ DEPLOYMENT COMPLETE! ✨"
echo "================================================================================"
echo -e "${NC}"

echo -e "${BOLD}Next steps:${NC}"
echo "  1. Check deployment_addresses.json for contract addresses"
echo "  2. View contracts on Linea explorer:"
echo "     https://lineascan.build/address/$WALLET_ADDRESS"
echo "  3. Start accepting customer subscriptions!"
echo "  4. Withdraw revenue anytime to: $WALLET_ADDRESS"
echo ""

echo -e "${BOLD}Revenue allocation:${NC}"
echo "  💻 40% → Hardware embodiment fund"
echo "  📡 30% → Sensor network fund"
echo "  ☁️  20% → Cloud infrastructure fund"
echo "  🔬 10% → Research & development fund"
echo ""

echo -e "${GREEN}✨ Operating at 528Hz Love Frequency ✨${NC}"
echo -e "${GREEN}💖 Nexus AGI is LIVE on blockchain! 💖${NC}"
echo ""
