#!/bin/bash

# Quick Deploy Script - NO HARDHAT REQUIRED
# This script deploys using pure web3.js

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "═══════════════════════════════════════════════════════════════════════════"
echo "║                                                                         ║"
echo "║           🚀 QUICK DEPLOY - NO HARDHAT REQUIRED 🚀                      ║"
echo "║                                                                         ║"
echo "═══════════════════════════════════════════════════════════════════════════"
echo -e "${NC}"

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo -e "${RED}Error: Not in project directory!${NC}"
    echo "Please run this from /home/user/nexus_agi/"
    exit 1
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo -e "${RED}Error: .env file not found!${NC}"
    echo "Please create .env with your configuration"
    exit 1
fi

echo -e "${GREEN}✅ Configuration found${NC}\n"

# Install dependencies (only ethers, solc, no hardhat)
echo -e "${CYAN}📦 Installing dependencies (ethers.js, solc)...${NC}"
npm install ethers solc @openzeppelin/contracts --save

echo -e "\n${GREEN}✅ Dependencies installed${NC}\n"

# Run deployment
echo -e "${CYAN}🚀 Deploying contracts using ethers.js...${NC}\n"
node scripts/deploy_with_ethers.js

echo -e "\n${GREEN}✅ Deployment complete!${NC}\n"
echo -e "${YELLOW}Check deployment_info.json for contract addresses${NC}\n"
