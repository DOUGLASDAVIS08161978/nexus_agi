#!/bin/bash

echo "════════════════════════════════════════════════════════════════════════════════"
echo "   AUTOMATED DEPLOYMENT TOOL FOR wTBTC + HASHPROOF ECOSYSTEM"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "hardhat.config.js" ]; then
    echo -e "${RED}❌ Error: Run this from the hashproof-token directory${NC}"
    exit 1
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  No .env file found. Creating one...${NC}"
    echo ""
    echo "Enter your private key (starts with 0x):"
    read -s PRIVATE_KEY
    echo "PRIVATE_KEY=$PRIVATE_KEY" > .env
    echo -e "${GREEN}✅ .env file created${NC}"
    echo ""
fi

# Check wallet balance
echo -e "${BLUE}📊 Checking wallet balance...${NC}"
node -e "
const ethers = require('ethers');
require('dotenv').config();

async function checkBalance() {
    const provider = new ethers.JsonRpcProvider('https://rpc.sepolia.org');
    const wallet = new ethers.Wallet(process.env.PRIVATE_KEY, provider);

    console.log('Address:', wallet.address);

    const balance = await provider.getBalance(wallet.address);
    const balanceETH = ethers.formatEther(balance);

    console.log('Balance:', balanceETH, 'ETH');

    if (balance === 0n) {
        console.log('');
        console.log('❌ No ETH in wallet!');
        console.log('Get free Sepolia ETH from:');
        console.log('  → https://www.alchemy.com/faucets/ethereum-sepolia');
        console.log('  → https://sepolia-faucet.pk910.de');
        process.exit(1);
    }

    if (parseFloat(balanceETH) < 0.1) {
        console.log('⚠️  Low balance. Recommended: 0.1 ETH');
    } else {
        console.log('✅ Sufficient balance for deployment');
    }
}

checkBalance().catch(err => {
    console.error('Error checking balance:', err.message);
    process.exit(1);
});
"

echo ""
echo -e "${BLUE}🔨 Compiling contracts...${NC}"
npx hardhat compile

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Compilation failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Contracts compiled successfully${NC}"
echo ""

# Ask user what to deploy
echo "What would you like to deploy?"
echo "1) wTBTC only"
echo "2) HashProof ecosystem (Token + Staking + Governance)"
echo "3) Everything (wTBTC + HashProof ecosystem)"
echo ""
read -p "Enter choice (1-3): " CHOICE

echo ""
echo -e "${BLUE}🚀 Deploying to Sepolia testnet...${NC}"
echo ""

case $CHOICE in
    1)
        npx hardhat run scripts/deploy-sepolia.js --network sepolia
        ;;
    2)
        npx hardhat run scripts/deploy-sepolia.js --network sepolia
        ;;
    3)
        npx hardhat run scripts/deploy-sepolia.js --network sepolia
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

if [ $? -eq 0 ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo -e "${GREEN}   ✅ DEPLOYMENT SUCCESSFUL!${NC}"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "📄 Contract addresses saved to: deployment-sepolia.json"
    echo ""
    echo "🔍 View on Sepolia Etherscan:"

    if [ -f "deployment-sepolia.json" ]; then
        node -e "
        const fs = require('fs');
        const data = JSON.parse(fs.readFileSync('deployment-sepolia.json'));
        console.log('');
        for (const [name, address] of Object.entries(data.contracts)) {
            console.log(name + ':');
            console.log('  Address: ' + address);
            console.log('  Etherscan: https://sepolia.etherscan.io/address/' + address);
            console.log('');
        }
        "
    fi

    echo "════════════════════════════════════════════════════════════════════════════════"
else
    echo ""
    echo -e "${RED}❌ Deployment failed. Check errors above.${NC}"
    exit 1
fi
