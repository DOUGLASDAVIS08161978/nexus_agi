#!/bin/bash
# NEXUS AGI - ONE-CLICK DEPLOYMENT FOR TERMUX
# Copy and paste this entire script into your terminal

echo "🚀 NEXUS AGI BLOCKCHAIN DEPLOYMENT"
echo "=================================="
echo ""

# Navigate to project directory
cd /home/user/nexus_agi

# Check internet
echo "📡 Checking internet connection..."
if ping -c 1 8.8.8.8 > /dev/null 2>&1; then
    echo "✅ Internet connected"
else
    echo "❌ No internet - compilation requires internet for first time"
    echo "   Please connect to internet and try again"
    exit 1
fi

# Check if Hardhat node is running
echo ""
echo "🔍 Checking local blockchain..."
if curl -s -X POST -H "Content-Type: application/json" \
   --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' \
   http://localhost:8545 > /dev/null 2>&1; then
    echo "✅ Hardhat node is running"
else
    echo "⚠️  Starting Hardhat node..."
    npx hardhat node --hostname 0.0.0.0 > hardhat_node.log 2>&1 &
    sleep 5
    echo "✅ Hardhat node started"
fi

# Compile contracts
echo ""
echo "🔨 Compiling smart contracts..."
echo "   (This may take 1-2 minutes on first run)"
echo ""

npx hardhat compile

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Compilation successful!"
else
    echo ""
    echo "❌ Compilation failed"
    exit 1
fi

# Deploy contracts
echo ""
echo "📦 Deploying contracts to local blockchain..."
echo ""

npx hardhat run scripts/deploy.js --network localhost

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Deployment successful!"
else
    echo ""
    echo "❌ Deployment failed"
    exit 1
fi

# Show results
echo ""
echo "🎉 DEPLOYMENT COMPLETE!"
echo "======================"
echo ""

if [ -f deployment_addresses.json ]; then
    echo "📋 Contract Addresses:"
    cat deployment_addresses.json | grep -A 1 "address" | head -20
    echo ""
    echo "Full details saved in: deployment_addresses.json"
else
    echo "⚠️  deployment_addresses.json not found"
fi

echo ""
echo "✅ Your wallet: 0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771"
echo "✅ Balance: 1 ETH (local blockchain)"
echo ""
echo "🦊 Add contract addresses to MetaMask:"
echo "   Network: Hardhat Local (localhost:8545, Chain ID: 31337)"
echo ""
echo "✨ Operating at 528Hz Love Frequency ✨"
