#!/bin/bash
# Complete deployment and liquidity setup for $WTBTC

echo "════════════════════════════════════════════════════════════════════════════"
echo "  🚀 DEPLOY CONTRACTS & ADD LIQUIDITY TO $WTBTC"
echo "════════════════════════════════════════════════════════════════════════════"

# Configuration
export POLYGON_RPC_URL="https://polygon-mainnet.infura.io/v3/5f5c1ddd0f2b469f83dc4b6a1cfa4057"
export PRIVATE_KEY="0eee6f45b0af8f5a6a24744a1a978346d5bd66b41c64dc30bd18a32e246515cd"
export INITIAL_WBTC_SUPPLY="100"
export INITIAL_BRIDGE_SUPPLY="1000"

echo ""
echo "Step 1: Deploying contracts..."
echo "════════════════════════════════════════════════════════════════════════════"
node scripts/deploy_with_ethers.js

if [ ! -f "deployment_info.json" ]; then
    echo "❌ Deployment failed - no deployment_info.json found"
    exit 1
fi

echo ""
echo "✅ Deployment complete!"
echo ""
echo "Step 2: Contract addresses saved to deployment_info.json"
cat deployment_info.json

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "  📋 NEXT STEPS TO ADD LIQUIDITY"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Option 1: Add liquidity on QuickSwap (Polygon DEX)"
echo "  1. Go to: https://quickswap.exchange/#/pool"
echo "  2. Connect your wallet (0x9FE74D9D6f1Ae0Ce1fb3B51d4a82c05b74e280f3)"
echo "  3. Click 'Add Liquidity'"
echo "  4. Select WTBTC token (paste contract address from above)"
echo "  5. Select MATIC as the other token"
echo "  6. Enter amounts and confirm"
echo ""
echo "Option 2: Add liquidity on Uniswap V3 (Polygon)"
echo "  1. Go to: https://app.uniswap.org/add/v2"
echo "  2. Connect wallet and switch to Polygon network"
echo "  3. Import WTBTC token using contract address"
echo "  4. Pair with WMATIC or USDC"
echo "  5. Set price range and add liquidity"
echo ""
echo "Option 3: Use the automated script"
echo "  Run: node scripts/add_liquidity.js"
echo ""
echo "════════════════════════════════════════════════════════════════════════════"
