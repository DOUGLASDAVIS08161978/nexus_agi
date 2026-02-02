#!/bin/bash
#
# Create Uniswap V3 Pool - TBTC Pegged to Bitcoin Price
# One-command deployment for Termux
#

echo "🦄 CREATING UNISWAP V3 POOL FOR TBTC"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "This will:"
echo "  ✅ Fetch current Bitcoin price"
echo "  ✅ Calculate ETH ratio (e.g., 1 BTC = 20 ETH)"
echo "  ✅ Create TBTC/WETH pool on Uniswap V3"
echo "  ✅ Set initial price to match Bitcoin"
echo "  ✅ Add liquidity to enable trading"
echo ""
echo "⚠️  You need:"
echo "  • Some Base Sepolia ETH (for gas + liquidity)"
echo "  • TBTC tokens (run ./LAUNCH_TBTC.sh first)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Make sure we're in the right directory
cd "$(dirname "$0")"

# Load environment variables
if [ -f .env ]; then
  export $(cat .env | grep -v '^#' | xargs)
fi

# Run the script
node scripts/create_uniswap_pool.js

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ Done!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
