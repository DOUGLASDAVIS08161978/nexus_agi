#!/bin/bash
#
# Create Uniswap V3 Pool - CURL Version (Termux Optimized)
# One-command deployment that bypasses Node.js networking issues
#

echo "🦄 UNISWAP V3 POOL CREATION - TERMUX OPTIMIZED"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "This script uses CURL for all network calls to avoid timeout issues"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Make sure we're in the right directory
cd "$(dirname "$0")"

# Step 1: Configure pool (uses curl for balance checks)
./scripts/create_pool_curl.sh

if [ $? -eq 0 ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "READY TO CREATE POOL"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Do you want to send transactions now? (y/n)"
    echo ""
    echo "This will:"
    echo "  • Wrap ETH to WETH"
    echo "  • Approve tokens"
    echo "  • Create Uniswap V3 pool"
    echo "  • Add liquidity"
    echo ""
    echo "You need:"
    echo "  • Base Sepolia ETH for gas (~0.002 ETH)"
    echo "  • Stable internet connection"
    echo ""
    read -p "Continue? (y/n): " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "SENDING TRANSACTIONS..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""

        # Set Node.js options for better connectivity
        export NODE_OPTIONS="--dns-result-order=ipv4first"

        # Send transactions
        node scripts/send_pool_transactions.js
    else
        echo ""
        echo "Cancelled. Pool configuration saved to pool_info.json"
        echo "Run 'node scripts/send_pool_transactions.js' when ready"
        echo ""
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ Done!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
