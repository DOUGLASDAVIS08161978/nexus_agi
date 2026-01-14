#!/bin/bash
###############################################################################
# NEXUS AGI - Single Command Balance Checker
# Copy this ENTIRE file to your local machine and run it!
###############################################################################

echo "================================================================================"
echo "               NEXUS AGI - INSTANT BITCOIN BALANCE CHECKER"
echo "================================================================================"
echo

# Your wallet addresses
PRIMARY="bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass"
SECONDARY="bc1q8z6z78dy5squapjpkeruem98jcezsw37hnae6qjyhxma6jmxyn6qsmqxce"
LIGHTNING="bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh"

echo "Checking 3 wallet balances on Bitcoin MAINNET..."
echo

# Function to get balance
get_balance() {
    local addr=$1
    local name=$2

    echo -n "💰 $name: "

    # Try blockstream.info API
    balance=$(curl -s "https://blockstream.info/api/address/$addr" | \
              python3 -c "import sys, json; data=json.load(sys.stdin); funded=data['chain_stats']['funded_txo_sum']; spent=data['chain_stats']['spent_txo_sum']; print(funded-spent)" 2>/dev/null)

    if [ -n "$balance" ]; then
        btc=$(python3 -c "print(f'{$balance/100_000_000:.8f}')")
        echo "$btc BTC ($balance satoshis)"
    else
        echo "Error retrieving balance"
    fi
}

# Check all wallets
get_balance "$PRIMARY" "PRIMARY  "
get_balance "$SECONDARY" "SECONDARY"
get_balance "$LIGHTNING" "LIGHTNING"

echo
echo "================================================================================"
echo "To see full details, run: ./quick_deploy.sh"
echo "================================================================================"
