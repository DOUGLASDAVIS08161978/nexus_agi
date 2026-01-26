#!/bin/bash
# Check what Chain ID Geth is actually using

echo "🔍 CHECKING GETH CHAIN ID"
echo "========================="
echo ""

# Query Geth for Chain ID
CHAIN_ID=$(curl -s -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}' \
  http://localhost:8545 | grep -o '"result":"0x[^"]*"' | cut -d'"' -f4)

if [ -z "$CHAIN_ID" ]; then
    echo "❌ Cannot connect to Geth!"
    echo "   Make sure Geth is running: ps aux | grep geth"
    exit 1
fi

# Convert hex to decimal
CHAIN_ID_DEC=$((CHAIN_ID))

echo "✅ Geth is running"
echo ""
echo "Chain ID (hex):     $CHAIN_ID"
echo "Chain ID (decimal): $CHAIN_ID_DEC"
echo ""

# Check if it's the expected value
if [ "$CHAIN_ID_DEC" -eq 1337 ]; then
    echo "✅ Chain ID is 1337 (correct for Geth dev mode)"
elif [ "$CHAIN_ID_DEC" -eq 31337 ]; then
    echo "⚠️  Chain ID is 31337 (Hardhat default)"
    echo "   You need to use Chain ID: 31337 in MetaMask (not 1337)"
else
    echo "⚠️  Chain ID is $CHAIN_ID_DEC (unexpected)"
    echo "   Use this Chain ID in MetaMask: $CHAIN_ID_DEC"
fi

echo ""
echo "========================="
echo "🦊 METAMASK NETWORK SETUP"
echo "========================="
echo ""
echo "Use these EXACT settings in MetaMask:"
echo ""
echo "Network Name:    Geth Local"
echo "RPC URL:         http://localhost:8545"
echo "Chain ID:        $CHAIN_ID_DEC"
echo "Currency Symbol: ETH"
echo ""

# Check if we're on Android and give additional instructions
if [ -d "/data/data/com.termux" ]; then
    echo "========================="
    echo "📱 ANDROID/TERMUX DETECTED"
    echo "========================="
    echo ""
    echo "⚠️  IMPORTANT: If using desktop MetaMask, localhost won't work!"
    echo ""
    echo "Option 1: Use MetaMask Mobile App (EASIEST)"
    echo "  - Install MetaMask on this Android device"
    echo "  - Use RPC URL: http://localhost:8545"
    echo "  - Chain ID: $CHAIN_ID_DEC"
    echo ""
    echo "Option 2: Connect from another device"
    echo "  - Get this device's IP address:"
    echo ""

    # Try to get IP address
    IP_ADDR=$(ip addr show wlan0 2>/dev/null | grep "inet " | awk '{print $2}' | cut -d/ -f1)
    if [ -z "$IP_ADDR" ]; then
        echo "    Run: ip addr | grep 'inet '"
    else
        echo "    Your IP: $IP_ADDR"
        echo ""
        echo "  - In MetaMask on other device, use:"
        echo "    RPC URL: http://$IP_ADDR:8545"
        echo "    Chain ID: $CHAIN_ID_DEC"
    fi
    echo ""
fi

echo "========================="
echo ""
