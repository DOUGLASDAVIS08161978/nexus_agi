#!/bin/bash
#
# Pure CURL/WGET Uniswap Pool Creator
# NO Node.js, NO .env dependencies - just pure bash!
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ============================================================================
# CONFIGURATION - Edit these values directly
# ============================================================================

# Your wallet details
WALLET_ADDRESS="0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3"
PRIVATE_KEY="0eee6f45b0af8f5a6a24744a1a978346d5bd66b41c64dc30bd18a32e246515cd"
RPC_URL="https://sepolia.base.org"

# Contract addresses (Base Sepolia)
TBTC_ADDRESS="0x5B060693a0eB04e8ea43E5aDfC99FE5B7B92d53e"
WETH_ADDRESS="0x4200000000000000000000000000000000000006"
POSITION_MANAGER="0x27F971cb582BF9E50F397e4d29a5C7A34f11faA2"
FACTORY="0x4752ba5dbc23f44d87826276bf6fd6b1c372ad24"

# ============================================================================

echo ""
echo -e "${BOLD}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}║  🦄 PURE CURL POOL CREATOR (NO NODE.JS, NO .ENV)                            ║${NC}"
echo -e "${BOLD}════════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

# Function to make JSON-RPC calls
rpc_call() {
    local method="$1"
    local params="$2"

    curl -s -X POST "$RPC_URL" \
        -H "Content-Type: application/json" \
        -d "{\"jsonrpc\":\"2.0\",\"method\":\"$method\",\"params\":$params,\"id\":1}" \
        --connect-timeout 10 \
        --max-time 30 2>/dev/null || wget -qO- --post-data="{\"jsonrpc\":\"2.0\",\"method\":\"$method\",\"params\":$params,\"id\":1}" \
        --header="Content-Type: application/json" \
        --timeout=30 "$RPC_URL" 2>/dev/null
}

# Get Bitcoin price
echo -e "${CYAN}📊 Fetching Bitcoin price...${NC}"

BTC_PRICE=$(curl -s --connect-timeout 5 "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT" 2>/dev/null | grep -o '"price":"[0-9.]*"' | cut -d'"' -f4)
ETH_PRICE=$(curl -s --connect-timeout 5 "https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT" 2>/dev/null | grep -o '"price":"[0-9.]*"' | cut -d'"' -f4)

# Fallback prices
if [ -z "$BTC_PRICE" ] || [ -z "$ETH_PRICE" ]; then
    echo -e "${YELLOW}   Using fallback prices${NC}"
    BTC_PRICE="98000"
    ETH_PRICE="3300"
fi

echo -e "${GREEN}   Bitcoin: \$$BTC_PRICE${NC}"
echo -e "${GREEN}   Ethereum: \$$ETH_PRICE${NC}"

# Calculate ratio
RATIO=$(awk "BEGIN {printf \"%.6f\", $BTC_PRICE / $ETH_PRICE}")
echo -e "${CYAN}   1 TBTC = $RATIO ETH${NC}"
echo ""

# Check balances
echo -e "${BOLD}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}║  CHECKING BALANCES                                                           ║${NC}"
echo -e "${BOLD}════════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${CYAN}Wallet: $WALLET_ADDRESS${NC}"

# Get ETH balance
echo -e "${YELLOW}Fetching ETH balance...${NC}"
ETH_RESULT=$(rpc_call "eth_getBalance" "[\"$WALLET_ADDRESS\",\"latest\"]")

if [ -z "$ETH_RESULT" ]; then
    echo -e "${RED}❌ Failed to connect to RPC${NC}"
    echo -e "${YELLOW}   Tried: $RPC_URL${NC}"
    exit 1
fi

ETH_HEX=$(echo "$ETH_RESULT" | grep -o '"result":"[^"]*"' | cut -d'"' -f4)
ETH_HEX_CLEAN=${ETH_HEX#0x}

if [ -n "$ETH_HEX_CLEAN" ] && [ "$ETH_HEX_CLEAN" != "0" ]; then
    ETH_WEI=$(printf "%d" "0x$ETH_HEX_CLEAN" 2>/dev/null || echo "0")
    ETH_BALANCE=$(awk "BEGIN {printf \"%.6f\", $ETH_WEI / 1000000000000000000}")
else
    ETH_BALANCE="0.000000"
fi

echo -e "${GREEN}✓ ETH Balance: $ETH_BALANCE ETH${NC}"

# Get TBTC balance
echo -e "${YELLOW}Fetching TBTC balance...${NC}"
BALANCE_SIG="0x70a08231"
WALLET_PADDED=$(echo "$WALLET_ADDRESS" | sed 's/0x//' | awk '{printf "000000000000000000000000%s", $0}')
CALL_DATA="${BALANCE_SIG}${WALLET_PADDED}"

TBTC_RESULT=$(rpc_call "eth_call" "[{\"to\":\"$TBTC_ADDRESS\",\"data\":\"$CALL_DATA\"},\"latest\"]")
TBTC_HEX=$(echo "$TBTC_RESULT" | grep -o '"result":"[^"]*"' | cut -d'"' -f4)
TBTC_HEX_CLEAN=${TBTC_HEX#0x}

if [ -n "$TBTC_HEX_CLEAN" ] && [ "$TBTC_HEX_CLEAN" != "0" ]; then
    # Handle large numbers by using Python or bc
    if command -v python3 &>/dev/null; then
        TBTC_BALANCE=$(python3 -c "print('{:.6f}'.format(int('0x$TBTC_HEX_CLEAN', 16) / 1e18))")
    elif command -v python &>/dev/null; then
        TBTC_BALANCE=$(python -c "print('{:.6f}'.format(int('0x$TBTC_HEX_CLEAN', 16) / 1e18))")
    elif command -v bc &>/dev/null; then
        TBTC_WEI=$(echo "ibase=16; ${TBTC_HEX_CLEAN^^}" | bc)
        TBTC_BALANCE=$(echo "scale=6; $TBTC_WEI / 1000000000000000000" | bc)
    else
        TBTC_BALANCE="UNKNOWN (install python or bc)"
    fi
else
    TBTC_BALANCE="0.000000"
fi

echo -e "${GREEN}✓ TBTC Balance: $TBTC_BALANCE TBTC${NC}"

# Check if we have enough
if [ "$ETH_BALANCE" = "0.000000" ]; then
    echo ""
    echo -e "${RED}❌ No ETH for gas!${NC}"
    echo -e "${YELLOW}   Get ETH from: https://www.alchemy.com/faucets/base-sepolia${NC}"
    exit 1
fi

if [ "$TBTC_BALANCE" = "0.000000" ]; then
    echo ""
    echo -e "${RED}❌ No TBTC for pool!${NC}"
    echo -e "${YELLOW}   Run: ./LAUNCH_TBTC.sh${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Balances OK!${NC}"
echo ""

# Save configuration for Node.js script
echo -e "${CYAN}Saving pool configuration...${NC}"

# Ensure private key has 0x prefix
if [[ ! "$PRIVATE_KEY" =~ ^0x ]]; then
    PRIVATE_KEY_WITH_PREFIX="0x$PRIVATE_KEY"
else
    PRIVATE_KEY_WITH_PREFIX="$PRIVATE_KEY"
fi

cat > pool_config.json << EOF
{
  "rpcUrl": "$RPC_URL",
  "privateKey": "$PRIVATE_KEY_WITH_PREFIX",
  "tbtcAddress": "$TBTC_ADDRESS",
  "wethAddress": "$WETH_ADDRESS",
  "positionManager": "$POSITION_MANAGER",
  "walletAddress": "$WALLET_ADDRESS",
  "ratio": "$RATIO"
}
EOF

echo -e "${GREEN}✓ Configuration saved!${NC}"
echo ""

echo -e "${BOLD}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}║  READY TO CREATE POOL                                                        ║${NC}"
echo -e "${BOLD}════════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${CYAN}Pool Details:${NC}"
echo -e "  TBTC: $TBTC_ADDRESS"
echo -e "  WETH: $WETH_ADDRESS"
echo -e "  Price: 1 TBTC = $RATIO ETH"
echo -e "  Your ETH: $ETH_BALANCE ETH"
echo -e "  Your TBTC: $TBTC_BALANCE TBTC"
echo ""

echo -e "${YELLOW}Next step: Run the transaction sender${NC}"
echo -e "${YELLOW}  node scripts/send_pool_transactions.js${NC}"
echo ""
