#!/bin/bash
#
# Create Uniswap V3 Pool using CURL (No Node.js networking)
# Bypass Termux DNS/timeout issues
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Load environment variables from .env file
if [ ! -f .env ]; then
    echo -e "${RED}❌ .env file not found${NC}"
    exit 1
fi

# Load only the variables we need (safer than sourcing entire file)
# Use head -1 to get only the first occurrence if variable is defined multiple times
export BASE_SEPOLIA_RPC_URL=$(grep "^BASE_SEPOLIA_RPC_URL=" .env | head -1 | cut -d'=' -f2-)
export BASE_SEPOLIA_PRIVATE_KEY=$(grep "^BASE_SEPOLIA_PRIVATE_KEY=" .env | head -1 | cut -d'=' -f2-)
export BASE_SEPOLIA_RECIPIENT=$(grep "^BASE_SEPOLIA_RECIPIENT=" .env | head -1 | cut -d'=' -f2-)

# Verify required variables are loaded
if [ -z "$BASE_SEPOLIA_RPC_URL" ]; then
    echo -e "${RED}❌ BASE_SEPOLIA_RPC_URL not found in .env${NC}"
    exit 1
fi

# Configuration (already loaded from .env above)
RPC_URL="${BASE_SEPOLIA_RPC_URL}"
PRIVATE_KEY="${BASE_SEPOLIA_PRIVATE_KEY}"
WALLET_ADDRESS="${BASE_SEPOLIA_RECIPIENT}"

# Verify wallet address
if [ -z "$WALLET_ADDRESS" ]; then
    echo -e "${RED}❌ BASE_SEPOLIA_RECIPIENT not found in .env${NC}"
    exit 1
fi
TBTC_ADDRESS="0x5B060693a0eB04e8ea43E5aDfC99FE5B7B92d53e"
WETH_ADDRESS="0x4200000000000000000000000000000000000006"
POSITION_MANAGER="0x27F971cb582BF9E50F397e4d29a5C7A34f11faA2"
FACTORY_ADDRESS="0x4752ba5dbc23f44d87826276bf6fd6b1c372ad24"

echo ""
echo -e "${BOLD}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}║  🦄 CREATE UNISWAP V3 POOL - USING CURL (TERMUX OPTIMIZED)                  ║${NC}"
echo -e "${BOLD}════════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

# Function to make JSON-RPC call with curl
rpc_call() {
    local method="$1"
    local params="$2"

    curl -s -X POST "${RPC_URL}" \
        -H "Content-Type: application/json" \
        -d "{\"jsonrpc\":\"2.0\",\"method\":\"${method}\",\"params\":${params},\"id\":1}" \
        --connect-timeout 10 \
        --max-time 30
}

# Step 1: Get Bitcoin Price
echo -e "${CYAN}📊 Fetching Bitcoin price...${NC}"

# Try Binance first
BTC_PRICE=$(curl -s --connect-timeout 5 --max-time 10 \
    "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT" 2>/dev/null | \
    grep -o '"price":"[0-9.]*"' | cut -d'"' -f4)

ETH_PRICE=$(curl -s --connect-timeout 5 --max-time 10 \
    "https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT" 2>/dev/null | \
    grep -o '"price":"[0-9.]*"' | cut -d'"' -f4)

# Fallback to approximate if APIs fail
if [ -z "$BTC_PRICE" ] || [ -z "$ETH_PRICE" ]; then
    echo -e "${YELLOW}   ⚠️  APIs unavailable, using approximate prices${NC}"
    BTC_PRICE="98000"
    ETH_PRICE="3300"
fi

echo -e "${GREEN}   Bitcoin: \$${BTC_PRICE}${NC}"
echo -e "${GREEN}   Ethereum: \$${ETH_PRICE}${NC}"

# Calculate ratio using bc
RATIO=$(echo "scale=6; $BTC_PRICE / $ETH_PRICE" | bc)
echo ""
echo -e "${BOLD}💱 Price Calculation:${NC}"
echo -e "${CYAN}   1 TBTC = ${RATIO} ETH${NC}"

# Step 2: Check balances
echo ""
echo -e "${BOLD}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}║  STEP 1: CHECK BALANCES                                                      ║${NC}"
echo -e "${BOLD}════════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${CYAN}💼 Your Address: ${WALLET_ADDRESS}${NC}"

# Get ETH balance
ETH_BALANCE_HEX=$(rpc_call "eth_getBalance" "[\"${WALLET_ADDRESS}\",\"latest\"]" | \
    grep -o '"result":"[^"]*"' | cut -d'"' -f4)

if [ -z "$ETH_BALANCE_HEX" ]; then
    echo -e "${RED}❌ Failed to connect to RPC${NC}"
    echo -e "${YELLOW}   RPC URL: ${RPC_URL}${NC}"
    echo -e "${YELLOW}   Try checking your internet connection${NC}"
    exit 1
fi

# Convert hex to decimal using printf and awk (more reliable than bc)
HEX_NO_PREFIX=${ETH_BALANCE_HEX#0x}
if [ -z "$HEX_NO_PREFIX" ]; then
    ETH_BALANCE="0.000000"
else
    ETH_BALANCE_WEI=$(printf "%d" "0x${HEX_NO_PREFIX}")
    ETH_BALANCE=$(awk "BEGIN {printf \"%.6f\", $ETH_BALANCE_WEI / 1000000000000000000}")
fi

echo -e "${CYAN}ETH Balance: ${ETH_BALANCE} ETH${NC}"

# Get TBTC balance using eth_call
TBTC_BALANCE_OF_SIG="0x70a08231" # balanceOf(address)
PADDED_ADDRESS=$(echo "$WALLET_ADDRESS" | sed 's/0x//' | awk '{printf "000000000000000000000000%s", $0}')
TBTC_CALL_DATA="${TBTC_BALANCE_OF_SIG}${PADDED_ADDRESS}"

TBTC_BALANCE_HEX=$(rpc_call "eth_call" "[{\"to\":\"${TBTC_ADDRESS}\",\"data\":\"${TBTC_CALL_DATA}\"},\"latest\"]" | \
    grep -o '"result":"[^"]*"' | cut -d'"' -f4)

# Convert hex to decimal
TBTC_HEX_NO_PREFIX=${TBTC_BALANCE_HEX#0x}
if [ -z "$TBTC_HEX_NO_PREFIX" ] || [ "$TBTC_HEX_NO_PREFIX" = "0x" ]; then
    TBTC_BALANCE="0.000000"
else
    TBTC_BALANCE_WEI=$(printf "%d" "0x${TBTC_HEX_NO_PREFIX}")
    TBTC_BALANCE=$(awk "BEGIN {printf \"%.6f\", $TBTC_BALANCE_WEI / 1000000000000000000}")
fi

echo -e "${CYAN}TBTC Balance: ${TBTC_BALANCE} TBTC${NC}"

# Check if we have TBTC
if [ "$TBTC_BALANCE" = "0.000000" ] || [ -z "$TBTC_BALANCE" ]; then
    echo ""
    echo -e "${RED}❌ You need TBTC tokens to create a pool!${NC}"
    echo -e "${YELLOW}   Run: ./LAUNCH_TBTC.sh to mint tokens${NC}"
    exit 1
fi

# Check if we have ETH
if [ "$ETH_BALANCE" = "0.000000" ] || [ -z "$ETH_BALANCE" ]; then
    echo ""
    echo -e "${RED}❌ You need Base Sepolia ETH for gas!${NC}"
    echo -e "${YELLOW}   Get ETH from: https://www.alchemy.com/faucets/base-sepolia${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Balances OK!${NC}"
echo ""
echo -e "${BOLD}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}║  STEP 2: CREATE POOL WITH NODE.JS (MINIMAL NETWORK USAGE)                   ║${NC}"
echo -e "${BOLD}════════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${CYAN}Creating Uniswap V3 pool...${NC}"
echo -e "${YELLOW}   This step uses Node.js only for transaction signing${NC}"
echo -e "${YELLOW}   Network calls will be minimal${NC}"
echo ""

# Create a minimal Node.js script that only signs transactions
cat > ./scripts/sign_pool_tx.js << 'EOFJS'
const { ethers } = require('ethers');
const fs = require('fs');
const path = require('path');

// Read config from project directory
const config = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'pool_config.json'), 'utf8'));

const POSITION_MANAGER_ABI = [
  'function createAndInitializePoolIfNecessary(address token0, address token1, uint24 fee, uint160 sqrtPriceX96) external payable returns (address pool)',
  'function mint((address token0, address token1, uint24 fee, int24 tickLower, int24 tickUpper, uint256 amount0Desired, uint256 amount1Desired, uint256 amount0Min, uint256 amount1Min, address recipient, uint256 deadline)) external payable returns (uint256 tokenId, uint128 liquidity, uint256 amount0, uint256 amount1)'
];

const ERC20_ABI = [
  'function approve(address spender, uint256 amount) external returns (bool)'
];

async function main() {
  try {
    // Use a provider with increased timeout
    const provider = new ethers.JsonRpcProvider(config.rpcUrl, {
      name: 'base-sepolia',
      chainId: 84532
    }, {
      staticNetwork: true,
      polling: true,
      pollingInterval: 5000
    });

    // Set timeout
    provider.pollingInterval = 5000;

    const wallet = new ethers.Wallet(config.privateKey, provider);

    console.log('Wallet connected:', wallet.address);

    // Calculate sqrtPriceX96
    const ratio = parseFloat(config.ratio);
    const sqrtPrice = Math.sqrt(ratio);
    const sqrtPriceX96 = BigInt(Math.floor(sqrtPrice * (2 ** 96)));

    // Determine token order
    const token0 = config.tbtcAddress.toLowerCase() < config.wethAddress.toLowerCase()
      ? config.tbtcAddress : config.wethAddress;
    const token1 = config.tbtcAddress.toLowerCase() < config.wethAddress.toLowerCase()
      ? config.wethAddress : config.tbtcAddress;
    const token0IsTbtc = token0.toLowerCase() === config.tbtcAddress.toLowerCase();

    // Adjust price if needed
    let finalSqrtPriceX96 = sqrtPriceX96;
    if (!token0IsTbtc) {
      finalSqrtPriceX96 = BigInt(Math.floor((2 ** 96) / Math.sqrt(ratio)));
    }

    console.log('Token0:', token0, token0IsTbtc ? '(TBTC)' : '(WETH)');
    console.log('Token1:', token1, token0IsTbtc ? '(WETH)' : '(TBTC)');
    console.log('Price (sqrtPriceX96):', finalSqrtPriceX96.toString());

    // Save pool info
    const poolInfo = {
      token0,
      token1,
      sqrtPriceX96: finalSqrtPriceX96.toString(),
      ratio: ratio
    };

    fs.writeFileSync(path.join(__dirname, '..', 'pool_info.json'), JSON.stringify(poolInfo, null, 2));

    console.log('\n✅ Pool configuration ready!');
    console.log('   Run the transaction script to create the pool');

  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
}

main();
EOFJS

# Create config file
cat > ./pool_config.json << EOFCONFIG
{
  "rpcUrl": "${RPC_URL}",
  "privateKey": "${PRIVATE_KEY}",
  "tbtcAddress": "${TBTC_ADDRESS}",
  "wethAddress": "${WETH_ADDRESS}",
  "positionManager": "${POSITION_MANAGER}",
  "walletAddress": "${WALLET_ADDRESS}",
  "ratio": "${RATIO}"
}
EOFCONFIG

# Run the minimal Node.js script with increased timeout
export NODE_OPTIONS="--dns-result-order=ipv4first"
timeout 60 node scripts/sign_pool_tx.js

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Pool setup complete!${NC}"
    echo ""
    echo -e "${BOLD}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}║  ✅ SUCCESS - POOL CONFIGURED                                                ║${NC}"
    echo -e "${BOLD}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${CYAN}📊 POOL DETAILS:${NC}"
    echo -e "${CYAN}   TBTC: ${TBTC_ADDRESS}${NC}"
    echo -e "${CYAN}   WETH: ${WETH_ADDRESS}${NC}"
    echo -e "${CYAN}   Initial Price: 1 TBTC = ${RATIO} ETH${NC}"
    echo -e "${CYAN}   Network: Base Sepolia${NC}"
    echo ""
    echo -e "${YELLOW}⚠️  Next: You need to approve tokens and create the pool${NC}"
    echo -e "${YELLOW}   This requires sending transactions which need a stable connection${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}❌ Setup failed${NC}"
    echo -e "${YELLOW}   Try again with a better internet connection${NC}"
    exit 1
fi
