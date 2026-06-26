# 🦄 Create Uniswap V3 Pool - TBTC Pegged to Bitcoin Price

## What This Does

This creates a **Uniswap V3 liquidity pool** on Base Sepolia that sets your TBTC's initial price to match Bitcoin's current market price.

### How It Works:

1. **Fetches Bitcoin Price** from CoinGecko/Binance API (e.g., $60,000)
2. **Fetches Ethereum Price** from same API (e.g., $3,000)
3. **Calculates Ratio**: 1 BTC = 20 ETH (in this example)
4. **Creates TBTC/WETH Pool** on Uniswap V3 with this initial price
5. **Adds Liquidity** so people can trade

## Important Notes

⚠️ **Price Will Change**: Once the pool is created, the price is determined by market trading. If people buy/sell TBTC, the price moves!

✅ **Your 1:1 Bitcoin Backing Remains**: The proof-of-reserves system ensures each TBTC is backed by real Bitcoin, regardless of market price.

## Prerequisites

Before running this command, you need:

1. ✅ **TBTC Deployed** on Base Sepolia
   - Run: `./LAUNCH_TBTC.sh` first

2. ✅ **Base Sepolia ETH** in your wallet
   - For gas fees (~$1)
   - For liquidity (~0.05 ETH minimum)
   - Get from: https://www.alchemy.com/faucets/base-sepolia

3. ✅ **TBTC Tokens** in your wallet
   - At least 1 TBTC for initial liquidity

## Single Copy-Paste Command

```bash
cd ~/nexus_agi && git pull origin claude/setup-nexus-agi-directory-3joXw && ./CREATE_UNISWAP_POOL.sh
```

## What Happens Step by Step

### Step 1: Check Balances
- Verifies you have ETH, TBTC

### Step 2: Wrap ETH to WETH
- Converts your ETH to WETH (Wrapped ETH)
- WETH is the ERC-20 version of ETH used in pools

### Step 3: Approve Tokens
- Approves Uniswap to use your TBTC and WETH

### Step 4: Create Pool
- Creates TBTC/WETH pool with 0.3% fee tier
- Sets initial price to match Bitcoin

### Step 5: Add Liquidity
- Adds 1 TBTC + equivalent WETH
- Creates full-range liquidity position

## After Pool Creation

### View Your Pool:
- **BaseScan**: `https://sepolia.basescan.org/address/[POOL_ADDRESS]`
- **Uniswap Interface**: `https://app.uniswap.org/pools/[POOL_ADDRESS]`

### Trading:
Anyone can now:
- Buy TBTC with ETH on Uniswap
- Sell TBTC for ETH on Uniswap
- Price changes based on supply/demand

### Your Position:
You'll receive an NFT representing your liquidity position. You earn 0.3% fees on all trades!

## Example Output

```
🦄 CREATE UNISWAP V3 POOL - TBTC PEGGED TO BITCOIN PRICE
════════════════════════════════════════════════════════

📊 Fetching Bitcoin price...
   Bitcoin: $95,000
   Ethereum: $3,500

💱 Price Calculation:
   1 TBTC = 27.1429 ETH
   1 ETH = 0.036842 TBTC

💼 Your Address: 0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3

STEP 1: CHECK BALANCES
══════════════════════

ETH Balance: 0.5 ETH
TBTC Balance: 1000000.0 TBTC

✅ UNISWAP V3 POOL CREATED SUCCESSFULLY!
════════════════════════════════════════

📊 POOL DETAILS:
   Pool Address: 0x...
   TBTC Address: 0x5B060693a0eB04e8ea43E5aDfC99FE5B7B92d53e
   WETH Address: 0x4200000000000000000000000000000000000006
   Fee Tier: 0.3%
   Initial Price: 1 TBTC = 27.1429 ETH
   Network: Base Sepolia

✨ Your TBTC is now tradeable on Uniswap! ✨
```

## How Price Discovery Works

### Initial Price (Set by You):
- 1 TBTC = 27.14 ETH (based on Bitcoin price)

### After Trading Starts:
- If someone **buys TBTC**: Price goes UP
- If someone **sells TBTC**: Price goes DOWN
- Market finds equilibrium based on demand

### Arbitrage:
If TBTC price differs from Bitcoin's real value:
- Arbitrage traders will buy/sell to profit
- This naturally pushes price toward Bitcoin's value
- Your 1:1 backing helps maintain the peg

## Troubleshooting

### Error: "TBTC not deployed"
**Solution**: Run `./LAUNCH_TBTC.sh` first

### Error: "Insufficient ETH balance"
**Solution**: Get Base Sepolia ETH from faucet:
- https://www.alchemy.com/faucets/base-sepolia

### Error: "You need TBTC tokens"
**Solution**: Mint TBTC by:
1. Get testnet4 BTC from faucet
2. Run: `node scripts/mint_with_bitcoin_proof.js <txid>`

### Error: "Pool already exists"
**Solution**: Pool is already created! You can add more liquidity through Uniswap interface

## Next Steps

After creating the pool:

1. **Share the pool link** so others can trade
2. **Add more liquidity** to reduce price impact
3. **Monitor trading** on BaseScan
4. **Earn fees** from every trade (0.3% of volume)

## Technical Details

### Uniswap V3 Addresses (Base Sepolia):
- Factory: `0x4752ba5dbc23f44d87826276bf6fd6b1c372ad24`
- Position Manager: `0x27F971cb582BF9E50F397e4d29a5C7A34f11faA2`
- WETH: `0x4200000000000000000000000000000000000006`

### Fee Tier: 0.3%
- Standard tier for most pairs
- You earn 0.3% of all trade volume

### Liquidity Range: Full Range
- Your liquidity is active at all prices
- Simpler but less capital efficient than concentrated liquidity

## Support

If you encounter issues, check:
1. Gas settings in `.env`
2. RPC endpoint is working
3. Wallet has sufficient funds
4. TBTC contract is deployed

---

**Built with ❤️ for Bitcoin-Ethereum bridge**
