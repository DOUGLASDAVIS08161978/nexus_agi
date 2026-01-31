# WalletManager Contract Deployment Information

## Sepolia Testnet Deployment

**Deployment Date:** January 28, 2026
**Network:** Sepolia Testnet (Chain ID: 11155111)
**Contract Address:** `0x798d8B4D8677c4cf3Bc9B0B38ba8dfe318DE60E4`
**Owner Address:** `0x24F6B1ce11C57d40B542f91AC85fA9eB61f78771`
**Block Number:** 10142767
**Transaction Hash:** Check Sepolia Etherscan

## Contract Configuration

- **Reward Rate:** 0.001 ETH per second per ETH deposited
- **Minimum Deposit:** 0.001 ETH
- **Owner Operator Status:** Enabled

## Network URLs

### Sepolia Testnet
- **Etherscan:** https://sepolia.etherscan.io/address/0x798d8B4D8677c4cf3Bc9B0B38ba8dfe318DE60E4
- **RPC URL:** https://ethereum-sepolia-rpc.publicnode.com

## Testing the Contract

### Using Hardhat Console
```bash
npx hardhat console --network sepolia
```

```javascript
const contract = await ethers.getContractAt("WalletManager", "0x798d8B4D8677c4cf3Bc9B0B38ba8dfe318DE60E4");

// Check owner
await contract.owner();

// Deposit 0.01 ETH
await contract.depositETH({ value: ethers.parseEther("0.01") });

// Check your balance
const balance = await contract.getETHBalance("0x24F6B1ce11C57d40B542f91AC85fA9eB61f78771");
console.log("Balance:", ethers.formatEther(balance), "ETH");

// Check pending rewards
const rewards = await contract.getPendingRewards("0x24F6B1ce11C57d40B542f91AC85fA9eB61f78771");
console.log("Rewards:", ethers.formatEther(rewards), "ETH");

// Claim rewards
await contract.claimRewards();

// Withdraw
await contract.withdrawETH(ethers.parseEther("0.005"));
```

### Using Interaction Script
```bash
# Update CONTRACT_ADDRESS in scripts/interact_wallet_manager.js first
export WALLET_MANAGER_ADDRESS=0x798d8B4D8677c4cf3Bc9B0B38ba8dfe318DE60E4
npx hardhat run scripts/interact_wallet_manager.js --network sepolia
```

## Frontend Integration

The contract address has been updated in:
- `frontend/src/web3-onboard-config.js`
- `frontend/index.html`

To connect via frontend:
1. Switch MetaMask to Sepolia network
2. Open the frontend application
3. Connect your wallet
4. Interact with the contract

## Get Sepolia Test ETH

If you need more Sepolia ETH for testing:
- https://sepoliafaucet.com/
- https://www.alchemy.com/faucets/ethereum-sepolia
- https://faucet.quicknode.com/ethereum/sepolia

## Mainnet Deployment (When Ready)

To deploy to Ethereum mainnet:
1. Ensure wallet `0x24F6B1ce11C57d40B542f91AC85fA9eB61f78771` has sufficient ETH (≥0.005 ETH)
2. Run: `npx hardhat run scripts/deploy_wallet_manager.js --network mainnet`
3. Update all configuration files with new mainnet address

## Contract Features

### User Functions
- `depositETH()` - Deposit ETH into contract
- `withdrawETH(amount)` - Withdraw specific amount
- `withdrawAllETH()` - Withdraw entire balance
- `claimRewards()` - Claim accumulated rewards
- `depositToken(token, amount)` - Deposit ERC20 tokens
- `withdrawToken(token, amount)` - Withdraw ERC20 tokens

### Owner Functions (0x24F6B1ce11C57d40B542f91AC85fA9eB61f78771)
- `setRewardRate(rate)` - Update reward rate
- `fundRewards()` - Add ETH to rewards pool
- `addOperator(address)` - Add authorized operators
- `removeOperator(address)` - Remove operators
- `pause()` / `unpause()` - Emergency controls
- `setMinDepositAmount(amount)` - Update minimum deposit

### View Functions
- `getETHBalance(user)` - Check user's deposited balance
- `getPendingRewards(user)` - Check pending rewards
- `getContractBalance()` - Total contract balance
- `owner()` - Contract owner address
- `isOperator(address)` - Check operator status

## Security Notes

- ✅ ReentrancyGuard protection enabled
- ✅ Pausable emergency stop available
- ✅ Owner-only administrative functions
- ✅ Minimum deposit requirements enforced
- ✅ Secure file permissions on .env (600)
- ✅ Private keys never committed to git

## Support & Documentation

For full documentation, see: `WALLET_MANAGER_README.md`
