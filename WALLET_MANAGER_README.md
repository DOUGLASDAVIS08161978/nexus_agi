# WalletManager Smart Contract & Web3-Onboard Integration

Complete setup for wallet interaction with smart contracts, specifically configured for wallet address: `0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771`

## 📋 Overview

This project includes:
- **WalletManager.sol** - Smart contract for managing ETH deposits, withdrawals, and rewards
- **Deployment Script** - Automated deployment with your wallet as owner
- **Web3-Onboard Integration** - Frontend wallet connection (MetaMask, etc.)
- **Contract Interaction Module** - Easy-to-use functions for interacting with the contract

## 🚀 Quick Start

### 1. Compile the Contract

```bash
npx hardhat compile
```

### 2. Deploy to Network

#### Local Hardhat Network (for testing)
```bash
# Terminal 1: Start local node
npx hardhat node

# Terminal 2: Deploy
npx hardhat run scripts/deploy_wallet_manager.js --network localhost
```

#### Sepolia Testnet
```bash
# Make sure you have PRIVATE_KEY in .env file
npx hardhat run scripts/deploy_wallet_manager.js --network sepolia
```

#### Ethereum Mainnet
```bash
npx hardhat run scripts/deploy_wallet_manager.js --network mainnet
```

### 3. Save Contract Address

After deployment, save the contract address from the output and update it in:
- `frontend/src/web3-onboard-config.js` (line 9)

## 📦 Contract Features

### WalletManager Contract

#### Deposit & Withdrawal
- `depositETH()` - Deposit ETH into the contract
- `withdrawETH(amount)` - Withdraw specific amount
- `withdrawAllETH()` - Withdraw entire balance

#### Token Support
- `depositToken(tokenAddress, amount)` - Deposit ERC20 tokens
- `withdrawToken(tokenAddress, amount)` - Withdraw ERC20 tokens
- `withdrawAllTokens(tokenAddress)` - Withdraw all tokens of a type

#### Rewards System
- Automatic reward accumulation based on deposited balance
- `claimRewards()` - Claim accumulated rewards
- `getPendingRewards(address)` - View pending rewards

#### View Functions
- `getETHBalance(address)` - Check user's ETH balance
- `getTokenBalance(address, token)` - Check token balance
- `getContractBalance()` - View total contract balance

#### Owner Functions (for 0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771)
- `setRewardRate(rate)` - Update reward rate
- `fundRewards()` - Add ETH for reward distribution
- `addOperator(address)` - Add authorized operators
- `pause()` / `unpause()` - Emergency controls

## 💻 Using Web3-Onboard

### Setup Frontend

1. Install dependencies (already done):
```bash
npm install @web3-onboard/core @web3-onboard/injected-wallets ethers
```

2. Import and use:
```javascript
import { connectWallet } from './frontend/src/web3-onboard-config.js';
import { depositETH } from './frontend/src/wallet-manager-contract.js';

// Connect wallet
const { provider, address } = await connectWallet();

// Deposit ETH
await depositETH(provider, '0.1'); // Deposit 0.1 ETH
```

## 🔧 Direct Contract Interaction

### Using Hardhat Console

```bash
npx hardhat console --network <network>
```

```javascript
// Get contract
const WalletManager = await ethers.getContractFactory("WalletManager");
const contract = await WalletManager.attach("YOUR_CONTRACT_ADDRESS");

// Check owner
await contract.owner();
// Returns: 0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771

// Deposit ETH
await contract.depositETH({ value: ethers.parseEther("0.1") });

// Check balance
const balance = await contract.getETHBalance("0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771");
console.log(ethers.formatEther(balance));

// Claim rewards
await contract.claimRewards();

// Withdraw
await contract.withdrawETH(ethers.parseEther("0.05"));
```

### Using Ethers.js Script

Create `interact.js`:

```javascript
const { ethers } = require("ethers");

const CONTRACT_ADDRESS = "YOUR_CONTRACT_ADDRESS";
const ABI = [ /* ABI from compilation */ ];

async function main() {
    // Connect to network
    const provider = new ethers.JsonRpcProvider("https://ethereum-sepolia-rpc.publicnode.com");

    // Connect wallet
    const wallet = new ethers.Wallet("YOUR_PRIVATE_KEY", provider);

    // Get contract
    const contract = new ethers.Contract(CONTRACT_ADDRESS, ABI, wallet);

    // Deposit
    const tx = await contract.depositETH({
        value: ethers.parseEther("0.1")
    });
    await tx.wait();
    console.log("Deposited!");
}

main();
```

## 🎯 Supported Networks

- **Ethereum Mainnet** (Chain ID: 1)
- **Sepolia Testnet** (Chain ID: 11155111)
- **Polygon Mainnet** (Chain ID: 137)
- **Mumbai Testnet** (Chain ID: 80001)
- **Hardhat Local** (Chain ID: 31337)

## 🔐 Security Features

- ReentrancyGuard protection
- Pausable emergency stop
- Owner-only administrative functions
- Operator role management
- Minimum deposit requirements

## 📊 Contract Owner

The contract owner is set to: **0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771**

This address has:
- Full administrative control
- Ability to set reward rates
- Emergency pause/unpause
- Operator management
- Default operator privileges

## 🧪 Testing

Run tests:
```bash
npx hardhat test
```

Check coverage:
```bash
npx hardhat coverage
```

## 📝 Example Workflow

1. **Deploy Contract**
   ```bash
   npx hardhat run scripts/deploy_wallet_manager.js --network sepolia
   ```

2. **Connect Wallet (Frontend)**
   - Open the dApp
   - Click "Connect Wallet"
   - Select MetaMask
   - Approve connection

3. **Deposit Funds**
   - Enter amount
   - Click "Deposit ETH"
   - Confirm transaction in MetaMask

4. **Wait for Rewards**
   - Rewards accumulate automatically
   - Check "Pending Rewards"

5. **Claim Rewards**
   - Click "Claim Rewards"
   - Confirm transaction

6. **Withdraw**
   - Enter amount or click "Withdraw All"
   - Confirm transaction

## 🛠️ Building the Frontend

For production, use a bundler like Vite:

```bash
npm install -g vite
cd frontend
npm init -y
npm install @web3-onboard/core @web3-onboard/injected-wallets ethers
vite
```

## 📚 Additional Resources

- [Web3-Onboard Docs](https://onboard.blocknative.com/)
- [Ethers.js Docs](https://docs.ethers.org/)
- [Hardhat Docs](https://hardhat.org/docs)
- [OpenZeppelin Contracts](https://docs.openzeppelin.com/contracts/)

## ⚠️ Important Notes

1. **Always test on testnet first** before mainnet deployment
2. **Keep private keys secure** - never commit them to git
3. **Verify contracts** on Etherscan after deployment
4. **Audit smart contracts** before handling significant funds
5. **Monitor gas prices** for cost-effective transactions

## 🤝 Support

For issues or questions:
- Check contract events in block explorer
- Use Hardhat console for debugging
- Review transaction receipts for errors

## 📄 License

MIT License - See contract files for details
