# Wrapped Testnet Bitcoin (wTBTC)

A simple ERC20-compatible token contract for wrapping Bitcoin testnet tokens on EVM-compatible chains.

## 📋 Contract Overview

**Token Details:**
- Name: Wrapped Testnet Bitcoin
- Symbol: wTBTC
- Decimals: 18
- Solidity Version: 0.8.20

**Features:**
- ✅ ERC20-compatible token
- ✅ Bridge minting (operator-controlled)
- ✅ User-initiated burning
- ✅ Pausable functionality
- ✅ Operator management

## 🔍 Security Analysis

### ✅ Strengths

1. **Overflow Protection**: Uses Solidity 0.8.20 with built-in overflow checks
2. **No Reentrancy**: No external calls that could cause reentrancy attacks
3. **Event Emissions**: Proper events for all state changes
4. **Clean ERC20**: Standard-compliant implementation
5. **Pausable**: Emergency pause mechanism

### ⚠️ Current Limitations

1. **Centralization Risk**: Single bridge operator with full control
2. **No Replay Protection**: Same bitcoinTxId could theoretically be used multiple times
3. **No Multisig**: Single point of failure for bridge operator
4. **Instant Pause**: No timelock for pausing (could lock funds immediately)
5. **No Proof Verification**: Operator can mint without cryptographic proof of BTC lock
6. **No Emergency Withdrawal**: If paused, funds are frozen indefinitely

## 🏗️ Architecture

```
Bitcoin Testnet                    EVM Chain
     [BTC]                         [wTBTC]
        │                              │
        │  User locks BTC              │
        ├─────────────────────────────>│
        │                              │
        │  Operator detects lock       │
        │  Operator calls mint()       │
        │                          ┌───┴───┐
        │                          │ wTBTC │
        │                          │ minted│
        │                          └───┬───┘
        │                              │
        │  User calls burn()           │
        │<─────────────────────────────┤
        │                              │
        │  Operator unlocks BTC        │
        │                              │
```

## 🚀 Deployment Guide

### Prerequisites

```bash
npm install -g hardhat
npm install @nomicfoundation/hardhat-toolbox
npm install dotenv
```

### Step 1: Set Up Hardhat Project

```bash
cd ~/nexus_agi/contracts/wTBTC/
npm init -y
npm install --save-dev hardhat @nomicfoundation/hardhat-toolbox
npx hardhat init
```

### Step 2: Configure Network

Create `.env` file:
```env
PRIVATE_KEY=your_deployer_private_key_here
SEPOLIA_RPC_URL=https://sepolia.infura.io/v3/YOUR_INFURA_KEY
ETHERSCAN_API_KEY=your_etherscan_api_key
BRIDGE_OPERATOR_ADDRESS=0x...  # Address that will control the bridge
```

Configure `hardhat.config.js`:
```javascript
require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

module.exports = {
  solidity: "0.8.20",
  networks: {
    sepolia: {
      url: process.env.SEPOLIA_RPC_URL,
      accounts: [process.env.PRIVATE_KEY]
    }
  },
  etherscan: {
    apiKey: process.env.ETHERSCAN_API_KEY
  }
};
```

### Step 3: Create Deployment Script

Create `scripts/deploy.js`:
```javascript
async function main() {
  const [deployer] = await ethers.getSigners();
  const bridgeOperator = process.env.BRIDGE_OPERATOR_ADDRESS;

  console.log("Deploying contracts with account:", deployer.address);
  console.log("Bridge operator will be:", bridgeOperator);

  const WrappedTestnetBTC = await ethers.getContractFactory("WrappedTestnetBTC");
  const wTBTC = await WrappedTestnetBTC.deploy(bridgeOperator);

  await wTBTC.waitForDeployment();
  const address = await wTBTC.getAddress();

  console.log("wTBTC deployed to:", address);
  console.log("Verify with:");
  console.log(`npx hardhat verify --network sepolia ${address} ${bridgeOperator}`);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
```

### Step 4: Deploy

```bash
# Deploy to Sepolia testnet
npx hardhat run scripts/deploy.js --network sepolia

# Verify on Etherscan
npx hardhat verify --network sepolia DEPLOYED_ADDRESS BRIDGE_OPERATOR_ADDRESS
```

## 📝 Usage Examples

### For Bridge Operator

```javascript
const wTBTC = await ethers.getContractAt("WrappedTestnetBTC", contractAddress);

// Mint wTBTC when BTC is locked
await wTBTC.mint(
  userAddress,
  ethers.parseEther("1.5"),  // 1.5 wTBTC
  "bitcoin_tx_hash_here"
);

// Pause contract in emergency
await wTBTC.setPaused(true);

// Change bridge operator
await wTBTC.changeBridgeOperator(newOperatorAddress);
```

### For Users

```javascript
const wTBTC = await ethers.getContractAt("WrappedTestnetBTC", contractAddress);

// Check balance
const balance = await wTBTC.balanceOf(userAddress);

// Transfer wTBTC
await wTBTC.transfer(recipientAddress, ethers.parseEther("0.5"));

// Approve spending
await wTBTC.approve(spenderAddress, ethers.parseEther("1.0"));

// Burn wTBTC to get BTC back
await wTBTC.burn(
  ethers.parseEther("1.0"),
  "tb1q... (your Bitcoin testnet address)"
);
```

## 🧪 Testing

Create `test/WrappedTestnetBTC.test.js`:
```javascript
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("WrappedTestnetBTC", function () {
  let wTBTC, operator, user1, user2;

  beforeEach(async function () {
    [operator, user1, user2] = await ethers.getSigners();

    const WrappedTestnetBTC = await ethers.getContractFactory("WrappedTestnetBTC");
    wTBTC = await WrappedTestnetBTC.deploy(operator.address);
  });

  it("Should mint tokens", async function () {
    await wTBTC.connect(operator).mint(
      user1.address,
      ethers.parseEther("10"),
      "btc_tx_123"
    );

    expect(await wTBTC.balanceOf(user1.address)).to.equal(ethers.parseEther("10"));
  });

  it("Should transfer tokens", async function () {
    await wTBTC.connect(operator).mint(user1.address, ethers.parseEther("10"), "tx1");
    await wTBTC.connect(user1).transfer(user2.address, ethers.parseEther("5"));

    expect(await wTBTC.balanceOf(user2.address)).to.equal(ethers.parseEther("5"));
  });

  it("Should burn tokens", async function () {
    await wTBTC.connect(operator).mint(user1.address, ethers.parseEther("10"), "tx1");
    await wTBTC.connect(user1).burn(ethers.parseEther("5"), "tb1qtest");

    expect(await wTBTC.balanceOf(user1.address)).to.equal(ethers.parseEther("5"));
  });

  it("Should pause contract", async function () {
    await wTBTC.connect(operator).setPaused(true);

    await expect(
      wTBTC.connect(operator).mint(user1.address, ethers.parseEther("10"), "tx1")
    ).to.be.revertedWith("Contract paused");
  });
});
```

Run tests:
```bash
npx hardhat test
```

## 🔐 Security Recommendations

### Before Production Deployment:

1. **Add Multisig**: Use Gnosis Safe or similar for bridge operator
2. **Implement Replay Protection**: Track used bitcoinTxIds
3. **Add Timelock**: Delay for critical operations
4. **Proof Verification**: Add SPV proofs or oracle verification
5. **Emergency Withdrawal**: Allow users to withdraw during pause
6. **Rate Limiting**: Limit minting per time period
7. **Professional Audit**: Get contract audited by reputable firm

### Enhanced Version Example:

```solidity
// Add these improvements:
mapping(string => bool) public usedBitcoinTxIds;  // Replay protection

function mint(...) {
    require(!usedBitcoinTxIds[bitcoinTxId], "TxId already used");
    usedBitcoinTxIds[bitcoinTxId] = true;
    // ... rest of mint logic
}
```

## 📊 Gas Estimates

| Function | Estimated Gas |
|----------|--------------|
| mint() | ~60,000 |
| burn() | ~45,000 |
| transfer() | ~52,000 |
| approve() | ~46,000 |
| transferFrom() | ~58,000 |

## 🌐 Supported Networks

**Testnets:**
- Ethereum Sepolia
- Ethereum Holesky
- Polygon Mumbai
- BSC Testnet
- Arbitrum Sepolia
- Optimism Sepolia

**Mainnets (after audit):**
- Ethereum
- Polygon
- BSC
- Arbitrum
- Optimism

## 📞 Bridge Operator Workflow

1. **Monitor Bitcoin Testnet**
   - Watch for incoming BTC transactions to bridge address
   - Verify confirmations (recommended: 6 confirmations)

2. **Verify Lock Transaction**
   - Confirm amount
   - Confirm destination address (in OP_RETURN or similar)
   - Record Bitcoin transaction ID

3. **Mint wTBTC**
   - Call `mint(userAddress, amount, bitcoinTxId)`
   - Wait for Ethereum confirmation
   - Notify user

4. **Monitor Burn Events**
   - Listen for `Burn` events
   - Verify user's wTBTC was actually burned
   - Send BTC to specified Bitcoin address
   - Wait for confirmations

5. **Security Monitoring**
   - Watch for unusual minting patterns
   - Monitor total supply vs locked BTC
   - Regular balance reconciliation
   - Alert on discrepancies

## 🛠️ Development Tools

```bash
# Compile contract
npx hardhat compile

# Run tests
npx hardhat test

# Deploy to local network
npx hardhat node
npx hardhat run scripts/deploy.js --network localhost

# Check contract size
npx hardhat size-contracts

# Generate ABI
npx hardhat export-abi
```

## 📄 License

MIT License - See LICENSE file for details

## ⚠️ Disclaimer

This is a testnet contract for educational and testing purposes. DO NOT use for mainnet production without:
- Professional security audit
- Extensive testing
- Proper multisig setup
- Insurance/backup mechanisms
- Legal compliance review

## 🤝 Contributing

Improvements welcome! Especially:
- Enhanced security features
- Better proof verification
- Multisig integration
- Emergency mechanisms
- Gas optimizations

## 📚 Additional Resources

- [ERC20 Standard](https://eips.ethereum.org/EIPS/eip-20)
- [Solidity Documentation](https://docs.soliditylang.org/)
- [Hardhat Documentation](https://hardhat.org/docs)
- [OpenZeppelin Contracts](https://docs.openzeppelin.com/contracts)
