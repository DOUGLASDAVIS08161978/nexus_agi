# 🚀 QUICK START - Deploy Your Token in 5 Minutes!

## ⚡ FASTEST METHOD: Use Remix IDE

### 1️⃣ Get Sepolia ETH (2 minutes)

Go to: **https://www.alchemy.com/faucets/ethereum-sepolia**
- Connect MetaMask
- Request free Sepolia ETH
- Wait for confirmation

### 2️⃣ Open Remix (30 seconds)

Go to: **https://remix.ethereum.org**

### 3️⃣ Copy Contract (1 minute)

1. In Remix, click **"contracts"** folder in left sidebar
2. Click **"+"** icon to create new file
3. Name it: `TestnetWBTC.sol`
4. **Copy-paste the entire contract** from below:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract TestnetWBTC is ERC20, ERC20Burnable, Ownable {
    uint8 private _decimals;

    constructor(uint256 initialSupply) ERC20("Testnet Wrapped Bitcoin", "tWBTC") Ownable(msg.sender) {
        _decimals = 8;
        _mint(msg.sender, initialSupply * 10**_decimals);
    }

    function decimals() public view virtual override returns (uint8) {
        return _decimals;
    }

    function mint(address to, uint256 amount) public onlyOwner {
        _mint(to, amount);
    }

    function batchMint(address[] calldata recipients, uint256[] calldata amounts) public onlyOwner {
        require(recipients.length == amounts.length, "Arrays must have same length");
        for (uint256 i = 0; i < recipients.length; i++) {
            _mint(recipients[i], amounts[i]);
        }
    }

    function airdrop(address[] calldata recipients, uint256 amount) public onlyOwner {
        for (uint256 i = 0; i < recipients.length; i++) {
            _mint(recipients[i], amount);
        }
    }
}
```

### 4️⃣ Compile (30 seconds)

1. Click **"Solidity Compiler"** icon (left sidebar - looks like "S")
2. Make sure compiler is **0.8.20+**
3. Click **"Compile TestnetWBTC.sol"**
4. Wait for green checkmark ✅

### 5️⃣ Deploy (1 minute)

1. Click **"Deploy & Run"** icon (left sidebar - looks like Ethereum logo)
2. In **"Environment"** dropdown → Select: **"Injected Provider - MetaMask"**
3. MetaMask pops up → Click **"Connect"**
4. Make sure MetaMask shows **"Sepolia"** network
5. In **"Contract"** dropdown → Select: **"TestnetWBTC"**
6. Next to orange Deploy button, enter: **5** (for 5 tokens)
7. Click orange **"Deploy"** button
8. MetaMask pops up → Click **"Confirm"**
9. Wait 15 seconds

### 6️⃣ Get Your Contract Address (30 seconds)

Look at bottom of Remix:
- You'll see: `[vm] from: 0x... to: TestnetWBTC.`**`(0x...)`**
- **Copy that 0x address** - that's your token contract!

---

## 📱 ADD TO METAMASK

1. Open MetaMask
2. Make sure you're on **Sepolia**
3. Click **"Import tokens"** at bottom
4. Click **"Custom token"**
5. Paste your contract address (from step 6)
6. Should auto-fill: Symbol: **tWBTC**, Decimals: **8**
7. Click **"Import"**

**You should now see: 5.00000000 tWBTC** ✅

---

## 💸 SEND TO YOUR FRIEND

Now you can send to: **0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771**

1. In MetaMask, click **tWBTC**
2. Click **"Send"**
3. Paste: `0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771`
4. Enter amount (e.g., 2.5)
5. Confirm
6. Done! 🎉

---

## ✅ THAT'S IT!

You now have:
- ✅ Your own ERC-20 token on Sepolia
- ✅ Real contract address
- ✅ 5 tWBTC in your wallet
- ✅ Can send to anyone
- ✅ Can mint more anytime

**Total time: ~5 minutes** ⚡

---

## 🪙 MINT MORE TOKENS (Optional)

In Remix, under **"Deployed Contracts"**:
1. Find **"mint"** function
2. Enter:
   - `to`: `0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771`
   - `amount`: `500000000` (= 5 tokens with 8 decimals)
3. Click **"transact"**
4. Confirm in MetaMask

---

## 🆘 PROBLEMS?

**"Insufficient funds"**
→ Get Sepolia ETH from faucet (step 1)

**"Compilation failed"**
→ Check compiler version is 0.8.20+

**"Can't see token in MetaMask"**
→ Make sure you're on Sepolia network
→ Double-check contract address

---

**Questions? Post your contract address and I'll help verify it!** 🚀
