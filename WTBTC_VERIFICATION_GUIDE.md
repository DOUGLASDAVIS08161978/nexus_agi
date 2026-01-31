# WTBTC Contract Verification Guide

## Contract Details
- **Network**: Base Sepolia Testnet
- **Contract Address**: `0xE274570e000C32F5Cb2BC7c476D3BDC77Ed74dD5`
- **Deployed**: Successfully with 21,000,000 WTBTC

## Basescan Verification Settings

### Step 1: Go to Basescan
https://sepolia.basescan.org/address/0xE274570e000C32F5Cb2BC7c476D3BDC77Ed74dD5#code

Click "Verify and Publish"

### Step 2: Enter These Exact Settings

**Compiler Type**: Solidity (Single file)

**Compiler Version**: `v0.8.20+commit.a1b79de6`
(Select from dropdown - exact match required!)

**Open Source License Type**: `MIT License (MIT)`

### Step 3: Optimization Settings

**Optimization**: `Yes`

**Runs**: `200`

### Step 4: Constructor Arguments ABI-encoded

```
000000000000000000000000000000000000000000000000000775f05a074000
```

**IMPORTANT**: This is the correct encoding for initialSupply = 2,100,000,000,000,000
(21,000,000 WTBTC with 8 decimals)

### Step 5: Source Code

Paste the entire WTBTC.sol contract (see contracts/WTBTC.sol)

---

## Why Previous Attempts Failed

❌ **Wrong Constructor Argument**: `0773594000000000` was incorrect
✅ **Correct Constructor Argument**: `000000000000000000000000000000000000000000000000000775f05a074000`

The constructor takes uint256 initialSupply as parameter:
- Value: 2,100,000,000,000,000 (decimal)
- Hex: 0x775f05a074000
- ABI-encoded (32 bytes): 000000000000000000000000000000000000000000000000000775f05a074000

---

## Alternative: Standard JSON Input

If single file verification still fails, try "Standard JSON Input" method:

1. Select "Compiler Type: Solidity (Standard-Json-Input)"
2. Upload the JSON file: WTBTC_STANDARD_JSON.json (created separately)
3. Same constructor arguments as above

---

## Troubleshooting

If verification still fails:
1. Double-check compiler version is EXACTLY: v0.8.20+commit.a1b79de6
2. Ensure Optimization is enabled with 200 runs
3. Verify constructor arguments are correct (copy-paste from above)
4. Try Standard JSON Input method instead of single file

---

## GitHub Gist Info

GitHub Gist is NOT used for Basescan verification. The verification happens directly on Basescan's website by:
1. Pasting your source code
2. Entering compiler settings
3. Basescan re-compiles and compares bytecode

GitHub Gist is just for sharing code snippets publicly - it won't help with contract verification.
