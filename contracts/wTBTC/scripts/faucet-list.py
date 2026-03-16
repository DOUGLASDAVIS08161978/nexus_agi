#!/usr/bin/env python3
"""
Quick Testnet BTC Faucet List
No input required - just displays the faucet URLs
"""

print("\n" + "="*70)
print("💰 BITCOIN TESTNET FAUCETS - QUICK REFERENCE")
print("="*70)
print()

print("🔑 STEP 1: GET A TESTNET BITCOIN ADDRESS")
print("-"*70)
print()
print("Use one of these wallets in TESTNET mode:")
print()
print("1. Electrum (Recommended - Easy)")
print("   • Download: https://electrum.org/")
print("   • Run with: electrum --testnet")
print("   • Get new address from 'Receive' tab")
print()
print("2. Sparrow Wallet (User-friendly)")
print("   • Download: https://sparrowwallet.com/")
print("   • Settings → Network → Bitcoin Core (Testnet)")
print("   • Create wallet → Get receive address")
print()
print("3. Bitcoin Core (Advanced)")
print("   • Download: https://bitcoin.org/en/download")
print("   • Run: bitcoind -testnet")
print("   • Get address: bitcoin-cli -testnet getnewaddress")
print()

print("="*70)
print("💸 STEP 2: GET FREE TESTNET BTC FROM FAUCETS")
print("="*70)
print()

faucets = [
    {
        "name": "Testnet Faucet",
        "url": "https://testnet-faucet.com/btc-testnet/",
        "amount": "0.01 tBTC",
        "notes": "Fast, reliable, high amount"
    },
    {
        "name": "CoinFaucet",
        "url": "https://coinfaucet.eu/en/btc-testnet/",
        "amount": "0.001 tBTC",
        "notes": "Quick, simple interface"
    },
    {
        "name": "Bitcoin Faucet",
        "url": "https://bitcoinfaucet.uo1.net/",
        "amount": "0.001 tBTC",
        "notes": "Reliable, established"
    },
    {
        "name": "Testnet QC",
        "url": "https://testnet.qc.to/",
        "amount": "Variable",
        "notes": "Community-run"
    }
]

for i, faucet in enumerate(faucets, 1):
    print(f"{i}. {faucet['name']}")
    print(f"   URL: {faucet['url']}")
    print(f"   Amount: {faucet['amount']}")
    print(f"   Notes: {faucet['notes']}")
    print()

print("="*70)
print("🔍 STEP 3: CHECK YOUR BALANCE")
print("="*70)
print()
print("Use a block explorer to check when funds arrive:")
print()
print("Blockstream Explorer:")
print("https://blockstream.info/testnet/")
print()
print("Just paste your address to see transactions and balance.")
print()

print("="*70)
print("⏱️ STEP 4: WAIT FOR CONFIRMATIONS")
print("="*70)
print()
print("• Faucets usually send within 1-5 minutes")
print("• Transactions need 1-6 confirmations")
print("• Check explorer for status")
print("• Most faucets limit 1 request per 24 hours")
print()

print("="*70)
print("🌉 STEP 5: USE WITH wTBTC BRIDGE")
print("="*70)
print()
print("Once you have testnet BTC:")
print()
print("1. Deploy wTBTC contract to Sepolia:")
print("   cd ~/nexus_agi/contracts/wTBTC")
print("   npm run deploy:sepolia")
print()
print("2. Lock your tBTC in bridge address")
print("3. Bridge operator mints wTBTC on Ethereum")
print("4. Use wTBTC in DeFi!")
print()

print("="*70)
print("📱 MOBILE-FRIENDLY WALLETS (FOR TERMUX)")
print("="*70)
print()
print("Since you're on Termux, try these:")
print()
print("1. Use Electrum wallet app on Android")
print("   • Install from F-Droid or official site")
print("   • Switch to testnet in settings")
print("   • Generate address")
print()
print("2. Use online wallet (for testing only!)")
print("   • https://testnet.blockchain.info/")
print("   • Create wallet")
print("   • Get receive address")
print()
print("⚠️  WARNING: Only use with small testnet amounts!")
print()

print("="*70)
print("✅ QUICK START")
print("="*70)
print()
print("1. Open: https://testnet-faucet.com/btc-testnet/")
print("2. Paste your testnet address")
print("3. Complete CAPTCHA")
print("4. Click 'Send'")
print("5. Wait 2-5 minutes")
print("6. Check: https://blockstream.info/testnet/")
print()
print("💚 You'll have testnet BTC to test your bridge!")
print()
