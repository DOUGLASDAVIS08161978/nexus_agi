#!/usr/bin/env python3
"""
Verify private key and address mapping
"""

from eth_account import Account

# The private key provided
private_key = "c411a4d4365560753ef3ceceac1652ec89240704346bf58ad900d65574f541c9"

# Create account from private key
account = Account.from_key(private_key)

print("\n" + "="*90)
print("  🔍 PRIVATE KEY ADDRESS VERIFICATION")
print("="*90 + "\n")

print(f"Private Key: {private_key}")
print(f"\n✅ Actual Address Controlled by This Private Key:")
print(f"   {account.address}")

print(f"\n📊 Transaction Details from Etherscan:")
print(f"   From: 0x24F6B1ce11C57d40B542f91AC85fA9eB61f78771")
print(f"   Status: FAILED - execution reverted")

print(f"\n⚠️  ADDRESSES WE'VE BEEN WORKING WITH:")
print(f"   Source (claimed): 0x12f4441ec006a6b00ae8bc8800c2443f9b0c775d")
print(f"   Destination:      0xD34beE1C52D05798BD1925318dF8d3292d0e49E6")

print(f"\n🔍 ISSUE FOUND:")
if account.address.lower() == "0x24F6B1ce11C57d40B542f91AC85fA9eB61f78771".lower():
    print(f"   ✅ Private key controls: {account.address}")
    print(f"   ❌ But we tried to send from: 0x12f4441ec006a6b00ae8bc8800c2443f9b0c775d")
    print(f"   ❌ These are DIFFERENT addresses!")
    print(f"\n💡 EXPLANATION:")
    print(f"   The private key you provided controls address:")
    print(f"   {account.address}")
    print(f"\n   But the WBTC (299.7 WBTC) is in a different address:")
    print(f"   0x12f4441ec006a6b00ae8bc8800c2443f9b0c775d")
    print(f"\n   ⚠️  You can only send WBTC from addresses you control!")
    print(f"   The transaction failed because your private key doesn't control")
    print(f"   the address with the WBTC.")
else:
    print(f"   Mismatch detected!")

print(f"\n📋 NEXT STEPS:")
print(f"   1. Check if {account.address} has any WBTC")
print(f"   2. If yes, we can transfer from that address")
print(f"   3. If no, you need the private key for 0x12f4441ec006a6b00ae8bc8800c2443f9b0c775d")
print(f"\n   Check balances at:")
print(f"   https://etherscan.io/token/0x2260fac5e5542a773aa44fbcfedf7c193bc2c599?a={account.address.lower()}")
print(f"   https://etherscan.io/token/0x2260fac5e5542a773aa44fbcfedf7c193bc2c599?a=0x12f4441ec006a6b00ae8bc8800c2443f9b0c775d")

print("\n" + "="*90 + "\n")
