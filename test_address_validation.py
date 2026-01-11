#!/usr/bin/env python3
"""
Demonstration: Why Mainnet Addresses Can't Receive Testnet Coins
"""

def validate_testnet_address(address: str) -> bool:
    """Check if address is valid for testnet"""
    testnet_prefixes = ['tb1', 'n', 'm', '2']

    print(f"\n{'='*80}")
    print(f"TESTING ADDRESS: {address}")
    print(f"{'='*80}")

    # Check prefix
    is_testnet = any(address.startswith(prefix) for prefix in testnet_prefixes)

    if is_testnet:
        print(f"✅ PREFIX CHECK: Valid testnet prefix detected")
        for prefix in testnet_prefixes:
            if address.startswith(prefix):
                print(f"   Prefix: '{prefix}' (testnet)")
        print(f"✅ RESULT: This address CAN be used with testnet")
        print(f"   - Can receive testnet Bitcoin")
        print(f"   - Can be used in our wallet system")
        print(f"   - Compatible with testnet network")
    else:
        print(f"❌ PREFIX CHECK: This appears to be a MAINNET address")

        # Identify mainnet type
        if address.startswith('bc1'):
            print(f"   Prefix: 'bc1' (mainnet SegWit)")
        elif address.startswith('1'):
            print(f"   Prefix: '1' (mainnet legacy)")
        elif address.startswith('3'):
            print(f"   Prefix: '3' (mainnet P2SH)")
        else:
            print(f"   Prefix: Unknown")

        print(f"❌ RESULT: This address CANNOT be used with testnet")
        print(f"   - Will REJECT testnet transactions")
        print(f"   - Only accepts real Bitcoin (mainnet)")
        print(f"   - Completely incompatible with testnet network")
        print(f"\n💡 WHY?")
        print(f"   Mainnet and testnet are separate blockchain networks")
        print(f"   Like trying to use a Visa card on Mastercard network")
        print(f"   The networks don't communicate or share coins")

    print(f"{'='*80}\n")
    return is_testnet


def main():
    """Test various Bitcoin addresses"""

    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║              BITCOIN ADDRESS VALIDATION DEMONSTRATION                       ║
║                                                                            ║
║  This demonstrates why mainnet addresses cannot receive testnet coins     ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

    # Test cases
    addresses = [
        ("bc1qyhkq7usdfhhhynkjksdqfx32u3rmv94y0htsal", "Your mainnet address"),
        ("tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx", "Example testnet SegWit"),
        ("tb1qva9h6chqy9x4jrp8rdjm69czhj5d83eyy3aqpk", "Our generated testnet address"),
        ("n33npN2oRNJFFpdoxkrm2CVyZAfRGFxALt", "Testnet legacy address"),
        ("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", "Mainnet legacy (Satoshi's address)"),
        ("3J98t1WpEZ73CNmYviecrnyiWrnqRhWNLy", "Mainnet P2SH"),
    ]

    results = []

    for address, description in addresses:
        print(f"\n📝 DESCRIPTION: {description}")
        is_valid = validate_testnet_address(address)
        results.append((address, description, is_valid))

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}\n")

    print("✅ TESTNET-COMPATIBLE ADDRESSES:")
    testnet_count = 0
    for addr, desc, valid in results:
        if valid:
            testnet_count += 1
            print(f"   {testnet_count}. {addr[:20]}... ({desc})")

    print("\n❌ MAINNET-ONLY ADDRESSES (REJECTED BY TESTNET):")
    mainnet_count = 0
    for addr, desc, valid in results:
        if not valid:
            mainnet_count += 1
            print(f"   {mainnet_count}. {addr[:20]}... ({desc})")

    print(f"\n{'='*80}")
    print(f"KEY TAKEAWAY")
    print(f"{'='*80}\n")
    print("""
Your address: bc1qyhkq7usdfhhhynkjksdqfx32u3rmv94y0htsal

❌ This is a MAINNET address (starts with 'bc1')
❌ CANNOT receive testnet coins
❌ Only accepts real Bitcoin
❌ Testnet network will REJECT transactions to this address

To use testnet, you need a TESTNET address like:
✅ tb1qva9h6chqy9x4jrp8rdjm69czhj5d83eyy3aqpk (our generated address)
✅ tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx (example address)

Think of it this way:
- Mainnet = Production database (real money, real consequences)
- Testnet = Staging database (fake money, safe to experiment)
- You can't transfer data between them - they're separate systems!

Our wallet system correctly prevents this mistake by validating addresses
before creating transactions. This is a SAFETY FEATURE, not a bug!
""")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
