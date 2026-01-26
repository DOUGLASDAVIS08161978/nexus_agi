#!/bin/bash
# ================================================================
# CHECK IF CONTRACTS HAVE REAL BYTECODE (Run in Termux)
# ================================================================

echo "🔍 CHECKING DEPLOYED CONTRACTS"
echo "==============================="
echo ""

# Check if deployment_addresses.json exists
if [ ! -f "deployment_addresses.json" ]; then
    echo "❌ deployment_addresses.json not found!"
    echo "   Run the deployment first"
    exit 1
fi

echo "📋 Deployment file found!"
echo ""
cat deployment_addresses.json
echo ""
echo "==============================="
echo ""

# Create Node.js script to check bytecode
cat > check_bytecode.cjs << 'CHECK_SCRIPT'
const { Web3 } = require('web3');
const fs = require('fs');

const web3 = new Web3('http://localhost:8545');

async function checkContracts() {
    console.log("🔍 Checking contract bytecode on blockchain...\n");

    // Read deployment file
    const deployment = JSON.parse(fs.readFileSync('deployment_addresses.json', 'utf8'));

    if (!deployment.contracts || deployment.contracts.length === 0) {
        console.log("❌ No contracts found in deployment file!");
        process.exit(1);
    }

    let allHaveBytecode = true;

    for (const contract of deployment.contracts) {
        const name = contract.name;
        const address = contract.address;

        console.log(`📋 ${name}: ${address}`);

        // Get bytecode
        const bytecode = await web3.eth.getCode(address);
        const length = bytecode.length;

        if (length > 2) {
            console.log(`   ✅ HAS BYTECODE: ${length} bytes`);
            console.log(`   → MetaMask SHOULD show this as CONTRACT\n`);
        } else {
            console.log(`   ❌ NO BYTECODE!`);
            console.log(`   → MetaMask shows as PERSONAL ADDRESS\n`);
            allHaveBytecode = false;
        }
    }

    console.log("=".repeat(80));

    if (allHaveBytecode) {
        console.log("✅ ALL CONTRACTS HAVE BYTECODE!\n");
        console.log("If MetaMask still shows 'personal address':");
        console.log("  1. Make sure MetaMask is on 'Geth Local' network");
        console.log("  2. RPC URL: http://localhost:8545");
        console.log("  3. Chain ID: 1337");
        console.log("  4. Try REMOVING and RE-ADDING the network in MetaMask");
        console.log("  5. Clear MetaMask cache (Settings → Advanced → Clear activity data)");
    } else {
        console.log("❌ CONTRACTS MISSING BYTECODE!");
        console.log("   Need to redeploy contracts");
    }

    process.exit(0);
}

checkContracts().catch(err => {
    console.error("Error:", err);
    process.exit(1);
});
CHECK_SCRIPT

echo "🔧 Running bytecode verification..."
echo ""

node check_bytecode.cjs
