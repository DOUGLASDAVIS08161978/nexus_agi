#!/usr/bin/env node
/**
 * Simple contract compiler
 * Compiles WalletManager.sol and generates deployment-ready files
 */

const fs = require('fs');
const path = require('path');
const solc = require('solc');

console.log("🔨 Compiling WalletManager contract...\n");

// Read contract source
const contractPath = path.join(__dirname, 'contracts', 'WalletManagerSimple.sol');
const source = fs.readFileSync(contractPath, 'utf8');

// Prepare compiler input
const input = {
    language: 'Solidity',
    sources: {
        'WalletManagerSimple.sol': {
            content: source
        }
    },
    settings: {
        outputSelection: {
            '*': {
                '*': ['abi', 'evm.bytecode']
            }
        },
        optimizer: {
            enabled: true,
            runs: 200
        }
    }
};

// Compile
const output = JSON.parse(solc.compile(JSON.stringify(input)));

// Check for errors
if (output.errors) {
    const errors = output.errors.filter(e => e.severity === 'error');
    if (errors.length > 0) {
        console.error("❌ Compilation failed:\n");
        errors.forEach(err => console.error(err.formattedMessage));
        process.exit(1);
    }
}

// Extract bytecode and ABI
const contract = output.contracts['WalletManagerSimple.sol']['WalletManagerSimple'];
const bytecode = '0x' + contract.evm.bytecode.object;
const abi = contract.abi;

// Save to files
const buildDir = path.join(__dirname, 'build');
if (!fs.existsSync(buildDir)) {
    fs.mkdirSync(buildDir);
}

fs.writeFileSync(
    path.join(buildDir, 'WalletManager.json'),
    JSON.stringify({ abi, bytecode }, null, 2)
);

console.log("✅ Contract compiled successfully!");
console.log(`\n📦 Bytecode length: ${bytecode.length} characters`);
console.log(`📋 ABI functions: ${abi.filter(x => x.type === 'function').length}`);
console.log(`\n💾 Saved to: build/WalletManager.json\n`);
