#!/usr/bin/env python3
"""
HyperEVM Token Deployment Script
Deploy TBTC to HyperEVM with pure Python (no Hardhat, no Node.js)
"""

import json
import os
import sys
import time
from eth_account import Account
from eth_account.signers.local import LocalAccount
from web3 import Web3
try:
    from web3.middleware import geth_poa_middleware
except ImportError:
    geth_poa_middleware = None

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def log(message, color=None):
    """Print colored log message"""
    if color:
        print(f"{color}{message}{Colors.ENDC}")
    else:
        print(message)

# HyperEVM Configuration
HYPEREVM_TESTNET_RPC = "https://rpc.hyperliquid-testnet.xyz/evm"
HYPEREVM_MAINNET_RPC = "https://rpc.hyperliquid.xyz/evm"
HYPEREVM_TESTNET_CHAIN_ID = 998
HYPEREVM_MAINNET_CHAIN_ID = 421614  # HyperEVM mainnet

# Use testnet by default
USE_TESTNET = True
RPC_URL = HYPEREVM_TESTNET_RPC if USE_TESTNET else HYPEREVM_MAINNET_RPC
CHAIN_ID = HYPEREVM_TESTNET_CHAIN_ID if USE_TESTNET else HYPEREVM_MAINNET_CHAIN_ID

log("=" * 80, Colors.BOLD)
log("║  🚀 HYPEREVM TOKEN DEPLOYMENT                                              ║", Colors.BOLD)
log("=" * 80, Colors.BOLD)
log("")

# Get private key from environment
PRIVATE_KEY = os.environ.get('PRIVATE_KEY')
if not PRIVATE_KEY:
    log("❌ Error: PRIVATE_KEY environment variable not set", Colors.FAIL)
    log("   Export it first: export PRIVATE_KEY='your_private_key'", Colors.WARNING)
    sys.exit(1)

# Ensure 0x prefix
if not PRIVATE_KEY.startswith('0x'):
    PRIVATE_KEY = '0x' + PRIVATE_KEY

# TBTC Token Contract (ERC20)
TBTC_CONTRACT_BYTECODE = "0x60806040523480156200001157600080fd5b506040518060400160405280600b81526020017f5465737420426974636f696e0000000000000000000000000000000000000000008152506040518060400160405280600481526020017f544254430000000000000000000000000000000000000000000000000000000081525081600390816200008f9190620004db565b508060049081620000a19190620004db565b505050620000c1620000b5620000c760201b60201c565b620000cf60201b60201c565b620005c2565b600033905090565b6000600560009054906101000a900473ffffffffffffffffffffffffffffffffffffffff16905081600560006101000a81548173ffffffffffffffffffffffffffffffffffffffff021916908373ffffffffffffffffffffffffffffffffffffffff1602179055508173ffffffffffffffffffffffffffffffffffffffff168173ffffffffffffffffffffffffffffffffffffffff167f8be0079c531659141344cd1fd0a4f28419497f9722a3daafe3b4186f6b6457e060405160405180910390a35050565b600081519050919050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052604160045260246000fd5b7f4e487b7100000000000000000000000000000000000000000000000000000000600052602260045260246000fd5b600060028204905060018216806200021d57607f821691505b602082108103620002335762000232620001d5565b5b50919050565b60008190508160005260206000209050919050565b60006020601f8301049050919050565b600082821b905092915050565b6000600883026200029d7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff826200025e565b620002a986836200025e565b95508019841693508086168417925050509392505050565b6000819050919050565b6000819050919050565b6000620002f6620002f0620002ea84620002c1565b620002cb565b620002c1565b9050919050565b6000819050919050565b6200031283620002d5565b6200032a6200032182620002fd565b8484546200026b565b825550505050565b600090565b6200034162000332565b6200034e81848462000307565b505050565b5b8181101562000376576200036a60008262000337565b60018101905062000354565b5050565b601f821115620003c5576200038f8162000239565b6200039a846200024e565b81016020851015620003aa578190505b620003c2620003b9856200024e565b83018262000353565b50505b505050565b600082821c905092915050565b6000620003ea60001984600802620003ca565b1980831691505092915050565b6000620004058383620003d7565b9150826002028217905092915050565b620004208262000195565b67ffffffffffffffff8111156200043c576200043b620001a0565b5b62000448825462000204565b620004558282856200037a565b600060209050601f8311600181146200048d576000841562000478578287015190505b620004848582620003f7565b865550620004f4565b601f1984166200049d8662000239565b60005b82811015620004c757848901518255600182019150602085019450602081019050620004a0565b86831015620004e75784890151620004e3601f891682620003d7565b8355505b6001600288020188555050505b505050505050565b6112f3806200050c6000396000f3fe608060405234801561001057600080fd5b50600436106100935760003560e01c8063715018a611610066578063715018a6146101425780638da5cb5b1461014c57806395d89b411461016a578063a9059cbb14610188578063dd62ed3e146101b857610093565b806306fdde0314610098578063095ea7b3146100b657806318160ddd146100e657806370a0823114610104575b600080fd5b6100a06101e8565b6040516100ad9190610e26565b60405180910390f35b6100d060048036038101906100cb9190610ee1565b61027a565b6040516100dd9190610f3c565b60405180910390f35b6100ee61029d565b6040516100fb9190610f66565b60405180910390f35b61011e60048036038101906101199190610f81565b6102a7565b60405161013d9190610f66565b60405180910390f35b61014a6102ef565b005b610154610303565b6040516101619190610fbd565b60405180910390f35b61017261032d565b60405161017f9190610e26565b60405180910390f35b6101a2600480360381019061019d9190610ee1565b6103bf565b6040516101af9190610f3c565b60405180910390f35b6101d260048036038101906101cd9190610fd8565b6103e2565b6040516101df9190610f66565b60405180910390f35b6060600380546101f790611047565b80601f016020809104026020016040519081016040528092919081815260200182805461022390611047565b80156102705780601f1061024557610100808354040283529160200191610270565b820191906000526020600020905b81548152906001019060200180831161025357829003601f168201915b5050505050905090565b600080610285610469565b9050610292818585610471565b600191505092915050565b6000600254905090565b60008060008373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff168152602001908152602001600020549050919050565b6102f761063a565b61030160006106b8565b565b6000600560009054906101000a900473ffffffffffffffffffffffffffffffffffffffff16905090565b60606004805461033c90611047565b80601f016020809104026020016040519081016040528092919081815260200182805461036890611047565b80156103b55780601f1061038a576101008083540402835291602001916103b5565b820191906000526020600020905b81548152906001019060200180831161039857829003601f168201915b5050505050905090565b6000806103ca610469565b90506103d781858561077e565b600191505092915050565b6000600160008473ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff16815260200190815260200160002060008373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff16815260200190815260200160002054905092915050565b600033905090565b600073ffffffffffffffffffffffffffffffffffffffff168373ffffffffffffffffffffffffffffffffffffffff16036104e0576040517f08c379a00000000000000000000000000000000000000000000000000000000081526004016104d7906110ea565b60405180910390fd5b600073ffffffffffffffffffffffffffffffffffffffff168273ffffffffffffffffffffffffffffffffffffffff160361054f576040517f08c379a00000000000000000000000000000000000000000000000000000000081526004016105469061117c565b60405180910390fd5b80600160008573ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff16815260200190815260200160002060008473ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff168152602001908152602001600020819055508173ffffffffffffffffffffffffffffffffffffffff168373ffffffffffffffffffffffffffffffffffffffff167f8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b9258360405161062d9190610f66565b60405180910390a3505050565b610642610469565b73ffffffffffffffffffffffffffffffffffffffff16610660610303565b73ffffffffffffffffffffffffffffffffffffffff16146106b6576040517f08c379a00000000000000000000000000000000000000000000000000000000081526004016106ad906111e8565b60405180910390fd5b565b6000600560009054906101000a900473ffffffffffffffffffffffffffffffffffffffff16905081600560006101000a81548173ffffffffffffffffffffffffffffffffffffffff021916908373ffffffffffffffffffffffffffffffffffffffff1602179055508173ffffffffffffffffffffffffffffffffffffffff168173ffffffffffffffffffffffffffffffffffffffff167f8be0079c531659141344cd1fd0a4f28419497f9722a3daafe3b4186f6b6457e060405160405180910390a35050565b600073ffffffffffffffffffffffffffffffffffffffff168373ffffffffffffffffffffffffffffffffffffffff16036107ed576040517f08c379a00000000000000000000000000000000000000000000000000000000081526004016107e49061127a565b60405180910390fd5b600073ffffffffffffffffffffffffffffffffffffffff168273ffffffffffffffffffffffffffffffffffffffff160361085c576040517f08c379a00000000000000000000000000000000000000000000000000000000081526004016108539061130c565b60405180910390fd5b610867838383610951565b60008060008573ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff168152602001908152602001600020549050818110156108ed576040517f08c379a00000000000000000000000000000000000000000000000000000000081526004016108e49061139e565b60405180910390fd5b8181036000808673ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff16815260200190815260200160002081905550816000808573ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff16815260200190815260200160002060008282540192505081905550505050565b505050565b600081519050919050565b600082825260208201905092915050565b60005b8381101561099057808201518184015260208101905061097557fd5b8381111561099f576000848401525b50505050565b6000601f19601f8301169050919050565b60006109c182610956565b6109cb8185610961565b93506109db818560208601610972565b6109e4816109a5565b840191505092915050565b60006020820190508181036000830152610a0981846109b6565b905092915050565b600080fd5b600073ffffffffffffffffffffffffffffffffffffffff82169050919050565b6000610a4182610a16565b9050919050565b610a5181610a36565b8114610a5c57600080fd5b50565b600081359050610a6e81610a48565b92915050565b6000819050919050565b610a8781610a74565b8114610a9257600080fd5b50565b600081359050610aa481610a7e565b92915050565b60008060408385031215610ac157610ac0610a11565b5b6000610acf85828601610a5f565b9250506020610ae085828601610a95565b9150509250929050565b60008115159050919050565b610aff81610aea565b82525050565b6000602082019050610b1a6000830184610af6565b92915050565b610b2981610a74565b82525050565b6000602082019050610b446000830184610b20565b92915050565b600060208284031215610b6057610b5f610a11565b5b6000610b6e84828501610a5f565b91505092915050565b610b8081610a36565b82525050565b6000602082019050610b9b6000830184610b77565b92915050565b60008060408385031215610bb857610bb7610a11565b5b6000610bc685828601610a5f565b9250506020610bd785828601610a5f565b9150509250929050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052602260045260246000fd5b60006002820490506001821680610c2857607f821691505b602082108103610c3b57610c3a610be1565b5b50919050565b7f45524332303a20617070726f76652066726f6d20746865207a65726f2061646460008201527f7265737300000000000000000000000000000000000000000000000000000000602082015250565b6000610c9d602483610961565b9150610ca882610c41565b604082019050919050565b60006020820190508181036000830152610ccc81610c90565b9050919050565b7f45524332303a20617070726f766520746f20746865207a65726f20616464726560008201527f7373000000000000000000000000000000000000000000000000000000000000602082015250565b6000610d2f602283610961565b9150610d3a82610cd3565b604082019050919050565b60006020820190508181036000830152610d5e81610d22565b9050919050565b7f4f776e61626c653a2063616c6c6572206973206e6f7420746865206f776e6572600082015250565b6000610d9b602083610961565b9150610da682610d65565b602082019050919050565b60006020820190508181036000830152610dca81610d8e565b9050919050565b7f45524332303a207472616e736665722066726f6d20746865207a65726f20616460008201527f6472657373000000000000000000000000000000000000000000000000000000602082015250565b6000610e2d602583610961565b9150610e3882610dd1565b604082019050919050565b60006020820190508181036000830152610e5c81610e20565b9050919050565b7f45524332303a207472616e7366657220746f20746865207a65726f206164647260008201527f6573730000000000000000000000000000000000000000000000000000000000602082015250565b6000610ebf602383610961565b9150610eca82610e63565b604082019050919050565b60006020820190508181036000830152610eee81610eb2565b9050919050565b7f45524332303a207472616e7366657220616d6f756e742065786365656473206260008201527f616c616e63650000000000000000000000000000000000000000000000000000602082015250565b6000610f51602683610961565b9150610f5c82610ef5565b604082019050919050565b60006020820190508181036000830152610f8081610f44565b905091905056fea264697066735822122084d8c07f7f6a424e8b3f5f38c0f0c0e8b8b7f7f7f7f7f7f7f7f7f7f7f7f7f764736f6c63430008110033"

TBTC_CONTRACT_ABI = [
    {
        "inputs": [],
        "stateMutability": "nonpayable",
        "type": "constructor"
    },
    {
        "inputs": [],
        "name": "name",
        "outputs": [{"internalType": "string", "name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "symbol",
        "outputs": [{"internalType": "string", "name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "decimals",
        "outputs": [{"internalType": "uint8", "name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "totalSupply",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "address", "name": "account", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "to", "type": "address"},
            {"internalType": "uint256", "name": "amount", "type": "uint256"}
        ],
        "name": "transfer",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "spender", "type": "address"},
            {"internalType": "uint256", "name": "amount", "type": "uint256"}
        ],
        "name": "approve",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "to", "type": "address"},
            {"internalType": "uint256", "name": "amount", "type": "uint256"}
        ],
        "name": "mint",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]

def deploy_contract():
    """Deploy TBTC contract to HyperEVM"""

    try:
        log(f"📡 Connecting to HyperEVM {'Testnet' if USE_TESTNET else 'Mainnet'}...", Colors.OKCYAN)
        log(f"   RPC: {RPC_URL}", Colors.OKCYAN)

        # Connect to HyperEVM
        w3 = Web3(Web3.HTTPProvider(RPC_URL, request_kwargs={'timeout': 60}))

        # Add PoA middleware if needed (for older web3 versions)
        if geth_poa_middleware:
            try:
                w3.middleware_onion.inject(geth_poa_middleware, layer=0)
            except:
                pass  # Not needed in newer versions

        if not w3.is_connected():
            log("❌ Failed to connect to HyperEVM", Colors.FAIL)
            return None

        log(f"✅ Connected to HyperEVM (Chain ID: {w3.eth.chain_id})", Colors.OKGREEN)

        # Load account
        account: LocalAccount = Account.from_key(PRIVATE_KEY)
        log(f"💼 Deployer Address: {account.address}", Colors.OKCYAN)

        # Check balance
        balance = w3.eth.get_balance(account.address)
        balance_eth = w3.from_wei(balance, 'ether')
        log(f"💰 Balance: {balance_eth} ETH", Colors.OKCYAN)

        if balance == 0:
            log("❌ No ETH for gas! Get testnet ETH from HyperEVM faucet", Colors.FAIL)
            return None

        log("")
        log("=" * 80, Colors.BOLD)
        log("║  DEPLOYING TBTC CONTRACT                                                   ║", Colors.BOLD)
        log("=" * 80, Colors.BOLD)
        log("")

        # Get nonce
        nonce = w3.eth.get_transaction_count(account.address)

        # Build transaction
        log("🔨 Building deployment transaction...", Colors.OKCYAN)

        transaction = {
            'from': account.address,
            'data': TBTC_CONTRACT_BYTECODE,
            'nonce': nonce,
            'chainId': w3.eth.chain_id,
            'gas': 2000000,
            'maxFeePerGas': w3.eth.gas_price,
            'maxPriorityFeePerGas': w3.to_wei(1, 'gwei'),
        }

        # Sign transaction
        log("✍️  Signing transaction...", Colors.OKCYAN)
        signed_txn = w3.eth.account.sign_transaction(transaction, PRIVATE_KEY)

        # Send transaction
        log("📤 Sending transaction...", Colors.OKCYAN)
        tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        log(f"   TX Hash: {tx_hash.hex()}", Colors.WARNING)

        # Wait for receipt
        log("⏳ Waiting for confirmation...", Colors.OKCYAN)
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)

        if tx_receipt.status == 1:
            contract_address = tx_receipt.contractAddress
            log("")
            log("=" * 80, Colors.OKGREEN)
            log("║  ✅ DEPLOYMENT SUCCESSFUL!                                                 ║", Colors.OKGREEN)
            log("=" * 80, Colors.OKGREEN)
            log("")
            log(f"📍 Contract Address: {contract_address}", Colors.OKGREEN)
            log(f"🔗 TX Hash: {tx_hash.hex()}", Colors.OKGREEN)
            log(f"⛽ Gas Used: {tx_receipt.gasUsed}", Colors.OKGREEN)
            log("")

            # Save deployment info
            deployment_info = {
                "network": "hyperevm-testnet" if USE_TESTNET else "hyperevm-mainnet",
                "chainId": w3.eth.chain_id,
                "rpcUrl": RPC_URL,
                "contracts": {
                    "TBTC": {
                        "address": contract_address,
                        "deploymentTx": tx_hash.hex(),
                        "deployer": account.address,
                        "gasUsed": tx_receipt.gasUsed
                    }
                }
            }

            with open('tbtc_hyperevm_deployment.json', 'w') as f:
                json.dump(deployment_info, f, indent=2)

            log("💾 Deployment info saved to: tbtc_hyperevm_deployment.json", Colors.OKCYAN)

            return contract_address

        else:
            log("❌ Deployment failed!", Colors.FAIL)
            return None

    except Exception as e:
        log(f"❌ Error: {str(e)}", Colors.FAIL)
        return None

if __name__ == "__main__":
    contract_address = deploy_contract()

    if contract_address:
        log("")
        log("=" * 80, Colors.BOLD)
        log("║  NEXT STEPS                                                                ║", Colors.BOLD)
        log("=" * 80, Colors.BOLD)
        log("")
        log(f"1. Add TBTC to your wallet:", Colors.OKCYAN)
        log(f"   Address: {contract_address}", Colors.OKGREEN)
        log(f"   Symbol: TBTC", Colors.OKGREEN)
        log(f"   Decimals: 18", Colors.OKGREEN)
        log("")
        log(f"2. View on HyperEVM Explorer:", Colors.OKCYAN)
        explorer = "https://explorer.hyperliquid-testnet.xyz" if USE_TESTNET else "https://explorer.hyperliquid.xyz"
        log(f"   {explorer}/address/{contract_address}", Colors.OKGREEN)
        log("")
        log("✨ TBTC is now live on HyperEVM! ✨", Colors.HEADER)
        sys.exit(0)
    else:
        sys.exit(1)
