# 📜 NEXUS AGI - SMART CONTRACT INTERACTION GUIDE

## Overview

Read-only smart contract interaction across all 8 EVM networks using `eth_call`. Query contract state, read token data, and interact with deployed contracts **without gas costs**.

### Supported Networks

- ✅ Ethereum Mainnet
- ✅ Ethereum Sepolia
- ✅ Polygon Mainnet
- ✅ Polygon Mumbai
- ✅ Arbitrum One
- ✅ Avalanche C-Chain
- ✅ Base Mainnet
- ✅ BNB Smart Chain

---

## 🚀 Quick Start

### Basic Contract Call

```python
from tools.contract_interactor import ContractInteractor

# Initialize interactor
interactor = ContractInteractor()

# Make eth_call
result = interactor.eth_call(
    network_name="ethereum_mainnet",
    contract_address="0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",  # WBTC
    data="0x18160ddd"  # totalSupply()
)

print(f"Total Supply (hex): {result}")
# Output: 0x00000000000000000000000000000000000000000000000000049e1b2d536f60
```

### Get Current Block Number

```python
from tools.contract_interactor import ContractInteractor

interactor = ContractInteractor()

# Get latest block on Ethereum
block = interactor.get_block_number("ethereum_mainnet")
print(f"Current block: {block:,}")
# Output: Current block: 18,900,000
```

### Get ETH Balance

```python
# Get balance for any address
balance = interactor.get_balance(
    "ethereum_mainnet",
    "0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771"
)

print(f"Balance: {balance:.4f} ETH")
# Output: Balance: 1.2345 ETH
```

---

## 💰 ERC-20 Token Interaction

### Get Token Information

```python
from tools.contract_interactor import ERC20Reader

erc20 = ERC20Reader()

# Get WBTC info on Ethereum
wbtc_info = erc20.get_token_info(
    network_name="ethereum_mainnet",
    token_address="0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"
)

print(f"Name: {wbtc_info['name']}")           # Wrapped BTC
print(f"Symbol: {wbtc_info['symbol']}")       # WBTC
print(f"Decimals: {wbtc_info['decimals']}")   # 8
print(f"Total Supply: {wbtc_info['total_supply']:,.2f}")  # 152,345.67 WBTC
```

### Get Token Balance

```python
# Check WBTC balance for an address
balance = erc20.get_balance(
    network_name="ethereum_mainnet",
    token_address="0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",  # WBTC
    holder_address="0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771"
)

print(f"WBTC Balance: {balance:.8f}")
# Output: WBTC Balance: 0.12345678
```

### Get Token Allowance

```python
# Check how much spender can spend on behalf of owner
allowance = erc20.get_allowance(
    network_name="ethereum_mainnet",
    token_address="0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",  # WBTC
    owner_address="0xOwnerAddress",
    spender_address="0xSpenderAddress"
)

print(f"Allowance: {allowance:.8f} WBTC")
```

---

## 🌐 Multi-Network Queries

### Query Token Across Networks

```python
from tools.contract_interactor import MultiNetworkTokenReader

multi_reader = MultiNetworkTokenReader()

# WBTC addresses on different networks
wbtc_addresses = {
    "ethereum_mainnet": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
    "polygon_mainnet": "0x1BFD67037B42Cf73acF2047067bd4F2C47D9BfD6",
    "arbitrum_mainnet": "0x2f2a2543B76A4166549F7aaB2e75Bef0aefC5B0f",
    "avalanche_mainnet": "0x50b7545627a5162F82A992c33b87aDc75187B218",
    "base_mainnet": "0x0555E30da8f98308EdB960aa94C0Db47230d2B9c"
}

# Get token info across all networks
results = multi_reader.get_token_across_networks(wbtc_addresses)

# Output:
# ================================================================================
# MULTI-NETWORK TOKEN QUERY
# ================================================================================
#
# 🔍 Querying ethereum_mainnet...
#    Token: 0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599
#    ✅ Wrapped BTC (WBTC)
#    Decimals: 8
#    Total Supply: 152,345.67
#
# 🔍 Querying polygon_mainnet...
#    Token: 0x1BFD67037B42Cf73acF2047067bd4F2C47D9BfD6
#    ✅ Wrapped BTC (WBTC)
#    Decimals: 8
#    Total Supply: 523.45
# ...
```

### Get Balances Across Networks

```python
# Check balance across multiple networks
holder = "0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771"

balances = multi_reader.get_balance_across_networks(wbtc_addresses, holder)

# Output:
# ================================================================================
# TOKEN BALANCES FOR 0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771
# ================================================================================
#    ethereum_mainnet              0.1234 WBTC
#    polygon_mainnet               0.5678 WBTC
#    arbitrum_mainnet              1.2345 WBTC
#    avalanche_mainnet             0.0000 WBTC
#    base_mainnet                  0.9876 WBTC
# ================================================================================

# Calculate total across all networks
total_wbtc = sum(balances.values())
print(f"Total WBTC across all networks: {total_wbtc:.8f}")
```

---

## 🔧 Advanced Usage

### Check if Address is Contract

```python
interactor = ContractInteractor()

# Check if address is a contract
is_contract = interactor.is_contract(
    "ethereum_mainnet",
    "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"
)

print(f"Is contract: {is_contract}")
# Output: Is contract: True
```

### Get Contract Bytecode

```python
# Get deployed contract bytecode
bytecode = interactor.get_code(
    "ethereum_mainnet",
    "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"
)

print(f"Bytecode length: {len(bytecode)} characters")
print(f"First 100 chars: {bytecode[:100]}")
```

### Get Transaction Count (Nonce)

```python
# Get nonce for an address
nonce = interactor.get_transaction_count(
    "ethereum_mainnet",
    "0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771"
)

print(f"Transaction count: {nonce}")
# Output: Transaction count: 1234
```

---

## 📊 Function Selectors Reference

### ERC-20 Function Selectors

```python
# Common ERC-20 function selectors
SELECTORS = {
    "name()": "0x06fdde03",
    "symbol()": "0x95d89b41",
    "decimals()": "0x313ce567",
    "totalSupply()": "0x18160ddd",
    "balanceOf(address)": "0x70a08231",
    "allowance(address,address)": "0xdd62ed3e",
    "transfer(address,uint256)": "0xa9059cbb",
    "approve(address,uint256)": "0x095ea7b3",
    "transferFrom(address,address,uint256)": "0x23b872dd"
}
```

### Manual Function Call

```python
# Manually construct function call
contract_address = "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"

# totalSupply() - no parameters
data = "0x18160ddd"

result = interactor.eth_call("ethereum_mainnet", contract_address, data)
total_supply_wei = int(result, 16)
total_supply = total_supply_wei / 10**8  # WBTC has 8 decimals

print(f"Total Supply: {total_supply:,.2f} WBTC")
```

### Function Call with Parameters

```python
# balanceOf(address) - requires address parameter
holder_address = "0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771"

# Function selector for balanceOf
function_selector = "0x70a08231"

# Encode address (remove 0x, pad to 32 bytes = 64 hex chars)
address_param = holder_address[2:].lower().zfill(64)

# Concatenate
data = function_selector + address_param

result = interactor.eth_call("ethereum_mainnet", contract_address, data)
balance_wei = int(result, 16)
balance = balance_wei / 10**8

print(f"Balance: {balance:.8f} WBTC")
```

---

## 🎯 Use Cases

### 1. Portfolio Tracking

Track token holdings across all networks:

```python
from tools.contract_interactor import MultiNetworkTokenReader

# Your wallet address
wallet = "0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771"

# Token addresses across networks
tokens = {
    "WBTC": {
        "ethereum_mainnet": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
        "polygon_mainnet": "0x1BFD67037B42Cf73acF2047067bd4F2C47D9BfD6",
        "arbitrum_mainnet": "0x2f2a2543B76A4166549F7aaB2e75Bef0aefC5B0f"
    },
    "USDC": {
        "ethereum_mainnet": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        "polygon_mainnet": "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",
        "arbitrum_mainnet": "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"
    }
}

multi_reader = MultiNetworkTokenReader()

for token_name, addresses in tokens.items():
    print(f"\n{token_name} Holdings:")
    balances = multi_reader.get_balance_across_networks(addresses, wallet)
    total = sum(b for b in balances.values() if b is not None)
    print(f"Total: {total:,.4f} {token_name}")
```

### 2. Token Supply Monitoring

Monitor total supply across networks:

```python
erc20 = ERC20Reader()

# WBTC supply on different networks
networks = {
    "ethereum_mainnet": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
    "polygon_mainnet": "0x1BFD67037B42Cf73acF2047067bd4F2C47D9BfD6"
}

total_supply = 0

for network, address in networks.items():
    info = erc20.get_token_info(network, address)
    supply = info.get('total_supply', 0)
    total_supply += supply
    print(f"{network}: {supply:,.2f} WBTC")

print(f"\nTotal WBTC across all networks: {total_supply:,.2f}")
```

### 3. Contract Verification

Verify contract deployment across networks:

```python
interactor = ContractInteractor()

# Your deployed contract addresses
contract_addresses = {
    "ethereum_sepolia": "0xYourContractOnSepolia",
    "polygon_mumbai": "0xYourContractOnMumbai",
    "arbitrum_sepolia": "0xYourContractOnArbitrumSepolia"
}

for network, address in contract_addresses.items():
    is_contract = interactor.is_contract(network, address)
    block = interactor.get_block_number(network)

    status = "✅ Deployed" if is_contract else "❌ Not Found"
    print(f"{network}: {status} (Block: {block:,})")
```

### 4. Allowance Checker

Check token allowances before operations:

```python
erc20 = ERC20Reader()

# Check if contract can spend your tokens
owner = "0xYourAddress"
spender = "0xUniswapRouterOrOtherContract"
token = "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"  # WBTC

allowance = erc20.get_allowance("ethereum_mainnet", token, owner, spender)

if allowance > 0:
    print(f"✅ Approved: {allowance:,.8f} WBTC")
else:
    print(f"❌ No allowance - need to approve first")
```

---

## 🔗 Integration with Other Modules

### With Price Data

```python
from tools.contract_interactor import ERC20Reader
from tools.public_api_integrator import PublicAPIIntegrator

erc20 = ERC20Reader()
api = PublicAPIIntegrator()

# Get WBTC balance
balance = erc20.get_balance(
    "ethereum_mainnet",
    "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
    "0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771"
)

# Get WBTC price
price = api.get_token_price_by_symbol("WBTC")

# Calculate value
value_usd = balance * price

print(f"Balance: {balance:.8f} WBTC")
print(f"Price: ${price:,.2f}")
print(f"Value: ${value_usd:,.2f}")
```

### With Bridge System

```python
from tools.contract_interactor import MultiNetworkTokenReader
from tools.multi_network_bridge_orchestrator import MultiNetworkBridgeOrchestrator

# Check current balances
multi_reader = MultiNetworkTokenReader()

wbtc_addresses = {
    "ethereum_mainnet": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
    "arbitrum_mainnet": "0x2f2a2543B76A4166549F7aaB2e75Bef0aefC5B0f"
}

wallet = "0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771"
balances = multi_reader.get_balance_across_networks(wbtc_addresses, wallet)

# If imbalanced, execute bridge
if balances["ethereum_mainnet"] > balances["arbitrum_mainnet"] * 2:
    print("Imbalance detected - bridging recommended")

    orchestrator = MultiNetworkBridgeOrchestrator()
    # ... execute bridge
```

---

## 📝 Best Practices

### 1. Error Handling

Always check for None results:

```python
balance = erc20.get_balance(network, token, holder)

if balance is not None:
    print(f"Balance: {balance:.4f}")
else:
    print("Failed to fetch balance")
```

### 2. Network Availability

Test network connectivity first:

```python
from config.network_config import get_network_manager

manager = get_network_manager()

# Find working RPC
working_rpc = manager.find_working_rpc("ethereum_mainnet")

if working_rpc:
    # Proceed with contract calls
    pass
else:
    print("No working RPC found")
```

### 3. Decimal Handling

Always use token decimals for proper conversion:

```python
# Get token info first to know decimals
token_info = erc20.get_token_info(network, token_address)
decimals = token_info.get('decimals', 18)

# Use decimals for conversion
balance_wei = int(result, 16)
balance = balance_wei / (10 ** decimals)
```

### 4. Batch Queries

Query multiple items efficiently:

```python
# Instead of individual calls
# BAD:
for address in addresses:
    balance = erc20.get_balance(network, token, address)

# GOOD: Use MultiNetworkTokenReader
multi_reader = MultiNetworkTokenReader()
all_balances = multi_reader.get_balance_across_networks(tokens, holder)
```

---

## 🚨 Limitations

### Read-Only Operations

This module supports **read-only** operations:

✅ **Can do:**
- Read contract state
- Get token balances
- Check allowances
- Get total supply
- Verify deployments
- Get block numbers

❌ **Cannot do:**
- Send transactions
- Transfer tokens
- Approve spending
- Mint tokens
- Deploy contracts
- Change contract state

For write operations, you need:
1. Private key management
2. Transaction signing
3. Gas payment
4. Transaction broadcasting

---

## 📚 Function Reference

### ContractInteractor

```python
class ContractInteractor:
    def eth_call(network_name, contract_address, data, block="latest") -> str
    def get_block_number(network_name) -> int
    def get_balance(network_name, address) -> float
    def get_transaction_count(network_name, address) -> int
    def get_code(network_name, address) -> str
    def is_contract(network_name, address) -> bool
```

### ERC20Reader

```python
class ERC20Reader(ContractInteractor):
    def get_token_info(network_name, token_address) -> Dict
    def get_balance(network_name, token_address, holder_address) -> float
    def get_allowance(network_name, token_address, owner, spender) -> float
```

### MultiNetworkTokenReader

```python
class MultiNetworkTokenReader:
    def get_token_across_networks(token_addresses: Dict) -> Dict
    def get_balance_across_networks(token_addresses: Dict, holder) -> Dict
```

---

## 🎓 Examples

### Complete Portfolio Tracker

```python
#!/usr/bin/env python3
"""
Complete portfolio tracker across all networks
"""

from tools.contract_interactor import MultiNetworkTokenReader
from tools.public_api_integrator import PublicAPIIntegrator

def track_portfolio(wallet_address):
    # Initialize
    multi_reader = MultiNetworkTokenReader()
    api = PublicAPIIntegrator()

    # Define tokens
    tokens = {
        "WBTC": {
            "ethereum_mainnet": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
            "polygon_mainnet": "0x1BFD67037B42Cf73acF2047067bd4F2C47D9BfD6",
            "arbitrum_mainnet": "0x2f2a2543B76A4166549F7aaB2e75Bef0aefC5B0f"
        }
    }

    total_value = 0

    for token_name, addresses in tokens.items():
        print(f"\n{'='*60}")
        print(f"{token_name} Portfolio")
        print(f"{'='*60}")

        # Get balances
        balances = multi_reader.get_balance_across_networks(addresses, wallet_address)

        # Get price
        price = api.get_token_price_by_symbol(token_name)

        # Calculate
        total_tokens = sum(b for b in balances.values() if b is not None)
        value = total_tokens * price if price else 0

        print(f"\nTotal {token_name}: {total_tokens:,.8f}")
        print(f"Price: ${price:,.2f}")
        print(f"Value: ${value:,.2f}")

        total_value += value

    print(f"\n{'='*60}")
    print(f"Total Portfolio Value: ${total_value:,.2f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    track_portfolio("0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771")
```

---

## 🔄 Curl Command Equivalents

### Original Curl

```bash
curl -X POST http://localhost:8545 \
  -H "Content-Type: application/json" \
  --data '{
    "jsonrpc":"2.0",
    "method":"eth_call",
    "params":[{
      "to": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
      "data": "0x18160ddd"
    }, "latest"],
    "id":1
  }'
```

### Python Equivalent

```python
from tools.contract_interactor import ContractInteractor

interactor = ContractInteractor()

result = interactor.eth_call(
    network_name="ethereum_mainnet",
    contract_address="0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
    data="0x18160ddd",
    block="latest"
)

print(result)
```

---

## ✅ Summary

**Contract Interaction Module Capabilities:**

✅ Read contract state across 8 EVM networks
✅ Query ERC-20 token data (name, symbol, decimals, supply)
✅ Check token balances and allowances
✅ Multi-network queries in parallel
✅ Get block numbers and account balances
✅ Verify contract deployments
✅ Automatic RPC failover
✅ No gas costs (read-only)
✅ No API keys required

**Ready to use with:**
- All Nexus AGI tokens (NEX, HASH, BRG, etc.)
- WBTC across all networks
- Any ERC-20 token
- Custom contracts

---

**Last Updated**: 2026-01-18
**Version**: 1.0
**Networks Supported**: 8 EVM networks
**Module**: `tools/contract_interactor.py`
