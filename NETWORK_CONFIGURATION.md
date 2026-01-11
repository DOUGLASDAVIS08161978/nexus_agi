# Nexus AGI Network Configuration Guide

## Overview
This guide explains how to configure network access for the Nexus AGI autonomous mining system to connect to Bitcoin Testnet 4.

## Required Network Access

### Bitcoin Testnet 4 Connections
- **Port**: 48333 (Testnet 4 P2P)
- **Protocol**: TCP
- **Direction**: Outbound
- **Purpose**: Connect to Bitcoin testnet nodes

### Bitcoin RPC (if using local node)
- **Port**: 48332 (Testnet 4 RPC)
- **Protocol**: HTTP/HTTPS
- **Direction**: Local only (127.0.0.1)
- **Authentication**: Username/password or cookie

### Mempool API
- **Endpoint**: https://mempool.space/testnet4/api/
- **Protocol**: HTTPS
- **Purpose**: Transaction broadcasting and block exploration

## Network Setup Options

### Option 1: Local Bitcoin Testnet Node (Recommended)

Install and run Bitcoin Core in testnet mode:

```bash
# Install Bitcoin Core
wget https://bitcoincore.org/bin/bitcoin-core-26.0/bitcoin-26.0-x86_64-linux-gnu.tar.gz
tar xzf bitcoin-26.0-x86_64-linux-gnu.tar.gz
sudo install -m 0755 -o root -g root -t /usr/local/bin bitcoin-26.0/bin/*

# Create bitcoin.conf for Testnet 4
mkdir -p ~/.bitcoin
cat > ~/.bitcoin/bitcoin.conf <<EOF
# Testnet 4 Configuration
testnet4=1
server=1
rpcuser=nexusagi
rpcpassword=YOUR_SECURE_PASSWORD_HERE
rpcallowip=127.0.0.1
rpcport=48332

# Network settings
listen=1
port=48333
maxconnections=125

# Logging
debug=0
EOF

# Start Bitcoin daemon
bitcoind -testnet4 -daemon
```

### Option 2: Remote Testnet Node

Connect to a remote Bitcoin testnet node:

```python
# Configuration for remote node
BITCOIN_NODE = {
    'host': 'your-node-ip',
    'port': 48332,
    'rpc_user': 'your-username',
    'rpc_password': 'your-password',
    'testnet': True
}
```

### Option 3: Public Testnet APIs

Use public testnet services (limited functionality):
- BlockCypher Testnet API
- Mempool.space Testnet API
- Blockchain.info Testnet

## Firewall Configuration

### Ubuntu/Debian (UFW)
```bash
# Allow Bitcoin Testnet 4 outbound
sudo ufw allow out 48333/tcp comment 'Bitcoin Testnet 4'

# Allow RPC local only
sudo ufw allow from 127.0.0.1 to any port 48332
```

### IPTables
```bash
# Allow outbound Testnet 4
sudo iptables -A OUTPUT -p tcp --dport 48333 -j ACCEPT

# Allow local RPC
sudo iptables -A INPUT -i lo -p tcp --dport 48332 -j ACCEPT
```

## Security Considerations

⚠️ **IMPORTANT SECURITY NOTES:**

1. **Never expose RPC to the internet** - Always use localhost or VPN
2. **Use strong RPC passwords** - Generate with: `openssl rand -hex 32`
3. **Testnet only** - Never use real Bitcoin private keys in testing
4. **Monitor resources** - Bitcoin node requires ~50GB disk space
5. **Rate limiting** - Mempool APIs have rate limits, cache locally

## Integration with Nexus AGI

Update `nexus_agi_autonomous_daemon.py` with network capabilities:

```python
class BitcoinTestnetConnector:
    """Real Bitcoin Testnet 4 network connector"""

    def __init__(self, rpc_user, rpc_password, rpc_host='127.0.0.1', rpc_port=48332):
        self.rpc_url = f'http://{rpc_user}:{rpc_password}@{rpc_host}:{rpc_port}'

    def broadcast_transaction(self, raw_tx):
        """Broadcast transaction to real network"""
        response = requests.post(self.rpc_url, json={
            'jsonrpc': '1.0',
            'id': 'nexus_agi',
            'method': 'sendrawtransaction',
            'params': [raw_tx]
        })
        return response.json()

    def get_network_info(self):
        """Get current network statistics"""
        response = requests.post(self.rpc_url, json={
            'jsonrpc': '1.0',
            'id': 'nexus_agi',
            'method': 'getnetworkinfo',
            'params': []
        })
        return response.json()
```

## Testing Network Connectivity

```bash
# Test Bitcoin node connection
bitcoin-cli -testnet4 getnetworkinfo

# Test RPC connection
curl --user nexusagi:YOUR_PASSWORD \
  --data-binary '{"jsonrpc":"1.0","id":"test","method":"getblockchaininfo","params":[]}' \
  http://127.0.0.1:48332

# Test mempool.space API
curl https://mempool.space/testnet4/api/blocks/tip/height
```

## Monitoring

Monitor network activity:
```bash
# Watch Bitcoin node connections
watch -n 5 'bitcoin-cli -testnet4 getpeerinfo | grep addr'

# Monitor network traffic
sudo nethogs -d 5

# Check daemon logs
journalctl -u nexus-agi-miner -f
```

## Troubleshooting

### Cannot connect to Bitcoin node
1. Check if bitcoind is running: `ps aux | grep bitcoind`
2. Verify port is listening: `netstat -tlnp | grep 48332`
3. Check firewall rules: `sudo ufw status`

### RPC authentication failed
1. Verify credentials in `~/.bitcoin/bitcoin.conf`
2. Check RPC allowip settings
3. Try cookie authentication: `cat ~/.bitcoin/testnet4/.cookie`

### Network unreachable
1. Confirm outbound connections allowed
2. Check DNS resolution: `dig mempool.space`
3. Test with telnet: `telnet <node-ip> 48333`

## Production Deployment

For 24/7 autonomous operation:

1. **Dedicated server** with stable internet
2. **Redundant connections** (primary + backup)
3. **Monitoring alerts** for network failures
4. **Automatic reconnection** logic
5. **Transaction queue** for offline periods

---

**Note**: This configuration enables legitimate Bitcoin testnet operations. The autonomous daemon requires proper network access to function with real Bitcoin networks. All network access should be properly configured through system administration, not by "bypassing restrictions."
