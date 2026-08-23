# Bitcoin Wallet Device & Network Tracking System

**Author:** Douglas Shane Davis & Claude
**Date:** 2025-12-18
**Version:** 1.0

## Overview

This system tracks device ID and network information for all Bitcoin wallet addresses used in the NexusAGI mining simulator. Each wallet is associated with detailed device specifications and network configuration data.

## Features

- ✅ **Device ID Tracking** - Unique device identifiers for each wallet
- ✅ **Network Information** - Complete network configuration (IP, MAC, hostname, etc.)
- ✅ **Device Specifications** - Hardware details including quantum capabilities
- ✅ **Mining Statistics** - Blocks mined and BTC earned per wallet
- ✅ **Quick Lookup** - Fast wallet information retrieval
- ✅ **JSON Storage** - Persistent data storage in structured format

## Files

### Core System Files

1. **wallet_device_tracker.py** - Main tracking system
   - Initialize and manage wallet tracking
   - Generate device IDs and network info
   - Save/load tracking data
   - Generate detailed reports

2. **wallet_tracking_data.json** - Data storage
   - Contains all wallet tracking information
   - Updated automatically by the tracker
   - Human-readable JSON format

3. **wallet_lookup.py** - Quick lookup tool
   - Command-line interface for wallet queries
   - Summary and list views
   - Fast information retrieval

## Tracked Wallets

The system currently tracks **6 wallet addresses**:

| # | Wallet Address | Device ID | Purpose |
|---|----------------|-----------|---------|
| 1 | bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh | MINER-72FA85432F2C7D55 | Primary mining wallet (Segwit) |
| 2 | 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa | MINER-9360688A3FD55626 | Legacy wallet (Genesis block style) |
| 3 | 3J98t1WpEZ73CNmYviecrnyiWrnqRhWNLy | MINER-6D673873E5902FC8 | P2SH wallet |
| 4 | **bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass** | **MINER-ADCC701CC7BC6D57** | **TARGET WALLET - Consolidation destination** |
| 5 | bc1q2m6w8au8yrjp85m7rqlcxqx9ypfla6h8st8u78 | MINER-885ED24A7AE5696E | Pool worker wallet (Slush Pool) |
| 6 | bc1q8z6z78dy5squapjpkeruem98jcezsw37hnae6qjyhxma6jmxyn6qsmqxce | MINER-1572039564FDAFC1 | Intermediate consolidation wallet |

## Usage

### Initialize Tracking System

```bash
python3 wallet_device_tracker.py
```

This will:
- Register all wallet addresses
- Generate device IDs and network info
- Create wallet_tracking_data.json
- Display detailed reports

### Lookup Specific Wallet

```bash
# Look up the target wallet
python3 wallet_lookup.py bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass
```

### View All Wallets Summary

```bash
python3 wallet_lookup.py --summary
```

### List All Tracked Wallets

```bash
python3 wallet_lookup.py --list
```

### Help

```bash
python3 wallet_lookup.py --help
```

## Example Output

### Target Wallet (bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass)

```
📍 WALLET: bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass

🖥️  DEVICE INFORMATION:
   Device ID:          MINER-ADCC701CC7BC6D57
   Device Name:        NexusMiner-003
   Type:               High-Performance ASIC Miner
   Manufacturer:       Bitmain
   Model:              Antminer S19 Pro
   Hash Rate:          120 TH/s
   Quantum Qubits:     16
   Quantum Enabled:    True

🌐 NETWORK INFORMATION:
   IP Address:         192.168.13.103
   MAC Address:        5d:21:95:ec:fc:8f
   Hostname:           miner-003.nexus-agi.local
   Port:               8333
   Protocol:           stratum+tcp
   Bandwidth:          10000 Mbps
   Latency:            26.48 ms
   ISP:                Comcast
   Geolocation:        Austin, TX

📊 MINING STATISTICS:
   Blocks Mined:       20
   BTC Earned:         160.00000000 BTC
   Status:             active
```

## Data Structure

Each wallet record contains:

### Device Information
- Device ID (unique identifier)
- Device name and type
- Manufacturer and model
- CPU, GPU, RAM specifications
- Hash rate and power consumption
- Operating system and firmware
- Quantum computing capabilities (qubit count)

### Network Information
- IP address (IPv4)
- MAC address
- Hostname
- Port number
- Network protocol
- DNS server and gateway
- Bandwidth and latency
- ISP and geolocation

### Mining Statistics
- Registration date
- Last active timestamp
- Total blocks mined
- Total BTC earned
- Current status
- Custom notes

## Integration with Mining Simulator

The tracking system integrates with:

1. **infinite_miner.py** - Main mining simulator
   - DOUGLAS_WALLETS array
   - Block mining and reward distribution

2. **bitcoin_validator_consolidator.py** - Blockchain validator
   - TARGET_WALLET configuration
   - Reward consolidation tracking

3. **pool_config.json** - Mining pool configuration
   - Pool worker wallet tracking
   - Stratum protocol settings

4. **quantum_mining_audit.json** - Audit records
   - Quantum-enhanced mining data
   - Device type correlation

## API / Programmatic Access

You can use the `WalletDeviceTracker` class in your Python scripts:

```python
from wallet_device_tracker import WalletDeviceTracker

# Load tracker
tracker = WalletDeviceTracker()

# Get wallet info
wallet_info = tracker.get_wallet_info("bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass")

# Access device ID
device_id = wallet_info.device_info.device_id
print(f"Device ID: {device_id}")

# Access network info
ip_address = wallet_info.network_info.ip_address
print(f"IP Address: {ip_address}")

# Update statistics
tracker.update_wallet_stats("bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass", blocks=1, btc=6.25)
tracker.save_data()
```

## Statistics

**Current System Totals:**
- Total Wallets Tracked: 6
- Total Blocks Mined: 86
- Total BTC Earned: 576.25 BTC
- Average Hash Rate: ~94 TH/s per device
- Quantum-Enhanced Devices: 6/6 (100%)

## Security Notes

- Device IDs are generated using SHA-256 hashing
- MAC addresses are deterministically generated from wallet addresses
- Network information represents simulated mining rig configurations
- All data is stored locally in JSON format

## Future Enhancements

Potential improvements:
- Real-time wallet monitoring
- Integration with live blockchain data
- Device health monitoring
- Network performance analytics
- Automated alert system
- Web dashboard interface
- Historical statistics tracking
- Export to CSV/Excel formats

## Support

For questions or issues, contact Douglas Shane Davis or refer to the main NexusAGI documentation.

---

**Last Updated:** 2025-12-18 04:39:03
**System Version:** 1.0
**Python Version:** 3.x required
