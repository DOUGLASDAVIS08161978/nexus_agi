#!/bin/bash

# NEXUS AGI - Monad Regtest Bridge Launcher
# Author: Douglas Shane Davis & Claude
# Date: 2026-01-22

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Banner
clear
echo -e "${CYAN}"
cat << "EOF"
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║         🌉 NEXUS AGI - BITCOIN REGTEST TO MONAD TESTNET BRIDGE 🌉           ║
║                                                                              ║
║  Mine Bitcoin on regtest → Bridge to Monad testnet → Receive WBTC          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}\n"

# Configuration Display
echo -e "${MAGENTA}📋 CURRENT CONFIGURATION${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  Receiving Address: ${GREEN}0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771${NC}"
echo -e "  Monad Testnet RPC: ${GREEN}https://testnet-rpc.monad.xyz${NC}"
echo -e "  Chain ID: ${GREEN}10143${NC}"
echo -e "  Block Reward: ${GREEN}50 BTC${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

# Check if Python is installed
echo -e "${BLUE}🔍 Checking system requirements...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed!${NC}"
    echo -e "${YELLOW}Please install Python 3 and try again.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python 3: $(python3 --version)${NC}"

# Check if web3 is installed
if ! python3 -c "import web3" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  web3.py not installed. Installing now...${NC}"
    pip3 install web3 eth-account --quiet
    echo -e "${GREEN}✅ web3.py installed${NC}"
else
    echo -e "${GREEN}✅ web3.py installed${NC}"
fi

# Check if script exists
if [ ! -f "monad_regtest_bridge.py" ]; then
    echo -e "${RED}❌ monad_regtest_bridge.py not found!${NC}"
    echo -e "${YELLOW}Please ensure you're in the correct directory.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Bridge script found${NC}\n"

# Main menu
while true; do
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${MAGENTA}🎮 MAIN MENU${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "  ${GREEN}1)${NC} 🚀 Run Full Mining & Bridge Session (Default: 10 blocks)"
    echo -e "  ${GREEN}2)${NC} ⚡ Quick Test (5 blocks, fast)"
    echo -e "  ${GREEN}3)${NC} 💪 Extended Session (20 blocks)"
    echo -e "  ${GREEN}4)${NC} 🎯 Custom Parameters"
    echo -e "  ${GREEN}5)${NC} 💰 Check Balance Only"
    echo -e "  ${GREEN}6)${NC} 📊 View Last Session Results"
    echo -e "  ${GREEN}7)${NC} 📖 View Documentation"
    echo -e "  ${GREEN}8)${NC} 🔧 Get Testnet Funds Info"
    echo -e "  ${GREEN}9)${NC} ❌ Exit"
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    read -p "$(echo -e ${YELLOW}Enter your choice [1-9]:${NC} )" choice

    case $choice in
        1)
            echo -e "\n${GREEN}🚀 Starting full mining & bridge session...${NC}\n"
            python3 monad_regtest_bridge.py
            ;;

        2)
            echo -e "\n${GREEN}⚡ Starting quick test session (5 blocks)...${NC}\n"
            python3 -c "
from monad_regtest_bridge import RegtestToMonadSystem

system = RegtestToMonadSystem(
    monad_private_key='c411a4d4365560753ef3ceceac1652ec89240704346bf58ad900d65574f541c9',
    wbtc_receiving_address='0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771'
)

session = system.run_mining_and_bridge_session(
    blocks_to_mine=5,
    bridge_every_n_blocks=1
)

system.export_session(session)
system.check_receiving_address_balance()
"
            ;;

        3)
            echo -e "\n${GREEN}💪 Starting extended session (20 blocks)...${NC}\n"
            python3 -c "
from monad_regtest_bridge import RegtestToMonadSystem

system = RegtestToMonadSystem(
    monad_private_key='c411a4d4365560753ef3ceceac1652ec89240704346bf58ad900d65574f541c9',
    wbtc_receiving_address='0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771'
)

session = system.run_mining_and_bridge_session(
    blocks_to_mine=20,
    bridge_every_n_blocks=3
)

system.export_session(session)
system.check_receiving_address_balance()
"
            ;;

        4)
            echo -e "\n${CYAN}🎯 Custom Parameters${NC}"
            echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            read -p "$(echo -e ${YELLOW}Number of blocks to mine:${NC} )" blocks
            read -p "$(echo -e ${YELLOW}Bridge every N blocks:${NC} )" bridge_freq

            echo -e "\n${GREEN}Starting custom session...${NC}"
            echo -e "  Blocks: $blocks"
            echo -e "  Bridge frequency: Every $bridge_freq block(s)\n"

            python3 -c "
from monad_regtest_bridge import RegtestToMonadSystem

system = RegtestToMonadSystem(
    monad_private_key='c411a4d4365560753ef3ceceac1652ec89240704346bf58ad900d65574f541c9',
    wbtc_receiving_address='0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771'
)

session = system.run_mining_and_bridge_session(
    blocks_to_mine=$blocks,
    bridge_every_n_blocks=$bridge_freq
)

system.export_session(session)
system.check_receiving_address_balance()
"
            ;;

        5)
            echo -e "\n${GREEN}💰 Checking balance...${NC}\n"
            python3 -c "
from monad_regtest_bridge import MonadTestnetBridge

bridge = MonadTestnetBridge(
    private_key='c411a4d4365560753ef3ceceac1652ec89240704346bf58ad900d65574f541c9',
    receiving_address='0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771'
)

print('\n' + '='*80)
print('💰 MONAD TESTNET BALANCE CHECK')
print('='*80)
print(f'Address: {bridge.receiving_address}')
balance = bridge.get_balance()
print(f'Balance: {balance:.8f} ETH')
print('='*80 + '\n')
"
            ;;

        6)
            echo -e "\n${GREEN}📊 Last session results:${NC}\n"
            latest_session=$(ls -t monad_regtest_session_*.json 2>/dev/null | head -1)
            if [ -z "$latest_session" ]; then
                echo -e "${YELLOW}No session files found.${NC}"
                echo -e "${YELLOW}Run a mining session first!${NC}\n"
            else
                echo -e "${CYAN}File: $latest_session${NC}\n"
                python3 -c "
import json
with open('$latest_session', 'r') as f:
    data = json.load(f)

print('Session ID:', data['session_id'])
print('Timestamp:', data['timestamp'])
print('Duration:', data['duration_seconds'], 'seconds')
print()
print('MINING:')
print('  Blocks Mined:', data['mining']['blocks_mined'])
print('  Total BTC:', data['mining']['total_btc'])
print()
print('BRIDGING:')
print('  Transactions:', data['bridging']['transactions'])
print('  Total WBTC:', data['bridging']['total_wbtc'])
print('  Receiving:', data['bridging']['receiving_address'])
print()
print('Bridge Transactions:')
for bridge in data['bridging']['bridges']:
    print(f'  • {bridge[\"bridge_id\"]} - {bridge[\"wbtc_amount\"]} WBTC - Status: {bridge[\"status\"]}')
    if bridge['status'] == 'CONFIRMED':
        print(f'    TX: {bridge[\"monad_txid\"]}')
"
            fi
            ;;

        7)
            echo -e "\n${GREEN}📖 Opening documentation...${NC}\n"
            if [ -f "MONAD_BRIDGE_GUIDE.md" ]; then
                if command -v less &> /dev/null; then
                    less MONAD_BRIDGE_GUIDE.md
                else
                    cat MONAD_BRIDGE_GUIDE.md
                fi
            else
                echo -e "${YELLOW}Documentation not found!${NC}"
            fi
            ;;

        8)
            echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            echo -e "${MAGENTA}💵 HOW TO GET MONAD TESTNET FUNDS${NC}"
            echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            echo ""
            echo -e "${GREEN}Your Receiving Address:${NC}"
            echo -e "${YELLOW}0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771${NC}"
            echo ""
            echo -e "${GREEN}Monad Testnet Faucets:${NC}"
            echo -e "  1. Official Monad Faucet: ${BLUE}https://faucet.monad.xyz${NC}"
            echo -e "  2. Monad Testnet Portal: ${BLUE}https://testnet.monad.xyz${NC}"
            echo -e "  3. Community Faucets: Check Monad Discord/Telegram"
            echo ""
            echo -e "${GREEN}Alternative Methods:${NC}"
            echo -e "  • Bridge from other testnets (if supported)"
            echo -e "  • Request from Monad community channels"
            echo -e "  • Check Monad documentation for latest faucet info"
            echo ""
            echo -e "${YELLOW}Note:${NC} You need testnet ETH for gas fees to bridge successfully."
            echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            ;;

        9)
            echo -e "\n${GREEN}👋 Goodbye!${NC}\n"
            exit 0
            ;;

        *)
            echo -e "${RED}Invalid choice. Please select 1-9.${NC}"
            ;;
    esac

    echo ""
    read -p "$(echo -e ${CYAN}Press Enter to return to main menu...${NC})"
    echo ""
done
