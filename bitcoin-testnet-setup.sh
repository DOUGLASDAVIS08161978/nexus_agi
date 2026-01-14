#!/bin/bash
# NEXUS AGI - Bitcoin Testnet Setup Script
# This script manages the Bitcoin testnet daemon for NEXUS AGI

set -e

BITCOIN_DIR="$HOME/.bitcoin"
BITCOIN_CONF="$BITCOIN_DIR/bitcoin.conf"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  NEXUS AGI - Bitcoin Testnet Manager  ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to start daemon
start_daemon() {
    echo -e "${GREEN}Starting Bitcoin testnet daemon...${NC}"
    if pgrep -x "bitcoind" > /dev/null; then
        echo -e "${RED}Bitcoin daemon is already running!${NC}"
        exit 1
    fi
    bitcoind -daemon -testnet
    echo -e "${GREEN}Bitcoin daemon started successfully!${NC}"
    echo "Waiting for daemon to initialize..."
    sleep 5
    show_status
}

# Function to stop daemon
stop_daemon() {
    echo -e "${GREEN}Stopping Bitcoin testnet daemon...${NC}"
    bitcoin-cli -testnet stop
    echo -e "${GREEN}Bitcoin daemon stopped.${NC}"
}

# Function to show status
show_status() {
    echo -e "${BLUE}Network Info:${NC}"
    bitcoin-cli -testnet getnetworkinfo | grep -E '"version"|"subversion"|"connections"' || true
    echo ""
    echo -e "${BLUE}Blockchain Info:${NC}"
    bitcoin-cli -testnet getblockchaininfo | grep -E '"chain"|"blocks"|"headers"|"verificationprogress"' || true
}

# Function to show mining info
mining_info() {
    echo -e "${BLUE}Mining Information:${NC}"
    bitcoin-cli -testnet getmininginfo
}

# Function to generate blocks (testnet mining)
mine_blocks() {
    BLOCKS=${1:-1}
    ADDRESS=$(bitcoin-cli -testnet getnewaddress)
    echo -e "${GREEN}Mining $BLOCKS block(s) to address: $ADDRESS${NC}"
    bitcoin-cli -testnet generatetoaddress $BLOCKS "$ADDRESS"
}

# Main command processing
case "${1:-start}" in
    start)
        start_daemon
        ;;
    stop)
        stop_daemon
        ;;
    status)
        show_status
        ;;
    restart)
        stop_daemon
        sleep 2
        start_daemon
        ;;
    mine)
        mine_blocks "${2:-1}"
        ;;
    mining-info)
        mining_info
        ;;
    help)
        echo "Usage: $0 {start|stop|restart|status|mine [blocks]|mining-info|help}"
        echo ""
        echo "Commands:"
        echo "  start        - Start the Bitcoin testnet daemon"
        echo "  stop         - Stop the Bitcoin testnet daemon"
        echo "  restart      - Restart the daemon"
        echo "  status       - Show network and blockchain status"
        echo "  mine [n]     - Mine n blocks (default: 1)"
        echo "  mining-info  - Show mining information"
        echo "  help         - Show this help message"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
