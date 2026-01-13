#!/bin/bash
###############################################################################
# NEXUS AGI - Quick Deployment Script for Unrestricted Network
# Run this on your local machine, VPS, or cloud server to check wallet balances
###############################################################################

set -e

echo "================================================================================"
echo "           NEXUS AGI - QUICK DEPLOYMENT & WALLET BALANCE CHECKER"
echo "================================================================================"
echo

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Python 3 is installed
echo -e "${BLUE}[1/4] Checking Python installation...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed!${NC}"
    echo "Please install Python 3.8 or higher:"
    echo "  Ubuntu/Debian: sudo apt-get install python3 python3-pip"
    echo "  CentOS/RHEL:   sudo yum install python3 python3-pip"
    echo "  macOS:         brew install python3"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo -e "${GREEN}✅ Python ${PYTHON_VERSION} found${NC}"
echo

# Check if pip is installed
echo -e "${BLUE}[2/4] Checking pip installation...${NC}"
if ! command -v pip3 &> /dev/null; then
    echo -e "${YELLOW}⚠️  pip3 not found, installing...${NC}"
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py --user
    rm get-pip.py
fi
echo -e "${GREEN}✅ pip3 found${NC}"
echo

# Install dependencies
echo -e "${BLUE}[3/4] Installing Python dependencies...${NC}"
echo "This may take a few minutes..."
echo

# Check if dependencies are already installed
if python3 -c "import requests, numpy, networkx" 2>/dev/null; then
    echo -e "${GREEN}✅ Core dependencies already installed${NC}"
else
    echo "Installing: requests numpy networkx bitcoinlib web3..."
    pip3 install --user --quiet requests numpy networkx bitcoinlib web3 || {
        echo -e "${RED}❌ Failed to install dependencies${NC}"
        echo "Trying with sudo..."
        sudo pip3 install requests numpy networkx bitcoinlib web3
    }
    echo -e "${GREEN}✅ Dependencies installed successfully${NC}"
fi
echo

# Check network connectivity
echo -e "${BLUE}[4/4] Testing network connectivity...${NC}"
if ! ping -c 1 8.8.8.8 &> /dev/null; then
    echo -e "${RED}❌ No internet connection detected${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Network connection active${NC}"
echo

# Run deployment test
echo "================================================================================"
echo "           RUNNING WALLET BALANCE CHECK & DEPLOYMENT TEST"
echo "================================================================================"
echo

python3 deploy_unrestricted_network.py

EXIT_CODE=$?

echo
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ DEPLOYMENT SUCCESSFUL${NC}"
else
    echo -e "${YELLOW}⚠️  DEPLOYMENT COMPLETED WITH WARNINGS${NC}"
fi
echo "================================================================================"
echo
echo "Report files created:"
ls -lh deployment_report_*.json 2>/dev/null | tail -1 | awk '{print "  📄 " $9 " (" $5 ")"}'
echo
echo "To view the full report:"
echo "  cat \$(ls -t deployment_report_*.json | head -1)"
echo
echo "To check balances again:"
echo "  python3 deploy_unrestricted_network.py"
echo
echo "For continuous monitoring (every 5 minutes):"
echo "  watch -n 300 python3 deploy_unrestricted_network.py"
echo

exit $EXIT_CODE
