#!/bin/bash

# Nexus AGI Bridge Deployment Launcher
# This script helps you deploy and interact with the bridge system

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Banner
echo -e "${CYAN}"
echo "═══════════════════════════════════════════════════════════════════════════"
echo "║                                                                         ║"
echo "║           🌉 NEXUS AGI BRIDGE - DEPLOYMENT LAUNCHER 🌉                  ║"
echo "║                                                                         ║"
echo "═══════════════════════════════════════════════════════════════════════════"
echo -e "${NC}"

# Check if we're in the right directory
if [ ! -f "deploy_metamask.html" ]; then
    echo -e "${RED}Error: deploy_metamask.html not found!${NC}"
    echo "Please run this script from /home/user/nexus_agi/"
    exit 1
fi

# Main menu
while true; do
    echo -e "\n${CYAN}Choose an option:${NC}\n"
    echo "  1) 🌐 Open MetaMask Deployment (Browser - RECOMMENDED)"
    echo "  2) 💻 Deploy via Hardhat CLI (Requires private key)"
    echo "  3) 🔍 Check Network Status"
    echo "  4) 📊 Run Bridge Interaction Dashboard"
    echo "  5) 📚 View Documentation"
    echo "  6) 🧪 Test Deployment on Local Network"
    echo "  7) ❌ Exit"
    echo ""
    read -p "Enter your choice (1-7): " choice

    case $choice in
        1)
            echo -e "\n${GREEN}Opening MetaMask deployment interface...${NC}"
            echo -e "${YELLOW}This will open in your default browser.${NC}"
            echo -e "${YELLOW}Make sure MetaMask is installed and unlocked!${NC}\n"

            # Try different methods to open browser
            if command -v xdg-open &> /dev/null; then
                xdg-open deploy_metamask.html
            elif command -v open &> /dev/null; then
                open deploy_metamask.html
            elif command -v start &> /dev/null; then
                start deploy_metamask.html
            else
                echo -e "${YELLOW}Could not auto-open browser. Please manually open:${NC}"
                echo "file://$(pwd)/deploy_metamask.html"
            fi

            echo -e "\n${CYAN}Instructions:${NC}"
            echo "1. Click 'Connect MetaMask' in the browser"
            echo "2. Approve the connection"
            echo "3. Click 'Deploy All (Recommended)'"
            echo "4. Approve each transaction in MetaMask"
            echo ""
            ;;

        2)
            echo -e "\n${YELLOW}Checking configuration...${NC}"

            # Check if private key is set
            if grep -q "PRIVATE_KEY=your_private_key_here" .env 2>/dev/null; then
                echo -e "${RED}❌ Private key not configured!${NC}\n"
                echo -e "${YELLOW}You need to add your private key to .env file:${NC}"
                echo "nano .env"
                echo ""
                echo -e "${YELLOW}Find the line:${NC}"
                echo "PRIVATE_KEY=your_private_key_here_without_0x_prefix"
                echo ""
                echo -e "${YELLOW}Replace with your actual 64-character private key${NC}"
                echo -e "${RED}⚠️  ONLY USE A TESTNET WALLET!${NC}\n"
                read -p "Press Enter to continue..."
            else
                echo -e "${GREEN}✅ Private key is set${NC}"
                echo -e "\n${CYAN}Choose network:${NC}"
                echo "1) Sepolia Testnet (FREE)"
                echo "2) Polygon Mainnet (requires MATIC)"
                echo "3) Cancel"
                read -p "Enter choice (1-3): " network_choice

                case $network_choice in
                    1)
                        echo -e "\n${GREEN}Deploying to Sepolia testnet...${NC}\n"
                        npx hardhat run scripts/deploy_testnet_bridge.js --network sepolia
                        ;;
                    2)
                        echo -e "\n${YELLOW}⚠️  WARNING: Deploying to Polygon MAINNET${NC}"
                        echo -e "${YELLOW}This requires real MATIC and will cost money!${NC}"
                        read -p "Are you sure? (yes/no): " confirm
                        if [ "$confirm" = "yes" ]; then
                            echo -e "\n${GREEN}Deploying to Polygon mainnet...${NC}\n"
                            npx hardhat run scripts/deploy_testnet_bridge.js --network polygon
                        else
                            echo "Deployment cancelled."
                        fi
                        ;;
                    3)
                        echo "Cancelled."
                        ;;
                esac
            fi
            ;;

        3)
            echo -e "\n${GREEN}Checking network status...${NC}\n"
            node scripts/check_network_status.js
            ;;

        4)
            echo -e "\n${GREEN}Running bridge interaction dashboard...${NC}\n"
            node scripts/interact_bridge.js
            ;;

        5)
            echo -e "\n${CYAN}Available Documentation:${NC}\n"
            echo "1) Quick Start Summary"
            echo "2) Testnet Bridge Deployment Guide"
            echo "3) MetaMask Deployment Guide"
            echo "4) API Integration Status"
            echo "5) Back to main menu"
            read -p "Choose document (1-5): " doc_choice

            case $doc_choice in
                1) cat QUICK_START_SUMMARY.md | less ;;
                2) cat TESTNET_BRIDGE_DEPLOYMENT_GUIDE.md | less ;;
                3) cat METAMASK_DEPLOYMENT_GUIDE.md | less ;;
                4) cat API_INTEGRATION_STATUS.md | less ;;
                5) continue ;;
            esac
            ;;

        6)
            echo -e "\n${GREEN}Testing deployment on local Hardhat network...${NC}"
            echo -e "${YELLOW}This is a simulation - no real tokens or gas fees${NC}\n"
            npx hardhat run scripts/deploy_testnet_bridge.js --network hardhat
            ;;

        7)
            echo -e "\n${GREEN}Goodbye!${NC}\n"
            exit 0
            ;;

        *)
            echo -e "${RED}Invalid option. Please choose 1-7.${NC}"
            ;;
    esac

    echo ""
    read -p "Press Enter to return to main menu..."
done
