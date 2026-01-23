#!/bin/bash

# ============================================================================
# 🌟 NEXUS AGI - BITCOIN NODE SETUP (myNode Integration)
# ============================================================================
#
# This script helps you set up a REAL Bitcoin node using myNode!
#
# Author: Douglas Shane Davis & Claude
# Date: 2026-01-23
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
WHITE='\033[1;37m'
NC='\033[0m'

print_banner() {
    clear
    echo -e "${CYAN}"
    cat << "EOF"
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              🌟 NEXUS AGI - BITCOIN NODE SETUP (myNode) 🌟                  ║
║                                                                              ║
║              Run Your Own REAL Bitcoin Node! 🎉                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

print_step() {
    echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${MAGENTA}$1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

ask_yes_no() {
    while true; do
        read -p "$(echo -e ${YELLOW}$1 [y/n]: ${NC})" yn
        case $yn in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer yes (y) or no (n).";;
        esac
    done
}

show_introduction() {
    print_banner
    print_step "🎓 WHAT IS myNode?"

    cat << 'EOF'
myNode is software that lets you run your OWN Bitcoin node!

╔═══════════════════════════════════════════════════════════════════════════╗
║                        WHY RUN YOUR OWN NODE?                             ║
╚═══════════════════════════════════════════════════════════════════════════╝

🔒 Privacy
   • Don't trust third-party servers
   • Verify transactions yourself
   • Nobody knows what you're checking

💪 Full Control
   • Own copy of the entire Bitcoin blockchain
   • Validate everything yourself
   • Support the Bitcoin network

🌟 Learn & Experiment
   • Understand how Bitcoin really works
   • Run on regtest, testnet, OR mainnet
   • Perfect for development

╔═══════════════════════════════════════════════════════════════════════════╗
║                         WHAT YOU'LL GET                                   ║
╚═══════════════════════════════════════════════════════════════════════════╝

✅ Bitcoin Core node (full node!)
✅ Lightning Network support
✅ Easy web interface
✅ Works with your existing bridge system
✅ Can switch between testnet and mainnet
✅ Professional-grade Bitcoin infrastructure

╔═══════════════════════════════════════════════════════════════════════════╗
║                         REQUIREMENTS                                      ║
╚═══════════════════════════════════════════════════════════════════════════╝

⚠️  IMPORTANT REQUIREMENTS:

💾 Storage: ~500GB minimum (for full blockchain)
   • Bitcoin blockchain is ~400GB
   • Growing every day
   • Need extra space for indices

💻 RAM: 2GB minimum (4GB+ recommended)

🌐 Internet: Unlimited or high cap
   • Initial sync downloads ~400GB
   • Ongoing: ~10-20GB per month

⏰ Time: 1-7 days for initial sync
   • Depends on internet speed
   • CPU power
   • Disk speed

EOF

    echo ""
    if ask_yes_no "Do you meet these requirements?"; then
        return 0
    else
        echo ""
        print_warning "Come back when you're ready!"
        echo ""
        print_info "For now, you can use:"
        echo "  • Regtest mode (local Bitcoin, no storage needed)"
        echo "  • Testnet mode (smaller blockchain, ~50GB)"
        echo ""
        exit 0
    fi
}

check_prerequisites() {
    print_step "🔍 CHECKING YOUR SYSTEM"

    # Check if running as root
    if [ "$EUID" -ne 0 ]; then
        print_error "This script needs to run with sudo!"
        echo ""
        echo "Please run:"
        echo "  sudo bash $0"
        echo ""
        exit 1
    fi

    print_success "Running with sudo ✓"

    # Check internet
    if ping -c 1 google.com &> /dev/null; then
        print_success "Internet connection ✓"
    else
        print_error "No internet connection!"
        exit 1
    fi

    # Check disk space
    local available=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')
    print_info "Available disk space: ${available}GB"

    if [ "$available" -lt 500 ]; then
        print_warning "You have less than 500GB available!"
        echo ""
        echo "Recommended: 500GB+ for full Bitcoin blockchain"
        echo "Minimum: 50GB for testnet"
        echo ""
        if ! ask_yes_no "Continue anyway?"; then
            exit 0
        fi
    else
        print_success "Sufficient disk space ✓"
    fi

    # Check RAM
    local ram_gb=$(free -g | awk '/^Mem:/{print $2}')
    print_info "Available RAM: ${ram_gb}GB"

    if [ "$ram_gb" -lt 2 ]; then
        print_warning "Less than 2GB RAM!"
        echo "This may be slow but should work."
        echo ""
    else
        print_success "Sufficient RAM ✓"
    fi

    echo ""
    print_success "System check complete!"
}

download_mynode() {
    print_step "📥 DOWNLOADING myNode SETUP SCRIPT"

    print_info "Downloading from official myNode repository..."
    echo ""

    if wget https://raw.githubusercontent.com/mynodebtc/mynode/latest_release/setup/setup_device.sh -O mynode_setup.sh; then
        print_success "Downloaded successfully!"
        chmod +x mynode_setup.sh
    else
        print_error "Download failed!"
        echo ""
        echo "Please check:"
        echo "  • Internet connection"
        echo "  • GitHub is accessible"
        echo ""
        exit 1
    fi
}

install_mynode() {
    print_step "🚀 INSTALLING myNode"

    print_warning "This will take 15-30 minutes!"
    echo ""
    echo "The installer will:"
    echo "  1. Install Bitcoin Core"
    echo "  2. Install Lightning Network"
    echo "  3. Set up web interface"
    echo "  4. Configure system services"
    echo ""

    if ! ask_yes_no "Ready to start installation?"; then
        echo "Installation cancelled."
        exit 0
    fi

    echo ""
    print_info "Installing myNode... This may take a while!"
    echo ""

    # Run the myNode installer
    if bash ./mynode_setup.sh online; then
        print_success "myNode installed successfully!"
    else
        print_error "Installation failed!"
        echo ""
        echo "Check the error messages above."
        echo "You may need to run the installer manually:"
        echo "  sudo bash mynode_setup.sh online"
        echo ""
        exit 1
    fi
}

configure_for_nexus() {
    print_step "🔧 INTEGRATING WITH NEXUS AGI"

    print_info "Configuring myNode to work with your bridge system..."
    echo ""

    # Add configuration to .env
    if [ -f "/home/user/nexus_agi/.env" ]; then
        echo "" >> /home/user/nexus_agi/.env
        echo "# Bitcoin Node Configuration (myNode)" >> /home/user/nexus_agi/.env
        echo "BITCOIN_NODE_URL=http://localhost:8332" >> /home/user/nexus_agi/.env
        echo "BITCOIN_RPC_USER=mynode" >> /home/user/nexus_agi/.env
        echo "BITCOIN_RPC_PASSWORD=bolt" >> /home/user/nexus_agi/.env
        echo "USE_LOCAL_BITCOIN_NODE=true" >> /home/user/nexus_agi/.env

        print_success "Configuration added to .env"
    fi

    echo ""
    print_info "Your bridge can now use your local Bitcoin node!"
}

show_next_steps() {
    print_step "🎉 INSTALLATION COMPLETE!"

    cat << 'EOF'
╔═══════════════════════════════════════════════════════════════════════════╗
║                          WHAT HAPPENS NEXT                                ║
╚═══════════════════════════════════════════════════════════════════════════╝

1️⃣  myNode is now running!

2️⃣  Access the web interface:
   • Open browser
   • Go to: http://mynode.local
   • Or: http://localhost

3️⃣  Initial Bitcoin sync will start
   • This takes 1-7 days
   • Downloads ~400GB
   • You can use your computer while it syncs!

4️⃣  After sync completes:
   • You'll have a full Bitcoin node
   • Can switch between mainnet/testnet
   • Works with your bridge system!

╔═══════════════════════════════════════════════════════════════════════════╗
║                         USING WITH YOUR BRIDGE                            ║
╚═══════════════════════════════════════════════════════════════════════════╝

Your bridge scripts are now configured to use your local node!

When Bitcoin is fully synced:
• Switch from regtest to real Bitcoin
• Use real testnet coins
• Or even use mainnet (with real BTC!)

╔═══════════════════════════════════════════════════════════════════════════╗
║                            USEFUL COMMANDS                                ║
╚═══════════════════════════════════════════════════════════════════════════╝

Check sync status:
  bitcoin-cli getblockchaininfo

Check node status:
  systemctl status bitcoind

View logs:
  journalctl -u bitcoind -f

Web interface:
  http://mynode.local

╔═══════════════════════════════════════════════════════════════════════════╗
║                              IMPORTANT                                    ║
╚═══════════════════════════════════════════════════════════════════════════╝

⚠️  Keep your computer running during initial sync!
⚠️  Don't shut down until sync is complete
⚠️  Check progress at: http://mynode.local

✅ You can still use your computer normally
✅ Syncing happens in the background
✅ Safe to use while syncing

EOF

    echo ""
    print_success "You now have your own Bitcoin node! 🎉"
    echo ""
}

main_menu() {
    while true; do
        print_banner
        echo -e "${WHITE}myNode Setup Options:${NC}\n"
        echo -e "  ${GREEN}1)${NC} 🎓 ${CYAN}What is myNode? (Beginners start here!)${NC}"
        echo -e "  ${GREEN}2)${NC} 🚀 ${CYAN}Full Installation (download + install + configure)${NC}"
        echo -e "  ${GREEN}3)${NC} 📥 ${CYAN}Just download setup script${NC}"
        echo -e "  ${GREEN}4)${NC} ⚙️  ${CYAN}Run installer (if already downloaded)${NC}"
        echo -e "  ${GREEN}5)${NC} 🔧 ${CYAN}Configure for Nexus AGI only${NC}"
        echo -e "  ${GREEN}6)${NC} 📊 ${CYAN}Check if myNode is installed${NC}"
        echo -e "  ${GREEN}7)${NC} 🚪 ${CYAN}Exit${NC}"
        echo ""

        read -p "$(echo -e ${YELLOW}Enter your choice [1-7]: ${NC})" choice

        case $choice in
            1)
                show_introduction
                ;;
            2)
                show_introduction
                check_prerequisites
                download_mynode
                install_mynode
                configure_for_nexus
                show_next_steps
                exit 0
                ;;
            3)
                check_prerequisites
                download_mynode
                print_info "Script downloaded! Run option 4 to install."
                read -p "Press ENTER to continue..."
                ;;
            4)
                if [ ! -f "mynode_setup.sh" ]; then
                    print_error "Setup script not found!"
                    echo "Run option 3 first to download it."
                else
                    install_mynode
                    configure_for_nexus
                    show_next_steps
                    exit 0
                fi
                read -p "Press ENTER to continue..."
                ;;
            5)
                configure_for_nexus
                print_success "Configuration complete!"
                read -p "Press ENTER to continue..."
                ;;
            6)
                echo ""
                if command -v bitcoin-cli &> /dev/null; then
                    print_success "Bitcoin Core is installed!"
                    bitcoin-cli --version
                else
                    print_warning "Bitcoin Core not found"
                fi
                echo ""
                if systemctl is-active --quiet bitcoind 2>/dev/null; then
                    print_success "Bitcoin node is running!"
                else
                    print_warning "Bitcoin node not running (or myNode not installed)"
                fi
                echo ""
                read -p "Press ENTER to continue..."
                ;;
            7)
                echo -e "\n${GREEN}👋 Goodbye!${NC}\n"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid choice. Please pick 1-7.${NC}"
                sleep 2
                ;;
        esac
    done
}

# Main execution
if [ "$EUID" -ne 0 ]; then
    print_banner
    print_error "This script must be run with sudo!"
    echo ""
    echo "Please run:"
    echo "  ${GREEN}sudo bash $0${NC}"
    echo ""
    exit 1
fi

main_menu
