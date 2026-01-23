#!/bin/bash

# ============================================================================
# 🚀 NEXUS AGI - BEGINNER-FRIENDLY CRYPTO LAUNCHER 🚀
# ============================================================================
#
# This is a ONE-CLICK launcher for complete beginners!
# Just run this and follow the simple instructions.
#
# Author: Douglas Shane Davis & Claude
# Date: 2026-01-23
# ============================================================================

set -e

# Colors for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Your wallet address
WALLET_ADDRESS="0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

print_banner() {
    clear
    echo -e "${CYAN}"
    cat << "EOF"
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                 🚀 NEXUS AGI - BEGINNER CRYPTO LAUNCHER 🚀                  ║
║                                                                              ║
║              Make Bitcoin, Bridge to Monad, Get Tokens!                     ║
║                                                                              ║
║                    Everything Made SUPER EASY! 😊                           ║
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

wait_for_enter() {
    echo -e "${CYAN}"
    read -p "Press ENTER to continue..."
    echo -e "${NC}"
}

show_progress() {
    local duration=$1
    local message=$2

    echo -ne "${BLUE}$message "
    for i in $(seq 1 $duration); do
        echo -ne "▓"
        sleep 1
    done
    echo -e " Done!${NC}"
}

# ============================================================================
# CHECK FUNCTIONS
# ============================================================================

check_python() {
    print_step "🔍 STEP 1: Checking if Python is installed..."

    if command -v python3 &> /dev/null; then
        local version=$(python3 --version 2>&1 | awk '{print $2}')
        print_success "Python is installed: $version"
        return 0
    else
        print_error "Python 3 is not installed!"
        echo ""
        echo -e "${YELLOW}📖 How to install Python:${NC}"
        echo "  1. Go to: https://www.python.org/downloads/"
        echo "  2. Download Python 3.10 or newer"
        echo "  3. Install it"
        echo "  4. Run this script again"
        echo ""
        exit 1
    fi
}

check_dependencies() {
    print_step "🔍 STEP 2: Checking if required tools are installed..."

    local missing_deps=()

    # Check for web3
    if ! python3 -c "import web3" 2>/dev/null; then
        missing_deps+=("web3")
    fi

    if [ ${#missing_deps[@]} -eq 0 ]; then
        print_success "All tools are installed!"
        return 0
    else
        print_warning "Some tools are missing. Installing them now..."
        echo ""

        pip3 install web3 eth-account --quiet

        print_success "Tools installed successfully!"
        return 0
    fi
}

check_wallet_balance() {
    print_step "💰 STEP 3: Checking if you have gas money (testnet ETH)..."

    # Run Python script to check balance
    local balance=$(python3 -c "
from web3 import Web3
w3 = Web3(Web3.HTTPProvider('https://testnet-rpc.monad.xyz'))
balance = w3.eth.get_balance('$WALLET_ADDRESS')
print(w3.from_wei(balance, 'ether'))
" 2>/dev/null || echo "0")

    echo -e "${BLUE}Your current balance: $balance testnet ETH${NC}"

    if (( $(echo "$balance >= 0.01" | bc -l) )); then
        print_success "You have enough gas! ✨"
        return 0
    else
        print_warning "You need testnet ETH to make transactions!"
        echo ""
        echo -e "${YELLOW}📖 How to get FREE testnet ETH:${NC}"
        echo ""
        echo "  1️⃣  Open your web browser"
        echo "  2️⃣  Go to: ${CYAN}https://faucet.monad.xyz${NC}"
        echo "  3️⃣  Enter your address:"
        echo "      ${GREEN}$WALLET_ADDRESS${NC}"
        echo "  4️⃣  Click 'Request' or 'Send Me ETH'"
        echo "  5️⃣  Wait 1-2 minutes"
        echo "  6️⃣  Come back and run this script again!"
        echo ""
        echo -e "${BLUE}💡 TIP: You can copy your address by clicking it!${NC}"
        echo ""

        if ask_yes_no "Do you want me to open the faucet website for you?"; then
            # Try to open browser
            if command -v xdg-open &> /dev/null; then
                xdg-open "https://faucet.monad.xyz" &>/dev/null &
            elif command -v open &> /dev/null; then
                open "https://faucet.monad.xyz" &>/dev/null &
            else
                echo -e "${YELLOW}Could not auto-open browser. Please visit manually:${NC}"
                echo "https://faucet.monad.xyz"
            fi
        fi

        echo ""
        echo -e "${RED}⚠️  You need testnet ETH before we can continue!${NC}"
        echo -e "${YELLOW}After you get testnet ETH, run this script again.${NC}"
        echo ""
        exit 0
    fi
}

# ============================================================================
# MAIN MENU
# ============================================================================

show_main_menu() {
    while true; do
        print_banner

        echo -e "${WHITE}Choose what you want to do:${NC}\n"
        echo -e "  ${GREEN}1)${NC} 🎓 ${CYAN}I'm brand new - Show me how everything works${NC}"
        echo -e "  ${GREEN}2)${NC} ⛏️  ${CYAN}Mine Bitcoin (regtest) and bridge to Monad${NC}"
        echo -e "  ${GREEN}3)${NC} 🪙 ${CYAN}Mint WBTC tokens to my wallet${NC}"
        echo -e "  ${GREEN}4)${NC} 💰 ${CYAN}Check my wallet balance${NC}"
        echo -e "  ${GREEN}5)${NC} 📱 ${CYAN}Show me how to add tokens to my mobile wallet${NC}"
        echo -e "  ${GREEN}6)${NC} 🔧 ${CYAN}Check if I have everything I need${NC}"
        echo -e "  ${GREEN}7)${NC} ❓ ${CYAN}Help - I'm confused!${NC}"
        echo -e "  ${GREEN}8)${NC} 🚪 ${CYAN}Exit${NC}"
        echo ""

        read -p "$(echo -e ${YELLOW}Enter your choice [1-8]: ${NC})" choice

        case $choice in
            1) show_tutorial ;;
            2) run_mining_and_bridge ;;
            3) mint_wbtc_tokens ;;
            4) check_balance ;;
            5) show_mobile_wallet_guide ;;
            6) run_full_check ;;
            7) show_help ;;
            8)
                echo -e "\n${GREEN}👋 Goodbye! Come back anytime!${NC}\n"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid choice. Please pick a number from 1 to 8.${NC}"
                sleep 2
                ;;
        esac
    done
}

# ============================================================================
# TUTORIAL FOR COMPLETE BEGINNERS
# ============================================================================

show_tutorial() {
    print_banner
    print_step "🎓 BEGINNER'S TUTORIAL - How Everything Works"

    cat << 'EOF'
📚 WELCOME TO CRYPTO! Here's what you're about to learn:

╔═══════════════════════════════════════════════════════════════════════════╗
║                         WHAT IS THIS ALL ABOUT?                           ║
╚═══════════════════════════════════════════════════════════════════════════╝

Think of this like a video game where you can:
  🎮 Mine virtual Bitcoin on your computer
  🌉 Convert it to tokens on another blockchain (Monad)
  💰 See those tokens in your crypto wallet!

╔═══════════════════════════════════════════════════════════════════════════╗
║                         STEP-BY-STEP EXPLANATION                          ║
╚═══════════════════════════════════════════════════════════════════════════╝

STEP 1: What is Bitcoin?
  • Bitcoin is digital money
  • We're using "testnet" Bitcoin (practice money, worth $0)
  • You can mine it on your computer!

STEP 2: What is a Blockchain?
  • It's like a super secure digital ledger
  • Monad is a blockchain (like Ethereum)
  • You need "gas" (testnet ETH) to do things on it

STEP 3: What is a Bridge?
  • It converts Bitcoin to WBTC (Wrapped Bitcoin)
  • WBTC is Bitcoin that works on Ethereum/Monad
  • Like exchanging dollars for euros!

STEP 4: What is a Wallet?
  • It's like a bank account for crypto
  • You have one! Address: 0x24f6b1ce...
  • You can see your tokens in MetaMask

╔═══════════════════════════════════════════════════════════════════════════╗
║                         WHAT YOU NEED TO DO                               ║
╚═══════════════════════════════════════════════════════════════════════════╝

1️⃣  Get FREE testnet ETH (gas money)
   → Go to: https://faucet.monad.xyz
   → Enter your address: 0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771
   → Click "Request"
   → Wait 1-2 minutes

2️⃣  Run this script again and choose option 2
   → It will mine Bitcoin
   → Bridge it to Monad
   → Create WBTC tokens

3️⃣  Add WBTC to your mobile wallet (option 5)
   → Follow the simple instructions
   → You'll see your tokens! 🎉

╔═══════════════════════════════════════════════════════════════════════════╗
║                         IMPORTANT THINGS TO KNOW                          ║
╚═══════════════════════════════════════════════════════════════════════════╝

✅ This is all TESTNET (practice mode)
   • Everything is FREE
   • The tokens have NO real value ($0)
   • It's safe to experiment!

✅ You can't lose real money
   • This is for learning and testing
   • No real Bitcoin or Ethereum is used

✅ If you make a mistake
   • Just start over!
   • Nothing bad will happen
   • It's all practice!

EOF

    wait_for_enter

    if ask_yes_no "Did this help? Do you want to get started now?"; then
        echo ""
        print_info "Great! Let's check if you have testnet ETH..."
        sleep 2
        check_wallet_balance
    fi
}

# ============================================================================
# MINING AND BRIDGING
# ============================================================================

run_mining_and_bridge() {
    print_banner
    print_step "⛏️  MINING BITCOIN AND BRIDGING TO MONAD"

    echo -e "${CYAN}This will:${NC}"
    echo "  1. Mine 10 Bitcoin blocks on regtest"
    echo "  2. Bridge the Bitcoin to Monad testnet"
    echo "  3. Create WBTC tokens in your wallet"
    echo ""

    if ! ask_yes_no "Ready to start?"; then
        return
    fi

    echo ""
    print_info "Starting the bridge..."
    echo ""

    if [ -f "monad_regtest_bridge.py" ]; then
        python3 monad_regtest_bridge.py
    else
        print_error "Bridge script not found!"
        echo "Please make sure you're in the nexus_agi directory."
        wait_for_enter
        return
    fi

    echo ""
    print_success "Bridge session complete!"
    echo ""
    print_info "What just happened:"
    echo "  • Mined 10 Bitcoin blocks (500 BTC)"
    echo "  • Attempted to bridge to Monad"
    echo "  • Check the output above for results"
    echo ""

    wait_for_enter
}

# ============================================================================
# MINT WBTC TOKENS
# ============================================================================

mint_wbtc_tokens() {
    print_banner
    print_step "🪙 MINTING WBTC TOKENS TO YOUR WALLET"

    echo -e "${CYAN}This will:${NC}"
    echo "  1. Check if you have testnet ETH (gas)"
    echo "  2. Deploy or connect to WBTC contract"
    echo "  3. Mint WBTC tokens to your address"
    echo "  4. Show you how to see them in your wallet"
    echo ""

    if ! ask_yes_no "Ready to mint tokens?"; then
        return
    fi

    echo ""
    print_info "Minting tokens..."
    echo ""

    if [ -f "mint_wbtc_tokens.py" ]; then
        python3 mint_wbtc_tokens.py
    else
        print_error "Mint script not found!"
        echo "Please make sure you're in the nexus_agi directory."
        wait_for_enter
        return
    fi

    echo ""
    wait_for_enter
}

# ============================================================================
# CHECK BALANCE
# ============================================================================

check_balance() {
    print_banner
    print_step "💰 CHECKING YOUR WALLET BALANCE"

    if [ -f "check_why_no_tokens.py" ]; then
        python3 check_why_no_tokens.py
    else
        print_error "Balance checker not found!"
        wait_for_enter
        return
    fi

    echo ""
    wait_for_enter
}

# ============================================================================
# MOBILE WALLET GUIDE
# ============================================================================

show_mobile_wallet_guide() {
    print_banner
    print_step "📱 HOW TO SEE TOKENS IN YOUR MOBILE WALLET"

    cat << 'EOF'
╔═══════════════════════════════════════════════════════════════════════════╗
║                    ADD MONAD TESTNET TO METAMASK                          ║
╚═══════════════════════════════════════════════════════════════════════════╝

STEP 1: Open MetaMask on your phone

STEP 2: Tap the hamburger menu (☰) → Settings

STEP 3: Tap "Networks" → "Add Network"

STEP 4: Fill in these EXACT values:

  Network Name:
  Monad Testnet

  RPC URL:
  https://testnet-rpc.monad.xyz

  Chain ID:
  10143

  Symbol:
  ETH

  Block Explorer URL:
  https://explorer-testnet.monad.xyz

STEP 5: Tap "Save"

╔═══════════════════════════════════════════════════════════════════════════╗
║                         ADD WBTC TOKEN                                    ║
╚═══════════════════════════════════════════════════════════════════════════╝

STEP 1: Switch to Monad Testnet network

STEP 2: Scroll down and tap "Import tokens"

STEP 3: Select "Custom token"

STEP 4: Enter:

  Token Address:
  [You'll get this after deploying WBTC contract]

  Token Symbol:
  WBTC

  Token Decimals:
  8

STEP 5: Tap "Import"

STEP 6: 🎉 You'll see your WBTC balance!

╔═══════════════════════════════════════════════════════════════════════════╗
║                              IMPORTANT                                    ║
╚═══════════════════════════════════════════════════════════════════════════╝

• You need testnet ETH first (from faucet)
• Then mint WBTC tokens (option 3 in main menu)
• Then add token to wallet using contract address
• Then tokens will appear!

EOF

    wait_for_enter
}

# ============================================================================
# RUN FULL CHECK
# ============================================================================

run_full_check() {
    print_banner
    print_step "🔧 RUNNING FULL SYSTEM CHECK"

    check_python
    check_dependencies
    check_wallet_balance

    echo ""
    print_success "System check complete!"
    echo ""
    wait_for_enter
}

# ============================================================================
# HELP SECTION
# ============================================================================

show_help() {
    print_banner
    print_step "❓ HELP - COMMON QUESTIONS"

    cat << 'EOF'
╔═══════════════════════════════════════════════════════════════════════════╗
║                         FREQUENTLY ASKED QUESTIONS                        ║
╚═══════════════════════════════════════════════════════════════════════════╝

Q: Why don't I see any tokens in my wallet?
A: You need two things:
   1. Testnet ETH for gas (get from faucet)
   2. Add the WBTC token to your wallet (see option 5)

Q: Where do I get testnet ETH?
A: Go to https://faucet.monad.xyz
   Enter your address: 0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771
   Click "Request" and wait 1-2 minutes

Q: Are these tokens worth real money?
A: NO! This is all TESTNET (practice mode).
   Testnet tokens = $0.00 value
   It's completely safe to experiment!

Q: What if I make a mistake?
A: No problem! Since it's testnet:
   • You can't lose real money
   • Just try again
   • Everything is reversible

Q: Do I need to know how to code?
A: Nope! This script does everything for you.
   Just follow the simple menu options.

Q: How do I add WBTC to my wallet?
A: Choose option 5 from the main menu.
   Follow the step-by-step instructions.

Q: The script says "insufficient balance"
A: You need testnet ETH! Get it from:
   https://faucet.monad.xyz

Q: How long does everything take?
A: • Mining Bitcoin: 5-10 seconds
   • Bridging: 10-30 seconds
   • Getting testnet ETH: 1-2 minutes
   • Total: About 5 minutes!

Q: Can I run this multiple times?
A: Yes! Run it as many times as you want.
   Each time will mine more Bitcoin and create more tokens.

Q: What's my wallet address?
A: 0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771
   (You can copy this to get testnet ETH)

╔═══════════════════════════════════════════════════════════════════════════╗
║                           STILL CONFUSED?                                 ║
╚═══════════════════════════════════════════════════════════════════════════╝

Start with option 1 (Beginner's Tutorial)
It explains everything in simple terms!

EOF

    wait_for_enter
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    # Make sure we're in the right directory
    if [ ! -f "monad_regtest_bridge.py" ]; then
        print_error "Please run this script from the /home/user/nexus_agi directory"
        exit 1
    fi

    # Show main menu
    show_main_menu
}

# Run the script
main
