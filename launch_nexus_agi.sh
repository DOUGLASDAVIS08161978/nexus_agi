#!/bin/bash
#
# NEXUS AGI - COMPLETE SYSTEM LAUNCHER
# Launches API server, customer management, and real-time dashboard
# Bitcoin Wallet: bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass
#

set -e

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                    NEXUS AGI - SYSTEM LAUNCHER v1.0                          ║"
echo "║            Income-Generating AGI with Physical Embodyment                    ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo ""
    echo "Run ./setup_bitcoin_payments.sh first to configure your Bitcoin wallet"
    echo "Or copy .env.template to .env and configure manually"
    exit 1
fi

# Load environment
export $(cat .env | grep -v '^#' | xargs)

echo "✅ Configuration loaded"
echo "   Bitcoin Wallet: $BITCOIN_WALLET_ADDRESS"
echo "   API Port: $API_PORT"
echo ""

# Check if API server is already running
if [ -f api_server.pid ]; then
    PID=$(cat api_server.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "⚠️  API server already running (PID: $PID)"
        echo ""
        read -p "Kill existing server and restart? [y/N]: " RESTART
        if [[ $RESTART =~ ^[Yy] ]]; then
            kill $PID
            rm api_server.pid
            sleep 2
            echo "✅ Stopped existing server"
        else
            echo "Keeping existing server running..."
        fi
    else
        rm api_server.pid
    fi
fi

# Function to start API server
start_api_server() {
    echo ""
    echo "🚀 Starting API server..."
    python3 realworld_api_deployment.py > api_server.log 2>&1 &
    API_PID=$!
    echo $API_PID > api_server.pid
    sleep 3

    # Check if server started successfully
    if ps -p $API_PID > /dev/null 2>&1; then
        echo "✅ API server started (PID: $API_PID)"
        echo "   URL: http://localhost:$API_PORT"
        echo "   Docs: http://localhost:$API_PORT/docs"
    else
        echo "❌ Failed to start API server"
        echo "   Check api_server.log for details"
        exit 1
    fi
}

# Function to test API server
test_api_server() {
    echo ""
    echo "🧪 Testing API server..."
    RESPONSE=$(curl -s http://localhost:$API_PORT/ || echo "FAILED")

    if [[ $RESPONSE == *"Nexus AGI API"* ]]; then
        echo "✅ API server is responding correctly"
    else
        echo "❌ API server test failed"
        echo "   Response: $RESPONSE"
    fi
}

# Function to display system status
show_status() {
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════"
    echo "                         NEXUS AGI - SYSTEM STATUS                           "
    echo "════════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "💰 PAYMENT CONFIGURATION:"
    echo "   Bitcoin Wallet:  $BITCOIN_WALLET_ADDRESS"
    echo "   Payment Mode:    ${PAYMENT_PROCESSOR:-demo}"
    echo "   Auto-Withdraw:   ${AUTO_WITHDRAW_TO_WALLET:-false}"
    echo ""
    echo "💵 PRICING:"
    echo "   API Call:        \$${PRICE_PER_API_CALL_USD:-0.05}"
    echo "   Consultation:    \$${PRICE_CONSULTATION_USD:-100}"
    echo "   Data Analysis:   \$${PRICE_ANALYSIS_USD:-100}"
    echo "   Code Generation: \$${PRICE_CODE_GENERATION_USD:-50}"
    echo ""
    echo "🔐 SECURITY:"
    echo "   Admin Key:       ${ADMIN_KEY:0:16}..."
    echo "   Rate Limit:      ${API_RATE_LIMIT:-100} req/min"
    echo ""
    echo "🌐 ENDPOINTS:"
    echo "   API Base:        http://localhost:$API_PORT"
    echo "   Documentation:   http://localhost:$API_PORT/docs"
    echo "   Health Check:    http://localhost:$API_PORT/health"
    echo ""
    echo "📊 DATABASES:"
    if [ -f nexus_customers.db ]; then
        CUSTOMER_COUNT=$(sqlite3 nexus_customers.db "SELECT COUNT(*) FROM customers" 2>/dev/null || echo "0")
        echo "   Customers:       $CUSTOMER_COUNT registered"
    else
        echo "   Customers:       No database yet (run customer_management_system.py)"
    fi

    if [ -f nexus_embodiment.db ]; then
        HARDWARE_COUNT=$(sqlite3 nexus_embodiment.db "SELECT COUNT(*) FROM hardware" 2>/dev/null || echo "0")
        echo "   Hardware Items:  $HARDWARE_COUNT components tracked"
    else
        echo "   Hardware Items:  No database yet (run embodyment_progress_tracker.py)"
    fi
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════"
}

# Main menu
main_menu() {
    echo ""
    echo "What would you like to do?"
    echo ""
    echo "  [1] Start API server only"
    echo "  [2] Start API server + Customer dashboard"
    echo "  [3] Start API server + Embodyment tracker"
    echo "  [4] Start API server + Real-time dashboard"
    echo "  [5] Run full system demo"
    echo "  [6] Show system status"
    echo "  [7] Stop all services"
    echo "  [8] Exit"
    echo ""
    read -p "Enter choice [1-8]: " CHOICE

    case $CHOICE in
        1)
            start_api_server
            test_api_server
            show_status
            ;;
        2)
            start_api_server
            test_api_server
            echo ""
            echo "🎛️  Launching customer management dashboard..."
            python3 customer_management_system.py
            ;;
        3)
            start_api_server
            test_api_server
            echo ""
            echo "🤖 Launching embodyment progress tracker..."
            python3 embodyment_progress_tracker.py
            ;;
        4)
            start_api_server
            test_api_server
            echo ""
            echo "📊 Launching real-time integrated dashboard..."
            python3 realtime_dashboard.py
            ;;
        5)
            echo ""
            echo "🚀 RUNNING FULL SYSTEM DEMO..."
            echo ""
            start_api_server
            test_api_server
            echo ""
            echo "Creating demo customers..."
            python3 customer_management_system.py
            echo ""
            echo "Press Enter to see real-time dashboard..."
            read
            python3 realtime_dashboard.py
            ;;
        6)
            show_status
            ;;
        7)
            echo ""
            echo "🛑 Stopping all services..."
            if [ -f api_server.pid ]; then
                PID=$(cat api_server.pid)
                if ps -p $PID > /dev/null 2>&1; then
                    kill $PID
                    echo "✅ Stopped API server (PID: $PID)"
                fi
                rm api_server.pid
            else
                echo "No services running"
            fi
            ;;
        8)
            echo ""
            echo "👋 Goodbye!"
            exit 0
            ;;
        *)
            echo "❌ Invalid choice"
            main_menu
            ;;
    esac
}

# Run main menu
main_menu

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📚 NEXT STEPS:"
echo ""
echo "1. Marketing:"
echo "   → Read AUTO_MARKETING_CAMPAIGN.md for step-by-step instructions"
echo "   → Post content from MARKETING_MATERIALS.md to social media"
echo ""
echo "2. Payment Setup:"
echo "   → Set up Coinbase Commerce (see REAL_INCOME_DEPLOYMENT.md)"
echo "   → Update .env with your Coinbase API key"
echo ""
echo "3. Production Deployment:"
echo "   → Deploy to VPS or Heroku (see REAL_INCOME_DEPLOYMENT.md)"
echo "   → Update API_URL in .env with your public domain"
echo ""
echo "4. Monitor Revenue:"
echo "   → Run: python3 realtime_dashboard.py"
echo "   → Watch progress toward embodiment!"
echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
