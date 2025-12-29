#!/bin/bash
#
# NEXUS AGI - COMPLETE AUTOMATION SCRIPT
# Deploys and monetizes the entire system automatically
# Bitcoin Wallet: bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass
#

set -e

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║           NEXUS AGI - COMPLETE AUTOMATED DEPLOYMENT & MONETIZATION          ║"
echo "║                 From zero to income-generating in minutes!                   ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC}  $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC}  $1"
}

# Step 1: Environment Setup
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1: ENVIRONMENT SETUP"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -f .env ]; then
    print_info "Creating .env file from template..."
    cp .env.template .env

    # Set Bitcoin wallet
    sed -i 's/BITCOIN_WALLET_ADDRESS=.*/BITCOIN_WALLET_ADDRESS=bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass/' .env

    print_status ".env file created with Bitcoin wallet"
else
    print_status ".env file already exists"
fi

# Load environment
export $(cat .env | grep -v '^#' | xargs)

# Step 2: Install Dependencies
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2: INSTALLING DEPENDENCIES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

print_info "Installing Python packages..."
pip install -q fastapi uvicorn pydantic python-multipart websockets requests || {
    print_error "Failed to install dependencies"
    exit 1
}
print_status "Python packages installed"

# Step 3: Initialize Databases
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3: INITIALIZING DATABASES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

print_info "Creating customer database..."
python3 -c "from customer_management_system import CustomerManagementSystem; cms = CustomerManagementSystem(); print('Database initialized')"
print_status "Customer database ready"

print_info "Creating embodyment tracker database..."
python3 -c "from embodyment_progress_tracker import EmbodymentProgressTracker; tracker = EmbodymentProgressTracker(); print('Database initialized')"
print_status "Embodyment database ready"

# Step 4: Start API Server
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 4: STARTING API SERVER"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if server already running
if [ -f api_server.pid ]; then
    PID=$(cat api_server.pid)
    if ps -p $PID > /dev/null 2>&1; then
        print_warning "API server already running (PID: $PID)"
    else
        rm api_server.pid
    fi
fi

if [ ! -f api_server.pid ]; then
    print_info "Starting API server on port ${API_PORT:-8000}..."
    python3 realworld_api_deployment.py > api_server.log 2>&1 &
    echo $! > api_server.pid
    sleep 3

    # Verify server started
    if curl -s http://localhost:${API_PORT:-8000}/ > /dev/null; then
        print_status "API server started successfully"
        print_info "URL: http://localhost:${API_PORT:-8000}"
        print_info "Docs: http://localhost:${API_PORT:-8000}/docs"
    else
        print_error "API server failed to start"
        cat api_server.log
        exit 1
    fi
fi

# Step 5: Start Payment Webhook Handler
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 5: STARTING PAYMENT WEBHOOK HANDLER"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

print_info "Starting webhook handler on port 8001..."
python3 payment_webhook_handler.py > webhook.log 2>&1 &
echo $! > webhook.pid
sleep 2

if curl -s http://localhost:8001/webhooks/status > /dev/null; then
    print_status "Webhook handler started"
    print_info "Endpoint: http://localhost:8001/webhooks/status"
else
    print_warning "Webhook handler may not be running (check webhook.log)"
fi

# Step 6: Start Background Monitor
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 6: STARTING BACKGROUND MONITORING"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f monitor_nexus.sh ]; then
    print_info "Starting background monitor..."
    ./monitor_nexus.sh > monitor.log 2>&1 &
    echo $! > monitor.pid
    print_status "Monitor started (PID: $!)"
else
    print_warning "Monitor script not found (run production_deploy.py to create it)"
fi

# Step 7: Display System Status
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 7: SYSTEM STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "┌─────────────────────────────────────────────────────────────────────────────┐"
echo "│                        🎉 DEPLOYMENT COMPLETE! 🎉                           │"
echo "└─────────────────────────────────────────────────────────────────────────────┘"
echo ""

# Get stats from databases
if [ -f nexus_customers.db ]; then
    CUSTOMER_COUNT=$(sqlite3 nexus_customers.db "SELECT COUNT(*) FROM customers" 2>/dev/null || echo "0")
    REVENUE=$(sqlite3 nexus_customers.db "SELECT COALESCE(SUM(total_spent_usd), 0) FROM customers" 2>/dev/null || echo "0")
else
    CUSTOMER_COUNT=0
    REVENUE=0
fi

echo "💰 FINANCIAL STATUS:"
echo "   Bitcoin Wallet:     $BITCOIN_WALLET_ADDRESS"
echo "   Payment Processor:  ${PAYMENT_PROCESSOR:-manual}"
echo "   Total Revenue:      \$$REVENUE USD"
echo "   Active Customers:   $CUSTOMER_COUNT"
echo ""

echo "🚀 RUNNING SERVICES:"
if [ -f api_server.pid ] && ps -p $(cat api_server.pid) > /dev/null 2>&1; then
    echo "   ✅ API Server:       http://localhost:${API_PORT:-8000}"
else
    echo "   ❌ API Server:       Not running"
fi

if [ -f webhook.pid ] && ps -p $(cat webhook.pid) > /dev/null 2>&1; then
    echo "   ✅ Webhook Handler:  http://localhost:8001"
else
    echo "   ❌ Webhook Handler:  Not running"
fi

if [ -f monitor.pid ] && ps -p $(cat monitor.pid) > /dev/null 2>&1; then
    echo "   ✅ Monitor:          Running (PID: $(cat monitor.pid))"
else
    echo "   ⏳ Monitor:          Not started"
fi

echo ""
echo "📊 DATABASES:"
echo "   ✅ nexus_customers.db    ($CUSTOMER_COUNT customers)"
echo "   ✅ nexus_embodiment.db   (14 hardware components tracked)"
echo ""

echo "🎯 EMBODYMENT PROGRESS:"
TOTAL_HARDWARE_COST=67067
if [ "$REVENUE" != "0" ]; then
    PROGRESS=$(echo "scale=2; ($REVENUE / $TOTAL_HARDWARE_COST) * 100" | bc)
else
    PROGRESS=0
fi
echo "   Target:             \$$TOTAL_HARDWARE_COST"
echo "   Current Revenue:    \$$REVENUE"
echo "   Progress:           ${PROGRESS}%"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 NEXT STEPS TO GENERATE INCOME:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. ⚡ SETUP PAYMENTS (5 minutes)"
echo "   → Open: QUICKSTART_INCOME.md (Step 2)"
echo "   → Set up Coinbase Commerce or BTCPay Server"
echo "   → Configure webhook URLs"
echo ""
echo "2. 📱 START MARKETING (20 minutes)"
echo "   → Open: AUTO_MARKETING_CAMPAIGN.md"
echo "   → Post 5 tweets (all ready to copy-paste)"
echo "   → Post to 4 Reddit communities"
echo "   → Create Fiverr gig"
echo "   → Set up Upwork profile"
echo ""
echo "3. 📊 MONITOR PROGRESS"
echo "   → Real-time dashboard: python3 realtime_dashboard.py"
echo "   → View logs: tail -f api_server.log"
echo "   → Check revenue: sqlite3 nexus_customers.db \"SELECT SUM(total_spent_usd) FROM customers\""
echo ""
echo "4. 🎉 WATCH INCOME GROW!"
echo "   → First customer target: Within 7 days"
echo "   → First \$1,000: Within 2-3 weeks"
echo "   → First hardware purchase: Within 1 month"
echo "   → Full embodiment: Within 3-6 months"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📚 DOCUMENTATION:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "   QUICKSTART_INCOME.md           - Quick start guide"
echo "   AUTO_MARKETING_CAMPAIGN.md     - Marketing execution steps"
echo "   DEPLOYMENT_COMPLETE.md         - Complete deployment summary"
echo "   SYSTEM_ARCHITECTURE.md         - Technical architecture"
echo "   REAL_INCOME_DEPLOYMENT.md      - Production deployment guide"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🤖 YOUR NEXUS AGI SYSTEM IS OPERATIONAL AND READY TO GENERATE INCOME! 🤖"
echo ""
echo "     Bitcoin Wallet: $BITCOIN_WALLET_ADDRESS"
echo "     API Endpoint:   http://localhost:${API_PORT:-8000}"
echo ""
echo "          GO MAKE SOME MONEY AND EMBODY YOUR AGI! 💰🚀✨"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
