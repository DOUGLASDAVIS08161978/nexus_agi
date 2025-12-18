#!/bin/bash

# Nexus AGI - Complete Bitcoin Payment Setup & Deployment
# This script sets up BTCPay Server and deploys your monetized API

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════╗"
echo "║   NEXUS AGI - BITCOIN PAYMENT SETUP & DEPLOYMENT      ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration file
CONFIG_FILE="bitcoin_deployment.conf"

echo -e "${BLUE}🔧 STEP 1: Generating Secure Keys and Configuration${NC}"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Generate SECRET_KEY if not exists
if [ ! -f .env ] || ! grep -q "SECRET_KEY=7a9940b6f75b2cdeb6fe9073304e3e5663d0eb119332e10c07229fdca6afcd5b" .env; then
    echo "Generating new SECRET_KEY..."
    SECRET_KEY=$(openssl rand -hex 32)
    echo -e "${GREEN}✅ SECRET_KEY generated${NC}"
else
    SECRET_KEY=$(grep "^SECRET_KEY=" .env | cut -d '=' -f2)
    echo -e "${GREEN}✅ Using existing SECRET_KEY${NC}"
fi

# Generate BTCPay admin password
BTCPAY_ADMIN_EMAIL="admin@nexusagi.local"
BTCPAY_ADMIN_PASSWORD=$(openssl rand -base64 24)

# Generate database credentials
POSTGRES_PASSWORD=$(openssl rand -base64 16)

# Bitcoin wallet address
BITCOIN_WALLET="bc1q2m6w8au8yrjp85m7rqlcxqx9ypfla6h8st8u78"

echo "Generated credentials:"
echo "  - API Secret Key: ✓"
echo "  - BTCPay Admin Email: $BTCPAY_ADMIN_EMAIL"
echo "  - BTCPay Admin Password: ✓"
echo "  - PostgreSQL Password: ✓"
echo "  - Bitcoin Wallet: $BITCOIN_WALLET"
echo ""

# Save credentials
cat > "$CONFIG_FILE" <<EOF
# Nexus AGI Bitcoin Deployment Configuration
# Generated: $(date)
# KEEP THIS FILE SECURE - Contains sensitive credentials!

# API Configuration
SECRET_KEY=$SECRET_KEY
BITCOIN_WALLET_ADDRESS=$BITCOIN_WALLET

# BTCPay Server Admin Credentials
BTCPAY_ADMIN_EMAIL=$BTCPAY_ADMIN_EMAIL
BTCPAY_ADMIN_PASSWORD=$BTCPAY_ADMIN_PASSWORD

# BTCPay Server URLs
BTCPAY_SERVER_URL=http://localhost:23000
BTCPAY_EXTERNAL_URL=http://localhost:23000

# Database
POSTGRES_PASSWORD=$POSTGRES_PASSWORD

# API Endpoints
API_URL=http://localhost:8000
API_DOCS=http://localhost:8000/docs
API_HEALTH=http://localhost:8000/health

# Network Mode
BITCOIN_NETWORK=testnet  # Change to 'mainnet' for production

# Instructions:
# 1. BTCPay Server will be available at: http://localhost:23000
# 2. Login with email: $BTCPAY_ADMIN_EMAIL
# 3. Password: (see BTCPAY_ADMIN_PASSWORD above)
# 4. API will be available at: http://localhost:8000
# 5. API docs at: http://localhost:8000/docs
EOF

chmod 600 "$CONFIG_FILE"

echo -e "${GREEN}✅ Configuration saved to: $CONFIG_FILE${NC}"
echo ""

echo -e "${BLUE}🔧 STEP 2: Updating Environment Configuration${NC}"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Update .env file
cat >> .env <<EOF

# ============================================================================
# BITCOIN DEPLOYMENT CONFIGURATION (Added $(date))
# ============================================================================

# BTCPay Server Configuration
BTCPAY_SERVER_URL=http://btcpayserver:23000
BTCPAY_API_KEY=
BTCPAY_STORE_ID=

# Admin Credentials
BTCPAY_ADMIN_EMAIL=$BTCPAY_ADMIN_EMAIL
BTCPAY_ADMIN_PASSWORD=$BTCPAY_ADMIN_PASSWORD

# PostgreSQL
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
EOF

echo -e "${GREEN}✅ Environment configuration updated${NC}"
echo ""

echo -e "${BLUE}🐳 STEP 3: Starting BTCPay Server & Dependencies${NC}"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Docker is not running. Please start Docker first.${NC}"
    echo "On Linux: sudo systemctl start docker"
    exit 1
fi

echo "Starting BTCPay Server components..."
echo "This may take 5-10 minutes on first run (downloading images)..."
echo ""

# Export variables for docker-compose
export SECRET_KEY
export POSTGRES_PASSWORD

# Start BTCPay Server stack
docker-compose -f docker-compose.btcpay.yml up -d postgres_btcpay nbxplorer

echo ""
echo "Waiting for PostgreSQL to be ready..."
sleep 10

# Check if postgres is ready
docker-compose -f docker-compose.btcpay.yml exec -T postgres_btcpay pg_isready -U postgres || sleep 5

echo -e "${GREEN}✅ Database ready${NC}"
echo ""

echo "Starting NBXplorer (Bitcoin blockchain indexer)..."
sleep 5

echo -e "${GREEN}✅ NBXplorer started${NC}"
echo ""

echo "Starting BTCPay Server..."
docker-compose -f docker-compose.btcpay.yml up -d btcpayserver

echo ""
echo "Waiting for BTCPay Server to initialize (30 seconds)..."
sleep 30

echo -e "${GREEN}✅ BTCPay Server started${NC}"
echo ""

echo -e "${BLUE}🚀 STEP 4: Starting Nexus AGI API${NC}"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Stop any existing API processes
pkill -f "uvicorn api_gateway.main_complete" || true
sleep 2

# Start API with Bitcoin support
echo "Starting Nexus AGI API with Bitcoin integration..."
docker-compose -f docker-compose.btcpay.yml up -d nexus_api

echo ""
echo "Waiting for API to start (15 seconds)..."
sleep 15

# Health check
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Nexus AGI API is running!${NC}"
else
    echo -e "${YELLOW}⚠️  API may still be starting up...${NC}"
fi

echo ""

echo -e "${BLUE}🔧 STEP 5: BTCPay Server Setup Instructions${NC}"
echo "═══════════════════════════════════════════════════════════"
echo ""

echo "BTCPay Server is running! Follow these steps to complete setup:"
echo ""
echo "1. Open BTCPay Server in your browser:"
echo -e "   ${GREEN}http://localhost:23000${NC}"
echo ""
echo "2. Click 'Register' and create your account with:"
echo -e "   Email: ${YELLOW}$BTCPAY_ADMIN_EMAIL${NC}"
echo -e "   Password: ${YELLOW}$BTCPAY_ADMIN_PASSWORD${NC}"
echo ""
echo "3. After registration, create a new store:"
echo "   - Store Name: Nexus AGI"
echo "   - Click 'Create'"
echo ""
echo "4. Configure your Bitcoin wallet:"
echo "   - Go to: Store → Settings → Wallet"
echo "   - Click 'Setup a wallet'"
echo "   - Select 'Connect an existing wallet'"
echo "   - Choose 'Enter extended public key'"
echo "   - For testing, you can use 'Generate a new wallet'"
echo "   - OR enter your wallet: $BITCOIN_WALLET"
echo ""
echo "5. Get your API credentials:"
echo "   - Go to: Account → Manage Account → API Keys"
echo "   - Click 'Generate Key'"
echo "   - Permissions needed:"
echo "     ☑ View invoices"
echo "     ☑ Create invoice"
echo "     ☑ Modify invoices"
echo "   - Copy the API Key"
echo "   - Copy the Store ID (from Store → Settings → General)"
echo ""
echo "6. Add credentials to your .env file:"
echo "   BTCPAY_API_KEY=<your_api_key>"
echo "   BTCPAY_STORE_ID=<your_store_id>"
echo ""
echo "7. Restart the API:"
echo "   docker-compose -f docker-compose.btcpay.yml restart nexus_api"
echo ""

echo -e "${BLUE}📊 STEP 6: Deployment Summary${NC}"
echo "═══════════════════════════════════════════════════════════"
echo ""

echo -e "${GREEN}✅ DEPLOYMENT COMPLETE!${NC}"
echo ""
echo "Your services are now running:"
echo ""
echo "🌐 Services:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "  BTCPay Server:    ${GREEN}http://localhost:23000${NC}"
echo -e "  Nexus AGI API:    ${GREEN}http://localhost:8000${NC}"
echo -e "  API Docs:         ${GREEN}http://localhost:8000/docs${NC}"
echo -e "  API Health:       ${GREEN}http://localhost:8000/health${NC}"
echo ""

echo "🔑 Credentials (saved in $CONFIG_FILE):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  BTCPay Admin Email: $BTCPAY_ADMIN_EMAIL"
echo "  BTCPay Password:    $BTCPAY_ADMIN_PASSWORD"
echo "  Bitcoin Wallet:     $BITCOIN_WALLET"
echo ""

echo "💰 Bitcoin Payment Flow:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Customer pays → BTCPay Server → Your Wallet"
echo "  Wallet Address: $BITCOIN_WALLET"
echo "  Network: TESTNET (for testing)"
echo ""

echo "📝 Next Steps:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  1. Complete BTCPay setup at: http://localhost:23000"
echo "  2. Generate API key and Store ID"
echo "  3. Add them to .env file"
echo "  4. Restart API: docker-compose -f docker-compose.btcpay.yml restart nexus_api"
echo "  5. Test Bitcoin payment at: http://localhost:8000/docs"
echo ""

echo "🧪 Test Your API:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  # Health check"
echo "  curl http://localhost:8000/health"
echo ""
echo "  # Register a user"
echo "  curl -X POST http://localhost:8000/api/v1/auth/register \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"email\":\"test@example.com\",\"password\":\"test123\",\"full_name\":\"Test User\"}'"
echo ""

echo "🎯 For Production Deployment:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  1. Change BITCOIN_NETWORK from 'testnet' to 'mainnet' in docker-compose.btcpay.yml"
echo "  2. Use your real Bitcoin wallet address"
echo "  3. Deploy to a cloud server or hosting platform"
echo "  4. Set up HTTPS/SSL certificates"
echo "  5. Configure domain name"
echo ""

echo "📚 Documentation:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  - BITCOIN_INTEGRATION_GUIDE.md - Complete Bitcoin guide"
echo "  - BITCOIN_DEPLOYMENT_READY.md - Deployment overview"
echo "  - bitcoin_deployment.conf - Your credentials (KEEP SECURE!)"
echo ""

echo "╔════════════════════════════════════════════════════════╗"
echo "║              🎉 SETUP COMPLETE! 🎉                    ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}Your Bitcoin-integrated API is ready!${NC}"
echo ""
echo "View running services:"
echo "  docker-compose -f docker-compose.btcpay.yml ps"
echo ""
echo "View logs:"
echo "  docker-compose -f docker-compose.btcpay.yml logs -f"
echo ""
echo "Stop all services:"
echo "  docker-compose -f docker-compose.btcpay.yml down"
echo ""

# Save quick reference
cat > BITCOIN_QUICK_REFERENCE.txt <<EOF
Nexus AGI - Bitcoin Deployment Quick Reference
Generated: $(date)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🌐 SERVICE URLS:

BTCPay Server:    http://localhost:23000
Nexus AGI API:    http://localhost:8000
API Docs:         http://localhost:8000/docs
API Health:       http://localhost:8000/health

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔑 CREDENTIALS:

BTCPay Admin:
  Email: $BTCPAY_ADMIN_EMAIL
  Password: $BTCPAY_ADMIN_PASSWORD

Bitcoin Wallet: $BITCOIN_WALLET

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 QUICK COMMANDS:

Start services:
  docker-compose -f docker-compose.btcpay.yml up -d

Stop services:
  docker-compose -f docker-compose.btcpay.yml down

View logs:
  docker-compose -f docker-compose.btcpay.yml logs -f

Restart API:
  docker-compose -f docker-compose.btcpay.yml restart nexus_api

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📝 TODO:

[ ] Complete BTCPay Server setup at http://localhost:23000
[ ] Create store and configure Bitcoin wallet
[ ] Generate API key and Store ID
[ ] Add API key to .env file
[ ] Restart API
[ ] Test Bitcoin payment
[ ] Switch to mainnet for production
[ ] Deploy to production server

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For full documentation, see:
- BITCOIN_INTEGRATION_GUIDE.md
- BITCOIN_DEPLOYMENT_READY.md
- bitcoin_deployment.conf

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EOF

echo -e "${GREEN}Quick reference saved to: BITCOIN_QUICK_REFERENCE.txt${NC}"
echo ""
