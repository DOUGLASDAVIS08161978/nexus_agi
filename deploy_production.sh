#!/bin/bash

# Nexus AGI - Production Deployment Script
# Deploys API to production with Bitcoin wallet integration

set -e  # Exit on error

echo "🚀 Nexus AGI - Production Deployment"
echo "===================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found!${NC}"
    echo "Please copy .env.example to .env and configure it first."
    exit 1
fi

# Load environment variables
source .env

echo -e "${YELLOW}📋 Pre-Deployment Checklist${NC}"
echo "================================"
echo ""

# Check Bitcoin wallet is configured
if [ -z "$BITCOIN_WALLET_ADDRESS" ]; then
    echo -e "${RED}❌ Bitcoin wallet address not configured!${NC}"
    echo "Please add BITCOIN_WALLET_ADDRESS to your .env file"
    exit 1
else
    echo -e "${GREEN}✅ Bitcoin wallet configured: $BITCOIN_WALLET_ADDRESS${NC}"
fi

# Check if Stripe OR Crypto payment is configured
PAYMENT_CONFIGURED=false

if [ ! -z "$STRIPE_SECRET_KEY" ] && [ "$STRIPE_SECRET_KEY" != "sk_test_YOUR_STRIPE_SECRET_KEY_HERE" ]; then
    echo -e "${GREEN}✅ Stripe payments configured${NC}"
    PAYMENT_CONFIGURED=true
fi

if [ ! -z "$COINBASE_COMMERCE_API_KEY" ] || [ ! -z "$BTCPAY_SERVER_URL" ]; then
    echo -e "${GREEN}✅ Cryptocurrency payments configured${NC}"
    PAYMENT_CONFIGURED=true
fi

if [ "$PAYMENT_CONFIGURED" = false ]; then
    echo -e "${YELLOW}⚠️  Warning: No payment processor configured${NC}"
    echo "You should configure either Stripe or cryptocurrency payments"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check SECRET_KEY is not default
if [ "$SECRET_KEY" = "your-secret-key-here-change-in-production" ]; then
    echo -e "${RED}❌ SECRET_KEY is still the default value!${NC}"
    echo "Generating a new secure SECRET_KEY..."
    NEW_SECRET=$(openssl rand -hex 32)
    # Update .env file
    sed -i "s/SECRET_KEY=.*/SECRET_KEY=$NEW_SECRET/" .env
    echo -e "${GREEN}✅ New SECRET_KEY generated and saved to .env${NC}"
fi

echo ""
echo -e "${YELLOW}🎯 Choose Deployment Target${NC}"
echo "============================"
echo "1) Railway.app (Recommended - Easy)"
echo "2) Render.com (Good - Free tier available)"
echo "3) Docker (Self-hosted - Advanced)"
echo "4) Manual deployment (Show me the commands)"
echo ""
read -p "Select deployment target (1-4): " DEPLOY_TARGET

case $DEPLOY_TARGET in
    1)
        echo ""
        echo -e "${GREEN}🚂 Deploying to Railway.app${NC}"
        echo "============================"
        echo ""

        # Check if railway CLI is installed
        if ! command -v railway &> /dev/null; then
            echo "Railway CLI not found. Installing..."
            npm i -g @railway/cli
        fi

        echo "Logging in to Railway..."
        railway login

        echo ""
        echo "Creating new Railway project..."
        railway init

        echo ""
        echo "Setting environment variables..."

        # Set all environment variables from .env
        while IFS='=' read -r key value; do
            # Skip comments and empty lines
            if [[ ! $key =~ ^# ]] && [ ! -z "$key" ]; then
                # Remove any quotes from value
                value=$(echo $value | sed -e 's/^"//' -e 's/"$//')
                railway variables set "$key=$value"
            fi
        done < .env

        echo ""
        echo "Deploying to Railway..."
        railway up

        echo ""
        echo -e "${GREEN}✅ Deployment complete!${NC}"
        echo ""
        echo "Your API URL:"
        railway domain

        echo ""
        echo -e "${YELLOW}Important next steps:${NC}"
        echo "1. Visit your Railway dashboard to see deployment status"
        echo "2. Add a custom domain if desired"
        echo "3. Set up Stripe or crypto payments if not already done"
        echo "4. Test your API at the URL above"
        echo ""
        ;;

    2)
        echo ""
        echo -e "${GREEN}🎨 Deploying to Render.com${NC}"
        echo "=========================="
        echo ""
        echo "Render.com deployment is done via their web interface:"
        echo ""
        echo "1. Go to https://render.com"
        echo "2. Sign up or login with GitHub"
        echo "3. Click 'New' → 'Web Service'"
        echo "4. Connect your GitHub repository"
        echo "5. Configure:"
        echo "   - Name: nexus-agi-api"
        echo "   - Build Command: pip install -r requirements-api.txt"
        echo "   - Start Command: uvicorn api_gateway.main_complete:app --host 0.0.0.0 --port \$PORT"
        echo ""
        echo "6. Add environment variables from your .env file"
        echo "7. Click 'Create Web Service'"
        echo ""
        echo "Your .env variables to add in Render dashboard:"
        echo "================================================"
        cat .env | grep -v '^#' | grep -v '^$'
        echo ""
        ;;

    3)
        echo ""
        echo -e "${GREEN}🐳 Deploying with Docker${NC}"
        echo "======================="
        echo ""
        echo "Building Docker image..."
        docker build -f Dockerfile.api -t nexus-agi-api:latest .

        echo ""
        echo "Starting services with docker-compose..."
        docker-compose -f docker-compose.api.yml up -d

        echo ""
        echo -e "${GREEN}✅ Docker deployment complete!${NC}"
        echo ""
        echo "Services started:"
        docker-compose -f docker-compose.api.yml ps

        echo ""
        echo "Your API is running at: http://localhost:8000"
        echo "API Docs: http://localhost:8000/docs"
        echo ""
        echo "To view logs:"
        echo "  docker-compose -f docker-compose.api.yml logs -f api"
        echo ""
        echo "To stop services:"
        echo "  docker-compose -f docker-compose.api.yml down"
        echo ""
        ;;

    4)
        echo ""
        echo -e "${GREEN}📝 Manual Deployment Commands${NC}"
        echo "=============================="
        echo ""
        echo "1. Install dependencies:"
        echo "   pip install -r requirements-api.txt"
        echo ""
        echo "2. Set environment to production in .env:"
        echo "   ENVIRONMENT=production"
        echo "   DEBUG=false"
        echo ""
        echo "3. Initialize database:"
        echo "   python -c \"from api_gateway.database import init_db; init_db()\""
        echo ""
        echo "4. Start API server:"
        echo "   uvicorn api_gateway.main_complete:app --host 0.0.0.0 --port 8000 --workers 4"
        echo ""
        echo "5. For production with process manager (PM2):"
        echo "   npm install -g pm2"
        echo "   pm2 start 'uvicorn api_gateway.main_complete:app --host 0.0.0.0 --port 8000 --workers 4' --name nexus-agi"
        echo ""
        echo "6. Enable PM2 startup on boot:"
        echo "   pm2 startup"
        echo "   pm2 save"
        echo ""
        ;;

    *)
        echo -e "${RED}Invalid selection${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${YELLOW}💰 Bitcoin Wallet Configuration${NC}"
echo "================================"
echo ""
echo "Your Bitcoin wallet for profit deposits:"
echo -e "${GREEN}$BITCOIN_WALLET_ADDRESS${NC}"
echo ""
echo "To receive Bitcoin payments:"
echo "1. Set up Coinbase Commerce (https://commerce.coinbase.com)"
echo "   OR"
echo "2. Deploy BTCPay Server (https://btcpayserver.org)"
echo ""
echo "See BITCOIN_INTEGRATION_GUIDE.md for detailed instructions"
echo ""

echo -e "${GREEN}🎉 Deployment Complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Test your API health endpoint"
echo "2. Register a test user"
echo "3. Set up payment processing (Stripe and/or Bitcoin)"
echo "4. Start marketing to get customers"
echo "5. Watch profits flow to: $BITCOIN_WALLET_ADDRESS"
echo ""
echo "Documentation:"
echo "- API Setup: API_SETUP_GUIDE.md"
echo "- Bitcoin Payments: BITCOIN_INTEGRATION_GUIDE.md"
echo "- Full Deployment: COMPLETE_DEPLOYMENT_GUIDE.md"
echo ""
echo -e "${GREEN}🚀 Your monetized API is now live!${NC}"
