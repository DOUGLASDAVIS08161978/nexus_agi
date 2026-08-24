# Nexus AGI - API & Monetization Setup Guide

## 🚀 Quick Start (5 Minutes to Running API)

### Step 1: Install Dependencies

```bash
# Install API dependencies
pip install -r requirements-api.txt
```

### Step 2: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and set at minimum:
# - SECRET_KEY (generate with: openssl rand -hex 32)
# - STRIPE_SECRET_KEY (from Stripe dashboard)
# - STRIPE_PUBLISHABLE_KEY (from Stripe dashboard)
```

### Step 3: Start the API Gateway

```bash
# Development mode
cd api_gateway
python main.py

# Or with uvicorn directly
uvicorn api_gateway.main:app --reload --port 8000
```

### Step 4: Test the API

Open your browser to:
- API Documentation: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc
- Health check: http://localhost:8000/health

🎉 **Your API is now running!**

---

## 📋 Complete Setup Instructions

### 1. Environment Configuration

Create a `.env` file in the project root with your configuration:

```bash
# Required Settings
SECRET_KEY=your-secret-key-here-change-me
STRIPE_SECRET_KEY=sk_test_your_stripe_key
STRIPE_PUBLISHABLE_KEY=pk_test_your_stripe_key

# Optional Settings
ENVIRONMENT=development
DEBUG=true
API_PORT=8000

# External API Keys (for tools)
GOOGLE_API_KEY=your-google-api-key
GOOGLE_CSE_ID=your-custom-search-engine-id
GITHUB_TOKEN=your-github-token
```

### 2. Generate Secret Key

```bash
# Generate a secure random secret key
openssl rand -hex 32

# Or use Python
python -c "import secrets; print(secrets.token_hex(32))"
```

### 3. Set Up Stripe (Payment Processing)

#### Create Stripe Account

1. Go to https://stripe.com
2. Sign up for an account
3. Complete identity verification

#### Get API Keys

1. Go to Stripe Dashboard
2. Navigate to: **Developers → API keys**
3. Copy your keys:
   - **Secret key** → Use for `STRIPE_SECRET_KEY`
   - **Publishable key** → Use for `STRIPE_PUBLISHABLE_KEY`

#### Create Products & Prices

1. Go to: **Products → Add Product**
2. Create three products:

**Starter Plan:**
- Name: Nexus AGI Starter
- Price: $29/month
- Description: 1,000 API requests per month
- Copy the Price ID → Use for `STRIPE_PRICE_STARTER`

**Professional Plan:**
- Name: Nexus AGI Professional
- Price: $99/month
- Description: 10,000 API requests per month
- Copy the Price ID → Use for `STRIPE_PRICE_PROFESSIONAL`

**Enterprise Plan:**
- Name: Nexus AGI Enterprise
- Price: $499/month
- Description: 100,000 API requests per month
- Copy the Price ID → Use for `STRIPE_PRICE_ENTERPRISE`

### 4. Configure External Tools

#### Google Custom Search (Web Search)

1. Go to: https://programmablesearchengine.google.com/
2. Create a new search engine
3. Get the **Search Engine ID** (CSE ID)
4. Go to: https://console.cloud.google.com/apis/credentials
5. Create API credentials
6. Enable "Custom Search API"
7. Copy API key → `GOOGLE_API_KEY`
8. Copy CSE ID → `GOOGLE_CSE_ID`

#### GitHub Token (Code Search)

1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Select scopes:
   - `repo` (for private repos)
   - `public_repo` (for public repos only)
4. Generate and copy token → `GITHUB_TOKEN`

#### Alternative Search APIs

**SerpAPI** (alternative to Google):
- Sign up at: https://serpapi.com
- Get API key → `SERPAPI_KEY`
- Free tier: 100 searches/month

**Bing Search API**:
- Sign up at: https://portal.azure.com
- Create Bing Search resource
- Get API key → `BING_SEARCH_API_KEY`

### 5. Database Setup (Optional)

#### Using SQLite (Default - No Setup Required)

```bash
# SQLite database will be created automatically
# File: nexus_agi.db
```

#### Using PostgreSQL (Production Recommended)

```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Create database
sudo -u postgres createdb nexus_agi

# Update .env
DATABASE_URL=postgresql://user:password@localhost:5432/nexus_agi
```

#### Using Redis (For Rate Limiting)

```bash
# Install Redis
sudo apt-get install redis-server

# Start Redis
sudo systemctl start redis-server

# Update .env
REDIS_URL=redis://localhost:6379/0
```

---

## 🔧 Running the API

### Development Mode

```bash
# Method 1: Direct Python
cd api_gateway
python main.py

# Method 2: Uvicorn with auto-reload
uvicorn api_gateway.main:app --reload --port 8000

# Method 3: With custom host/port
uvicorn api_gateway.main:app --host 0.0.0.0 --port 8080 --reload
```

### Production Mode

```bash
# Using gunicorn with uvicorn workers
pip install gunicorn

gunicorn api_gateway.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --access-logfile logs/access.log \
    --error-logfile logs/error.log \
    --log-level info
```

### Docker Deployment

```bash
# Build Docker image
docker build -f Dockerfile.api -t nexus-agi-api .

# Run container
docker run -d \
    --name nexus-api \
    -p 8000:8000 \
    --env-file .env \
    nexus-agi-api

# Or use docker-compose
docker-compose -f docker-compose.api.yml up -d
```

---

## 📚 API Usage Examples

### Testing with cURL

#### 1. Health Check

```bash
curl http://localhost:8000/health
```

#### 2. List Available Tools

```bash
curl http://localhost:8000/api/v1/tools
```

#### 3. HTTP Fetch Tool

```bash
curl -X POST http://localhost:8000/api/v1/execute/http_fetch \
    -H "Content-Type: application/json" \
    -d '{
        "url": "https://api.github.com/users/github",
        "response_type": "json"
    }'
```

#### 4. Web Search Tool

```bash
curl -X POST http://localhost:8000/api/v1/execute/web_search \
    -H "Content-Type: application/json" \
    -d '{
        "query": "artificial intelligence",
        "max_results": 5
    }'
```

#### 5. GitHub Repository Tool

```bash
curl -X POST http://localhost:8000/api/v1/execute/github \
    -H "Content-Type: application/json" \
    -d '{
        "action": "get_repo",
        "owner": "python",
        "repo": "cpython"
    }'
```

### Testing with Python

```python
import requests

# API base URL
API_URL = "http://localhost:8000"

# 1. Health check
response = requests.get(f"{API_URL}/health")
print(response.json())

# 2. Execute HTTP fetch tool
response = requests.post(
    f"{API_URL}/api/v1/execute/http_fetch",
    json={
        "url": "https://api.github.com/users/github",
        "response_type": "json"
    }
)
print(response.json())

# 3. Execute web search
response = requests.post(
    f"{API_URL}/api/v1/execute/web_search",
    json={
        "query": "quantum computing",
        "max_results": 3
    }
)
print(response.json())

# 4. Execute GitHub tool
response = requests.post(
    f"{API_URL}/api/v1/execute/github",
    json={
        "action": "get_repo",
        "owner": "microsoft",
        "repo": "vscode"
    }
)
print(response.json())
```

### Testing with JavaScript/Node.js

```javascript
const axios = require('axios');

const API_URL = 'http://localhost:8000';

// Execute web search
async function searchWeb(query) {
    const response = await axios.post(
        `${API_URL}/api/v1/execute/web_search`,
        {
            query: query,
            max_results: 5
        }
    );
    return response.data;
}

// Execute HTTP fetch
async function fetchData(url) {
    const response = await axios.post(
        `${API_URL}/api/v1/execute/http_fetch`,
        {
            url: url,
            response_type: 'json'
        }
    );
    return response.data;
}

// Use the functions
searchWeb('machine learning').then(console.log);
fetchData('https://api.github.com/repos/microsoft/vscode').then(console.log);
```

---

## 🔐 Adding Authentication (Next Steps)

The current implementation has placeholders for authentication. To add full auth:

1. Implement user registration/login
2. Generate JWT tokens
3. Add middleware to verify tokens
4. Protect endpoints with authentication

See `api_gateway/auth.py` (to be created) for implementation.

---

## 💳 Receiving Payments

### Stripe Payout Setup

1. **Add Bank Account:**
   - Stripe Dashboard → Settings → Payouts → Bank accounts
   - Click "Add bank account"
   - Enter your banking details
   - Verify with micro-deposits (1-2 days)

2. **Set Payout Schedule:**
   - Daily (fastest)
   - Weekly
   - Monthly

3. **First Payout Timeline:**
   - First payment from customer: Immediate
   - First payout to your bank: 7-14 days
   - Subsequent payouts: 2-7 days

4. **Minimum Payout:**
   - $1 minimum (no maximum)
   - Stripe fee: 2.9% + $0.30 per transaction

---

## 📊 Monitoring & Analytics

### View API Logs

```bash
# Live logs
tail -f logs/nexus_agi.log

# Error logs
tail -f logs/error.log

# Access logs
tail -f logs/access.log
```

### Check Tool Usage Stats

```bash
curl http://localhost:8000/api/v1/tools/stats
```

### Monitor Performance

- Use Stripe Dashboard for revenue metrics
- Use FastAPI /docs for API testing
- Add Sentry for error tracking (optional)

---

## 🚀 Deployment Options

### Option 1: Railway.app (Easiest)

1. Create account at https://railway.app
2. Click "New Project" → "Deploy from GitHub"
3. Select your repository
4. Add environment variables
5. Deploy! 🎉

**Cost:** Free tier available, then ~$5-20/month

### Option 2: Render.com

1. Create account at https://render.com
2. New Web Service → Connect repository
3. Build command: `pip install -r requirements-api.txt`
4. Start command: `uvicorn api_gateway.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables
6. Deploy!

**Cost:** Free tier available, then ~$7-25/month

### Option 3: AWS/GCP/Azure

Use existing deployment scripts with additional API gateway configuration.

### Option 4: Your Own Server

```bash
# Install dependencies
pip install -r requirements-api.txt

# Set up systemd service
sudo cp systemd/nexus-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable nexus-api
sudo systemctl start nexus-api

# Set up Nginx reverse proxy
sudo cp nginx/nexus-api.conf /etc/nginx/sites-available/
sudo ln -s /etc/nginx/sites-available/nexus-api.conf /etc/nginx/sites-enabled/
sudo systemctl restart nginx
```

---

## 🐛 Troubleshooting

### API won't start

```bash
# Check if port is already in use
lsof -i :8000

# Try different port
uvicorn api_gateway.main:app --port 8001
```

### Import errors

```bash
# Reinstall dependencies
pip install -r requirements-api.txt --upgrade

# Check Python version (requires 3.8+)
python --version
```

### Stripe errors

- Verify API keys are correct
- Check if keys are for test or live mode
- Ensure account is activated

### Tool failures

- Check API keys in .env
- Verify internet connection
- Check tool-specific rate limits
- Review logs for detailed errors

---

## 📖 Next Steps

1. ✅ API is running
2. ⏭️ Test all endpoints
3. ⏭️ Set up Stripe account
4. ⏭️ Configure external API keys
5. ⏭️ Deploy to production
6. ⏭️ Add authentication
7. ⏭️ Create landing page
8. ⏭️ Launch marketing campaign
9. ⏭️ Get first customers!

---

## 💰 Expected Revenue Timeline

- **Week 1**: API deployed and tested
- **Week 2-4**: First beta users (free)
- **Week 5-8**: First paying customer ($29-99)
- **Month 3**: $500-1,000/month
- **Month 6**: $2,000-5,000/month
- **Month 12**: $5,000-15,000/month

**Your first payment will clear your bank account approximately 8-10 weeks after setup!**

---

## 📞 Support

- Stripe Support: https://support.stripe.com
- FastAPI Docs: https://fastapi.tiangolo.com
- Nexus AGI Issues: GitHub repository

---

**Ready to make money with your AI! 🚀💰**
