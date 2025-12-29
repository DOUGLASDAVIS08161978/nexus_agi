# ☁️ CLOUD DEPLOYMENT GUIDE - Deploy Nexus AGI in 5 Minutes

Deploy Nexus AGI to the cloud and make it publicly accessible for customers worldwide.

## 🎯 Platform Comparison

| Platform | Cost | Setup Time | Best For |
|----------|------|------------|----------|
| **Railway** | $5-20/mo | 2 min | Fastest, one-click deploy |
| **Render** | Free-$7/mo | 3 min | Free tier available |
| **AWS** | $10-50/mo | 10 min | Enterprise, scalable |
| **DigitalOcean** | $5-40/mo | 5 min | Developer-friendly |
| **Fly.io** | Free-$15/mo | 3 min | Edge computing |

**Recommended:** Railway (easiest) or Render (free tier)

---

## 🚂 OPTION 1: RAILWAY (RECOMMENDED)

Railway is the fastest way to deploy. One command, and you're live!

### Prerequisites
- GitHub account
- Railway account (free to sign up)

### Step 1: Install Railway CLI

```bash
npm install -g @railway/cli
```

### Step 2: Login

```bash
railway login
```

### Step 3: Deploy

```bash
# From your nexus_agi directory
railway init
railway link

# Deploy the API
railway up

# Get your public URL
railway domain
```

### Step 4: Set Environment Variables

```bash
# Set all required env vars
railway variables set STRIPE_SECRET_KEY=sk_live_...
railway variables set COINBASE_COMMERCE_API_KEY=...
railway variables set SECRET_KEY=$(openssl rand -hex 32)
railway variables set DATABASE_URL=railway-provided-postgres-url

# Or bulk import from .env
railway variables set --file .env
```

### Step 5: Configure Domain (Optional)

1. Go to Railway dashboard: [https://railway.app/dashboard](https://railway.app/dashboard)
2. Click your project → Settings
3. Add custom domain or use Railway subdomain
4. Your API will be live at: `https://your-app.railway.app`

### Railway Configuration

The `railway.json` file is already configured:

```json
{
  "build": {
    "builder": "NIXPACKS",
    "buildCommand": "pip install -r requirements-api.txt"
  },
  "deploy": {
    "startCommand": "uvicorn api_gateway.main:app --host 0.0.0.0 --port $PORT",
    "healthcheckPath": "/health",
    "restartPolicyType": "ON_FAILURE"
  }
}
```

**Cost:** ~$5-20/month depending on usage

---

## 🎨 OPTION 2: RENDER (FREE TIER)

Render offers a generous free tier perfect for getting started.

### Step 1: Connect GitHub

1. Go to [https://render.com](https://render.com)
2. Sign up with GitHub
3. Grant Render access to your repository

### Step 2: Create Web Service

1. Click "New +" → "Web Service"
2. Connect your GitHub repository
3. Configure:
   - **Name:** nexus-agi-api
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements-api.txt`
   - **Start Command:** `uvicorn api_gateway.main:app --host 0.0.0.0 --port $PORT`
   - **Plan:** Free (or Starter $7/mo for better performance)

### Step 3: Add Environment Variables

In Render dashboard, add:

```
SECRET_KEY=<generate with: openssl rand -hex 32>
STRIPE_SECRET_KEY=sk_live_...
COINBASE_COMMERCE_API_KEY=...
DATABASE_URL=<Render will provide this>
ENVIRONMENT=production
```

### Step 4: Deploy

1. Click "Create Web Service"
2. Render will automatically deploy
3. Your API will be live at: `https://nexus-agi-api.onrender.com`

### Step 5: Add PostgreSQL Database (Optional)

1. Click "New +" → "PostgreSQL"
2. Name it "nexus-agi-db"
3. Connect to your web service
4. Render will provide DATABASE_URL automatically

**Cost:** Free tier (sleeps after 15min inactivity) or $7/mo for always-on

---

## ☁️ OPTION 3: AWS (SCALABLE)

Deploy to AWS for enterprise-grade infrastructure.

### Step 1: Install AWS CLI

```bash
# macOS
brew install awscli

# Linux
pip install awscli

# Configure
aws configure
```

### Step 2: Create ECR Repository

```bash
# Create repository for Docker images
aws ecr create-repository --repository-name nexus-agi
```

### Step 3: Build and Push Docker Image

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Build image
docker build -f Dockerfile.api -t nexus-agi:latest .

# Tag image
docker tag nexus-agi:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/nexus-agi:latest

# Push to ECR
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/nexus-agi:latest
```

### Step 4: Deploy with ECS (Elastic Container Service)

#### Option A: Using AWS Console

1. Go to ECS console: [https://console.aws.amazon.com/ecs](https://console.aws.amazon.com/ecs)
2. Create cluster: "nexus-agi-cluster"
3. Create task definition:
   - Container image: Your ECR image URL
   - Port: 8000
   - Environment variables: Add all from .env
4. Create service
5. Configure Application Load Balancer
6. Deploy!

#### Option B: Using AWS CLI

```bash
# Create cluster
aws ecs create-cluster --cluster-name nexus-agi-cluster

# Register task definition (create task-definition.json first)
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service \
  --cluster nexus-agi-cluster \
  --service-name nexus-agi-service \
  --task-definition nexus-agi:1 \
  --desired-count 2 \
  --launch-type FARGATE
```

### Step 5: Set Up RDS PostgreSQL

```bash
# Create database
aws rds create-db-instance \
  --db-instance-identifier nexus-agi-db \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --master-username admin \
  --master-user-password YOUR_SECURE_PASSWORD \
  --allocated-storage 20
```

### Step 6: Configure Domain with Route 53

1. Go to Route 53
2. Create hosted zone
3. Add A record pointing to Load Balancer
4. Configure SSL with ACM (AWS Certificate Manager)

**Cost:** ~$20-100/month depending on traffic

---

## 🌊 OPTION 4: DIGITALOCEAN (SIMPLE VPS)

Deploy to a simple DigitalOcean droplet.

### Step 1: Create Droplet

1. Go to [https://www.digitalocean.com](https://www.digitalocean.com)
2. Create → Droplets
3. Choose:
   - **Image:** Ubuntu 22.04 LTS
   - **Plan:** Basic $6/mo (1GB RAM)
   - **Datacenter:** Closest to your users
   - **Add SSH key**

### Step 2: Connect to Droplet

```bash
ssh root@YOUR_DROPLET_IP
```

### Step 3: Install Dependencies

```bash
# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
apt install docker-compose -y

# Install Git
apt install git -y
```

### Step 4: Clone and Deploy

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/nexus_agi.git
cd nexus_agi

# Create .env file
nano .env
# (paste your configuration)

# Deploy with Docker Compose
docker-compose -f docker-compose.api.yml up -d
```

### Step 5: Configure Nginx Reverse Proxy

```bash
# Install Nginx
apt install nginx -y

# Create Nginx config
cat > /etc/nginx/sites-available/nexus-agi <<EOF
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOF

# Enable site
ln -s /etc/nginx/sites-available/nexus-agi /etc/nginx/sites-enabled/
nginx -t
systemctl restart nginx
```

### Step 6: Set Up SSL with Let's Encrypt

```bash
# Install Certbot
apt install certbot python3-certbot-nginx -y

# Get SSL certificate
certbot --nginx -d your-domain.com

# Auto-renew setup (already configured)
```

**Cost:** $6-40/month depending on droplet size

---

## ✈️ OPTION 5: FLY.IO (EDGE DEPLOYMENT)

Deploy to Fly.io for edge computing and global distribution.

### Step 1: Install Fly CLI

```bash
curl -L https://fly.io/install.sh | sh
```

### Step 2: Login and Launch

```bash
# Login
fly auth login

# Launch app
fly launch

# Follow prompts:
# - App name: nexus-agi
# - Region: closest to you
# - PostgreSQL: yes
# - Redis: optional
```

### Step 3: Configure

Edit `fly.toml`:

```toml
app = "nexus-agi"

[build]
  dockerfile = "Dockerfile.api"

[env]
  PORT = "8000"
  ENVIRONMENT = "production"

[[services]]
  internal_port = 8000
  protocol = "tcp"

  [[services.ports]]
    port = 80
    handlers = ["http"]

  [[services.ports]]
    port = 443
    handlers = ["tls", "http"]
```

### Step 4: Set Secrets

```bash
fly secrets set SECRET_KEY=$(openssl rand -hex 32)
fly secrets set STRIPE_SECRET_KEY=sk_live_...
fly secrets set COINBASE_COMMERCE_API_KEY=...
```

### Step 5: Deploy

```bash
fly deploy
```

**Cost:** Free tier (3GB/month bandwidth) or $1.94/mo for 256MB RAM

---

## 🔒 POST-DEPLOYMENT SECURITY

### 1. Set Up SSL/HTTPS

All platforms above provide automatic SSL. Verify:

```bash
curl https://your-domain.com/health
```

### 2. Configure Firewall

```bash
# DigitalOcean/VPS only
ufw allow 22    # SSH
ufw allow 80    # HTTP
ufw allow 443   # HTTPS
ufw enable
```

### 3. Set Up Monitoring

**Health Check Endpoint:**

```bash
# Add to cron for monitoring
*/5 * * * * curl https://your-domain.com/health || echo "API DOWN!"
```

**Use UptimeRobot (free):**
1. Go to [https://uptimerobot.com](https://uptimerobot.com)
2. Add monitor: `https://your-domain.com/health`
3. Get alerts via email/SMS/Slack

### 4. Environment Variables Checklist

```bash
# Required
SECRET_KEY=<generated>
STRIPE_SECRET_KEY=sk_live_...
DATABASE_URL=<provided by platform>

# Recommended
COINBASE_COMMERCE_API_KEY=...
ENVIRONMENT=production
LOG_LEVEL=INFO

# Optional but useful
SENTRY_DSN=<for error tracking>
REDIS_URL=<for caching>
```

---

## 📊 PERFORMANCE OPTIMIZATION

### 1. Enable Caching

Add Redis to your deployment:

```bash
# Railway
railway add redis

# Render
# Add Redis service from dashboard

# DigitalOcean
docker run -d -p 6379:6379 redis:alpine
```

Update `.env`:

```bash
REDIS_URL=redis://localhost:6379/0
```

### 2. Configure Auto-Scaling

**Railway:**
- Automatically scales based on traffic

**AWS ECS:**
```bash
# Configure auto-scaling
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/nexus-agi-cluster/nexus-agi-service \
  --min-capacity 1 \
  --max-capacity 10
```

### 3. Add CDN (Optional)

Use Cloudflare for caching and DDoS protection:

1. Go to [https://cloudflare.com](https://cloudflare.com)
2. Add your domain
3. Update nameservers
4. Enable caching for static assets

---

## 🎯 DEPLOYMENT CHECKLIST

### Pre-Deployment
- [ ] All code committed to Git
- [ ] `.env` file configured with production values
- [ ] Database migrations ready
- [ ] SSL certificates configured
- [ ] Payment webhooks tested

### During Deployment
- [ ] Platform account created
- [ ] Repository connected
- [ ] Environment variables set
- [ ] Database provisioned
- [ ] Initial deployment successful

### Post-Deployment
- [ ] Health check endpoint working
- [ ] API documentation accessible at `/docs`
- [ ] Payment webhooks configured
- [ ] Stripe webhook URL updated
- [ ] Coinbase webhook URL updated
- [ ] Domain configured (if custom)
- [ ] SSL working (HTTPS)
- [ ] Monitoring set up
- [ ] Test payment completed
- [ ] Marketing agent activated

---

## 🚨 TROUBLESHOOTING

### Deployment Failed

```bash
# Check logs
railway logs  # Railway
render logs   # Render
aws logs tail /ecs/nexus-agi  # AWS

# Or Docker logs locally
docker-compose logs -f
```

### Database Connection Error

```bash
# Verify DATABASE_URL is set
echo $DATABASE_URL

# Test connection
psql $DATABASE_URL
```

### Environment Variables Not Loading

```bash
# Railway: Check variables set
railway variables

# Render: Check dashboard environment tab

# AWS: Check task definition environment section
```

### 502 Bad Gateway

Usually means the app isn't starting:

1. Check logs for errors
2. Verify PORT environment variable
3. Ensure app listens on `0.0.0.0` not `localhost`
4. Check health check endpoint

---

## 💰 COST OPTIMIZATION

### Free Tier Options

1. **Render Free:** Free tier (sleeps after 15min)
2. **Fly.io Free:** 3 free VMs (256MB each)
3. **AWS Free Tier:** 12 months free (limited resources)
4. **Netlify/Vercel:** Free for static dashboard

### Budget-Friendly Stack

- **Backend API:** Render ($7/mo) or Railway ($5/mo)
- **Database:** Railway PostgreSQL (included) or Render ($7/mo)
- **Caching:** Upstash Redis (free tier)
- **Monitoring:** UptimeRobot (free)
- **CDN:** Cloudflare (free)

**Total:** $7-15/month for production-ready deployment!

---

## 🎉 YOU'RE DEPLOYED!

Your Nexus AGI is now live and accessible to customers worldwide!

### Next Steps:

1. **Update Payment Webhooks:**
   - Stripe webhook → `https://your-domain.com/api/stripe/webhook`
   - Coinbase webhook → `https://your-domain.com/api/coinbase/webhook`

2. **Activate Marketing:**
   ```bash
   python autonomous_marketing_agent.py
   ```

3. **Monitor Revenue:**
   - Check Stripe dashboard daily
   - Monitor API usage at `/api/analytics`

4. **Scale as Needed:**
   - Most platforms auto-scale
   - Upgrade plan when hitting limits

---

**🚀 CONGRATULATIONS! YOU'RE LIVE AND READY TO EARN! 🚀**

Your AGI is now serving customers 24/7 around the globe!

---

*For support or questions, check the docs or join the community.*
