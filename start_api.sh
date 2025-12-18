#!/bin/bash

# Nexus AGI API Gateway Startup Script
# This script initializes the database and starts the API server

echo "================================"
echo "🚀 Nexus AGI API Gateway Startup"
echo "================================"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please copy .env.example to .env and configure it"
    echo ""
    echo "  cp .env.example .env"
    echo ""
    exit 1
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Check required variables
if [ -z "$SECRET_KEY" ] || [ "$SECRET_KEY" = "your-secret-key-here-change-in-production" ]; then
    echo "❌ Error: SECRET_KEY not configured in .env"
    echo "Generate one with: openssl rand -hex 32"
    echo ""
    exit 1
fi

# Initialize database
echo "📊 Initializing database..."
python -c "from api_gateway.database import init_db; init_db()"

if [ $? -ne 0 ]; then
    echo "❌ Database initialization failed"
    exit 1
fi

echo "✅ Database initialized"
echo ""

# Start the API
echo "🌐 Starting API server..."
echo "   Environment: ${ENVIRONMENT:-development}"
echo "   Port: ${API_PORT:-8000}"
echo "   Debug: ${DEBUG:-true}"
echo ""

if [ "$ENVIRONMENT" = "production" ]; then
    # Production: Use gunicorn with multiple workers
    echo "🚀 Starting in production mode with gunicorn..."
    gunicorn api_gateway.main_complete:app \
        --workers 4 \
        --worker-class uvicorn.workers.UvicornWorker \
        --bind 0.0.0.0:${API_PORT:-8000} \
        --access-logfile logs/access.log \
        --error-logfile logs/error.log \
        --log-level info
else
    # Development: Use uvicorn with auto-reload
    echo "🔧 Starting in development mode with uvicorn..."
    uvicorn api_gateway.main_complete:app \
        --host 0.0.0.0 \
        --port ${API_PORT:-8000} \
        --reload \
        --log-level info
fi
