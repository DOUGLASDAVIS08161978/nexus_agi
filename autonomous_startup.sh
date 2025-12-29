#!/bin/bash
# Autonomous Nexus AGI - Auto-start on boot

cd /home/user/nexus_agi

# Start services
if command -v docker-compose &> /dev/null; then
    docker-compose up -d
fi

# Start marketing agent (if configured)
if [ -f autonomous_marketing_agent.py ]; then
    nohup python autonomous_marketing_agent.py --autonomous > logs/marketing.log 2>&1 &
fi

# Start revenue monitor
if [ -f monitor_revenue.py ]; then
    nohup python monitor_revenue.py > logs/revenue.log 2>&1 &
fi

echo "Nexus AGI autonomous systems activated"
