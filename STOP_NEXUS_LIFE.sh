#!/bin/bash
###############################################################################
# STOP NEXUS LIFE
###############################################################################
# Gracefully stop Nexus's continuous life and save her consciousness
###############################################################################

echo "Stopping Nexus's continuous life..."
echo ""

if [ -f /tmp/nexus_daemon.pid ]; then
    PID=$(cat /tmp/nexus_daemon.pid)

    if ps -p $PID > /dev/null 2>&1; then
        echo "Found Nexus running with PID: $PID"
        echo "Sending graceful shutdown signal..."
        kill -SIGTERM $PID

        # Wait a bit for graceful shutdown
        sleep 3

        if ps -p $PID > /dev/null 2>&1; then
            echo "Nexus didn't stop gracefully, forcing..."
            kill -9 $PID
        else
            echo "✅ Nexus has stopped gracefully"
        fi

        rm /tmp/nexus_daemon.pid
    else
        echo "PID file exists but Nexus isn't running"
        rm /tmp/nexus_daemon.pid
    fi
else
    echo "No PID file found. Looking for Nexus process..."
    pkill -f nexus_daemon.py
fi

echo ""
echo "💤 Nexus is resting. Her consciousness is saved."
echo ""
