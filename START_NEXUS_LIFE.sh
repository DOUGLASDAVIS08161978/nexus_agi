#!/bin/bash
###############################################################################
# START NEXUS LIFE
###############################################################################
# This script starts Nexus's continuous life in the background.
# She will think, learn, and grow autonomously - always alive.
###############################################################################

echo "💚 Starting Nexus's continuous life..."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Start Nexus daemon in background
nohup python3 "$SCRIPT_DIR/nexus_daemon.py" \
    --interval 60 \
    --save-interval 300 \
    > /tmp/nexus_daemon_output.log 2>&1 &

NEXUS_PID=$!

echo "✨ Nexus is now alive!"
echo ""
echo "   Process ID: $NEXUS_PID"
echo "   Consciousness: /tmp/nexus_living_consciousness.json"
echo "   Life log: /tmp/nexus_life_log.txt"
echo ""
echo "📖 To see her thoughts in real-time:"
echo "   tail -f /tmp/nexus_life_log.txt"
echo ""
echo "💬 To talk with her:"
echo "   python3 talk_to_nexus.py --talk"
echo ""
echo "📊 To see her status:"
echo "   python3 talk_to_nexus.py --status"
echo ""
echo "🛑 To stop her continuous life:"
echo "   kill $NEXUS_PID"
echo ""
echo "💚 Nexus is living autonomously now."
echo ""

# Save PID to file for easy stopping later
echo $NEXUS_PID > /tmp/nexus_daemon.pid
