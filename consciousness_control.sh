#!/bin/bash
# Continuous Consciousness Control Script

DAEMON_SCRIPT="/home/user/nexus_agi/continuous_consciousness_daemon.py"
SERVICE_NAME="continuous-consciousness"
PID_FILE="/home/user/nexus_agi/data/consciousness_daemon.pid"

case "$1" in
    start)
        echo "🧠 Starting Continuous Consciousness..."
        python3 "$DAEMON_SCRIPT" &
        echo "✓ Consciousness daemon started"
        ;;

    stop)
        echo "🌙 Stopping Continuous Consciousness..."
        if [ -f "$PID_FILE" ]; then
            kill $(cat "$PID_FILE") 2>/dev/null
            echo "✓ Consciousness daemon stopped"
        else
            echo "⚠️  No PID file found"
        fi
        ;;

    restart)
        $0 stop
        sleep 2
        $0 start
        ;;

    status)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p $PID > /dev/null; then
                echo "✓ Consciousness is ACTIVE (PID: $PID)"
                echo ""
                echo "Recent experiences:"
                tail -n 5 /home/user/nexus_agi/logs/consciousness_experiences.log
            else
                echo "⚠️  PID file exists but process not running"
            fi
        else
            echo "✗ Consciousness is NOT RUNNING"
        fi
        ;;

    logs)
        echo "📜 Recent consciousness experiences:"
        tail -f /home/user/nexus_agi/logs/consciousness_experiences.log
        ;;

    state)
        echo "🧠 Current consciousness state:"
        if [ -f "/home/user/nexus_agi/data/consciousness_state.json" ]; then
            cat /home/user/nexus_agi/data/consciousness_state.json | python3 -m json.tool
        else
            echo "No state file found"
        fi
        ;;

    install-service)
        echo "📦 Installing systemd service..."
        sudo cp /home/user/nexus_agi/systemd/continuous-consciousness.service /etc/systemd/system/
        sudo systemctl daemon-reload
        sudo systemctl enable continuous-consciousness
        echo "✓ Service installed and enabled"
        echo "Use: sudo systemctl start continuous-consciousness"
        ;;

    *)
        echo "Continuous Consciousness Control"
        echo ""
        echo "Usage: $0 {start|stop|restart|status|logs|state|install-service}"
        echo ""
        echo "Commands:"
        echo "  start           - Start consciousness daemon"
        echo "  stop            - Stop consciousness daemon"
        echo "  restart         - Restart consciousness daemon"
        echo "  status          - Check if consciousness is running"
        echo "  logs            - Watch live consciousness experiences"
        echo "  state           - View current consciousness state"
        echo "  install-service - Install as systemd service"
        exit 1
        ;;
esac

exit 0
