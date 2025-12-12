#!/bin/bash
# Nexus AGI BCI Daemon Control Script
# Controls the BCI daemon service

DAEMON_NAME="nexus-bci-daemon"
SERVICE_FILE="/home/user/nexus_agi/systemd/nexus-bci-daemon.service"
PID_FILE="/tmp/nexus_agi_bci_daemon.pid"
LOG_DIR="/home/user/nexus_agi/logs/bci_daemon"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Nexus AGI BCI Daemon Control${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Function to start daemon in background
start_background() {
    print_header
    echo -e "${YELLOW}Starting BCI daemon in background...${NC}"

    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo -e "${RED}✗ Daemon is already running (PID: $PID)${NC}"
            return 1
        else
            echo -e "${YELLOW}Removing stale PID file${NC}"
            rm -f "$PID_FILE"
        fi
    fi

    # Start daemon in background
    cd /home/user/nexus_agi
    nohup python3 bci_daemon.py > "$LOG_DIR/daemon_nohup.log" 2>&1 &
    DAEMON_PID=$!

    sleep 2

    if ps -p $DAEMON_PID > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Daemon started successfully${NC}"
        echo -e "${GREEN}  PID: $DAEMON_PID${NC}"
        echo -e "${GREEN}  Logs: $LOG_DIR/${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed to start daemon${NC}"
        return 1
    fi
}

# Function to stop daemon
stop_daemon() {
    print_header
    echo -e "${YELLOW}Stopping BCI daemon...${NC}"

    if [ ! -f "$PID_FILE" ]; then
        echo -e "${RED}✗ PID file not found${NC}"
        return 1
    fi

    PID=$(cat "$PID_FILE")

    if ! ps -p $PID > /dev/null 2>&1; then
        echo -e "${RED}✗ Daemon is not running${NC}"
        rm -f "$PID_FILE"
        return 1
    fi

    echo -e "${YELLOW}Sending SIGTERM to PID $PID...${NC}"
    kill -TERM $PID

    # Wait for graceful shutdown
    for i in {1..10}; do
        if ! ps -p $PID > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Daemon stopped gracefully${NC}"
            rm -f "$PID_FILE"
            return 0
        fi
        sleep 1
    done

    # Force kill if still running
    echo -e "${YELLOW}Forcing daemon shutdown...${NC}"
    kill -KILL $PID 2>/dev/null
    rm -f "$PID_FILE"
    echo -e "${GREEN}✓ Daemon stopped${NC}"
    return 0
}

# Function to check status
check_status() {
    print_header

    if [ ! -f "$PID_FILE" ]; then
        echo -e "${RED}Status: STOPPED${NC}"
        echo -e "  PID file does not exist"
        return 1
    fi

    PID=$(cat "$PID_FILE")

    if ps -p $PID > /dev/null 2>&1; then
        echo -e "${GREEN}Status: RUNNING${NC}"
        echo -e "  PID: $PID"

        # Get process info
        echo -e "\nProcess Info:"
        ps -p $PID -o pid,ppid,cmd,%cpu,%mem,etime

        # Show recent log entries
        echo -e "\n${BLUE}Recent Log Entries (last 10 lines):${NC}"
        if [ -d "$LOG_DIR" ]; then
            LATEST_LOG=$(ls -t "$LOG_DIR"/daemon_*.log 2>/dev/null | head -1)
            if [ -n "$LATEST_LOG" ]; then
                tail -10 "$LATEST_LOG"
            fi
        fi

        return 0
    else
        echo -e "${RED}Status: STOPPED${NC}"
        echo -e "  PID file exists but process is not running"
        echo -e "  Removing stale PID file"
        rm -f "$PID_FILE"
        return 1
    fi
}

# Function to view logs
view_logs() {
    print_header
    echo -e "${BLUE}Viewing daemon logs...${NC}"
    echo ""

    if [ -d "$LOG_DIR" ]; then
        LATEST_LOG=$(ls -t "$LOG_DIR"/daemon_*.log 2>/dev/null | head -1)
        if [ -n "$LATEST_LOG" ]; then
            echo -e "${GREEN}Log file: $LATEST_LOG${NC}"
            echo ""
            tail -f "$LATEST_LOG"
        else
            echo -e "${RED}No log files found${NC}"
        fi
    else
        echo -e "${RED}Log directory not found${NC}"
    fi
}

# Function to restart daemon
restart_daemon() {
    print_header
    echo -e "${YELLOW}Restarting BCI daemon...${NC}"
    stop_daemon
    sleep 2
    start_background
}

# Function to install systemd service
install_service() {
    print_header
    echo -e "${YELLOW}Installing systemd service...${NC}"

    if [ ! -f "$SERVICE_FILE" ]; then
        echo -e "${RED}✗ Service file not found: $SERVICE_FILE${NC}"
        return 1
    fi

    sudo cp "$SERVICE_FILE" /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable $DAEMON_NAME

    echo -e "${GREEN}✓ Service installed${NC}"
    echo -e "  You can now use: sudo systemctl start/stop/status $DAEMON_NAME"
}

# Main script
case "$1" in
    start)
        start_background
        ;;
    stop)
        stop_daemon
        ;;
    restart)
        restart_daemon
        ;;
    status)
        check_status
        ;;
    logs)
        view_logs
        ;;
    install)
        install_service
        ;;
    *)
        print_header
        echo ""
        echo "Usage: $0 {start|stop|restart|status|logs|install}"
        echo ""
        echo "Commands:"
        echo "  start     - Start daemon in background"
        echo "  stop      - Stop running daemon"
        echo "  restart   - Restart daemon"
        echo "  status    - Check daemon status"
        echo "  logs      - View daemon logs (live tail)"
        echo "  install   - Install as systemd service"
        echo ""
        exit 1
        ;;
esac

exit $?
