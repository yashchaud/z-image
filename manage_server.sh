#!/bin/bash

# Z-Image Server Management Script
# Easy commands to start, stop, restart, and monitor the server

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="/tmp/z-image-server.pid"
LOG_FILE="/tmp/z-image-server.log"

start_server() {
    if [ -f "$PID_FILE" ] && ps -p $(cat "$PID_FILE") > /dev/null 2>&1; then
        echo "❌ Server is already running (PID: $(cat $PID_FILE))"
        exit 1
    fi

    echo "Starting Z-Image server..."
    cd "$PROJECT_DIR"

    # Activate venv if exists
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi

    nohup python src/server.py > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"

    sleep 3

    if ps -p $(cat "$PID_FILE") > /dev/null 2>&1; then
        echo "✅ Server started successfully!"
        echo "PID: $(cat $PID_FILE)"
        echo "Logs: $LOG_FILE"
    else
        echo "❌ Failed to start. Check logs: tail $LOG_FILE"
        exit 1
    fi
}

stop_server() {
    if [ ! -f "$PID_FILE" ]; then
        echo "❌ Server is not running (no PID file)"
        exit 1
    fi

    PID=$(cat "$PID_FILE")

    if ! ps -p $PID > /dev/null 2>&1; then
        echo "❌ Server is not running"
        rm -f "$PID_FILE"
        exit 1
    fi

    echo "Stopping server (PID: $PID)..."
    kill $PID

    # Wait for graceful shutdown
    for i in {1..10}; do
        if ! ps -p $PID > /dev/null 2>&1; then
            echo "✅ Server stopped"
            rm -f "$PID_FILE"
            return 0
        fi
        sleep 1
    done

    # Force kill if still running
    echo "Force killing..."
    kill -9 $PID
    rm -f "$PID_FILE"
    echo "✅ Server killed"
}

restart_server() {
    echo "Restarting server..."
    stop_server 2>/dev/null
    sleep 2
    start_server
}

status_server() {
    if [ ! -f "$PID_FILE" ]; then
        echo "❌ Server is not running (no PID file)"
        return 1
    fi

    PID=$(cat "$PID_FILE")

    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Server is running"
        echo "PID: $PID"
        echo "Uptime: $(ps -p $PID -o etime= | xargs)"
        echo "Memory: $(ps -p $PID -o rss= | awk '{print $1/1024 " MB"}')"

        # Check if responding
        if command -v curl &> /dev/null; then
            if curl -s http://localhost:8000/health > /dev/null; then
                echo "Status: ✅ Responding to requests"
            else
                echo "Status: ⚠️  Process running but not responding"
            fi
        fi
    else
        echo "❌ Server is not running"
        rm -f "$PID_FILE"
        return 1
    fi
}

logs_server() {
    if [ ! -f "$LOG_FILE" ]; then
        echo "❌ No log file found at $LOG_FILE"
        exit 1
    fi

    if [ "$1" == "-f" ] || [ "$1" == "--follow" ]; then
        echo "Following logs (Ctrl+C to exit)..."
        tail -f "$LOG_FILE"
    else
        echo "Last 50 lines of logs:"
        tail -n 50 "$LOG_FILE"
        echo ""
        echo "To follow logs in real-time: $0 logs -f"
    fi
}

# Main command handler
case "$1" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        restart_server
        ;;
    status)
        status_server
        ;;
    logs)
        logs_server "$2"
        ;;
    *)
        echo "Z-Image Server Management"
        echo ""
        echo "Usage: $0 {start|stop|restart|status|logs}"
        echo ""
        echo "Commands:"
        echo "  start    - Start the server"
        echo "  stop     - Stop the server"
        echo "  restart  - Restart the server"
        echo "  status   - Check if server is running"
        echo "  logs     - View recent logs"
        echo "  logs -f  - Follow logs in real-time"
        echo ""
        exit 1
        ;;
esac
