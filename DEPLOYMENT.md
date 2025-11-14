# Service Deployment Guide

This guide explains how to deploy the Nexus AGI and ARIA systems as continuous services.

## Overview

Both systems can run as services in continuous loops, processing problems and queries indefinitely. This guide covers three deployment methods:

1. **Direct execution** - Run services directly with Python/Node.js
2. **Docker deployment** - Deploy as containers using Docker Compose
3. **Systemd services** - Deploy as Linux system services

## Quick Start

### Direct Execution

**Python Nexus Service:**
```bash
# Default: runs every 5 minutes
python3 nexus_service.py

# Custom interval (e.g., every 60 seconds)
python3 nexus_service.py --interval 60

# Enable debug logging
python3 nexus_service.py --log-level DEBUG
```

**JavaScript ARIA Service:**
```bash
# Default: runs every 5 minutes
node aria_service.js

# Custom interval (e.g., every 120 seconds)
node aria_service.js --interval 120
```

### Docker Deployment

**Prerequisites:**
- Docker and Docker Compose installed

**Steps:**

1. **Build and start both services:**
```bash
docker-compose up -d
```

2. **View logs:**
```bash
# Nexus AGI logs
docker-compose logs -f nexus-service

# ARIA logs
docker-compose logs -f aria-service

# Both services
docker-compose logs -f
```

3. **Check service status:**
```bash
docker-compose ps
```

4. **Stop services:**
```bash
docker-compose down
```

5. **Restart services:**
```bash
docker-compose restart
```

**Configuration:**

Edit `docker-compose.yml` to customize:
- Interval between cycles (default: 300 seconds)
- Volume mounts for logs and data
- Resource limits
- Environment variables

### Systemd Deployment (Linux)

**Prerequisites:**
- Linux system with systemd
- Root/sudo access

**Installation Steps:**

1. **Install application files:**
```bash
# Create directory
sudo mkdir -p /opt/nexus_agi
sudo mkdir -p /var/log/nexus_agi
sudo mkdir -p /var/log/aria

# Copy files
sudo cp nexus_agi.py nexus_service.py /opt/nexus_agi/
sudo cp aria.js aria_service.js /opt/nexus_agi/

# Install Python dependencies
cd /opt/nexus_agi
sudo pip3 install -r requirements.txt
```

2. **Create service users:**
```bash
# For Nexus AGI
sudo useradd -r -s /bin/false nexus
sudo chown -R nexus:nexus /opt/nexus_agi
sudo chown -R nexus:nexus /var/log/nexus_agi

# For ARIA
sudo useradd -r -s /bin/false aria
sudo chown -R aria:aria /var/log/aria
```

3. **Install systemd service files:**
```bash
sudo cp systemd/nexus-agi.service /etc/systemd/system/
sudo cp systemd/aria.service /etc/systemd/system/
sudo systemctl daemon-reload
```

4. **Enable and start services:**
```bash
# Nexus AGI
sudo systemctl enable nexus-agi
sudo systemctl start nexus-agi

# ARIA
sudo systemctl enable aria
sudo systemctl start aria
```

5. **Check service status:**
```bash
sudo systemctl status nexus-agi
sudo systemctl status aria
```

6. **View logs:**
```bash
# Nexus AGI logs
sudo journalctl -u nexus-agi -f

# ARIA logs
sudo journalctl -u aria -f

# Or check log files directly
sudo tail -f /var/log/nexus_agi/service.log
sudo tail -f /var/log/aria/service.log
```

7. **Stop services:**
```bash
sudo systemctl stop nexus-agi
sudo systemctl stop aria
```

## Service Configuration

### Nexus Service Parameters

**Command-line options:**
```bash
python3 nexus_service.py --help

Options:
  --interval SECONDS    Seconds between cycles (default: 300)
  --log-level LEVEL     Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
```

**Environment variables:**
- `PYTHONUNBUFFERED=1` - Disable output buffering for real-time logs
- `HF_HUB_OFFLINE=1` - Use offline mode for HuggingFace models

### ARIA Service Parameters

**Command-line options:**
```bash
node aria_service.js --help

Options:
  --interval <seconds>  Seconds between cycles (default: 300)
  --help               Show help message
```

**Environment variables:**
- `NODE_ENV=production` - Run in production mode

## Service Behavior

### Processing Loop

Both services follow this pattern:

1. **Initialize** - Load models and components
2. **Process** - Execute problem solving or query processing
3. **Log results** - Record outcomes and metrics
4. **Wait** - Sleep for configured interval
5. **Repeat** - Go to step 2

### Graceful Shutdown

Services handle shutdown signals (SIGINT, SIGTERM) gracefully:
- Complete current processing cycle
- Log final statistics
- Clean up resources
- Exit with status code

To stop a running service:
- **Direct execution**: Press Ctrl+C
- **Docker**: `docker-compose stop`
- **Systemd**: `sudo systemctl stop <service-name>`

## Monitoring

### Log Files

**Direct execution:**
- `nexus_service.log` - Nexus AGI service logs
- `aria_service.log` - ARIA service logs

**Docker:**
- Logs available via `docker-compose logs`
- Also written to `./logs/` directory

**Systemd:**
- `/var/log/nexus_agi/service.log`
- `/var/log/aria/service.log`
- Also available via `journalctl`

### Metrics

Both services log:
- Cycle count
- Processing time
- Success/failure status
- System metrics (for Nexus: ethics scores, effectiveness; for ARIA: quantum confidence, consciousness levels)
- Uptime statistics

### Health Checks

**Docker:**
```bash
# Check if containers are running
docker-compose ps

# Check container health
docker inspect nexus-agi-service
docker inspect aria-service
```

**Systemd:**
```bash
# Check service status
sudo systemctl is-active nexus-agi
sudo systemctl is-active aria

# View recent logs
sudo journalctl -u nexus-agi --since "1 hour ago"
```

## Troubleshooting

### Service Won't Start

**Check dependencies:**
```bash
# Python
pip3 install -r requirements.txt

# Node.js
node --version  # Should be >= 14.0.0
```

**Check permissions:**
```bash
# Make sure scripts are executable
chmod +x nexus_service.py aria_service.js

# Check log directory permissions
ls -la /var/log/nexus_agi /var/log/aria
```

**View error logs:**
```bash
# Docker
docker-compose logs

# Systemd
sudo journalctl -u nexus-agi -n 50
sudo journalctl -u aria -n 50
```

### High Resource Usage

**Adjust cycle interval:**
```bash
# Increase time between cycles
python3 nexus_service.py --interval 600  # 10 minutes
node aria_service.js --interval 600
```

**Docker resource limits:**

Edit `docker-compose.yml`:
```yaml
services:
  nexus-service:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
```

### Service Crashes

**Check logs** for error messages and stack traces.

**Common issues:**
- Out of memory: Increase system memory or reduce cycle frequency
- Missing dependencies: Install required packages
- Permission errors: Check file/directory ownership

**Automatic restart:**
- Docker: `restart: unless-stopped` (already configured)
- Systemd: `Restart=always` (already configured)

## Production Recommendations

### Resource Requirements

**Minimum:**
- Nexus AGI: 4GB RAM, 2 CPU cores
- ARIA: 2GB RAM, 1 CPU core

**Recommended:**
- Nexus AGI: 8GB+ RAM, 4+ CPU cores
- ARIA: 4GB RAM, 2 CPU cores

### Security

**Docker:**
- Run containers as non-root users
- Use network isolation
- Limit resource usage
- Regular security updates

**Systemd:**
- Run services as dedicated users (not root)
- Use security settings in service files
- Enable firewall rules as needed
- Regular system updates

### Backup

**Important data:**
- Log files: `/var/log/nexus_agi/`, `/var/log/aria/`
- Nexus data: `~/nexus_core/` or `/opt/nexus_agi/nexus_core/`

**Backup script example:**
```bash
#!/bin/bash
BACKUP_DIR="/backup/nexus_agi_$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"
cp -r /var/log/nexus_agi "$BACKUP_DIR/"
cp -r ~/nexus_core "$BACKUP_DIR/"
tar czf "$BACKUP_DIR.tar.gz" "$BACKUP_DIR"
```

### Monitoring & Alerts

Consider integrating with:
- Prometheus for metrics collection
- Grafana for visualization
- Alertmanager for notifications

## Advanced Configuration

### Custom Problem Sources

Edit `nexus_service.py` to integrate with:
- Message queues (RabbitMQ, Kafka)
- REST APIs
- Databases
- File watchers

Example integration point in `process_problem()` method:
```python
def process_problem(self):
    # Replace this with your problem source
    problem = fetch_from_queue()  # Your custom function
    
    solution = self.core.solve_complex_problem(problem, constraints)
    # ... rest of processing
```

### Custom Query Sources

Edit `aria_service.js` to integrate with:
- WebSocket servers
- REST APIs
- Message brokers

Example integration point in `generateQuery()` method:
```javascript
generateQuery() {
    // Replace this with your query source
    return await fetchFromAPI();  // Your custom function
}
```

## Support

For issues or questions:
1. Check logs for error messages
2. Review this deployment guide
3. Consult the main README.md
4. Open an issue on GitHub

## Summary

The Nexus AGI and ARIA systems can now run as continuous services using any of the three deployment methods. Choose the method that best fits your infrastructure:

- **Direct execution**: Simple, good for development
- **Docker**: Portable, isolated, easy to manage
- **Systemd**: Native Linux integration, production-ready

All methods support graceful shutdown, automatic restart, and comprehensive logging.
