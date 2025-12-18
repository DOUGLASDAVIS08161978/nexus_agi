# Nexus AGI - Startup Methods Guide

This guide explains the three convenient ways to start the Nexus AGI system.

## Prerequisites

Before running the system, ensure you have installed the required dependencies:

```bash
pip install -r requirements.txt
```

## Method 1: Main Entry Point (Recommended)

The `main.py` script provides a unified, user-friendly CLI interface.

### Basic Usage

```bash
# Run Nexus Core demonstration
python main.py

# Run OMEGA ASI system
python main.py --system omega

# Show all available options
python main.py --help
```

### Service Mode

Run the system as a continuous service that processes problems at regular intervals:

```bash
# Run as service (default: every 5 minutes)
python main.py --service

# Custom interval (e.g., every 60 seconds)
python main.py --service --interval 60

# Run OMEGA ASI as service
python main.py --system omega --service
```

### Advantages
- ✅ Simple and intuitive
- ✅ Built-in help system
- ✅ Flexible CLI options
- ✅ Works on all platforms

## Method 2: Python Module Interface

Run the system as a Python module using the `-m` flag.

### Basic Usage

```bash
# Run Nexus Core demonstration
python -m nexus_agi

# Run OMEGA ASI system
python -m nexus_agi --system omega

# Show all available options
python -m nexus_agi --help
```

### Service Mode

```bash
# Run as service
python -m nexus_agi --service

# Custom interval
python -m nexus_agi --service --interval 60
```

### Advantages
- ✅ Standard Python module convention
- ✅ Works with virtual environments
- ✅ Easy to integrate with other Python tools
- ✅ Same functionality as main.py

## Method 3: Docker Compose

Deploy the entire system as containerized services.

### Basic Usage

```bash
# Start all services in background
docker compose up -d

# View logs
docker compose logs -f

# Stop all services
docker compose down
```

### Service Management

```bash
# Start specific service
docker compose up -d nexus-service

# Restart services
docker compose restart

# View service status
docker compose ps
```

### Advantages
- ✅ Isolated environment
- ✅ Production-ready
- ✅ Easy deployment
- ✅ Includes monitoring dashboard
- ✅ Automatic restart on failure

### Services Included
- **nexus-service**: Nexus Core AGI system
- **aria-service**: ARIA quantum-enhanced AI
- **dashboard**: Web-based monitoring dashboard (port 8080)

## Direct Script Execution (Legacy)

For backwards compatibility, you can still run scripts directly:

```bash
# Run Nexus Core
python3 nexus_agi.py

# Run as service
python3 nexus_service.py --interval 300

# Run OMEGA ASI
python3 omega_asi.py

# Run ARIA (requires Node.js)
node aria.js
```

## Quick Comparison

| Method | Best For | Pros |
|--------|----------|------|
| `python main.py` | Local development, testing | Simple, flexible, user-friendly |
| `python -m nexus_agi` | Integration, automation | Standard Python convention |
| `docker compose up` | Production, deployment | Isolated, scalable, production-ready |
| Direct scripts | Legacy compatibility | Direct access to individual components |

## Troubleshooting

### ImportError: No module named 'nexus_agi'
- **Solution**: Ensure you're in the project root directory
- **Solution**: Install dependencies: `pip install -r requirements.txt`

### Permission Denied
- **Solution**: Make scripts executable: `chmod +x main.py`
- **Or**: Run with python explicitly: `python main.py`

### Docker command not found
- **Solution**: Install Docker and Docker Compose
- **Alternative**: Use Method 1 or 2 instead

### Module dependencies missing
- **Solution**: Install all requirements: `pip install -r requirements.txt`
- **Note**: Some optional features (PySwip, Pennylane) require additional setup

## Next Steps

After starting the system, you can:

1. **Customize the problem**: Edit the problem definition in the code
2. **Adjust constraints**: Modify solution constraints
3. **Monitor performance**: Check the logs directory for output
4. **Scale up**: Deploy with Docker Compose for production use

For more detailed information, see:
- [README.md](README.md) - Complete system documentation
- [DEPLOYMENT.md](DEPLOYMENT.md) - Production deployment guide
- [OMEGA_ASI_README.md](OMEGA_ASI_README.md) - OMEGA ASI documentation

## Support

For issues or questions:
- Check the [README.md](README.md) troubleshooting section
- Review the code examples in the documentation
- Open an issue on GitHub
