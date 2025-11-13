# Nexus AGI Deployment Guide

This guide provides instructions for deploying and running Nexus AGI in various environments.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Local Deployment](#local-deployment)
- [Docker Deployment](#docker-deployment)
- [Running the Application](#running-the-application)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- **Operating System**: Linux, macOS, or Windows (with WSL2 for Docker)
- **RAM**: Minimum 8GB, Recommended 16GB+
- **CPU**: Multi-core processor recommended
- **Disk Space**: At least 5GB free space

### Software Requirements
- **Python**: 3.8 or higher (Python 3.11 recommended)
- **pip**: Latest version
- **Docker** (optional): For containerized deployment
- **docker-compose** (optional): For easy orchestration

## Local Deployment

### Method 1: Using the Setup Script (Recommended)

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/DOUGLASDAVIS08161978/nexus_agi.git
   cd nexus_agi
   ```

2. **Run the setup script**:
   ```bash
   ./setup.sh
   ```
   
   This will:
   - Create a Python virtual environment
   - Install all required dependencies
   - Prepare the environment for running

3. **Activate the virtual environment**:
   ```bash
   source venv/bin/activate
   ```

### Method 2: Manual Installation

1. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Docker Deployment

Docker provides an isolated, reproducible environment for running Nexus AGI.

### Method 1: Using Docker Compose (Recommended)

1. **Build and start the container**:
   ```bash
   docker-compose up -d
   ```

2. **View logs**:
   ```bash
   docker-compose logs -f nexus-agi
   ```

3. **Stop the container**:
   ```bash
   docker-compose down
   ```

### Method 2: Using Docker CLI

1. **Build the Docker image**:
   ```bash
   docker build -t nexus-agi:latest .
   ```

2. **Run the container**:
   ```bash
   docker run -d \
     --name nexus-agi \
     -v nexus-data:/root/nexus_core \
     nexus-agi:latest
   ```

3. **View logs**:
   ```bash
   docker logs -f nexus-agi
   ```

4. **Stop and remove the container**:
   ```bash
   docker stop nexus-agi
   docker rm nexus-agi
   ```

## Running the Application

### Local Execution

#### Option 1: Using the run script
```bash
./run.sh
```

#### Option 2: Direct Python execution
```bash
python3 nexus_agi
```

#### Option 3: With the shebang (if executable)
```bash
./nexus_agi
```

### Docker Execution

The application runs automatically when the container starts. To run it interactively:

```bash
docker run -it --rm nexus-agi:latest
```

Or with docker-compose:
```bash
docker-compose run --rm nexus-agi
```

## Configuration

### Data Persistence

Nexus AGI creates a directory structure at `~/nexus_core` for storing models and data:
- `~/nexus_core/models/` - Trained models and weights
- `~/nexus_core/data/` - Generated data and results

### Environment Variables

You can customize behavior using environment variables:

```bash
# Example: Set custom base path
export NEXUS_BASE_PATH=/custom/path
python3 nexus_agi
```

### Docker Volume Management

Data persists in Docker volumes. To inspect:
```bash
docker volume ls
docker volume inspect nexus-data
```

To backup the volume:
```bash
docker run --rm -v nexus-data:/data -v $(pwd):/backup \
  alpine tar czf /backup/nexus-backup.tar.gz /data
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
**Problem**: ModuleNotFoundError for various packages

**Solution**: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

#### 2. Permission Denied
**Problem**: Cannot execute scripts

**Solution**: Make scripts executable:
```bash
chmod +x setup.sh run.sh nexus_agi
```

#### 3. Memory Issues
**Problem**: Out of memory errors during execution

**Solution**: 
- Reduce batch sizes in the code
- Increase system RAM or swap space
- For Docker, increase memory limits in docker-compose.yml

#### 4. PyTorch/CUDA Issues
**Problem**: PyTorch fails to initialize or use GPU

**Solution**:
- For CPU-only: The default requirements.txt works
- For GPU support: Install PyTorch with CUDA support:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu118
  ```

#### 5. Docker Build Failures
**Problem**: Docker build runs out of space or memory

**Solution**:
- Clean Docker cache: `docker system prune -a`
- Increase Docker resources in Docker Desktop settings
- Use multi-stage build (already implemented in Dockerfile)

### Getting Help

If you encounter issues not covered here:
1. Check the main README.md for project documentation
2. Review the code comments in the nexus_agi file
3. Open an issue on the GitHub repository

## Advanced Usage

### Running with Different Problem Scenarios

The main script includes a demonstration scenario. You can modify the `if __name__ == "__main__"` block to solve different problems.

### Integration with Other Systems

Nexus AGI can be imported as a module:

```python
from nexus_agi import MetaAlgorithm_NexusCore

core = MetaAlgorithm_NexusCore()
solution = core.solve_complex_problem(problem_definition, constraints)
```

### Continuous Deployment

For production deployments, consider:
1. Using a process manager like systemd or supervisor
2. Setting up monitoring with Prometheus/Grafana
3. Implementing health checks
4. Using Kubernetes for orchestration at scale

## Performance Optimization

### Local Optimization
- Use Python 3.11+ for better performance
- Enable PyTorch optimizations if using GPU
- Monitor resource usage with `htop` or similar tools

### Docker Optimization
- Use specific Python versions instead of `latest`
- Minimize image layers
- Use multi-stage builds (already implemented)
- Set appropriate resource limits in docker-compose.yml

## Security Considerations

When deploying in production:
1. Don't run containers as root (add USER directive to Dockerfile)
2. Use specific version tags for dependencies
3. Regularly update dependencies for security patches
4. Implement network isolation if needed
5. Use secrets management for sensitive data

## Next Steps

After successful deployment:
1. Review the output logs to ensure proper initialization
2. Monitor resource usage during execution
3. Customize problem scenarios for your use case
4. Integrate with your existing workflows

For more information about the Nexus AGI system capabilities, see the main README.md file.
