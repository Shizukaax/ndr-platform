# Enhanced NDR Platform Deployment Guide

## 🚀 Enhanced Quick Start *(Version 2.1.0)*

### ✨ **Enhanced Features Support**
The deployment now fully supports:
- **📊 Enterprise Anomaly Tracking** - Persistent storage with baseline learning
- **🤖 6 Configurable ML Models** - Isolation Forest, LOF, One-Class SVM, KNN, HDBSCAN, Ensemble
- **📈 Anomaly History Dashboard** - Time-filtered historical analysis
- **🎯 Baseline Learning System** - Automatic pattern detection
- **⚡ Real-time Integration** - Seamless ML detection with persistent storage

### Smart Deployment (Recommended)
**Windows:**
```cmd
# Smart deployment - automatically detects if rebuild is needed
deploy.bat

# Force full rebuild (recommended after major updates)
deploy.bat rebuild

# Check status and verify anomaly tracking
deploy.bat status
```

**Linux/macOS:**
```bash
# Smart deployment - automatically detects if rebuild is needed
./deploy.sh

# Force full rebuild (recommended after major updates)
./deploy.sh rebuild

# Check status and verify anomaly tracking
./deploy.sh status
```

### Enhanced Development Environment
```bash
# Clone and setup enhanced platform
git clone <repository-url>
cd ndr-platform

# Install enhanced dependencies (includes ML libraries)
pip install -r requirements.txt

# Configure data source and ML models
cp config/config.example.yaml config/config.yaml
# Edit config.yaml to set:
#   - Your data directory
#   - ML model preferences (ensemble recommended)
#   - Auto-refresh intervals
#   - Anomaly tracking settings

# Test enhanced anomaly tracking system
python test_tracking_system.py

# Run enhanced application
streamlit run run.py
```

### Manual Docker Deployment

#### Docker Compose (Production)
```bash
# Build and deploy
docker-compose -f guides/deployment/docker-compose.yml up -d

# View logs
docker-compose -f guides/deployment/docker-compose.yml logs -f ndr-platform

# Stop
docker-compose -f guides/deployment/docker-compose.yml down
```

#### Manual Docker
```bash
# Build image
docker build -f guides/deployment/Dockerfile -t ndr-platform .

# Run container
docker run -d \
  --name ndr-platform \
  -p 8501:8501 \
  -v /opt/arkime/json:/app/data:ro \
  -v ./models:/app/models \
  -v ./logs:/app/logs \
  ndr-platform
```

## 🔄 When to Rebuild vs Restart

### Smart Deployment Logic

The deployment scripts automatically determine when a full rebuild is needed vs when a simple restart is sufficient:

**Full Rebuild Required When:**
- `requirements.txt` has changed (new Python dependencies)
- `Dockerfile` has been modified
- Container or image doesn't exist
- First-time deployment

**Restart Sufficient When:**
- Only Python code files have changed (`.py` files in `app/`, `core/`)
- Configuration files updated (`config.yaml`)
- Data files added/modified

### Manual Control
```bash
# Let script decide (recommended)
./deploy.sh

# Force full rebuild
./deploy.sh rebuild

# Just restart (if you're sure no dependencies changed)
./deploy.sh restart
```

## 📦 Docker Configuration Status

### Current Dockerfile Features
✅ **Multi-stage optimized build**
✅ **Security hardened** (non-root user)
✅ **Health checks** built-in
✅ **Proper layer caching** for faster rebuilds
✅ **All dependencies** included in requirements.txt
✅ **Volume mounts** for persistent data

### Container Configuration
- **Base Image:** `python:3.11-slim` (lightweight, secure)
- **Working Directory:** `/app`
- **Exposed Port:** `8501` (Streamlit default)
- **User:** `appuser` (non-root for security)
- **Health Check:** Built-in Streamlit health endpoint

### Volume Mounts
```yaml
volumes:
  - /opt/arkime/json:/app/data:ro     # Arkime JSON files (read-only)
  - ./models:/app/models              # Trained models
  - ./logs:/app/logs                  # Application logs
  - ./results:/app/results            # Analysis results
  - ./feedback:/app/feedback          # User feedback data
  - ./config:/app/config              # Configuration files
  - ./cache:/app/cache                # Temporary cache
  - ./reports:/app/reports            # Generated reports
```

**Development (Windows)**:
```yaml
# config/config.yaml
data_source:
  directory: "C:\\Users\\justinchua\\Desktop\\newnewapp\\data"
  file_pattern: "*.json"
  max_files: 100
  auto_refresh: true
```

**Production (Linux)**:
```yaml
# config/config.yaml
data_source:
  directory: "/opt/arkime/json"
  file_pattern: "*.json"
  max_files: 1000
  auto_refresh: true
```

### Environment Variables
```bash
# .env file
STREAMLIT_SERVER_PORT=8501
DATA_DIRECTORY=/opt/arkime/json
LOG_LEVEL=INFO
ENVIRONMENT=production
```

## Monitoring

### Health Checks
- Application: `http://localhost:8501/_stcore/health`
- Container: Built-in Docker healthcheck
- Logs: `docker-compose logs -f ndr-platform`

### Performance Tuning
- Batch size: Adjust `BATCH_SIZE` environment variable
- Memory: Configure Docker memory limits
- Processing: Tune `max_files` in config

### Troubleshooting

**Common Issues:**
1. **No data loading**: Check data directory path and permissions
2. **High memory usage**: Reduce `max_files` or increase container memory
3. **Slow processing**: Adjust batch size and processing interval

**Debug Commands:**
```bash
# Check container logs
docker logs ndr-platform

# Access container shell
docker exec -it ndr-platform /bin/bash

# Test configuration
python -c "from core.config_loader import load_config; print(load_config())"
```

## Scaling

### Horizontal Scaling
```bash
# Scale to multiple instances
docker-compose up -d --scale ndr-platform=3

# Load balancer (nginx)
docker-compose --profile production up -d
```

### Resource Allocation
```yaml
# docker-compose.yml
services:
  ndr-platform:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

## Security

### Production Considerations
- Use read-only data mounts
- Run containers as non-root user
- Enable SSL/TLS with reverse proxy
- Restrict network access
- Regular security updates

### Network Security
```yaml
# docker-compose.yml
networks:
  ndr-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```
