# NDR Platform Deployment Guide

## Quick Start

### Development Environment
```bash
# Clone and setup
git clone <repository-url>
cd ndr-platform

# Install dependencies
pip install -r requirements.txt

# Configure data source
cp config/config.example.yaml config/config.yaml
# Edit config.yaml to set your data directory

# Run application
streamlit run run.py
```

### Production Deployment

#### Docker Compose (Recommended)
```bash
# Build and deploy
docker-compose up -d

# View logs
docker-compose logs -f ndr-platform

# Stop
docker-compose down
```

#### Manual Docker
```bash
# Build image
docker build -t ndr-platform .

# Run container
docker run -d \
  --name ndr-platform \
  -p 8501:8501 \
  -v /opt/arkime/json:/app/data:ro \
  -v ./models:/app/models \
  -v ./logs:/app/logs \
  ndr-platform
```

## Configuration

### Environment-Specific Setup

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
