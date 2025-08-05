# NDR Platform - Deployment Guide

## 🚀 Complete Deployment Documentation

### 🏗️ Deployment Architecture

```
Production Deployment Architecture:
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer / Reverse Proxy            │
│                         (nginx)                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                   Docker Compose Stack                      │
├─────────────────────┬───────────────────────────────────────┤
│   NDR Application   │          Data Sources                 │
│   (Streamlit App)   │       (/opt/arkime/json)              │
│                     │                                       │
│   Core Services:    │          Volume Mounts:               │
│   - Data Manager    │       - Data: Read-only               │
│   - Model Manager   │       - Logs: Read-write              │
│   - Analytics       │       - Models: Read-write            │
│   - MITRE Mapping   │       - Results: Read-write           │
└─────────────────────┴───────────────────────────────────────┘
```

## 📦 Container Deployment

### 1. **Docker Configuration**

#### **Dockerfile**
```dockerfile
# Multi-stage build for production optimization
FROM python:3.11-slim as base

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r ndr && useradd -r -g ndr -d /app -s /bin/bash ndr

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/models /app/results /app/data && \
    chown -R ndr:ndr /app

# Switch to non-root user
USER ndr

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Expose port
EXPOSE 8501

# Run application
CMD ["streamlit", "run", "run.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"]
```

#### **Docker Compose - Production**
```yaml
# docker-compose.yml - Production Configuration
version: '3.8'

services:
  ndr-app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: ndr-platform
    restart: unless-stopped
    
    # Environment configuration
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
      - DATA_DIRECTORY=/app/data
      - STREAMLIT_SERVER_HEADLESS=true
      - PYTHONDONTWRITEBYTECODE=1
      - PYTHONUNBUFFERED=1
    
    # Volume mounts
    volumes:
      # Data source (read-only for security)
      - /opt/arkime/json:/app/data:ro
      # Persistent storage (read-write)
      - ./logs:/app/logs
      - ./models:/app/models
      - ./results:/app/results
      - ./feedback:/app/feedback
      - ./cache:/app/cache
    
    # Port mapping
    ports:
      - "8501:8501"
    
    # Resource limits
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
    
    # Health check
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    
    # Logging configuration
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"
    
    # Network
    networks:
      - ndr-network

  # Nginx reverse proxy
  nginx:
    image: nginx:alpine
    container_name: ndr-nginx
    restart: unless-stopped
    
    ports:
      - "80:80"
      - "443:443"
    
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    
    depends_on:
      - ndr-app
    
    networks:
      - ndr-network

  # Optional: Log aggregation
  fluentd:
    image: fluent/fluentd:v1.14-debian
    container_name: ndr-fluentd
    restart: unless-stopped
    
    volumes:
      - ./fluentd/fluent.conf:/fluentd/etc/fluent.conf
      - ./logs:/app/logs:ro
    
    networks:
      - ndr-network

networks:
  ndr-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/24

# Named volumes for persistent data
volumes:
  ndr-logs:
    driver: local
  ndr-models:
    driver: local
  ndr-results:
    driver: local
```

#### **Nginx Configuration**
```nginx
# nginx.conf - Production Reverse Proxy
events {
    worker_connections 1024;
}

http {
    upstream ndr_app {
        server ndr-app:8501;
    }
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # Gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    server {
        listen 80;
        server_name your-domain.com;
        
        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }
    
    server {
        listen 443 ssl http2;
        server_name your-domain.com;
        
        # SSL configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_session_timeout 1d;
        ssl_session_cache shared:SSL:50m;
        ssl_session_tickets off;
        
        # Modern configuration
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
        ssl_prefer_server_ciphers off;
        
        # HSTS
        add_header Strict-Transport-Security "max-age=63072000" always;
        
        # Proxy configuration
        location / {
            proxy_pass http://ndr_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support for Streamlit
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            
            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
            
            # Rate limiting
            limit_req zone=api burst=20 nodelay;
        }
        
        # Health check endpoint
        location /health {
            proxy_pass http://ndr_app/_stcore/health;
            access_log off;
        }
        
        # Static files caching
        location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
```

### 2. **Kubernetes Deployment**

#### **Kubernetes Manifests**
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ndr-platform

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ndr-config
  namespace: ndr-platform
data:
  config.yaml: |
    app:
      name: "NDR Platform"
      version: "2.0.0"
      debug: false
      log_level: "INFO"
    data_source:
      directory: "/app/data"
      auto_load: true
      max_files: 10000
    models:
      default_algorithm: "IsolationForest"
      auto_retrain: true

---
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: ndr-secrets
  namespace: ndr-platform
type: Opaque
data:
  # Base64 encoded secrets
  database-password: <base64-encoded-password>
  api-key: <base64-encoded-api-key>

---
# k8s/pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ndr-data-pvc
  namespace: ndr-platform
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: fast-ssd

---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ndr-app
  namespace: ndr-platform
  labels:
    app: ndr-platform
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: ndr-platform
  template:
    metadata:
      labels:
        app: ndr-platform
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
        fsGroup: 1001
      containers:
      - name: ndr-app
        image: your-registry/ndr-platform:v2.0.0
        ports:
        - containerPort: 8501
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
        - name: DATABASE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: ndr-secrets
              key: database-password
        
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        - name: data-volume
          mountPath: /app/data
        - name: logs-volume
          mountPath: /app/logs
        
        livenessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
      
      volumes:
      - name: config-volume
        configMap:
          name: ndr-config
      - name: data-volume
        hostPath:
          path: /opt/arkime/json
          type: Directory
      - name: logs-volume
        persistentVolumeClaim:
          claimName: ndr-data-pvc

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ndr-service
  namespace: ndr-platform
spec:
  selector:
    app: ndr-platform
  ports:
  - port: 80
    targetPort: 8501
    protocol: TCP
  type: ClusterIP

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ndr-ingress
  namespace: ndr-platform
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - ndr.your-domain.com
    secretName: ndr-tls-secret
  rules:
  - host: ndr.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ndr-service
            port:
              number: 80
```

## 🔧 Production Setup

### 1. **System Requirements**

#### **Minimum Requirements**
- **CPU**: 4 cores, 2.4 GHz
- **RAM**: 8 GB
- **Storage**: 100 GB SSD
- **Network**: 1 Gbps
- **OS**: Ubuntu 20.04 LTS / CentOS 8 / RHEL 8

#### **Recommended Requirements**
- **CPU**: 8 cores, 3.0 GHz
- **RAM**: 16 GB
- **Storage**: 500 GB NVMe SSD
- **Network**: 10 Gbps
- **OS**: Ubuntu 22.04 LTS

### 2. **Pre-deployment Checklist**

```bash
#!/bin/bash
# pre-deployment-check.sh

echo "=== NDR Platform Pre-deployment Check ==="

# Check system requirements
echo "1. Checking system requirements..."
cores=$(nproc)
memory=$(free -g | awk '/^Mem:/{print $2}')
disk=$(df -h / | awk 'NR==2{print $4}')

echo "   CPU Cores: $cores"
echo "   Memory: ${memory}GB"
echo "   Disk Space: $disk"

# Check Docker installation
echo "2. Checking Docker..."
if command -v docker &> /dev/null; then
    docker_version=$(docker --version)
    echo "   ✅ Docker installed: $docker_version"
else
    echo "   ❌ Docker not installed"
    exit 1
fi

# Check Docker Compose
echo "3. Checking Docker Compose..."
if command -v docker-compose &> /dev/null; then
    compose_version=$(docker-compose --version)
    echo "   ✅ Docker Compose installed: $compose_version"
else
    echo "   ❌ Docker Compose not installed"
    exit 1
fi

# Check data directory
echo "4. Checking data directory..."
if [ -d "/opt/arkime/json" ]; then
    file_count=$(find /opt/arkime/json -name "*.json" | wc -l)
    echo "   ✅ Data directory exists with $file_count JSON files"
else
    echo "   ❌ Data directory /opt/arkime/json not found"
    exit 1
fi

# Check ports
echo "5. Checking port availability..."
if ! netstat -tuln | grep :8501 &> /dev/null; then
    echo "   ✅ Port 8501 available"
else
    echo "   ❌ Port 8501 already in use"
    exit 1
fi

echo "=== Pre-deployment check completed ✅ ==="
```

### 3. **Deployment Scripts**

#### **Quick Deploy Script**
```bash
#!/bin/bash
# deploy.sh - Quick deployment script

set -e

echo "=== NDR Platform Deployment ==="

# Configuration
REPO_URL="https://github.com/your-org/ndr-platform.git"
DEPLOY_DIR="/opt/ndr-platform"
BACKUP_DIR="/opt/ndr-platform/backups"

# Create deployment directory
echo "1. Setting up deployment directory..."
sudo mkdir -p $DEPLOY_DIR
sudo chown $USER:$USER $DEPLOY_DIR
cd $DEPLOY_DIR

# Clone or update repository
if [ -d ".git" ]; then
    echo "2. Updating existing repository..."
    git pull origin main
else
    echo "2. Cloning repository..."
    git clone $REPO_URL .
fi

# Create environment file
echo "3. Creating environment configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "   ⚠️  Please edit .env file with your configuration"
    read -p "   Press enter to continue after editing .env..."
fi

# Create necessary directories
echo "4. Creating directories..."
mkdir -p logs models results feedback cache backups

# Set permissions
echo "5. Setting permissions..."
chmod 755 scripts/*.sh
chmod 600 .env

# Build and start services
echo "6. Building and starting services..."
docker-compose build
docker-compose up -d

# Wait for services to be ready
echo "7. Waiting for services to start..."
sleep 30

# Health check
echo "8. Performing health check..."
if curl -f http://localhost:8501/_stcore/health &> /dev/null; then
    echo "   ✅ Application is healthy"
else
    echo "   ❌ Application health check failed"
    docker-compose logs ndr-app
    exit 1
fi

# Setup log rotation
echo "9. Setting up log rotation..."
sudo cp scripts/logrotate.conf /etc/logrotate.d/ndr-platform

# Setup monitoring
echo "10. Setting up monitoring..."
./scripts/setup-monitoring.sh

echo "=== Deployment completed successfully! ==="
echo "Access the application at: http://localhost:8501"
echo "Logs: docker-compose logs -f"
echo "Stop: docker-compose down"
```

#### **Production Deploy Script**
```bash
#!/bin/bash
# production-deploy.sh - Zero-downtime production deployment

set -e

echo "=== NDR Platform Production Deployment ==="

# Configuration
NEW_VERSION=$1
CURRENT_VERSION=$(cat VERSION 2>/dev/null || echo "unknown")
BACKUP_DIR="/opt/ndr-platform/backups/$(date +%Y%m%d_%H%M%S)"

if [ -z "$NEW_VERSION" ]; then
    echo "Usage: $0 <version>"
    exit 1
fi

# Pre-deployment backup
echo "1. Creating backup..."
mkdir -p $BACKUP_DIR
cp -r models results feedback config $BACKUP_DIR/
docker-compose exec ndr-app pg_dump > $BACKUP_DIR/database.sql

# Health check current version
echo "2. Checking current application health..."
if ! curl -f http://localhost:8501/_stcore/health &> /dev/null; then
    echo "   ❌ Current application is unhealthy, aborting deployment"
    exit 1
fi

# Pull new image
echo "3. Pulling new image..."
docker pull your-registry/ndr-platform:$NEW_VERSION

# Update docker-compose with new version
echo "4. Updating configuration..."
sed -i "s/image: your-registry\/ndr-platform:.*/image: your-registry\/ndr-platform:$NEW_VERSION/" docker-compose.yml

# Rolling update
echo "5. Performing rolling update..."
docker-compose up -d --no-deps ndr-app

# Wait for new version to be ready
echo "6. Waiting for new version..."
timeout=300
count=0
while [ $count -lt $timeout ]; do
    if curl -f http://localhost:8501/_stcore/health &> /dev/null; then
        echo "   ✅ New version is healthy"
        break
    fi
    count=$((count + 10))
    sleep 10
done

if [ $count -ge $timeout ]; then
    echo "   ❌ New version failed to become healthy, rolling back..."
    docker-compose down
    sed -i "s/image: your-registry\/ndr-platform:.*/image: your-registry\/ndr-platform:$CURRENT_VERSION/" docker-compose.yml
    docker-compose up -d
    exit 1
fi

# Update version file
echo $NEW_VERSION > VERSION

# Cleanup old images
echo "7. Cleaning up..."
docker image prune -f

echo "=== Production deployment completed successfully! ==="
echo "Previous version: $CURRENT_VERSION"
echo "New version: $NEW_VERSION"
echo "Backup location: $BACKUP_DIR"
```

## 📊 Monitoring & Maintenance

### 1. **Health Monitoring**

```bash
#!/bin/bash
# health-check.sh - Comprehensive health monitoring

echo "=== NDR Platform Health Check ==="

# Service status
echo "1. Service Status:"
docker-compose ps

# Application health
echo "2. Application Health:"
health_status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8501/_stcore/health)
if [ "$health_status" = "200" ]; then
    echo "   ✅ Application: Healthy"
else
    echo "   ❌ Application: Unhealthy (HTTP $health_status)"
fi

# Resource usage
echo "3. Resource Usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"

# Disk usage
echo "4. Disk Usage:"
df -h | grep -E "(logs|models|results)"

# Log errors
echo "5. Recent Errors:"
docker-compose logs --tail=50 | grep -i error | tail -10

# Model status
echo "6. Model Status:"
ls -la models/ | tail -5
```

### 2. **Backup Strategy**

```bash
#!/bin/bash
# backup.sh - Automated backup script

BACKUP_DIR="/opt/ndr-platform/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="$BACKUP_DIR/$DATE"

echo "Creating backup: $BACKUP_PATH"

# Create backup directory
mkdir -p $BACKUP_PATH

# Backup application data
tar -czf $BACKUP_PATH/models.tar.gz models/
tar -czf $BACKUP_PATH/results.tar.gz results/
tar -czf $BACKUP_PATH/feedback.tar.gz feedback/
tar -czf $BACKUP_PATH/config.tar.gz config/
tar -czf $BACKUP_PATH/logs.tar.gz logs/

# Backup database (if using external database)
# docker-compose exec postgres pg_dump -U ndr ndr_platform > $BACKUP_PATH/database.sql

# Create manifest
cat > $BACKUP_PATH/manifest.txt << EOF
Backup Date: $(date)
Application Version: $(cat VERSION)
Docker Images: $(docker images --format "{{.Repository}}:{{.Tag}}" | grep ndr)
Configuration Hash: $(md5sum config/config.yaml | cut -d' ' -f1)
EOF

# Cleanup old backups (keep last 30 days)
find $BACKUP_DIR -type d -mtime +30 -exec rm -rf {} +

echo "Backup completed: $BACKUP_PATH"
```

This comprehensive deployment guide ensures reliable, secure, and scalable production deployment of the NDR Platform.
