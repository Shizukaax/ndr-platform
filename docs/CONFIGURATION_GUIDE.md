# NDR Platform - Configuration Guide

## 🔧 Complete Configuration Reference

### 🏗️ Configuration Architecture

The NDR Platform uses a hierarchical configuration system:

```
Configuration Hierarchy:
1. Environment Variables (.env)     # Runtime overrides
2. config.yaml                     # Main configuration  
3. Default Values                   # Fallback defaults
```

## 📁 Configuration Files

### 1. **Main Configuration (`config/config.yaml`)**

```yaml
# NDR Platform Main Configuration
app:
  name: "Network Detection and Response Platform"
  version: "2.0.0"
  debug: false
  log_level: "INFO"
  timezone: "UTC"

# Data Source Configuration
data_source:
  # Primary data directory (production: /opt/arkime/json)
  directory: "data/"
  
  # Auto-loading settings
  auto_load: true
  max_files: 1000
  file_patterns:
    - "*.json"
    - "*.pcap_filtered.json"
  
  # Real-time monitoring
  watch_directory: true
  refresh_interval: 30  # seconds
  
  # Data validation
  validate_on_load: true
  required_fields:
    - "timestamp" 
    - "src_ip"
    - "dst_ip"

# Model Configuration
models:
  # Default algorithm
  default_algorithm: "IsolationForest"
  
  # Training parameters
  default_params:
    IsolationForest:
      contamination: 0.1
      n_estimators: 100
      random_state: 42
    LocalOutlierFactor:
      n_neighbors: 20
      contamination: 0.1
    OneClassSVM:
      nu: 0.1
      kernel: "rbf"
      gamma: "scale"
  
  # Model management
  auto_retrain: true
  retrain_threshold: 0.8
  performance_window: 1000
  max_models: 10
  
  # Feature engineering
  feature_selection:
    enabled: true
    max_features: 50
    selection_method: "variance"

# MITRE ATT&CK Integration
mitre:
  enabled: true
  data_file: "config/mitre_attack_data.json"
  auto_mapping: true
  confidence_threshold: 0.7
  
  # Risk scoring
  risk_scoring:
    enabled: true
    weights:
      technique_severity: 0.4
      anomaly_score: 0.3
      prevalence: 0.2
      context: 0.1

# Security Settings
security:
  # Alert thresholds
  alert_threshold: 0.8
  critical_threshold: 0.95
  
  # Threat intelligence
  threat_feeds:
    enabled: true
    sources:
      - "internal"
      - "mitre"
    update_interval: 3600  # seconds
  
  # IP geolocation
  geoip:
    enabled: true
    database_path: "lib/geoip/"
    
# Monitoring Configuration
monitoring:
  # Real-time metrics
  metrics_enabled: true
  metrics_interval: 60  # seconds
  
  # Health checks
  health_check:
    enabled: true
    interval: 30  # seconds
    endpoints:
      - "/health"
      - "/metrics"
  
  # Performance monitoring
  performance:
    track_response_time: true
    track_memory_usage: true
    alert_on_degradation: true

# Notification Settings  
notifications:
  enabled: true
  channels:
    - "streamlit"
    - "logs"
  
  # Alert levels
  levels:
    critical:
      color: "red"
      icon: "🚨"
      persistence: 300  # seconds
    warning:
      color: "orange" 
      icon: "⚠️"
      persistence: 60
    info:
      color: "blue"
      icon: "ℹ️"
      persistence: 30

# Storage Configuration
storage:
  # Model storage
  models_directory: "models/"
  model_format: "pickle"
  compress_models: true
  
  # Results storage
  results_directory: "results/"
  results_format: "json"
  max_results_age: 30  # days
  
  # Backup settings
  backup:
    enabled: true
    directory: "models/backups/"
    retention_days: 90
    compression: true

# Logging Configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  
  # File settings
  files:
    app:
      path: "logs/app.log"
      max_size: "10MB"
      backup_count: 5
    errors:
      path: "logs/errors.log"
      max_size: "5MB"
      backup_count: 3
    models:
      path: "logs/models.log" 
      max_size: "5MB"
      backup_count: 3
  
  # Console output
  console:
    enabled: true
    level: "WARNING"  # Reduced spam
    
# UI Configuration
ui:
  # Streamlit settings
  streamlit:
    port: 8501
    host: "0.0.0.0"
    headless: true
    
  # Page settings  
  max_rows_display: 1000
  chart_height: 400
  enable_caching: true
  
  # Theme
  theme:
    primary_color: "#1f77b4"
    background_color: "#ffffff"
    text_color: "#262730"

# Development Settings
development:
  # Debug features
  debug_mode: false
  profiling: false
  verbose_logging: false
  
  # Testing
  test_data_size: 1000
  mock_services: false
  
  # Hot reloading
  auto_reload: true
  watch_files: true
```

### 2. **Environment Configuration (`.env.example`)**

```bash
# NDR Platform Environment Configuration
# Copy to .env and customize for your environment

# === CORE APPLICATION SETTINGS ===
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Python Environment
PYTHONPATH=/app
PYTHONDONTWRITEBYTECODE=1
PYTHONUNBUFFERED=1

# === DATA CONFIGURATION ===
# Data source directory
# Development: ./data
# Production: /opt/arkime/json
DATA_DIRECTORY=/opt/arkime/json
MAX_FILES=1000
AUTO_REFRESH=true
WATCH_DIRECTORY=true

# Data validation
VALIDATE_ON_LOAD=true
REQUIRED_FIELDS=timestamp,src_ip,dst_ip

# === LOGGING CONFIGURATION ===
LOG_LEVEL=INFO
LOG_FILE=/app/logs/ndr-platform.log
LOG_MAX_SIZE=10MB
LOG_BACKUP_COUNT=5
CONSOLE_LOG_LEVEL=WARNING

# === MODEL CONFIGURATION ===
DEFAULT_MODEL=IsolationForest
MODEL_CONTAMINATION=0.1
BATCH_SIZE=1000
PROCESSING_INTERVAL=30
AUTO_RETRAIN=true
RETRAIN_THRESHOLD=0.8

# === SECURITY SETTINGS ===
ALERT_THRESHOLD=0.8
CRITICAL_THRESHOLD=0.95
THREAT_FEEDS_ENABLED=true
GEOIP_ENABLED=true
MITRE_MAPPING_ENABLED=true

# === MONITORING & METRICS ===
METRICS_ENABLED=true
HEALTH_CHECK_INTERVAL=30
PERFORMANCE_MONITORING=true

# === EXTERNAL SERVICES ===
# Database (if using external storage)
# DATABASE_URL=postgresql://user:password@localhost:5432/ndr_platform

# Redis (for caching and session storage)
# REDIS_URL=redis://redis:6379/0

# Elasticsearch (for log aggregation)
# ELASTIC_URL=http://elasticsearch:9200

# === DEPLOYMENT ENVIRONMENT ===
ENVIRONMENT=production
DEBUG=false
PROFILING=false

# === SECURITY & ACCESS ===
# API Keys (if using external threat feeds)
# VIRUSTOTAL_API_KEY=your_api_key_here
# SHODAN_API_KEY=your_api_key_here

# JWT Secret (if implementing authentication)
# JWT_SECRET=your-super-secret-jwt-key

# === PERFORMANCE TUNING ===
MAX_WORKERS=4
MEMORY_LIMIT=4GB
CACHE_SIZE=1000
CONNECTION_TIMEOUT=30

# === BACKUP & RETENTION ===
BACKUP_ENABLED=true
BACKUP_DIRECTORY=/app/backups
RETENTION_DAYS=90
COMPRESS_BACKUPS=true
```

### 3. **Docker Environment (`.env.docker`)**

```bash
# Docker-specific environment variables
COMPOSE_PROJECT_NAME=ndr-platform
COMPOSE_FILE=docker-compose.yml

# Container settings
CONTAINER_MEMORY=4g
CONTAINER_CPUS=2.0

# Volume mappings
DATA_VOLUME=/opt/arkime/json:/app/data:ro
LOGS_VOLUME=./logs:/app/logs
MODELS_VOLUME=./models:/app/models

# Network settings
EXTERNAL_PORT=8501
INTERNAL_PORT=8501
NETWORK_NAME=ndr-network

# Health check settings
HEALTH_CHECK_INTERVAL=30s
HEALTH_CHECK_TIMEOUT=10s
HEALTH_CHECK_RETRIES=3
```

## ⚙️ Configuration Loading

### Configuration Loader Implementation

```python
# core/config_loader.py
import yaml
import os
from pathlib import Path

def load_config(config_path=None, environment=None):
    """
    Load configuration with environment override support.
    
    Args:
        config_path: Path to config file (default: config/config.yaml)
        environment: Environment name (dev, staging, production)
    
    Returns:
        dict: Merged configuration
    """
    # Default config path
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    
    # Load base configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Environment-specific overrides
    if environment:
        env_config_path = config_path.parent / f"config.{environment}.yaml"
        if env_config_path.exists():
            with open(env_config_path, 'r') as f:
                env_config = yaml.safe_load(f)
            config = merge_configs(config, env_config)
    
    # Environment variable overrides
    config = apply_env_overrides(config)
    
    return config

def merge_configs(base, override):
    """Recursively merge configuration dictionaries."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            merge_configs(base[key], value)
        else:
            base[key] = value
    return base

def apply_env_overrides(config):
    """Apply environment variable overrides."""
    # Data directory override
    if 'DATA_DIRECTORY' in os.environ:
        config['data_source']['directory'] = os.environ['DATA_DIRECTORY']
    
    # Logging level override
    if 'LOG_LEVEL' in os.environ:
        config['app']['log_level'] = os.environ['LOG_LEVEL']
    
    # Model configuration overrides
    if 'DEFAULT_MODEL' in os.environ:
        config['models']['default_algorithm'] = os.environ['DEFAULT_MODEL']
    
    if 'MODEL_CONTAMINATION' in os.environ:
        contamination = float(os.environ['MODEL_CONTAMINATION'])
        for model in config['models']['default_params']:
            if 'contamination' in config['models']['default_params'][model]:
                config['models']['default_params'][model]['contamination'] = contamination
    
    # Security overrides
    if 'ALERT_THRESHOLD' in os.environ:
        config['security']['alert_threshold'] = float(os.environ['ALERT_THRESHOLD'])
    
    return config
```

## 🔧 Configuration Validation

### Validation Schema

```python
# Configuration validation schema
CONFIG_SCHEMA = {
    "app": {
        "required": ["name", "version"],
        "properties": {
            "name": {"type": "string"},
            "version": {"type": "string", "pattern": r"^\d+\.\d+\.\d+$"},
            "debug": {"type": "boolean"},
            "log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR"]}
        }
    },
    "data_source": {
        "required": ["directory"],
        "properties": {
            "directory": {"type": "string"},
            "auto_load": {"type": "boolean"},
            "max_files": {"type": "integer", "minimum": 1}
        }
    },
    "models": {
        "required": ["default_algorithm"],
        "properties": {
            "default_algorithm": {
                "type": "string",
                "enum": ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]
            }
        }
    }
}

def validate_config(config):
    """Validate configuration against schema."""
    errors = []
    
    # Validate required sections
    required_sections = ["app", "data_source", "models"]
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
    
    # Validate data types and values
    if "models" in config:
        contamination = config["models"]["default_params"]["IsolationForest"]["contamination"]
        if not 0 < contamination < 1:
            errors.append("Model contamination must be between 0 and 1")
    
    return errors
```

## 🌍 Environment-Specific Configurations

### Development (`config.dev.yaml`)
```yaml
app:
  debug: true
  log_level: "DEBUG"

data_source:
  directory: "./data"
  max_files: 100

monitoring:
  metrics_enabled: false

ui:
  streamlit:
    headless: false
```

### Production (`config.production.yaml`)  
```yaml
app:
  debug: false
  log_level: "INFO"

data_source:
  directory: "/opt/arkime/json"
  max_files: 10000

security:
  alert_threshold: 0.9
  critical_threshold: 0.95

monitoring:
  metrics_enabled: true
  health_check:
    enabled: true
```

## 📋 Configuration Best Practices

### 1. **Secrets Management**
```bash
# Never commit secrets to config files
# Use environment variables for sensitive data
DATABASE_PASSWORD=${DATABASE_PASSWORD}
API_KEY=${API_KEY}
JWT_SECRET=${JWT_SECRET}
```

### 2. **Environment Separation**
```yaml
# Use environment-specific config files
config.yaml          # Base configuration
config.dev.yaml      # Development overrides  
config.staging.yaml  # Staging overrides
config.production.yaml # Production overrides
```

### 3. **Validation**
```python
# Always validate configuration on startup
config = load_config()
errors = validate_config(config)
if errors:
    raise ConfigurationError(f"Invalid configuration: {errors}")
```

### 4. **Documentation**
```yaml
# Document all configuration options
data_source:
  directory: "data/"              # Path to data files
  auto_load: true                 # Auto-load on startup
  max_files: 1000                # Maximum files to process
  refresh_interval: 30            # Refresh interval in seconds
```

## 🔄 Runtime Configuration Updates

### Dynamic Configuration
```python
class ConfigManager:
    def __init__(self):
        self.config = load_config()
        self.watchers = []
    
    def update_config(self, section, key, value):
        """Update configuration at runtime."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self.notify_watchers(section, key, value)
    
    def get_config(self, section, key, default=None):
        """Get configuration value with fallback."""
        return self.config.get(section, {}).get(key, default)
    
    def watch_config(self, callback):
        """Register callback for configuration changes."""
        self.watchers.append(callback)
```

This comprehensive configuration guide ensures the NDR Platform can be properly configured for any environment while maintaining security and flexibility.
