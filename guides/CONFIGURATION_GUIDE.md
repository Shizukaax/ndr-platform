# NDR Platform v2.1.0 - Configuration Guide

## 🔧 Production Configuration Reference *(August 2025)*

### 🏗️ Configuration Architecture

The NDR Platform v2.1.0 uses a robust, hierarchical configuration system designed for enterprise deployment:

```
Configuration Hierarchy:
1. config.yaml                     # ✅ Main system configuration
2. Environment Variables (.env)     # Runtime overrides  
3. Default Values                   # Intelligent fallbacks
4. Runtime Settings                 # Dynamic adjustments
```

---

## 📋 **Critical Configuration Updates**

### **✅ Fixed Configuration Issues**
Following recent platform fixes, all configuration management has been enhanced:

- **Directory Creation:** `run.py` uses config.yaml for all paths
- **Results Storage:** Proper integration with configured directories
- **Feedback Management:** Correct path resolution for user feedback
- **Error Handling:** Robust fallback mechanisms

---

## 📁 **Main Configuration File**

### **`config/config.yaml` - Enhanced Structure**

```yaml
# NDR Platform v2.1.0 Configuration
app:
  name: "Network Detection and Response Platform"
  version: "2.1.0"
  debug: false
  log_level: "INFO"
  timezone: "UTC"

# ✅ Fixed: System paths (used by run.py)
system:
  data_dir: "data"
  results_dir: "data/results"
  models_dir: "models"
  logs_dir: "logs"
  cache_dir: "cache"

# ✅ Fixed: Feedback configuration
feedback:
  storage_dir: "data/feedback"
  enable_auto_save: true
  validation_required: false

# ✅ Enhanced: Anomaly detection configuration
anomaly_detection:
  save_results: true
  models:
    ensemble:
      enabled: true
      weights:
        isolation_forest: 0.3
        local_outlier_factor: 0.25
        one_class_svm: 0.2
        knn: 0.15
        hdbscan: 0.1
    
    isolation_forest:
      n_estimators: 100
      contamination: 0.1
      random_state: 42
    
    local_outlier_factor:
      n_neighbors: 20
      contamination: 0.1
    
    one_class_svm:
      kernel: "rbf"
      gamma: "scale"
      nu: 0.1
    
    knn:
      n_neighbors: 5
      contamination: 0.1
    
    hdbscan:
      min_cluster_size: 5
      min_samples: 3

# ✅ Enhanced: Real-time monitoring
monitoring:
  real_time:
    enable_auto_refresh: true
    auto_refresh_interval: 300  # 5 minutes
    max_data_points: 1000
    
    anomaly_detection:
      enabled: true
      model_type: "ensemble"
      confidence_threshold: 0.7
      
    alerts:
      enabled: true
      threshold: 0.8
      notification_methods: ["email", "dashboard"]

# Data processing configuration
data:
  processing:
    chunk_size: 1000
    max_file_size_mb: 100
    supported_formats: ["json", "csv", "parquet"]
    
  validation:
    strict_mode: false
    required_fields: ["timestamp", "source_ip", "dest_ip"]
    
  storage:
    compression: true
    retention_days: 90
    backup_enabled: true

# ✅ Enhanced: Logging configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  
  files:
    app: "logs/app.log"
    anomaly: "logs/anomaly_detection.log"
    model: "logs/model_manager.log"
    data: "logs/data_processing.log"
    error: "logs/errors.log"
  
  rotation:
    max_size_mb: 10
    backup_count: 5

# Security configuration
security:
  authentication:
    enabled: false  # Set to true for production
    method: "oauth2"
    
  data_encryption:
    enabled: false  # Set to true for sensitive data
    algorithm: "AES-256"
    
  audit_logging:
    enabled: true
    include_data_access: true

# Performance optimization
performance:
  parallel_processing:
    enabled: true
    max_workers: 4
    
  caching:
    enabled: true
    ttl_seconds: 3600
    max_size_mb: 500
    
  memory:
    max_usage_mb: 2048
    garbage_collection: true

# UI/UX configuration
ui:
  theme: "dark"
  sidebar_width: 300
  chart_height: 400
  
  features:
    real_time_charts: true
    export_options: true
    advanced_filters: true
    
  pagination:
    default_page_size: 50
    max_page_size: 1000
```

---

## 🔧 **Environment-Specific Configurations**

### **Development Environment**
```yaml
# config/config.dev.yaml
app:
  debug: true
  log_level: "DEBUG"

anomaly_detection:
  models:
    ensemble:
      enabled: false  # Use single models for testing
```

### **Production Environment**
```yaml
# config/config.prod.yaml
app:
  debug: false
  log_level: "WARNING"

security:
  authentication:
    enabled: true
  data_encryption:
    enabled: true
    
performance:
  parallel_processing:
    max_workers: 8
  memory:
    max_usage_mb: 8192
```

### **Testing Environment**
```yaml
# config/config.test.yaml
system:
  data_dir: "test_data"
  
data:
  storage:
    retention_days: 1  # Clean up test data quickly
```

---

## 🎯 **Configuration Validation**

### **Startup Validation**
The platform validates configuration on startup:

```python
# Automatic validation in run.py
def validate_config():
    """Validate configuration and create required directories."""
    required_dirs = [
        config.get('system.data_dir'),
        config.get('system.results_dir'),
        config.get('feedback.storage_dir'),
        config.get('system.logs_dir')
    ]
    
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
```

### **Runtime Validation**
- **Directory Existence:** All paths verified and created if missing
- **Model Configuration:** ML parameters validated before training
- **Resource Limits:** Memory and storage limits enforced
- **Security Settings:** Authentication and encryption verified

---

## ⚙️ **Advanced Configuration**

### **Dynamic Model Selection**
Configure ML models at runtime:

```yaml
# Switch between algorithms
anomaly_detection:
  active_model: "isolation_forest"  # Current: ensemble, isolation_forest, etc.
  
  auto_selection:
    enabled: true
    performance_threshold: 0.85
    fallback_model: "isolation_forest"
```

### **Custom Thresholds**
Environment-specific sensitivity:

```yaml
# Production: Conservative
monitoring:
  anomaly_detection:
    confidence_threshold: 0.9
    
# Development: Sensitive  
monitoring:
  anomaly_detection:
    confidence_threshold: 0.5
```

### **Storage Optimization**
```yaml
data:
  storage:
    # Optimize for performance
    compression: true
    compression_level: 6
    
    # Lifecycle management
    retention_days: 90
    archive_after_days: 30
    
    # Backup strategy
    backup_enabled: true
    backup_schedule: "daily"
    backup_retention: 7
```

---

## 🔍 **Configuration Troubleshooting**

### **Common Issues**

**Issue:** Directories not created properly
- **Cause:** ✅ **FIXED** - `run.py` now uses config.yaml
- **Solution:** Configuration-driven directory creation implemented

**Issue:** Results not saving to configured paths
- **Cause:** ✅ **FIXED** - ModelManager integration enhanced
- **Solution:** Proper config integration for all result storage

**Issue:** Feedback stored in wrong location
- **Cause:** ✅ **FIXED** - Path resolution corrected
- **Solution:** Uses `feedback.storage_dir` from config

### **Validation Commands**
```powershell
# Verify configuration
python -c "from core.config_loader import load_config; print('Config valid:', load_config())"

# Check directory structure
python scripts/verify_structure.py

# Test model configuration
python -c "from core.model_manager import ModelManager; ModelManager().validate_config()"
```

---

## 📖 **Related Documentation**

For implementation details:
- **Deployment:** See `DEPLOYMENT_GUIDE.md`
- **User Guide:** See `USER_GUIDE.md`
- **API Reference:** See `API_DOCUMENTATION.md`
- **Project Structure:** See `PROJECT_ORGANIZATION.md`

**Configuration Status:** Production Ready v2.1.0 with all fixes applied.
      
      # Ensemble configuration (NEW)
      ensemble_config:
        method: "weighted_average"  # Options: weighted_average, majority_vote, max_confidence
        models:
          isolation_forest:
            enabled: true
            weight: 0.3      # 0.0 to 1.0
          local_outlier_factor:
            enabled: true
            weight: 0.25
          one_class_svm:
            enabled: true
            weight: 0.2
          knn:
            enabled: false   # Can disable individual models
            weight: 0.15
          hdbscan:
            enabled: true
            weight: 0.1

# Data Source Configuration (Enhanced)

## **📊 Anomaly Storage Configuration**

### **🔒 Persistent Storage Settings**
Configure where anomaly history is stored (NEVER use cache directories for persistent data):

```yaml
# Anomaly History Storage (Persistent Data)
anomaly_storage:
  history_dir: "data/anomaly_history"     # ✅ PERSISTENT location (NOT cache)
  max_retention_days: 90                  # Keep anomaly history for X days
  backup_enabled: true                    # Enable automatic backups
  backup_interval_hours: 24               # Backup frequency
  compression_enabled: false              # Enable/disable compression
```

### **⚠️ Storage Location Guidelines**
- **✅ GOOD**: `data/anomaly_history/` - Persistent data directory
- **❌ BAD**: `cache/anomalies/` - Temporary cache directory (may be cleared)
- **✅ GOOD**: `persistent/anomaly_data/` - Any permanent storage path
- **❌ BAD**: `temp/anomalies/` - Temporary directory (may be cleared)

### **📁 Storage Structure**
```
data/anomaly_history/
├── anomaly_history.json      # Main anomaly detection records
├── daily_summary.json        # Daily aggregated statistics
├── baseline_metrics.json     # Learned baseline patterns
└── backups/                  # Automatic backups (if enabled)
```
data_source:
  # Primary data directory (production: /opt/arkime/json)
  directory: "data/"
  
  # Auto-loading settings
  auto_load: true
  max_files: 1000
  file_patterns:
    - "*.json"
    - "*.pcap_filtered.json"
  
  # Enhanced real-time monitoring
  watch_directory: true
  refresh_interval: 30  # seconds (deprecated - use monitoring.real_time.auto_refresh_interval)
  
  # Data validation
  validate_on_load: true
  required_fields:
    - "timestamp" 
    - "src_ip"
    - "dst_ip"

# Enhanced Anomaly Detection Configuration (NEW)
anomaly_detection:
  # Global settings
  default_threshold: 0.8
  max_anomalies: 1000
  
  # Storage and tracking (NEW)
  storage:
    enabled: true
    directory: "cache/anomalies"
    backup_enabled: true
    retention_days: 90
  
  # Baseline learning (NEW)
  baseline:
    enabled: true
    learning_period_days: 7
    min_samples: 100
    auto_adjust_threshold: true
    deviation_sensitivity: 2.0  # Standard deviations
  
  # Individual model configurations
  models:
    isolation_forest:
      enabled: true
      contamination: 0.05
      n_estimators: 100
      max_samples: "auto"
      random_state: 42
      
    local_outlier_factor:
      enabled: true
      n_neighbors: 20
      contamination: 0.05
      algorithm: "auto"
      
    one_class_svm:
      enabled: true
      nu: 0.05
      kernel: "rbf"
      gamma: "scale"
      
    knn:
      enabled: true
      n_neighbors: 5
      algorithm: "auto"
      metric: "minkowski"
      
    hdbscan:
      enabled: true
      min_cluster_size: 5
      min_samples: 3
      alpha: 1.0

# Enhanced Model Configuration
models:
  # Default algorithm (can be overridden by monitoring.real_time.anomaly_detection.model_type)
  default_algorithm: "ensemble"  # NEW: ensemble support
  
  # Training parameters (enhanced)
  default_params:
    IsolationForest:
      contamination: 0.05  # Reduced for better sensitivity
      n_estimators: 100
      random_state: 42
    LocalOutlierFactor:
      n_neighbors: 20
      contamination: 0.05
    OneClassSVM:
      nu: 0.05
      kernel: "rbf"
      gamma: "scale"
    KNN:
      n_neighbors: 5
      algorithm: "auto"
    HDBSCAN:
      min_cluster_size: 5
      min_samples: 3
  
  # Enhanced model management
  auto_retrain: true
  retrain_threshold: 0.8
  performance_window: 1000
  max_models: 10
  
  # Enhanced feature engineering
  feature_selection:
    enabled: true
    max_features: 50
    selection_method: "variance"
    
  # Model ensemble settings (NEW)
  ensemble:
    enabled: true
    default_method: "weighted_average"
    voting_threshold: 0.6
    confidence_weighting: true
```

## 🎯 **Enhanced Configuration Examples** *(NEW)*

### **Scenario 1: High-Security Environment**
For environments requiring maximum anomaly detection sensitivity:

```yaml
monitoring:
  real_time:
    enable_auto_refresh: true
    auto_refresh_interval: 60  # 1 minute updates
    anomaly_detection:
      enabled: true
      model_type: "ensemble"
      confidence_threshold: 0.5  # Lower threshold = more sensitive
      ensemble_config:
        method: "weighted_average"
        models:
          isolation_forest:
            enabled: true
            weight: 0.4  # Higher weight for proven algorithm
          local_outlier_factor:
            enabled: true
            weight: 0.3
          one_class_svm:
            enabled: true
            weight: 0.3

anomaly_detection:
  default_threshold: 0.6  # More sensitive
  baseline:
    enabled: true
    learning_period_days: 14  # Longer learning period
    deviation_sensitivity: 1.5  # More sensitive to deviations
```

### **Scenario 2: Performance-Optimized Environment**
For environments with high data volumes requiring efficient processing:

```yaml
monitoring:
  real_time:
    enable_auto_refresh: true
    auto_refresh_interval: 600  # 10 minutes
    anomaly_detection:
      enabled: true
      model_type: "isolation_forest"  # Single fast model
      confidence_threshold: 0.8  # Higher threshold = less sensitive but faster

data_source:
  max_files: 100  # Limit data processing
  
anomaly_detection:
  max_anomalies: 500  # Limit stored anomalies
  storage:
    retention_days: 30  # Shorter retention
```

### **Scenario 3: Balanced Production Environment**
Recommended settings for most production environments:

```yaml
monitoring:
  real_time:
    enable_auto_refresh: true
    auto_refresh_interval: 300  # 5 minutes
    anomaly_detection:
      enabled: true
      model_type: "ensemble"
      confidence_threshold: 0.7  # Balanced sensitivity
      ensemble_config:
        method: "weighted_average"
        models:
          isolation_forest:
            enabled: true
            weight: 0.3
          local_outlier_factor:
            enabled: true
            weight: 0.25
          one_class_svm:
            enabled: true
            weight: 0.25
          hdbscan:
            enabled: true
            weight: 0.2

anomaly_detection:
  default_threshold: 0.8
  baseline:
    enabled: true
    learning_period_days: 7
    deviation_sensitivity: 2.0
  storage:
    retention_days: 90
```

## 🔧 **Configuration Management**

### **Loading Configuration Priorities**
1. **Environment Variables** (highest priority)
2. **config.yaml** file settings
3. **Default hardcoded values** (lowest priority)

### **Environment Variable Overrides**
Override any setting using environment variables:

```bash
# Override auto-refresh interval
export NDR_MONITORING_REAL_TIME_AUTO_REFRESH_INTERVAL=120

# Override anomaly detection model
export NDR_MONITORING_REAL_TIME_ANOMALY_DETECTION_MODEL_TYPE="isolation_forest"

# Override confidence threshold
export NDR_MONITORING_REAL_TIME_ANOMALY_DETECTION_CONFIDENCE_THRESHOLD=0.8
```

### **Baseline Learning Configuration**
The baseline learning system adapts to your network patterns:

```yaml
baseline:
  enabled: true
  learning_period_days: 7      # How long to learn patterns
  min_samples: 100             # Minimum samples before baseline is active
  auto_adjust_threshold: true  # Automatically adjust thresholds
  deviation_sensitivity: 2.0   # Standard deviations for alerts (1.0-3.0)
```

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
