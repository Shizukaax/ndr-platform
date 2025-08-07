# NDR Platform v2.1.0 - Logging Configuration Guide

## 📋 Overview

Following recent critical fixes, the NDR Platform v2.1.0 features a comprehensive, production-ready logging system designed for enterprise deployment with proper error tracking and debugging capabilities.

---

## 🏗️ **Enhanced Logging Architecture**

### **📁 Log File Structure**

```
logs/
├── 📄 app.log                    # ✅ Main application logs (INFO+)
├── 📄 errors.log                 # ✅ Error tracking (ERROR+)
├── 📄 anomaly_detection.log      # ✅ ML detection processes
├── 📄 model_manager.log          # ✅ Model lifecycle operations
├── 📄 data_processing.log        # ✅ Data pipeline operations
├── 📄 streamlit_app.log          # ✅ UI application logs
├── 📄 visualization.log          # ✅ Chart generation logs
└── 📄 data_manager.log           # ✅ Data management logs
```

### **🎯 Log Level Configuration**

```python
# Production logging levels
LOGGING_LEVELS = {
    'console': 'WARNING',      # Reduced console spam
    'file': 'INFO',           # Comprehensive file logging
    'error_file': 'ERROR',    # Critical issues only
    'debug_mode': 'DEBUG'     # Development debugging
}
```

---

## 🔧 **Configuration Management**

### **📋 Configuration via config.yaml**

```yaml
# Enhanced logging configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  
  # ✅ Log file paths
  files:
    app: "logs/app.log"
    anomaly: "logs/anomaly_detection.log"
    model: "logs/model_manager.log"
    data: "logs/data_processing.log"
    error: "logs/errors.log"
  
  # ✅ File rotation settings
  rotation:
    max_size_mb: 10
    backup_count: 5
    
  # ✅ Console logging
  console:
    enabled: true
    level: "WARNING"
    format: "%(levelname)s: %(message)s"
```

### **🚀 Programmatic Setup**

```python
from core.logging_config import setup_logging

# ✅ Initialize enhanced logging system
logger = setup_logging(
    log_dir="logs",
    log_level=logging.INFO,
    enable_file_rotation=True
)

# ✅ Module-specific logging
def get_module_logger(module_name):
    """Get logger for specific module"""
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)
    return logger
```

---

## 📊 **Critical Issue Logging** *(Recent Fixes)*

### **✅ Fixed Issues Tracking**

The logging system now properly tracks all recent fixes:

```python
# ✅ NaN handling logging
logger.info("Port value cleaned: %s -> %s", original_port, cleaned_port)

# ✅ Arrow compatibility logging
logger.debug("DataFrame cleaned for Arrow display: %d rows", len(df))

# ✅ Results saving logging
logger.info("Results saved to configured directory: %s", results_path)

# ✅ Directory creation logging
logger.info("Directory created per config: %s", dir_path)

# ✅ Configuration validation logging
logger.info("Configuration validated successfully")
```

### **🚨 Error Tracking Examples**

```python
# Model Manager errors
try:
    results = model_manager.train_model(data, "isolation_forest")
    if 'anomaly_indices' not in results:
        logger.error("Missing anomaly_indices in results - applying fix")
        results['anomaly_indices'] = []
except Exception as e:
    logger.error("Model training failed: %s", str(e), exc_info=True)

# Data processing errors
try:
    clean_df = clean_dataframe_for_arrow(df)
    logger.info("DataFrame cleaned successfully: %d rows", len(clean_df))
except Exception as e:
    logger.error("DataFrame cleaning failed: %s", str(e), exc_info=True)

# Configuration errors
try:
    config = load_config()
    logger.info("Configuration loaded successfully")
except Exception as e:
    logger.error("Configuration loading failed: %s", str(e), exc_info=True)
```

---

## 🔍 **Monitoring & Debugging**

### **📈 Log Analysis Commands**

```bash
# Monitor real-time application logs
tail -f logs/app.log

# Check for errors
grep -i "error" logs/app.log logs/errors.log

# Monitor model operations
tail -f logs/model_manager.log

# Check anomaly detection performance
grep -i "anomaly" logs/anomaly_detection.log | tail -20

# Monitor configuration issues
grep -i "config" logs/app.log
```

### **🔧 Debug Mode Activation**

```python
# Enable debug logging for troubleshooting
import logging
from core.logging_config import setup_logging

# Temporary debug mode
debug_logger = setup_logging(
    log_level=logging.DEBUG,
    console_level=logging.DEBUG
)

# Debug specific modules
logging.getLogger('core.model_manager').setLevel(logging.DEBUG)
logging.getLogger('app.pages.explain_feedback').setLevel(logging.DEBUG)
```

---

## 🚀 **Production Logging Best Practices**

### **✅ Performance Optimization**

```python
# Efficient logging practices
logger = logging.getLogger(__name__)

# Use lazy formatting
logger.info("Processing %d records", record_count)

# Conditional debug logging
if logger.isEnabledFor(logging.DEBUG):
    logger.debug("Complex operation result: %s", complex_calculation())

# Context managers for operation tracking
import contextlib
import time

@contextlib.contextmanager
def log_operation(operation_name):
    start_time = time.time()
    logger.info("Starting %s", operation_name)
    try:
        yield
        duration = time.time() - start_time
        logger.info("Completed %s in %.2f seconds", operation_name, duration)
    except Exception as e:
        logger.error("Failed %s: %s", operation_name, str(e), exc_info=True)
        raise
```

### **🔒 Security Considerations**

```python
# Safe logging practices
def safe_log_data(data, max_length=100):
    """Log data safely without exposing sensitive information"""
    if isinstance(data, dict):
        # Log structure without values
        keys = list(data.keys())
        logger.info("Processing data with keys: %s", keys)
    elif isinstance(data, str) and len(data) > max_length:
        # Truncate long strings
        logger.info("Processing text data (length: %d): %s...", 
                   len(data), data[:max_length])
    else:
        logger.info("Processing data: %s", data)
```

---

## 📊 **Log Rotation & Maintenance**

### **🔄 Automatic Rotation**

```python
# Enhanced rotation configuration
import logging.handlers

def setup_rotating_logger(name, filepath, max_bytes=10*1024*1024, backup_count=5):
    """Setup rotating file handler"""
    handler = logging.handlers.RotatingFileHandler(
        filepath, 
        maxBytes=max_bytes, 
        backupCount=backup_count,
        encoding='utf-8'
    )
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    return logger
```

### **🧹 Log Cleanup Scripts**

```bash
# Clean old logs (keep last 7 days)
find logs/ -name "*.log.*" -mtime +7 -delete

# Archive logs monthly
tar -czf logs/archive/logs-$(date +%Y%m).tar.gz logs/*.log

# Monitor log disk usage
du -sh logs/
```

---

## 🚨 **Troubleshooting Logging Issues**

### **Common Problems & Solutions**

**Issue:** Duplicate log entries
```python
# ✅ FIXED: Global initialization flag prevents duplicates
_logging_initialized = False

def setup_logging():
    global _logging_initialized
    if _logging_initialized:
        return logging.getLogger()
    # ... initialization code
    _logging_initialized = True
```

**Issue:** Missing log entries
```python
# Verify logger configuration
logger = logging.getLogger(__name__)
logger.info("Test message")  # Should appear in logs
```

**Issue:** Large log files
```python
# Enable log rotation
handler = logging.handlers.RotatingFileHandler(
    'logs/app.log', maxBytes=10*1024*1024, backupCount=3
)
```

---

## 📖 **Related Documentation**

For platform details:
- **Configuration:** See `CONFIGURATION_GUIDE.md`
- **Deployment:** See `DEPLOYMENT_GUIDE.md`
- **User Guide:** See `USER_GUIDE.md`
- **API Reference:** See `API_DOCUMENTATION.md`

**Logging Status:** Production Ready v2.1.0 with comprehensive error tracking.  
- 3-5 backup files maintained
- UTF-8 encoding for all log files

## Usage Examples

### Basic Logging
```python
from core.logging_config import get_logger

logger = get_logger("my_module")
logger.info("Operation completed successfully")
logger.error("Something went wrong", exc_info=True)
```

### Module-Specific Logging
```python
# Data processing
data_logger = get_logger("data_processing")
data_logger.info("Processing 1000 records")

# Model operations
model_logger = get_logger("model_manager")
model_logger.info("Training isolation forest model")
```

### Startup/Important Messages
```python
from core.logging_config import get_logger

startup_logger = get_logger("app_startup")
startup_logger.info("Application initialized successfully")
```

## Configuration Functions

### Core Functions
```python
# Initialize logging (call once)
setup_logging(log_dir="/path/to/logs", log_level=logging.INFO)

# Get logger instance
logger = get_logger("module_name")

# Check if initialized
if is_logging_initialized():
    logger.info("Logging ready")

# Reset for development
reset_logging()

# Change log level
set_log_level(logging.DEBUG)
```

## Best Practices

### ✅ DO
- Use module-specific loggers: `get_logger("your_module")`
- Include exception info for errors: `logger.error("msg", exc_info=True)`
- Use appropriate log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Log important state changes and operations
- Include relevant context in log messages

### ❌ DON'T
- Call `setup_logging()` multiple times unnecessarily
- Use print() statements instead of logging
- Log sensitive information (passwords, tokens)
- Create excessive DEBUG logs in production
- Log inside tight loops without rate limiting

## Log Level Guidelines

### DEBUG
```python
logger.debug("Variable x = %s", x)
logger.debug("Entering function with params: %s", params)
```

### INFO
```python
logger.info("Starting data processing for %d records", count)
logger.info("Model training completed in %.2f seconds", duration)
```

### WARNING
```python
logger.warning("Deprecated feature used: %s", feature_name)
logger.warning("Performance threshold exceeded: %d ms", response_time)
```

### ERROR
```python
logger.error("Failed to process file: %s", filename, exc_info=True)
logger.error("Database connection lost, attempting reconnect")
```

### CRITICAL
```python
logger.critical("Application unable to start: %s", error_msg)
logger.critical("Security breach detected from IP: %s", ip_address)
```

## Monitoring and Maintenance

### Log File Monitoring
```python
# Check log file sizes
import os
log_dir = "logs"
for filename in os.listdir(log_dir):
    if filename.endswith('.log'):
        size = os.path.getsize(os.path.join(log_dir, filename))
        print(f"{filename}: {size/1024/1024:.1f} MB")
```

### Performance Impact
- File I/O is asynchronous where possible
- Log rotation prevents unlimited file growth
- Console output limited to WARNING+ to reduce overhead
- Module loggers prevent root logger spam

## Troubleshooting

### Issue: Duplicate Log Messages
**Solution**: Check if `setup_logging()` is called multiple times
```python
from core.logging_config import is_logging_initialized
if not is_logging_initialized():
    setup_logging()
```

### Issue: Missing Log Files
**Solution**: Ensure log directory exists and has write permissions
```python
import os
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
```

### Issue: Log Files Too Large
**Solution**: Logs auto-rotate, but you can manually clear if needed
```python
from core.logging_config import reset_logging
reset_logging()
setup_logging()
```

### Issue: Console Spam
**Solution**: Console is set to WARNING+, module loggers don't propagate
```python
from core.logging_config import set_log_level
import logging
set_log_level(logging.ERROR)  # Only show errors in console
```

## Development vs Production

### Development Settings
```python
# More verbose logging
setup_logging(log_level=logging.DEBUG)

# Enable all console output
console_handler.setLevel(logging.INFO)
```

### Production Settings
```python
# Standard logging
setup_logging(log_level=logging.INFO)

# Minimal console output
console_handler.setLevel(logging.WARNING)
```

## Testing Logging

Run the logging test to verify configuration:
```bash
python test_logging.py
```

Expected output:
- ✅ Initialization protection working
- ✅ No handler duplication  
- ✅ Module loggers configured
- ✅ Reload simulation successful

## Configuration Summary

The current logging configuration:
- **Prevents duplication** during Streamlit reloading
- **Module-specific logs** for different components
- **Rotating files** to manage disk space
- **UTF-8 encoding** for international characters
- **Console spam reduction** with WARNING+ threshold
- **Startup logger** for important application messages
- **Comprehensive error tracking** with separate error logs

This configuration ensures reliable, maintainable logging that works well with Streamlit's development and production environments.
