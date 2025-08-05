# NDR Platform - Logging Configuration Guide

## Overview

The NDR Platform now has a comprehensive, non-duplicating logging system designed to handle Streamlit's module reloading behavior while providing detailed logging for debugging and monitoring.

## Logging Structure

### 🏗️ Architecture

```
logs/
├── app.log              # General application logs (INFO+)
├── errors.log           # Error logs only (ERROR+)
├── data_manager.log     # Data management operations
├── model_manager.log    # ML model operations
├── anomaly_detection.log # Anomaly detection processes
├── streamlit_app.log    # Streamlit-specific logs
├── data_processing.log  # Data processing operations
└── visualization.log    # Chart and graph generation
```

### 📊 Log Levels

- **Console Output**: WARNING+ (reduced spam)
- **File Logging**: INFO+ (comprehensive)
- **Error Files**: ERROR+ (critical issues only)
- **Startup Logger**: INFO+ (important startup messages)

## Key Features

### ✅ Duplication Prevention
```python
# Global flag prevents multiple initialization
_logging_initialized = False

def setup_logging(log_dir=None, log_level=logging.INFO, force_reinit=False):
    global _logging_initialized
    if _logging_initialized and not force_reinit:
        return logging.getLogger()
```

### ✅ Streamlit Compatibility
- Handles hot reloading without log duplication
- Module-specific loggers don't propagate to root (prevents spam)
- Startup messages logged only once per session

### ✅ File Rotation
- 10MB max file size for main logs
- 5MB max for module logs  
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
