"""
Logging configuration for the Network Anomaly Detection Platform.
Configures loggers for different parts of the application.
"""

import logging
import logging.handlers
import os
from datetime import datetime
import sys

# Global flag to prevent multiple logging setups
_logging_initialized = False

def setup_logging(log_dir=None, log_level=logging.INFO, force_reinit=False):
    """
    Set up logging for the application.
    
    Args:
        log_dir (str, optional): Directory to store log files
        log_level: Logging level
        force_reinit (bool): Force re-initialization even if already set up
    
    Returns:
        logging.Logger: Root logger
    """
    global _logging_initialized
    
    # Prevent duplicate initialization unless forced
    if _logging_initialized and not force_reinit:
        return logging.getLogger()
    
    # Create logs directory if it doesn't exist
    if log_dir is None:
        log_dir = os.path.join(os.getcwd(), "logs")
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Define log file paths
    app_log_path = os.path.join(log_dir, "app.log")
    error_log_path = os.path.join(log_dir, "errors.log")
    model_log_path = os.path.join(log_dir, "models.log")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicates
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    # Format for log messages
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (WARNING level and above to reduce spam)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    # Set encoding for Windows console
    if hasattr(console_handler.stream, 'reconfigure'):
        try:
            console_handler.stream.reconfigure(encoding='utf-8')
        except:
            pass  # Fallback if reconfigure is not available
    root_logger.addHandler(console_handler)
    
    # App log file handler (INFO level and above)
    app_handler = logging.handlers.RotatingFileHandler(
        app_log_path, maxBytes=10485760, backupCount=5, encoding='utf-8'
    )
    app_handler.setLevel(logging.INFO)
    app_handler.setFormatter(formatter)
    root_logger.addHandler(app_handler)
    
    # Error log file handler (ERROR level and above)
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_path, maxBytes=10485760, backupCount=5, encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)
    
    # Model-specific logger
    model_logger = logging.getLogger("model_operations")
    model_logger.propagate = False  # Don't propagate to root logger
    
    model_handler = logging.handlers.RotatingFileHandler(
        model_log_path, maxBytes=10485760, backupCount=5, encoding='utf-8'
    )
    model_handler.setLevel(logging.INFO)
    model_handler.setFormatter(formatter)
    model_logger.addHandler(model_handler)
    
    # Set up other specific loggers that are actually used in the codebase
    setup_module_logger("data_manager", log_dir, log_level, formatter)
    setup_module_logger("model_manager", log_dir, log_level, formatter)
    setup_module_logger("data_processing", log_dir, log_level, formatter)
    setup_module_logger("visualization", log_dir, log_level, formatter)
    setup_module_logger("streamlit_app", log_dir, log_level, formatter)
    setup_module_logger("anomaly_detection", log_dir, log_level, formatter)
    
    # Create a special logger for application startup/important messages
    app_logger = logging.getLogger("app_startup")
    if not app_logger.handlers:
        startup_console_handler = logging.StreamHandler(sys.stdout)
        startup_console_handler.setLevel(logging.INFO)
        startup_console_handler.setFormatter(formatter)
        app_logger.addHandler(startup_console_handler)
        app_logger.propagate = False
    
    # Mark logging as initialized
    _logging_initialized = True
    
    # Log initialization message only once
    root_logger.info(f"Logging initialized with log directory: {log_dir}")
    
    return root_logger

def setup_module_logger(name, log_dir, log_level, formatter):
    """
    Set up a logger for a specific module.
    
    Args:
        name (str): Name of the module
        log_dir (str): Directory to store log file
        log_level: Logging level
        formatter: Log formatter
    
    Returns:
        logging.Logger: Module logger
    """
    logger = logging.getLogger(name)
    
    # Don't setup if already has handlers (prevents duplicates)
    if logger.handlers:
        return logger
        
    logger.propagate = False  # Don't propagate to root logger
    logger.setLevel(log_level)
    
    log_path = os.path.join(log_dir, f"{name}.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5242880, backupCount=3, encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def get_logger(name):
    """
    Get a logger with the given name.
    
    Args:
        name (str): Name of the logger
    
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)

def reset_logging():
    """
    Reset logging configuration (useful for development).
    """
    global _logging_initialized
    _logging_initialized = False
    
    # Clear all existing loggers
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = True
    
    # Clear root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

def is_logging_initialized():
    """
    Check if logging has been initialized.
    
    Returns:
        bool: True if logging is initialized
    """
    return _logging_initialized

def set_log_level(level):
    """
    Set the log level for all loggers.
    
    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    for handler in root_logger.handlers:
        handler.setLevel(level)