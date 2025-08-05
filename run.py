"""
Main entry point for the Network Anomaly Detection Platform.
Sets up logging, environment, and launches the Streamlit application.
"""

import os
import sys
import streamlit as st
from pathlib import Path
import logging

# Add project root to Python path for absolute imports
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Import logging configuration early
from core.logging_config import setup_logging, get_logger

# Set up logging before any other imports (only once)
log_dir = os.path.join(project_root, "logs")
logger = setup_logging(log_dir=log_dir)
startup_logger = get_logger("app_startup")

# Only log startup once during initial setup
if not hasattr(setup_logging, '_startup_logged'):
    startup_logger.info("Starting Network Anomaly Detection Platform")
    
    # Create necessary directories
    directories = ["data", "models", "logs", "feedback", "cache", "config", "app/assets"]
    created_dirs = []
    for directory in directories:
        dir_path = Path(project_root, directory)
        if not dir_path.exists():
            os.makedirs(dir_path, exist_ok=True)
            created_dirs.append(directory)
    
    # Only log directories that were actually created
    if created_dirs:
        startup_logger.info(f"Created directories: {', '.join(created_dirs)}")
    
    # Mark that startup has been logged
    setup_logging._startup_logged = True

# Import and run the Streamlit app
from app.main import main

if __name__ == "__main__":
    try:
        # Only log application launch once
        if not hasattr(setup_logging, '_app_launched'):
            startup_logger.info("Launching Streamlit application")
            setup_logging._app_launched = True
        main()
    except Exception as e:
        logger.exception(f"Unhandled exception in main application: {str(e)}")