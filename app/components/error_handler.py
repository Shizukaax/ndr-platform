"""
Error handling utilities for the Network Anomaly Detection Platform.
Provides decorators and functions for consistent error handling across the application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import traceback
import logging
import functools
import os
from datetime import datetime

# Setup logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "anomaly_detection.log")

# Configure logger
logger = logging.getLogger("anomaly_detection")
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)

# Create formatter and add it to handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def handle_error(func):
    """
    Decorator to handle errors in Streamlit pages.
    Catches exceptions, logs them, and displays user-friendly messages.
    
    Args:
        func: The function to decorate
    
    Returns:
        The decorated function with error handling
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            
            st.error(f"An error occurred: {str(e)}")
            with st.expander("Error details"):
                st.code(traceback.format_exc())
            
            return None
    
    return wrapper

def validate_inputs(**kwargs):
    """
    Validate input parameters for functions.
    
    Args:
        **kwargs: Parameters to validate with their requirements
    
    Returns:
        bool: True if all validations pass, False otherwise
    """
    all_valid = True
    
    for param_name, (param_value, validation_func, error_msg) in kwargs.items():
        if not validation_func(param_value):
            st.error(f"Invalid {param_name}: {error_msg}")
            all_valid = False
    
    return all_valid

def is_dataframe(df):
    """Check if value is a pandas DataFrame."""
    return isinstance(df, pd.DataFrame)

def is_numeric_array(arr):
    """Check if value is a numeric numpy array."""
    return isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.number)

def is_positive(val):
    """Check if value is positive."""
    return val > 0

def log_action(action, details=None):
    """
    Log user actions for auditing.
    
    Args:
        action (str): The action performed
        details (dict, optional): Additional details about the action
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if details:
        logger.info(f"ACTION [{timestamp}] {action}: {details}")
    else:
        logger.info(f"ACTION [{timestamp}] {action}")