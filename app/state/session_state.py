"""
Manages the application's session state to persist data and user selections across tabs.
Initializes default values and provides accessor functions.
"""

import streamlit as st
import os
import glob
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import re

def init_session_state():
    """Initialize session state variables if they don't exist."""
    
    # Basic session state variables
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        
        # Data management
        st.session_state.data_files = []  # List of loaded file paths
        st.session_state.selected_files = []  # List of user-selected files
        st.session_state.raw_data = {}  # Dictionary mapping file paths to raw JSON data
        st.session_state.processed_data = {}  # Dictionary mapping file paths to processed DataFrames
        st.session_state.combined_data = None  # Combined DataFrame of all selected files
        st.session_state.data_source = 'none'  # Current data source: auto_loaded, realtime, manual, none
        st.session_state.data_load_time = None  # When data was last loaded
        st.session_state.loaded_files = []  # List of files currently loaded
        
        # Model management
        st.session_state.selected_model = None  # Currently selected model
        st.session_state.model_results = {}  # Results from different models
        st.session_state.anomalies = []  # List of detected anomalies
        st.session_state.anomaly_indices = []  # Indices of detected anomalies
        
        # Initialize model manager
        try:
            from core.model_manager import ModelManager
            st.session_state.model_manager = ModelManager()
        except Exception as e:
            print(f"Warning: Could not initialize ModelManager: {e}")
            st.session_state.model_manager = None
        
        # Visualization state
        st.session_state.visualization_config = {
            'chart_type': 'bar',
            'color_theme': 'blue',
            'show_legend': True
        }
        
        # MITRE mapping state
        st.session_state.mitre_mappings = {}  # Maps anomalies to MITRE techniques
        
        # Feedback collection
        st.session_state.feedback = {}  # Analyst feedback on anomalies
        
        # Application settings
        st.session_state.settings = {
            'anomaly_threshold': 0.8,
            'max_anomalies': 100,
            'auto_refresh': False,
            'dark_mode': False
        }
        
        # Load any existing data files from the data directory
        refresh_data_files()
        
        # Initialize data manager and auto-load data
        try:
            from core.data_manager import get_data_manager
            data_manager = get_data_manager()
            data_manager.initialize_default_data()
        except Exception as e:
            print(f"Warning: Could not initialize data manager: {e}")

def refresh_data_files():
    """Scan the data directory for available JSON files and update session state."""
    # Get all JSON files in the data directory and its subdirectories
    json_files = []
    for path in Path('data').rglob('*.json'):
        if path.is_file():
            json_files.append(str(path))
    
    # Update session state
    st.session_state.data_files = json_files

def get_data_files():
    """Return the list of available data files."""
    return st.session_state.data_files

def load_json_file(file_path):
    """
    Load a JSON file, process it, and return the DataFrame.
    Returns None if loading/processing fails.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Ensure session state dictionaries exist
        if not hasattr(st.session_state, 'raw_data') or st.session_state.raw_data is None:
            st.session_state.raw_data = {}
        if not hasattr(st.session_state, 'processed_data') or st.session_state.processed_data is None:
            st.session_state.processed_data = {}
        
        # Store raw data in session state
        st.session_state.raw_data[file_path] = data
        
        # Process the data and store the processed DataFrame
        processed_df = process_json_data(data)
        
        # Validate the DataFrame
        if processed_df is not None and not processed_df.empty:
            st.session_state.processed_data[file_path] = processed_df
            return processed_df
        else:
            print(f"Warning: Processed DataFrame is empty for {file_path}")
            return None
            
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")
        return None

def process_json_data(data):
    """
    Process raw JSON data into a flat DataFrame.
    Handles the specific JSON structure with _source.layers and list values.
    """
    records = []
    
    # Handle single entry or list of entries
    if isinstance(data, dict):
        # Single entry
        flat_record = flatten_json_entry(data)
        if flat_record:  # Only add non-empty records
            records.append(flat_record)
    elif isinstance(data, list):
        # List of entries
        for entry in data:
            flat_record = flatten_json_entry(entry)
            if flat_record:  # Only add non-empty records
                records.append(flat_record)
    
    # Check if we have any valid records
    if not records:
        print("Warning: No valid records found in JSON data")
        return None
    
    # Convert to DataFrame
    try:
        df = pd.DataFrame(records)
    except Exception as e:
        print(f"Error creating DataFrame: {str(e)}")
        return None
    
    # Validate DataFrame
    if df.empty:
        print("Warning: DataFrame is empty after processing")
        return None
    
    # Apply type conversions for numeric fields
    numeric_columns = [
        'frame_number', 'frame_len', 'tcp_srcport', 'tcp_dstport', 
        'udp_srcport', 'udp_dstport'
    ]
    
    for col in df.columns:
        # Convert all potential numeric columns to numeric types
        try:
            # Try to convert to numeric, forcing errors to NaN
            numeric_values = pd.to_numeric(df[col], errors='coerce')
            
            # If the conversion worked and didn't create too many NaNs, use it
            if not numeric_values.isna().all() and numeric_values.notna().sum() > 0:
                df[col] = numeric_values
        except:
            pass
    
    # Convert timestamp strings to datetime objects
    if 'frame_time' in df.columns:
        try:
            # Try multiple date formats with explicit patterns
            date_formats = [
                '%b %d, %Y %H:%M:%S.%f000 %z',   # Jul 22, 2025 16:53:50.356985000 +08
                '%b %d, %Y %H:%M:%S.%f %z',      # Jul 22, 2025 16:53:50.356985 +08
                '%Y-%m-%d %H:%M:%S.%f',          # 2025-07-22 16:53:50.356985
                '%Y-%m-%d %H:%M:%S',             # 2025-07-22 16:53:50
                '%Y-%m-%d'                       # 2025-07-22
            ]
            
            # Try each format until one works
            df['timestamp'] = None
            for fmt in date_formats:
                try:
                    # Clean up microseconds to avoid format issues
                    clean_times = df['frame_time'].astype(str).apply(
                        lambda x: re.sub(r'(\.\d{6})\d+', r'\1', x) if isinstance(x, str) else x
                    )
                    
                    # Parse with the current format
                    df['timestamp'] = pd.to_datetime(clean_times, format=fmt, errors='coerce')
                    
                    # If we got valid dates, break the loop
                    if not df['timestamp'].isna().all():
                        break
                except:
                    continue
            
            # If all explicit formats failed, try pandas automatic parsing
            if df['timestamp'].isna().all():
                df['timestamp'] = pd.to_datetime(df['frame_time'], errors='coerce')
        except Exception as e:
            print(f"Error parsing timestamps: {str(e)}")
            # Create a timestamp column from frame_number as fallback
            if 'frame_number' in df.columns:
                base_date = datetime(2025, 7, 22)  # Default to July 22, 2025
                df['timestamp'] = pd.to_datetime(base_date) + pd.to_timedelta(df['frame_number'], unit='s')
    
    # Add row number as a numeric feature
    df['row_id'] = np.arange(len(df))
    
    return df

def flatten_json_entry(entry):
    """
    Flatten a single JSON entry with the structure _source.layers.
    Normalize dotted keys to underscores.
    """
    flat_entry = {}
    
    # Check if the entry has the expected structure
    if isinstance(entry, dict) and '_source' in entry and 'layers' in entry['_source']:
        layers = entry['_source']['layers']
        
        # Process each field in layers
        for key, value in layers.items():
            # Normalize key (replace dots with underscores)
            normalized_key = key.replace('.', '_')
            
            # Extract value from single-element list if needed
            if isinstance(value, list) and len(value) == 1:
                extracted_value = value[0]
                # Try to convert to numeric if it looks like a number
                try:
                    if isinstance(extracted_value, str) and extracted_value.replace('.', '', 1).isdigit():
                        if '.' in extracted_value:
                            flat_entry[normalized_key] = float(extracted_value)
                        else:
                            flat_entry[normalized_key] = int(extracted_value)
                    else:
                        flat_entry[normalized_key] = extracted_value
                except:
                    flat_entry[normalized_key] = extracted_value
            elif isinstance(value, list) and len(value) > 1:
                # Handle multi-value lists by joining as string
                flat_entry[normalized_key] = str(value)
            else:
                flat_entry[normalized_key] = value
    
    return flat_entry if flat_entry else None

def update_combined_data():
    """Combine all selected processed data into a single DataFrame."""
    if not st.session_state.selected_files:
        st.session_state.combined_data = None
        return
    
    # Collect DataFrames from all selected files
    dfs = []
    for file_path in st.session_state.selected_files:
        if file_path in st.session_state.processed_data:
            dfs.append(st.session_state.processed_data[file_path])
    
    # Combine all DataFrames
    if dfs:
        st.session_state.combined_data = pd.concat(dfs, ignore_index=True)
    else:
        st.session_state.combined_data = None