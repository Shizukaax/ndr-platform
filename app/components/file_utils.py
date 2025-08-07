"""
File handling utilities for the Network Anomaly Detection Platform.
Provides functions for loading, parsing, and processing data files.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
import tempfile
import shutil
from core.config_loader import load_config

from app.state.session_state import load_json_file, refresh_data_files

def upload_file(uploaded_file, target_dir="data"):
    """
    Save an uploaded file to the target directory.
    
    Args:
        uploaded_file (UploadedFile): The uploaded file from st.file_uploader
        target_dir (str): Target directory to save the file
    
    Returns:
        str: Path to the saved file, or None if failed
    """
    if uploaded_file is None:
        return None
    
    try:
        # Create target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)
        
        # Create file path
        file_path = os.path.join(target_dir, uploaded_file.name)
        
        # Save the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Refresh the list of data files
        refresh_data_files()
        
        return file_path
    
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def validate_json(content):
    """
    Validate that a string is valid JSON with the expected structure.
    
    Args:
        content (str): JSON content to validate
    
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # Parse JSON
        data = json.loads(content)
        
        # Check for expected structure
        if not isinstance(data, dict) and not isinstance(data, list):
            return False, "JSON must be an object or array"
        
        if isinstance(data, dict):
            # Single entry
            if '_source' not in data:
                return False, "Missing '_source' key in JSON"
            if 'layers' not in data['_source']:
                return False, "Missing 'layers' key in _source"
        
        elif isinstance(data, list):
            # Multiple entries
            if not data:
                return False, "JSON array is empty"
            
            # Check first entry for structure
            if not isinstance(data[0], dict):
                return False, "Array items must be JSON objects"
            if '_source' not in data[0]:
                return False, "Missing '_source' key in JSON array items"
            if 'layers' not in data[0]['_source']:
                return False, "Missing 'layers' key in _source of array items"
        
        return True, ""
    
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {str(e)}"
    except Exception as e:
        return False, f"Error validating JSON: {str(e)}"

def load_and_process_json(file_path):
    """
    Load a JSON file and process it into a pandas DataFrame.
    
    Args:
        file_path (str): Path to the JSON file
    
    Returns:
        tuple: (success, result), where result is either a DataFrame or error message
    """
    try:
        # Load the file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process the data
        df = process_json_data(data)
        
        return True, df
    
    except Exception as e:
        return False, f"Error loading or processing file: {str(e)}"

def process_json_data(data):
    """
    Process JSON data with the structure _source.layers into a flat DataFrame.
    
    Args:
        data (dict or list): JSON data to process
    
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    records = []
    
    # Handle both single entry and list of entries
    if isinstance(data, dict):
        # Single entry
        records.append(flatten_json_entry(data))
    elif isinstance(data, list):
        # List of entries
        for entry in data:
            records.append(flatten_json_entry(entry))
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    # Convert timestamp strings to datetime objects if they exist
    if 'frame_time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['frame_time'], errors='coerce')
    
    # Add row number as a numeric feature
    df['row_id'] = np.arange(len(df))
    
    # Convert common numeric columns if they exist
    numeric_cols = ['frame_number', 'frame_len', 'tcp_srcport', 'tcp_dstport', 'udp_srcport', 'udp_dstport']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Try to convert all columns that could be numeric
    for col in df.columns:
        try:
            if any(num_name in col.lower() for num_name in ['len', 'port', 'number', 'count', 'seq', 'ack', 'win']):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            pass
    
    return df

def flatten_json_entry(entry):
    """
    Flatten a single JSON entry with the structure _source.layers.
    
    Args:
        entry (dict): JSON entry to flatten
    
    Returns:
        dict: Flattened entry
    """
    flat_entry = {}
    
    # Check if the entry has the expected structure
    if '_source' in entry and 'layers' in entry['_source']:
        layers = entry['_source']['layers']
        
        # Process each field in layers
        for key, value in layers.items():
            # Normalize key (replace dots with underscores)
            normalized_key = key
            
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
                        continue
                except:
                    pass
                
                flat_entry[normalized_key] = extracted_value
            else:
                flat_entry[normalized_key] = value
    
    return flat_entry

def get_json_files(directory="data", recursive=True):
    """
    Get all JSON files in the specified directory.
    
    Args:
        directory (str): Directory to search for JSON files
        recursive (bool): Whether to search recursively
    
    Returns:
        list: List of JSON file paths
    """
    json_files = []
    
    if recursive:
        for path in Path(directory).rglob('*.json'):
            if path.is_file():
                json_files.append(str(path))
    else:
        for path in Path(directory).glob('*.json'):
            if path.is_file():
                json_files.append(str(path))
    
    return json_files

def parse_timestamp(timestamp_str):
    """
    Parse a timestamp string into a datetime object.
    Handles various timestamp formats.
    
    Args:
        timestamp_str (str): Timestamp string to parse
    
    Returns:
        datetime: Parsed datetime object
    """
    try:
        # Try various formats
        formats = [
            '%b %d, %Y %H:%M:%S.%f %z',  # Jul 22, 2025 16:53:50.356985000 +08
            '%Y-%m-%d %H:%M:%S.%f',      # 2025-07-22 16:53:50.356985
            '%Y-%m-%d %H:%M:%S',         # 2025-07-22 16:53:50
            '%Y-%m-%dT%H:%M:%S.%fZ',     # 2025-07-22T16:53:50.356Z
            '%Y-%m-%dT%H:%M:%SZ'         # 2025-07-22T16:53:50Z
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        # If none of the formats match, use pandas to parse
        return pd.to_datetime(timestamp_str)
    
    except Exception:
        # Return current time if parsing fails
        return datetime.now()

def create_example_json():
    """
    Create an example JSON file with the expected structure.
    
    Returns:
        str: Path to the created file
    """
    example_data = {
        "_source": {
            "layers": {
                "frame.number": ["1"],
                "frame.time": ["Jul 22, 2025 16:53:50.356985000 +08"],
                "frame.len": ["118"],
                "ip.src": ["192.168.102.101"],
                "ip.dst": ["192.168.3.66"],
                "eth.src": ["40:64:dd:1b:66:44"],
                "eth.dst": ["c8:7f:54:07:cf:b3"],
                "_ws.col.Protocol": ["SSH"],
                "tcp.srcport": ["56197"],
                "tcp.dstport": ["22"],
                "_ws.col.Info": ["Client: Encrypted packet (len=64)"]
            }
        }
    }
    
    # Get data directory from config and create examples subdirectory
    config = load_config()
    data_dir = config.get('system', {}).get('data_dir', 'data')
    examples_dir = f"{data_dir}/examples"
    os.makedirs(examples_dir, exist_ok=True)
    
    # Save example file
    file_path = f"{examples_dir}/example_network.json"
    with open(file_path, 'w') as f:
        json.dump(example_data, f, indent=2)
    
    return file_path

def convert_csv_to_json(csv_file_path):
    """
    Convert a CSV file to the expected JSON format.
    
    Args:
        csv_file_path (str): Path to the CSV file
        
    Returns:
        str: Path to the created JSON file
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_file_path, encoding='utf-8')
        
        # Convert to JSON format
        json_data = []
        
        for _, row in df.iterrows():
            entry = {
                "_source": {
                    "layers": {}
                }
            }
            
            # Map columns to the expected JSON structure
            for col in df.columns:
                # Skip NaN values
                if pd.isna(row[col]):
                    continue
                
                # Normalize column name (for JSON keys)
                key = col.replace(' ', '.').lower()
                
                # Special mappings for common column names
                if 'number' in col.lower():
                    key = 'frame.number'
                elif 'time' in col.lower() or 'date' in col.lower():
                    key = 'frame.time'
                elif 'length' in col.lower() or 'len' in col.lower():
                    key = 'frame.len'
                elif 'source ip' in col.lower() or 'src ip' in col.lower():
                    key = 'ip.src'
                elif 'dest ip' in col.lower() or 'dst ip' in col.lower():
                    key = 'ip.dst'
                elif 'source mac' in col.lower() or 'src mac' in col.lower():
                    key = 'eth.src'
                elif 'dest mac' in col.lower() or 'dst mac' in col.lower():
                    key = 'eth.dst'
                elif 'protocol' in col.lower():
                    key = '_ws.col.Protocol'
                elif 'source port' in col.lower() or 'src port' in col.lower():
                    key = 'tcp.srcport'
                elif 'dest port' in col.lower() or 'dst port' in col.lower():
                    key = 'tcp.dstport'
                elif 'info' in col.lower() or 'description' in col.lower():
                    key = '_ws.col.Info'
                
                # Add to layers
                entry['_source']['layers'][key] = [str(row[col])]
            
            json_data.append(entry)
        
        # Save as JSON
        output_path = os.path.splitext(csv_file_path)[0] + '.json'
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        return output_path
    
    except Exception as e:
        st.error(f"Error converting CSV to JSON: {str(e)}")
        return None