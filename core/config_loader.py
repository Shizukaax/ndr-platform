"""
Configuration loader for the Network Anomaly Detection Platform.
Loads and validates configuration from config files.
"""

import os
import yaml
import json
from pathlib import Path

DEFAULT_CONFIG = {
    "anomaly_detection": {
        "default_threshold": 0.8,
        "max_anomalies": 100,
        "models": {
            "isolation_forest": {
                "enabled": True,
                "contamination": 0.05,
                "n_estimators": 100,
                "max_samples": "auto",
                "random_state": 42
            },
            "local_outlier_factor": {
                "enabled": True,
                "n_neighbors": 20,
                "contamination": 0.05,
                "algorithm": "auto"
            },
            "one_class_svm": {
                "enabled": True,
                "nu": 0.05,
                "kernel": "rbf",
                "gamma": "scale"
            },
            "dbscan": {
                "enabled": True,
                "eps": 0.5,
                "min_samples": 5,
                "algorithm": "auto"
            },
            "knn": {
                "enabled": True,
                "n_neighbors": 5,
                "algorithm": "auto",
                "metric": "minkowski"
            },
            "hdbscan": {
                "enabled": False,
                "min_cluster_size": 5,
                "min_samples": None,
                "alpha": 1.0
            },
            "ensemble": {
                "enabled": True,
                "combination_method": "weighted_average"
            }
        }
    },
    "visualization": {
        "color_theme": "blue",
        "max_points_scatter": 5000,
        "max_nodes_network": 100,
        "default_chart_height": 400
    },
    "reports": {
        "output_dir": "reports",
        "include_charts": True,
        "include_raw_data": False,
        "default_format": "html"
    },
    "mitre": {
        "techniques_file": "config/mitre_attack_data.json",
        "confidence_threshold": 0.6
    },
    "feedback": {
        "storage_dir": "feedback",
        "use_feedback_for_training": True
    },
    "system": {
        "cache_dir": "cache",
        "models_dir": "models",
        "data_dir": "data",
        "log_level": "INFO"
    }
}

def load_config(config_path=None):
    """
    Load configuration from a YAML file, falling back to defaults if needed.
    
    Args:
        config_path (str, optional): Path to the configuration file
    
    Returns:
        dict: Configuration dictionary
    """
    # Use default path if not specified
    if config_path is None:
        config_path = Path("config") / "config.yaml"
    
    # Try to load config file
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Merge with defaults to ensure all required fields exist
            merged_config = DEFAULT_CONFIG.copy()
            
            # Simple recursive dictionary merge function
            def merge_dicts(default_dict, override_dict):
                for key, value in override_dict.items():
                    if key in default_dict and isinstance(default_dict[key], dict) and isinstance(value, dict):
                        merge_dicts(default_dict[key], value)
                    else:
                        default_dict[key] = value
            
            # Merge loaded config with defaults
            merge_dicts(merged_config, config)
            return merged_config
        else:
            # Create default config file if it doesn't exist
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)
            return DEFAULT_CONFIG
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        print(f"Falling back to default configuration")
        return DEFAULT_CONFIG

def save_config(config, config_path=None):
    """
    Save configuration to a YAML file.
    
    Args:
        config (dict): Configuration dictionary to save
        config_path (str, optional): Path to the configuration file
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Use default path if not specified
    if config_path is None:
        config_path = Path("config") / "config.yaml"
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Save config
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        
        return True
    except Exception as e:
        print(f"Error saving configuration: {str(e)}")
        return False

def get_model_config(model_name):
    """
    Get configuration for a specific model.
    
    Args:
        model_name (str): Name of the model
    
    Returns:
        dict: Model configuration
    """
    config = load_config()
    models_config = config.get('anomaly_detection', {}).get('models', {})
    
    if model_name in models_config:
        return models_config[model_name]
    else:
        return {}

def load_mitre_data(file_path=None):
    """
    Load MITRE ATT&CK data from a JSON file.
    
    Args:
        file_path (str, optional): Path to the MITRE data file
    
    Returns:
        dict: MITRE ATT&CK data
    """
    # Use default path if not specified
    if file_path is None:
        config = load_config()
        file_path = config['mitre']['techniques_file']
    
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                mitre_data = json.load(f)
            return mitre_data
        else:
            print(f"MITRE ATT&CK data file not found at {file_path}")
            return {"techniques": [], "tactics": []}
    except Exception as e:
        print(f"Error loading MITRE ATT&CK data: {str(e)}")
        return {"techniques": [], "tactics": []}