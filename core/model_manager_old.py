"""
Model manager for the Network Anomaly Detection Platform.
Manages model persistence, loading, and continuous learning from new data.
Maintains separate models for different algorithm types.
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
from pathlib import Path
import pickle
import tempfile
import shutil
import stat
import time
from typing import List, Dict, Any, Optional, Union
from sklearn.preprocessing import StandardScaler

# Setup logger
logger = logging.getLogger("model_manager")

class ModelManager:
    """
    Manages anomaly detection models including saving, loading, and continuous learning.
    Maintains separate models for different algorithm types (Isolation Forest, LOF, etc.).
    Each algorithm type has its own model file that gets updated with new data.
    """
    
    def __init__(self, models_dir=None, data_dir=None, results_dir=None):
        """
        Initialize the model manager with config-aware paths.
        
        Args:
            models_dir (str, optional): Directory to store models
            data_dir (str, optional): Directory where input data is stored
            results_dir (str, optional): Directory to store analysis results
        """
        # Import config loader to get default paths
        try:
            from core.config_loader import load_config
            config = load_config()
        except Exception as e:
            logger.warning(f"Could not load config: {e}. Using default paths.")
            config = {}
        
        # Use config paths by default, fall back to provided or hardcoded paths
        if models_dir is None:
            models_dir = config.get('system', {}).get('models_dir', 'models')
        
        if data_dir is None:
            data_dir = config.get('system', {}).get('data_dir', 'data')
            
        if results_dir is None:
            results_dir = config.get('system', {}).get('results_dir', 'data/results')
        
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.results_dir = results_dir
        
        # Create required directories if they don't exist
        self._ensure_directory(self.models_dir)
        self._ensure_directory(self.data_dir)
        self._ensure_directory(self.results_dir)
        
        # Initialize cache
        self._model_cache = {}
        self._threshold_cache = {}
        
        logger.info(f"Initialized ModelManager with models directory: {self.models_dir}")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Results directory: {self.results_dir}")
    
    def _ensure_directory(self, directory):
        """
        Ensure a directory exists with proper permissions.
        
        Args:
            directory (str): Directory path to ensure
        """
        try:
            # Create directory if it doesn't exist
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory: {directory}")
            
            # Check if directory is writable
            test_file = os.path.join(directory, f"test_write_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tmp")
            with open(test_file, 'w') as f:
                f.write("Test write")
            
            # Clean up test file
            os.remove(test_file)
            logger.info(f"Verified directory is writable: {directory}")
            
            # Set directory permissions to 777 (read/write/execute for all)
            try:
                os.chmod(directory, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                logger.info(f"Set full permissions on directory: {directory}")
            except Exception as e:
                logger.warning(f"Could not set permissions on directory {directory}: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error ensuring directory {directory}: {str(e)}")
            # Try alternative locations
            alt_dir = os.path.join(os.path.dirname(os.getcwd()), os.path.basename(directory))
            logger.info(f"Trying alternative directory: {alt_dir}")
            
            try:
                os.makedirs(alt_dir, exist_ok=True)
                # If we're creating a tracked directory, update the instance variable
                if directory == self.models_dir:
                    self.models_dir = alt_dir
                elif directory == self.data_dir:
                    self.data_dir = alt_dir
                elif directory == self.results_dir:
                    self.results_dir = alt_dir
                logger.info(f"Created alternative directory: {alt_dir}")
            except Exception as alt_e:
                logger.error(f"Error creating alternative directory {alt_dir}: {str(alt_e)}")
                # Last resort - use temp directory
                temp_dir = os.path.join(tempfile.gettempdir(), os.path.basename(directory))
                logger.info(f"Trying temp directory: {temp_dir}")
                
                try:
                    os.makedirs(temp_dir, exist_ok=True)
                    # Update instance variable
                    if directory == self.models_dir:
                        self.models_dir = temp_dir
                    elif directory == self.data_dir:
                        self.data_dir = temp_dir
                    elif directory == self.results_dir:
                        self.results_dir = temp_dir
                    logger.info(f"Created temp directory: {temp_dir}")
                except Exception as temp_e:
                    logger.error(f"Error creating temp directory {temp_dir}: {str(temp_e)}")
    
    def _get_model_filename(self, model_type: str) -> str:
        """
        Get the filename for a specific model type.
        
        Args:
            model_type: Type of model (e.g., "IsolationForest", "LocalOutlierFactor")
            
        Returns:
            str: Filename for the model
        """
        # Clean model type name - remove spaces and special characters
        model_type = model_type.replace(" ", "").replace("-", "_")
        return f"{model_type}_model.pkl"
    
    def _get_model_path(self, model_type: str) -> str:
        """
        Get the full path for a specific model type.
        
        Args:
            model_type: Type of model (e.g., "IsolationForest", "LocalOutlierFactor")
            
        Returns:
            str: Full path to the model file
        """
        filename = self._get_model_filename(model_type)
        return os.path.join(self.models_dir, filename)
    
    def _get_metadata_path(self, model_type: str) -> str:
        """
        Get the metadata path for a specific model type.
        
        Args:
            model_type: Type of model (e.g., "IsolationForest", "LocalOutlierFactor")
            
        Returns:
            str: Path to the metadata file
        """
        model_path = self._get_model_path(model_type)
        return os.path.splitext(model_path)[0] + "_metadata.json"
    
    def _make_json_serializable(self, obj):
        """
        Convert a potentially non-serializable object to a JSON-serializable form.
        
        Args:
            obj: Object to make serializable
            
        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, dict):
            # Handle dictionaries - recursively process each value
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # Handle lists - recursively process each item
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            # Handle numpy arrays
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            # Handle objects - convert to dictionary of attributes
            if hasattr(obj, '__class__'):
                # For model objects, just save their type and parameters
                if 'Detector' in obj.__class__.__name__:
                    detector_info = {
                        "type": obj.__class__.__name__,
                    }
                    # Add parameters if available
                    if hasattr(obj, 'get_params'):
                        try:
                            detector_info["parameters"] = self._make_json_serializable(obj.get_params())
                        except:
                            detector_info["parameters"] = "Parameters not available"
                    return detector_info
            # For other objects, convert to dict of attributes
            return self._make_json_serializable(obj.__dict__)
        elif isinstance(obj, (int, float, str, bool, type(None))):
            # Basic types are already serializable
            return obj
        else:
            # For other types, convert to string representation
            try:
                return str(obj)
            except:
                return "Non-serializable object"
    
    def save_model(self, model, create_backup: bool = True, threshold: float = None) -> str:
        """
        Save/update a model based on its type.
        
        Args:
            model: Model to save
            create_backup: Whether to backup the existing model before saving
            threshold: Anomaly threshold to save with the model
            
        Returns:
            str: Path to the saved model
        """
        if not model:
            logger.error("No model provided to save")
            raise ValueError("No model provided to save")
        
        # Get model type from the model class name or metadata
        if hasattr(model, 'metadata') and 'type' in model.metadata:
            model_type = model.metadata['type']
        else:
            model_type = model.__class__.__name__.replace("Detector", "")
        
        # Initialize metadata if not present
        if not hasattr(model, 'metadata'):
            logger.warning(f"Model {model_type} does not have metadata, initializing it")
            model.metadata = {
                "type": model_type,
                "trained": True,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "parameters": {}
            }
        
        # Update metadata
        model.metadata["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model.metadata["update_count"] = model.metadata.get("update_count", 0) + 1
        
        # Store feature names in metadata
        if hasattr(model, 'feature_names') and model.feature_names:
            model.metadata["feature_names"] = model.feature_names
            
        # Store threshold in metadata if provided
        if threshold is not None:
            model.metadata["anomaly_threshold"] = float(threshold)
            # Also cache it
            self._threshold_cache[model_type] = threshold
        
        # Get model path based on its type
        model_path = self._get_model_path(model_type)
        metadata_path = self._get_metadata_path(model_type)
        
        # Create backup if requested and model exists
        if create_backup and os.path.exists(model_path):
            try:
                backup_dir = os.path.join(self.models_dir, "backups")
                os.makedirs(backup_dir, exist_ok=True)
                
                backup_filename = f"{model_type}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                backup_path = os.path.join(backup_dir, backup_filename)
                
                shutil.copy2(model_path, backup_path)
                logger.info(f"Created backup of {model_type} model at {backup_path}")
                
                # Also backup metadata
                if os.path.exists(metadata_path):
                    backup_metadata_path = os.path.splitext(backup_path)[0] + "_metadata.json"
                    shutil.copy2(metadata_path, backup_metadata_path)
            except Exception as e:
                logger.error(f"Failed to create backup: {str(e)}")
        
        # Save the model
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Use temp file approach for more reliable saving
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                pickle.dump(model, tmp_file)
                tmp_path = tmp_file.name
            
            # Move temp file to final location
            shutil.move(tmp_path, model_path)
            
            # Make metadata JSON-serializable
            serializable_metadata = self._make_json_serializable(model.metadata)
            
            # Save metadata separately
            with open(metadata_path, "w", encoding='utf-8') as f:
                json.dump(serializable_metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"{model_type} model updated successfully at {model_path}")
            
            # Update cache
            cache_key = f"model_{model_type}"
            self._model_cache[cache_key] = model
            
            return model_path
            
        except Exception as e:
            logger.error(f"Error saving {model_type} model: {str(e)}")
            
            # Try alternative locations if main save fails
            try:
                alt_path = os.path.join(self.models_dir, "fallback", self._get_model_filename(model_type))
                os.makedirs(os.path.dirname(alt_path), exist_ok=True)
                
                with open(alt_path, "wb") as f:
                    pickle.dump(model, f)
                
                # Save metadata (ensuring it's serializable)
                alt_metadata_path = os.path.splitext(alt_path)[0] + "_metadata.json"
                serializable_metadata = self._make_json_serializable(model.metadata)
                
                with open(alt_metadata_path, "w", encoding='utf-8') as f:
                    json.dump(serializable_metadata, f, indent=2, ensure_ascii=False)
                
                logger.info(f"{model_type} model saved to alternative location: {alt_path}")
                return alt_path
                
            except Exception as alt_e:
                logger.error(f"Alternative save also failed: {str(alt_e)}")
                raise IOError(f"Failed to save {model_type} model: {str(e)}. Alternative save also failed: {str(alt_e)}")
    
    def update_model_with_feedback(self, model_type: str, feedback_data) -> str:
        """
        Update a specific model type with feedback.
        
        Args:
            model_type: Type of model to update
            feedback_data: Feedback data to incorporate into the model
            
        Returns:
            str: Path to the updated model
        """
        # Load the model
        model = self.load_model(model_type)
        
        # Add feedback to model metadata
        if not hasattr(model, 'metadata'):
            model.metadata = {}
        
        if 'feedback' not in model.metadata:
            model.metadata['feedback'] = []
        
        # Add new feedback with timestamp
        feedback_entry = {
            "data": feedback_data,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        model.metadata['feedback'].append(feedback_entry)
        
        # Update model parameters if the model supports it
        if hasattr(model, 'update_with_feedback') and callable(getattr(model, 'update_with_feedback')):
            model.update_with_feedback(feedback_data)
            logger.info(f"{model_type} model updated with feedback using model's update_with_feedback method")
        
        # Save updated model - preserve the existing threshold
        threshold = None
        if model_type in self._threshold_cache:
            threshold = self._threshold_cache[model_type]
        elif hasattr(model, 'metadata') and 'anomaly_threshold' in model.metadata:
            threshold = model.metadata['anomaly_threshold']
            
        # Save updated model with the original threshold
        return self.save_model(model, threshold=threshold)
    
    def load_model(self, model_type: str):
        """
        Load a model by its type.
        
        Args:
            model_type: Type of model to load
            
        Returns:
            The loaded model
        
        Raises:
            FileNotFoundError: If model doesn't exist
        """
        # Check cache first
        cache_key = f"model_{model_type}"
        if cache_key in self._model_cache:
            logger.info(f"Using cached {model_type} model")
            return self._model_cache[cache_key]
        
        # Get model path
        model_path = self._get_model_path(model_type)
        
        # Check if model exists
        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                
                # Add to cache
                self._model_cache[cache_key] = model
                
                # Also cache the threshold if present in metadata
                if hasattr(model, 'metadata') and 'anomaly_threshold' in model.metadata:
                    self._threshold_cache[model_type] = model.metadata['anomaly_threshold']
                
                logger.info(f"{model_type} model loaded from {model_path}")
                return model
            except Exception as e:
                logger.error(f"Error loading {model_type} model: {str(e)}")
                
                # Try loading from backups
                backup_dir = os.path.join(self.models_dir, "backups")
                if os.path.exists(backup_dir):
                    # Find the most recent backup
                    backups = [f for f in os.listdir(backup_dir) if f.startswith(f"{model_type}_backup_") and f.endswith(".pkl")]
                    if backups:
                        # Sort by timestamp (newest first)
                        backups.sort(reverse=True)
                        newest_backup = os.path.join(backup_dir, backups[0])
                        
                        try:
                            with open(newest_backup, "rb") as f:
                                model = pickle.load(f)
                            
                            # Also try to load the threshold from backup
                            backup_metadata_path = os.path.splitext(newest_backup)[0] + "_metadata.json"
                            if os.path.exists(backup_metadata_path):
                                try:
                                    with open(backup_metadata_path, "r", encoding='utf-8') as f:
                                        metadata = json.load(f)
                                        if 'anomaly_threshold' in metadata:
                                            self._threshold_cache[model_type] = metadata['anomaly_threshold']
                                except Exception as meta_e:
                                    logger.warning(f"Could not load threshold from backup metadata: {str(meta_e)}")
                            
                            logger.info(f"Loaded {model_type} model from backup: {newest_backup}")
                            return model
                        except Exception as backup_e:
                            logger.error(f"Error loading backup model: {str(backup_e)}")
                
                raise
        else:
            # Try alternative locations
            alt_path = os.path.join(self.models_dir, "fallback", self._get_model_filename(model_type))
            if os.path.exists(alt_path):
                try:
                    with open(alt_path, "rb") as f:
                        model = pickle.load(f)
                    
                    # Also try to load the threshold from alternative metadata
                    alt_metadata_path = os.path.splitext(alt_path)[0] + "_metadata.json"
                    if os.path.exists(alt_metadata_path):
                        try:
                            with open(alt_metadata_path, "r") as f:
                                metadata = json.load(f)
                                if 'anomaly_threshold' in metadata:
                                    self._threshold_cache[model_type] = metadata['anomaly_threshold']
                        except Exception as meta_e:
                            logger.warning(f"Could not load threshold from alternative metadata: {str(meta_e)}")
                    
                    logger.info(f"{model_type} model loaded from alternative location: {alt_path}")
                    return model
                except Exception as alt_e:
                    logger.error(f"Error loading model from alternative location: {str(alt_e)}")
            
            # If no model found, raise error
            raise FileNotFoundError(f"No {model_type} model found. Please train a model first.")
    
    def has_model(self, model_type: str) -> bool:
        """
        Check if a model of the specified type exists.
        
        Args:
            model_type: Type of model to check
            
        Returns:
            bool: True if model exists, False otherwise
        """
        # Get model path
        model_path = self._get_model_path(model_type)
        
        if os.path.exists(model_path):
            return True
        
        # Check alternative location
        alt_path = os.path.join(self.models_dir, "fallback", self._get_model_filename(model_type))
        if os.path.exists(alt_path):
            return True
        
        return False
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_type: Type of model to get info for
            
        Returns:
            Dict with model information
        """
        if not self.has_model(model_type):
            return {"exists": False, "type": model_type}
        
        # Get model path
        model_path = self._get_model_path(model_type)
        metadata_path = self._get_metadata_path(model_type)
        
        # Check alternative path if needed
        if not os.path.exists(model_path):
            model_path = os.path.join(self.models_dir, "fallback", self._get_model_filename(model_type))
            metadata_path = os.path.splitext(model_path)[0] + "_metadata.json"
        
        # Get metadata
        metadata = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding='utf-8') as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata for {model_type}: {str(e)}")
        
        # Get file info
        try:
            file_size = os.path.getsize(model_path)
            modified_time = os.path.getmtime(model_path)
            modified_date = datetime.fromtimestamp(modified_time).strftime("%Y-%m-%d %H:%M:%S")
            
            info = {
                "exists": True,
                "type": model_type,
                "path": model_path,
                "size": file_size,
                "size_mb": file_size / (1024 * 1024),
                "last_modified": modified_date,
                "metadata": metadata
            }
            
            return info
        except Exception as e:
            logger.error(f"Error getting model info for {model_type}: {str(e)}")
            return {"exists": True, "type": model_type, "error": str(e), "metadata": metadata}
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models.
        
        Returns:
            List of dictionaries with model information
        """
        models = []
        
        # Check models directory
        if os.path.exists(self.models_dir):
            for file in os.listdir(self.models_dir):
                if file.endswith('_model.pkl'):
                    try:
                        model_path = os.path.join(self.models_dir, file)
                        model_type = file.replace('_model.pkl', '')
                        
                        # Get metadata
                        metadata_path = self._get_metadata_path(model_type)
                        metadata = {}
                        
                        if os.path.exists(metadata_path):
                            try:
                                with open(metadata_path, "r", encoding='utf-8') as f:
                                    metadata = json.load(f)
                            except Exception as e:
                                logger.warning(f"Could not load metadata for {file}: {str(e)}")
                        
                        # Get file info
                        file_size = os.path.getsize(model_path)
                        modified_time = os.path.getmtime(model_path)
                        modified_date = datetime.fromtimestamp(modified_time).strftime("%Y-%m-%d %H:%M:%S")
                        
                        models.append({
                            "type": model_type,
                            "path": model_path,
                            "size": file_size,
                            "size_mb": file_size / (1024 * 1024),
                            "last_modified": modified_date,
                            "metadata": metadata
                        })
                    except Exception as e:
                        logger.error(f"Error processing model file {file}: {str(e)}")
        
        # Also check fallback directory
        fallback_dir = os.path.join(self.models_dir, "fallback")
        if os.path.exists(fallback_dir):
            for file in os.listdir(fallback_dir):
                if file.endswith('_model.pkl'):
                    try:
                        model_path = os.path.join(fallback_dir, file)
                        model_type = file.replace('_model.pkl', '')
                        
                        # Check if we already added this model type
                        if any(model["type"] == model_type for model in models):
                            continue
                        
                        # Get metadata
                        metadata_path = os.path.splitext(model_path)[0] + "_metadata.json"
                        metadata = {}
                        
                        if os.path.exists(metadata_path):
                            try:
                                with open(metadata_path, "r") as f:
                                    metadata = json.load(f)
                            except Exception as e:
                                logger.warning(f"Could not load metadata for {file}: {str(e)}")
                        
                        # Get file info
                        file_size = os.path.getsize(model_path)
                        modified_time = os.path.getmtime(model_path)
                        modified_date = datetime.fromtimestamp(modified_time).strftime("%Y-%m-%d %H:%M:%S")
                        
                        models.append({
                            "type": model_type,
                            "path": model_path,
                            "size": file_size,
                            "size_mb": file_size / (1024 * 1024),
                            "last_modified": modified_date,
                            "metadata": metadata
                        })
                    except Exception as e:
                        logger.error(f"Error processing fallback model file {file}: {str(e)}")
        
        return models
    
    def delete_model(self, model_type: str) -> bool:
        """
        Delete a model and its metadata.
        
        Args:
            model_type: Type of model to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.has_model(model_type):
            logger.warning(f"Cannot delete {model_type} model - it doesn't exist")
            return False
        
        # Get model paths
        model_path = self._get_model_path(model_type)
        metadata_path = self._get_metadata_path(model_type)
        
        # Check alternative path if needed
        alt_model_path = None
        alt_metadata_path = None
        if not os.path.exists(model_path):
            alt_model_path = os.path.join(self.models_dir, "fallback", self._get_model_filename(model_type))
            alt_metadata_path = os.path.splitext(alt_model_path)[0] + "_metadata.json"
        
        success = True
        
        # Delete main model if it exists
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
                logger.info(f"Deleted {model_type} model at {model_path}")
            except Exception as e:
                logger.error(f"Error deleting {model_type} model: {str(e)}")
                success = False
        
        # Delete metadata if it exists
        if os.path.exists(metadata_path):
            try:
                os.remove(metadata_path)
                logger.info(f"Deleted {model_type} metadata at {metadata_path}")
            except Exception as e:
                logger.error(f"Error deleting {model_type} metadata: {str(e)}")
                success = False
        
        # Delete alternative model if it exists
        if alt_model_path and os.path.exists(alt_model_path):
            try:
                os.remove(alt_model_path)
                logger.info(f"Deleted {model_type} model at alternative location {alt_model_path}")
            except Exception as e:
                logger.error(f"Error deleting {model_type} model from alternative location: {str(e)}")
                success = False
        
        # Delete alternative metadata if it exists
        if alt_metadata_path and os.path.exists(alt_metadata_path):
            try:
                os.remove(alt_metadata_path)
                logger.info(f"Deleted {model_type} metadata at alternative location {alt_metadata_path}")
            except Exception as e:
                logger.error(f"Error deleting {model_type} metadata from alternative location: {str(e)}")
                success = False
        
        # Clear cache entries
        cache_key = f"model_{model_type}"
        if cache_key in self._model_cache:
            del self._model_cache[cache_key]
            
        if model_type in self._threshold_cache:
            del self._threshold_cache[model_type]
        
        return success

    def apply_model_to_data(self, model_type: str, data: pd.DataFrame, 
                           feature_names: List[str], recalculate_threshold: bool = False,
                           custom_contamination: float = None,
                           save_results: bool = True) -> Dict[str, Any]:
        """
        Apply a specific model to detect anomalies in data.
        
        Args:
            model_type: Type of model to apply
            data: Data to apply the model to
            feature_names: Features to use from the data
            recalculate_threshold: Whether to recalculate the anomaly threshold
            custom_contamination: Custom contamination value to use if recalculating threshold
            save_results: Whether to save results to disk
            
        Returns:
            Dict with detection results including scores, threshold, and anomalies
        """
        if not self.has_model(model_type):
            raise FileNotFoundError(f"No {model_type} model found. Please train a model first.")
        
        start_time = time.time()
        
        # Load the model
        model = self.load_model(model_type)
        
        # Validate feature names
        model_features = []
        if hasattr(model, 'feature_names') and model.feature_names:
            model_features = model.feature_names
        elif hasattr(model, 'metadata') and 'feature_names' in model.metadata:
            model_features = model.metadata['feature_names']
        
        # Find matching features between model and data
        if model_features:
            # Filter the provided feature names to only include those in the model
            matching_features = [f for f in feature_names if f in model_features]
            
            # If no matching features, try to use all model features
            if not matching_features:
                # Check which model features are in the data
                matching_features = [f for f in model_features if f in data.columns]
                
                if not matching_features:
                    raise ValueError(f"No matching features found between model and data. Model features: {model_features}")
                
                logger.warning(f"Using model features instead of provided features. Matching features: {matching_features}")
        else:
            # If model doesn't have feature names, use the provided ones
            matching_features = feature_names
        
        # Extract features from data
        try:
            X = data[matching_features].copy()
        except KeyError as e:
            # If we can't extract features, list what's available
            available_features = [col for col in data.columns if col in matching_features]
            missing_features = [col for col in matching_features if col not in data.columns]
            
            logger.error(f"Missing features: {missing_features}")
            logger.error(f"Available features: {available_features}")
            
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Handle missing values - fill with 0 for simplicity
        # In a real-world scenario, this should be handled more carefully
        X = X.fillna(0)
        
        # Generate anomaly scores
        scores = model.predict(X)
        
        # Determine the threshold
        used_saved_threshold = False
        
        if recalculate_threshold:
            # Recalculate threshold based on contamination
            contamination = custom_contamination if custom_contamination is not None else 0.01
            
            # Different models may have different threshold interpretations
            if model_type == "OneClassSVM" or model_type == "One_ClassSVM":
                # For One-Class SVM, the threshold is typically 0
                # Scores < 0 are anomalies
                threshold = 0
            else:
                # For most other models, use percentile based on contamination
                # Higher scores are anomalies
                threshold = np.percentile(scores, 100 * (1 - contamination))
        else:
            # Use the saved threshold if available
            if model_type in self._threshold_cache:
                threshold = self._threshold_cache[model_type]
                used_saved_threshold = True
            elif hasattr(model, 'metadata') and 'anomaly_threshold' in model.metadata:
                threshold = model.metadata['anomaly_threshold']
                used_saved_threshold = True
            else:
                # Default to using 95th percentile if no threshold is saved
                threshold = np.percentile(scores, 95)
                logger.warning(f"No saved threshold found for {model_type}. Using 95th percentile: {threshold}")
        
        # Identify anomalies
        if model_type == "OneClassSVM" or model_type == "One_ClassSVM":
            # For One-Class SVM, lower scores (below threshold) are anomalies
            anomaly_indices = np.where(scores < threshold)[0]
            # Invert scores for consistency with other models (higher = more anomalous)
            display_scores = -scores
        else:
            # For most models, higher scores (above threshold) are anomalies
            anomaly_indices = np.where(scores > threshold)[0]
            display_scores = scores
        
        # Create anomalies dataframe
        anomalies = data.iloc[anomaly_indices].copy() if len(anomaly_indices) > 0 else pd.DataFrame()
        if len(anomalies) > 0:
            anomalies['anomaly_score'] = display_scores[anomaly_indices]
        
        # Calculate processing time
        end_time = time.time()
        analysis_time = end_time - start_time
        
        # Compile results
        results = {
            "model": model,
            "model_type": model_type,
            "scores": scores,
            "threshold": threshold,
            "anomalies": anomalies,
            "features": matching_features,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "analysis_time": analysis_time,
            "used_saved_threshold": used_saved_threshold,
            "matching_features": matching_features
        }
        
        # Save results if requested
        if save_results:
            try:
                # Create results directory if it doesn't exist
                results_dir = os.path.join(self.results_dir, model_type)
                os.makedirs(results_dir, exist_ok=True)
                
                # Save anomalies to CSV
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                anomalies_path = os.path.join(results_dir, f"anomalies_{timestamp}.csv")
                
                if len(anomalies) > 0:
                    anomalies.to_csv(anomalies_path, index=False, encoding='utf-8')
                    logger.info(f"Saved {len(anomalies)} anomalies to {anomalies_path}")
                
                # Save summary as JSON
                summary = {
                    "model_type": model_type,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "record_count": len(data),
                    "anomaly_count": len(anomalies),
                    "anomaly_percent": (len(anomalies) / len(data)) * 100 if len(data) > 0 else 0,
                    "threshold": threshold,
                    "threshold_source": "saved" if used_saved_threshold else "calculated",
                    "analysis_time": analysis_time,
                    "features": matching_features
                }
                
                summary_path = os.path.join(results_dir, f"summary_{timestamp}.json")
                with open(summary_path, "w", encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                    
                logger.info(f"Saved analysis summary to {summary_path}")
                
                # Add paths to results
                results["anomalies_path"] = anomalies_path
                results["summary_path"] = summary_path
                
            except Exception as e:
                logger.error(f"Error saving results: {str(e)}")
        
        return results

    def fix_model_thresholds(self, model_type: str = None) -> bool:
        """
        Fix problematic thresholds for models (like extremely small values).
        
        Args:
            model_type: Specific model type to fix or None for all models
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # If model_type is None, fix all models
            if model_type is None:
                models_to_fix = self.list_models()
                model_types = [model["type"] for model in models_to_fix]
            else:
                model_types = [model_type]
            
            for m_type in model_types:
                # Load the model
                model = self.load_model(m_type)
                
                # Get current threshold
                current_threshold = None
                if hasattr(model, 'metadata') and 'anomaly_threshold' in model.metadata:
                    current_threshold = model.metadata['anomaly_threshold']
                
                # Check if threshold is problematic
                is_problematic = (
                    current_threshold is not None and 
                    (current_threshold < 0.0001 or current_threshold > 100)  # Too small or too large
                )
                
                if is_problematic:
                    # Determine a better threshold based on model type
                    if m_type == "IsolationForest":
                        # Isolation Forest typically has scores from 0 to ~1
                        new_threshold = 0.2  # A reasonable default threshold
                    elif m_type == "LocalOutlierFactor":
                        # LOF can have very large values
                        if current_threshold > 10:
                            # Already a large threshold, probably OK
                            new_threshold = current_threshold
                        else:
                            # Too small, set a reasonable default
                            new_threshold = 1.5
                    elif m_type == "OneClassSVM" or m_type == "One_ClassSVM":
                        # OneClassSVM typically uses 0 as threshold (scores below 0 are anomalies)
                        # But our platform expects higher scores to be anomalies, so we use positive threshold
                        new_threshold = 0.1
                    else:
                        # Generic fallback - use a reasonable value
                        new_threshold = 0.5
                    
                    # Update model metadata
                    model.metadata["anomaly_threshold"] = new_threshold
                    
                    # Save the model with updated threshold
                    self.save_model(model, create_backup=True, threshold=new_threshold)
                    
                    logger.info(f"Fixed threshold for {m_type} model: {current_threshold} -> {new_threshold}")
                
            return True
        except Exception as e:
            logger.error(f"Error fixing model thresholds: {str(e)}")
            return False