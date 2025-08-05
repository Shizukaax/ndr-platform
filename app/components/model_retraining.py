"""
Model retraining utilities for the Network Anomaly Detection Platform.
Provides functions to retrain models with new parameters or additional data.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import tempfile
import os
import pickle

from core.models.isolation_forest import IsolationForestDetector
from core.models.local_outlier_factor import LocalOutlierFactorDetector
from core.models.one_class_svm import OneClassSVMDetector
from core.models.dbscan import DBSCANDetector
from core.models.knn import KNNDetector
from core.models.ensemble import EnsembleDetector

try:
    from core.models.hdbscan import HDBSCANDetector
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

from core.model_manager import ModelManager

logger = logging.getLogger("streamlit_app")

def get_default_parameters(model_type: str, current_parameters: Dict = None) -> Dict[str, Any]:
    """
    Get default parameters for a model type, or use current parameters if available.
    
    Args:
        model_type: Type of model
        current_parameters: Current parameters if available
        
    Returns:
        Dictionary of default parameters
    """
    if current_parameters is None:
        current_parameters = {}
    
    defaults = {}
    
    if model_type == "IsolationForest":
        defaults = {
            "contamination": current_parameters.get("contamination", 0.01),
            "n_estimators": current_parameters.get("n_estimators", 100),
            "random_state": 42
        }
    elif model_type == "LocalOutlierFactor":
        defaults = {
            "contamination": current_parameters.get("contamination", 0.01),
            "n_neighbors": current_parameters.get("n_neighbors", 20),
            "algorithm": "auto"
        }
    elif model_type == "OneClassSVM":
        defaults = {
            "nu": current_parameters.get("nu", 0.01),
            "kernel": current_parameters.get("kernel", "rbf"),
            "gamma": "scale"
        }
    elif model_type == "DBSCAN":
        defaults = {
            "eps": current_parameters.get("eps", 0.5),
            "min_samples": current_parameters.get("min_samples", 5),
            "algorithm": "auto",
            "contamination": current_parameters.get("contamination", 0.01)
        }
    elif model_type == "KNN":
        defaults = {
            "n_neighbors": current_parameters.get("n_neighbors", 5),
            "contamination": current_parameters.get("contamination", 0.01),
            "metric": current_parameters.get("metric", "minkowski"),
            "algorithm": "auto"
        }
    elif model_type == "HDBSCAN" and HDBSCAN_AVAILABLE:
        defaults = {
            "min_cluster_size": current_parameters.get("min_cluster_size", 5),
            "min_samples": current_parameters.get("min_samples", 5),
            "alpha": current_parameters.get("alpha", 1.0),
            "contamination": current_parameters.get("contamination", 0.01)
        }
    else:
        # Generic defaults
        defaults = {
            "contamination": current_parameters.get("contamination", 0.01)
        }
    
    return defaults

def create_parameter_inputs(model_type: str, current_parameters: Dict = None) -> Dict[str, Any]:
    """
    Create Streamlit inputs for model parameters and return user selections.
    
    Args:
        model_type: Type of model
        current_parameters: Current parameters if available
        
    Returns:
        Dictionary of selected parameters
    """
    if current_parameters is None:
        current_parameters = {}
    
    new_params = {}
    
    # Common parameter for most models
    if model_type != "OneClassSVM":  # OneClassSVM uses nu instead
        new_params["contamination"] = st.slider(
            "Contamination (expected proportion of outliers)",
            min_value=0.001,
            max_value=0.2,
            value=float(current_parameters.get("contamination", 0.01)),
            step=0.001,
            format="%.3f"
        )
    
    # Model-specific parameters
    if model_type == "IsolationForest":
        new_params["n_estimators"] = st.slider(
            "Number of estimators",
            min_value=50,
            max_value=500,
            value=int(current_parameters.get("n_estimators", 100)),
            step=10
        )
        new_params["random_state"] = 42
        
    elif model_type == "LocalOutlierFactor":
        new_params["n_neighbors"] = st.slider(
            "Number of neighbors",
            min_value=5,
            max_value=50,
            value=int(current_parameters.get("n_neighbors", 20)),
            step=1
        )
        new_params["algorithm"] = "auto"
        
    elif model_type == "OneClassSVM":
        new_params["nu"] = st.slider(
            "Nu parameter",
            min_value=0.001,
            max_value=0.2,
            value=float(current_parameters.get("nu", 0.01)),
            step=0.001,
            format="%.3f"
        )
        new_params["kernel"] = st.selectbox(
            "Kernel",
            options=["rbf", "linear", "poly", "sigmoid"],
            index=["rbf", "linear", "poly", "sigmoid"].index(current_parameters.get("kernel", "rbf"))
            if current_parameters.get("kernel") in ["rbf", "linear", "poly", "sigmoid"] else 0
        )
        
    elif model_type == "DBSCAN":
        new_params["eps"] = st.slider(
            "Epsilon",
            min_value=0.1,
            max_value=5.0,
            value=float(current_parameters.get("eps", 0.5)),
            step=0.1
        )
        new_params["min_samples"] = st.slider(
            "Minimum samples",
            min_value=2,
            max_value=20,
            value=int(current_parameters.get("min_samples", 5)),
            step=1
        )
        
    elif model_type == "KNN":
        new_params["n_neighbors"] = st.slider(
            "Number of neighbors",
            min_value=1,
            max_value=50,
            value=int(current_parameters.get("n_neighbors", 5)),
            step=1
        )
        new_params["metric"] = st.selectbox(
            "Distance metric",
            options=["minkowski", "euclidean", "manhattan", "chebyshev"],
            index=["minkowski", "euclidean", "manhattan", "chebyshev"].index(current_parameters.get("metric", "minkowski"))
            if current_parameters.get("metric") in ["minkowski", "euclidean", "manhattan", "chebyshev"] else 0
        )
    
    return new_params

def create_new_model(model_type: str, parameters: Dict[str, Any]) -> Any:
    """
    Create a new model instance with the specified parameters.
    
    Args:
        model_type: Type of model to create
        parameters: Parameters for the model
        
    Returns:
        New model instance
    """
    if model_type == "IsolationForest":
        return IsolationForestDetector(**parameters)
    elif model_type == "LocalOutlierFactor":
        return LocalOutlierFactorDetector(**parameters)
    elif model_type == "OneClassSVM":
        return OneClassSVMDetector(**parameters)
    elif model_type == "DBSCAN":
        return DBSCANDetector(**parameters)
    elif model_type == "KNN":
        return KNNDetector(**parameters)
    elif model_type == "HDBSCAN" and HDBSCAN_AVAILABLE:
        return HDBSCANDetector(**parameters)
    elif model_type == "Ensemble":
        # Create component models based on selection
        if "models" in parameters and isinstance(parameters["models"], list) and all(isinstance(m, str) for m in parameters["models"]):
            component_models = []
            contamination = parameters.get("contamination", 0.01)
            
            if "Isolation Forest" in parameters["models"]:
                component_models.append(IsolationForestDetector(contamination=contamination))
            if "Local Outlier Factor" in parameters["models"]:
                component_models.append(LocalOutlierFactorDetector(contamination=contamination))
            if "One-Class SVM" in parameters["models"]:
                component_models.append(OneClassSVMDetector(nu=contamination))
            if "KNN" in parameters["models"]:
                component_models.append(KNNDetector(contamination=contamination))
            if "DBSCAN" in parameters["models"]:
                component_models.append(DBSCANDetector(contamination=contamination))
                
            parameters["models"] = component_models
        
        return EnsembleDetector(**parameters)
    else:
        # Default to Isolation Forest if model type is unknown
        logger.warning(f"Unknown model type: {model_type}. Defaulting to Isolation Forest.")
        return IsolationForestDetector(contamination=parameters.get("contamination", 0.01))

def retrain_model(model_type: str, 
                 data: pd.DataFrame, 
                 feature_names: List[str], 
                 parameters: Dict[str, Any]) -> Tuple[Any, float, pd.DataFrame]:
    """
    Train a new model with the specified parameters and data.
    
    Args:
        model_type: Type of model to train
        data: Training data
        feature_names: Features to use for training
        parameters: Parameters for the model
        
    Returns:
        Tuple of (trained model, threshold, anomalies DataFrame)
    """
    # Create a new model with the specified parameters
    model = create_new_model(model_type, parameters)
    
    # Set feature names
    model.feature_names = feature_names
    
    # Extract features from data
    X = data[feature_names].copy()
    
    # Train the model
    model.fit(X)
    
    # Generate predictions
    scores = model.predict(X)
    
    # Calculate threshold based on contamination
    contamination = parameters.get("contamination", 0.01)
    if model_type == "OneClassSVM":
        contamination = parameters.get("nu", 0.01)
    
    threshold = np.percentile(scores, 100 * (1 - contamination))
    
    # Identify anomalies
    anomaly_indices = np.where(scores > threshold)[0]
    anomalies = data.iloc[anomaly_indices].copy() if len(anomaly_indices) > 0 else pd.DataFrame()
    if len(anomalies) > 0:
        anomalies['anomaly_score'] = scores[anomaly_indices]
    
    return model, threshold, anomalies

def preprocess_data_for_training(data: pd.DataFrame, 
                                feature_names: List[str], 
                                handle_missing: str = "Fill missing values with mean") -> pd.DataFrame:
    """
    Preprocess data for model training.
    
    Args:
        data: Input dataframe
        feature_names: Features to use
        handle_missing: Strategy for handling missing values
        
    Returns:
        Preprocessed dataframe
    """
    # Create a copy to avoid modifying the original
    X = data.copy()
    
    # Extract only the required features
    features_df = X[feature_names].copy()
    
    # Handle missing values
    if handle_missing == "Drop rows with missing values":
        features_df = features_df.dropna()
        # Update the original dataframe to match
        X = X.loc[features_df.index]
    elif handle_missing == "Fill missing values with mean":
        features_df = features_df.fillna(features_df.mean())
        # Fill any remaining NaNs with 0 (for columns that are all NaN)
        features_df = features_df.fillna(0)
    elif handle_missing == "Fill missing values with median":
        features_df = features_df.fillna(features_df.median())
        # Fill any remaining NaNs with 0
        features_df = features_df.fillna(0)
    elif handle_missing == "Fill missing values with 0":
        features_df = features_df.fillna(0)
    
    # Update the dataframe with the processed features
    for feature in feature_names:
        X[feature] = features_df[feature]
    
    return X