"""
Model card visualization utilities for the Network Anomaly Detection Platform.
Provides functions to generate visual representations of models and their metadata.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import logging

logger = logging.getLogger("streamlit_app")

def generate_model_card(model_info: Dict[str, Any]) -> str:
    """
    Generate an HTML card for a model to display in the UI.
    
    Args:
        model_info: Dictionary containing model metadata
        
    Returns:
        HTML string for the model card
    """
    model_type = model_info.get("type", "Unknown")
    path = model_info.get("path", "Unknown")
    last_modified = model_info.get("last_modified", "Unknown")
    size_mb = model_info.get("size_mb", 0)
    
    # Extract metadata
    metadata = model_info.get("metadata", {})
    feature_count = len(metadata.get("feature_names", []))
    trained_at = metadata.get("trained_at", "Unknown")
    threshold = metadata.get("anomaly_threshold", "Auto")
    
    # Extract algorithm-specific info for the card
    algo_info = ""
    parameters = metadata.get("parameters", {})
    if isinstance(parameters, dict):
        if model_type == "IsolationForest":
            n_estimators = parameters.get("n_estimators", "Unknown")
            contamination = parameters.get("contamination", "Unknown")
            algo_info = f"<p><strong>Estimators:</strong> {n_estimators}</p>"
            algo_info += f"<p><strong>Contamination:</strong> {contamination}</p>"
        elif model_type == "LocalOutlierFactor":
            n_neighbors = parameters.get("n_neighbors", "Unknown")
            contamination = parameters.get("contamination", "Unknown")
            algo_info = f"<p><strong>Neighbors:</strong> {n_neighbors}</p>"
            algo_info += f"<p><strong>Contamination:</strong> {contamination}</p>"
        elif model_type == "OneClassSVM":
            nu = parameters.get("nu", "Unknown")
            kernel = parameters.get("kernel", "Unknown")
            algo_info = f"<p><strong>Nu:</strong> {nu}</p>"
            algo_info += f"<p><strong>Kernel:</strong> {kernel}</p>"
    
    # Create card
    card_html = f"""
    <div style="border:1px solid #ddd; border-radius:5px; padding:15px; margin-bottom:15px;">
        <h3 style="color:#4B56D2;">{model_type}</h3>
        <p><strong>Features:</strong> {feature_count}</p>
        <p><strong>Threshold:</strong> {threshold}</p>
        {algo_info}
        <p><strong>Trained:</strong> {trained_at}</p>
        <p><strong>Last Modified:</strong> {last_modified}</p>
        <p><strong>Size:</strong> {size_mb:.2f} MB</p>
        <p style="font-size:0.8em; color:#888;">{path}</p>
    </div>
    """
    return card_html

def plot_to_base64(fig):
    """
    Convert a matplotlib figure to base64 encoded string for display in HTML.
    
    Args:
        fig: Matplotlib figure object
        
    Returns:
        Base64 encoded string of the image
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

def create_model_comparison_table(models: List[Dict[str, Any]]) -> str:
    """
    Generate an HTML table for comparing multiple models.
    
    Args:
        models: List of model information dictionaries
        
    Returns:
        HTML string with comparison table
    """
    if not models:
        return "No models available for comparison"
    
    table_html = """
    <table style="width:100%; border-collapse:collapse;">
        <tr style="background-color:#f2f2f2;">
            <th style="padding:8px; text-align:left; border:1px solid #ddd;">Model Type</th>
            <th style="padding:8px; text-align:center; border:1px solid #ddd;">Features</th>
            <th style="padding:8px; text-align:center; border:1px solid #ddd;">Threshold</th>
            <th style="padding:8px; text-align:center; border:1px solid #ddd;">Algorithm</th>
            <th style="padding:8px; text-align:center; border:1px solid #ddd;">Last Updated</th>
        </tr>
    """
    
    for model in models:
        model_type = model.get("type", "Unknown")
        metadata = model.get("metadata", {})
        feature_count = len(metadata.get("feature_names", []))
        threshold = metadata.get("anomaly_threshold", "Auto")
        last_modified = model.get("last_modified", "Unknown")
        
        # Get algorithm-specific info
        algo_info = "N/A"
        parameters = metadata.get("parameters", {})
        if isinstance(parameters, dict):
            if model_type == "IsolationForest":
                algo_info = f"IF (n={parameters.get('n_estimators', 'N/A')})"
            elif model_type == "LocalOutlierFactor":
                algo_info = f"LOF (k={parameters.get('n_neighbors', 'N/A')})"
            elif model_type == "OneClassSVM":
                algo_info = f"SVM ({parameters.get('kernel', 'N/A')})"
            elif model_type == "DBSCAN":
                algo_info = f"DBSCAN (eps={parameters.get('eps', 'N/A')})"
            elif model_type == "KNN":
                algo_info = f"KNN (k={parameters.get('n_neighbors', 'N/A')})"
        
        table_html += f"""
        <tr>
            <td style="padding:8px; text-align:left; border:1px solid #ddd;">{model_type}</td>
            <td style="padding:8px; text-align:center; border:1px solid #ddd;">{feature_count}</td>
            <td style="padding:8px; text-align:center; border:1px solid #ddd;">{threshold}</td>
            <td style="padding:8px; text-align:center; border:1px solid #ddd;">{algo_info}</td>
            <td style="padding:8px; text-align:center; border:1px solid #ddd;">{last_modified}</td>
        </tr>
        """
    
    table_html += "</table>"
    return table_html

def extract_parameter_data(models: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Extract parameters from models for comparison.
    
    Args:
        models: List of model information dictionaries
        
    Returns:
        DataFrame with model parameters for comparison
    """
    param_data = []
    
    for model in models:
        # Handle both string and dict models
        if isinstance(model, str):
            model_type = model
            metadata = {}
            parameters = {}
        else:
            model_type = model.get("type", "Unknown")
            metadata = model.get("metadata", {})
            parameters = metadata.get("parameters", {})
        
        # Extract common parameters
        if isinstance(parameters, dict):
            if model_type == "IsolationForest":
                param_data.append({
                    "Model": model_type,
                    "Algorithm": "Isolation Forest",
                    "Contamination": parameters.get("contamination", "N/A"),
                    "Estimators": parameters.get("n_estimators", "N/A"),
                    "Threshold": metadata.get("anomaly_threshold", "Auto")
                })
            elif model_type == "LocalOutlierFactor":
                param_data.append({
                    "Model": model_type,
                    "Algorithm": "Local Outlier Factor",
                    "Contamination": parameters.get("contamination", "N/A"),
                    "Neighbors": parameters.get("n_neighbors", "N/A"),
                    "Threshold": metadata.get("anomaly_threshold", "Auto")
                })
            elif model_type == "OneClassSVM":
                param_data.append({
                    "Model": model_type,
                    "Algorithm": "One-Class SVM",
                    "Nu": parameters.get("nu", "N/A"),
                    "Kernel": parameters.get("kernel", "N/A"),
                    "Threshold": metadata.get("anomaly_threshold", "Auto")
                })
            elif model_type == "DBSCAN":
                param_data.append({
                    "Model": model_type,
                    "Algorithm": "DBSCAN",
                    "Eps": parameters.get("eps", "N/A"),
                    "Min Samples": parameters.get("min_samples", "N/A"),
                    "Threshold": metadata.get("anomaly_threshold", "Auto")
                })
            elif model_type == "KNN":
                param_data.append({
                    "Model": model_type,
                    "Algorithm": "K-Nearest Neighbors",
                    "Contamination": parameters.get("contamination", "N/A"),
                    "Neighbors": parameters.get("n_neighbors", "N/A"),
                    "Threshold": metadata.get("anomaly_threshold", "Auto")
                })
            elif model_type == "Ensemble":
                param_data.append({
                    "Model": model_type,
                    "Algorithm": "Ensemble",
                    "Contamination": parameters.get("contamination", "N/A"),
                    "Method": parameters.get("combination_method", "N/A"),
                    "Threshold": metadata.get("anomaly_threshold", "Auto")
                })
            else:
                # Generic entry for other model types
                param_data.append({
                    "Model": model_type,
                    "Algorithm": model_type,
                    "Contamination": parameters.get("contamination", "N/A"),
                    "Threshold": metadata.get("anomaly_threshold", "Auto")
                })
    
    return pd.DataFrame(param_data)