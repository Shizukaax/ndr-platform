"""
Anomaly Detection page for the Network Anomaly Detection Platform.
Allows users to detect and analyze network anomalies using different algorithms.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import json
import logging
import time
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import plotly.graph_objects as go

from core.model_manager import ModelManager
from core.models.isolation_forest import IsolationForestDetector
from core.auto_analysis import auto_analysis_service
from core.notification_service import notification_service
from core.session_manager import session_manager
from core.models.local_outlier_factor import LocalOutlierFactorDetector
from core.models.one_class_svm import OneClassSVMDetector
from core.models.knn import KNNDetector
from core.models.hdbscan_detector import HDBSCANDetector
from core.models.ensemble import EnsembleDetector
from app.components.error_handler import handle_error, validate_inputs
from app.components.data_source_selector import ensure_data_available, show_compact_data_status
from app.components.visualization import (
    plot_anomaly_scores, plot_anomaly_scatter, plot_anomaly_timeline,
    plot_feature_importance, plot_network_graph
)
# Import the explainer factory to use SHAP and LIME
from core.explainers.explainer_factory import get_explainer

# Setup logger
logger = logging.getLogger("anomaly_detection")

def get_default_features(df: pd.DataFrame) -> List[str]:
    """
    Get all available numeric features for anomaly detection.
    Users can deselect any they don't want to use.
    
    Args:
        df: DataFrame to extract features from
        
    Returns:
        List of all available numeric feature names
    """
    # Get all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude columns that are definitely not useful for anomaly detection
    exclude_cols = [
        'index', 'id', '_id', 'row_id', 'frame.number', 'frame_number',
        'timestamp', 'time', 'epoch', 'unix_time'  # Time columns should be handled separately
    ]
    
    # Filter out excluded columns (case-insensitive)
    available_features = []
    for col in numeric_cols:
        col_lower = col.lower()
        if not any(exclude.lower() in col_lower for exclude in exclude_cols):
            available_features.append(col)
    
    # Sort features for consistent ordering
    available_features.sort()
    
    # Log what we found for debugging
    logger.info(f"Found {len(available_features)} numeric features: {available_features}")
    
    return available_features

@handle_error
def prepare_protocol_agnostic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare protocol-agnostic features from packet data.
    This adds derived features that work across different protocols.
    
    Args:
        df: DataFrame with packet data
        
    Returns:
        DataFrame with additional protocol-agnostic features
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Process source and destination ports
    if 'src_port' not in result_df.columns:
        # Try to extract from tcp.srcport or udp.srcport
        if 'tcp.srcport' in result_df.columns:
            result_df['src_port'] = pd.to_numeric(result_df['tcp.srcport'], errors='coerce')
        elif 'udp.srcport' in result_df.columns:
            result_df['src_port'] = pd.to_numeric(result_df['udp.srcport'], errors='coerce')
        elif 'tcp_srcport' in result_df.columns:
            result_df['src_port'] = pd.to_numeric(result_df['tcp_srcport'], errors='coerce')
        elif 'udp_srcport' in result_df.columns:
            result_df['src_port'] = pd.to_numeric(result_df['udp_srcport'], errors='coerce')
    
    if 'dst_port' not in result_df.columns:
        # Try to extract from tcp.dstport or udp.dstport
        if 'tcp.dstport' in result_df.columns:
            result_df['dst_port'] = pd.to_numeric(result_df['tcp.dstport'], errors='coerce')
        elif 'udp.dstport' in result_df.columns:
            result_df['dst_port'] = pd.to_numeric(result_df['udp.dstport'], errors='coerce')
        elif 'tcp_dstport' in result_df.columns:
            result_df['dst_port'] = pd.to_numeric(result_df['tcp_dstport'], errors='coerce')
        elif 'udp_dstport' in result_df.columns:
            result_df['dst_port'] = pd.to_numeric(result_df['udp_dstport'], errors='coerce')
    
    # Process packet length
    if 'packet_length' not in result_df.columns:
        if 'frame.len' in result_df.columns:
            result_df['packet_length'] = pd.to_numeric(result_df['frame.len'], errors='coerce')
        elif 'frame_len' in result_df.columns:
            result_df['packet_length'] = pd.to_numeric(result_df['frame_len'], errors='coerce')
        elif 'length' in result_df.columns:
            result_df['packet_length'] = pd.to_numeric(result_df['length'], errors='coerce')
    
    # Extract last octet from IP addresses if not already present
    if 'src_ip_last_octet' not in result_df.columns and ('ip.src' in result_df.columns or 'ip_src' in result_df.columns):
        ip_col = 'ip.src' if 'ip.src' in result_df.columns else 'ip_src'
        # Extract the last octet using string operations
        result_df['src_ip_last_octet'] = result_df[ip_col].astype(str).str.split('.').str[-1].astype(float)
    
    if 'dst_ip_last_octet' not in result_df.columns and ('ip.dst' in result_df.columns or 'ip_dst' in result_df.columns):
        ip_col = 'ip.dst' if 'ip.dst' in result_df.columns else 'ip_dst'
        # Extract the last octet using string operations
        result_df['dst_ip_last_octet'] = result_df[ip_col].astype(str).str.split('.').str[-1].astype(float)
    
    # Process timestamps to calculate time deltas if not already present
    if 'time_delta_ms' not in result_df.columns:
        # Find timestamp column
        timestamp_cols = ['frame.time', 'frame_time', 'timestamp', 'time']
        timestamp_col = None
        
        for col in timestamp_cols:
            if col in result_df.columns:
                timestamp_col = col
                break
        
        if timestamp_col:
            # Convert to datetime
            try:
                # First try a specific format from your examples
                if result_df[timestamp_col].dtype == 'object':
                    sample = result_df[timestamp_col].iloc[0] if not result_df[timestamp_col].empty else ""
                    if 'Jul' in str(sample) or 'Aug' in str(sample) or 'Sep' in str(sample):
                        result_df['timestamp'] = pd.to_datetime(
                            result_df[timestamp_col].astype(str).str.split('+').str[0].str.strip(), 
                            format='%b %d, %Y %H:%M:%S.%f', 
                            errors='coerce'
                        )
                    else:
                        result_df['timestamp'] = pd.to_datetime(result_df[timestamp_col], errors='coerce')
                else:
                    result_df['timestamp'] = pd.to_datetime(result_df[timestamp_col], errors='coerce')
                
                # Sort by timestamp
                result_df = result_df.sort_values('timestamp')
                
                # Calculate time deltas in milliseconds
                result_df['time_delta_ms'] = result_df['timestamp'].diff().dt.total_seconds() * 1000
                
                # Fill NA values in the first row
                result_df['time_delta_ms'] = result_df['time_delta_ms'].fillna(0)
                
            except Exception as e:
                st.warning(f"Could not process timestamps: {str(e)}")
    
    # Fill any remaining NaN values with 0
    for col in ['src_port', 'dst_port', 'packet_length', 'time_delta_ms', 'src_ip_last_octet', 'dst_ip_last_octet']:
        if col in result_df.columns:
            result_df[col] = result_df[col].fillna(0)
    
    return result_df

def display_confidence_level(anomaly_score: float, threshold: float):
    """
    Display a confidence level gauge for anomaly detection.
    Handles both high-is-anomaly and low-is-anomaly algorithms correctly.
    
    Args:
        anomaly_score: The anomaly score value
        threshold: The anomaly detection threshold
    """
    # Check what algorithm is being used to determine scoring logic
    algorithm = st.session_state.get('selected_model', 'Unknown')
    
    # For Isolation Forest: Lower scores = higher anomaly confidence
    # For most others: Higher scores = higher anomaly confidence
    isolation_forest_like = algorithm in ['Isolation Forest', 'IsolationForest']
    
    if isolation_forest_like:
        # For Isolation Forest: score < threshold = anomaly (lower is more anomalous)
        is_anomaly = anomaly_score < threshold
        if is_anomaly:
            # Calculate how far below threshold (more negative = more anomalous)
            distance_ratio = (threshold - anomaly_score) / max(abs(threshold), 0.001)
            
            # Determine confidence level based on how far below threshold
            if distance_ratio >= 2.0:  # Score much lower than threshold
                confidence_level = "Confirmed Threat"
                confidence_percent = min(85 + distance_ratio * 5, 99.9)
                color = "red"
            elif distance_ratio >= 1.0:  # Score significantly lower than threshold
                confidence_level = "Likely Threat"  
                confidence_percent = min(70 + distance_ratio * 10, 95.0)
                color = "orange"
            elif distance_ratio >= 0.5:  # Score moderately lower than threshold
                confidence_level = "Suspicious"
                confidence_percent = min(55 + distance_ratio * 15, 85.0)
                color = "yellow"
            else:  # Score just below threshold
                confidence_level = "Possible Anomaly"
                confidence_percent = min(40 + distance_ratio * 20, 70.0)
                color = "blue"
        else:
            # Score is above threshold - normal behavior for Isolation Forest
            distance_above = (anomaly_score - threshold) / max(abs(threshold), 0.001)
            confidence_level = "Normal Behavior"
            confidence_percent = max(30 - distance_above * 5, 5.0)
            color = "green"
    else:
        # For other algorithms: score > threshold = anomaly (higher is more anomalous)
        is_anomaly = anomaly_score > threshold
        
        if is_anomaly:
            # Score is above threshold - this is flagged as anomaly
            distance_ratio = (anomaly_score - threshold) / max(threshold, 0.001)
            
            # Determine confidence level based on how far above threshold
            if distance_ratio >= 1.0:  # Score is 2x or more than threshold
                confidence_level = "Confirmed Threat"
                confidence_percent = min(85 + distance_ratio * 10, 99.9)
                color = "red"
            elif distance_ratio >= 0.5:  # Score is 1.5x threshold
                confidence_level = "Likely Threat"  
                confidence_percent = min(70 + distance_ratio * 20, 95.0)
                color = "orange"
            elif distance_ratio >= 0.2:  # Score is 1.2x threshold
                confidence_level = "Suspicious"
                confidence_percent = min(55 + distance_ratio * 25, 85.0)
                color = "yellow"
            else:  # Score just above threshold
                confidence_level = "Possible Anomaly"
                confidence_percent = min(40 + distance_ratio * 30, 70.0)
                color = "blue"
        else:
            # Score is below threshold - normal behavior for most algorithms
            distance_below = (threshold - anomaly_score) / max(threshold, 0.001)
            confidence_level = "Normal Behavior"
            confidence_percent = max(30 - distance_below * 10, 5.0)
            color = "green"
    
    # Create columns for display
    col1, col2 = st.columns([1, 2])
    
    # Display confidence level
    with col1:
        st.metric("Confidence Level", confidence_level)
    
    # Display gauge chart for confidence percentage
    with col2:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = confidence_percent,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Confidence"},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 60], 'color': 'lightgray'},
                    {'range': [60, 80], 'color': 'lightblue'},
                    {'range': [80, 90], 'color': 'lightyellow'},
                    {'range': [90, 95], 'color': 'orange'},
                    {'range': [95, 100], 'color': 'lightcoral'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 95
                }
            }
        ))
        
        # Update layout
        fig.update_layout(
            height=150,
            margin=dict(l=30, r=30, t=30, b=0),
            paper_bgcolor="white",
            font=dict(size=12)
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional explanatory text based on confidence
    if confidence_level == "Confirmed Threat":
        st.error("🚨 This anomaly is significant and should be investigated immediately.")
    elif confidence_level == "Likely Threat":
        st.warning("⚠️ This anomaly is concerning and should be investigated.")
    elif confidence_level == "Suspicious":
        st.warning("🔍 This anomaly may warrant investigation.")
    elif confidence_level == "Possible Anomaly":
        st.info("ℹ️ This is a weak anomaly signal - investigate if part of a pattern.")
    else:  # Normal Behavior
        st.success("✅ This appears to be normal network behavior.")
    
    # Show score details for transparency
    with st.expander("📊 Score Details"):
        st.write(f"**Anomaly Score:** {anomaly_score:.4f}")
        st.write(f"**Detection Threshold:** {threshold:.4f}")
        st.write(f"**Classification:** {'Anomaly' if is_anomaly else 'Normal'}")
        
        # Show algorithm-specific interpretation
        algorithm = st.session_state.get('selected_model', 'Unknown')
        if algorithm in ['Isolation Forest', 'IsolationForest']:
            st.write(f"**Algorithm:** {algorithm} (Lower scores = more anomalous)")
            if is_anomaly:
                st.write(f"**Distance Below Threshold:** {(threshold - anomaly_score):.4f}")
                st.write("*Note: For Isolation Forest, scores below threshold indicate anomalies*")
            else:
                st.write(f"**Distance Above Threshold:** {(anomaly_score - threshold):.4f}")
                st.write("*Note: Score above threshold indicates normal behavior*")
        else:
            st.write(f"**Algorithm:** {algorithm} (Higher scores = more anomalous)")
            if is_anomaly:
                st.write(f"**Distance Above Threshold:** {(anomaly_score - threshold):.4f}")
            else:
                st.write(f"**Distance Below Threshold:** {(threshold - anomaly_score):.4f}")

@handle_error
def show_anomaly_detection():
    """Display the anomaly detection page."""
    
    st.header("🔍 Network Anomaly Detection")
    st.markdown("**Detect and analyze network anomalies using advanced machine learning algorithms.**")
    
    # Show current data source status
    show_compact_data_status()
    
    # Ensure data is available
    if not ensure_data_available():
        return
    
    # Get the data
    df = st.session_state.combined_data
    
    # Data summary
    st.subheader("Data Summary")
    
    # Show record count
    st.write(f"Number of records: {len(df)}")
    
    # Create feature processing button
    if st.button("Prepare Protocol-Agnostic Features"):
        with st.spinner("Processing features..."):
            df = prepare_protocol_agnostic_features(df)
            st.session_state.combined_data = df
            st.success("Protocol-agnostic features created successfully.")
            st.rerun()
    
    # Create tabs for the different sections
    tab1, tab2, tab3 = st.tabs(["Train Models", "Analyze Results", "Explanation"])
    
    # Training tab
    with tab1:
        st.subheader("Anomaly Detection Models")
        
        # Initialize model manager
        model_manager = ModelManager()
        
        st.write("### Select Anomaly Detection Algorithm")
        
        # Feature selection
        st.write("Select features to use for anomaly detection:")
        
        # Get all available numeric features
        default_features = get_default_features(df)
        all_numeric_features = df.select_dtypes(include=np.number).columns.tolist()
        
        # Display info about available features
        st.info(f"Found {len(all_numeric_features)} numeric features in your data. All are selected by default - uncheck any you don't want to use.")
        
        # Select features (all selected by default)
        selected_features = st.multiselect(
            "Features", 
            options=all_numeric_features,
            default=default_features,
            help="All numeric features are selected by default. Uncheck any you don't want to include in the analysis."
        )
        
        # Validate selected features (filter out None values)
        selected_features = [f for f in selected_features if f is not None and f in df.columns]
        
        if not selected_features:
            st.error("Please select at least one valid feature for analysis.")
            return
        
        # Show what features are actually available in the data
        with st.expander("📊 Available Features in Your Data"):
            feature_info = []
            for feature in all_numeric_features:
                sample_value = df[feature].iloc[0] if not df[feature].empty else "N/A"
                non_null_count = df[feature].count()
                total_count = len(df)
                feature_info.append({
                    "Feature": feature,
                    "Sample Value": sample_value,
                    "Non-null Count": f"{non_null_count}/{total_count}",
                    "Data Type": str(df[feature].dtype)
                })
            
            feature_df = pd.DataFrame(feature_info)
            st.dataframe(feature_df, use_container_width=True)
        
        # Verify feature selection
        if not selected_features:
            st.warning("Please select at least one feature for anomaly detection.")
            return
        
        # Create columns for algorithm selection and contamination
        col1, col2 = st.columns(2)
        
        with col1:
            # Algorithm selection - expanded list
            algorithm = st.selectbox(
                "Choose Algorithm",
                options=[
                    "Isolation Forest", 
                    "Local Outlier Factor", 
                    "One-Class SVM", 
                    "K-Nearest Neighbors",
                    "HDBSCAN",
                    "Ensemble"
                ],
                help="Each algorithm automatically uses its own saved model if available, or trains a new one"
            )
        
        with col2:
            # Contamination parameter (proportion of outliers)
            contamination = st.slider(
                "Contamination (expected proportion of outliers)",
                min_value=0.001,
                max_value=0.5,
                value=0.33,
                step=0.001,
                format="%.3f"
            )
        
        # Advanced parameters section
        with st.expander("Advanced Parameters"):
            # Algorithm-specific parameters
            if algorithm == "Isolation Forest":
                n_estimators = st.slider("Number of estimators", 50, 500, 100, 10)
                max_samples = st.select_slider(
                    "Max samples", 
                    options=["auto", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"],
                    value="auto"
                )
                
                # Convert max_samples to the right format
                if max_samples != "auto":
                    max_samples = float(max_samples.replace("%", "")) / 100
                
                # FIX: Only include parameters that IsolationForestDetector accepts
                model_params = {
                    "n_estimators": n_estimators,
                    "max_samples": max_samples,
                    "contamination": contamination,
                    "random_state": 42
                }
                
            elif algorithm == "Local Outlier Factor":
                n_neighbors = st.slider("Number of neighbors", 5, 50, 20, 1)
                
                # FIX: Only include parameters that LocalOutlierFactorDetector accepts
                model_params = {
                    "n_neighbors": n_neighbors,
                    "contamination": contamination,
                    "novelty": True
                }
                
            elif algorithm == "One-Class SVM":
                kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
                nu = st.slider("Nu", 0.01, 0.99, contamination, 0.01)
                gamma = st.select_slider(
                    "Gamma", 
                    options=["scale", "auto", "0.001", "0.01", "0.1", "1", "10"],
                    value="scale"
                )
                
                # Convert gamma to the right format
                if gamma not in ["scale", "auto"]:
                    gamma = float(gamma)
                
                model_params = {
                    "kernel": kernel,
                    "nu": nu,
                    "gamma": gamma
                }
                
            elif algorithm == "K-Nearest Neighbors":
                n_neighbors = st.slider("Number of neighbors", 5, 50, 20, 1)
                
                model_params = {
                    "n_neighbors": n_neighbors,
                    "contamination": contamination
                }
                
            elif algorithm == "HDBSCAN":
                min_cluster_size = st.slider("Minimum cluster size", 5, 50, 15, 1)
                min_samples = st.slider("Minimum samples", 1, 20, 5, 1)
                
                model_params = {
                    "min_cluster_size": min_cluster_size,
                    "min_samples": min_samples,
                    "contamination": contamination
                }
                
            elif algorithm == "Ensemble":
                use_isolation_forest = st.checkbox("Use Isolation Forest", value=True)
                use_lof = st.checkbox("Use Local Outlier Factor", value=True)
                use_ocsvm = st.checkbox("Use One-Class SVM", value=False)
                use_knn = st.checkbox("Use K-Nearest Neighbors", value=False)
                
                ensemble_method = st.selectbox(
                    "Ensemble Method", 
                    options=["average", "maximum", "weighted"],
                    help="How to combine scores from different models"
                )
                
                model_params = {
                    "contamination": contamination,
                    "models": [],
                    "method": ensemble_method,
                    "random_state": 42
                }
                
                if use_isolation_forest:
                    model_params["models"].append({
                        "type": "IsolationForest",
                        "params": {
                            "n_estimators": 100,
                            "contamination": contamination,
                            "random_state": 42
                        }
                    })
                
                if use_lof:
                    model_params["models"].append({
                        "type": "LocalOutlierFactor",
                        "params": {
                            "n_neighbors": 20,
                            "contamination": contamination,
                            "novelty": True
                        }
                    })
                
                if use_ocsvm:
                    model_params["models"].append({
                        "type": "OneClassSVM",
                        "params": {
                            "nu": contamination,
                            "gamma": "scale"
                        }
                    })
                
                if use_knn:
                    model_params["models"].append({
                        "type": "KNN",
                        "params": {
                            "n_neighbors": 20,
                            "contamination": contamination
                        }
                    })
                
                if not model_params["models"]:
                    st.warning("Please select at least one model for the ensemble.")
                    return
        
        # Data preprocessing options
        with st.expander("Data Preprocessing Options"):
            handle_missing = st.radio(
                "Handle missing values",
                options=["Fill missing values with mean", "Fill missing values with median", 
                         "Fill missing values with 0", "Drop rows with missing values"],
                index=0,
                key="handle_missing_train"
            )
            
            normalize_data = st.checkbox("Normalize data", value=True)
        
        # Run Analysis button
        if st.button("🔍 Run Anomaly Detection", type="primary", help="Automatically uses existing model if available, or trains a new one"):
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Validate inputs
                if not selected_features:
                    st.error("Please select at least one feature.")
                    return
                
                # Extract features
                X = df[selected_features].copy()
                
                # Handle missing values
                if handle_missing == "Fill missing values with mean":
                    X = X.fillna(X.mean())
                elif handle_missing == "Fill missing values with median":
                    X = X.fillna(X.median())
                elif handle_missing == "Fill missing values with 0":
                    X = X.fillna(0)
                else:  # Drop rows
                    X = X.dropna()
                
                # Check for existing model of this type and auto-load if available
                model_type_name = algorithm.replace(" ", "")
                existing_model = None
                
                status_text.text(f"Checking for existing {algorithm} model...")
                progress_bar.progress(10)
                
                try:
                    existing_model = model_manager.load_model(model_type_name)
                    if existing_model:
                        st.info(f"🔄 **Found existing {algorithm} model - using saved model**")
                        model = existing_model
                        
                        # Update metadata to show it was reused
                        if not hasattr(model, 'metadata'):
                            model.metadata = {}
                        model.metadata.update({
                            "last_used": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "reuse_count": model.metadata.get("reuse_count", 0) + 1
                        })
                    else:
                        raise FileNotFoundError("No existing model")
                        
                except:
                    # Create new model if none exists
                    status_text.text(f"Creating new {algorithm} model...")
                    progress_bar.progress(15)
                    
                    # Initialize the appropriate model
                    if algorithm == "Isolation Forest":
                        model = IsolationForestDetector(**model_params)
                    elif algorithm == "Local Outlier Factor":
                        model = LocalOutlierFactorDetector(**model_params)
                    elif algorithm == "One-Class SVM":
                        model = OneClassSVMDetector(**model_params)
                    elif algorithm == "K-Nearest Neighbors":
                        model = KNNDetector(**model_params)
                    elif algorithm == "HDBSCAN":
                        model = HDBSCANDetector(**model_params)
                    elif algorithm == "Ensemble":
                        # Create actual model instances for ensemble
                        ensemble_models = []
                        
                        # Create the individual models based on selected options
                        if use_isolation_forest:
                            from core.models.isolation_forest import IsolationForestDetector
                            if_model = IsolationForestDetector(
                                n_estimators=100,
                                contamination=contamination,
                                random_state=42
                            )
                            ensemble_models.append(if_model)
                        
                        if use_lof:
                            from core.models.local_outlier_factor import LocalOutlierFactorDetector
                            lof_model = LocalOutlierFactorDetector(
                                n_neighbors=20,
                                contamination=contamination,
                                novelty=True
                            )
                            ensemble_models.append(lof_model)
                        
                        if use_ocsvm:
                            from core.models.one_class_svm import OneClassSVMDetector
                            svm_model = OneClassSVMDetector(
                                nu=contamination,
                                gamma="scale"
                            )
                            ensemble_models.append(svm_model)
                        
                        if use_knn:
                            from core.models.knn import KNNDetector
                            knn_model = KNNDetector(
                                n_neighbors=20,
                                contamination=contamination
                            )
                            ensemble_models.append(knn_model)
                        
                        # Create the ensemble with actual model instances
                        if ensemble_models:
                            model = EnsembleDetector(
                                models=ensemble_models,
                                contamination=contamination,
                                combination_method=ensemble_method
                            )
                        else:
                            st.error("Please select at least one model for the ensemble.")
                            return
                    
                    # Set feature names
                    model.feature_names = selected_features
                    
                    # Train the model
                    status_text.text("Training new model...")
                    progress_bar.progress(30)
                    model.fit(X)
                    
                    st.success(f"✅ **Created and trained new {algorithm} model**")
                
                # Apply model to data and save results using ModelManager
                status_text.text("Applying model and saving results...")
                progress_bar.progress(60)
                
                # Use ModelManager's apply_model_to_data method which handles everything
                try:
                    results = model_manager.apply_model_to_data(
                        model_type=model_type_name,
                        data=df,  # Use full dataframe, not just X
                        feature_names=selected_features,
                        recalculate_threshold=False,  # Use existing threshold
                        save_results=True  # This will save to data/results
                    )
                    
                    # Extract results
                    scores = results['scores']
                    threshold = results['threshold'] 
                    anomalies = results['anomalies']
                    anomaly_indices = results['anomaly_indices']
                    
                    st.success(f"✅ **Results saved to {results.get('results_path', 'data/results')}**")
                    
                except Exception as e:
                    logger.error(f"Error using apply_model_to_data: {e}")
                    st.warning(f"Could not save results automatically: {e}")
                    
                    # Fallback to manual prediction (without saving)
                    scores = model.predict(X)
                    threshold = np.percentile(scores, 100 * (1 - contamination))
                    
                    # Identify anomalies based on algorithm type
                    if algorithm == "Isolation Forest":
                        is_anomaly = scores < threshold
                    else:
                        is_anomaly = scores > threshold
                        
                    anomaly_indices = np.where(is_anomaly)[0]
                    anomalies = X.iloc[anomaly_indices].copy() if len(anomaly_indices) > 0 else pd.DataFrame()
                    
                    # Add anomaly scores
                    if len(anomalies) > 0:
                        anomalies['anomaly_score'] = scores[anomaly_indices]
                
                # Save the model
                status_text.text("Saving model...")
                progress_bar.progress(80)
                
                # Update model metadata with timestamp
                if not hasattr(model, 'metadata'):
                    model.metadata = {}
                    
                # Determine if this was a new model or existing one
                was_existing_model = existing_model is not None
                    
                model.metadata.update({
                    "trained_at" if not was_existing_model else "last_used": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "feature_names": selected_features,
                    "contamination": contamination,
                    "anomaly_threshold": threshold,
                    "parameters": model_params,
                    "type": model_type_name
                })
                
                # Save model with threshold (creates backup automatically)
                if not was_existing_model:
                    model_path = model_manager.save_model(model, threshold=threshold, create_backup=True)
                else:
                    # For existing models, just update the metadata
                    model_path = f"models/{model_type_name}_model.pkl"
                
                # Store results in session state (filter out None values from features)
                valid_features = [f for f in selected_features if f is not None and f in df.columns]
                st.session_state.anomaly_model = model
                st.session_state.anomaly_scores = scores
                st.session_state.anomaly_threshold = threshold
                st.session_state.anomaly_features = valid_features
                st.session_state.anomalies = anomalies
                st.session_state.anomaly_indices = anomaly_indices
                st.session_state.selected_model = algorithm  # Store the algorithm name
                
                # 🤖 AUTOMATIC ANALYSIS: Run MITRE mapping and risk scoring automatically
                if len(anomalies) > 0:
                    with st.spinner("🤖 Running automatic analysis (MITRE mapping & risk scoring)..."):
                        auto_results = auto_analysis_service.run_automatic_analysis(anomalies)
                        notification_service.show_auto_analysis_results(auto_results)
                
                # Update progress
                progress_bar.progress(100)
                status_text.text("Model analysis complete!")
                
                # Success message
                action_text = "reused existing model and analyzed" if was_existing_model else "trained new model and saved to"
                if was_existing_model:
                    st.success(f"✅ **{action_text} data successfully!**")
                else:
                    st.success(f"✅ **Model {action_text} {model_path}**")
                
                # Display quick stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", len(X))
                with col2:
                    st.metric("Anomalies", len(anomalies))
                with col3:
                    anomaly_percent = (len(anomalies) / len(X)) * 100 if len(X) > 0 else 0
                    st.metric("Anomaly %", f"{anomaly_percent:.2f}%")
                
            except Exception as e:
                error_msg = f"Error training model: {str(e)}"
                st.error(error_msg)
                logger.exception(error_msg)
    
    # Results analysis tab
    with tab2:
        st.subheader("Analyze Anomaly Detection Results")
        
        # Check if model has been trained
        if 'anomaly_model' not in st.session_state or 'anomaly_scores' not in st.session_state:
            st.info("Train a model in the Training tab first.")
            return
        
        # Get results from session state
        model = st.session_state.anomaly_model
        scores = st.session_state.anomaly_scores
        threshold = st.session_state.anomaly_threshold
        features = st.session_state.anomaly_features
        anomalies = st.session_state.anomalies
        
        # Validate features (filter out None values and ensure they exist in DataFrame)
        if features:
            features = [f for f in features if f is not None and f in df.columns]
        
        if not features:
            st.error("No valid features found in session state. Please re-run the anomaly detection.")
            return
        
        # Results summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Anomalies", len(anomalies))
        with col3:
            anomaly_percent = (len(anomalies) / len(df)) * 100 if len(df) > 0 else 0
            st.metric("Anomaly %", f"{anomaly_percent:.2f}%")
        
        # Plot anomaly score distribution
        st.subheader("Anomaly Score Distribution")
        fig = plot_anomaly_scores(df[features], scores, threshold)
        # FIX: Add unique key to plotly_chart
        st.plotly_chart(fig, use_container_width=True, key="results_score_dist")
        
        # Feature visualization
        st.subheader("Feature Visualization")
        
        if len(features) >= 2:
            # Select features for scatter plot
            col1, col2 = st.columns(2)
            with col1:
                x_feature = st.selectbox("X-axis feature", options=features, index=0)
            with col2:
                y_feature = st.selectbox("Y-axis feature", options=features, index=min(1, len(features)-1))
            
            # Plot scatter
            fig = plot_anomaly_scatter(df[features], scores, threshold, x_feature, y_feature)
            # FIX: Add unique key to plotly_chart
            st.plotly_chart(fig, use_container_width=True, key="scatter_plot")
        
        # Timeline visualization if timestamp is available
        st.subheader("Anomaly Timeline")
        fig = plot_anomaly_timeline(df, scores, threshold)
        if fig:
            # FIX: Add unique key to plotly_chart
            st.plotly_chart(fig, use_container_width=True, key="timeline_plot")
        else:
            st.info("Timeline visualization requires a timestamp column.")
        
        # Feature importance
        st.subheader("Feature Importance")
        fig = plot_feature_importance(df[features], scores, threshold)
        if fig:
            # FIX: Add unique key to plotly_chart
            st.plotly_chart(fig, use_container_width=True, key="feature_importance_plot")
        else:
            st.info("Could not calculate feature importance.")
        
        # Network graph if IP addresses are available
        st.subheader("Network Graph")
        
        # Detect IP columns
        ip_src_col = None
        ip_dst_col = None
        
        if 'ip_src' in df.columns:
            ip_src_col = 'ip_src'
        elif 'ip.src' in df.columns:
            ip_src_col = 'ip.src'
            
        if 'ip_dst' in df.columns:
            ip_dst_col = 'ip_dst'
        elif 'ip.dst' in df.columns:
            ip_dst_col = 'ip.dst'
        
        if ip_src_col and ip_dst_col:
            # Generate network graph
            with st.spinner("Generating network graph..."):
                html = plot_network_graph(df, ip_src_col, ip_dst_col, scores, threshold)
                if html:
                    # FIX: Remove the key parameter from html component
                    st.components.v1.html(html, height=600)
                else:
                    st.info("Could not generate network graph.")
        else:
            st.info("Network graph visualization requires source and destination IP columns.")
        
        # Anomaly details table
        st.subheader("Anomaly Details")
        
        if len(anomalies) > 0:
            # Get all columns from the original DataFrame for the anomalies
            anomaly_indices = anomalies.index
            full_anomalies = df.loc[anomaly_indices].copy()
            
            # Add anomaly scores
            full_anomalies['anomaly_score'] = scores[anomaly_indices]
            
            # Sort by anomaly score (descending)
            full_anomalies = full_anomalies.sort_values('anomaly_score', ascending=False)
            
            # Display the anomalies
            st.dataframe(full_anomalies, use_container_width=True)
            
            # Export options
            col1, col2 = st.columns(2)
            with col1:
                csv = full_anomalies.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download CSV",
                    data=csv,
                    file_name=f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_csv"
                )
            with col2:
                json_str = full_anomalies.to_json(orient='records', date_format='iso')
                st.download_button(
                    "Download JSON",
                    data=json_str,
                    file_name=f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="download_json"
                )
        else:
            st.info("No anomalies detected.")
    
    # Explanation tab
    with tab3:
        st.subheader("Model Explanation")
        
        # Check if model has been trained
        if 'anomaly_model' not in st.session_state or 'anomaly_scores' not in st.session_state:
            st.info("Train a model in the Training tab first.")
            return
        
        # Get results from session state
        model = st.session_state.anomaly_model
        scores = st.session_state.anomaly_scores
        threshold = st.session_state.anomaly_threshold
        features = st.session_state.anomaly_features
        anomalies = st.session_state.anomalies
        
        # Validate features (filter out None values and ensure they exist in DataFrame)
        if features:
            features = [f for f in features if f is not None and f in df.columns]
        
        if not features:
            st.error("No valid features found in session state. Please re-run the anomaly detection.")
            return
        
        # Model information
        st.write("### Model Information")
        
        # Display model metadata
        metadata = model.metadata if hasattr(model, 'metadata') else {}
        
        # Create columns for metadata
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Model Type:**", metadata.get("type", type(model).__name__))
            st.write("**Trained At:**", metadata.get("trained_at", "Unknown"))
            st.write("**Features:**", ", ".join(features))
        
        with col2:
            st.write("**Anomaly Threshold:**", threshold)
            st.write("**Contamination:**", metadata.get("contamination", "Unknown"))
            st.write("**Total Anomalies:**", len(anomalies))
        
        # Model parameters
        st.write("### Model Parameters")
        parameters = metadata.get("parameters", {})
        if parameters:
            # Convert parameters to a readable format
            param_text = []
            for param, value in parameters.items():
                if param != "models":  # Skip detailed model list for ensembles
                    param_text.append(f"**{param}:** {value}")
            
            st.markdown("<br>".join(param_text), unsafe_allow_html=True)
        else:
            st.info("No parameter information available.")
        
        # Toggle between SHAP and LIME
        explanation_method = st.radio(
            "Select explanation method:",
            options=["SHAP", "LIME"],
            index=0,
            horizontal=True
        )
        
        # Get appropriate explainer
        try:
            if explanation_method.lower() == "shap":
                explainer = get_explainer("shap", model=model, X=df[features], feature_names=features)
            else:
                explainer = get_explainer("lime", model=model, X=df[features], feature_names=features)
        except Exception as e:
            st.error(f"Error initializing {explanation_method} explainer: {str(e)}")
            return
        
        if explanation_method == "SHAP":
            st.write("### SHAP Feature Importance")
            
            # Global model explanation
            with st.spinner("Calculating SHAP values (this may take a while)..."):
                try:
                    # Use the explainer to get global explanations
                    explanation = explainer.explain_global(df[features])
                    
                    # Check for errors
                    if "error" in explanation:
                        st.error(f"Error calculating SHAP values: {explanation['error']}")
                    else:
                        # Use the explainer's built-in plotting methods
                        try:
                            # Summary plot (bar)
                            st.write("#### Global Feature Importance (Bar)")
                            try:
                                fig_bar = explainer.plot_summary(explanation, plot_type="bar", max_display=10)
                                st.pyplot(fig_bar)
                                plt.close(fig_bar)
                            except Exception as e:
                                st.warning(f"Could not create SHAP bar plot: {str(e)}")
                            
                            # Summary plot (beeswarm)
                            st.write("#### Feature Impact Distribution (Beeswarm)")
                            try:
                                fig_beeswarm = explainer.plot_summary(explanation, plot_type="beeswarm", max_display=10)
                                st.pyplot(fig_beeswarm)
                                plt.close(fig_beeswarm)
                            except Exception as e:
                                st.warning(f"Could not create SHAP beeswarm plot: {str(e)}")
                            
                            # Dependence plots for top features
                            st.write("#### SHAP Dependence Plots")
                            try:
                                # Get feature importance to determine top features
                                feature_importance = explainer.get_feature_importance(explanation)
                                top_features = list(feature_importance.keys())[:3]  # Top 3 features
                                
                                for i, feature in enumerate(top_features):
                                    try:
                                        fig_dep = explainer.plot_dependence(explanation, feature)
                                        st.pyplot(fig_dep)
                                        plt.close(fig_dep)
                                    except Exception as e:
                                        st.warning(f"Could not create dependence plot for {feature}: {str(e)}")
                            except Exception as e:
                                st.warning(f"Could not create dependence plots: {str(e)}")
                            
                            # Feature importance table
                            st.write("#### Feature Importance Table")
                            try:
                                feature_importance = explainer.get_feature_importance(explanation)
                                importance_df = pd.DataFrame([
                                    {"Feature": k, "Importance": v} for k, v in feature_importance.items()
                                ])
                                
                                # Create interactive bar chart
                                fig_importance = px.bar(
                                    importance_df,
                                    x='Feature',
                                    y='Importance',
                                    title="Global Feature Importance (Mean |SHAP|)",
                                    color='Importance',
                                    color_continuous_scale='viridis'
                                )
                                st.plotly_chart(fig_importance, use_container_width=True, key="shap_global_importance")
                                st.dataframe(importance_df, use_container_width=True)
                            except Exception as e:
                                st.warning(f"Could not create feature importance table: {str(e)}")
                            
                        except Exception as e:
                            st.warning(f"Could not create SHAP visualizations: {str(e)}")
                                
                except Exception as e:
                    st.error(f"Error calculating SHAP values: {str(e)}")
                    logger.exception(f"Error calculating SHAP values: {str(e)}")
                    
                    # Show feature importance from the model instead
                    st.write("### Alternative Feature Importance")
                    try:
                        X = df[features].copy()
                        X = X.fillna(0)
                        
                        # Calculate feature importance using permutation importance or correlation
                        try:
                            # Try to get feature importance if the method exists
                            importance = model.get_feature_importance(X)
                        except:
                            # Calculate simple correlation-based importance
                            scores_subset = scores[:len(X)]  # Make sure arrays match
                            importance = {}
                            for feature in features:
                                correlation = np.corrcoef(X[feature], scores_subset)[0, 1]
                                importance[feature] = abs(correlation) if not np.isnan(correlation) else 0
                        
                        # FIX: Check if importance is None before accessing keys
                        if importance is not None and len(importance) > 0:
                            # Convert to DataFrame for plotting
                            importance_df = pd.DataFrame({
                                'Feature': list(importance.keys()),
                                'Importance': list(importance.values())
                            }).sort_values('Importance', ascending=False)
                            
                            # Plot as a bar chart
                            fig = px.bar(
                                importance_df, 
                                x='Feature', 
                                y='Importance',
                                title="Feature Importance (Correlation with Anomaly Scores)",
                                color='Importance',
                                color_continuous_scale='viridis'
                            )
                            # Add unique key to plotly_chart
                            st.plotly_chart(fig, use_container_width=True, key="alt_feature_importance")
                        else:
                            st.info("Feature importance calculation returned no results.")
                    except Exception as e:
                        st.error(f"Error calculating feature importance: {str(e)}")
                        logger.exception(f"Error calculating feature importance: {str(e)}")
        else:  # LIME explanation
            st.write("### LIME Feature Importance")
            
            # Global explanation approach for LIME
            st.info("LIME provides local explanations. Below are comprehensive visualizations based on multiple anomaly samples.")
            
            # Show sample explanations for multiple anomalies
            if len(anomalies) > 0:
                st.write("#### Aggregated LIME Analysis")
                
                # Get multiple anomalies for analysis
                anomaly_indices = anomalies.index.tolist()
                sample_size = min(10, len(anomaly_indices))  # Sample up to 10 anomalies
                sample_anomalies = sorted(anomaly_indices, key=lambda x: scores[x], reverse=True)[:sample_size]
                
                # Collect feature importance from multiple samples
                all_feature_importance = {}
                sample_explanations = []
                
                progress_bar = st.progress(0)
                for i, idx in enumerate(sample_anomalies):
                    try:
                        sample_explanation = explainer.explain_instance(df[features], idx)
                        
                        # Check if explanation contains an error
                        if "error" in sample_explanation:
                            st.warning(f"Could not analyze anomaly {idx}: {sample_explanation['error']}")
                            continue
                            
                        feature_importance = sample_explanation.get("feature_importance", {})
                        
                        # Accumulate feature importance
                        for feat, weight in feature_importance.items():
                            if feat not in all_feature_importance:
                                all_feature_importance[feat] = []
                            all_feature_importance[feat].append(weight)
                        
                        sample_explanations.append((idx, sample_explanation))
                        progress_bar.progress((i + 1) / sample_size)
                        
                    except Exception as e:
                        st.warning(f"Could not analyze anomaly {idx}: {str(e)}")
                
                progress_bar.empty()
                
                if all_feature_importance:
                    # Calculate aggregate statistics
                    st.write("#### Aggregate Feature Importance Across Anomalies")
                    
                    agg_stats = []
                    for feat, weights in all_feature_importance.items():
                        agg_stats.append({
                            'Feature': feat,
                            'Mean_Weight': np.mean(weights),
                            'Std_Weight': np.std(weights),
                            'Mean_Abs_Weight': np.mean(np.abs(weights)),
                            'Max_Weight': np.max(weights),
                            'Min_Weight': np.min(weights),
                            'Count': len(weights)
                        })
                    
                    agg_df = pd.DataFrame(agg_stats).sort_values('Mean_Abs_Weight', ascending=False)
                    
                    # Mean importance bar chart
                    fig_mean = px.bar(
                        agg_df,
                        x='Feature',
                        y='Mean_Weight',
                        title="Average LIME Feature Weights Across Anomalies",
                        color='Mean_Weight',
                        color_continuous_scale='RdBu_r',
                        hover_data=['Std_Weight', 'Count']
                    )
                    fig_mean.add_hline(y=0, line_dash="dash", line_color="black")
                    st.plotly_chart(fig_mean, use_container_width=True, key="lime_mean_weights")
                    
                    # Absolute importance
                    fig_abs = px.bar(
                        agg_df,
                        x='Feature',
                        y='Mean_Abs_Weight',
                        title="Average Absolute LIME Feature Importance",
                        color='Mean_Abs_Weight',
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig_abs, use_container_width=True, key="lime_abs_weights")
                    
                    # Box plot showing distribution of weights per feature
                    st.write("#### Feature Weight Distributions")
                    
                    # Prepare data for box plot
                    box_data = []
                    for feat, weights in all_feature_importance.items():
                        for weight in weights:
                            box_data.append({'Feature': feat, 'Weight': weight})
                    
                    if box_data:
                        box_df = pd.DataFrame(box_data)
                        fig_box = px.box(
                            box_df,
                            x='Feature',
                            y='Weight',
                            title="Distribution of LIME Weights by Feature"
                        )
                        fig_box.add_hline(y=0, line_dash="dash", line_color="black")
                        st.plotly_chart(fig_box, use_container_width=True, key="lime_weight_distribution")
                    
                    # Heatmap of weights across anomalies
                    st.write("#### Feature Weights Heatmap")
                    
                    # Create matrix for heatmap
                    heatmap_data = []
                    for idx, explanation in sample_explanations:
                        feature_importance = explanation.get("feature_importance", {})
                        row = [feature_importance.get(feat, 0) for feat in features]
                        heatmap_data.append(row)
                    
                    if heatmap_data:
                        heatmap_df = pd.DataFrame(heatmap_data, 
                                                columns=features,
                                                index=[f"Anomaly_{idx}" for idx, _ in sample_explanations])
                        
                        fig_heatmap = px.imshow(
                            heatmap_df.T,  # Transpose to have features on y-axis
                            title="LIME Feature Weights Across Different Anomalies",
                            color_continuous_scale='RdBu_r',
                            aspect='auto'
                        )
                        fig_heatmap.update_layout(
                            xaxis_title="Anomaly Instances",
                            yaxis_title="Features"
                        )
                        st.plotly_chart(fig_heatmap, use_container_width=True, key="lime_heatmap")
                    
                    # Summary statistics table
                    st.write("#### Summary Statistics")
                    st.dataframe(agg_df, use_container_width=True)
                    
                # Individual sample explanations in expandable sections
                st.write("#### Individual LIME Explanations")
                for i, (idx, explanation) in enumerate(sample_explanations[:5]):  # Show top 5
                    with st.expander(f"Detailed Analysis - Anomaly ID {idx} (Score: {scores[idx]:.4f})"):
                        feature_importance = explanation.get("feature_importance", {})
                        
                        if feature_importance:
                            importance_df = pd.DataFrame([
                                {"Feature": k, "Weight": v} for k, v in feature_importance.items()
                            ]).sort_values('Weight', key=abs, ascending=False)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Bar chart
                                fig_individual = px.bar(
                                    importance_df,
                                    x='Feature',
                                    y='Weight',
                                    title=f"Feature Weights for Anomaly {idx}",
                                    color='Weight',
                                    color_continuous_scale='RdBu_r'
                                )
                                fig_individual.add_hline(y=0, line_dash="dash", line_color="black")
                                st.plotly_chart(fig_individual, use_container_width=True, key=f"lime_individual_{i}_{idx}")
                            
                            with col2:
                                # Feature values for this anomaly
                                feature_values = df.loc[idx, features].to_dict()
                                values_df = pd.DataFrame([
                                    {"Feature": k, "Value": v, "Weight": feature_importance.get(k, 0)}
                                    for k, v in feature_values.items()
                                ])
                                st.dataframe(values_df, use_container_width=True)
            else:
                st.info("No anomalies detected to create LIME analysis.")
            
        # Explain specific anomalies
        st.write("### Explain Specific Anomalies")
        
        if len(anomalies) > 0:
            # Let user select an anomaly to explain
            anomaly_indices = anomalies.index.tolist()
            
            # Get some identifier for the anomalies (time, IP, etc.)
            identifier_col = None
            for col in ['timestamp', 'frame.time', 'ip_src', 'ip.src']:
                if col in df.columns:
                    identifier_col = col
                    break
            
            # Create options for selection
            options = []
            for idx in anomaly_indices:
                score = scores[idx]
                if identifier_col:
                    identifier = df.loc[idx, identifier_col]
                    option_text = f"ID: {idx} - {identifier_col}: {identifier} (Score: {score:.4f})"
                else:
                    option_text = f"ID: {idx} (Score: {score:.4f})"
                options.append((idx, option_text))
            
            # Sort by score (descending)
            options.sort(key=lambda x: scores[x[0]], reverse=True)
            
            # Limit to top 20 for better usability
            options = options[:20]
            
            # Create a selectbox for user to choose an anomaly
            selected_option = st.selectbox(
                "Select an anomaly to explain",
                options=[text for _, text in options],
                index=0,
                key="anomaly_select"
            )
            
            # Get the index of the selected anomaly
            selected_idx = options[[text for _, text in options].index(selected_option)][0]
            
            # Display the selected anomaly's details
            st.write("#### Selected Anomaly Details")
            anomaly_details = df.loc[selected_idx].to_dict()
            anomaly_details['anomaly_score'] = scores[selected_idx]
            
            # Create a more readable display of the details
            col1, col2 = st.columns(2)
            
            # Display score and basic info in first column
            with col1:
                st.write(f"**Anomaly Score:** {anomaly_details['anomaly_score']:.4f}")
                if 'timestamp' in anomaly_details:
                    st.write(f"**Timestamp:** {anomaly_details['timestamp']}")
                if 'ip_src' in anomaly_details or 'ip.src' in anomaly_details:
                    ip_src = anomaly_details.get('ip_src', anomaly_details.get('ip.src', 'N/A'))
                    st.write(f"**Source IP:** {ip_src}")
                if 'ip_dst' in anomaly_details or 'ip.dst' in anomaly_details:
                    ip_dst = anomaly_details.get('ip_dst', anomaly_details.get('ip.dst', 'N/A'))
                    st.write(f"**Destination IP:** {ip_dst}")
            
            # Display protocol and port info in second column
            with col2:
                if 'protocol' in anomaly_details or '_ws.col.Protocol' in anomaly_details:
                    protocol = anomaly_details.get('protocol', anomaly_details.get('_ws.col.Protocol', 'N/A'))
                    st.write(f"**Protocol:** {protocol}")
                if 'src_port' in anomaly_details:
                    st.write(f"**Source Port:** {anomaly_details['src_port']}")
                if 'dst_port' in anomaly_details:
                    st.write(f"**Destination Port:** {anomaly_details['dst_port']}")
                if 'packet_length' in anomaly_details or 'frame.len' in anomaly_details:
                    length = anomaly_details.get('packet_length', anomaly_details.get('frame.len', 'N/A'))
                    st.write(f"**Packet Length:** {length}")
            
            # Display confidence level
            display_confidence_level(anomaly_details['anomaly_score'], threshold)
            
            # Try to explain the anomaly with the selected explainer
            try:
                with st.spinner(f"Calculating {explanation_method} explanation..."):
                    try:
                        # Use the explainer to get local explanations
                        local_explanation = explainer.explain_instance(df[features], selected_idx)
                    except Exception as e:
                        st.error(f"Error calculating {explanation_method} explanation: {str(e)}")
                        local_explanation = {"error": str(e)}
                    
                    if "error" in local_explanation:
                        st.error(f"Error calculating explanation: {local_explanation['error']}")
                        
                        # Provide fallback simple analysis
                        st.write("#### Fallback Analysis")
                        st.write("Since detailed explanation failed, here's a basic feature comparison:")
                        
                        # Show feature values for this anomaly
                        feature_values = df.loc[selected_idx, features]
                        feature_df = pd.DataFrame({
                            'Feature': features,
                            'Value': feature_values.values
                        })
                        
                        # Add comparison to normal data if available
                        non_anomaly_indices = np.where(scores <= threshold)[0]
                        if len(non_anomaly_indices) > 0:
                            normal_data = df.iloc[non_anomaly_indices][features]
                            feature_df['Normal_Mean'] = [normal_data[feat].mean() for feat in features]
                            feature_df['Difference'] = feature_df['Value'] - feature_df['Normal_Mean']
                            feature_df['Abs_Difference'] = feature_df['Difference'].abs()
                            feature_df = feature_df.sort_values('Abs_Difference', ascending=False)
                            
                            # Create a simple comparison chart
                            fig_fallback = px.bar(
                                feature_df,
                                x='Feature',
                                y='Difference',
                                title=f"Feature Deviation from Normal for Anomaly {selected_idx}",
                                color='Difference',
                                color_continuous_scale='RdBu_r'
                            )
                            fig_fallback.add_hline(y=0, line_dash="dash", line_color="black")
                            st.plotly_chart(fig_fallback, use_container_width=True, key=f"fallback_{selected_idx}")
                        
                        st.dataframe(feature_df, use_container_width=True)
                        
                    else:
                        st.write("#### Feature Contributions")
                        st.write(f"This visualization shows how each feature contributed to the anomaly score of this specific data point.")
                        
                        if explanation_method == "SHAP":
                            # Create SHAP visualizations using the explainer's built-in methods
                            try:
                                # Force plot
                                st.write("##### SHAP Force Plot")
                                try:
                                    force_plot = explainer.plot_force(local_explanation, instance_index=0)
                                    # SHAP force plots are interactive, we'll create a static version
                                    fig_force = plt.figure(figsize=(12, 4))
                                    
                                    shap_values = local_explanation.get("shap_values")
                                    if shap_values is not None:
                                        if len(shap_values.shape) == 2:
                                            shap_vals = shap_values[0]
                                        else:
                                            shap_vals = shap_values
                                        
                                        # Create horizontal bar plot
                                        feature_vals = df.loc[selected_idx, features].values
                                        y_pos = np.arange(len(features))
                                        colors = ['red' if x > 0 else 'blue' for x in shap_vals]
                                        
                                        plt.barh(y_pos, shap_vals, color=colors, alpha=0.7)
                                        plt.yticks(y_pos, [f"{feat}={val:.3f}" for feat, val in zip(features, feature_vals)])
                                        plt.xlabel('SHAP Value (impact on model output)')
                                        plt.title(f'SHAP Values for Anomaly {selected_idx}')
                                        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                                        plt.tight_layout()
                                        
                                        st.pyplot(fig_force)
                                        plt.close(fig_force)
                                except Exception as e:
                                    st.warning(f"Could not create SHAP force plot: {str(e)}")
                                
                                # Feature importance for this instance
                                st.write("##### Individual Feature Importance")
                                try:
                                    shap_values = local_explanation.get("shap_values")
                                    if shap_values is not None:
                                        if len(shap_values.shape) == 2:
                                            shap_vals = shap_values[0]
                                        else:
                                            shap_vals = shap_values
                                        
                                        importance_df = pd.DataFrame({
                                            'Feature': features,
                                            'SHAP_Value': shap_vals,
                                            'Abs_Impact': np.abs(shap_vals),
                                            'Feature_Value': df.loc[selected_idx, features].values
                                        }).sort_values('Abs_Impact', ascending=False)
                                        
                                        # Create interactive bar chart
                                        fig_bar = px.bar(
                                            importance_df,
                                            x='Feature',
                                            y='SHAP_Value',
                                            title=f"SHAP Feature Contributions for Anomaly {selected_idx}",
                                            color='SHAP_Value',
                                            color_continuous_scale='RdBu_r',
                                            hover_data=['Feature_Value', 'Abs_Impact']
                                        )
                                        fig_bar.add_hline(y=0, line_dash="dash", line_color="black")
                                        st.plotly_chart(fig_bar, use_container_width=True, key=f"shap_bar_{selected_idx}")
                                        
                                        # Create a waterfall-style chart
                                        st.write("##### SHAP Waterfall Chart")
                                        cumulative = np.cumsum(np.concatenate([[0], shap_vals]))
                                        base_value = local_explanation.get("base_value", 0)
                                        
                                        waterfall_df = pd.DataFrame({
                                            'Feature': ['Base'] + features + ['Final'],
                                            'Value': [base_value] + list(shap_vals) + [base_value + np.sum(shap_vals)],
                                            'Cumulative': [base_value] + list(base_value + cumulative[1:]) + [base_value + np.sum(shap_vals)]
                                        })
                                        
                                        fig_waterfall = px.bar(
                                            waterfall_df[1:-1],  # Exclude base and final
                                            x='Feature',
                                            y='Value',
                                            title=f"SHAP Waterfall for Anomaly {selected_idx}",
                                            color='Value',
                                            color_continuous_scale='RdBu_r'
                                        )
                                        fig_waterfall.add_hline(y=0, line_dash="dash", line_color="black")
                                        st.plotly_chart(fig_waterfall, use_container_width=True, key=f"shap_waterfall_{selected_idx}")
                                        
                                        # Show data table
                                        st.dataframe(importance_df, use_container_width=True)
                                        
                                except Exception as e:
                                    st.warning(f"Could not create individual importance plots: {str(e)}")
                                    
                            except Exception as e:
                                st.error(f"Error creating SHAP visualizations: {str(e)}")
                                
                        else:  # LIME explanation
                            try:
                                # Use LIME explainer's built-in plotting
                                st.write("##### LIME Explanation Plot")
                                try:
                                    fig_lime = explainer.plot_explanation(local_explanation)
                                    st.pyplot(fig_lime)
                                    plt.close(fig_lime)
                                except Exception as e:
                                    st.warning(f"Could not create LIME plot: {str(e)}")
                                
                                # Feature importance charts
                                lime_explanation = local_explanation.get("lime_explanation")
                                feature_importance = local_explanation.get("feature_importance", {})
                                
                                if feature_importance:
                                    # Main feature weights bar chart
                                    st.write("##### LIME Feature Weights")
                                    importance_df = pd.DataFrame([
                                        {"Feature": k, "Weight": v} for k, v in feature_importance.items()
                                    ]).sort_values('Weight', key=abs, ascending=False)
                                    
                                    fig_weights = px.bar(
                                        importance_df,
                                        x='Feature',
                                        y='Weight',
                                        title=f"LIME Feature Weights for Anomaly {selected_idx}",
                                        color='Weight',
                                        color_continuous_scale='RdBu_r',
                                        hover_data=['Weight']
                                    )
                                    fig_weights.add_hline(y=0, line_dash="dash", line_color="black")
                                    st.plotly_chart(fig_weights, use_container_width=True, key=f"lime_weights_{selected_idx}")
                                    
                                    # Horizontal bar chart (alternative view)
                                    st.write("##### Feature Impact (Horizontal)")
                                    fig_horizontal = px.bar(
                                        importance_df,
                                        x='Weight',
                                        y='Feature',
                                        orientation='h',
                                        title=f"LIME Feature Impact for Anomaly {selected_idx}",
                                        color='Weight',
                                        color_continuous_scale='RdBu_r'
                                    )
                                    fig_horizontal.add_vline(x=0, line_dash="dash", line_color="black")
                                    st.plotly_chart(fig_horizontal, use_container_width=True, key=f"lime_horizontal_{selected_idx}")
                                    
                                    # Pie chart for absolute importance
                                    st.write("##### Absolute Feature Importance")
                                    importance_df['Abs_Weight'] = importance_df['Weight'].abs()
                                    if importance_df['Abs_Weight'].sum() > 0:
                                        fig_pie = px.pie(
                                            importance_df,
                                            values='Abs_Weight',
                                            names='Feature',
                                            title=f"Feature Contribution Distribution for Anomaly {selected_idx}"
                                        )
                                        st.plotly_chart(fig_pie, use_container_width=True, key=f"lime_pie_{selected_idx}")
                                    
                                    # Data table with feature values
                                    st.write("##### Feature Analysis Table")
                                    analysis_df = importance_df.copy()
                                    analysis_df['Feature_Value'] = [df.loc[selected_idx, feat] for feat in analysis_df['Feature']]
                                    analysis_df['Abs_Weight'] = analysis_df['Weight'].abs()
                                    st.dataframe(analysis_df, use_container_width=True)
                                
                                # Raw LIME explanation rules
                                if lime_explanation is not None and hasattr(lime_explanation, "as_list"):
                                    st.write("##### LIME Rules and Conditions")
                                    explanation_data = lime_explanation.as_list()
                                    if explanation_data:
                                        rules_df = pd.DataFrame(explanation_data, columns=["Rule", "Weight"])
                                        rules_df['Abs_Weight'] = rules_df['Weight'].abs()
                                        rules_df = rules_df.sort_values('Abs_Weight', ascending=False)
                                        
                                        # Show top rules
                                        st.write("**Top Rules:**")
                                        st.dataframe(rules_df.head(10), use_container_width=True)
                                        
                                        # Rules bar chart
                                        fig_rules = px.bar(
                                            rules_df.head(10),
                                            x=range(len(rules_df.head(10))),
                                            y='Weight',
                                            title="Top 10 LIME Rules by Weight",
                                            color='Weight',
                                            color_continuous_scale='RdBu_r',
                                            hover_data=['Rule']
                                        )
                                        fig_rules.update_layout(xaxis_title="Rule Index")
                                        fig_rules.add_hline(y=0, line_dash="dash", line_color="black")
                                        st.plotly_chart(fig_rules, use_container_width=True, key=f"lime_rules_{selected_idx}")
                                
                                # Model prediction info
                                st.write("##### Prediction Details")
                                instance_score = local_explanation.get("score", "N/A")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Anomaly Score", f"{instance_score:.4f}" if instance_score != "N/A" else "N/A")
                                with col2:
                                    if instance_score != "N/A":
                                        prediction_class = "Anomaly" if instance_score > threshold else "Normal"
                                        st.metric("Prediction", prediction_class)
                                        
                            except Exception as e:
                                st.error(f"Error creating LIME visualizations: {str(e)}")
                            
            except Exception as e:
                st.error(f"Error explaining anomaly: {str(e)}")
                logger.exception(f"Error explaining anomaly: {str(e)}")
                
                # Fallback to a simpler explanation
                st.write("#### Feature Values Compared to Normal Data")
                
                # Compare this anomaly's values to the average values
                non_anomaly_indices = np.where(scores <= threshold)[0]
                normal_data = df.iloc[non_anomaly_indices][features] if len(non_anomaly_indices) > 0 else pd.DataFrame()
                
                if not normal_data.empty:
                    # Calculate means and standard deviations for normal data
                    normal_means = normal_data.mean()
                    normal_stds = normal_data.std()
                    
                    # Calculate z-scores for the anomaly
                    anomaly_values = df.loc[selected_idx, features]
                    z_scores = (anomaly_values - normal_means) / normal_stds.replace(0, 1)  # Avoid division by zero
                    
                    # Create comparison dataframe
                    comparison = pd.DataFrame({
                        'Feature': features,
                        'Anomaly Value': anomaly_values.values,
                        'Normal Mean': normal_means.values,
                        'Z-Score': z_scores.values
                    })
                    
                    # Sort by absolute z-score
                    comparison['Abs Z-Score'] = np.abs(comparison['Z-Score'])
                    comparison = comparison.sort_values('Abs Z-Score', ascending=False)
                    
                    # Plot the comparison
                    fig = px.bar(
                        comparison,
                        x='Feature',
                        y='Z-Score',
                        color='Abs Z-Score',
                        color_continuous_scale='RdBu_r',
                        title="Deviation from Normal (Z-Scores)",
                        hover_data=['Anomaly Value', 'Normal Mean']
                    )
                    # Add unique key to plotly_chart
                    st.plotly_chart(fig, use_container_width=True, key="zscore_comparison")
                    
                    # Display the comparison table
                    st.dataframe(comparison[['Feature', 'Anomaly Value', 'Normal Mean', 'Z-Score']], use_container_width=True)
                else:
                    st.info("Not enough normal data to create comparison.")
        else:
            st.info("No anomalies detected to explain.")