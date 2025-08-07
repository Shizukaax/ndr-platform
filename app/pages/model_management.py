"""
Model Management page for the Network Anomaly Detection Platform.
Allows users to view, manage, apply, compare and retrain models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import logging
import time
import tempfile
import pickle
from datetime import datetime
from typing import Dict, List, Any

from core.model_manager import ModelManager
from app.components.error_handler import handle_error, validate_inputs, is_dataframe
from app.components.data_source_selector import ensure_data_available, show_compact_data_status
from app.components.visualization import (
    plot_anomaly_scores, plot_anomaly_scatter, plot_anomaly_timeline,
    plot_feature_importance, plot_network_graph, find_timestamp_column
)
from app.components.model_cards import (
    generate_model_card, plot_to_base64, extract_parameter_data
)
from app.components.model_comparison import (
    normalize_scores, calculate_optimal_threshold, create_score_distribution_plot,
    analyze_model_overlap, create_overlap_matrix_plot, provide_expert_analysis
)
from app.components.model_retraining import (
    get_default_parameters, create_parameter_inputs, retrain_model,
    preprocess_data_for_training
)

# Get logger
logger = logging.getLogger("streamlit_app")

def fix_model_thresholds_ui(model_manager):
    """
    Create UI for fixing model thresholds.
    
    Args:
        model_manager: The model manager instance
    """
    with st.expander("Model Threshold Tools", expanded=False):
        st.write("Fix problematic model thresholds that might cause too many anomalies to be flagged.")
        st.info("Use this if your models are flagging too many or too few anomalies.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Get model types for selection
            model_types = model_manager.list_models()
            
            selected_model = st.selectbox(
                "Select model to fix threshold",
                options=["All models"] + model_types,
                index=0,
                key="fix_threshold_model_select"
            )
        
        with col2:
            if st.button("Fix Threshold", key="fix_threshold_button"):
                try:
                    model_type = None if selected_model == "All models" else selected_model
                    
                    success = model_manager.fix_model_thresholds(model_type)
                    
                    if success:
                        st.success(f"Successfully fixed thresholds for {selected_model}.")
                        st.info("Reload the page to see the updated threshold values.")
                        
                        # Add reload button
                        if st.button("Reload Now", key="reload_after_fix"):
                            st.rerun()
                    else:
                        st.error(f"Failed to fix thresholds for {selected_model}.")
                except Exception as e:
                    st.error(f"Error fixing thresholds: {str(e)}")
                    logging.exception(f"Error fixing thresholds: {str(e)}")

@handle_error
def show_model_management():
    """Display the Model Management page"""
    
    st.header("Model Management")
    
    # Initialize model manager
    model_manager = ModelManager()
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Saved Models", "Apply to New Data"])
    
    # Saved Models Tab
    with tab1:
        st.subheader("Saved Models")
        
        # Get all saved models
        models = model_manager.list_models()
        
        if not models:
            st.info("No models found. Train a model in the Anomaly Detection page.")
            return
        
        # Refresh button
        col1, col2 = st.columns([0.85, 0.15])
        with col2:
            if st.button("🔄 Refresh", key="refresh_models"):
                st.rerun()
                
        # Display models in a grid
        num_models = len(models)
        st.write(f"Found {num_models} saved models.")
        
        # Add threshold fixing UI
        fix_model_thresholds_ui(model_manager)
        
        # Display models in a table for easy comparison
        # Get detailed model information for the table
        detailed_models = model_manager.list_models()
        model_data = extract_parameter_data(detailed_models)
        # Convert 'N/A' strings to actual N/A values to avoid Arrow serialization issues
        for col in model_data.columns:
            model_data[col] = model_data[col].apply(lambda x: np.nan if x == 'N/A' else x)
        st.dataframe(model_data, use_container_width=True)
        
        # Visual grid of models
        st.subheader("Model Cards")
        
        # Create model cards in a grid layout
        cols = st.columns(3)
        for i, model in enumerate(models):
            with cols[i % 3]:
                st.markdown(generate_model_card(model), unsafe_allow_html=True)
                
                # Add buttons for this model
                model_type = model.get("type", "Unknown")
                
                # Action buttons for each model
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Details", key=f"details_{model_type}"):
                        # Set session state for model details
                        st.session_state.selected_model_type = model_type
                        st.session_state.show_model_details = True
                
                with col2:
                    if st.button("Delete", key=f"delete_{model_type}"):
                        # Set session state for model deletion
                        st.session_state.delete_model_type = model_type
                        st.session_state.confirm_delete = True
        
        # Model details dialog
        if hasattr(st.session_state, 'show_model_details') and st.session_state.show_model_details:
            selected_model_type = st.session_state.selected_model_type
            selected_model_info = next((model for model in models if model["type"] == selected_model_type), None)
            
            if selected_model_info:
                with st.expander("Model Details", expanded=True):
                    st.write(f"## {selected_model_type} Model")
                    st.write(f"**Path:** {selected_model_info['path']}")
                    st.write(f"**Size:** {selected_model_info['size_mb']:.2f} MB")
                    st.write(f"**Last Modified:** {selected_model_info['last_modified']}")
                    
                    # Display metadata
                    metadata = selected_model_info.get("metadata", {})
                    if metadata:
                        st.subheader("Metadata")
                        
                        # Show features
                        feature_names = metadata.get("feature_names", [])
                        if feature_names:
                            st.write(f"**Features ({len(feature_names)}):**")
                            st.write(", ".join(feature_names))
                        
                        # Show threshold
                        if "anomaly_threshold" in metadata:
                            st.write(f"**Anomaly Threshold:** {metadata['anomaly_threshold']}")
                        
                        # Show training time
                        if "trained_at" in metadata:
                            st.write(f"**Trained At:** {metadata['trained_at']}")
                        
                        # Show parameters
                        params = metadata.get("parameters", {})
                        if params:
                            st.subheader("Model Parameters")
                            if isinstance(params, dict):
                                for param, value in params.items():
                                    st.write(f"**{param}:** {value}")
                            else:
                                st.write(str(params))
                    
                    # Download and delete buttons
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Get model path
                        model_path = selected_model_info["path"]
                        
                        # Read model file as binary
                        with open(model_path, "rb") as f:
                            model_bytes = f.read()
                        
                        # Get model filename from path
                        model_filename = os.path.basename(model_path)
                        
                        # Offer download
                        st.download_button(
                            label="Download Model",
                            data=model_bytes,
                            file_name=model_filename,
                            mime="application/octet-stream",
                            key="download_model_details"
                        )
                    
                    with col2:
                        if st.button("Delete Model", key="delete_model_details"):
                            # Confirm deletion
                            st.session_state.delete_model_type = selected_model_type
                            st.session_state.confirm_delete = True
                    
                    # Close button
                    if st.button("Close", key="close_details"):
                        st.session_state.show_model_details = False
                        st.rerun()
        
        # Deletion confirmation dialog
        if hasattr(st.session_state, 'confirm_delete') and st.session_state.confirm_delete:
            delete_model_type = st.session_state.delete_model_type
            
            st.warning(f"Are you sure you want to delete the {delete_model_type} model?")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, Delete", key="confirm_delete_button"):
                    # Delete model
                    success = model_manager.delete_model(delete_model_type)
                    
                    if success:
                        st.success(f"{delete_model_type} model deleted successfully.")
                        # Reset session state
                        st.session_state.confirm_delete = False
                        if hasattr(st.session_state, 'delete_model_type'):
                            delattr(st.session_state, 'delete_model_type')
                        st.rerun()
                    else:
                        st.error(f"Failed to delete {delete_model_type} model.")
            
            with col2:
                if st.button("Cancel", key="cancel_delete_button"):
                    # Reset session state
                    st.session_state.confirm_delete = False
                    if hasattr(st.session_state, 'delete_model_type'):
                        delattr(st.session_state, 'delete_model_type')
                    st.rerun()
        
        # Upload new model section
        with st.expander("Upload Model", expanded=False):
            st.write("Upload a previously exported model file.")
            
            uploaded_file = st.file_uploader("Choose a model file", type=["pkl"])
            
            if uploaded_file is not None:
                # Save the uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Try to load the model to verify it's valid
                    with open(tmp_path, "rb") as f:
                        model = pickle.load(f)
                    
                    # Check if it's a valid model
                    if hasattr(model, 'predict') and callable(getattr(model, 'predict')):
                        # Get model type
                        if hasattr(model, 'metadata') and 'type' in model.metadata:
                            model_type = model.metadata['type']
                        else:
                            model_type = model.__class__.__name__.replace("Detector", "")
                        
                        # Save the model
                        save_path = model_manager.save_model(model)
                        
                        st.success(f"Model {model_type} uploaded and saved successfully to {save_path}")
                        
                        # Offer to refresh
                        if st.button("Refresh Model List", key="refresh_after_upload"):
                            st.rerun()
                    else:
                        st.error("The uploaded file is not a valid anomaly detection model.")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
    
    # Apply to New Data Tab
    with tab2:
        st.subheader("Apply Model to New Data")
        
        # Show current data source status
        show_compact_data_status()
        
        # Ensure data is available
        if not ensure_data_available():
            return
        
        # Get the combined data
        df = st.session_state.combined_data
        
        # Display data summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Records", len(df))
        with col2:
            if 'ip_src' in df.columns:
                st.metric("Unique Source IPs", df['ip_src'].nunique())
            elif 'ip.src' in df.columns:
                st.metric("Unique Source IPs", df['ip.src'].nunique())
        with col3:
            if 'ip_dst' in df.columns:
                st.metric("Unique Destination IPs", df['ip_dst'].nunique())
            elif 'ip.dst' in df.columns:
                st.metric("Unique Destination IPs", df['ip.dst'].nunique())
        
        # Select model to apply
        if not models:
            st.warning("No models found. Train a model in the Anomaly Detection page.")
            return
        
        # Select a model to apply
        model_types = [model["type"] for model in models]
        selected_model_type = st.selectbox("Select model to apply", model_types, key="apply_model_select")
        
        # Get selected model info
        selected_model_info = next((model for model in models if model["type"] == selected_model_type), None)
        
        if selected_model_info:
            # Display selected model details
            metadata = selected_model_info.get("metadata", {})
            feature_names = metadata.get("feature_names", [])
            threshold = metadata.get("anomaly_threshold", "Auto")
            
            # Display model info
            with st.expander("Model Information", expanded=True):
                st.write(f"**Model Type:** {selected_model_type}")
                st.write(f"**Saved Threshold:** {threshold}")
                
                # Algorithm-specific info
                parameters = metadata.get("parameters", {})
                if isinstance(parameters, dict):
                    st.write("**Algorithm Parameters:**")
                    for param, value in parameters.items():
                        if param != "models":  # Skip detailed model list for ensembles
                            st.write(f"- {param}: {value}")
                
                if feature_names:
                    st.write(f"**Required Features ({len(feature_names)}):**")
                    
                    # Check which features are available in the dataset
                    available_features = []
                    missing_features = []
                    
                    for feature in feature_names:
                        if feature in df.columns:
                            available_features.append(feature)
                        else:
                            missing_features.append(feature)
                    
                    # Show availability status
                    if missing_features:
                        st.error(f"Missing features: {', '.join(missing_features)}")
                        st.info("You may need to use the 'Prepare Protocol-Agnostic Features' button in the Anomaly Detection page.")
                    else:
                        st.success(f"All {len(feature_names)} required features are available in the dataset.")
                    
                    # Show feature list
                    feature_df = pd.DataFrame({
                        "Feature": feature_names,
                        "Available": [feature in df.columns for feature in feature_names]
                    })
                    st.dataframe(feature_df, use_container_width=True)
            
            # Data preprocessing options
            with st.expander("Data Preprocessing Options", expanded=True):
                handle_missing = st.radio(
                    "Handle missing values",
                    options=["Fill missing values with mean", "Fill missing values with median", 
                             "Fill missing values with 0", "Drop rows with missing values"],
                    index=0,
                    key="apply_handle_missing"  # Added unique key
                )
                
                normalize_data = st.checkbox("Normalize data", value=True, key="apply_normalize")
                
                recalculate_threshold = st.checkbox(
                    "Recalculate anomaly threshold", 
                    value=False,
                    help="If checked, will calculate a new threshold based on the current data instead of using the saved threshold",
                    key="apply_recalculate_threshold"
                )
                
                if recalculate_threshold:
                    # Allow custom contamination if recalculating threshold
                    contamination = st.slider(
                        "Contamination (expected proportion of anomalies)",
                        min_value=0.001,
                        max_value=0.2,
                        value=0.01,
                        step=0.001,
                        format="%.3f",
                        key="apply_contamination"
                    )
                else:
                    contamination = 0.01  # Default value
                
                save_results = st.checkbox(
                    "Save results to disk", 
                    value=True,
                    help="If checked, will save the analysis results to the results directory",
                    key="apply_save_results"
                )
            
            # Check if we can proceed
            can_proceed = len(missing_features) == 0
            
            # Button to apply model
            if can_proceed:
                if st.button("Run Anomaly Detection", key="apply_model_button", type="primary"):
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Show progress
                        status_text.text("Preprocessing data...")
                        progress_bar.progress(20)
                        
                        # Preprocess data
                        X = preprocess_data_for_training(df, feature_names, handle_missing)
                        
                        # Apply model to data
                        status_text.text(f"Applying {selected_model_type} model to data...")
                        progress_bar.progress(50)
                        
                        # Apply model with options
                        results = model_manager.apply_model_to_data(
                            selected_model_type, 
                            X, 
                            feature_names, 
                            recalculate_threshold=recalculate_threshold,
                            custom_contamination=contamination if recalculate_threshold else None,
                            save_results=save_results
                        )
                        
                        # Show completion
                        progress_bar.progress(100)
                        status_text.text("Analysis complete!")
                        
                        # Display results
                        anomaly_count = len(results['anomalies'])
                        total_count = len(X)
                        anomaly_percent = (anomaly_count / total_count) * 100 if total_count > 0 else 0
                        
                        # Show metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Records", total_count)
                        
                        with col2:
                            st.metric("Anomalies", anomaly_count)
                        
                        with col3:
                            st.metric("Anomaly %", f"{anomaly_percent:.2f}%")
                        
                        # Show threshold used
                        threshold_type = "saved" if results.get('used_saved_threshold', False) else "calculated"
                        st.info(f"Used {threshold_type} threshold: {results['threshold']:.4f}")
                        
                        # Plot anomaly scores
                        st.subheader("Anomaly Score Distribution")
                        fig = plot_anomaly_scores(X, results['scores'], results['threshold'])
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Store results in session state for other tabs
                        st.session_state.current_results = results
                        st.session_state.selected_model_type = selected_model_type
                        st.session_state.analyzed_data = X
                        
                        # Show anomalies if any were found
                        if anomaly_count > 0:
                            st.success(f"Found {anomaly_count} anomalies. View them in the table below.")
                            
                            # Show anomalies table
                            st.subheader("Detected Anomalies")
                            anomalies = results['anomalies']
                            
                            # Determine which columns to display
                            display_cols = ['anomaly_score']
                            
                            # Add useful columns if available
                            cols_to_check = [
                                'ip_src', 'ip.src', 'ip_dst', 'ip.dst',
                                '_ws_col_Protocol', '_ws.col_Protocol', 'protocol',
                                'frame.len', 'packet_length', '_ws_col_Info'
                            ]
                            
                            # Add timestamp if available
                            time_col, _ = find_timestamp_column(anomalies)
                            if time_col:
                                display_cols.insert(0, time_col)
                            
                            # Add other informative columns
                            for col in cols_to_check:
                                if col in anomalies.columns:
                                    display_cols.append(col)
                            
                            # If we have only the score, add all features
                            if len(display_cols) <= 1:
                                display_cols = ['anomaly_score'] + feature_names
                            
                            # Sort by anomaly score (descending)
                            anomalies_sorted = anomalies.sort_values('anomaly_score', ascending=False)
                            
                            # Show the table
                            st.dataframe(anomalies_sorted[display_cols], use_container_width=True)
                            
                            # Provide download options
                            col1, col2 = st.columns(2)
                            with col1:
                                csv = anomalies.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    "Download Anomalies as CSV",
                                    data=csv,
                                    file_name=f"{selected_model_type}_anomalies.csv",
                                    mime="text/csv",
                                    key="download_anomalies_csv"
                                )
                                
                            with col2:
                                json_str = anomalies.to_json(orient='records', date_format='iso')
                                st.download_button(
                                    "Download Anomalies as JSON",
                                    data=json_str,
                                    file_name=f"{selected_model_type}_anomalies.json",
                                    mime="application/json",
                                    key="download_anomalies_json"
                                )
                            
                        else:
                            st.success("No anomalies detected in this dataset.")
                        
                    except Exception as e:
                        st.error(f"Error applying model: {str(e)}")
                        logger.exception(f"Error applying model: {str(e)}")
            else:
                st.error("Cannot proceed: missing required features for this model.")
                st.info("Please prepare the data with the required features first using the 'Prepare Protocol-Agnostic Features' button in the Anomaly Detection page.")