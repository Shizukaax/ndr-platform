"""
Model Comparison page for the Network Anomaly Detection Platform.
Compares performance of multiple anomaly detection models on the same dataset.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
from typing import List, Dict, Any

from core.models.isolation_forest import IsolationForestDetector
from core.models.local_outlier_factor import LocalOutlierFactorDetector
from core.models.one_class_svm import OneClassSVMDetector
from core.models.dbscan import DBSCANDetector
from core.models.knn import KNNDetector
from core.models.ensemble import EnsembleDetector
from core.config_loader import load_config
from core.model_manager import ModelManager

try:
    from core.models.hdbscan import HDBSCANDetector
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

from app.components.visualization import (
    plot_anomaly_scores, 
    plot_anomaly_scatter,
    find_timestamp_column
)
from app.components.error_handler import handle_error

# Set up logger
logger = logging.getLogger("streamlit_app")

def get_default_features(df: pd.DataFrame) -> List[str]:
    """
    Get all available numeric features for model comparison.
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
    logger.info(f"Found {len(available_features)} numeric features for model comparison: {available_features}")
    
    return available_features

@handle_error
def show_model_comparison():
    """Display the Model Comparison page."""
    
    st.header("Model Comparison")
    
    # Check if data is loaded
    if not st.session_state.get('combined_data') is not None:
        st.info("No data loaded. Please go to the Data Upload page to select JSON files.")
        return
    
    # Get the combined data
    df = st.session_state.combined_data
    
    # Feature selection
    st.subheader("Select Features for Comparison")
    
    # Get only numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if not numeric_cols:
        st.error("No numeric features found in the dataset. Please ensure your data contains numeric values.")
        st.info("Try using the 'Prepare Protocol-Agnostic Features' button in the Anomaly Detection page.")
        return
    
    # Get all available numeric features as default
    default_features = get_default_features(df)
    
    # Display info about available features
    st.info(f"Found {len(numeric_cols)} numeric features in your data. All are selected by default - uncheck any you don't want to use for comparison.")
    
    # Allow selecting features for model comparison
    selected_features = st.multiselect(
        "Select features for model comparison",
        options=numeric_cols,
        default=default_features,
        help="All numeric features are selected by default. Uncheck any you don't want to include in the comparison."
    )
    
    # Show what features are actually available in the data
    with st.expander("📊 Available Features in Your Data"):
        feature_info = []
        for feature in numeric_cols:
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
    
    if not selected_features:
        st.warning("Please select at least one feature for model comparison.")
        return
    
    # Model selection
    st.subheader("Select Models to Compare")
    
    # Load configuration
    config = load_config()
    
    # Define model options with optimal contamination for synthetic data (33% anomalies)
    model_options = [
        {"name": "Isolation Forest", "selected": True, "params": {
            "contamination": 0.33, 
            "n_estimators": 100
        }},
        {"name": "Local Outlier Factor", "selected": True, "params": {
            "n_neighbors": 20,
            "contamination": 0.33
        }},
        {"name": "One-Class SVM", "selected": True, "params": {
            "nu": 0.33,
            "kernel": "rbf"
        }},
        {"name": "DBSCAN", "selected": True, "params": {
            "eps": 0.5,
            "min_samples": 5,
            "contamination": 0.33
        }},
        {"name": "KNN", "selected": True, "params": {
            "n_neighbors": 5,
            "contamination": 0.33
        }}
    ]
    
    # Add HDBSCAN if available
    if HDBSCAN_AVAILABLE:
        model_options.append({
            "name": "HDBSCAN", 
            "selected": False, 
            "params": {
                "min_cluster_size": 5,
                "min_samples": 5,
                "contamination": 0.33
            }
        })
    
    # Add Ensemble model
    model_options.append({
        "name": "Ensemble", 
        "selected": False,
        "params": {
            "combination_method": "weighted_average",
            "contamination": 0.33
        }
    })
    
    # Allow user to select which models to compare
    selected_models = []
    for i, model in enumerate(model_options):
        col1, col2 = st.columns([1, 3])
        with col1:
            model["selected"] = st.checkbox(f"Use {model['name']}", value=model["selected"])
        
        if model["selected"]:
            with col2:
                if model["name"] == "Isolation Forest":
                    model["params"]["contamination"] = st.slider(
                        f"{model['name']} Contamination", 
                        min_value=0.01, 
                        max_value=0.3, 
                        value=model["params"]["contamination"],
                        key=f"if_contamination_{i}"
                    )
                    model["params"]["n_estimators"] = st.slider(
                        f"{model['name']} n_estimators", 
                        min_value=50, 
                        max_value=300, 
                        value=model["params"]["n_estimators"],
                        key=f"if_n_estimators_{i}"
                    )
                
                elif model["name"] == "Local Outlier Factor":
                    model["params"]["contamination"] = st.slider(
                        f"{model['name']} Contamination", 
                        min_value=0.01, 
                        max_value=0.3, 
                        value=model["params"]["contamination"],
                        key=f"lof_contamination_{i}"
                    )
                    model["params"]["n_neighbors"] = st.slider(
                        f"{model['name']} n_neighbors", 
                        min_value=5, 
                        max_value=50, 
                        value=model["params"]["n_neighbors"],
                        key=f"lof_n_neighbors_{i}"
                    )
                
                elif model["name"] == "One-Class SVM":
                    model["params"]["nu"] = st.slider(
                        f"{model['name']} nu", 
                        min_value=0.01, 
                        max_value=0.3, 
                        value=model["params"]["nu"],
                        key=f"ocsvm_nu_{i}"
                    )
                    model["params"]["kernel"] = st.selectbox(
                        f"{model['name']} kernel",
                        options=["rbf", "linear", "poly", "sigmoid"],
                        index=0,
                        key=f"ocsvm_kernel_{i}"
                    )
                
                elif model["name"] == "DBSCAN":
                    model["params"]["eps"] = st.slider(
                        f"{model['name']} eps", 
                        min_value=0.1, 
                        max_value=2.0, 
                        value=model["params"]["eps"],
                        key=f"dbscan_eps_{i}"
                    )
                    model["params"]["min_samples"] = st.slider(
                        f"{model['name']} min_samples", 
                        min_value=2, 
                        max_value=20, 
                        value=model["params"]["min_samples"],
                        key=f"dbscan_min_samples_{i}"
                    )
                    model["params"]["contamination"] = st.slider(
                        f"{model['name']} Contamination (for threshold)", 
                        min_value=0.01, 
                        max_value=0.3, 
                        value=model["params"]["contamination"],
                        key=f"dbscan_contamination_{i}"
                    )
                
                elif model["name"] == "KNN":
                    model["params"]["n_neighbors"] = st.slider(
                        f"{model['name']} n_neighbors", 
                        min_value=1, 
                        max_value=50, 
                        value=model["params"]["n_neighbors"],
                        key=f"knn_n_neighbors_{i}"
                    )
                    model["params"]["contamination"] = st.slider(
                        f"{model['name']} Contamination", 
                        min_value=0.01, 
                        max_value=0.3, 
                        value=model["params"]["contamination"],
                        key=f"knn_contamination_{i}"
                    )
                
                elif model["name"] == "HDBSCAN":
                    model["params"]["min_cluster_size"] = st.slider(
                        f"{model['name']} min_cluster_size", 
                        min_value=2, 
                        max_value=20, 
                        value=model["params"]["min_cluster_size"],
                        key=f"hdbscan_min_cluster_size_{i}"
                    )
                    model["params"]["min_samples"] = st.slider(
                        f"{model['name']} min_samples", 
                        min_value=1, 
                        max_value=20, 
                        value=model["params"]["min_samples"],
                        key=f"hdbscan_min_samples_{i}"
                    )
                    model["params"]["contamination"] = st.slider(
                        f"{model['name']} Contamination (for threshold)", 
                        min_value=0.01, 
                        max_value=0.3, 
                        value=model["params"]["contamination"],
                        key=f"hdbscan_contamination_{i}"
                    )
                
                elif model["name"] == "Ensemble":
                    model["params"]["contamination"] = st.slider(
                        f"{model['name']} Contamination", 
                        min_value=0.01, 
                        max_value=0.3, 
                        value=model["params"]["contamination"],
                        key=f"ensemble_contamination_{i}"
                    )
                    model["params"]["combination_method"] = st.selectbox(
                        f"{model['name']} Combination Method",
                        options=["weighted_average", "average", "max", "majority_vote"],
                        index=0,
                        key=f"ensemble_method_{i}"
                    )
            
            selected_models.append(model)
    
    if not any(model["selected"] for model in model_options):
        st.warning("Please select at least one model for comparison.")
        return
    
    # Data preprocessing
    st.subheader("Data Preprocessing")
    
    # Handle missing values options
    handle_missing = st.radio(
        "Handle missing values",
        options=["Fill missing values with mean", "Fill missing values with median", "Fill missing values with 0", "Drop rows with missing values"],
        index=0
    )
    
    # Normalize data
    normalize_data = st.checkbox("Normalize data", value=True)
    
    # Run comparison button
    if st.button("Run Model Comparison"):
        if not selected_features:
            st.error("Please select at least one feature.")
            return
        
        # Extract selected data
        X = df[selected_features].copy()
        
        # Check for missing values
        missing_count = X.isna().sum().sum()
        if missing_count > 0:
            st.warning(f"Found {missing_count} missing values in selected features.")
            
            # Show which columns have missing values
            missing_by_col = X.isna().sum()
            missing_cols = missing_by_col[missing_by_col > 0]
            if not missing_cols.empty:
                st.write("Missing values by column:")
                st.write(missing_cols)
        
        # Handle missing values
        if handle_missing == "Drop rows with missing values":
            rows_before = len(X)
            X = X.dropna()
            rows_after = len(X)
            rows_dropped = rows_before - rows_after
            
            if rows_dropped > 0:
                st.info(f"Dropped {rows_dropped} rows with missing values. {rows_after} rows remaining.")
            
            if len(X) == 0:
                st.error("No data remains after dropping missing values. Try a different approach.")
                return
        elif handle_missing == "Fill missing values with mean":
            X = X.fillna(X.mean())
            # If there are still NaNs (columns with all NaNs), fill with 0
            if X.isna().any().any():
                X = X.fillna(0)
        elif handle_missing == "Fill missing values with median":
            X = X.fillna(X.median())
            # If there are still NaNs, fill with 0
            if X.isna().any().any():
                X = X.fillna(0)
        elif handle_missing == "Fill missing values with 0":
            X = X.fillna(0)
        
        # Normalize data if requested
        if normalize_data:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Progress bar and status
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize models and results storage
        models = []
        results = {}
        all_scores = {}
        training_times = {}
        prediction_times = {}
        all_anomalies = {}
        
        # Count total models
        total_models = sum(1 for model in model_options if model["selected"])
        
        try:
            # Initialize and train each selected model
            for i, model_option in enumerate([m for m in model_options if m["selected"]]):
                model_name = model_option["name"]
                params = model_option["params"]
                
                # Update status
                progress_value = (i / total_models) * 0.5  # First half for initialization
                progress_bar.progress(progress_value)
                status_text.text(f"Initializing {model_name}...")
                
                # Initialize the appropriate model
                if model_name == "Isolation Forest":
                    model = IsolationForestDetector(**params)
                elif model_name == "Local Outlier Factor":
                    model = LocalOutlierFactorDetector(**params)
                elif model_name == "One-Class SVM":
                    model = OneClassSVMDetector(**params)
                elif model_name == "DBSCAN":
                    model = DBSCANDetector(**params)
                elif model_name == "KNN":
                    model = KNNDetector(**params)
                elif model_name == "HDBSCAN" and HDBSCAN_AVAILABLE:
                    model = HDBSCANDetector(**params)
                elif model_name == "Ensemble":
                    # For ensemble, create component models
                    component_models = []
                    component_models.append(IsolationForestDetector(contamination=params["contamination"]))
                    component_models.append(LocalOutlierFactorDetector(contamination=params["contamination"]))
                    component_models.append(KNNDetector(contamination=params["contamination"]))
                    
                    # Create ensemble model
                    params_copy = params.copy()
                    params_copy["models"] = component_models
                    model = EnsembleDetector(**params_copy)
                else:
                    continue  # Skip if model not recognized
                
                # Set feature names before training
                model.feature_names = selected_features
                
                # Train the model
                status_text.text(f"Training {model_name}...")
                start_time = time.time()
                model.fit(X)
                training_time = time.time() - start_time
                
                # Predict anomaly scores
                status_text.text(f"Predicting with {model_name}...")
                start_time = time.time()
                scores = model.predict(X)
                prediction_time = time.time() - start_time
                
                # Calculate threshold based on percentile
                if hasattr(model, 'contamination'):
                    contamination = model.contamination
                elif model_name == "One-Class SVM" and hasattr(model, 'nu'):
                    contamination = model.nu
                else:
                    contamination = params.get("contamination", 0.33)
                
                threshold = np.percentile(scores, 100 * (1 - contamination))
                
                # Store results
                models.append(model)
                results[model_name] = {
                    "model": model,
                    "scores": scores,
                    "threshold": threshold
                }
                all_scores[model_name] = scores
                training_times[model_name] = training_time
                prediction_times[model_name] = prediction_time
                
                # Identify anomalies for this model
                anomalies = X.index[scores > threshold].tolist()
                all_anomalies[model_name] = anomalies
                
                # Update progress
                progress_value = 0.5 + (i + 1) / total_models * 0.5  # Second half for training/prediction
                progress_bar.progress(progress_value)
            
            # Complete progress
            progress_bar.progress(1.0)
            status_text.text("Model comparison completed!")
            
            # Store results in session state for later use
            st.session_state.model_comparison_results = {
                "models": results,
                "all_scores": all_scores,
                "training_times": training_times,
                "prediction_times": prediction_times,
                "all_anomalies": all_anomalies,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "trained_models": models,  # Store actual model objects
                "selected_features": selected_features
            }
            
            # Show results
            st.subheader("Model Comparison Results")
            
            # Create tabs for different result views
            tab1, tab2, tab3, tab4 = st.tabs([
                "Score Distribution", "Performance Metrics", "Training Time", "Anomaly Agreement"
            ])
            
            # Tab 1: Score Distribution
            with tab1:
                st.subheader("Anomaly Score Distribution by Model")
                
                # Create a DataFrame with all scores for plotting
                scores_df = pd.DataFrame({
                    model_name: all_scores[model_name] for model_name in all_scores.keys()
                })
                
                # Boxplot of score distributions
                fig = px.box(scores_df, title="Anomaly Score Distribution by Model")
                st.plotly_chart(fig, use_container_width=True)
                
                # Score distributions as KDE plots
                fig = plt.figure(figsize=(10, 6))
                for model_name, scores in all_scores.items():
                    sns.kdeplot(scores, label=model_name)
                
                plt.title("Anomaly Score Density by Model")
                plt.xlabel("Anomaly Score")
                plt.ylabel("Density")
                plt.legend()
                st.pyplot(fig)
                
                # Individual model score histograms
                selected_model = st.selectbox(
                    "Select model to view score distribution",
                    options=list(all_scores.keys()),
                    key="select_model_histogram"
                )
                
                if selected_model:
                    fig = px.histogram(
                        x=all_scores[selected_model],
                        nbins=50,
                        title=f"Anomaly Score Distribution: {selected_model}",
                        labels={"x": "Anomaly Score", "y": "Count"}
                    )
                    
                    # Add threshold line
                    threshold = results[selected_model]["threshold"]
                    fig.add_vline(
                        x=threshold,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Threshold: {threshold:.3f}",
                        annotation_position="top right"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show top anomalies for selected model
                    st.subheader(f"Top 10 Anomalies: {selected_model}")
                    
                    # Get indices of top anomalies
                    scores_array = all_scores[selected_model]
                    top_indices = np.argsort(scores_array)[-10:][::-1]
                    
                    # Create a table with original data and anomaly scores
                    top_anomalies = df.iloc[top_indices].copy()
                    top_anomalies["anomaly_score"] = scores_array[top_indices]
                    
                    # Display the table
                    st.dataframe(top_anomalies, use_container_width=True)
            
            # Tab 2: Performance Metrics
            with tab2:
                st.subheader("Performance Comparison")
                
                # Create table with performance metrics
                metrics = []
                
                for model_name in all_scores.keys():
                    # Get anomaly count
                    anomaly_count = len(all_anomalies[model_name])
                    anomaly_percent = anomaly_count / len(X) * 100
                    
                    # Get threshold
                    threshold = results[model_name]["threshold"]
                    
                    # Add to metrics
                    metrics.append({
                        "Model": model_name,
                        "Anomalies Detected": anomaly_count,
                        "Anomaly %": f"{anomaly_percent:.2f}%",
                        "Threshold": f"{threshold:.3f}"
                    })
                
                # Display metrics as a table
                metrics_df = pd.DataFrame(metrics)
                st.dataframe(metrics_df, use_container_width=True)
                
                # Plot anomaly counts as a bar chart
                fig = px.bar(
                    metrics_df,
                    x="Model",
                    y="Anomalies Detected",
                    title="Number of Anomalies Detected by Model"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Tab 3: Training Time
            with tab3:
                st.subheader("Training and Prediction Time")
                
                # Create table with timing information
                timing = []
                
                for model_name in all_scores.keys():
                    # Add to timing
                    timing.append({
                        "Model": model_name,
                        "Training Time (s)": f"{training_times[model_name]:.3f}",
                        "Prediction Time (s)": f"{prediction_times[model_name]:.3f}",
                        "Training Time (ms)": training_times[model_name] * 1000,
                        "Prediction Time (ms)": prediction_times[model_name] * 1000
                    })
                
                # Display timing as a table
                timing_df = pd.DataFrame(timing)
                st.dataframe(timing_df[["Model", "Training Time (s)", "Prediction Time (s)"]], use_container_width=True)
                
                # Plot training time as a bar chart
                fig = px.bar(
                    timing_df,
                    x="Model",
                    y="Training Time (ms)",
                    title="Training Time by Model (ms)"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Tab 4: Anomaly Agreement
            with tab4:
                st.subheader("Model Agreement on Anomalies")
                
                # Create a matrix showing how many anomalies are shared between models
                model_names = list(all_scores.keys())
                agreement_matrix = np.zeros((len(model_names), len(model_names)))
                
                for i, model_i in enumerate(model_names):
                    for j, model_j in enumerate(model_names):
                        # Get anomalies for both models
                        anomalies_i = set(all_anomalies[model_i])
                        anomalies_j = set(all_anomalies[model_j])
                        
                        # Count overlapping anomalies
                        overlap = len(anomalies_i.intersection(anomalies_j))
                        
                        # Store in matrix
                        agreement_matrix[i, j] = overlap
                
                # Create heatmap
                fig = plt.figure(figsize=(10, 8))
                sns.heatmap(
                    agreement_matrix,
                    annot=True,
                    fmt=".0f",
                    cmap="viridis",
                    xticklabels=model_names,
                    yticklabels=model_names
                )
                plt.title("Number of Shared Anomalies Between Models")
                plt.tight_layout()
                st.pyplot(fig)
            
            # Add download options
            st.subheader("Export Results")
            
            # Prepare comparison results for download
            comparison_results = pd.DataFrame({
                "Record ID": range(len(df))
            })
            
            for model_name, scores in all_scores.items():
                comparison_results[f"{model_name} Score"] = scores
                comparison_results[f"{model_name} Is Anomaly"] = scores > results[model_name]["threshold"]
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                csv = comparison_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Comparison Results (CSV)",
                    data=csv,
                    file_name="model_comparison_results.csv",
                    mime="text/csv"
                )
            
            with col2:
                json_str = comparison_results.to_json(orient='records')
                st.download_button(
                    "Download Comparison Results (JSON)",
                    data=json_str,
                    file_name="model_comparison_results.json",
                    mime="application/json"
                )
                
        except Exception as e:
            st.error(f"Error during model comparison: {str(e)}")
            logger.exception(f"Error during model comparison: {str(e)}")
    
    # Show save models section if comparison has been run
    if 'model_comparison_results' in st.session_state:
        st.subheader("Save Models from Previous Comparison")
        
        # Initialize model manager
        model_manager = ModelManager()
        
        comparison_results = st.session_state.model_comparison_results
        trained_models = comparison_results.get("trained_models", [])
        model_names = list(comparison_results.get("all_scores", {}).keys())
        selected_features = comparison_results.get("selected_features", [])
        
        if not trained_models:
            st.info("No trained models available for saving.")
        else:
            # Let user select which models to save
            st.write("Select models to save for future use:")
            st.info(f"Available models from comparison run at: {comparison_results['timestamp']}")
            
            # Create columns for save options
            save_cols = st.columns(min(len(trained_models), 4))
            save_selections = []
            
            for i, model in enumerate(trained_models):
                if i < len(model_names):
                    model_name = model_names[i]
                    col_idx = i % len(save_cols)
                    with save_cols[col_idx]:
                        save_selections.append(st.checkbox(f"Save {model_name}", key=f"save_model_persist_{i}_{hash(str(comparison_results['timestamp']))}"))
            
            if st.button("Save Selected Models", key=f"save_models_persist_btn_{hash(str(comparison_results['timestamp']))}"):
                saved_models = []
                for i, (save, model) in enumerate(zip(save_selections, trained_models)):
                    if save and i < len(model_names):
                        model_name = model_names[i]
                        try:
                            # Set feature names and other metadata
                            model.feature_names = selected_features
                            if not hasattr(model, 'metadata'):
                                model.metadata = {}
                            model.metadata["feature_names"] = selected_features
                            model.metadata["data_shape"] = st.session_state.combined_data[selected_features].shape if 'combined_data' in st.session_state else (0, len(selected_features))
                            model.metadata["training_time"] = comparison_results["training_times"].get(model_name, 0)
                            model.metadata["prediction_time"] = comparison_results["prediction_times"].get(model_name, 0)
                            model.metadata["comparison_timestamp"] = comparison_results["timestamp"]
                            
                            # Generate model type with timestamp for comparison models
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            model_type = model.__class__.__name__.replace("Detector", "")
                            comparison_model_type = f"{model_type}_comparison_{timestamp}"
                            
                            # Update model metadata with comparison-specific type
                            model.metadata["type"] = comparison_model_type
                            model.metadata["original_type"] = model_type
                            
                            # Save the model (ModelManager will generate the filename based on type)
                            saved_path = model_manager.save_model(model, create_backup=False)
                            saved_models.append((model_name, saved_path))
                        except Exception as e:
                            st.error(f"Error saving {model_name} model: {str(e)}")
                            logger.exception(f"Error saving {model_name} model: {str(e)}")
                
                if saved_models:
                    st.success(f"Successfully saved {len(saved_models)} models:")
                    for model_name, path in saved_models:
                        st.write(f"- {model_name}: {path}")
                else:
                    st.warning("No models were selected for saving.")
