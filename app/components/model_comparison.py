"""
Model comparison utilities for the Network Anomaly Detection Platform.
Provides functions to compare and evaluate multiple anomaly detection models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger("streamlit_app")

def normalize_scores(scores: np.ndarray, model_type: str) -> np.ndarray:
    """
    Normalize scores based on model type to ensure fair comparison.
    Different models produce different score ranges and interpretations.
    
    Args:
        scores: Array of anomaly scores
        model_type: Type of the model that produced the scores
        
    Returns:
        Normalized scores (higher = more anomalous)
    """
    # Make a copy to avoid modifying the original
    normalized = np.array(scores).copy()
    
    # Handle different model types
    if model_type == "IsolationForest":
        # Isolation Forest: Higher score = more anomalous (already correct)
        pass
    elif model_type == "LocalOutlierFactor":
        # LOF: Higher score = more anomalous (already correct)
        pass
    elif model_type == "OneClassSVM":
        # One-Class SVM: Lower score (negative) = more anomalous, invert and shift
        # Typically produces scores from -1 to 1, where <0 is anomalous
        normalized = -normalized  # Invert so higher = more anomalous
    elif model_type == "DBSCAN":
        # DBSCAN: -1 = anomaly, everything else is a cluster ID, convert to binary
        normalized = np.where(normalized == -1, 1.0, 0.0)
    elif model_type == "KNN":
        # KNN: Higher distance = more anomalous (already correct)
        pass
    elif model_type == "HDBSCAN":
        # HDBSCAN: Similar to DBSCAN, convert outlier scores
        pass
    elif model_type == "Ensemble":
        # Ensemble: Already normalized by the ensemble method
        pass
    
    # If all scores are the same, add a small noise to avoid all-same thresholds
    if np.all(normalized == normalized[0]):
        normalized = normalized + np.random.normal(0, 0.01, size=normalized.shape)
    
    # If distribution is inverted (most points are anomalies), flip it
    if np.mean(normalized) > 0.5:
        # Check for bimodal distribution or if we need to flip
        lower_half = np.mean(normalized < np.median(normalized))
        if lower_half < 0.1:  # Very few points below median suggests we should flip
            normalized = -normalized + np.max(normalized)
    
    return normalized

def calculate_optimal_threshold(scores: np.ndarray, contamination: float = 0.01, model_type: str = None) -> float:
    """
    Calculate optimal threshold based on expected contamination and model type.
    
    Args:
        scores: Array of anomaly scores
        contamination: Expected proportion of anomalies
        model_type: Type of model (affects threshold calculation)
    
    Returns:
        Optimal threshold value
    """
    # Handle edge cases
    if len(scores) == 0:
        return 0.0
    
    # For models with specific threshold logic
    if model_type == "OneClassSVM":
        # One-Class SVM typically uses 0 as threshold
        return 0.0
    elif model_type == "DBSCAN":
        # DBSCAN is already binary (-1 = outlier)
        return 0.5  # Midpoint between 0 and 1 after normalization
    
    # For most models, use percentile based on contamination
    return np.percentile(scores, 100 * (1 - contamination))

def create_score_distribution_plot(score_dict: Dict[str, np.ndarray], threshold_dict: Dict[str, float]) -> go.Figure:
    """
    Create a plotly figure comparing score distributions from multiple models.
    
    Args:
        score_dict: Dictionary mapping model names to their score arrays
        threshold_dict: Dictionary mapping model names to their thresholds
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    for model_name, scores in score_dict.items():
        # Add histogram for this model's scores
        fig.add_trace(go.Histogram(
            x=scores,
            name=model_name,
            opacity=0.6,
            nbinsx=50
        ))
        
        # Add threshold line if available
        if model_name in threshold_dict:
            threshold = threshold_dict[model_name]
            fig.add_vline(
                x=threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"{model_name} threshold",
                annotation_position="top right"
            )
    
    fig.update_layout(
        title="Score Distribution Comparison",
        xaxis_title="Anomaly Score",
        yaxis_title="Count",
        barmode='overlay',
        legend_title="Model"
    )
    
    return fig

def analyze_model_overlap(model_scores: Dict[str, np.ndarray], 
                          model_thresholds: Dict[str, float]) -> Tuple[List[Dict], np.ndarray]:
    """
    Analyze the overlap between anomalies detected by different models.
    
    Args:
        model_scores: Dictionary mapping model names to their score arrays
        model_thresholds: Dictionary mapping model names to their thresholds
        
    Returns:
        Tuple containing overlap data and anomaly sets
    """
    # Create sets of anomaly indices for each model
    anomaly_sets = {}
    model_names = list(model_scores.keys())
    
    for model_name in model_names:
        scores = model_scores[model_name]
        threshold = model_thresholds[model_name]
        anomaly_indices = np.where(scores > threshold)[0]
        anomaly_sets[model_name] = set(anomaly_indices)
    
    # Calculate pairwise overlap
    overlap_data = []
    
    for i, model1 in enumerate(model_names):
        for model2 in model_names[i+1:]:
            set1 = anomaly_sets[model1]
            set2 = anomaly_sets[model2]
            
            if not set1 or not set2:
                overlap_percent = 0
                jaccard = 0
            else:
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                overlap_percent = (intersection / min(len(set1), len(set2))) * 100 if min(len(set1), len(set2)) > 0 else 0
                jaccard = (intersection / union) * 100 if union > 0 else 0
            
            overlap_data.append({
                "Model 1": model1,
                "Model 2": model2,
                "Model 1 Anomalies": len(set1),
                "Model 2 Anomalies": len(set2),
                "Overlap Count": len(set1.intersection(set2)),
                "Overlap %": f"{overlap_percent:.1f}%",
                "Jaccard Similarity": f"{jaccard:.1f}%"
            })
    
    return overlap_data, anomaly_sets

def create_overlap_matrix_plot(anomaly_sets: Dict[str, set], data_size: int) -> plt.Figure:
    """
    Create a confusion matrix style visualization for two models.
    
    Args:
        anomaly_sets: Dictionary mapping model names to sets of anomaly indices
        data_size: Total number of data points
        
    Returns:
        Matplotlib figure object
    """
    if len(anomaly_sets) < 2:
        return None
    
    # Get the first two models for visualization
    model_names = list(anomaly_sets.keys())
    model1, model2 = model_names[0], model_names[1]
    
    set1 = anomaly_sets[model1]
    set2 = anomaly_sets[model2]
    
    # Calculate confusion matrix values
    true_pos = len(set1.intersection(set2))  # Both models agree it's an anomaly
    false_pos = len(set1 - set2)  # Only model1 says it's an anomaly
    false_neg = len(set2 - set1)  # Only model2 says it's an anomaly
    true_neg = data_size - (true_pos + false_pos + false_neg)  # Both agree it's normal
    
    # Create confusion matrix data
    conf_matrix = np.array([
        [true_neg, false_pos],
        [false_neg, true_pos]
    ])
    
    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt='d',
        cmap='Blues',
        xticklabels=[f'Normal ({model1})', f'Anomaly ({model1})'],
        yticklabels=[f'Normal ({model2})', f'Anomaly ({model2})']
    )
    plt.title(f'Anomaly Detection Agreement: {model1} vs {model2}')
    plt.ylabel(model2)
    plt.xlabel(model1)
    
    return fig

def provide_expert_analysis(metrics_df: pd.DataFrame) -> List[str]:
    """
    Provide expert analysis and recommendations based on model comparison metrics.
    
    Args:
        metrics_df: DataFrame with model comparison metrics
        
    Returns:
        List of analysis points and recommendations
    """
    analysis = []
    
    # Extract anomaly percentages
    if 'Anomaly %' in metrics_df.columns:
        metrics_df['Anomaly Percentage'] = metrics_df['Anomaly %'].apply(
            lambda x: float(x.replace('%', '')) if isinstance(x, str) else x
        )
        
        # Calculate variance in anomaly detection rates
        anomaly_variance = metrics_df['Anomaly Percentage'].std()
        
        if anomaly_variance > 20:
            analysis.append("⚠️ **High variance in anomaly detection rates between models**. Models disagree significantly about what constitutes an anomaly.")
            analysis.append("- Use the model most suitable for your data distribution (e.g., Isolation Forest for high-dimensional data, LOF for density-based anomalies)")
            analysis.append("- Verify model parameters align with your expectations about anomaly prevalence")
            analysis.append("- Consider an ensemble approach to reduce false positives")
        elif anomaly_variance < 5 and len(metrics_df) > 2:
            analysis.append("✅ **Models show strong agreement** on anomaly detection rates.")
            analysis.append("- The anomalies in your data are well-defined and consistently detectable")
            analysis.append("- You can choose models based on performance or interpretability preferences")
            analysis.append("- Consider the model with the fastest analysis time for production use")
        else:
            analysis.append("ℹ️ **Models show moderate agreement** on anomaly detection.")
            analysis.append("- Examine overlap analysis to understand which models detect similar or different anomalies")
            analysis.append("- Check if certain models detect specific types of anomalies that others miss")
            analysis.append("- Consider using multiple models for different purposes")
    
    # Add model-specific advice
    models = metrics_df['Model'].unique()
    
    model_advice = []
    if "IsolationForest" in models:
        model_advice.append("- **Isolation Forest**: Good for high-dimensional data with mixed types of anomalies. Generally robust and fast.")
    if "LocalOutlierFactor" in models:
        model_advice.append("- **Local Outlier Factor**: Effective at detecting local density-based anomalies. Works well when normal data forms clusters.")
    if "OneClassSVM" in models:
        model_advice.append("- **One-Class SVM**: Good for datasets with clear boundaries between normal and anomalous points. Can be computationally intensive for large datasets.")
    if "KNN" in models:
        model_advice.append("- **KNN**: Simple and intuitive. Works well when normal data is clustered and anomalies are isolated.")
    if "DBSCAN" in models:
        model_advice.append("- **DBSCAN**: Excellent for density-based anomalies and clusters of varying shapes. No need to specify number of clusters.")
    if "Ensemble" in models:
        model_advice.append("- **Ensemble**: Combines strengths of multiple models. Often more robust but potentially slower and less interpretable.")
    
    if model_advice:
        analysis.append("\n**Model-Specific Considerations:**")
        analysis.extend(model_advice)
    
    return analysis