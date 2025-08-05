"""
Enhanced Explainers for the Network Anomaly Detection Platform.
Provides SHAP and LIME explainers with interactive features and natural language explanations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import shap
import logging
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Tuple, Optional, Union

# Try to import LIME (with graceful fallback if not available)
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

logger = logging.getLogger("streamlit_app")

def explain_with_shap(model, X: pd.DataFrame, instance_idx: Optional[int] = None, 
                     features: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Generate SHAP explanations for a model.
    
    Args:
        model: The trained model to explain
        X: Input data (features)
        instance_idx: Index of a specific instance to explain (optional)
        features: List of feature names (optional)
    
    Returns:
        Dictionary with SHAP plots and values
    """
    results = {}
    
    try:
        # Check if model has underlying model attribute (for our detector wrappers)
        if hasattr(model, 'model'):
            underlying_model = model.model
        else:
            underlying_model = model
        
        # Use appropriate explainer based on model type
        if hasattr(underlying_model, 'estimators_') or 'RandomForest' in str(type(underlying_model)) or 'IsolationForest' in str(type(underlying_model)):
            # For tree-based models
            explainer = shap.TreeExplainer(underlying_model)
        else:
            # For other model types, fall back to KernelExplainer
            # (this is slower but works with black-box models)
            if instance_idx is not None:
                # For a single instance, we don't need background data
                explainer = shap.KernelExplainer(underlying_model.predict, X.iloc[:10])
            else:
                # Sample background data for efficiency
                background = shap.kmeans(X, 10)
                explainer = shap.KernelExplainer(underlying_model.predict, background)
        
        # Get feature names
        if features is None and hasattr(model, 'feature_names'):
            features = model.feature_names
        
        # Calculate SHAP values
        if instance_idx is not None:
            # Explain a single instance
            X_instance = X.iloc[[instance_idx]]
            shap_values = explainer.shap_values(X_instance)
            
            # Create plots
            plt.figure(figsize=(10, 4))
            force_plot = shap.force_plot(
                explainer.expected_value, 
                shap_values[0] if isinstance(shap_values, list) else shap_values,
                X_instance.iloc[0],
                feature_names=features,
                matplotlib=True, 
                show=False
            )
            results['force_plot'] = force_plot
            
            # Waterfall plot
            plt.figure(figsize=(10, 6))
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # For multi-class models, use first class
                values_for_waterfall = shap_values[0]
                if len(values_for_waterfall.shape) > 1:
                    values_for_waterfall = values_for_waterfall[0]  # First instance
            else:
                # For single output models
                values_for_waterfall = shap_values
                if len(values_for_waterfall.shape) > 1:
                    values_for_waterfall = values_for_waterfall[0]  # First instance
            
            waterfall_plot = shap.waterfall_plot(
                shap.Explanation(
                    values=values_for_waterfall,
                    base_values=explainer.expected_value,
                    data=X_instance.iloc[0].values,
                    feature_names=features
                ),
                show=False
            )
            results['waterfall_plot'] = waterfall_plot
            
            # Store values
            results['shap_values'] = shap_values
            results['expected_value'] = explainer.expected_value
            
        else:
            # Explain the entire dataset
            shap_values = explainer.shap_values(X)
            
            # Summary plot
            plt.figure(figsize=(10, 6))
            summary_plot = shap.summary_plot(shap_values, X, feature_names=features, show=False)
            results['summary_plot'] = summary_plot
            
            # Bar plot
            plt.figure(figsize=(10, 6))
            bar_plot = shap.summary_plot(shap_values, X, feature_names=features, plot_type="bar", show=False)
            results['bar_plot'] = bar_plot
            
            # Get top features
            importance_vals = np.abs(shap_values).mean(0)
            top_indices = importance_vals.argsort()[-3:][::-1]
            
            # Dependence plots for top features
            dependence_plots = []
            for i, idx in enumerate(top_indices):
                plt.figure(figsize=(10, 6))
                dep_plot = shap.dependence_plot(idx, shap_values, X, feature_names=features, show=False)
                dependence_plots.append(dep_plot)
            
            results['dependence_plots'] = dependence_plots
            results['shap_values'] = shap_values
            results['expected_value'] = explainer.expected_value
        
        return results
    
    except Exception as e:
        logger.error(f"Error in SHAP explanation: {str(e)}")
        return {"error": str(e)}

def explain_with_lime(model, X: pd.DataFrame, instance_idx: int, 
                     features: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Generate LIME explanations for a model.
    
    Args:
        model: The trained model to explain
        X: Input data (features)
        instance_idx: Index of the instance to explain
        features: List of feature names (optional)
    
    Returns:
        Dictionary with LIME explanation
    """
    if not LIME_AVAILABLE:
        return {"error": "LIME is not installed. Install with: pip install lime"}
    
    results = {}
    
    try:
        # Get feature names
        if features is None and hasattr(model, 'feature_names'):
            features = model.feature_names
        elif features is None:
            features = X.columns.tolist()
        
        # Create a prediction function for LIME
        def predict_fn(instances):
            if isinstance(instances, np.ndarray):
                instances_df = pd.DataFrame(instances, columns=X.columns)
            else:
                instances_df = instances
            return model.predict(instances_df)
        
        # Create explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X.values,
            feature_names=features,
            class_names=["normal", "anomaly"],
            discretize_continuous=True,
            mode="regression"  # Use regression for anomaly scores
        )
        
        # Get explanation for the instance
        exp = explainer.explain_instance(
            X.iloc[instance_idx].values, 
            predict_fn,
            num_features=min(10, len(features))
        )
        
        # Store explanation
        results['explanation'] = exp
        
        # Generate plot
        fig = exp.as_pyplot_figure(label=1)
        results['figure'] = fig
        
        return results
    
    except Exception as e:
        logger.error(f"Error in LIME explanation: {str(e)}")
        return {"error": str(e)}

def display_shap_explanation(shap_results: Dict[str, Any], st_container=None):
    """
    Display SHAP explanation results in a Streamlit container.
    
    Args:
        shap_results: Results from explain_with_shap
        st_container: Streamlit container to use (optional)
    """
    if st_container is None:
        st_container = st
    
    if "error" in shap_results:
        st_container.error(f"SHAP explanation error: {shap_results['error']}")
        return
    
    # Display plots based on what's available
    if 'force_plot' in shap_results:
        st_container.write("#### SHAP Force Plot")
        st_container.write("This plot shows how each feature contributes to the prediction for this specific instance.")
        st_container.pyplot(plt, key="shap_force_display")
    
    if 'waterfall_plot' in shap_results:
        st_container.write("#### SHAP Waterfall Plot")
        st_container.write("This plot shows the cumulative effect of features on the prediction.")
        st_container.pyplot(plt, key="shap_waterfall_display")
    
    if 'summary_plot' in shap_results:
        st_container.write("#### SHAP Summary Plot")
        st_container.write("This plot shows the impact of each feature across all instances.")
        st_container.pyplot(plt, key="shap_summary_display")
    
    if 'bar_plot' in shap_results:
        st_container.write("#### SHAP Feature Importance")
        st_container.write("This plot shows the average impact of each feature on model output magnitude.")
        st_container.pyplot(plt, key="shap_bar_display")
    
    if 'dependence_plots' in shap_results and shap_results['dependence_plots']:
        st_container.write("#### SHAP Dependence Plots")
        st_container.write("These plots show how the feature's effect depends on its value.")
        for i, plot in enumerate(shap_results['dependence_plots']):
            st_container.pyplot(plt, key=f"shap_dependence_{i}_display")

def display_lime_explanation(lime_results: Dict[str, Any], st_container=None):
    """
    Display LIME explanation results in a Streamlit container.
    
    Args:
        lime_results: Results from explain_with_lime
        st_container: Streamlit container to use (optional)
    """
    if st_container is None:
        st_container = st
    
    if "error" in lime_results:
        st_container.error(f"LIME explanation error: {lime_results['error']}")
        return
    
    if 'figure' in lime_results:
        st_container.write("#### LIME Feature Importance")
        st_container.write("This plot shows which features were most important for this prediction.")
        st_container.pyplot(lime_results['figure'], key="lime_figure_display")
    
    if 'explanation' in lime_results:
        exp = lime_results['explanation']
        
        st_container.write("#### LIME Explanation Details")
        
        # Display feature weights as a table
        explanation_data = exp.as_list()
        explanation_df = pd.DataFrame(explanation_data, columns=["Feature", "Weight"])
        st_container.dataframe(explanation_df, use_container_width=True)

def create_interactive_shap_plot(shap_values: np.ndarray, X: pd.DataFrame, 
                                feature_names: List[str], instance_idx: Optional[int] = None) -> go.Figure:
    """
    Create an interactive SHAP plot using Plotly.
    
    Args:
        shap_values: SHAP values array
        X: Feature data
        feature_names: List of feature names
        instance_idx: Index of specific instance (optional)
        
    Returns:
        Interactive Plotly figure
    """
    if instance_idx is not None:
        # Single instance waterfall plot
        values = shap_values[instance_idx] if shap_values.ndim > 1 else shap_values
        feature_values = X.iloc[instance_idx] if X.ndim > 1 else X
        
        # Sort by absolute SHAP value
        sorted_indices = np.argsort(np.abs(values))[::-1]
        
        # Take top 10 features for readability
        top_indices = sorted_indices[:10]
        top_values = values[top_indices]
        top_features = [feature_names[i] for i in top_indices]
        top_feature_values = [feature_values.iloc[i] if hasattr(feature_values, 'iloc') else feature_values[i] for i in top_indices]
        
        # Create waterfall chart
        fig = go.Figure(go.Waterfall(
            name="SHAP Values",
            orientation="v",
            measure=["relative"] * len(top_values),
            x=top_features,
            y=top_values,
            text=[f"{val:.3f}<br>Feature Value: {fval:.2f}" for val, fval in zip(top_values, top_feature_values)],
            textposition="auto",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "red"}},
            decreasing={"marker": {"color": "green"}},
        ))
        
        fig.update_layout(
            title="Interactive SHAP Waterfall Plot",
            xaxis_title="Features",
            yaxis_title="SHAP Value (Impact on Prediction)",
            showlegend=False
        )
        
    else:
        # Summary plot for all instances
        mean_abs_shap = np.abs(shap_values).mean(0)
        sorted_indices = np.argsort(mean_abs_shap)[::-1]
        
        # Take top 15 features
        top_indices = sorted_indices[:15]
        top_features = [feature_names[i] for i in top_indices]
        top_importance = mean_abs_shap[top_indices]
        
        # Create horizontal bar chart
        fig = go.Figure(go.Bar(
            x=top_importance,
            y=top_features,
            orientation='h',
            marker_color='lightblue',
            text=[f"{val:.3f}" for val in top_importance],
            textposition='auto',
        ))
        
        fig.update_layout(
            title="Interactive SHAP Feature Importance",
            xaxis_title="Mean |SHAP Value|",
            yaxis_title="Features",
            yaxis={'categoryorder': 'total ascending'}
        )
    
    return fig

def generate_natural_language_explanation(shap_values: np.ndarray, X: pd.DataFrame, 
                                        feature_names: List[str], instance_idx: int,
                                        model_type: str, anomaly_score: float) -> str:
    """
    Generate a natural language explanation for an anomaly prediction.
    
    Args:
        shap_values: SHAP values for the instance
        X: Feature data
        feature_names: List of feature names  
        instance_idx: Index of the instance
        model_type: Type of the model
        anomaly_score: The anomaly score
        
    Returns:
        Natural language explanation string
    """
    # Safe indexing with bounds checking
    try:
        if shap_values.ndim > 1:
            # Check if instance_idx is within bounds
            if instance_idx >= shap_values.shape[0]:
                instance_idx = 0  # Use first instance as fallback
            values = shap_values[instance_idx]
        else:
            values = shap_values
        
        # Check if instance_idx is within DataFrame bounds
        if instance_idx >= len(X):
            instance_idx = 0  # Use first instance as fallback
        feature_values = X.iloc[instance_idx]
    except (IndexError, KeyError) as e:
        # Fallback to first instance if indexing fails
        values = shap_values[0] if shap_values.ndim > 1 else shap_values
        feature_values = X.iloc[0]
        instance_idx = 0
    
    # Sort features by absolute SHAP value
    sorted_indices = np.argsort(np.abs(values))[::-1]
    
    # Get top contributing features
    top_3_indices = sorted_indices[:3]
    top_3_features = [feature_names[i] for i in top_3_indices]
    top_3_values = values[top_3_indices]
    top_3_feature_values = [feature_values.iloc[i] for i in top_3_indices]
    
    # Determine if it's an anomaly based on score comparison with a reasonable threshold
    # Note: Different algorithms use different scoring conventions
    # Most use: higher score = more anomalous
    # Some (like Isolation Forest) use: lower score = more anomalous
    
    # For this explanation, we'll use a simple heuristic:
    # If anomaly_score is very low (< 0.2), it's likely normal
    # If anomaly_score is high (> 0.5), it's likely anomalous
    is_anomaly = anomaly_score > 0.3  # More conservative threshold for explanations
    
    explanation = []
    
    # Opening statement with better logic
    if is_anomaly:
        confidence = "HIGH" if anomaly_score > 0.7 else "MEDIUM" if anomaly_score > 0.5 else "LOW"
        explanation.append(f"🚨 **This network event was flagged as an ANOMALY** (score: {anomaly_score:.3f}, confidence: {confidence}) by the {model_type} model.")
    else:
        explanation.append(f"✅ **This network event appears NORMAL** (score: {anomaly_score:.3f}) according to the {model_type} model.")
    
    explanation.append("")
    explanation.append("**Key Contributing Factors:**")
    
    # Explain top features
    for i, (feature, shap_val, feat_val) in enumerate(zip(top_3_features, top_3_values, top_3_feature_values)):
        rank = ["Primary", "Secondary", "Tertiary"][i]
        direction = "INCREASES" if shap_val > 0 else "DECREASES"
        impact = "strongly" if abs(shap_val) > 0.1 else "moderately" if abs(shap_val) > 0.05 else "slightly"
        
        # Feature-specific explanations
        feature_explanation = get_feature_specific_explanation(feature, feat_val, shap_val)
        
        explanation.append(f"{i+1}. **{rank} factor ({feature})**: {feature_explanation}")
        explanation.append(f"   - This feature {impact} {direction.lower()} the anomaly likelihood (impact: {shap_val:+.3f})")
    
    # Add contextual advice
    explanation.append("")
    if is_anomaly:
        explanation.append("**Recommended Actions:**")
        explanation.append("- Review this event for potential security implications")
        explanation.append("- Check if similar patterns exist in recent network traffic")
        explanation.append("- Consider correlating with threat intelligence feeds")
        explanation.append("- Validate against known good network behavior patterns")
    else:
        explanation.append("**Analysis Notes:**")
        explanation.append("- This event follows expected network behavior patterns")
        explanation.append("- The feature values are within normal operational ranges")
        explanation.append("- No immediate action required unless part of broader investigation")
    
    return "\n".join(explanation)

def get_feature_specific_explanation(feature_name: str, feature_value: float, shap_value: float) -> str:
    """
    Generate feature-specific explanations for network data.
    Updated to handle JSON column names like frame.len, tcp.srcport, etc.
    
    Args:
        feature_name: Name of the feature
        feature_value: Value of the feature
        shap_value: SHAP value for this feature
        
    Returns:
        Human-readable explanation for this feature
    """
    feature_lower = feature_name.lower()
    
    # Frame length / packet size features
    if 'frame.len' in feature_lower or 'len' in feature_lower or 'size' in feature_lower:
        if shap_value > 0:
            return f"Unusually large frame/packet size ({feature_value:.0f} bytes) compared to typical network patterns"
        else:
            return f"Normal frame/packet size ({feature_value:.0f} bytes) consistent with expected traffic"
    
    # TCP/UDP port features (handles tcp.srcport, tcp.dstport, udp.srcport, udp.dstport)
    elif any(port_term in feature_lower for port_term in ['port', 'tcp.srcport', 'tcp.dstport', 'udp.srcport', 'udp.dstport']):
        port_type = "source" if 'src' in feature_lower else "destination" if 'dst' in feature_lower else ""
        protocol = "TCP" if 'tcp' in feature_lower else "UDP" if 'udp' in feature_lower else ""
        
        if shap_value > 0:
            return f"Uncommon {protocol} {port_type} port usage (port {feature_value:.0f}) potentially indicating non-standard services"
        else:
            return f"Standard {protocol} {port_type} port usage (port {feature_value:.0f}) consistent with known services"
    
    # Frame number (usually not useful for anomaly detection but might be present)
    elif 'frame.number' in feature_lower or 'frame_number' in feature_lower:
        if shap_value > 0:
            return f"Frame position ({feature_value:.0f}) may indicate temporal clustering of suspicious activity"
        else:
            return f"Frame position ({feature_value:.0f}) within normal packet sequence"
    
    # Row ID (if somehow included)
    elif 'row_id' in feature_lower:
        if shap_value > 0:
            return f"Data position ({feature_value:.0f}) may correlate with batch processing or temporal patterns"
        else:
            return f"Data position ({feature_value:.0f}) shows normal processing order"
    
    # Protocol-related features
    elif 'protocol' in feature_lower:
        if shap_value > 0:
            return f"Protocol usage ({feature_value:.2f}) deviates from typical communication patterns"
        else:
            return f"Standard protocol usage ({feature_value:.2f}) following normal network protocols"
    
    # Time-related features
    elif 'time' in feature_lower or 'delta' in feature_lower:
        if shap_value > 0:
            return f"Unusual timing pattern ({feature_value:.2f}) suggesting non-standard communication intervals"
        else:
            return f"Normal timing pattern ({feature_value:.2f}) typical for this type of traffic"
    
    # Packet count features
    elif 'packet' in feature_lower or 'count' in feature_lower:
        if shap_value > 0:
            return f"Elevated packet count ({feature_value:.0f}) indicating potential bulk data transfer or scanning"
        else:
            return f"Standard packet count ({feature_value:.0f}) within normal communication patterns"
    
    # IP address features (typically encoded)
    elif 'ip' in feature_lower or 'addr' in feature_lower:
        if shap_value > 0:
            return f"IP address pattern ({feature_value:.2f}) associated with suspicious or uncommon behavior"
        else:
            return f"IP address pattern ({feature_value:.2f}) consistent with known legitimate hosts"
    
    # Frequency/rate features
    elif 'rate' in feature_lower or 'freq' in feature_lower:
        if shap_value > 0:
            return f"High activity rate ({feature_value:.2f}) suggesting automated or bulk operations"
        else:
            return f"Normal activity rate ({feature_value:.2f}) consistent with human or standard automated behavior"
    
    # Generic numeric features (fallback for any unrecognized column)
    else:
        feature_display = feature_name.replace('.', ' ').replace('_', ' ').title()
        if shap_value > 0:
            return f"{feature_display} value ({feature_value:.3f}) is unusual compared to normal network behavior"
        else:
            return f"{feature_display} value ({feature_value:.3f}) aligns with expected network characteristics"

def create_counterfactual_explanation(model, X: pd.DataFrame, instance_idx: int, 
                                     feature_names: List[str], target_prediction: float = 0.0) -> Dict[str, Any]:
    """
    Generate counterfactual explanations - "what if" scenarios.
    
    Args:
        model: Trained model
        X: Feature data
        instance_idx: Index of instance to explain
        feature_names: List of feature names
        target_prediction: Target prediction value (default: 0.0 for normal)
        
    Returns:
        Dictionary with counterfactual analysis
    """
    try:
        original_instance = X.iloc[instance_idx].copy()
        
        # Handle NaN values - fill with median values
        if original_instance.isna().any():
            print("Warning: NaN values detected in original instance, filling with median values")
            for col in original_instance.index:
                if pd.isna(original_instance[col]):
                    median_val = X[col].median()
                    if pd.isna(median_val):
                        # If median is also NaN, use 0
                        original_instance[col] = 0
                    else:
                        original_instance[col] = median_val
        
        original_prediction = model.predict([original_instance])[0]
        
        results = {
            'original_prediction': original_prediction,
            'target_prediction': target_prediction,
            'counterfactuals': []
        }
        
        # Try modifying each feature to see impact
        for i, feature in enumerate(feature_names):
            try:
                # Calculate feature statistics for realistic ranges - handle NaN values
                feature_values = X.iloc[:, i].dropna()  # Remove NaN values for stats
                
                if len(feature_values) == 0:
                    continue  # Skip if all values are NaN
                
                feature_min, feature_max = feature_values.min(), feature_values.max()
                feature_mean, feature_std = feature_values.mean(), feature_values.std()
                
                # Skip if std is NaN (single value)
                if pd.isna(feature_std):
                    feature_std = 0
                
                # Try different values
                test_values = [
                    feature_mean,  # Mean value
                    feature_mean - feature_std,  # One std below mean
                    feature_mean + feature_std,  # One std above mean
                    feature_min,  # Minimum value
                    feature_max   # Maximum value
                ]
                
                for test_value in test_values:
                    if pd.isna(test_value):
                        continue  # Skip NaN test values
                    
                    # Create modified instance
                    modified_instance = original_instance.copy()
                    modified_instance.iloc[i] = test_value
                    
                    # Ensure no NaN values in modified instance
                    if modified_instance.isna().any():
                        continue
                    
                    # Get prediction for modified instance
                    new_prediction = model.predict([modified_instance])[0]
                    
                    # Calculate improvement (closer to target)
                    original_distance = abs(original_prediction - target_prediction)
                    new_distance = abs(new_prediction - target_prediction)
                    improvement = original_distance - new_distance
                    
                    if improvement > 0.01:  # Only keep meaningful improvements
                        results['counterfactuals'].append({
                            'feature': feature,
                            'original_value': original_instance.iloc[i],
                            'modified_value': test_value,
                            'original_prediction': original_prediction,
                            'new_prediction': new_prediction,
                            'improvement': improvement,
                            'change_magnitude': abs(test_value - original_instance.iloc[i])
                        })
            except Exception as e:
                print(f"Error processing feature {feature}: {str(e)}")
                continue
        
        # Sort by improvement
        results['counterfactuals'] = sorted(
            results['counterfactuals'], 
            key=lambda x: x['improvement'], 
            reverse=True
        )
        
    except Exception as e:
        print(f"Error in counterfactual analysis: {str(e)}")
        return {
            'original_prediction': 0,
            'target_prediction': target_prediction,
            'counterfactuals': [],
            'error': str(e)
        }
    
    return results

def display_enhanced_explanations(model, X: pd.DataFrame, instance_idx: int, 
                                feature_names: List[str], model_type: str, 
                                anomaly_score: float, st_container=None):
    """
    Display comprehensive enhanced explanations in Streamlit.
    
    Args:
        model: Trained model
        X: Feature data
        instance_idx: Index of instance to explain
        feature_names: List of feature names
        model_type: Type of model
        anomaly_score: Anomaly score
        st_container: Streamlit container (optional)
    """
    if st_container is None:
        st_container = st
    
    # Generate SHAP explanations
    with st_container.spinner("Generating explanations..."):
        shap_results = explain_with_shap(model, X, instance_idx, feature_names)
    
    if "error" not in shap_results:
        # Natural language explanation
        st_container.subheader("🗣️ Natural Language Explanation")
        
        nl_explanation = generate_natural_language_explanation(
            shap_results['shap_values'], X, feature_names, 
            instance_idx, model_type, anomaly_score
        )
        st_container.markdown(nl_explanation)
        
        # Interactive SHAP plot
        st_container.subheader("📊 Interactive Feature Impact Analysis")
        
        interactive_fig = create_interactive_shap_plot(
            shap_results['shap_values'], X, feature_names, instance_idx
        )
        st_container.plotly_chart(interactive_fig, use_container_width=True)
        
        # Counterfactual analysis
        st_container.subheader("🔄 What-If Analysis")
        st_container.write("This shows how changing specific features could affect the prediction:")
        
        counterfactual_results = create_counterfactual_explanation(
            model, X, instance_idx, feature_names
        )
        
        if counterfactual_results['counterfactuals']:
            cf_df = pd.DataFrame(counterfactual_results['counterfactuals'][:5])  # Top 5
            cf_df['change_description'] = cf_df.apply(
                lambda row: f"Change {row['feature']} from {row['original_value']:.3f} to {row['modified_value']:.3f}", 
                axis=1
            )
            
            st_container.dataframe(
                cf_df[['change_description', 'new_prediction', 'improvement']].rename(columns={
                    'change_description': 'Suggested Change',
                    'new_prediction': 'New Prediction',
                    'improvement': 'Improvement'
                }),
                use_container_width=True
            )
        else:
            st_container.info("No significant counterfactual changes found that would meaningfully alter the prediction.")
    
    else:
        st_container.error(f"Could not generate explanations: {shap_results['error']}")

def create_explanation_comparison(models: Dict[str, Any], X: pd.DataFrame, 
                                instance_idx: int, feature_names: List[str]) -> go.Figure:
    """
    Compare explanations across multiple models for the same instance.
    
    Args:
        models: Dictionary of model name -> model object
        X: Feature data
        instance_idx: Index of instance to explain
        feature_names: List of feature names
        
    Returns:
        Plotly figure comparing explanations
    """
    model_explanations = {}
    
    # Get SHAP values for each model
    for model_name, model in models.items():
        try:
            shap_results = explain_with_shap(model, X, instance_idx, feature_names)
            if "error" not in shap_results:
                values = shap_results['shap_values']
                if values.ndim > 1:
                    values = values[instance_idx]
                model_explanations[model_name] = values
        except Exception as e:
            logger.warning(f"Could not get SHAP values for {model_name}: {e}")
    
    if not model_explanations:
        return None
    
    # Create comparison plot
    fig = make_subplots(
        rows=1, cols=len(model_explanations),
        subplot_titles=list(model_explanations.keys()),
        shared_yaxis=True
    )
    
    for i, (model_name, shap_values) in enumerate(model_explanations.items()):
        # Get top features by absolute importance
        top_indices = np.argsort(np.abs(shap_values))[-10:][::-1]
        top_features = [feature_names[idx] for idx in top_indices]
        top_values = shap_values[top_indices]
        
        fig.add_trace(
            go.Bar(
                x=top_values,
                y=top_features,
                orientation='h',
                name=model_name,
                showlegend=i == 0
            ),
            row=1, col=i+1
        )
    
    fig.update_layout(
        title="Model Explanation Comparison",
        height=600,
        xaxis_title="SHAP Value"
    )
    
    return fig