"""
Explanation and Feedback page for the Network Anomaly Detection Platform.
Provides model explanations and collects user feedback on anomalies.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import shap
import time
from datetime import datetime
from pathlib import Path

from core.explainers.explainer_factory import ExplainerFactory
from core.explainers.shap_explainer import ShapExplainer
from core.explainers.lime_explainer import LimeExplainer
from core.config_loader import load_config, load_mitre_data
from app.components.error_handler import handle_error
from app.components.visualization import find_timestamp_column

def clean_dataframe_for_arrow(df):
    """
    Clean dataframe for Arrow serialization by converting problematic data types.
    """
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].apply(lambda x: 
                'N/A' if pd.isna(x) 
                else str(x) if isinstance(x, (pd.Timestamp, datetime, list, dict, np.ndarray))
                else x
            )
        elif pd.api.types.is_datetime64_any_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].astype(str)
    return df_clean

# Enhanced imports for new features
try:
    from app.components.explainers import (
        display_enhanced_explanations, create_explanation_comparison
    )
    from core.security_intelligence import ThreatIntelligenceManager, display_threat_intelligence_dashboard
    from core.advanced_analytics import BehavioralAnalytics, PredictiveAnalytics, display_advanced_analytics_dashboard
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    ENHANCED_FEATURES_AVAILABLE = False
    # Note: Enhanced features not available, but basic functionality will work

@handle_error
def show_explain_feedback():
    """Display the Explanation and Feedback page."""
    
    # Ensure plotly is available for this function
    import plotly.express as px
    import plotly.graph_objects as go
    
    st.header("Model Explanation & Analyst Feedback")
    
    # Check if anomalies exist
    anomalies = st.session_state.get('anomalies')
    if anomalies is None or anomalies.empty:
        st.info("No anomalies detected yet. Please run anomaly detection first.")
        return
    
    # Get original data
    df = st.session_state.combined_data
    
    # Get model from session state (stored as anomaly_model in anomaly_detection.py)
    model = st.session_state.get('anomaly_model')
    if model is None:
        st.warning("No model found in session state. Please run anomaly detection first.")
        return
    
    # Create tabs for explanation and feedback
    tab1, tab2, tab3 = st.tabs(["Model Explanation", "Anomaly Feedback", "Advanced Analytics"])
    
    # Model Explanation Tab
    with tab1:
        st.subheader("Model Explanation")
        
        # Get feature names used for model training
        features = st.session_state.get('anomaly_features', [])
        if not features:
            st.error("No features found for model explanation.")
            return
        
        # Get selected model type
        model_type = st.session_state.get('selected_model', 'Unknown')
        st.write(f"Explaining {model_type} model")
        
        # Risk-based filtering
        st.subheader("🎯 Filter Anomalies by Risk Level")
        
        # Check if risk scores are available
        if 'risk_scores' in st.session_state and st.session_state.risk_scores:
            risk_results = st.session_state.risk_scores
            if 'individual_scores' in risk_results:
                individual_scores = risk_results['individual_scores']
                
                # Create risk level filter
                col1, col2 = st.columns(2)
                
                with col1:
                    available_levels = list(set(item['risk_level'] for item in individual_scores))
                    selected_risk_levels = st.multiselect(
                        "Filter by Risk Level",
                        options=['Critical', 'High', 'Medium', 'Low', 'Minimal'],
                        default=available_levels if available_levels else ['High', 'Critical']
                    )
                
                with col2:
                    min_risk_score = st.slider(
                        "Minimum Risk Score",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.1
                    )
                
                # Filter anomalies based on risk criteria
                if selected_risk_levels:
                    # Create a mapping of anomaly_id to risk info
                    risk_map = {item['anomaly_id']: item for item in individual_scores}
                    
                    # Filter anomalies
                    filtered_indices = []
                    for idx in anomalies.index:
                        if idx in risk_map:
                            risk_info = risk_map[idx]
                            if (risk_info['risk_level'] in selected_risk_levels and 
                                risk_info['risk_score'] >= min_risk_score):
                                filtered_indices.append(idx)
                    
                    if filtered_indices:
                        anomalies = anomalies.loc[filtered_indices]
                        st.success(f"📊 Filtered to {len(anomalies)} anomalies matching risk criteria")
                    else:
                        st.warning("⚠️ No anomalies match the selected risk criteria")
                        return
                else:
                    st.warning("⚠️ Please select at least one risk level")
                    return
        else:
            st.info("💡 **Tip:** Risk-based filtering is available after running anomaly detection with auto-analysis enabled.")
        
        # Select explainer type
        explainer_type = st.selectbox(
            "Select explainer type",
            options=["SHAP", "LIME"],
            index=0
        )
        
        # Select anomaly to explain
        st.subheader("Select Anomaly to Explain")
        
        # Display anomalies table with pagination
        items_per_page = 10
        num_pages = (len(anomalies) + items_per_page - 1) // items_per_page
        
        if num_pages > 1:
            page_number = st.selectbox("Page", options=range(1, num_pages + 1), index=0)
            start_idx = (page_number - 1) * items_per_page
        else:
            start_idx = 0
        
        end_idx = min(start_idx + items_per_page, len(anomalies))
        
        # Show the anomalies table
        anomalies_display = anomalies.iloc[start_idx:end_idx].copy()
        
        # Find time column if available
        time_col, _ = find_timestamp_column(anomalies_display)
        
        # Get IP columns
        src_ip_col = next((col for col in ['ip_src', 'ip.src'] if col in anomalies_display.columns), None)
        dst_ip_col = next((col for col in ['ip_dst', 'ip.dst'] if col in anomalies_display.columns), None)
        
        # Determine columns to display
        display_cols = ['anomaly_score']
        
        if time_col:
            display_cols.insert(0, time_col)
        
        if src_ip_col:
            display_cols.append(src_ip_col)
            
        if dst_ip_col:
            display_cols.append(dst_ip_col)
        
        # Add protocol column if available
        protocol_col = next((col for col in ['_ws_col_Protocol', 'protocol'] if col in anomalies_display.columns), None)
        if protocol_col:
            display_cols.append(protocol_col)
        
        # Sort by anomaly score
        anomalies_display = anomalies_display.sort_values('anomaly_score', ascending=False)
        
        # Show select box for anomaly selection
        def format_anomaly_option(x):
            try:
                score = anomalies_display.loc[x, 'anomaly_score']
                if isinstance(score, (int, float)):
                    return f"Anomaly {x} (Score: {score:.3f})"
                else:
                    return f"Anomaly {x} (Score: {score})"
            except Exception:
                return f"Anomaly {x} (Score: N/A)"
                
        selected_index = st.selectbox(
            "Select anomaly to explain",
            options=anomalies_display.index.tolist(),
            format_func=format_anomaly_option
        )
        
        # Show the selected anomaly
        st.write("Selected anomaly details:")
        # Clean dataframe for Arrow compatibility
        anomaly_display = clean_dataframe_for_arrow(anomalies_display.loc[[selected_index]])
        st.dataframe(anomaly_display, use_container_width=True)
        
        # Generate explanation when button is clicked
        if st.button("Generate Explanation"):
            # Show a spinner while generating explanation
            with st.spinner("Generating explanation... This may take a moment."):
                try:
                    # Create explainer
                    if explainer_type == "SHAP":
                        # Get feature matrix for explanation
                        X = df[features]
                        
                        # Show information about the model and features
                        st.info(f"Explaining model using {len(features)} features: {', '.join(features)}")
                        
                        try:
                            # Try SHAP explainer first
                            explainer = ExplainerFactory.create_explainer(
                                'shap',
                                model=model,
                                feature_names=features,
                                background_data=X
                            )
                            explainer_name = "SHAP"
                            
                            # Get the index of the selected anomaly in the original dataset
                            rel_index = df.index.get_loc(selected_index)
                            
                            # Generate explanation
                            explanation = explainer.explain_instance(X, rel_index)
                            
                            # Display SHAP explanation
                            st.subheader("Feature Importance")
                            
                            # Check if we have a valid explanation
                            if "shap_values" not in explanation:
                                st.error("Could not generate SHAP explanation. Trying LIME instead.")
                                raise Exception("Invalid SHAP explanation")
                                
                            # Plot feature importance
                            fig = explainer.plot_summary(explanation)
                            st.pyplot(fig)
                            
                            # Get feature importance values
                            importance_dict = explainer.get_feature_importance(explanation)
                            
                            # Show feature importance as a table
                            importance_df = pd.DataFrame({
                                "Feature": list(importance_dict.keys()),
                                "Importance": list(importance_dict.values())
                            }).sort_values("Importance", ascending=False)
                            
                            st.write("Feature importance values:")
                            importance_clean = clean_dataframe_for_arrow(importance_df)
                            st.dataframe(importance_clean, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"SHAP explainer error: {str(e)}")
                            st.info("Falling back to LIME explainer...")
                            
                            # Fall back to LIME explainer
                            explainer = ExplainerFactory.create_explainer(
                                'lime',
                                model=model,
                                feature_names=features,
                                training_data=X
                            )
                            explainer_name = "LIME"
                            
                            # Get the index of the selected anomaly in the original dataset
                            rel_index = df.index.get_loc(selected_index)
                            
                            # Generate explanation
                            explanation = explainer.explain_instance(X, rel_index)
                            
                            # Display LIME explanation
                            st.subheader("Feature Importance (LIME)")
                            
                            # Plot explanation
                            fig = explainer.plot_explanation(explanation)
                            st.pyplot(fig)
                            
                    else:  # LIME explainer
                        # Get feature matrix for explanation
                        X = df[features]
                        
                        # Create LIME explainer
                        explainer = ExplainerFactory.create_explainer(
                            'lime',
                            model=model,
                            feature_names=features,
                            training_data=X
                        )
                        explainer_name = "LIME"
                        
                        # Get the index of the selected anomaly in the original dataset
                        rel_index = df.index.get_loc(selected_index)
                        
                        # Generate explanation
                        explanation = explainer.explain_instance(X, rel_index)
                        
                        # Display LIME explanation
                        st.subheader("Feature Importance (LIME)")
                        
                        # Plot explanation
                        fig = explainer.plot_explanation(explanation)
                        st.pyplot(fig)
                    
                    # Store explanation in session state
                    st.session_state.current_explanation = {
                        "anomaly_index": selected_index,
                        "explanation": explanation,
                        "explainer_type": explainer_name,
                        "model_type": model_type,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    st.success(f"Successfully generated {explainer_name} explanation!")
                    
                    # Show anomaly details with context
                    st.subheader("Anomaly Context")
                    
                    # Get the context (surrounding records)
                    context_window = 5  # Number of records before and after
                    start_idx = max(0, selected_index - context_window)
                    end_idx = min(len(df), selected_index + context_window + 1)
                    
                    # Get records before and after the anomaly
                    context_df = df.loc[range(start_idx, end_idx)].copy()
                    
                    # Mark the anomaly in the context
                    context_df['is_selected_anomaly'] = context_df.index == selected_index
                    
                    # Format the context dataframe for display
                    display_cols = ['is_selected_anomaly']
                    if time_col:
                        display_cols.append(time_col)
                    if src_ip_col:
                        display_cols.append(src_ip_col)
                    if dst_ip_col:
                        display_cols.append(dst_ip_col)
                    if protocol_col:
                        display_cols.append(protocol_col)
                    
                    # Add other interesting columns
                    for col in ['anomaly_score', '_ws_col_Info', 'frame.len']:
                        if col in context_df.columns:
                            display_cols.append(col)
                    
                    # Display the context
                    st.write("Context (surrounding records):")
                    context_clean = clean_dataframe_for_arrow(context_df[display_cols])
                    st.dataframe(
                        context_clean,
                        use_container_width=True,
                        height=300
                    )
                    
                except Exception as e:
                    st.error(f"Error generating explanation: {str(e)}")
                    st.exception(e)
                    
            # Show recommendation for action
            if st.session_state.get('current_explanation') is not None:
                st.subheader("Recommendation")
                
                # Get anomaly score
                score = anomalies_display.loc[selected_index, 'anomaly_score']
                
                if score > 0.9:
                    st.error("⚠️ **Critical anomaly detected**. Immediate investigation recommended.")
                elif score > 0.7:
                    st.warning("🚨 **High-risk anomaly detected**. Investigation recommended.")
                elif score > 0.5:
                    st.info("ℹ️ **Potential anomaly detected**. Review when possible.")
                else:
                    st.success("✅ **Low-risk anomaly**. Likely a false positive.")
    
    # Anomaly Feedback Tab
    with tab2:
        st.subheader("Provide Feedback on Anomalies")
        
        # Check if we have anomalies
        if anomalies.empty:
            st.info("No anomalies to provide feedback on. Please run anomaly detection first.")
            return
        
        # Initialize feedback storage if not exists
        if 'feedback' not in st.session_state:
            st.session_state.feedback = {}
        
        # Sort anomalies by score
        anomalies_sorted = anomalies.sort_values('anomaly_score', ascending=False)
        
        # Display anomalies table with pagination
        items_per_page = 5
        num_pages = (len(anomalies_sorted) + items_per_page - 1) // items_per_page
        
        if num_pages > 1:
            page_number = st.selectbox("Page", options=range(1, num_pages + 1), index=0, key="feedback_page")
            start_idx = (page_number - 1) * items_per_page
        else:
            start_idx = 0
        
        end_idx = min(start_idx + items_per_page, len(anomalies_sorted))
        
        # Show the anomalies table for feedback
        anomalies_for_feedback = anomalies_sorted.iloc[start_idx:end_idx].copy()
        
        # Determine columns to display
        display_cols = ['anomaly_score']
        
        # Add timestamp if available
        time_col, _ = find_timestamp_column(anomalies_for_feedback)
        if time_col:
            display_cols.insert(0, time_col)
        
        # Add IP columns if available
        src_ip_col = next((col for col in ['ip_src', 'ip.src'] if col in anomalies_for_feedback.columns), None)
        if src_ip_col:
            display_cols.append(src_ip_col)
            
        dst_ip_col = next((col for col in ['ip_dst', 'ip.dst'] if col in anomalies_for_feedback.columns), None)
        if dst_ip_col:
            display_cols.append(dst_ip_col)
        
        # Add protocol column if available
        protocol_col = next((col for col in ['_ws_col_Protocol', 'protocol'] if col in anomalies_for_feedback.columns), None)
        if protocol_col:
            display_cols.append(protocol_col)
        
        # Add packet info if available
        if '_ws_col_Info' in anomalies_for_feedback.columns:
            display_cols.append('_ws_col_Info')
        
        # Show feedback form for each anomaly
        for idx, row in anomalies_for_feedback.iterrows():
            # Create enhanced title with key information
            src_ip = row.get(src_ip_col, 'N/A') if src_ip_col else 'N/A'
            dst_ip = row.get(dst_ip_col, 'N/A') if dst_ip_col else 'N/A'
            protocol = row.get(protocol_col, 'N/A') if protocol_col else 'N/A'
            
            anomaly_title = f"Anomaly {idx} | Score: {row['anomaly_score']:.4f} | {src_ip} → {dst_ip} ({protocol})"
            
            with st.expander(anomaly_title):
                # Enhanced anomaly details display
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Network Information:**")
                    if src_ip_col and src_ip != 'N/A':
                        st.write(f"• **Source IP:** {src_ip}")
                    if dst_ip_col and dst_ip != 'N/A':
                        st.write(f"• **Destination IP:** {dst_ip}")
                    if protocol_col and protocol != 'N/A':
                        st.write(f"• **Protocol:** {protocol}")
                    
                    # Add port information if available
                    src_port_col = next((col for col in ['tcp_srcport', 'udp_srcport', 'tcp.srcport', 'udp.srcport'] if col in row.index), None)
                    dst_port_col = next((col for col in ['tcp_dstport', 'udp_dstport', 'tcp.dstport', 'udp.dstport'] if col in row.index), None)
                    if src_port_col:
                        src_port = row.get(src_port_col)
                        if pd.notna(src_port) and src_port != '':
                            st.write(f"• **Source Port:** {src_port}")
                        else:
                            st.write("• **Source Port:** N/A")
                    if dst_port_col:
                        dst_port = row.get(dst_port_col)
                        if pd.notna(dst_port) and dst_port != '':
                            st.write(f"• **Destination Port:** {dst_port}")
                        else:
                            st.write("• **Destination Port:** N/A")
                
                with col2:
                    st.write("**Anomaly Details:**")
                    st.write(f"• **Anomaly Score:** {row['anomaly_score']:.6f}")
                    if time_col:
                        st.write(f"• **Timestamp:** {row.get(time_col, 'N/A')}")
                    
                    # Add packet length/size if available
                    length_col = next((col for col in ['frame_len', 'ip_len', 'frame.len', 'ip.len'] if col in row.index), None)
                    if length_col:
                        st.write(f"• **Packet Length:** {row.get(length_col, 'N/A')} bytes")
                    
                    # Add packet info if available
                    if '_ws_col_Info' in row.index:
                        st.write(f"• **Info:** {row.get('_ws_col_Info', 'N/A')}")
                
                # Show detailed packet data
                st.write("**📋 Detailed Packet Data:**")
                if st.button(f"Toggle All Data View", key=f"toggle_data_{idx}"):
                    # Toggle state for this specific anomaly
                    toggle_key = f"show_data_{idx}"
                    st.session_state[toggle_key] = not st.session_state.get(toggle_key, False)
                
                # Show data if toggled on
                if st.session_state.get(f"show_data_{idx}", False):
                    # Enhanced detailed view with key network info prominently displayed
                    st.markdown("**🌐 Key Network Details:**")
                    
                    # Create a summary row with the most important information
                    key_info = {}
                    if src_ip_col and src_ip != 'N/A':
                        key_info['Source IP'] = src_ip
                    if dst_ip_col and dst_ip != 'N/A':
                        key_info['Destination IP'] = dst_ip
                    if protocol_col and protocol != 'N/A':
                        key_info['Protocol'] = protocol
                    
                    # Add port information
                    src_port_col = next((col for col in ['tcp_srcport', 'udp_srcport', 'tcp.srcport', 'udp.srcport'] if col in row.index), None)
                    dst_port_col = next((col for col in ['tcp_dstport', 'udp_dstport', 'tcp.dstport', 'udp.dstport'] if col in row.index), None)
                    if src_port_col and pd.notna(row.get(src_port_col)):
                        key_info['Source Port'] = row.get(src_port_col)
                    if dst_port_col and pd.notna(row.get(dst_port_col)):
                        key_info['Destination Port'] = row.get(dst_port_col)
                    
                    # Add timestamp and packet info
                    if time_col and pd.notna(row.get(time_col)):
                        key_info['Timestamp'] = row.get(time_col)
                    if '_ws_col_Info' in row.index and pd.notna(row.get('_ws_col_Info')):
                        key_info['Packet Info'] = row.get('_ws_col_Info')
                    
                    # Display key info as a nice table
                    if key_info:
                        # Convert values to string to avoid Arrow serialization issues
                        key_info_clean = {}
                        for k, v in key_info.items():
                            if pd.isna(v):
                                key_info_clean[k] = 'N/A'
                            elif isinstance(v, (pd.Timestamp, datetime)):
                                key_info_clean[k] = str(v)
                            else:
                                key_info_clean[k] = str(v)
                        
                        key_df = pd.DataFrame(list(key_info_clean.items()), columns=['Field', 'Value'])
                        st.dataframe(key_df, use_container_width=True, hide_index=True)
                    
                    # Show full raw data
                    st.markdown("**📊 Complete Raw Data:**")
                    # Convert problematic data types to avoid Arrow issues
                    display_data = row[display_cols].copy()
                    for col in display_data.index:
                        val = display_data[col]
                        if pd.isna(val):
                            display_data[col] = 'N/A'
                        elif isinstance(val, (pd.Timestamp, datetime)):
                            display_data[col] = str(val)
                        elif isinstance(val, (list, dict, np.ndarray)):
                            display_data[col] = str(val)
                        elif hasattr(val, 'dtype') and 'object' in str(val.dtype):
                            display_data[col] = str(val)
                    
                    st.dataframe(display_data.to_frame().T, use_container_width=True)
                
                # Check if we already have feedback for this anomaly
                existing_feedback = st.session_state.feedback.get(idx, {})
                
                # Feedback form
                col1, col2 = st.columns(2)
                
                with col1:
                    # Classification
                    classification = st.selectbox(
                        "Classification",
                        options=["True Positive", "False Positive", "Uncertain"],
                        index=0 if existing_feedback.get('classification') == "True Positive" else 
                               1 if existing_feedback.get('classification') == "False Positive" else 
                               2,
                        key=f"classification_{idx}"
                    )
                    
                    # Priority
                    priority = st.selectbox(
                        "Priority",
                        options=["Critical", "High", "Medium", "Low"],
                        index=0 if existing_feedback.get('priority') == "Critical" else 
                               1 if existing_feedback.get('priority') == "High" else 
                               2 if existing_feedback.get('priority') == "Medium" else 
                               3,
                        key=f"priority_{idx}"
                    )
                
                with col2:
                    # MITRE ATT&CK technique
                    mitre_techniques = load_mitre_data().get('techniques', [])
                    technique_options = ["None"] + sorted([t.get('name', 'Unknown') for t in mitre_techniques if t.get('name')])
                    
                    technique = st.selectbox(
                        "MITRE ATT&CK Technique",
                        options=technique_options,
                        index=technique_options.index(existing_feedback.get('technique')) if existing_feedback.get('technique') in technique_options else 0,
                        key=f"technique_{idx}"
                    )
                    
                    # Action taken
                    action_taken = st.selectbox(
                        "Action Taken",
                        options=["None", "Investigated", "Blocked", "Added to watchlist", "False positive - ignored"],
                        index=0 if existing_feedback.get('action_taken') == "None" else 
                               1 if existing_feedback.get('action_taken') == "Investigated" else 
                               2 if existing_feedback.get('action_taken') == "Blocked" else 
                               3 if existing_feedback.get('action_taken') == "Added to watchlist" else 
                               4,
                        key=f"action_{idx}"
                    )
                
                # Comments
                comments = st.text_area(
                    "Comments",
                    value=existing_feedback.get('comments', ''),
                    height=100,
                    key=f"comments_{idx}"
                )
                
                # Submit button
                if st.button("Save Feedback", key=f"save_{idx}"):
                    # Get risk information if available
                    risk_score = None
                    risk_level = None
                    if 'risk_scores' in st.session_state and st.session_state.risk_scores:
                        risk_results = st.session_state.risk_scores
                        if 'individual_scores' in risk_results:
                            risk_map = {item['anomaly_id']: item for item in risk_results['individual_scores']}
                            if idx in risk_map:
                                risk_score = risk_map[idx]['risk_score']
                                risk_level = risk_map[idx]['risk_level']
                    
                    # Create feedback entry
                    feedback_entry = {
                        "anomaly_id": idx,
                        "anomaly_score": float(row['anomaly_score']),
                        "risk_score": risk_score,
                        "risk_level": risk_level,
                        "classification": classification,
                        "priority": priority,
                        "technique": technique if technique != "None" else None,
                        "action_taken": action_taken,
                        "comments": comments,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "analyst": "current_user"  # In a real system, get the actual user
                    }
                    
                    # Store feedback in session state
                    st.session_state.feedback[idx] = feedback_entry
                    
                    # Save feedback using unified feedback manager
                    from core.feedback_manager import FeedbackManager
                    feedback_manager = FeedbackManager()
                    
                    # Convert feedback entry to the format expected by FeedbackManager
                    feedback_data = {
                        "classification": feedback_entry.get("classification"),
                        "priority": feedback_entry.get("priority"),
                        "technique": feedback_entry.get("technique"),
                        "action_taken": feedback_entry.get("action_taken"),
                        "comments": feedback_entry.get("comments"),
                        "analyst": feedback_entry.get("analyst"),
                        "anomaly_score": feedback_entry.get("anomaly_score"),
                        "risk_score": feedback_entry.get("risk_score"),
                        "risk_level": feedback_entry.get("risk_level")
                    }
                    
                    # Use anomaly_id as the key, with fallback to index
                    anomaly_id = feedback_entry.get("anomaly_id", f"anomaly_{idx}")
                    feedback_manager.add_feedback(str(anomaly_id), feedback_data)
                    
                    st.success("Feedback saved successfully!")
        
        # Display all feedback
        if st.session_state.feedback:
            st.subheader("All Feedback")
            
            # Convert feedback dictionary to DataFrame
            feedback_df = pd.DataFrame.from_dict(st.session_state.feedback, orient='index')
            
            # Sort by timestamp
            if 'timestamp' in feedback_df.columns:
                feedback_df = feedback_df.sort_values('timestamp', ascending=False)
            
            # Display feedback
            feedback_clean = clean_dataframe_for_arrow(feedback_df)
            st.dataframe(feedback_clean, use_container_width=True)
            
            # Model Retraining Section
            st.subheader("Model Retraining")
            
            st.write("### Retrain Model with Analyst Feedback")
            
            # Count feedback types
            true_positives = len(feedback_df[feedback_df['classification'] == 'True Positive'])
            false_positives = len(feedback_df[feedback_df['classification'] == 'False Positive'])
            uncertain = len(feedback_df[feedback_df['classification'] == 'Uncertain'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("True Positives", true_positives)
            with col2:
                st.metric("False Positives", false_positives)
            with col3:
                st.metric("Uncertain", uncertain)
            
            # Retraining options
            st.write("#### Retraining Options")
            
            retrain_method = st.selectbox(
                "Select retraining method:",
                options=[
                    "Update contamination parameter based on feedback",
                    "Retrain with feedback as labels (supervised)",
                    "Adjust threshold based on feedback"
                ]
            )
            
            # Show current model info
            if model:
                st.write("**Current Model:**")
                metadata = model.metadata if hasattr(model, 'metadata') else {}
                st.write(f"- Type: {metadata.get('type', type(model).__name__)}")
                st.write(f"- Current Contamination: {metadata.get('contamination', 'Unknown')}")
                st.write(f"- Current Threshold: {st.session_state.get('anomaly_threshold', 'Unknown')}")
            
            if st.button("Retrain Model with Feedback", type="primary"):
                with st.spinner("Retraining model with analyst feedback..."):
                    try:
                        # Import model manager
                        from core.model_manager import ModelManager
                        model_manager = ModelManager()
                        
                        # Get current data and features
                        X = df[features].copy()
                        
                        # Handle missing values before retraining
                        if X.isna().any().any():
                            st.warning("Found missing values in features. Filling with median values...")
                            X = X.fillna(X.median())
                            # If there are still NaNs (columns with all NaNs), fill with 0
                            if X.isna().any().any():
                                X = X.fillna(0)
                        
                        # Get the model type for proper saving
                        model_type = st.session_state.get('selected_model', 'Unknown')
                        model_type_name = model_type.replace(" ", "")
                        
                        # Update model metadata to track retraining
                        if not hasattr(model, 'metadata'):
                            model.metadata = {}
                        
                        # Track feedback-based retraining
                        model.metadata.update({
                            "feedback_retrained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "feedback_count": len(feedback_df),
                            "true_positives": true_positives,
                            "false_positives": false_positives,
                            "retraining_method": retrain_method,
                            "retrain_count": model.metadata.get("retrain_count", 0) + 1
                        })
                        
                        if retrain_method == "Update contamination parameter based on feedback":
                            # Calculate new contamination based on feedback
                            total_feedback = len(feedback_df)
                            if total_feedback > 0:
                                # New contamination = (true positives) / total analyzed
                                new_contamination = max(0.001, min(0.2, true_positives / total_feedback))
                                
                                # Update model contamination
                                if hasattr(model, 'contamination'):
                                    model.contamination = new_contamination
                                
                                # Retrain with new contamination
                                model.fit(X)
                                
                                # Update metadata
                                model.metadata["contamination"] = new_contamination
                                model.metadata["retrained_with_feedback"] = True
                                
                                st.success(f"✅ **Model updated** with contamination: {new_contamination:.3f} (Retrain #{str(model.metadata['retrain_count'])})")
                                
                        elif retrain_method == "Adjust threshold based on feedback":
                            # Adjust threshold to minimize false positives
                            scores = st.session_state.get('anomaly_scores', [])
                            if len(scores) > 0:
                                # Get scores for feedback items
                                fp_scores = []
                                tp_scores = []
                                
                                for idx, feedback in st.session_state.feedback.items():
                                    if idx < len(scores):
                                        if feedback['classification'] == 'False Positive':
                                            fp_scores.append(scores[idx])
                                        elif feedback['classification'] == 'True Positive':
                                            tp_scores.append(scores[idx])
                                
                                if fp_scores and tp_scores:
                                    # New threshold: between max FP and min TP
                                    max_fp = max(fp_scores)
                                    min_tp = min(tp_scores)
                                    new_threshold = (max_fp + min_tp) / 2
                                else:
                                    # Fallback: adjust based on FP rate
                                    fp_rate = false_positives / total_feedback if total_feedback > 0 else 0
                                    current_threshold = st.session_state.get('anomaly_threshold', np.percentile(scores, 90))
                                    new_threshold = current_threshold * (1 + fp_rate)
                                
                                # Update threshold
                                st.session_state.anomaly_threshold = new_threshold
                                model.metadata["anomaly_threshold"] = new_threshold
                                model.metadata["retrained_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                
                                st.success(f"Threshold adjusted to: {new_threshold:.4f}")
                        
                        elif retrain_method == "Retrain with feedback as labels (supervised)":
                            # Create labels from feedback
                            y = np.zeros(len(X))  # Default: normal
                            
                            for idx, feedback in st.session_state.feedback.items():
                                if idx < len(y):
                                    if feedback['classification'] == 'True Positive':
                                        y[idx] = 1  # Anomaly
                                    elif feedback['classification'] == 'False Positive':
                                        y[idx] = 0  # Normal
                            
                            # For unsupervised models, we can't directly use labels
                            # Instead, adjust the contamination based on the label ratio
                            anomaly_ratio = np.sum(y) / len(y) if len(y) > 0 else 0.01
                            new_contamination = max(0.001, min(0.2, anomaly_ratio))
                            
                            if hasattr(model, 'contamination'):
                                model.contamination = new_contamination
                            
                            # Retrain
                            model.fit(X)
                            
                            # Update metadata
                            model.metadata.update({
                                "contamination": new_contamination,
                                "supervised_retrain": True,
                                "anomaly_ratio_from_feedback": anomaly_ratio
                            })
                            
                            st.success(f"✅ **Model updated** with supervised feedback. New contamination: {new_contamination:.3f} (Retrain #{str(model.metadata['retrain_count'])})")
                        
                        # Save the updated model (instead of creating new one)
                        model_path = model_manager.save_model(model, create_backup=True)
                        st.success(f"✅ **Updated model saved** to: {model_path}")
                        
                        # Update session state with the updated model
                        st.session_state.anomaly_model = model
                        
                        # Re-run anomaly detection with retrained model
                        scores = model.predict(X)
                        threshold = st.session_state.get('anomaly_threshold', np.percentile(scores, 90))
                        
                        # Identify new anomalies
                        is_anomaly = scores > threshold
                        anomaly_indices = np.where(is_anomaly)[0]
                        anomalies = X.iloc[anomaly_indices].copy() if len(anomaly_indices) > 0 else pd.DataFrame()
                        
                        # Update session state with new results
                        st.session_state.anomaly_scores = scores
                        st.session_state.anomalies = anomalies
                        
                        st.info(f"New analysis complete: {len(anomalies)} anomalies detected with retrained model.")
                        
                    except Exception as e:
                        st.error(f"Error retraining model: {str(e)}")
                        st.exception(e)
            
            # Export feedback
            if st.button("Export All Feedback"):
                # Export as CSV
                csv = feedback_df.to_csv(index=False)
                st.download_button(
                    label="Download Feedback CSV",
                    data=csv,
                    file_name=f"anomaly_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # Advanced Analytics Tab 
    with tab3:
        st.subheader("Advanced Analytics & Intelligence")
        
        # Create sub-tabs for different analytics
        analytics_tab1, analytics_tab2, analytics_tab3, analytics_tab4 = st.tabs([
            "Anomaly Patterns", "Temporal Analysis", "Network Intelligence", "Model Insights"
        ])
        
        with analytics_tab1:
            st.write("### Anomaly Pattern Analysis")
            
            if not anomalies.empty and st.session_state.feedback:
                # Analyze patterns in anomalies
                feedback_df = pd.DataFrame.from_dict(st.session_state.feedback, orient='index')
                
                # Pattern analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("#### Classification Distribution")
                    if 'classification' in feedback_df.columns:
                        classification_counts = feedback_df['classification'].value_counts()
                        
                        import plotly.express as px
                        fig = px.pie(
                            values=classification_counts.values,
                            names=classification_counts.index,
                            title="Anomaly Classification Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write("#### Priority Distribution")
                    if 'priority' in feedback_df.columns:
                        priority_counts = feedback_df['priority'].value_counts()
                        
                        fig = px.bar(
                            x=priority_counts.index,
                            y=priority_counts.values,
                            title="Anomaly Priority Distribution",
                            labels={'x': 'Priority', 'y': 'Count'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Score vs Classification Analysis
                st.write("#### Anomaly Score vs Analyst Classification")
                if 'classification' in feedback_df.columns and 'anomaly_score' in feedback_df.columns:
                    fig = px.box(
                        feedback_df,
                        x='classification',
                        y='anomaly_score',
                        title="Anomaly Score Distribution by Classification",
                        color='classification'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate model accuracy metrics
                    st.write("#### Model Performance Metrics")
                    
                    true_positives = len(feedback_df[feedback_df['classification'] == 'True Positive'])
                    false_positives = len(feedback_df[feedback_df['classification'] == 'False Positive'])
                    total_feedback = len(feedback_df[feedback_df['classification'] != 'Uncertain'])
                    
                    if total_feedback > 0:
                        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                        recall_estimate = true_positives / total_feedback  # This is an estimate
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Estimated Precision", f"{precision:.2%}")
                        with col2:
                            st.metric("True Positive Rate", f"{recall_estimate:.2%}")
                        with col3:
                            st.metric("False Positive Rate", f"{false_positives/total_feedback:.2%}")
            else:
                st.info("Provide feedback on anomalies to see pattern analysis.")
        
        with analytics_tab2:
            st.write("### Temporal Analysis")
            
            if not anomalies.empty:
                # Time-based analysis
                time_col, _ = find_timestamp_column(df)
                
                if time_col:
                    st.write("#### Anomaly Distribution Over Time")
                    
                    # Get timestamps for anomalies
                    anomaly_timestamps = df.loc[anomalies.index, time_col]
                    
                    # Convert to datetime if needed
                    try:
                        anomaly_timestamps = pd.to_datetime(anomaly_timestamps)
                        
                        # Create time-based bins
                        time_bins = st.selectbox(
                            "Select time granularity:",
                            options=["Hour", "Day", "Week"],
                            index=1
                        )
                        
                        if time_bins == "Hour":
                            anomaly_timestamps = anomaly_timestamps.dt.floor('H')
                        elif time_bins == "Day":
                            anomaly_timestamps = anomaly_timestamps.dt.floor('D')
                        else:  # Week
                            anomaly_timestamps = anomaly_timestamps.dt.floor('W')
                        
                        # Count anomalies per time bin
                        time_counts = anomaly_timestamps.value_counts().sort_index()
                        
                        # Plot time series
                        fig = px.line(
                            x=time_counts.index,
                            y=time_counts.values,
                            title=f"Anomalies Over Time ({time_bins}ly)",
                            labels={'x': 'Time', 'y': 'Number of Anomalies'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detect temporal patterns
                        st.write("#### Temporal Patterns")
                        
                        # Hour of day analysis
                        if len(anomaly_timestamps) > 0:
                            hour_analysis = pd.to_datetime(df.loc[anomalies.index, time_col]).dt.hour.value_counts().sort_index()
                            
                            fig = px.bar(
                                x=hour_analysis.index,
                                y=hour_analysis.values,
                                title="Anomalies by Hour of Day",
                                labels={'x': 'Hour', 'y': 'Count'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Day of week analysis
                            dow_analysis = pd.to_datetime(df.loc[anomalies.index, time_col]).dt.day_name().value_counts()
                            
                            fig = px.bar(
                                x=dow_analysis.index,
                                y=dow_analysis.values,
                                title="Anomalies by Day of Week",
                                labels={'x': 'Day', 'y': 'Count'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    except Exception as e:
                        st.warning(f"Could not process timestamps: {str(e)}")
                else:
                    st.info("No timestamp column found for temporal analysis.")
            else:
                st.info("No anomalies detected for temporal analysis.")
        
        with analytics_tab3:
            st.write("### Network Intelligence")
            
            if not anomalies.empty:
                # Network-based analysis
                src_ip_col = next((col for col in ['ip_src', 'ip.src'] if col in df.columns), None)
                dst_ip_col = next((col for col in ['ip_dst', 'ip.dst'] if col in df.columns), None)
                
                if src_ip_col or dst_ip_col:
                    st.write("#### IP Address Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if src_ip_col:
                            # Top source IPs in anomalies
                            src_ip_counts = df.loc[anomalies.index, src_ip_col].value_counts().head(10)
                            
                            fig = px.bar(
                                x=src_ip_counts.values,
                                y=src_ip_counts.index,
                                orientation='h',
                                title="Top Source IPs in Anomalies",
                                labels={'x': 'Count', 'y': 'Source IP'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        if dst_ip_col:
                            # Top destination IPs in anomalies
                            dst_ip_counts = df.loc[anomalies.index, dst_ip_col].value_counts().head(10)
                            
                            fig = px.bar(
                                x=dst_ip_counts.values,
                                y=dst_ip_counts.index,
                                orientation='h',
                                title="Top Destination IPs in Anomalies",
                                labels={'x': 'Count', 'y': 'Destination IP'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Port analysis
                    st.write("#### Port Analysis")
                    
                    port_cols = [col for col in ['src_port', 'dst_port', 'tcp.srcport', 'tcp.dstport', 'udp.srcport', 'udp.dstport'] if col in df.columns]
                    
                    if port_cols:
                        selected_port_col = st.selectbox("Select port column:", port_cols)
                        
                        port_counts = df.loc[anomalies.index, selected_port_col].value_counts().head(15)
                        
                        fig = px.bar(
                            x=port_counts.index.astype(str),
                            y=port_counts.values,
                            title=f"Top Ports in Anomalies ({selected_port_col})",
                            labels={'x': 'Port', 'y': 'Count'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Protocol analysis
                    protocol_cols = [col for col in ['_ws_col_Protocol', 'protocol', 'ip.proto'] if col in df.columns]
                    
                    if protocol_cols:
                        st.write("#### Protocol Analysis")
                        
                        selected_proto_col = st.selectbox("Select protocol column:", protocol_cols)
                        
                        proto_counts = df.loc[anomalies.index, selected_proto_col].value_counts()
                        
                        fig = px.pie(
                            values=proto_counts.values,
                            names=proto_counts.index,
                            title="Protocol Distribution in Anomalies"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.info("No IP address columns found for network analysis.")
            else:
                st.info("No anomalies detected for network analysis.")
        
        with analytics_tab4:
            st.write("### Model Performance Insights")
            
            # Model metadata analysis
            if model and hasattr(model, 'metadata'):
                metadata = model.metadata
                
                st.write("#### Model Information")
                
                info_col1, info_col2 = st.columns(2)
                
                with info_col1:
                    st.write(f"**Model Type:** {metadata.get('type', 'Unknown')}")
                    st.write(f"**Training Date:** {metadata.get('trained_at', 'Unknown')}")
                    st.write(f"**Features Used:** {len(metadata.get('feature_names', []))}")
                    st.write(f"**Contamination:** {metadata.get('contamination', 'Unknown')}")
                
                with info_col2:
                    st.write(f"**Retrain Count:** {metadata.get('retrain_count', 0)}")
                    st.write(f"**Last Retrained:** {metadata.get('feedback_retrained_at', 'Never')}")
                    st.write(f"**Feedback Count:** {metadata.get('feedback_count', 0)}")
                    st.write(f"**Reuse Count:** {metadata.get('reuse_count', 0)}")
                
                # Feature importance from model
                if hasattr(model, 'get_feature_importance') and not anomalies.empty:
                    st.write("#### Model Feature Importance")
                    
                    try:
                        features = st.session_state.get('anomaly_features', [])
                        if features:
                            X = df[features].fillna(0)
                            importance = model.get_feature_importance(X)
                            
                            if importance:
                                importance_df = pd.DataFrame({
                                    'Feature': list(importance.keys()),
                                    'Importance': list(importance.values())
                                }).sort_values('Importance', ascending=False)
                                
                                fig = px.bar(
                                    importance_df,
                                    x='Feature',
                                    y='Importance',
                                    title="Model Feature Importance",
                                    color='Importance',
                                    color_continuous_scale='viridis'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not calculate feature importance: {str(e)}")
                
                # Anomaly score distribution
                if 'anomaly_scores' in st.session_state:
                    st.write("#### Anomaly Score Distribution")
                    
                    scores = st.session_state['anomaly_scores']
                    threshold = st.session_state.get('anomaly_threshold', 0.5)
                    
                    fig = px.histogram(
                        x=scores,
                        nbins=50,
                        title="Distribution of Anomaly Scores",
                        labels={'x': 'Anomaly Score', 'y': 'Count'}
                    )
                    fig.add_vline(x=threshold, line_dash="dash", line_color="red", 
                                 annotation_text="Threshold")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Score statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean Score", f"{np.mean(scores):.4f}")
                    with col2:
                        st.metric("Std Score", f"{np.std(scores):.4f}")
                    with col3:
                        st.metric("Max Score", f"{np.max(scores):.4f}")
                    with col4:
                        st.metric("Min Score", f"{np.min(scores):.4f}")
            else:
                st.info("No model metadata available for analysis.")
        
        # Enhanced Features Section (if available)
        if ENHANCED_FEATURES_AVAILABLE:
            st.subheader("🚀 Enhanced Analytics")
            
            # Initialize enhanced analytics components
            if 'threat_intelligence_manager' not in st.session_state:
                st.session_state.threat_intelligence_manager = ThreatIntelligenceManager()
            
            if 'behavioral_analytics' not in st.session_state:
                st.session_state.behavioral_analytics = BehavioralAnalytics()
            
            if 'predictive_analytics' not in st.session_state:
                st.session_state.predictive_analytics = PredictiveAnalytics()
            
            # Enhanced analytics tabs
            enhanced_tab1, enhanced_tab2, enhanced_tab3 = st.tabs([
                "🛡️ Threat Intelligence", "🧠 Behavioral Analytics", "🔮 Predictive Analytics"
            ])
            
            with enhanced_tab1:
                st.write("### Threat Intelligence & IOC Correlation")
                
                # Threat intelligence dashboard
                display_threat_intelligence_dashboard(
                    st.session_state.threat_intelligence_manager
                )
                
                # Enrich current anomalies with threat intelligence
                if not anomalies.empty:
                    if st.button("🔍 Enrich Anomalies with Threat Intel"):
                        with st.spinner("Enriching anomalies with security intelligence..."):
                            enriched_anomalies = st.session_state.threat_intelligence_manager.enrich_anomalies(anomalies)
                            
                            # Display enriched results
                            if 'threat_score' in enriched_anomalies.columns:
                                high_threat = enriched_anomalies[enriched_anomalies['threat_score'] > 0.5]
                                
                                if not high_threat.empty:
                                    st.warning(f"🚨 Found {len(high_threat)} anomalies with threat intelligence matches!")
                                    
                                    # Display threat matches
                                    threat_cols = ['threat_score', 'threat_types', 'ioc_sources', 'security_risk_score']
                                    available_cols = [col for col in threat_cols if col in enriched_anomalies.columns]
                                    
                                    display_cols = list(enriched_anomalies.columns[:5]) + available_cols
                                    high_threat_clean = clean_dataframe_for_arrow(high_threat[display_cols])
                                    st.dataframe(
                                        high_threat_clean, 
                                        use_container_width=True
                                    )
                                    
                                    # Security risk visualization
                                    from core.security_intelligence import create_security_risk_visualization
                                    risk_plots = create_security_risk_visualization(enriched_anomalies)
                                    
                                    for plot_name, fig in risk_plots.items():
                                        st.plotly_chart(fig, use_container_width=True)
                                
                                else:
                                    st.success("✅ No threat intelligence matches found in current anomalies")
                            else:
                                st.info("Threat intelligence enrichment completed, but no matches found")
            
            with enhanced_tab2:
                st.write("### Behavioral Baseline Analysis")
                
                # Display behavioral analytics dashboard
                display_advanced_analytics_dashboard(
                    st.session_state.behavioral_analytics,
                    st.session_state.predictive_analytics,
                    df
                )
                
                # Enhanced explanations for selected anomaly
                if not anomalies.empty:
                    st.write("#### 🧠 Enhanced Anomaly Explanations")
                    
                    def format_anomaly_option(x):
                        """Format anomaly option for selectbox"""
                        score = anomalies.iloc[x].get('anomaly_score', 'N/A')
                        if isinstance(score, (int, float)) and score != 'N/A':
                            return f"Anomaly {x+1} (Score: {score:.3f})"
                        else:
                            return f"Anomaly {x+1} (Score: {score})"
                    
                    selected_anomaly_idx = st.selectbox(
                        "Select anomaly for enhanced explanation:",
                        options=range(len(anomalies)),
                        format_func=format_anomaly_option
                    )
                    
                    if st.button("🔍 Generate Enhanced Explanation"):
                        anomaly_row_idx = anomalies.index[selected_anomaly_idx]
                        
                        if model and hasattr(model, 'feature_names'):
                            feature_names = model.feature_names
                        else:
                            feature_names = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
                        
                        model_type = type(model).__name__ if model else "Unknown"
                        anomaly_score = anomalies.iloc[selected_anomaly_idx].get('anomaly_score', 0.0)
                        
                        # Get filtered feature data for the anomalies and find positional index
                        feature_data = df[feature_names]
                        anomaly_feature_data = feature_data.loc[anomalies.index]
                        
                        # Convert DataFrame index to positional index for explanation functions
                        positional_idx = selected_anomaly_idx  # This is already the position in the anomalies list
                        
                        # Display enhanced explanations
                        display_enhanced_explanations(
                            model=model,
                            X=anomaly_feature_data,
                            instance_idx=positional_idx,
                            feature_names=feature_names,
                            model_type=model_type,
                            anomaly_score=anomaly_score
                        )
            
            with enhanced_tab3:
                st.write("### Predictive Analytics & Forecasting")
                
                # Time-based analysis for predictions
                time_col, _ = find_timestamp_column(df)
                
                if time_col:
                    st.write(f"Using time column: **{time_col}**")
                    
                    # Verify the column actually exists in current data
                    if time_col not in df.columns:
                        st.error(f"Time column '{time_col}' was detected but not found in current data. Available columns: {list(df.columns)}")
                        time_col = None
                
                if time_col:
                    # Forecast anomaly trends
                    forecast_hours = st.slider("Forecast horizon (hours):", 1, 72, 24, key="forecast_hours")
                    
                    if st.button("📈 Generate Anomaly Forecast"):
                        with st.spinner("Generating predictive analytics..."):
                            # Use anomalies for forecasting
                            forecast_data = st.session_state.predictive_analytics.forecast_anomaly_trends(
                                anomalies if not anomalies.empty else df,
                                time_col,
                                forecast_hours
                            )
                            
                            if 'error' in forecast_data:
                                st.error(f"Forecasting error: {forecast_data['error']}")
                            else:
                                # Display forecast visualization
                                from core.advanced_analytics import create_predictive_analytics_plots
                                plots = create_predictive_analytics_plots(forecast_data)
                                
                                for plot_name, fig in plots.items():
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Display insights
                                if 'insights' in forecast_data and forecast_data['insights']:
                                    st.subheader("🎯 Predictive Insights")
                                    
                                    for insight in forecast_data['insights']:
                                        insight_text = f"""
                                        **{insight.prediction_type.replace('_', ' ').title()}**
                                        - Confidence: {insight.confidence:.1%}
                                        - Time Horizon: {insight.time_horizon}
                                        - Recommendation: {insight.recommendation}
                                        """
                                        st.info(insight_text)
                                
                                # Overall forecast confidence
                                confidence = forecast_data.get('confidence', 0.0)
                                st.metric("Forecast Confidence", f"{confidence:.1%}")
                
                # Risk escalation predictions (regardless of time column availability)
                st.write("#### 🚨 Risk Escalation Analysis")
                
                if st.button("⚠️ Analyze Risk Escalation Patterns"):
                    with st.spinner("Analyzing risk escalation patterns..."):
                        # Simple risk escalation analysis based on anomaly trends
                        if not anomalies.empty and time_col and time_col in anomalies.columns:
                            # Group anomalies by time periods
                            anomalies_with_time = anomalies.copy()
                            anomalies_with_time[time_col] = pd.to_datetime(anomalies_with_time[time_col])
                            
                            # Calculate hourly anomaly counts
                            hourly_counts = anomalies_with_time.groupby(
                                anomalies_with_time[time_col].dt.floor('H')
                            ).size()
                            
                            if len(hourly_counts) > 1:
                                # Calculate trend
                                x = np.arange(len(hourly_counts))
                                trend_coef = np.polyfit(x, hourly_counts.values, 1)[0]
                                
                                if trend_coef > 0.1:
                                    st.warning(f"⚠️ **Escalating Risk Detected**: Anomaly count increasing by {trend_coef:.2f} per hour")
                                    st.write("**Recommended Actions:**")
                                    st.write("- Increase monitoring frequency")
                                    st.write("- Prepare incident response team")
                                    st.write("- Review recent security events")
                                elif trend_coef < -0.1:
                                    st.success(f"✅ **Risk De-escalation**: Anomaly count decreasing by {abs(trend_coef):.2f} per hour")
                                else:
                                    st.info("📊 **Stable Risk Level**: No significant trend in anomaly frequency")
                                
                                # Visualize trend
                                import plotly.express as px
                                
                                trend_fig = px.line(
                                    x=hourly_counts.index,
                                    y=hourly_counts.values,
                                    title="Anomaly Count Trend",
                                    labels={'x': 'Time', 'y': 'Anomaly Count'}
                                )
                                
                                # Add trend line
                                trend_line = trend_coef * x + hourly_counts.values[0]
                                trend_fig.add_scatter(
                                    x=hourly_counts.index,
                                    y=trend_line,
                                    mode='lines',
                                    name='Trend',
                                    line=dict(dash='dash', color='red')
                                )
                                
                                st.plotly_chart(trend_fig, use_container_width=True)
                            else:
                                st.info("Insufficient temporal data for risk escalation analysis")
                        else:
                            st.info("⚠️ Risk escalation analysis requires temporal data. Please ensure your data contains a time column.")
                else:
                    st.warning("⚠️ No suitable time column found in the data. Forecasting is not available.")
                    st.info("💡 **Tip**: Ensure your data has a column with timestamp information (e.g., 'timestamp', 'frame.time', 'datetime', etc.)")
        
        else:
            st.info("💡 Enhanced features (Threat Intelligence, Behavioral Analytics, Predictive Analytics) are not available. Install required dependencies to enable these features.")

def save_feedback(feedback_entry):
    """Save feedback to a file."""
    # Load config to get proper paths
    from core.config_loader import load_config
    config = load_config()
    
    # Get feedback storage directory from config
    feedback_dir = config.get('feedback', {}).get('storage_dir', 'data/feedback')
    
    # Create feedback directory if it doesn't exist
    os.makedirs(feedback_dir, exist_ok=True)
    
    # Create filename based on date
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = os.path.join(feedback_dir, f"feedback_{date_str}.json")
    
    # Load existing feedback if file exists
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            try:
                feedback_data = json.load(f)
            except json.JSONDecodeError:
                feedback_data = {"feedback": []}
    else:
        feedback_data = {"feedback": []}
    
    # Add new feedback
    feedback_data["feedback"].append(feedback_entry)
    
    # Save feedback to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(feedback_data, f, indent=4)

def create_sample_mitre_data():
    """Create a sample MITRE ATT&CK data file."""
    sample_data = {
        "techniques": [
            {
                "id": "T1046",
                "name": "Network Service Discovery",
                "description": "Adversaries may attempt to get a listing of services running on remote hosts, including those that may be vulnerable to remote software exploitation.",
                "tactic": "Discovery"
            },
            {
                "id": "T1110",
                "name": "Brute Force",
                "description": "Adversaries may use brute force techniques to gain access to accounts when passwords are unknown or when password hashes are obtained.",
                "tactic": "Credential Access"
            },
            {
                "id": "T1059",
                "name": "Command and Scripting Interpreter",
                "description": "Adversaries may abuse command and script interpreters to execute commands, scripts, or binaries.",
                "tactic": "Execution"
            },
            {
                "id": "T1190",
                "name": "Exploit Public-Facing Application",
                "description": "Adversaries may attempt to exploit a public-facing application, which can enable access to systems without requiring separate credentials or user accounts.",
                "tactic": "Initial Access"
            },
            {
                "id": "T1021",
                "name": "Remote Services",
                "description": "Adversaries may use remote services to access and persist within a network environment.",
                "tactic": "Lateral Movement"
            },
            {
                "id": "T1078",
                "name": "Valid Accounts",
                "description": "Adversaries may obtain and abuse credentials of existing accounts as a means of gaining Initial Access, Persistence, Privilege Escalation, or Defense Evasion.",
                "tactic": "Defense Evasion"
            },
            {
                "id": "T1133",
                "name": "External Remote Services",
                "description": "Adversaries may leverage external-facing remote services to initially access or persist within a network.",
                "tactic": "Persistence"
            },
            {
                "id": "T1068",
                "name": "Exploitation for Privilege Escalation",
                "description": "Adversaries may exploit software vulnerabilities in an attempt to elevate privileges.",
                "tactic": "Privilege Escalation"
            },
            {
                "id": "T1213",
                "name": "Data from Information Repositories",
                "description": "Adversaries may leverage information repositories to mine valuable information.",
                "tactic": "Collection"
            },
            {
                "id": "T1020",
                "name": "Automated Exfiltration",
                "description": "Adversaries may exfiltrate data, such as sensitive documents, through automated processing or scripted transfers.",
                "tactic": "Exfiltration"
            },
            {
                "id": "T1041",
                "name": "Exfiltration Over C2 Channel",
                "description": "Adversaries may steal data by exfiltrating it over an existing command and control channel.",
                "tactic": "Exfiltration"
            },
            {
                "id": "T1090",
                "name": "Proxy",
                "description": "Adversaries may use a connection proxy to direct network traffic between systems or act as an intermediary for network communications.",
                "tactic": "Command and Control"
            },
            {
                "id": "T1499",
                "name": "Endpoint Denial of Service",
                "description": "Adversaries may perform endpoint denial of service (DoS) attacks to disrupt access to or impair the availability of targeted resources.",
                "tactic": "Impact"
            },
            {
                "id": "T1505",
                "name": "Server Software Component",
                "description": "Adversaries may abuse server software components to establish persistent access to systems.",
                "tactic": "Persistence"
            },
            {
                "id": "T1189",
                "name": "Drive-by Compromise",
                "description": "Adversaries may gain access to a system through a user visiting a website over the normal course of browsing.",
                "tactic": "Initial Access"
            }
        ],
        "tactics": [
            {
                "id": "TA0043",
                "name": "Reconnaissance",
                "description": "The adversary is trying to gather information they can use to plan future operations."
            },
            {
                "id": "TA0042",
                "name": "Resource Development",
                "description": "The adversary is trying to establish resources they can use to support operations."
            },
            {
                "id": "TA0001",
                "name": "Initial Access",
                "description": "The adversary is trying to get into your network."
            },
            {
                "id": "TA0002",
                "name": "Execution",
                "description": "The adversary is trying to run malicious code."
            },
            {
                "id": "TA0003",
                "name": "Persistence",
                "description": "The adversary is trying to maintain their foothold."
            },
            {
                "id": "TA0004",
                "name": "Privilege Escalation",
                "description": "The adversary is trying to gain higher-level permissions."
            },
            {
                "id": "TA0005",
                "name": "Defense Evasion",
                "description": "The adversary is trying to avoid being detected."
            },
            {
                "id": "TA0006",
                "name": "Credential Access",
                "description": "The adversary is trying to steal account names and passwords."
            },
            {
                "id": "TA0007",
                "name": "Discovery",
                "description": "The adversary is trying to figure out your environment."
            },
            {
                "id": "TA0008",
                "name": "Lateral Movement",
                "description": "The adversary is trying to move through your environment."
            },
            {
                "id": "TA0009",
                "name": "Collection",
                "description": "The adversary is trying to gather data of interest to their goal."
            },
            {
                "id": "TA0011",
                "name": "Command and Control",
                "description": "The adversary is trying to communicate with compromised systems to control them."
            },
            {
                "id": "TA0010",
                "name": "Exfiltration",
                "description": "The adversary is trying to steal data."
            },
            {
                "id": "TA0040",
                "name": "Impact",
                "description": "The adversary is trying to manipulate, interrupt, or destroy your systems and data."
            }
        ]
    }
    
    # Get config directory path and create if it doesn't exist
    config_dir = "config"  # This is correct as it's for configuration files
    os.makedirs(config_dir, exist_ok=True)
    
    # Save sample data to file
    with open("config/mitre_attack_data.json", 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=4)