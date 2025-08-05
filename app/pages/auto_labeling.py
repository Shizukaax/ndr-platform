"""
Auto Labeling page for the Network Anomaly Detection Platform.
Provides AI-powered labeling suggestions based on historical feedback.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging

from core.auto_labeler import AutoLabeler
from core.feedback_manager import FeedbackManager
from core.risk_scorer import RiskScorer
from core.search_engine import SearchEngine
from app.components.error_handler import handle_error
from app.components.chart_factory import chart_factory
from core.notification_service import notification_service

# Set up logger
logger = logging.getLogger("streamlit_app")

@handle_error
def show_auto_labeling():
    """Display the Auto Labeling page."""
    
    st.header("🤖 AI-Powered Auto Labeling")
    st.markdown("Train AI models to automatically suggest labels for anomalies based on your historical feedback.")
    
    # Initialize components
    feedback_manager = FeedbackManager()
    auto_labeler = AutoLabeler(feedback_manager)
    risk_scorer = RiskScorer()
    search_engine = SearchEngine()
    
    # Check if we have session data
    if not st.session_state.get('combined_data') is not None:
        st.info("No data loaded. Please go to the Data Upload page to select JSON files.")
        return
    
    # Check if we have anomalies detected
    if not st.session_state.get('anomaly_scores') is not None:
        st.info("No anomalies detected yet. Please run anomaly detection first.")
        return
    
    # Create tabs - Removed duplicate Risk Scoring tab
    tab1, tab2, tab3 = st.tabs([
        "🧠 Model Training", "🏷️ Auto Label", "🔍 Smart Search"
    ])
    
    with tab1:
        st.subheader("Train Auto-Labeling Models")
        st.markdown("Train AI models based on your historical feedback to automatically suggest labels for new anomalies.")
        
        # Check feedback availability
        feedback_df = feedback_manager.get_feedback_dataframe()
        
        # Also check session state feedback
        session_feedback = st.session_state.get('feedback', {})
        
        # If no feedback in files but exists in session, show session feedback count
        if feedback_df.empty and session_feedback:
            st.info(f"Found {len(session_feedback)} feedback records in current session.")
            st.warning("💾 **Note**: Feedback exists in session but may not be saved to file yet.")
            st.info("💡 **Tip**: Continue using the feedback in session or go to 'Model Explanation & Analyst Feedback' page to save more feedback.")
            
            # Create a temporary DataFrame from session feedback for training
            session_records = []
            for anomaly_id, feedback_data in session_feedback.items():
                if isinstance(feedback_data, dict):
                    record = {"anomaly_id": anomaly_id}
                    record.update(feedback_data)
                    session_records.append(record)
            
            if session_records:
                feedback_df = pd.DataFrame(session_records)
                st.success(f"Using {len(feedback_df)} feedback records from current session for training.")
        
        if feedback_df.empty:
            st.warning("No feedback data available for training. Please provide feedback on some anomalies first.")
            st.info("💡 **Tip**: Go to 'Model Explanation & Analyst Feedback' page to start labeling anomalies.")
        else:
            st.success(f"Found {len(feedback_df)} feedback records for training.")
            
            # Show feedback summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'classification' in feedback_df.columns:
                    classification_counts = feedback_df['classification'].value_counts()
                    st.metric("True Positives", classification_counts.get('True Positive', 0))
            
            with col2:
                if 'category' in feedback_df.columns:
                    unique_categories = feedback_df['category'].nunique()
                    st.metric("Unique Categories", unique_categories)
            
            with col3:
                if 'severity' in feedback_df.columns:
                    unique_severities = feedback_df['severity'].nunique()
                    st.metric("Severity Levels", unique_severities)
            
            # Training options
            st.subheader("Training Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                force_retrain = st.checkbox("Force retrain existing models", value=False)
            with col2:
                min_samples = st.slider("Minimum samples for training", min_value=5, max_value=50, value=10)
            
            # Update auto_labeler minimum samples
            auto_labeler.min_samples = min_samples
            
            # Train models button
            if st.button("🚀 Train Auto-Labeling Models"):
                with st.spinner("Training models..."):
                    training_results = auto_labeler.train_models(force_retrain=force_retrain)
                
                if training_results["status"] == "success":
                    st.success("✅ Models trained successfully!")
                    
                    # Display training results
                    results = training_results["results"]
                    
                    for label_type, result in results.items():
                        with st.expander(f"📈 {label_type.title()} Model Results"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Accuracy", f"{result['accuracy']:.3f}")
                            with col2:
                                st.metric("Model Type", result['model_type'])
                            with col3:
                                st.metric("Training Samples", result['training_samples'])
                            
                            st.write("**Features used:**")
                            st.write(", ".join(result['features']))
                    
                    # Store models in session state
                    st.session_state.auto_labeler = auto_labeler
                else:
                    st.error(f"❌ Training failed: {training_results['message']}")
    
    with tab2:
        st.subheader("Auto Label New Anomalies")
        st.markdown("Use trained models to automatically suggest labels for detected anomalies.")
        
        # Check if models are trained
        if 'auto_labeler' not in st.session_state:
            st.info("No trained models available. Please train models first in the 'Model Training' tab.")
        else:
            auto_labeler = st.session_state.auto_labeler
            
            # Get current anomalies
            if 'anomaly_indices' in st.session_state and 'combined_data' in st.session_state:
                df = st.session_state.combined_data
                anomaly_indices = st.session_state.anomaly_indices
                anomalies = df.iloc[anomaly_indices].copy()
                
                st.success(f"Found {len(anomalies)} anomalies to label.")
                
                # Auto-label button
                if st.button("🏷️ Generate Labels for All Anomalies"):
                    with st.spinner("Generating labels..."):
                        labeled_anomalies = auto_labeler.predict_labels(anomalies)
                    
                    st.success("✅ Labels generated successfully!")
                    
                    # Show labeled anomalies
                    st.subheader("📊 Labeled Anomalies")
                    
                    # Filter columns to show
                    display_cols = ['src_port', 'dst_port', 'protocol_type', 'packet_length']
                    prediction_cols = [col for col in labeled_anomalies.columns if col.startswith('predicted_')]
                    
                    available_display_cols = [col for col in display_cols if col in labeled_anomalies.columns]
                    show_cols = available_display_cols + prediction_cols
                    
                    if show_cols:
                        st.dataframe(labeled_anomalies[show_cols], use_container_width=True)
                    else:
                        st.dataframe(labeled_anomalies, use_container_width=True)
                    
                    # Store labeled anomalies
                    st.session_state.labeled_anomalies = labeled_anomalies
                    
                    # Download option
                    csv = labeled_anomalies.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Labeled Anomalies",
                        data=csv,
                        file_name=f"labeled_anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.info("No anomalies detected. Please run anomaly detection first.")
    
    with tab3:
        st.subheader("🔍 Smart Search Engine")
        st.markdown("Advanced search capabilities with regex support and intelligent filtering.")
        
        # Get data
        df = st.session_state.combined_data
        
        # Search interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "Search across all data:",
                placeholder="e.g., SSH, 192.168.1.*, malware, port:22"
            )
        
        with col2:
            search_options = st.multiselect(
                "Options:",
                ["Case Sensitive", "Regex", "Anomalies Only"],
                default=[]
            )
        
        # Advanced filters
        with st.expander("🔧 Advanced Filters"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'src_port' in df.columns:
                    port_filter = st.text_input("Source Port", placeholder="22,80,443 or 1000-2000")
            
            with col2:
                if 'protocol_type' in df.columns:
                    protocols = df['protocol_type'].unique() if 'protocol_type' in df.columns else []
                    protocol_filter = st.multiselect("Protocols", protocols)
            
            with col3:
                if 'packet_length' in df.columns:
                    size_range = st.slider("Packet Size Range", 0, 2000, (0, 2000))
        
        # Perform search
        if search_query or any(search_options):
            case_sensitive = "Case Sensitive" in search_options
            regex = "Regex" in search_options
            anomalies_only = "Anomalies Only" in search_options
            
            # Start with full dataset or anomalies only
            search_df = df.copy()
            
            if anomalies_only and 'anomaly_indices' in st.session_state:
                search_df = df.iloc[st.session_state.anomaly_indices]
            
            # Apply search query
            if search_query:
                search_df = search_engine.search(
                    search_df, 
                    search_query, 
                    case_sensitive=case_sensitive, 
                    regex=regex,
                    limit=1000
                )
            
            # Apply additional filters
            if 'port_filter' in locals() and port_filter:
                # Parse port filter (e.g., "22,80,443" or "1000-2000")
                if '-' in port_filter:
                    start_port, end_port = map(int, port_filter.split('-'))
                    search_df = search_df[
                        (search_df['src_port'] >= start_port) & 
                        (search_df['src_port'] <= end_port)
                    ]
                else:
                    ports = [int(p.strip()) for p in port_filter.split(',') if p.strip().isdigit()]
                    if ports:
                        search_df = search_df[search_df['src_port'].isin(ports)]
            
            if 'protocol_filter' in locals() and protocol_filter:
                search_df = search_df[search_df['protocol_type'].isin(protocol_filter)]
            
            if 'size_range' in locals():
                search_df = search_df[
                    (search_df['packet_length'] >= size_range[0]) & 
                    (search_df['packet_length'] <= size_range[1])
                ]
            
            # Display results
            st.subheader(f"🎯 Search Results ({len(search_df)} matches)")
            
            if not search_df.empty:
                # Show summary statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if 'protocol_type' in search_df.columns:
                        protocol_counts = search_df['protocol_type'].value_counts()
                        st.write("**Top Protocols:**")
                        st.write(protocol_counts.head())
                
                with col2:
                    if 'src_port' in search_df.columns:
                        port_counts = search_df['src_port'].value_counts()
                        st.write("**Top Ports:**")
                        st.write(port_counts.head())
                
                with col3:
                    if 'packet_length' in search_df.columns:
                        avg_size = search_df['packet_length'].mean()
                        st.metric("Avg Packet Size", f"{avg_size:.0f} bytes")
                
                # Display results table
                st.dataframe(search_df, use_container_width=True)
                
                # Export results
                csv = search_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Export Search Results",
                    data=csv,
                    file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No matches found. Try adjusting your search criteria.")
