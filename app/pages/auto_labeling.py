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
                min_samples = st.slider("Minimum samples for training", min_value=5, max_value=50, value=5)
            
            # Train models button
            if st.button("🚀 Train Auto-Labeling Models"):
                with st.spinner("Training models..."):
                    training_results = auto_labeler.train_models(force_retrain=force_retrain, min_samples=min_samples)
                
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
            
            # Show a quick train button for convenience
            if st.button("🚀 Quick Train Models"):
                with st.spinner("Training models..."):
                    training_result = auto_labeler.train_models(force_retrain=True)
                    if training_result["status"] == "success":
                        st.success("✅ Models trained successfully!")
                        st.session_state.auto_labeler = auto_labeler
                        st.rerun()
                    else:
                        st.error(f"❌ Training failed: {training_result['message']}")
                        st.info("💡 **Tip**: Ensure you have feedback data in the system.")
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
                        # Ensure models are trained first
                        if 'auto_labeler' not in st.session_state or not auto_labeler.models:
                            st.info("Training models first...")
                            training_result = auto_labeler.train_models(force_retrain=True)
                            if training_result["status"] == "error":
                                st.error(f"❌ Model training failed: {training_result['message']}")
                                st.info("💡 **Tip**: Add more feedback data in the 'Model Explanation & Analyst Feedback' page to enable auto-labeling.")
                                return
                            else:
                                st.success("✅ Models trained successfully!")
                        
                        # Generate labels
                        labeled_anomalies = auto_labeler.predict_labels(anomalies)
                        
                        # Verify that predictions were added
                        prediction_cols = [col for col in labeled_anomalies.columns if col.startswith('predicted_')]
                        confidence_cols = [col for col in labeled_anomalies.columns if 'confidence' in col]
                        
                        if not prediction_cols:
                            st.warning("⚠️ No predictions were generated. This may be due to:")
                            st.write("• Insufficient training data")
                            st.write("• Missing required features in anomaly data")
                            st.write("• Model training failure")
                            st.info("💡 **Tip**: Check the Model Training tab for more details.")
                            return
                    
                    st.success("✅ Labels generated successfully!")
                    
                    # Store labeled anomalies in session state immediately
                    st.session_state.labeled_anomalies = labeled_anomalies
                    st.session_state.auto_labeling_complete = True
                
                # Check if we have labeled anomalies (either just generated or from previous run)
                if hasattr(st.session_state, 'labeled_anomalies') and st.session_state.labeled_anomalies is not None:
                    labeled_anomalies = st.session_state.labeled_anomalies
                    
                    # Show status indicator
                    if st.session_state.get('auto_labeling_complete'):
                        st.success("✅ Auto-labeling completed! Results are persistent across interactions.")
                    else:
                        st.info("📊 Displaying previously generated labels.")
                    
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
                    
                    # Check if confidence columns exist before proceeding
                    confidence_col = None
                    possible_confidence_cols = [
                        'predicted_true_positive_confidence',
                        'predicted_category_confidence',
                        'predicted_severity_confidence',
                        'confidence',
                        'prediction_confidence',
                        'score'
                    ]
                    
                    for col in possible_confidence_cols:
                        if col in labeled_anomalies.columns:
                            confidence_col = col
                            break
                    
                    if confidence_col is not None:
                        # Confidence-based workflow
                        st.subheader("🎯 Confirm Predictions")
                        st.markdown("Review and confirm the auto-generated labels to improve future predictions.")
                        
                        # Confidence threshold for bulk confirmation
                        confidence_threshold = st.slider(
                            "Confidence threshold for bulk confirmation:",
                            min_value=0.5,
                            max_value=1.0,
                            value=0.8,
                            step=0.05,
                            help="Predictions above this confidence will be auto-confirmed",
                            key="confidence_threshold_slider"
                        )
                        
                        # Count high-confidence predictions
                        high_conf_count = len(labeled_anomalies[
                            labeled_anomalies[confidence_col] >= confidence_threshold
                        ])
                        st.info(f"🎯 {high_conf_count} predictions above {confidence_threshold:.0%} confidence using column '{confidence_col}'")
                    else:
                        # Fallback workflow without confidence scores
                        st.warning("⚠️ No confidence scores available in predictions. This may be due to insufficient training data or model training issues.")
                        st.subheader("🎯 Review Predictions")
                        st.markdown("Review and confirm the auto-generated labels to improve future predictions.")
                        
                        # Show available prediction columns for debugging
                        pred_cols = [col for col in labeled_anomalies.columns if col.startswith('predicted_')]
                        if pred_cols:
                            st.info(f"Available prediction columns: {', '.join(pred_cols)}")
                        else:
                            st.warning("No prediction columns found. This suggests the auto-labeling models may not have been trained properly.")
                        
                        confidence_threshold = 0.8  # Default value
                        confidence_col = None
                    
                    # Bulk confirmation options
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        confirm_key = f"confirm_high_conf_{len(labeled_anomalies)}"
                        if confidence_col is not None:
                            button_label = "✅ Confirm High-Confidence"
                            button_help = "Accept all predictions above threshold"
                        else:
                            button_label = "✅ Confirm All Predictions"
                            button_help = "Accept all auto-generated predictions"
                        
                        if st.button(button_label, key=confirm_key, help=button_help):
                            # Add predictions as feedback
                            if confidence_col is not None:
                                # Filter by confidence
                                high_conf_mask = labeled_anomalies[confidence_col] >= confidence_threshold
                                confirm_anomalies = labeled_anomalies[high_conf_mask]
                            else:
                                # Confirm all predictions
                                confirm_anomalies = labeled_anomalies
                            
                            confirmed_count = 0
                            success_messages = []
                            
                            for idx, row in confirm_anomalies.iterrows():
                                anomaly_id = f"auto_{idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                
                                # Extract prediction data safely
                                is_true_positive = row.get('predicted_true_positive', True)
                                if isinstance(is_true_positive, str):
                                    is_true_positive = is_true_positive.lower() in ['true', 'yes', '1']
                                
                                confidence_score = row.get(confidence_col, 0.8) if confidence_col else 0.8
                                
                                feedback_data = {
                                    "classification": "True Positive" if is_true_positive else "False Positive",
                                    "priority": "High" if row.get('predicted_priority', False) else "Medium",
                                    "technique": row.get('predicted_category', None),
                                    "action_taken": "Auto-confirmed by AI",
                                    "comments": f"Auto-confirmed with {confidence_score:.1%} confidence" if confidence_col else "Auto-confirmed (no confidence score)",
                                    "analyst": "auto_labeler",
                                    "anomaly_score": row.get('anomaly_score', 0),
                                    "risk_score": 0,
                                    "risk_level": "Medium"
                                }
                                
                                try:
                                    if feedback_manager.add_feedback(anomaly_id, feedback_data):
                                        confirmed_count += 1
                                        success_messages.append(f"Added feedback for anomaly {idx}")
                                except Exception as e:
                                    st.error(f"Error adding feedback for anomaly {idx}: {e}")
                            
                            if confirmed_count > 0:
                                if confidence_col:
                                    st.success(f"✅ Successfully confirmed {confirmed_count} high-confidence predictions!")
                                else:
                                    st.success(f"✅ Successfully confirmed {confirmed_count} predictions!")
                                st.info("These predictions have been added to the feedback database for model training.")
                                
                                # Store recent activity for persistence
                                if confidence_col:
                                    confirmation_msg = f"Confirmed {confirmed_count} predictions with ≥{confidence_threshold:.0%} confidence at {datetime.now().strftime('%H:%M:%S')}"
                                else:
                                    confirmation_msg = f"Confirmed {confirmed_count} predictions at {datetime.now().strftime('%H:%M:%S')}"
                                st.session_state.recent_confirmations = confirmation_msg
                                
                                # Show some details
                                with st.expander("📋 Confirmation Details"):
                                    for msg in success_messages[:5]:  # Show first 5
                                        st.text(f"• {msg}")
                                    if len(success_messages) > 5:
                                        st.text(f"... and {len(success_messages) - 5} more")
                            else:
                                st.warning("⚠️ No predictions were confirmed. Please check the logs for errors.")
                    
                    with col2:
                        review_key = f"review_low_conf_{len(labeled_anomalies)}"
                        if confidence_col is not None:
                            button_label = "🔍 Review Low-Confidence"
                            button_help = "Show predictions that need manual review"
                        else:
                            button_label = "🔍 Review All Predictions"
                            button_help = "Show all predictions for manual review"
                        
                        if st.button(button_label, key=review_key, help=button_help):
                            # Filter and display predictions for review
                            if confidence_col is not None:
                                low_conf_mask = labeled_anomalies[confidence_col] < confidence_threshold
                                review_anomalies = labeled_anomalies[low_conf_mask]
                                
                                if not review_anomalies.empty:
                                    st.warning(f"⚠️ {len(review_anomalies)} predictions need manual review:")
                                    
                                    # Store in session state for persistence
                                    st.session_state.low_confidence_predictions = review_anomalies
                                    
                                    # Display the predictions
                                    st.dataframe(review_anomalies[show_cols] if show_cols else review_anomalies, use_container_width=True)
                                    
                                    st.info("💡 **Tip**: Review these in the 'Explain & Feedback' page to improve future predictions")
                                    
                                    # Add quick action buttons for low confidence items
                                    st.markdown("**Quick Actions:**")
                                    if st.button("📝 Take me to Explain & Feedback", key="goto_explain"):
                                        # This would ideally switch to the explain page, but we can't do that directly
                                        st.info("👆 Please navigate to the 'Explain & Feedback' page using the sidebar to review these predictions.")
                                else:
                                    st.success("🎉 All predictions are high-confidence!")
                            else:
                                # No confidence scores - show all predictions for review
                                st.info(f"📊 All {len(labeled_anomalies)} predictions available for review:")
                                
                                # Store in session state for persistence
                                st.session_state.low_confidence_predictions = labeled_anomalies
                                
                                # Display the predictions
                                st.dataframe(labeled_anomalies[show_cols] if show_cols else labeled_anomalies, use_container_width=True)
                                
                                st.info("💡 **Tip**: Review these in the 'Explain & Feedback' page to improve future predictions")
                                
                                # Add quick action buttons
                                st.markdown("**Quick Actions:**")
                                if st.button("📝 Take me to Explain & Feedback", key="goto_explain_all"):
                                    st.info("👆 Please navigate to the 'Explain & Feedback' page using the sidebar to review these predictions.")
                    
                    with col3:
                        stats_key = f"show_stats_{len(labeled_anomalies)}"
                        if st.button("📊 Prediction Summary", key=stats_key, help="Show detailed prediction statistics"):
                            if confidence_col is not None:
                                # Show comprehensive statistics
                                st.subheader("📊 Prediction Statistics")
                                
                                total_predictions = len(labeled_anomalies)
                                high_conf_count = len(labeled_anomalies[
                                    labeled_anomalies[confidence_col] >= confidence_threshold
                                ])
                                low_conf_count = total_predictions - high_conf_count
                                avg_confidence = labeled_anomalies[confidence_col].mean()
                                
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Total Predictions", total_predictions)
                                with col_b:
                                    st.metric("High Confidence", high_conf_count, f"{high_conf_count/total_predictions:.1%}")
                                with col_c:
                                    st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                                
                                # Confidence distribution
                                st.markdown("**Confidence Distribution:**")
                                confidence_bins = pd.cut(labeled_anomalies[confidence_col], 
                                                       bins=[0, 0.5, 0.7, 0.8, 0.9, 1.0], 
                                                       labels=['Very Low (0-50%)', 'Low (50-70%)', 'Medium (70-80%)', 'High (80-90%)', 'Very High (90-100%)'])
                                confidence_counts = confidence_bins.value_counts().sort_index()
                                
                                for range_label, count in confidence_counts.items():
                                    if count > 0:
                                        percentage = count / total_predictions * 100
                                        st.text(f"• {range_label}: {count} predictions ({percentage:.1f}%)")
                            else:
                                # Show basic statistics without confidence
                                st.subheader("📊 Prediction Statistics")
                                
                                total_predictions = len(labeled_anomalies)
                                
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Total Predictions", total_predictions)
                                with col_b:
                                    # Count true positives if available
                                    true_pos_col = None
                                    for col in ['predicted_true_positive', 'is_true_positive', 'classification']:
                                        if col in labeled_anomalies.columns:
                                            true_pos_col = col
                                            break
                                    
                                    if true_pos_col:
                                        try:
                                            if labeled_anomalies[true_pos_col].dtype == bool:
                                                true_positives = labeled_anomalies[true_pos_col].sum()
                                            else:
                                                true_positives = len(labeled_anomalies[labeled_anomalies[true_pos_col] == True])
                                            st.metric("Predicted True Positives", true_positives)
                                        except:
                                            st.metric("Prediction Columns", len([col for col in labeled_anomalies.columns if col.startswith('predicted_')]))
                                    else:
                                        st.metric("Prediction Columns", len([col for col in labeled_anomalies.columns if col.startswith('predicted_')]))
                                with col_c:
                                    # Show unique categories if available
                                    cat_col = None
                                    for col in ['predicted_category', 'category', 'technique']:
                                        if col in labeled_anomalies.columns:
                                            cat_col = col
                                            break
                                    
                                    if cat_col:
                                        try:
                                            unique_categories = labeled_anomalies[cat_col].nunique()
                                            st.metric("Unique Categories", unique_categories)
                                        except:
                                            st.metric("Available Columns", len(labeled_anomalies.columns))
                                    else:
                                        st.metric("Available Columns", len(labeled_anomalies.columns))
                                
                                st.info("💡 **Note**: No confidence scores available. Consider training models with confidence prediction.")
                                
                                # Show column information
                                st.markdown("**Available Prediction Columns:**")
                                pred_cols = [col for col in labeled_anomalies.columns if col.startswith('predicted_')]
                                if pred_cols:
                                    for col in pred_cols:
                                        st.text(f"• {col}")
                                else:
                                    st.text("• No prediction columns found")
                    
                    # Show recent confirmation activity if any
                    if 'recent_confirmations' in st.session_state and st.session_state.recent_confirmations:
                        st.info(f"✅ **Recent Activity**: {st.session_state.recent_confirmations}")
                    
                    # Show persistent low confidence results if they exist
                    if 'low_confidence_predictions' in st.session_state and st.session_state.low_confidence_predictions is not None:
                        st.subheader("⚠️ Low Confidence Predictions Requiring Review")
                        low_conf_data = st.session_state.low_confidence_predictions
                        st.dataframe(low_conf_data[show_cols] if show_cols else low_conf_data, use_container_width=True)
                        st.info("💡 **Tip**: Review these in the 'Explain & Feedback' page to improve future predictions")
                    
                    # Download options
                    st.subheader("💾 Export Options")
                    
                    col_download1, col_download2 = st.columns(2)
                    
                    with col_download1:
                        # Download option
                        csv = labeled_anomalies.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download Labeled Anomalies",
                            data=csv,
                            file_name=f"labeled_anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with col_download2:
                        # Clear results button
                        if st.button("🗑️ Clear Results", help="Clear the current labeling results"):
                            if 'labeled_anomalies' in st.session_state:
                                del st.session_state.labeled_anomalies
                            if 'auto_labeling_complete' in st.session_state:
                                del st.session_state.auto_labeling_complete
                            if 'recent_confirmations' in st.session_state:
                                del st.session_state.recent_confirmations
                            if 'low_confidence_predictions' in st.session_state:
                                del st.session_state.low_confidence_predictions
                            st.rerun()
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
