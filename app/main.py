"""
Main Streamlit application for Network Anomaly Detection Platform.
Manages navigation, page routing, and overall application structure.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import logging
from pathlib import Path

from app.about import show_about
from app.settings import show_settings
from app.pages.anomaly_detection import show_anomaly_detection
from app.pages.model_comparison import show_model_comparison
from app.pages.explain_feedback import show_explain_feedback
from app.pages.auto_labeling import show_auto_labeling
from app.pages.analytics_dashboard import show_analytics_dashboard
from app.pages.real_time_monitoring import show_real_time_monitoring
from app.pages.mitre_mapping import show_mitre_mapping
from app.pages.model_management import show_model_management
from app.pages.file_diagnostics import show_file_diagnostics
from app.pages.reporting import show_reporting
from app.state.session_state import init_session_state
from app.components.data_source_selector import show_compact_data_status
from core.logging_config import get_logger
from core.data_manager import get_data_manager
from core.notification_service import notification_service
from core.session_manager import session_manager

# Get logger
logger = get_logger("streamlit_app")

def main():
    """Main function to run the Streamlit app."""
    
    # Set page config
    st.set_page_config(
        page_title="NDR Platform - Network Detection & Response",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Initialize data manager and auto-load data
    data_manager = get_data_manager()
    data_manager.initialize_default_data()
    
    # Show notifications in sidebar
    notification_service.show_notifications()
    
    # Set up sidebar
    with st.sidebar:
        st.title("🛡️ NDR Platform")
        
        # Try to load logo if it exists, otherwise show an emoji
        logo_path = Path("app/assets/network_logo.png")
        if logo_path.exists():
            try:
                st.image(str(logo_path), use_container_width=True)
            except Exception as e:
                logger.warning(f"Failed to load logo image: {str(e)}")
                st.markdown("## 🛡️ 🌐")
        else:
            logo_path.parent.mkdir(parents=True, exist_ok=True)
            st.markdown("## 🛡️ 🌐")
            logger.info(f"Logo image not found at {logo_path}, using emoji fallback")
        
        # Navigation
        st.header("Navigation")
        
        # Workflow guidance
        st.markdown("**🎯 NDR Workflow:**")
        st.markdown("""
        1. **🔍 Anomaly Detection** - Train ML models & detect threats
        2. **📊 Analytics Dashboard** - Analyze risk patterns & trends  
        3. **� Explain & Feedback** - Investigate & provide feedback
        4. **🏷️ AI Auto Labeling** - Train classification models
        5. **� Reporting** - Generate & export analysis reports
        """)
        
        st.divider()
        
        page = st.radio(
            "Select Page",
            options=[
                "🔍 Anomaly Detection", 
                "📊 Analytics Dashboard", 
                "� Explain & Feedback", 
                "🏷️ AI Auto Labeling",
                "� Real-time Monitoring",
                "🛡️ MITRE Mapping",
                "🤖 Model Management", 
                "⚖️ Model Comparison",
                "📄 Reporting",
                "🗂️ File Diagnostics", 
                "⚙️ Settings", 
                "ℹ️ About"
            ],
            index=0  # Default to Anomaly Detection
        )
        
        st.divider()
        
        # Workflow Status
        st.write("#### 📊 Data Status")
        
        # Use the compact data status component
        show_compact_data_status()
    
    # Main content area
    try:
        # Route to appropriate page based on selection
        if page == "🔍 Anomaly Detection":
            show_anomaly_detection()
        elif page == "📊 Analytics Dashboard":
            show_analytics_dashboard()
        elif page == "� Explain & Feedback":
            show_explain_feedback()
        elif page == "🏷️ AI Auto Labeling":
            show_auto_labeling()
        elif page == "� Real-time Monitoring":
            show_real_time_monitoring()
        elif page == "🛡️ MITRE Mapping":
            show_mitre_mapping()
        elif page == "🤖 Model Management":
            show_model_management()
        elif page == "⚖️ Model Comparison":
            show_model_comparison()
        elif page == "📄 Reporting":
            show_reporting()
        elif page == "🗂️ File Diagnostics":
            show_file_diagnostics()
        elif page == "⚙️ Settings":
            show_settings()
        elif page == "ℹ️ About":
            show_about()
        else:
            st.error(f"Unknown page: {page}")
            
    except Exception as e:
        st.error(f"Error loading page '{page}': {str(e)}")
        logger.error(f"Error in page '{page}': {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
