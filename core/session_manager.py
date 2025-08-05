"""
Session state manager for the Network Anomaly Detection Platform.
Centralizes session state management and provides clean APIs.
"""

import streamlit as st
import pandas as pd
from typing import Any, Optional, Dict
import logging

logger = logging.getLogger("streamlit_app")

class SessionStateManager:
    """Manages application session state with clean APIs."""
    
    def __init__(self):
        """Initialize session state manager."""
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize all session state keys with default values."""
        defaults = {
            # Data management
            'combined_data': None,
            'data_source_info': {},
            'feature_engineering_done': False,
            
            # Model and analysis results
            'anomaly_model': None,
            'anomaly_scores': None,
            'anomaly_threshold': None,
            'anomaly_features': None,
            'anomalies': pd.DataFrame(),
            'anomaly_indices': [],
            'selected_model': None,
            'model_results': {},
            
            # Auto analysis results
            'mitre_mappings': None,
            'mitre_auto_mapped': False,
            'risk_scores': None,
            'risk_auto_calculated': False,
            'auto_analysis_complete': False,
            
            # UI state
            'notifications': [],
            'current_page': 'anomaly_detection',
            'analysis_in_progress': False,
            
            # Comparison results
            'model_comparison_results': {},
            'comparison_features': [],
            'comparison_data': None,
            
            # Feedback and explanations
            'feedback_data': [],
            'explanation_cache': {},
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    # Data management methods
    def get_data(self) -> Optional[pd.DataFrame]:
        """Get the combined data."""
        return st.session_state.get('combined_data')
    
    def set_data(self, data: pd.DataFrame, source_info: Dict = None):
        """Set the combined data and source info."""
        st.session_state.combined_data = data
        if source_info:
            st.session_state.data_source_info = source_info
        logger.info(f"Data updated: {len(data)} records")
    
    def has_data(self) -> bool:
        """Check if data is loaded."""
        return st.session_state.get('combined_data') is not None
    
    def clear_data(self):
        """Clear all data and reset related state."""
        st.session_state.combined_data = None
        st.session_state.data_source_info = {}
        st.session_state.feature_engineering_done = False
        self.clear_analysis_results()
    
    # Model and analysis methods
    def get_anomalies(self) -> pd.DataFrame:
        """Get detected anomalies."""
        return st.session_state.get('anomalies', pd.DataFrame())
    
    def set_anomalies(self, anomalies: pd.DataFrame, model=None, scores=None, 
                     threshold=None, features=None, indices=None):
        """Set anomaly detection results."""
        st.session_state.anomalies = anomalies
        if model is not None:
            st.session_state.anomaly_model = model
        if scores is not None:
            st.session_state.anomaly_scores = scores
        if threshold is not None:
            st.session_state.anomaly_threshold = threshold
        if features is not None:
            st.session_state.anomaly_features = features
        if indices is not None:
            st.session_state.anomaly_indices = indices
        
        logger.info(f"Anomalies updated: {len(anomalies)} anomalies detected")
    
    def has_anomalies(self) -> bool:
        """Check if anomalies are detected."""
        anomalies = st.session_state.get('anomalies', pd.DataFrame())
        return not anomalies.empty
    
    def get_model_results(self) -> Dict:
        """Get model results."""
        return st.session_state.get('model_results', {})
    
    def set_model_results(self, results: Dict):
        """Set model results."""
        st.session_state.model_results = results
    
    def clear_analysis_results(self):
        """Clear all analysis results."""
        st.session_state.anomaly_model = None
        st.session_state.anomaly_scores = None
        st.session_state.anomaly_threshold = None
        st.session_state.anomaly_features = None
        st.session_state.anomalies = pd.DataFrame()
        st.session_state.anomaly_indices = []
        st.session_state.selected_model = None
        st.session_state.model_results = {}
        self.clear_auto_analysis_results()
    
    # Auto analysis methods
    def get_mitre_mappings(self) -> Optional[Dict]:
        """Get MITRE mappings."""
        return st.session_state.get('mitre_mappings')
    
    def set_mitre_mappings(self, mappings: Dict):
        """Set MITRE mappings."""
        st.session_state.mitre_mappings = mappings
        st.session_state.mitre_auto_mapped = True
    
    def get_risk_scores(self) -> Optional[Dict]:
        """Get risk scores."""
        return st.session_state.get('risk_scores')
    
    def set_risk_scores(self, scores: Dict):
        """Set risk scores."""
        st.session_state.risk_scores = scores
        st.session_state.risk_auto_calculated = True
    
    def is_auto_analysis_complete(self) -> bool:
        """Check if auto analysis is complete."""
        return st.session_state.get('auto_analysis_complete', False)
    
    def set_auto_analysis_complete(self, complete: bool = True):
        """Mark auto analysis as complete."""
        st.session_state.auto_analysis_complete = complete
    
    def clear_auto_analysis_results(self):
        """Clear auto analysis results."""
        st.session_state.mitre_mappings = None
        st.session_state.mitre_auto_mapped = False
        st.session_state.risk_scores = None
        st.session_state.risk_auto_calculated = False
        st.session_state.auto_analysis_complete = False
    
    # UI state methods
    def is_analysis_in_progress(self) -> bool:
        """Check if analysis is in progress."""
        return st.session_state.get('analysis_in_progress', False)
    
    def set_analysis_in_progress(self, in_progress: bool):
        """Set analysis progress state."""
        st.session_state.analysis_in_progress = in_progress
    
    def get_current_page(self) -> str:
        """Get current page."""
        return st.session_state.get('current_page', 'anomaly_detection')
    
    def set_current_page(self, page: str):
        """Set current page."""
        st.session_state.current_page = page
    
    # Notification methods
    def add_notification(self, message: str, notification_type: str = "info"):
        """Add a notification."""
        if 'notifications' not in st.session_state:
            st.session_state.notifications = []
        
        notification = {
            'message': message,
            'type': notification_type,
            'timestamp': pd.Timestamp.now(),
            'id': len(st.session_state.notifications)
        }
        
        st.session_state.notifications.append(notification)
    
    def get_notifications(self) -> list:
        """Get all notifications."""
        return st.session_state.get('notifications', [])
    
    def clear_notifications(self):
        """Clear all notifications."""
        st.session_state.notifications = []
    
    # Utility methods
    def get_data_summary(self) -> Dict[str, Any]:
        """Get a summary of current data state."""
        data = self.get_data()
        anomalies = self.get_anomalies()
        
        summary = {
            'has_data': self.has_data(),
            'data_records': len(data) if data is not None else 0,
            'has_anomalies': self.has_anomalies(),
            'anomaly_count': len(anomalies) if not anomalies.empty else 0,
            'has_mitre_mappings': self.get_mitre_mappings() is not None,
            'has_risk_scores': self.get_risk_scores() is not None,
            'auto_analysis_complete': self.is_auto_analysis_complete(),
            'analysis_in_progress': self.is_analysis_in_progress()
        }
        
        return summary
    
    def reset_session(self):
        """Reset the entire session state."""
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Reinitialize with defaults
        self._initialize_session_state()
        logger.info("Session state reset")

# Singleton instance
session_manager = SessionStateManager()
