"""
Notification service for the Network Anomaly Detection Platform.
Handles automatic notifications and alerts for analysis results.
"""

import streamlit as st
import logging
from typing import Dict, Any, List
from datetime import datetime
import re

logger = logging.getLogger("streamlit_app")

def sanitize_for_log(text: str) -> str:
    """Remove or replace emoji characters for logging to prevent encoding issues."""
    # More comprehensive emoji pattern
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # dingbats
                               u"\U000024C2-\U0001F251"  # enclosed characters
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    # Replace emojis with descriptive text or remove them
    cleaned = emoji_pattern.sub('', text)
    # Clean up multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

class NotificationService:
    """Service for handling notifications and alerts."""
    
    def __init__(self):
        """Initialize the notification service."""
        self.notifications = []
    
    def show_auto_analysis_results(self, analysis_results: Dict[str, Any]) -> None:
        """
        Display automatic analysis results in the UI.
        
        Args:
            analysis_results (Dict[str, Any]): Results from auto analysis
        """
        if not analysis_results.get('success', False):
            self._show_analysis_errors(analysis_results.get('errors', []))
            return
        
        # Show success notification
        st.success("🤖 **Automatic Analysis Complete!**")
        
        # Show analysis summary
        summary = analysis_results.get('analysis_summary', {})
        if summary:
            self._show_analysis_summary(summary)
        
        # Show MITRE mapping results
        mitre_results = analysis_results.get('mitre_mappings')
        if mitre_results:
            self._show_mitre_notification(mitre_results, summary.get('mitre_analysis', {}))
        
        # Show risk scoring results
        risk_results = analysis_results.get('risk_scores')
        if risk_results:
            self._show_risk_notification(risk_results, summary.get('risk_analysis', {}))
        
        # Show recommendations
        recommendations = summary.get('recommendations', [])
        if recommendations:
            self._show_recommendations(recommendations)
    
    def _show_analysis_errors(self, errors: List[str]) -> None:
        """Show analysis errors."""
        st.error("⚠️ **Automatic Analysis Encountered Errors:**")
        for error in errors:
            st.error(f"• {error}")
    
    def _show_analysis_summary(self, summary: Dict[str, Any]) -> None:
        """Show analysis summary metrics."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🔍 Anomalies Analyzed", summary.get('total_anomalies', 0))
        
        with col2:
            mitre_analysis = summary.get('mitre_analysis', {})
            mapped_count = mitre_analysis.get('mapped_anomalies', 0)
            st.metric("🛡️ MITRE Mapped", mapped_count)
        
        with col3:
            risk_analysis = summary.get('risk_analysis', {})
            high_risk = risk_analysis.get('high_risk_count', 0)
            st.metric("⚠️ High Risk", high_risk)
        
        with col4:
            recommendations = summary.get('recommendations', [])
            critical_recs = len([r for r in recommendations if r.get('priority') == 'Critical'])
            st.metric("🚨 Critical Actions", critical_recs)
    
    def _show_mitre_notification(self, mitre_results: Dict, mitre_analysis: Dict) -> None:
        """Show MITRE mapping notification."""
        if not mitre_results:
            return
        
        st.success("🛡️ **MITRE ATT&CK Mapping Results**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Mapping Summary:**")
            st.write(f"• **Anomalies Mapped:** {mitre_analysis.get('mapped_anomalies', 0)}")
            st.write(f"• **Unique Techniques:** {mitre_analysis.get('unique_techniques', 0)}")
            st.write(f"• **Unique Tactics:** {mitre_analysis.get('unique_tactics', 0)}")
        
        with col2:
            top_techniques = mitre_analysis.get('top_techniques', [])
            if top_techniques:
                st.write("**Top Techniques Detected:**")
                for technique in top_techniques[:3]:
                    st.write(f"• {technique}")
        
        st.info("💡 **Tip:** View detailed mappings in the MITRE Mapping page")
    
    def _show_risk_notification(self, risk_results: Dict, risk_analysis: Dict) -> None:
        """Show risk scoring notification."""
        if not risk_results:
            return
        
        st.warning("⚠️ **Risk Assessment Results**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Risk Summary:**")
            st.write(f"• **Average Risk Score:** {risk_analysis.get('avg_risk_score', 0):.2f}")
            st.write(f"• **Highest Risk Score:** {risk_analysis.get('max_risk_score', 0):.2f}")
            st.write(f"• **High-Risk Anomalies:** {risk_analysis.get('high_risk_count', 0)}")
        
        with col2:
            distribution = risk_analysis.get('risk_level_distribution', {})
            if distribution:
                    st.write("**Risk Level Distribution:**")
                    for level, count in distribution.items():
                        st.write(f"• **{level}:** {count}")
            
            st.info("💡 **Tip:** View detailed risk analysis in the Auto Labeling page")
    
    def _show_recommendations(self, recommendations: List[Dict]) -> None:
        """Show actionable recommendations."""
        if not recommendations:
            return
        
        st.info("💡 **Automated Recommendations**")
        # Group by priority
        critical = [r for r in recommendations if r.get('priority') == 'Critical']
        high = [r for r in recommendations if r.get('priority') == 'High']
        medium = [r for r in recommendations if r.get('priority') == 'Medium']
        low = [r for r in recommendations if r.get('priority') == 'Low']
        
        # Show critical recommendations first
        if critical:
            st.error("🚨 **Critical Actions Required:**")
            for rec in critical:
                st.write(f"• **{rec['category']}:** {rec['action']}")
                st.write(f"  *Reason:* {rec['reason']}")
        
        # Show high priority recommendations
        if high:
            st.warning("⚠️ **High Priority Actions:**")
            for rec in high:
                st.write(f"• **{rec['category']}:** {rec['action']}")
                st.write(f"  *Reason:* {rec['reason']}")
        
        # Show medium priority recommendations
        if medium:
            st.info("📋 **Medium Priority Actions:**")
            for rec in medium:
                st.write(f"• **{rec['category']}:** {rec['action']}")
        
        # Show low priority recommendations
        if low:
            st.info("📝 **Additional Recommendations:**")
            for rec in low:
                st.write(f"• **{rec['category']}:** {rec['action']}")
    
    def add_notification(self, message: str, notification_type: str = "info", 
                        timestamp: datetime = None) -> None:
        """
        Add a notification to the queue.
        
        Args:
            message (str): Notification message
            notification_type (str): Type of notification (info, success, warning, error)
            timestamp (datetime): When the notification was created
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        notification = {
            'message': message,
            'type': notification_type,
            'timestamp': timestamp,
            'id': len(self.notifications)
        }
        
        self.notifications.append(notification)
        
        # Store in session state for persistence
        if 'notifications' not in st.session_state:
            st.session_state.notifications = []
        st.session_state.notifications.append(notification)
    
    def show_notifications(self) -> None:
        """Show pending notifications."""
        notifications = st.session_state.get('notifications', [])
        
        if not notifications:
            return
        
        with st.sidebar:
            st.subheader("🔔 Notifications")
            
            for notification in notifications[-5:]:  # Show last 5 notifications
                timestamp = notification['timestamp'].strftime("%H:%M:%S")
                message = f"[{timestamp}] {notification['message']}"
                
                if notification['type'] == 'success':
                    st.success(message)
                elif notification['type'] == 'warning':
                    st.warning(message)
                elif notification['type'] == 'error':
                    st.error(message)
                else:
                    st.info(message)
            
            if len(notifications) > 5:
                st.write(f"... and {len(notifications) - 5} more notifications")
            
            if st.button("Clear Notifications"):
                st.session_state.notifications = []
                st.rerun()

# Singleton instance
notification_service = NotificationService()
