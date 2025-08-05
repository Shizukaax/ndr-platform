"""
Real-time Monitoring page for the Network Anomaly Detection Platform.
Provides live monitoring capabilities with Arkime JSON file integration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import logging
import json
import os

from core.model_manager import ModelManager
from core.risk_scorer import RiskScorer
from core.config_loader import load_config
from core.file_watcher import (
    start_realtime_monitoring, 
    stop_realtime_monitoring, 
    get_monitoring_status,
    get_live_metrics,
    get_recent_anomalies
)
from app.components.error_handler import handle_error

# Set up logger
logger = logging.getLogger("streamlit_app")

@handle_error
def show_real_time_monitoring():
    """Display the Real-time Monitoring page."""
    
    st.header("🔴 Real-time Network Monitoring")
    st.markdown("**Live monitoring dashboard with real-time anomaly detection and Arkime integration.**")
    
    # Highlight the purpose prominently
    st.info("🎯 **Primary Use Case**: This page monitors your configured data directory for real-time threat detection.")
    
    # Load configuration
    config = load_config()
    data_source_config = config.get('data_source', {})
    data_directory = data_source_config.get('directory', 'data')
    realtime_config = config.get('realtime_monitoring', {})
    arkime_config = realtime_config.get('arkime', {})
    
    # Quick status at the top
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.get('file_monitor') and hasattr(st.session_state.file_monitor, 'is_monitoring'):
            if st.session_state.file_monitor.is_monitoring:
                st.success("🟢 **LIVE MONITORING ACTIVE**")
            else:
                st.error("🔴 **MONITORING STOPPED**")
        else:
            st.warning("🟡 **MONITORING READY**")
    
    with col2:
        st.write(f"📁 **Source:** `{data_directory}`")
    
    with col3:
        file_pattern = arkime_config.get('file_pattern', '*.json')
        st.write(f"📄 **Pattern:** `{file_pattern}`")
    
    # Quick Start Section
    st.markdown("---")
    st.subheader("🚀 Quick Start")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("**Ready to monitor your Arkime packet capture data in real-time!**")
    with col2:
        if st.button("▶️ **START MONITORING**", type="primary"):
            st.session_state.quick_start_monitoring = True
    with col3:
        if st.button("⏹️ Stop Monitoring"):
            if 'file_monitor' in st.session_state:
                st.session_state.file_monitor.stop_monitoring()
                st.success("Monitoring stopped.")
    
    st.markdown("---")
    
    # Initialize components
    model_manager = ModelManager()
    risk_scorer = RiskScorer()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔴 Live Dashboard", "🚨 Alert Center", "📊 Performance", "⚙️ Settings"
    ])
    
    with tab1:
        st.subheader("🔴 Live Network Monitoring")
        
        # Control panel
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Default to active if quick start was clicked
            default_monitoring = st.session_state.get('quick_start_monitoring', False)
            monitoring_active = st.toggle("Enable Live Monitoring", value=default_monitoring)
            
            # Clear quick start flag after use
            if 'quick_start_monitoring' in st.session_state:
                del st.session_state['quick_start_monitoring']
        
        with col2:
            refresh_interval = st.selectbox(
                "Refresh Interval",
                options=[1, 5, 10, 30, 60],
                index=2,
                format_func=lambda x: f"{x} seconds"
            )
        
        with col3:
            # Get default threshold from config
            default_threshold = realtime_config.get('alerts', {}).get('default_threshold', 0.7)
            alert_threshold = st.slider("Alert Threshold", 0.0, 1.0, default_threshold, 0.1)
        
        # Status indicators
        st.subheader("📊 System Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if monitoring_active:
                st.success("🟢 Monitoring Active")
            else:
                st.warning("🟡 Monitoring Paused")
        
        with col2:
            # Mock model health
            model_health = "Healthy" if st.session_state.get('model_manager') else "Not Loaded"
            if model_health == "Healthy":
                st.success(f"🟢 Models: {model_health}")
            else:
                st.error(f"🔴 Models: {model_health}")
        
        with col3:
            # Data freshness
            data_age = "Real-time" if monitoring_active else "Static"
            st.info(f"📡 Data: {data_age}")
        
        with col4:
            # Alert status
            alert_count = st.session_state.get('alert_count', 0)
            if alert_count > 0:
                st.error(f"🚨 Alerts: {alert_count}")
            else:
                st.success("✅ No Alerts")
        
        # Live metrics
        st.subheader("📈 Live Network Metrics")
        
        # Create placeholder for live updates
        metrics_placeholder = st.empty()
        chart_placeholder = st.empty()
        
        if monitoring_active:
            # Start real-time Arkime file monitoring
            if 'file_monitor' not in st.session_state:
                from core.file_watcher import RealTimeMonitor
                
                # Check for session overrides, otherwise use config
                override_dir = st.session_state.get('arkime_directory_override')
                override_pattern = st.session_state.get('arkime_file_pattern_override')
                
                st.session_state.file_monitor = RealTimeMonitor(
                    watch_directory=override_dir,
                    file_pattern=override_pattern
                )
                
                arkime_dir = st.session_state.file_monitor.watch_directory
                file_pattern = st.session_state.file_monitor.file_pattern
                
                if st.session_state.file_monitor.start_monitoring():
                    st.success(f"🟢 Started monitoring Arkime JSON files at {arkime_dir} (pattern: {file_pattern})")
                else:
                    st.error(f"❌ Failed to start file monitoring. Check if {arkime_dir} exists and is accessible.")
            
            # Get real-time metrics from file monitor
            metrics = st.session_state.file_monitor.get_current_metrics()
            
            with metrics_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    # Real packets per second from Arkime data
                    pps = metrics.get('packets_per_second', 0)
                    pps_delta = metrics.get('pps_delta', 0)
                    st.metric("Packets/sec", f"{pps:,}", f"{pps_delta:+d}")
                
                with col2:
                    # Real bandwidth from Arkime data
                    bandwidth = metrics.get('bandwidth_mbps', 0)
                    bw_delta = metrics.get('bandwidth_delta', 0)
                    st.metric("Bandwidth (Mbps)", f"{bandwidth:.1f}", f"{bw_delta:+.1f}")
                
                with col3:
                    # Real active connections from Arkime
                    connections = metrics.get('active_connections', 0)
                    conn_delta = metrics.get('connections_delta', 0)
                    st.metric("Active Connections", f"{connections:,}", f"{conn_delta:+d}")
                
                with col4:
                    # Real anomaly rate from detection results
                    anomaly_rate = metrics.get('anomaly_rate_percent', 0)
                    anomaly_delta = metrics.get('anomaly_delta', 0)
                    st.metric("Anomaly Rate (%)", f"{anomaly_rate:.1f}", f"{anomaly_delta:+.1f}")
            
            # Real-time chart with actual Arkime data
            with chart_placeholder.container():
                # Get historical data from file monitor
                historical_data = st.session_state.file_monitor.get_historical_data()
                
                if historical_data and len(historical_data) > 0:
                    # Create subplots
                    from plotly.subplots import make_subplots
                    
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=('Packets/Second', 'Anomaly Scores', 'Bandwidth', 'Protocol Distribution'),
                        specs=[[{'secondary_y': False}, {'secondary_y': False}],
                               [{'secondary_y': False}, {'type': 'pie'}]]
                    )
                    
                    # Convert to DataFrame for easier handling
                    df_hist = pd.DataFrame(historical_data)
                    
                    # Packets per second
                    fig.add_trace(
                        go.Scatter(
                            x=df_hist['timestamp'],
                            y=df_hist['packets_per_second'],
                            mode='lines',
                            name='Packets/sec',
                            line=dict(color='blue')
                        ),
                        row=1, col=1
                    )
                    
                    # Anomaly scores
                    fig.add_trace(
                        go.Scatter(
                            x=df_hist['timestamp'],
                            y=df_hist['anomaly_score'],
                            mode='lines',
                            name='Anomaly Score',
                            line=dict(color='red')
                        ),
                        row=1, col=2
                    )
                    
                    # Add alert threshold line
                    fig.add_hline(
                        y=alert_threshold,
                        line_dash="dash",
                        line_color="orange",
                        row=1, col=2
                    )
                    
                    # Bandwidth
                    fig.add_trace(
                        go.Scatter(
                            x=df_hist['timestamp'],
                            y=df_hist['bandwidth_mbps'],
                            mode='lines',
                            name='Bandwidth',
                            line=dict(color='green')
                        ),
                        row=2, col=1
                    )
                    
                    # Protocol distribution from real data
                    protocol_data = st.session_state.file_monitor.get_protocol_distribution()
                    if protocol_data:
                        fig.add_trace(
                            go.Pie(
                                labels=list(protocol_data.keys()),
                                values=list(protocol_data.values()),
                                name="Protocols"
                            ),
                            row=2, col=2
                        )
                    
                    fig.update_layout(
                        height=600,
                        title_text="Live Network Monitoring Dashboard - Real Arkime Data",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("📊 Waiting for Arkime data... No JSON files processed yet.")
        
        else:
            # Stop monitoring if it was active
            if 'file_monitor' in st.session_state:
                st.session_state.file_monitor.stop_monitoring()
                st.info("🔴 File monitoring stopped.")
            
            with metrics_placeholder.container():
                st.info("🔴 Monitoring is paused. Toggle 'Enable Live Monitoring' to start.")
        
        # Recent activity
        st.subheader("🕐 Recent Activity")
        
        if monitoring_active:
            # Get real recent anomalies from file monitor
            recent_anomalies = st.session_state.file_monitor.get_recent_anomalies(limit=10)
            
            if recent_anomalies:
                # Convert to DataFrame for display
                recent_df = pd.DataFrame(recent_anomalies)
                
                # Format the dataframe
                if 'timestamp' in recent_df.columns:
                    recent_df['Timestamp'] = pd.to_datetime(recent_df['timestamp']).dt.strftime('%H:%M:%S')
                if 'anomaly_score' in recent_df.columns:
                    recent_df['Score'] = recent_df['anomaly_score'].round(3)
                if 'source_ip' in recent_df.columns:
                    recent_df['Source IP'] = recent_df['source_ip']
                if 'dest_ip' in recent_df.columns:
                    recent_df['Dest IP'] = recent_df['dest_ip']
                if 'protocol' in recent_df.columns:
                    recent_df['Protocol'] = recent_df['protocol']
                
                # Select relevant columns for display
                display_cols = ['Timestamp', 'Score', 'Source IP', 'Dest IP', 'Protocol']
                display_df = recent_df[display_cols] if all(col in recent_df.columns for col in display_cols) else recent_df
                
                st.dataframe(display_df, use_container_width=True)
            else:
                st.info("🔍 No anomalies detected yet. Monitor will show anomalies as they are detected in real-time.")
        else:
            st.info("Enable monitoring to see recent activity.")
    
    with tab2:
        st.subheader("🚨 Alert Management Center")
        
        # Alert configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Alert Thresholds:**")
            
            high_risk_threshold = st.slider("High Risk Threshold", 0.7, 1.0, 0.8, 0.05)
            medium_risk_threshold = st.slider("Medium Risk Threshold", 0.3, 0.7, 0.5, 0.05)
            
            alert_types = st.multiselect(
                "Alert Types",
                ["Anomaly Detection", "High Risk Score", "Unusual Protocol", "Port Scanning", "Large Packets"],
                default=["Anomaly Detection", "High Risk Score"]
            )
        
        with col2:
            st.write("**Notification Settings:**")
            
            email_alerts = st.checkbox("Email Notifications", value=False)
            if email_alerts:
                email_address = st.text_input("Email Address", placeholder="admin@company.com")
            
            slack_alerts = st.checkbox("Slack Notifications", value=False)
            if slack_alerts:
                slack_webhook = st.text_input("Slack Webhook URL", placeholder="https://hooks.slack.com/...")
            
            sms_alerts = st.checkbox("SMS Alerts", value=False)
            if sms_alerts:
                phone_number = st.text_input("Phone Number", placeholder="+1234567890")
        
        # Alert history
        st.subheader("📋 Alert History")
        
        # Mock alert history
        alert_history = []
        for i in range(10):
            alert_history.append({
                'Timestamp': datetime.now() - timedelta(hours=np.random.randint(1, 72)),
                'Type': np.random.choice(['Anomaly', 'High Risk', 'Protocol Alert', 'Port Scan']),
                'Severity': np.random.choice(['Low', 'Medium', 'High', 'Critical']),
                'Message': f"Detected {np.random.choice(['suspicious activity', 'anomalous behavior', 'high risk connection'])}",
                'Status': np.random.choice(['Active', 'Resolved', 'Investigating']),
                'Source IP': f"192.168.{np.random.randint(0, 255)}.{np.random.randint(1, 255)}"
            })
        
        alert_df = pd.DataFrame(alert_history)
        
        # Filter controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            severity_filter = st.multiselect(
                "Filter by Severity",
                options=alert_df['Severity'].unique(),
                default=alert_df['Severity'].unique()
            )
        
        with col2:
            status_filter = st.multiselect(
                "Filter by Status",
                options=alert_df['Status'].unique(),
                default=alert_df['Status'].unique()
            )
        
        with col3:
            time_filter = st.selectbox(
                "Time Range",
                options=["Last Hour", "Last 24 Hours", "Last 7 Days", "All Time"],
                index=1
            )
        
        # Apply filters
        filtered_alerts = alert_df[
            (alert_df['Severity'].isin(severity_filter)) &
            (alert_df['Status'].isin(status_filter))
        ]
        
        # Display alerts
        st.dataframe(filtered_alerts, use_container_width=True)
        
        # Alert actions
        st.subheader("⚡ Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔕 Silence All Alerts"):
                st.success("All alerts silenced for 1 hour")
        
        with col2:
            if st.button("✅ Mark All as Resolved"):
                st.success("All alerts marked as resolved")
        
        with col3:
            if st.button("📤 Export Alert Report"):
                csv = filtered_alerts.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download CSV",
                    data=csv,
                    file_name=f"alert_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    with tab3:
        st.subheader("📊 System Performance Monitoring")
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # CPU usage (simulated)
            cpu_usage = np.random.uniform(20, 80)
            st.metric("CPU Usage", f"{cpu_usage:.1f}%", f"{np.random.uniform(-5, 5):.1f}%")
        
        with col2:
            # Memory usage (simulated)
            memory_usage = np.random.uniform(30, 90)
            st.metric("Memory Usage", f"{memory_usage:.1f}%", f"{np.random.uniform(-2, 8):.1f}%")
        
        with col3:
            # Processing speed
            processing_speed = np.random.randint(500, 2000)
            st.metric("Records/sec", f"{processing_speed:,}", f"{np.random.randint(-100, 200)}")
        
        with col4:
            # Model accuracy
            accuracy = np.random.uniform(0.85, 0.98)
            st.metric("Model Accuracy", f"{accuracy:.1%}", f"{np.random.uniform(-0.02, 0.01):.1%}")
        
        # Performance charts
        st.subheader("📈 Performance Trends")
        
        # Generate mock performance data
        time_points = pd.date_range(
            start=datetime.now() - timedelta(hours=24),
            end=datetime.now(),
            freq='1H'
        )
        
        perf_data = pd.DataFrame({
            'timestamp': time_points,
            'cpu_usage': np.random.uniform(20, 80, len(time_points)),
            'memory_usage': np.random.uniform(30, 90, len(time_points)),
            'throughput': np.random.randint(500, 2000, len(time_points)),
            'latency': np.random.uniform(10, 100, len(time_points))
        })
        
        # Create performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(
                perf_data,
                x='timestamp',
                y=['cpu_usage', 'memory_usage'],
                title="Resource Usage Over Time",
                labels={'value': 'Usage (%)', 'variable': 'Resource'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(
                perf_data,
                x='timestamp',
                y='throughput',
                title="Processing Throughput",
                labels={'throughput': 'Records/sec'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Model performance
        st.subheader("🤖 Model Performance")
        
        if st.session_state.get('model_manager'):
            # Mock model performance data
            models = ['IsolationForest', 'LocalOutlierFactor', 'OneClassSVM']
            
            model_metrics = []
            for model in models:
                model_metrics.append({
                    'Model': model,
                    'Accuracy': np.random.uniform(0.80, 0.95),
                    'Precision': np.random.uniform(0.75, 0.90),
                    'Recall': np.random.uniform(0.70, 0.88),
                    'F1-Score': np.random.uniform(0.72, 0.89),
                    'Processing Time (ms)': np.random.randint(50, 500)
                })
            
            model_df = pd.DataFrame(model_metrics)
            model_df = model_df.round(3)
            st.dataframe(model_df, use_container_width=True)
        else:
            st.info("No models loaded. Load models to see performance metrics.")
    
    with tab4:
        st.subheader("⚙️ Monitoring Configuration")
        
        # Configuration info
        st.info("📝 **Configuration Source**: Real-time monitoring settings are loaded from `config/config.yaml`. Changes made here are temporary session overrides.")
        
        # Data source configuration
        st.write("**Data Source Settings:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            data_source = st.selectbox(
                "Data Source Type",
                options=["Arkime JSON Files", "File Upload", "Network Stream", "Database", "API Endpoint"],
                index=0
            )
            
            if data_source == "Arkime JSON Files":
                # Load current config values from data_source section
                current_data_dir = data_directory
                current_file_pattern = arkime_config.get('file_pattern', '*.json')
                
                arkime_path = st.text_input(
                    "Data Directory", 
                    value=current_data_dir,
                    help="Path to directory where data files are stored (configured in config.yaml data_source.directory)"
                )
                file_pattern = st.text_input(
                    "File Pattern", 
                    value=current_file_pattern,
                    help="Pattern to match JSON files (e.g., *.json, *.pcap_filtered.json)"
                )
                
                # Configuration info
                st.info(f"💡 Current config: {current_data_dir} with pattern {current_file_pattern}")
                
                if st.button("🔄 Update Configuration"):
                    # This would require updating config.yaml or session overrides
                    st.warning("⚠️ To permanently change these settings, update config/config.yaml data_source.directory and restart the application.")
                    # For now, store as session overrides
                    st.session_state.data_directory_override = arkime_path
                    st.session_state.arkime_file_pattern_override = file_pattern
                    st.success("✅ Configuration updated for this session. Restart monitoring to apply changes.")
                    
            elif data_source == "Network Stream":
                stream_ip = st.text_input("Stream IP Address", "127.0.0.1")
                stream_port = st.number_input("Stream Port", value=9999)
            elif data_source == "Database":
                db_host = st.text_input("Database Host", "localhost")
                db_name = st.text_input("Database Name", "network_data")
            elif data_source == "API Endpoint":
                api_url = st.text_input("API URL", "http://localhost:8080/api/data")
                api_key = st.text_input("API Key", type="password")
        
        with col2:
            # Real-time monitoring settings from config
            st.write("**Real-time Settings:**")
            
            # Load current values from config
            buffer_size = st.number_input(
                "Buffer Size (records)", 
                value=arkime_config.get('max_buffer_size', 1000), 
                min_value=100
            )
            
            retention_hours = st.number_input(
                "Data Retention (hours)", 
                value=arkime_config.get('retention_hours', 24), 
                min_value=1
            )
            
            polling_interval = st.number_input(
                "Polling Interval (seconds)", 
                value=arkime_config.get('polling_interval', 1), 
                min_value=1
            )
            
            # Alert thresholds from config
            alerts_config = realtime_config.get('alerts', {})
            default_alert_threshold = st.number_input(
                "Default Alert Threshold", 
                value=alerts_config.get('default_threshold', 0.7), 
                min_value=0.0, 
                max_value=1.0, 
                step=0.1
            )
            
            high_risk_threshold = st.number_input(
                "High Risk Threshold", 
                value=alerts_config.get('high_risk_threshold', 0.8), 
                min_value=0.0, 
                max_value=1.0, 
                step=0.05
            )
        
        # Model configuration
        st.write("**Model Configuration:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            auto_retrain = st.checkbox("Auto-retrain models", value=True)
            if auto_retrain:
                retrain_interval = st.selectbox(
                    "Retrain Interval",
                    options=["Daily", "Weekly", "Monthly"],
                    index=1
                )
                min_samples = st.number_input("Min samples for retrain", value=100)
        
        with col2:
            model_ensemble = st.checkbox("Use ensemble models", value=True)
            if model_ensemble:
                ensemble_models = st.multiselect(
                    "Ensemble Models",
                    options=["IsolationForest", "LocalOutlierFactor", "OneClassSVM"],
                    default=["IsolationForest", "LocalOutlierFactor"]
                )
        
        # Export/Import configuration
        st.write("**Configuration Management:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Configuration"):
                config = {
                    'data_source': data_source,
                    'arkime_directory': arkime_path if data_source == "Arkime JSON Files" else None,
                    'file_pattern': file_pattern if data_source == "Arkime JSON Files" else None,
                    'buffer_size': buffer_size,
                    'retention_hours': retention_hours,
                    'polling_interval': polling_interval,
                    'default_alert_threshold': default_alert_threshold,
                    'high_risk_threshold': high_risk_threshold,
                    'auto_retrain': auto_retrain,
                    'model_ensemble': model_ensemble,
                    'timestamp': datetime.now().isoformat()
                }
                
                config_json = json.dumps(config, indent=2)
                st.download_button(
                    "Download Config",
                    data=config_json,
                    file_name=f"monitoring_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            uploaded_config = st.file_uploader("📂 Import Configuration", type="json")
            if uploaded_config:
                try:
                    config = json.load(uploaded_config)
                    st.success("✅ Configuration imported successfully!")
                    st.json(config)
                except Exception as e:
                    st.error(f"❌ Failed to import configuration: {e}")
        
        with col3:
            if st.button("🔄 Reset to Defaults"):
                st.success("Configuration reset to defaults")
