"""
Real-time Network Monitoring Dashboard
File-based refresh - updates when JSON files change
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
import time
import json
import numpy as np
import os
import glob
import logging

# Import modular components
from app.components.security_metrics import SecurityMetrics
from core.predictive_security import PredictiveSecurityEngine
from core.anomaly_tracker import AnomalyTracker

def _get_data_files():
    """Get data files from directory - will refresh with auto-refresh timer"""
    # Load config directly each time to detect changes
    try:
        from core.config_loader import load_config
        config = load_config()
        data_dir = config.get('data_source', {}).get('directory', './data/')
        file_pattern = config.get('data_source', {}).get('file_pattern', '*.json')
    except:
        data_dir = './data/'
        file_pattern = '*.json'
    
    # Get current file list
    data_files = glob.glob(os.path.join(data_dir, file_pattern))
    
    # Store file count for display
    if 'previous_file_count' not in st.session_state:
        st.session_state.previous_file_count = len(data_files)
    elif st.session_state.previous_file_count != len(data_files):
        # File count changed - show notification
        if len(data_files) > st.session_state.previous_file_count:
            st.success(f"ðŸ”„ New files detected! ({st.session_state.previous_file_count} â†’ {len(data_files)})")
        elif len(data_files) < st.session_state.previous_file_count:
            st.info(f"ðŸ“ Files removed ({st.session_state.previous_file_count} â†’ {len(data_files)})")
        st.session_state.previous_file_count = len(data_files)
    
    return data_files

def show_real_time_monitoring():
    """Main real-time monitoring page"""
    st.title("ðŸ”´ Real-time Network Monitoring")
    st.markdown("---")
    
    # Initialize components
    if 'security_metrics' not in st.session_state:
        st.session_state.security_metrics = SecurityMetrics()
    if 'predictive_engine' not in st.session_state:
        st.session_state.predictive_engine = PredictiveSecurityEngine()
    if 'anomaly_tracker' not in st.session_state:
        st.session_state.anomaly_tracker = AnomalyTracker()
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "ðŸš€ Quick Start", 
        "ðŸ“¡ Live Dashboard", 
        "ðŸ“Š Security Metrics",
        "ðŸ”® Predictive Security",
        "ðŸ“ˆ Anomaly History"
    ])
    
    with tab1:
        _render_quick_start_section()
    
    with tab2:
        _render_live_dashboard_tab()
    
    with tab3:
        _render_real_data_security_metrics()
    
    with tab4:
        _render_predictive_security_tab()
    
    with tab5:
        _render_anomaly_history_dashboard()
    
    # Note: Dashboard auto-refresh interval is configurable in config/config.yaml under monitoring.real_time.auto_refresh_interval

def _render_quick_start_section():
    """Render the quick start section"""
    st.markdown("### ðŸš€ Quick Start")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Real-time Network Monitoring** analyzes your network traffic in real-time using advanced ML models.
        
        ðŸ”¹ **Anomaly Detection** - Detects unusual network patterns  
        ðŸ”¹ **Protocol Analysis** - Deep packet inspection and analysis  
        ðŸ”¹ **Threat Intelligence** - Real-time threat correlation  
        ðŸ”¹ **Predictive Security** - Forecast potential threats  
        """)
    
    with col2:
        # Get data files to detect changes
        data_files = _get_data_files()
        data_dir = st.session_state.get('config_cache', {}).get('data_dir', './data/')
        
        # Check monitoring status
        is_monitoring = st.session_state.get('file_monitor') and hasattr(st.session_state.file_monitor, 'is_monitoring') and st.session_state.file_monitor.is_monitoring
        
        if is_monitoring:
            st.success("ðŸŸ¢ **MONITORING ACTIVE**")
            st.write(f"Monitoring: `{data_dir}`")
            st.write(f"Files found: {len(data_files)} JSON files")
        else:
            if len(data_files) > 0:
                st.info("ðŸ”µ **READY TO MONITOR**")
                st.write(f"Directory: `{data_dir}`")
                st.write(f"Found: {len(data_files)} JSON files")
            else:
                st.warning("âš ï¸ **NO DATA FILES**")
                st.write(f"Directory: `{data_dir}`")
                st.write("No JSON files found to monitor")

def _render_live_dashboard_tab():
    """Render the live dashboard tab with file-based refresh"""
    st.markdown("### ðŸ“¡ Live Dashboard")
    
    # Load config to get auto-refresh settings
    try:
        from core.config_loader import load_config
        config = load_config()
        auto_refresh_interval = config.get('monitoring', {}).get('real_time', {}).get('auto_refresh_interval', 300)  # Default 5 minutes
        enable_auto_refresh = config.get('monitoring', {}).get('real_time', {}).get('enable_auto_refresh', True)
    except:
        auto_refresh_interval = 300  # Default 5 minutes in seconds
        enable_auto_refresh = True
    
    # Auto-refresh based on config settings
    if enable_auto_refresh:
        st_autorefresh(interval=auto_refresh_interval * 1000, key="file_watcher")  # Convert to milliseconds
    
    # Show current data status
    data_files = _get_data_files()
    refresh_minutes = auto_refresh_interval // 60
    refresh_seconds = auto_refresh_interval % 60
    if refresh_minutes > 0:
        refresh_display = f"{refresh_minutes} minute{'s' if refresh_minutes != 1 else ''}"
        if refresh_seconds > 0:
            refresh_display += f" {refresh_seconds} second{'s' if refresh_seconds != 1 else ''}"
    else:
        refresh_display = f"{refresh_seconds} second{'s' if refresh_seconds != 1 else ''}"
    
    if enable_auto_refresh:
        st.info(f"ðŸ”„ **Auto-refresh every {refresh_display}** | Current files: {len(data_files)} JSON files")
    else:
        st.info(f"â¸ï¸ **Auto-refresh disabled** | Current files: {len(data_files)} JSON files")
    
    # Simplified monitoring controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Check if monitoring is active
        is_monitoring = st.session_state.get('file_monitor') and hasattr(st.session_state.file_monitor, 'is_monitoring') and st.session_state.file_monitor.is_monitoring
        
        if is_monitoring:
            if st.button("ðŸ›‘ Stop Monitoring", type="primary"):
                if hasattr(st.session_state.file_monitor, 'stop_monitoring'):
                    st.session_state.file_monitor.stop_monitoring()
                    st.success("Monitoring stopped")
                    st.rerun()
        else:
            if st.button("â–¶ï¸ Start Monitoring", type="primary"):
                try:
                    from core.file_watcher import RealTimeMonitor
                    if 'file_monitor' not in st.session_state:
                        st.session_state.file_monitor = RealTimeMonitor()
                    
                    st.session_state.file_monitor.start_monitoring()
                    st.success("Monitoring started!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to start monitoring: {e}")
    
    with col2:
        if st.button("ðŸ”„ Manual Refresh"):
            st.rerun()
    
    # Configuration info
    st.markdown("**âš™ï¸ Configuration:**")
    if enable_auto_refresh:
        st.caption(f"â€¢ Auto-refresh: âœ… Enabled ({refresh_display})")
    else:
        st.caption("â€¢ Auto-refresh: âŒ Disabled")
    st.caption(f"â€¢ Configurable in: `config/config.yaml` â†’ `monitoring.real_time.auto_refresh_interval`")
    
    st.markdown("---")
    
    # Real-time metrics from actual data
    _render_real_time_metrics_from_data()
    
    st.markdown("---")
    
    # Real-time charts
    col1, col2 = st.columns(2)
    
    with col1:
        _render_network_activity_chart()
    
    with col2:
        _render_threat_level_chart()
    
    st.markdown("---")
    
    # NEW: Real-time security dashboard
    _render_real_time_security_dashboard()
    
    st.markdown("---")
    
    # Additional real data charts
    col1, col2 = st.columns(2)
    
    with col1:
        _render_protocol_distribution_chart()
    
    with col2:
        _render_ip_activity_chart()
    
    st.markdown("---")
    
    # NEW: Real-time packet stream and threat detection
    _render_live_packet_stream()
    
    st.markdown("---")
    
    # NEW: Network topology visualization
    _render_network_topology()
    
    st.markdown("---")
    
    # Live alerts section
    _render_live_alerts()

def _render_real_time_security_dashboard():
    """Render real-time security dashboard with live threat detection"""
    st.markdown("### ðŸ›¡ï¸ Real-time Security Dashboard")
    
    # Load configuration for anomaly detection and auto-refresh
    try:
        from core.config_loader import load_config
        config = load_config()
        rt_config = config.get('monitoring', {}).get('real_time', {})
        anomaly_config = rt_config.get('anomaly_detection', {})
        anomaly_enabled = anomaly_config.get('enabled', True)
        model_type = anomaly_config.get('model_type', 'ensemble')
        confidence_threshold = anomaly_config.get('confidence_threshold', 0.7)
        # Load auto-refresh settings for status display
        enable_auto_refresh = rt_config.get('enable_auto_refresh', True)
        auto_refresh_interval = rt_config.get('auto_refresh_interval', 300)
    except:
        anomaly_enabled = True
        model_type = 'ensemble'
        confidence_threshold = 0.7
        enable_auto_refresh = True
        auto_refresh_interval = 300
    
    # Show current anomaly detection configuration
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("#### âš™ï¸ Anomaly Detection Configuration")
    with col2:
        if st.button("ðŸ”§ Reconfigure"):
            st.info("Edit `config/config.yaml` â†’ `monitoring.real_time.anomaly_detection` to change settings")
    
    config_info = f"**Model**: {model_type.replace('_', ' ').title()} | **Enabled**: {'âœ…' if anomaly_enabled else 'âŒ'} | **Threshold**: {confidence_threshold}"
    st.caption(config_info)
    
    try:
        # Get data files - will trigger refresh when files change
        data_files = _get_data_files()
        
        if data_files:
            # Analyze the most recent file for real-time threats
            latest_file = max(data_files, key=os.path.getmtime)
            
            try:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list) and len(data) > 0:
                    # Security metrics
                    external_connections = 0
                    suspicious_protocols = 0
                    large_transfers = 0
                    unique_external_ips = set()
                    port_scans = {}
                    
                    # Define private IP ranges
                    private_ranges = ['192.168.', '10.', '172.16.', '172.17.', '172.18.', '172.19.', '172.20.', '172.21.', '172.22.', '172.23.', '172.24.', '172.25.', '172.26.', '172.27.', '172.28.', '172.29.', '172.30.', '172.31.', '127.']
                    
                    # Suspicious protocols list
                    risky_protocols = ['SMB', 'SMB2', 'RDP', 'FTP', 'TELNET', 'SSH', 'SNMP']
                    
                    # Analyze all packets (no limit for comprehensive analysis)
                    recent_packets = data
                    
                    for packet in recent_packets:
                        if isinstance(packet, dict) and '_source' in packet:
                            layers = packet['_source'].get('layers', {})
                            
                            src_ip = layers.get('ip.src', [''])[0] if isinstance(layers.get('ip.src'), list) else layers.get('ip.src', '')
                            dst_ip = layers.get('ip.dst', [''])[0] if isinstance(layers.get('ip.dst'), list) else layers.get('ip.dst', '')
                            protocol = layers.get('_ws.col.Protocol', [''])[0] if isinstance(layers.get('_ws.col.Protocol'), list) else layers.get('_ws.col.Protocol', '')
                            frame_len = layers.get('frame.len', ['0'])[0] if isinstance(layers.get('frame.len'), list) else layers.get('frame.len', '0')
                            
                            # Check for external connections
                            if src_ip and dst_ip:
                                src_private = any(src_ip.startswith(pr) for pr in private_ranges)
                                dst_private = any(dst_ip.startswith(pr) for pr in private_ranges)
                                
                                if not src_private or not dst_private:
                                    external_connections += 1
                                    if not src_private:
                                        unique_external_ips.add(src_ip)
                                    if not dst_private:
                                        unique_external_ips.add(dst_ip)
                            
                            # Check for suspicious protocols
                            if any(risky_proto.lower() in protocol.lower() for risky_proto in risky_protocols):
                                suspicious_protocols += 1
                            
                            # Check for large transfers
                            try:
                                if int(frame_len) > 1500:
                                    large_transfers += 1
                            except:
                                pass
                            
                            # Port scan detection
                            if dst_ip and 'tcp.dstport' in layers:
                                port = layers['tcp.dstport'][0] if isinstance(layers['tcp.dstport'], list) else layers['tcp.dstport']
                                key = f"{src_ip}->{dst_ip}"
                                if key not in port_scans:
                                    port_scans[key] = set()
                                port_scans[key].add(port)
                    
                    # Security metrics display
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if external_connections > 100:
                            st.metric("ðŸŒ External Connections", external_connections, delta="High Risk", delta_color="inverse")
                        else:
                            st.metric("ðŸŒ External Connections", external_connections)
                    
                    with col2:
                        if suspicious_protocols > 50:
                            st.metric("âš ï¸ Risky Protocols", suspicious_protocols, delta="Alert", delta_color="inverse")
                        else:
                            st.metric("âš ï¸ Risky Protocols", suspicious_protocols)
                    
                    with col3:
                        if large_transfers > 200:
                            st.metric("ðŸ“¦ Large Transfers", large_transfers, delta="Monitor", delta_color="inverse")
                        else:
                            st.metric("ðŸ“¦ Large Transfers", large_transfers)
                    
                    with col4:
                        st.metric("ðŸŒ External IPs", len(unique_external_ips))
                    
                    # Threat level indicator
                    st.markdown("#### ðŸš¨ Current Threat Level")
                    
                    # Calculate overall threat score
                    threat_score = 0
                    if external_connections > 100:
                        threat_score += 30
                    if suspicious_protocols > 50:
                        threat_score += 40
                    if large_transfers > 200:
                        threat_score += 20
                    if len(unique_external_ips) > 20:
                        threat_score += 10
                    
                    # Port scan detection
                    potential_scans = [conn for conn, ports in port_scans.items() if len(ports) > 5]
                    if potential_scans:
                        threat_score += 30
                    
                    # Display threat level
                    if threat_score >= 70:
                        st.error(f"ðŸš¨ HIGH THREAT LEVEL: {threat_score}%")
                        st.error("Immediate attention required!")
                    elif threat_score >= 40:
                        st.warning(f"âš ï¸ MEDIUM THREAT LEVEL: {threat_score}%")
                        st.warning("Enhanced monitoring recommended")
                    elif threat_score >= 20:
                        st.info(f"ðŸ” LOW THREAT LEVEL: {threat_score}%")
                        st.info("Normal monitoring sufficient")
                    else:
                        st.success(f"âœ… MINIMAL THREAT: {threat_score}%")
                        st.success("Network appears secure")
                    
                    # Recent security events
                    st.markdown("#### ðŸ” Recent Security Events")
                    
                    security_events = []
                    if external_connections > 50:
                        security_events.append(f"ðŸŒ {external_connections} external connections detected")
                    if suspicious_protocols > 20:
                        security_events.append(f"âš ï¸ {suspicious_protocols} suspicious protocol communications")
                    if large_transfers > 100:
                        security_events.append(f"ðŸ“¦ {large_transfers} large data transfers detected")
                    if potential_scans:
                        security_events.append(f"ðŸ” {len(potential_scans)} potential port scans detected")
                    if len(unique_external_ips) > 10:
                        security_events.append(f"ðŸŒ Communication with {len(unique_external_ips)} external IP addresses")
                    
                    if security_events:
                        for event in security_events:
                            st.write(f"â€¢ {event}")
                    else:
                        st.info("No significant security events detected")
                    
                    # Real-time monitoring status
                    st.markdown("#### âš¡ Live Monitoring Status")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"ðŸ“Š **Analyzing:** `{os.path.basename(latest_file)}`")
                        st.write(f"ðŸ“¦ **Total Packets:** {len(data):,}")
                        st.write(f"ðŸ” **Analyzed:** {len(recent_packets):,} packets (full analysis)")
                    
                    with col2:
                        current_time = datetime.now().strftime("%H:%M:%S")
                        st.write(f"ðŸ• **Last Update:** {current_time}")
                        st.write(f"ðŸ“ **Data Files:** {len(data_files)} available")
                        # Use the actual config value for auto-refresh status
                        if enable_auto_refresh:
                            st.write(f"ðŸ”„ **Auto-refresh:** âœ… Enabled ({auto_refresh_interval//60}min {auto_refresh_interval%60}s)")
                        else:
                            st.write("ðŸ”„ **Auto-refresh:** âŒ Disabled")
                    
                    # Advanced ML-based Anomaly Detection (configurable)
                    if anomaly_enabled:
                        st.markdown("#### ðŸ¤– ML-Based Anomaly Detection")
                        
                        # Create DataFrame for ML analysis
                        try:
                            df_data = []
                            for packet in recent_packets:
                                if isinstance(packet, dict) and '_source' in packet:
                                    layers = packet['_source'].get('layers', {})
                                    
                                    # Extract numeric features for ML - handle list/string conversion safely
                                    def safe_extract_value(field_data, default='0'):
                                        if isinstance(field_data, list):
                                            return field_data[0] if len(field_data) > 0 else default
                                        return field_data if field_data is not None else default
                                    
                                    def safe_int_convert(value, default=0):
                                        try:
                                            return int(value)
                                        except (ValueError, TypeError):
                                            return default
                                    
                                    def safe_protocol_hash(protocol_data):
                                        protocol_str = safe_extract_value(protocol_data, 'UNKNOWN')
                                        if isinstance(protocol_str, str):
                                            return hash(protocol_str) % 1000
                                        return 0
                                    
                                    frame_len = safe_int_convert(safe_extract_value(layers.get('frame.len', '0')))
                                    src_port = safe_int_convert(safe_extract_value(layers.get('tcp.srcport', '0')))
                                    dst_port = safe_int_convert(safe_extract_value(layers.get('tcp.dstport', '0')))
                                    protocol_num = safe_protocol_hash(layers.get('_ws.col.Protocol'))
                                    
                                    # Check external connection safely
                                    src_ip = safe_extract_value(layers.get('ip.src', ''), '')
                                    dst_ip = safe_extract_value(layers.get('ip.dst', ''), '')
                                    src_private = any(src_ip.startswith(pr) for pr in private_ranges) if src_ip else True
                                    dst_private = any(dst_ip.startswith(pr) for pr in private_ranges) if dst_ip else True
                                    external_conn = 1 if not src_private or not dst_private else 0
                                    
                                    packet_features = {
                                        'frame_len': frame_len,
                                        'src_port': src_port,
                                        'dst_port': dst_port,
                                        'protocol_num': protocol_num,
                                        'external_conn': external_conn
                                    }
                                    df_data.append(packet_features)
                            
                            if len(df_data) > 10:  # Need minimum data for ML
                                df = pd.DataFrame(df_data)
                                
                                # Perform configurable anomaly detection
                                anomaly_results, analysis_status = _perform_configurable_security_analysis(df, anomaly_config)
                                
                                # Record anomaly detection in tracker
                                tracker = st.session_state.anomaly_tracker
                                detection_record = tracker.record_anomaly_detection(
                                    anomalies=anomaly_results,
                                    model_type=model_type,
                                    confidence_threshold=confidence_threshold,
                                    source_file=os.path.basename(latest_file),
                                    total_packets=len(df)
                                )
                                
                                # Enhanced display with detailed tracking
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.caption(f"ðŸ” {analysis_status}")
                                with col2:
                                    if detection_record:
                                        st.caption(f"Model: **{model_type.replace('_', ' ').title()}**")
                                    else:
                                        st.caption(f"Model: **{model_type.replace('_', ' ').title()}** (duplicate skipped)")
                                
                                if anomaly_results is not None and len(anomaly_results) > 0:
                                    # Get baseline deviation info
                                    current_rate = detection_record["anomaly_rate"]
                                    baseline_info = tracker.get_baseline_deviation(current_rate)
                                    
                                    # Display enhanced anomaly information
                                    severity = detection_record["severity"]
                                    severity_colors = {
                                        "low": "ðŸŸ¢",
                                        "medium": "ðŸŸ¡", 
                                        "high": "ðŸŸ ",
                                        "critical": "ðŸ”´"
                                    }
                                    
                                    st.error(f"{severity_colors.get(severity, 'âšª')} **{len(anomaly_results)} anomalies detected!** (Severity: {severity.upper()})")
                                    
                                    # Anomaly details in expandable sections
                                    with st.expander("ðŸ” Detailed Anomaly Analysis", expanded=True):
                                        
                                        # Summary metrics
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Anomaly Rate", f"{current_rate:.2f}%", 
                                                     delta=f"{baseline_info.get('deviation', 0):.0f}% vs baseline" if baseline_info.get('baseline_rate', 0) > 0 else None)
                                        with col2:
                                            st.metric("Detection ID", detection_record["detection_id"][-8:])  # Last 8 chars
                                        with col3:
                                            if baseline_info["status"] != "no_baseline":
                                                st.metric("Baseline Status", baseline_info["status"].title(), 
                                                         delta=baseline_info["message"])
                                            else:
                                                st.info("ðŸ“Š Building baseline...")
                                        
                                        # Anomaly breakdown
                                        details = detection_record.get("details", {})
                                        if details:
                                            st.markdown("**ðŸŽ¯ Anomaly Breakdown:**")
                                            
                                            if "size_anomalies" in details and details["size_anomalies"]:
                                                size_info = details["size_anomalies"]
                                                st.write(f"ðŸ“¦ **Large Transfers**: {size_info['count']} packets (max: {size_info['max_size']:,} bytes)")
                                            
                                            if "external_connections" in details:
                                                st.write(f"ðŸŒ **External Connections**: {details['external_connections']} anomalous external connections")
                                            
                                            if "affected_ports" in details and details["affected_ports"]:
                                                ports_str = ", ".join(map(str, details["affected_ports"][:5]))
                                                st.write(f"ðŸ”Œ **Affected Ports**: {ports_str}{'...' if len(details['affected_ports']) > 5 else ''}")
                                        
                                        # Sample anomalous data
                                        st.markdown("**ðŸ“‹ Sample Anomalous Packets:**")
                                        sample_anomalies = anomaly_results.head(3)
                                        st.dataframe(sample_anomalies, use_container_width=True)
                                        
                                        # Quick actions
                                        st.markdown("**âš¡ Quick Actions:**")
                                        action_col1, action_col2, action_col3 = st.columns(3)
                                        with action_col1:
                                            if st.button("âœ… Acknowledge", key=f"ack_{detection_record['detection_id']}"):
                                                tracker.acknowledge_anomaly(detection_record['detection_id'], "Acknowledged from dashboard")
                                                st.success("Anomaly acknowledged!")
                                                st.experimental_rerun()
                                        with action_col2:
                                            if st.button("ðŸ“‹ Export Details", key=f"export_{detection_record['detection_id']}"):
                                                # Create export data
                                                export_data = {
                                                    "detection_record": detection_record,
                                                    "anomaly_data": anomaly_results.to_dict('records')
                                                }
                                                st.download_button(
                                                    "â¬‡ï¸ Download JSON",
                                                    data=json.dumps(export_data, indent=2),
                                                    file_name=f"anomaly_export_{detection_record['detection_id']}.json",
                                                    mime="application/json"
                                                )
                                        with action_col3:
                                            if st.button("ðŸ“Š View History", key=f"history_{detection_record['detection_id']}"):
                                                # Show recent anomalies
                                                st.info("Recent anomalies history shown below...")
                                else:
                                    st.success("âœ… No ML anomalies detected in current data")
                                    
                                    # Still show baseline info for context
                                    if len(df) > 0:
                                        baseline_info = tracker.get_baseline_deviation(0)  # 0% anomaly rate
                                        if baseline_info["status"] != "no_baseline":
                                            st.caption(f"ðŸ“Š Baseline: {baseline_info['message']}")
                            else:
                                st.info("ðŸ“Š Insufficient data for ML analysis (need > 10 packets)")
                                
                        except Exception as e:
                            st.error(f"ML anomaly detection error: {str(e)}")
                    else:
                        st.info("ðŸ¤– ML-based anomaly detection is disabled in configuration")
                
            except Exception as e:
                st.error(f"Error analyzing security data: {e}")
        
        else:
            st.info("ðŸ“­ No data files found for security analysis")
    
    except Exception as e:
        st.error(f"Error in security dashboard: {e}")

def _render_anomaly_history_dashboard():
    """Render anomaly history and trends dashboard"""
    st.markdown("### ðŸ“Š Anomaly History & Trends")
    
    try:
        tracker = st.session_state.anomaly_tracker
        
        # Time range selector
        col1, col2 = st.columns([3, 1])
        with col1:
            time_range = st.selectbox(
                "ðŸ“… Time Range",
                ["Last 24 Hours", "Last 7 Days", "Last 30 Days"],
                index=1
            )
        with col2:
            if st.button("ðŸ”„ Refresh History"):
                st.experimental_rerun()
        
        # Map time range to hours
        hours_map = {
            "Last 24 Hours": 24,
            "Last 7 Days": 24 * 7,
            "Last 30 Days": 24 * 30
        }
        hours = hours_map[time_range]
        
        # Get recent anomalies
        recent_anomalies = tracker.get_recent_anomalies(hours=hours)
        
        if not recent_anomalies:
            st.info(f"ðŸ“­ No anomalies detected in the {time_range.lower()}")
            return
        
        # Summary metrics
        st.markdown("#### ðŸ“ˆ Summary Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        total_anomalies = sum(a["anomaly_count"] for a in recent_anomalies)
        total_detections = len(recent_anomalies)
        avg_anomaly_rate = sum(a["anomaly_rate"] for a in recent_anomalies) / len(recent_anomalies)
        
        # Severity distribution
        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for anomaly in recent_anomalies:
            severity_counts[anomaly["severity"]] += 1
        
        with col1:
            st.metric("Total Detections", total_detections)
        with col2:
            st.metric("Total Anomalies", total_anomalies)
        with col3:
            st.metric("Avg Anomaly Rate", f"{avg_anomaly_rate:.2f}%")
        with col4:
            highest_severity = max(severity_counts.items(), key=lambda x: x[1])
            st.metric("Most Common Severity", highest_severity[0].title(), delta=f"{highest_severity[1]} occurrences")
        
        # Recent anomalies table
        st.markdown("#### ðŸ“‹ Recent Anomaly Detections")
        
        if recent_anomalies:
            # Create display data
            display_data = []
            for anomaly in recent_anomalies[:10]:  # Show last 10
                display_data.append({
                    "Time": datetime.fromisoformat(anomaly["timestamp"]).strftime("%m/%d %H:%M"),
                    "ID": anomaly["detection_id"][-8:],  # Last 8 chars
                    "Model": anomaly["model_type"].replace("_", " ").title(),
                    "Anomalies": anomaly["anomaly_count"],
                    "Rate %": f"{anomaly['anomaly_rate']:.2f}",
                    "Severity": anomaly['severity'].title(),
                    "Status": anomaly.get("status", "new").title()
                })
            
            df_display = pd.DataFrame(display_data)
            st.dataframe(df_display, use_container_width=True)
                
    except Exception as e:
        st.error(f"Error loading anomaly history: {e}")

def _perform_configurable_security_analysis(df, anomaly_config):
    """Perform real-time security analysis using configurable models"""
    try:
        model_type = anomaly_config.get('model_type', 'ensemble')
        confidence_threshold = anomaly_config.get('confidence_threshold', 0.7)
        
        # Select and load the appropriate model
        model_results = None
        
        if model_type == 'ensemble':
            # Load ensemble configuration
            ensemble_config = anomaly_config.get('ensemble_config', {})
            method = ensemble_config.get('method', 'weighted_average')
            
            # Get model weights from the correct nested structure
            models_config = ensemble_config.get('models', {})
            
            # Extract enabled models and their weights
            enabled_models = {}
            for model_name, model_config in models_config.items():
                if isinstance(model_config, dict) and model_config.get('enabled', False):
                    weight = model_config.get('weight', 0)
                    if weight > 0:
                        enabled_models[model_name] = weight
            
            if not enabled_models:
                return None, "No models enabled in ensemble configuration"
            
            # Debug info for troubleshooting
            import logging
            logging.info(f"Ensemble models enabled: {enabled_models}")
            
            # Use ensemble approach
            try:
                from core.models.ensemble import EnsembleAnomalyDetector
                ensemble_detector = EnsembleAnomalyDetector()
                
                # Configure ensemble with enabled models and weights
                model_results = ensemble_detector.detect_anomalies(
                    df, 
                    method=method,
                    model_weights=enabled_models,
                    confidence_threshold=confidence_threshold
                )
            except ImportError:
                return _fallback_to_simple_analysis(df, confidence_threshold)
            
        else:
            # Load single model based on configuration
            model_map = {
                'isolation_forest': 'core.models.isolation_forest',
                'local_outlier_factor': 'core.models.local_outlier_factor', 
                'one_class_svm': 'core.models.one_class_svm',
                'knn': 'core.models.knn_detector',
                'hdbscan': 'core.models.hdbscan_detector'
            }
            
            if model_type in model_map:
                try:
                    module_name = model_map[model_type]
                    module = __import__(module_name, fromlist=[''])
                    
                    # Get the detector class (assuming standard naming)
                    class_name = model_type.replace('_', '').title() + 'AnomalyDetector'
                    if hasattr(module, class_name):
                        detector_class = getattr(module, class_name)
                        detector = detector_class()
                        model_results = detector.detect_anomalies(df, confidence_threshold=confidence_threshold)
                    else:
                        # Fallback to simple analysis if class not found
                        return _fallback_to_simple_analysis(df, confidence_threshold)
                        
                except ImportError:
                    # Fallback to simple analysis if import fails
                    return _fallback_to_simple_analysis(df, confidence_threshold)
            else:
                return _fallback_to_simple_analysis(df, confidence_threshold)
        
        if model_results is not None and len(model_results) > 0:
            # Calculate risk scores
            total_events = len(df)
            anomaly_count = len(model_results)
            risk_score = min((anomaly_count / total_events) * 100, 100) if total_events > 0 else 0
            
            return model_results, f"Detected {anomaly_count} anomalies using {model_type.replace('_', ' ').title()} (Risk: {risk_score:.1f}%)"
        else:
            return None, f"No anomalies detected using {model_type.replace('_', ' ').title()}"
            
    except Exception as e:
        return None, f"Analysis error: {str(e)}"

def _fallback_to_simple_analysis(df, confidence_threshold=0.7):
    """Fallback to simple statistical analysis if ML models fail"""
    try:
        # Simple statistical anomaly detection on frame length
        if 'frame_len' in df.columns and len(df) > 5:
            Q1 = df['frame_len'].quantile(0.25)
            Q3 = df['frame_len'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            anomalies = df[(df['frame_len'] < lower_bound) | (df['frame_len'] > upper_bound)]
            
            if len(anomalies) > 0:
                anomaly_count = len(anomalies)
                total_events = len(df)
                risk_score = min((anomaly_count / total_events) * 100, 100) if total_events > 0 else 0
                return anomalies, f"Detected {anomaly_count} anomalies using Statistical Fallback (Risk: {risk_score:.1f}%)"
            else:
                return None, "No anomalies detected using Statistical Fallback"
        else:
            return None, "Insufficient data for statistical analysis"
            
    except Exception as e:
        return None, f"Fallback analysis error: {str(e)}"

def _render_real_time_metrics_from_data():
    """Render real-time metrics from actual data"""
    st.markdown("### ðŸ“Š Real-time Metrics")
    
    # Get all available data from files - will trigger refresh when files change
    try:
        # Get data files - will trigger refresh when files change
        data_files = _get_data_files()
        
        if data_files:
            total_packets = 0
            unique_ips = set()
            protocols = {}
            total_bytes = 0
            
            # Process all files to get complete metrics (with performance limit)
            file_count = 0
            max_files_to_process = 10  # Reasonable limit for performance
            
            for file_path in data_files:
                if file_count >= max_files_to_process:
                    st.info(f"ðŸ“Š Processing first {max_files_to_process} files for performance. Total files available: {len(data_files)}")
                    break
                    
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    if isinstance(data, list):
                        total_packets += len(data)
                        
                        for packet in data:
                            if isinstance(packet, dict) and '_source' in packet:
                                layers = packet['_source'].get('layers', {})
                                
                                # Extract IPs
                                if 'ip.src' in layers:
                                    unique_ips.add(layers['ip.src'][0] if isinstance(layers['ip.src'], list) else layers['ip.src'])
                                if 'ip.dst' in layers:
                                    unique_ips.add(layers['ip.dst'][0] if isinstance(layers['ip.dst'], list) else layers['ip.dst'])
                                
                                # Extract protocols
                                if '_ws.col.Protocol' in layers:
                                    protocol = layers['_ws.col.Protocol'][0] if isinstance(layers['_ws.col.Protocol'], list) else layers['_ws.col.Protocol']
                                    protocols[protocol] = protocols.get(protocol, 0) + 1
                                
                                # Extract frame length
                                if 'frame.len' in layers:
                                    frame_len = layers['frame.len'][0] if isinstance(layers['frame.len'], list) else layers['frame.len']
                                    try:
                                        total_bytes += int(frame_len)
                                    except:
                                        pass
                except Exception as e:
                    st.warning(f"Error processing {os.path.basename(file_path)}: {e}")
                    continue
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Packets", f"{total_packets:,}")
            
            with col2:
                st.metric("Unique IPs", f"{len(unique_ips)}")
            
            with col3:
                st.metric("Total Bytes", f"{total_bytes:,}")
            
            with col4:
                top_protocol = max(protocols.items(), key=lambda x: x[1])[0] if protocols else "N/A"
                st.metric("Top Protocol", top_protocol)
            
            # Show protocol breakdown
            if protocols:
                st.markdown("**Protocol Distribution:**")
                protocol_df = pd.DataFrame(list(protocols.items()), columns=['Protocol', 'Count'])
                st.dataframe(protocol_df.sort_values('Count', ascending=False), use_container_width=True)
                
        else:
            st.info("No data files found to analyze")
            
    except Exception as e:
        st.error(f"Error processing data: {e}")
        _render_fallback_metrics()

def _render_fallback_metrics():
    """Render fallback metrics when no real data is available"""
    st.markdown("### ðŸ“Š Real-time Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Packets", "0")
    
    with col2:
        st.metric("Unique IPs", "0")
    
    with col3:
        st.metric("Total Bytes", "0")
    
    with col4:
        st.metric("Top Protocol", "N/A")

def _render_network_activity_chart():
    """Render network activity chart using real data"""
    st.markdown("#### ðŸ“ˆ Network Activity")
    
    # Load real data for time series
    try:
        # Get data files - will trigger refresh when files change
        data_files = _get_data_files()
        
        if data_files:
            # Get sample of data to create time series
            timestamps = []
            packet_counts = []
            
            for file_path in data_files:  # Process all files instead of limiting to 2
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list) and len(data) > 0:
                        # Extract timestamps and create time series
                        file_packets = []
                        for packet in data:  # Process all packets instead of just first 100
                            if isinstance(packet, dict) and '_source' in packet:
                                layers = packet['_source'].get('layers', {})
                                if 'frame.time' in layers:
                                    time_str = layers['frame.time'][0] if isinstance(layers['frame.time'], list) else layers['frame.time']
                                    try:
                                        # Parse Arkime time format: "Jul 22, 2025 16:40:25.207432000 +08"
                                        time_parts = time_str.split(' ')
                                        if len(time_parts) >= 3:
                                            date_part = ' '.join(time_parts[:3])
                                            packet_time = datetime.strptime(date_part, "%b %d, %Y")
                                            file_packets.append(packet_time)
                                    except:
                                        continue
                        
                        # Group by minute and count
                        if file_packets:
                            from collections import Counter
                            time_groups = Counter([t.replace(second=0, microsecond=0) for t in file_packets])
                            for time_group, count in time_groups.items():
                                timestamps.append(time_group)
                                packet_counts.append(count)
                                
                except Exception as e:
                    continue
            
            if timestamps and packet_counts:
                # Sort by time
                sorted_data = sorted(zip(timestamps, packet_counts))
                timestamps, packet_counts = zip(*sorted_data)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=timestamps, 
                    y=packet_counts,
                    mode='lines+markers',
                    name='Packets/min',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                fig.update_layout(
                    height=300,
                    showlegend=False,
                    margin=dict(l=0, r=0, t=30, b=0),
                    xaxis_title="Time",
                    yaxis_title="Packets/min"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                return
                
    except Exception as e:
        st.warning(f"Could not load real time data: {e}")
    
    # Fallback to sample data
    timestamps = [datetime.now() - timedelta(minutes=x) for x in range(60, 0, -1)]
    values = [10] * len(timestamps)  # Flat line indicating no real data
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, 
        y=values,
        mode='lines',
        name='No real data',
        line=dict(color='#cccccc', width=1, dash='dash')
    ))
    
    fig.update_layout(
        height=300,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="Time",
        yaxis_title="Packets/min"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.info("ðŸ“Š Real-time chart will show actual data once monitoring is active")

def _render_threat_level_chart():
    """Render threat level chart"""
    st.markdown("#### âš ï¸ Threat Levels")
    
    # Threat level data
    threats = ['Low', 'Medium', 'High', 'Critical']
    values = [75, 20, 4, 1]  # Sample values
    colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545']
    
    fig = go.Figure(data=[go.Pie(
        labels=threats, 
        values=values,
        marker_colors=colors,
        hole=0.4
    )])
    
    fig.update_layout(
        height=300,
        showlegend=True,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def _render_live_packet_stream():
    """Render live packet stream with real-time analysis"""
    st.markdown("### ðŸ“¡ Live Packet Stream")
    
    try:
        # Get data files - will trigger refresh when files change
        data_files = _get_data_files()
        
        if data_files:
            # Get latest packets from most recent file
            latest_file = max(data_files, key=os.path.getmtime)
            
            try:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list) and len(data) > 0:
                    # Show last 10 packets with real-time analysis
                    recent_packets = data[-10:] if len(data) >= 10 else data
                    
                    st.markdown(f"**ðŸ“Š Latest {len(recent_packets)} packets from:** `{os.path.basename(latest_file)}`")
                    
                    # Create columns for packet details
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.markdown("**ðŸ” Packet Analysis**")
                    with col2:
                        st.markdown("**âš¡ Real-time Metrics**")
                    with col3:
                        st.markdown("**ðŸš¨ Threat Score**")
                    
                    # Analyze recent packets
                    for i, packet in enumerate(reversed(recent_packets)):
                        if isinstance(packet, dict) and '_source' in packet:
                            layers = packet['_source'].get('layers', {})
                            
                            # Extract packet info
                            src_ip = layers.get('ip.src', ['Unknown'])[0] if isinstance(layers.get('ip.src'), list) else layers.get('ip.src', 'Unknown')
                            dst_ip = layers.get('ip.dst', ['Unknown'])[0] if isinstance(layers.get('ip.dst'), list) else layers.get('ip.dst', 'Unknown')
                            protocol = layers.get('_ws.col.Protocol', ['Unknown'])[0] if isinstance(layers.get('_ws.col.Protocol'), list) else layers.get('_ws.col.Protocol', 'Unknown')
                            frame_len = layers.get('frame.len', ['0'])[0] if isinstance(layers.get('frame.len'), list) else layers.get('frame.len', '0')
                            
                            # Calculate threat score based on heuristics
                            threat_score = 0
                            threat_reasons = []
                            
                            # Private IP ranges
                            private_ranges = ['192.168.', '10.', '172.16.', '172.17.', '172.18.', '172.19.', '172.20.', '172.21.', '172.22.', '172.23.', '172.24.', '172.25.', '172.26.', '172.27.', '172.28.', '172.29.', '172.30.', '172.31.']
                            
                            # Check for external communication
                            if not any(src_ip.startswith(pr) for pr in private_ranges) or not any(dst_ip.startswith(pr) for pr in private_ranges):
                                threat_score += 30
                                threat_reasons.append("External communication")
                            
                            # Check for suspicious protocols
                            suspicious_protocols = ['SMB', 'SMB2', 'RDP', 'FTP', 'TELNET', 'SSH']
                            if any(sp.lower() in protocol.lower() for sp in suspicious_protocols):
                                threat_score += 25
                                threat_reasons.append(f"Sensitive protocol: {protocol}")
                            
                            # Check for large packets
                            try:
                                if int(frame_len) > 1500:
                                    threat_score += 15
                                    threat_reasons.append("Large packet size")
                            except:
                                pass
                            
                            # Display packet info
                            with st.container():
                                col1, col2, col3 = st.columns([2, 1, 1])
                                
                                with col1:
                                    st.text(f"ðŸ”¹ {src_ip} â†’ {dst_ip}")
                                    st.text(f"   Protocol: {protocol} | Size: {frame_len} bytes")
                                
                                with col2:
                                    timestamp = layers.get('frame.time', ['Now'])[0] if isinstance(layers.get('frame.time'), list) else layers.get('frame.time', 'Now')
                                    if len(timestamp) > 20:
                                        timestamp = timestamp[:20] + "..."
                                    st.text(f"â° {timestamp}")
                                
                                with col3:
                                    if threat_score > 50:
                                        st.error(f"ðŸš¨ {threat_score}%")
                                    elif threat_score > 25:
                                        st.warning(f"âš ï¸ {threat_score}%")
                                    else:
                                        st.success(f"âœ… {threat_score}%")
                                
                                # Show threat reasons if any
                                if threat_reasons:
                                    st.caption(f"ðŸ” {', '.join(threat_reasons)}")
                                
                                st.markdown("---")
                                
                                # Limit to 5 packets for performance
                                if i >= 4:
                                    break
                    
                    # Real-time statistics
                    st.markdown("### ðŸ“ˆ Live Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_packets = len(data)
                        st.metric("Total Packets", f"{total_packets:,}")
                    
                    with col2:
                        unique_protocols = len(set([
                            layers.get('_ws.col.Protocol', ['Unknown'])[0] if isinstance(layers.get('_ws.col.Protocol'), list) 
                            else layers.get('_ws.col.Protocol', 'Unknown')
                            for packet in data 
                            if isinstance(packet, dict) and '_source' in packet
                            for layers in [packet['_source'].get('layers', {})]
                        ]))
                        st.metric("Unique Protocols", unique_protocols)
                    
                    with col3:
                        unique_ips = len(set([
                            ip for packet in data 
                            if isinstance(packet, dict) and '_source' in packet
                            for layers in [packet['_source'].get('layers', {})]
                            for ip_key in ['ip.src', 'ip.dst']
                            if ip_key in layers
                            for ip in [layers[ip_key][0] if isinstance(layers[ip_key], list) else layers[ip_key]]
                        ]))
                        st.metric("Unique IPs", unique_ips)
                    
                    with col4:
                        # Calculate average threat score
                        threat_scores = []
                        for packet in recent_packets:
                            if isinstance(packet, dict) and '_source' in packet:
                                layers = packet['_source'].get('layers', {})
                                score = 0
                                protocol = layers.get('_ws.col.Protocol', [''])[0] if isinstance(layers.get('_ws.col.Protocol'), list) else layers.get('_ws.col.Protocol', '')
                                if any(sp.lower() in protocol.lower() for sp in ['SMB', 'SMB2', 'RDP', 'FTP', 'TELNET', 'SSH']):
                                    score += 25
                                threat_scores.append(score)
                        
                        avg_threat = sum(threat_scores) / len(threat_scores) if threat_scores else 0
                        st.metric("Avg Threat Score", f"{avg_threat:.1f}%")
                
            except Exception as e:
                st.error(f"Error processing packet stream: {e}")
        else:
            st.info("ðŸ“­ No data files found for live packet stream")
    
    except Exception as e:
        st.error(f"Error in live packet stream: {e}")

def _render_network_topology():
    """Render network topology visualization from real data"""
    st.markdown("### ðŸ•¸ï¸ Network Topology")
    
    try:
        # Get data files - will trigger refresh when files change
        data_files = _get_data_files()
        
        if data_files:
            # Analyze network connections
            connections = {}
            node_sizes = {}
            
            # Process a sample of files for performance
            sample_files = data_files[:3] if len(data_files) > 3 else data_files
            
            for file_path in sample_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list):
                        for packet in data:
                            if isinstance(packet, dict) and '_source' in packet:
                                layers = packet['_source'].get('layers', {})
                                
                                src_ip = layers.get('ip.src', ['Unknown'])[0] if isinstance(layers.get('ip.src'), list) else layers.get('ip.src', 'Unknown')
                                dst_ip = layers.get('ip.dst', ['Unknown'])[0] if isinstance(layers.get('ip.dst'), list) else layers.get('ip.dst', 'Unknown')
                                
                                if src_ip != 'Unknown' and dst_ip != 'Unknown':
                                    # Count connections
                                    connection_key = f"{src_ip} -> {dst_ip}"
                                    connections[connection_key] = connections.get(connection_key, 0) + 1
                                    
                                    # Count node activity
                                    node_sizes[src_ip] = node_sizes.get(src_ip, 0) + 1
                                    node_sizes[dst_ip] = node_sizes.get(dst_ip, 0) + 1
                
                except Exception:
                    continue
            
            if connections:
                # Show top connections
                st.markdown("#### ðŸ”— Top Network Connections")
                top_connections = sorted(connections.items(), key=lambda x: x[1], reverse=True)[:15]
                
                # Create a simple network visualization using a table
                connection_data = []
                for connection, count in top_connections:
                    src, dst = connection.split(' -> ')
                    connection_data.append({
                        'Source IP': src,
                        'Destination IP': dst,
                        'Packet Count': count,
                        'Traffic Volume': 'ðŸ”¥' * min(int(count / 10), 5) if count > 0 else 'â–«ï¸'
                    })
                
                df = pd.DataFrame(connection_data)
                st.dataframe(df, use_container_width=True)
                
                # Show network statistics
                st.markdown("#### ðŸ“Š Network Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_connections = len(connections)
                    st.metric("Unique Connections", total_connections)
                
                with col2:
                    total_nodes = len(node_sizes)
                    st.metric("Active Nodes", total_nodes)
                
                with col3:
                    if node_sizes:
                        most_active_ip = max(node_sizes.items(), key=lambda x: x[1])
                        st.metric("Most Active IP", most_active_ip[0][:15] + "..." if len(most_active_ip[0]) > 15 else most_active_ip[0])
                
                # Show top talkers
                if node_sizes:
                    st.markdown("#### ðŸ‘¥ Top Talkers")
                    top_talkers = sorted(node_sizes.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    talker_data = []
                    for ip, count in top_talkers:
                        talker_data.append({
                            'IP Address': ip,
                            'Packet Count': count,
                            'Activity Level': 'ðŸ”¥' * min(int(count / 50), 5) if count > 0 else 'â–«ï¸'
                        })
                    
                    talker_df = pd.DataFrame(talker_data)
                    st.dataframe(talker_df, use_container_width=True)
            
            else:
                st.info("No network connections found in data")
        
        else:
            st.info("ðŸ“­ No data files found for network topology")
    
    except Exception as e:
        st.error(f"Error rendering network topology: {e}")

def _render_live_alerts():
    """Render live alerts section"""
    st.markdown("### ðŸš¨ Live Alerts")
    
    # Load real data directly from files instead of using cached historical data
    try:
        # Get data files - will trigger refresh when files change
        data_files = _get_data_files()
        
        if data_files:
            total_packets = 0
            unique_src_ips = set()
            unique_dst_ips = set()
            protocols = {}
            
            # Process all files to get complete data for alerts
            for file_path in data_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    if isinstance(data, list):
                        total_packets += len(data)
                        
                        for packet in data:
                            if isinstance(packet, dict) and '_source' in packet:
                                layers = packet['_source'].get('layers', {})
                                
                                # Extract source IPs
                                if 'ip.src' in layers:
                                    src_ip = layers['ip.src'][0] if isinstance(layers['ip.src'], list) else layers['ip.src']
                                    unique_src_ips.add(src_ip)
                                
                                # Extract destination IPs
                                if 'ip.dst' in layers:
                                    dst_ip = layers['ip.dst'][0] if isinstance(layers['ip.dst'], list) else layers['ip.dst']
                                    unique_dst_ips.add(dst_ip)
                                
                                # Extract protocols
                                if '_ws.col.Protocol' in layers:
                                    protocol = layers['_ws.col.Protocol'][0] if isinstance(layers['_ws.col.Protocol'], list) else layers['_ws.col.Protocol']
                                    protocols[protocol] = protocols.get(protocol, 0) + 1
                                
                except Exception as e:
                    continue
            
            st.success(f"ðŸ“Š Processing {total_packets:,} packets from real data")
            
            # Generate alerts based on real data analysis
            alerts = []
            
            # Check for high number of unique source IPs
            if len(unique_src_ips) > 100:
                alerts.append(f"âš ï¸ High number of unique source IPs detected: {len(unique_src_ips):,}")
            
            # Check for high number of unique destination IPs
            if len(unique_dst_ips) > 50:
                alerts.append(f"âš ï¸ High number of unique destination IPs detected: {len(unique_dst_ips):,}")
            
            # Check for protocol diversity
            if len(protocols) > 15:
                alerts.append(f"âš ï¸ High protocol diversity detected: {len(protocols)} different protocols")
            
            # Check for suspicious ports or protocols
            suspicious_protocols = ['SMB', 'SMB2', 'RDP', 'SSH', 'FTP', 'TELNET']
            found_suspicious = [p for p in protocols.keys() if any(sp.lower() in p.lower() for sp in suspicious_protocols)]
            if found_suspicious:
                alerts.append(f"ðŸ” Potentially sensitive protocols detected: {', '.join(found_suspicious)}")
            
            # Display alerts or normal status
            if alerts:
                for alert in alerts:
                    st.warning(alert)
            else:
                st.info("âœ… Normal traffic patterns detected")
                
            # Show summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Source IPs:** {len(unique_src_ips):,}")
            with col2:
                st.write(f"**Destination IPs:** {len(unique_dst_ips):,}")
            with col3:
                st.write(f"**Protocols:** {len(protocols)}")
                
        else:
            st.info("ðŸ“­ No data files found for alert analysis")
            
    except Exception as e:
        st.error(f"Error checking for alerts: {e}")
        st.info("ðŸ“­ No alerts - unable to process data files")

def _render_predictive_security_tab():
    """Render the predictive security tab"""
    st.markdown("### ðŸ”® Predictive Security Analytics")
    
    if 'predictive_engine' not in st.session_state:
        return
    
    engine = st.session_state.predictive_engine
    
    # Control panel
    col1, col2 = st.columns(2)
    
    with col1:
        time_horizon = st.selectbox(
            "Prediction Horizon",
            [6, 12, 24, 48],
            index=2,
            format_func=lambda x: f"{x} hours"
        )
    
    with col2:
        if st.button("ðŸ”„ Update Predictions"):
            st.rerun()
    
    st.markdown("---")
    
    # Threat probability predictions
    try:
        threat_predictions = engine.predict_threat_probability(time_horizon)
        
        if threat_predictions:
            st.markdown("#### ðŸ“ˆ Threat Probability Forecast")
            
            # Create prediction chart
            times = [pred.get('timestamp', datetime.now() + timedelta(hours=i)) for i, pred in enumerate(threat_predictions)]
            probabilities = [pred.get('probability', 0) for pred in threat_predictions]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=times,
                y=probabilities,
                mode='lines+markers',
                name='Threat Probability',
                line=dict(color='#ff6b6b', width=2)
            ))
            
            fig.update_layout(
                height=300,
                xaxis_title="Time",
                yaxis_title="Threat Probability",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show current risk level
            current_risk = threat_predictions[0] if threat_predictions else None
            if current_risk:
                risk_level = current_risk.get('risk_category', 'Unknown')
                probability = current_risk.get('probability', 0)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Current Risk Level", risk_level)
                
                with col2:
                    st.metric("Threat Probability", f"{probability:.1%}")
                
                with col3:
                    confidence = current_risk.get('confidence', 0)
                    st.metric("Confidence", f"{confidence:.1%}")
    
    except Exception as e:
        st.error(f"Error generating predictions: {e}")
        st.info("ðŸ“Š Predictive analytics will be available once monitoring starts")

def _render_real_data_security_metrics():
    """Render security metrics using real Arkime data instead of mock data"""
    st.subheader("ðŸ“Š Real Data Security Metrics")
    
    try:
        # Get data files - will trigger refresh when files change
        data_files = _get_data_files()
        
        if data_files:
            # Process all data to get real metrics
            total_packets = 0
            unique_src_ips = set()
            unique_dst_ips = set()
            protocols = {}
            suspicious_ips = set()
            port_analysis = {}
            total_data_bytes = 0
            file_count = 0  # Initialize file_count variable
            
            for file_path in data_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    if isinstance(data, list):
                        total_packets += len(data)
                        
                        for packet in data:
                            if isinstance(packet, dict) and '_source' in packet:
                                layers = packet['_source'].get('layers', {})
                                
                                # Extract source and destination IPs
                                if 'ip.src' in layers:
                                    src_ip = layers['ip.src'][0] if isinstance(layers['ip.src'], list) else layers['ip.src']
                                    unique_src_ips.add(src_ip)
                                    
                                if 'ip.dst' in layers:
                                    dst_ip = layers['ip.dst'][0] if isinstance(layers['ip.dst'], list) else layers['ip.dst']
                                    unique_dst_ips.add(dst_ip)
                                
                                # Protocol analysis
                                if '_ws.col.Protocol' in layers:
                                    protocol = layers['_ws.col.Protocol'][0] if isinstance(layers['_ws.col.Protocol'], list) else layers['_ws.col.Protocol']
                                    protocols[protocol] = protocols.get(protocol, 0) + 1
                                
                                # Port analysis
                                if 'tcp.dstport' in layers:
                                    port = layers['tcp.dstport'][0] if isinstance(layers['tcp.dstport'], list) else layers['tcp.dstport']
                                    port_analysis[f"TCP:{port}"] = port_analysis.get(f"TCP:{port}", 0) + 1
                                elif 'udp.dstport' in layers:
                                    port = layers['udp.dstport'][0] if isinstance(layers['udp.dstport'], list) else layers['udp.dstport']
                                    port_analysis[f"UDP:{port}"] = port_analysis.get(f"UDP:{port}", 0) + 1
                                
                                # Data size analysis
                                if 'frame.len' in layers:
                                    frame_len = layers['frame.len'][0] if isinstance(layers['frame.len'], list) else layers['frame.len']
                                    try:
                                        total_data_bytes += int(frame_len)
                                    except (ValueError, TypeError):
                                        pass
                                
                except Exception as e:
                    st.warning(f"Error processing file {file_path}: {e}")
                
                file_count += 1
            
            # Display real data metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("ðŸ“¦ Total Packets", f"{total_packets:,}")
            
            with col2:
                unique_ips_total = len(unique_src_ips.union(unique_dst_ips))
                st.metric("ðŸŒ Unique IPs", f"{unique_ips_total}")
            
            with col3:
                protocol_count = len(protocols)
                st.metric("ðŸ”Œ Protocols", f"{protocol_count}")
            
            with col4:
                data_mb = total_data_bytes / (1024 * 1024) if total_data_bytes > 0 else 0
                st.metric("ðŸ’¾ Data Size", f"{data_mb:.1f} MB")
            
            st.markdown("---")
            
            # Protocol Distribution Chart
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ðŸ”Œ Protocol Distribution")
                if protocols:
                    # Top 10 protocols
                    top_protocols = sorted(protocols.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    fig = go.Figure(data=[
                        go.Pie(
                            labels=[p[0] for p in top_protocols],
                            values=[p[1] for p in top_protocols],
                            hole=0.3
                        )
                    ])
                    fig.update_layout(
                        title="Top 10 Protocols by Packet Count",
                        height=400,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No protocol data available")
            
            with col2:
                st.markdown("### ðŸšª Port Activity")
                if port_analysis:
                    # Top 10 ports
                    top_ports = sorted(port_analysis.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=[p[0] for p in top_ports],
                            y=[p[1] for p in top_ports],
                            marker_color='lightblue'
                        )
                    ])
                    fig.update_layout(
                        title="Top 10 Ports by Activity",
                        height=400,
                        margin=dict(l=0, r=0, t=30, b=0),
                        xaxis_title="Port",
                        yaxis_title="Packet Count"
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No port activity data available")
            
            # Security Analysis Section
            st.markdown("---")
            st.markdown("### ðŸ›¡ï¸ Security Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**ðŸ“Š Traffic Analysis**")
                communication_pairs = len(unique_src_ips) * len(unique_dst_ips)
                st.write(f"â€¢ Source IPs: {len(unique_src_ips)}")
                st.write(f"â€¢ Destination IPs: {len(unique_dst_ips)}")
                st.write(f"â€¢ Potential Communications: {communication_pairs:,}")
            
            with col2:
                st.markdown("**ðŸ” Anomaly Indicators**")
                # Simple heuristics for potential anomalies
                high_traffic_threshold = total_packets / len(data_files) if len(data_files) > 0 else 0
                if total_packets > high_traffic_threshold * 2:
                    st.write("âš ï¸ High traffic volume detected")
                if len(unique_src_ips) > 100:
                    st.write("âš ï¸ Many unique source IPs")
                if len(protocols) > 10:
                    st.write("âš ï¸ Multiple protocols in use")
                if not any([total_packets > high_traffic_threshold * 2, len(unique_src_ips) > 100, len(protocols) > 10]):
                    st.write("âœ… Normal traffic patterns")
            
            with col3:
                st.markdown("**ðŸ“ˆ Performance Metrics**")
                avg_packet_size = total_data_bytes / total_packets if total_packets > 0 else 0
                st.write(f"â€¢ Avg Packet Size: {avg_packet_size:.0f} bytes")
                st.write(f"â€¢ Files Processed: {len(data_files)}")
                packets_per_file = total_packets / len(data_files) if len(data_files) > 0 else 0
                st.write(f"â€¢ Avg Packets/File: {packets_per_file:.0f}")
                
        else:
            st.warning("ðŸ“­ No data files found. Please check your data directory configuration.")
            st.info("Configure data source in `config/config.yaml`")
    
    except Exception as e:
        st.error(f"Error loading real data metrics: {e}")
        st.info("Please check your configuration and data files.")

def _render_protocol_distribution_chart():
    """Render protocol distribution chart using real Arkime data"""
    st.markdown("### ðŸ”Œ Protocol Distribution")
    
    try:
        # Get data files - will trigger refresh when files change
        data_files = _get_data_files()
        protocols = {}
        
        if data_files:
            for file_path in data_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    if isinstance(data, list):
                        for packet in data:
                            if isinstance(packet, dict) and '_source' in packet:
                                layers = packet['_source'].get('layers', {})
                                if '_ws.col.Protocol' in layers:
                                    protocol = layers['_ws.col.Protocol'][0] if isinstance(layers['_ws.col.Protocol'], list) else layers['_ws.col.Protocol']
                                    protocols[protocol] = protocols.get(protocol, 0) + 1
                except Exception:
                    continue
            
            if protocols:
                # Top 8 protocols for better visibility
                top_protocols = sorted(protocols.items(), key=lambda x: x[1], reverse=True)[:8]
                
                fig = go.Figure(data=[
                    go.Pie(
                        labels=[p[0] for p in top_protocols],
                        values=[p[1] for p in top_protocols],
                        hole=0.3
                    )
                ])
                fig.update_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=30, b=0),
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No protocol data available")
        else:
            st.info("No data files found")
    except Exception as e:
        st.error(f"Error rendering protocol chart: {e}")

def _render_ip_activity_chart():
    """Render IP activity chart using real Arkime data"""
    st.markdown("### ðŸŒ IP Activity")
    
    try:
        # Get data files - will trigger refresh when files change
        data_files = _get_data_files()
        src_ips = {}
        dst_ips = {}
        
        if data_files:
            for file_path in data_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    if isinstance(data, list):
                        for packet in data:
                            if isinstance(packet, dict) and '_source' in packet:
                                layers = packet['_source'].get('layers', {})
                                
                                if 'ip.src' in layers:
                                    src_ip = layers['ip.src'][0] if isinstance(layers['ip.src'], list) else layers['ip.src']
                                    src_ips[src_ip] = src_ips.get(src_ip, 0) + 1
                                    
                                if 'ip.dst' in layers:
                                    dst_ip = layers['ip.dst'][0] if isinstance(layers['ip.dst'], list) else layers['ip.dst']
                                    dst_ips[dst_ip] = dst_ips.get(dst_ip, 0) + 1
                except Exception:
                    continue
            
            if src_ips or dst_ips:
                # Top 10 source IPs
                top_src = sorted(src_ips.items(), key=lambda x: x[1], reverse=True)[:10]
                top_dst = sorted(dst_ips.items(), key=lambda x: x[1], reverse=True)[:10]
                
                fig = go.Figure()
                
                if top_src:
                    fig.add_trace(go.Bar(
                        name='Source IPs',
                        x=[ip[0] for ip in top_src],
                        y=[ip[1] for ip in top_src],
                        marker_color='lightblue'
                    ))
                
                if top_dst:
                    fig.add_trace(go.Bar(
                        name='Destination IPs',
                        x=[ip[0] for ip in top_dst],
                        y=[ip[1] for ip in top_dst],
                        marker_color='lightcoral'
                    ))
                
                fig.update_layout(
                    title="Top 10 IPs by Packet Count",
                    height=300,
                    margin=dict(l=0, r=0, t=30, b=0),
                    xaxis_title="IP Address",
                    yaxis_title="Packet Count",
                    barmode='group'
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No IP activity data available")
        else:
            st.info("No data files found")
    except Exception as e:
        st.error(f"Error rendering IP activity chart: {e}")

if __name__ == "__main__":
    show_real_time_monitoring()
