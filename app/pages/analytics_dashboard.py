"""
Advanced Analytics Dashboard for the Network Anomaly Detection Platform.
Provides comprehensive data insights, trends, and intelligent analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
from collections import Counter

from core.search_engine import SearchEngine
from core.risk_scorer import RiskScorer
from app.components.error_handler import handle_error
from app.components.data_source_selector import ensure_data_available, show_compact_data_status

# Set up logger
logger = logging.getLogger("streamlit_app")

@handle_error
def show_analytics_dashboard():
    """Display the Analytics Dashboard page."""
    
    st.header("📊 Advanced Analytics Dashboard")
    st.markdown("Comprehensive data insights and intelligent analysis for network traffic patterns.")
    
    # Show current data source status
    show_compact_data_status()
    
    # Ensure data is available
    if not ensure_data_available():
        return
    
    df = st.session_state.combined_data
    search_engine = SearchEngine()
    risk_scorer = RiskScorer()
    
    # Create tabs for different analytics
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", "🌐 Network Patterns", "⚠️ Anomaly Analysis", 
        "🔍 Deep Insights", "📋 Reports"
    ])
    
    with tab1:
        st.subheader("📊 Data Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        
        with col2:
            if 'anomaly_indices' in st.session_state:
                anomaly_count = len(st.session_state.anomaly_indices)
                anomaly_rate = (anomaly_count / len(df)) * 100
                st.metric("Anomalies Detected", f"{anomaly_count:,}", f"{anomaly_rate:.2f}%")
            else:
                st.metric("Anomalies Detected", "Not analyzed")
        
        with col3:
            if 'protocol_type' in df.columns:
                unique_protocols = df['protocol_type'].nunique()
                st.metric("Unique Protocols", unique_protocols)
            else:
                st.metric("Unique Protocols", "N/A")
        
        with col4:
            if 'timestamp' in df.columns:
                time_span = "Dynamic"
            else:
                time_span = "Static Data"
            st.metric("Time Span", time_span)
        
        # Data quality assessment
        st.subheader("🔍 Data Quality Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Missing values analysis
            missing_data = df.isnull().sum()
            missing_pct = (missing_data / len(df)) * 100
            
            if missing_data.sum() > 0:
                st.warning("⚠️ Missing Data Detected")
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Count': missing_data.values,
                    'Missing %': missing_pct.values
                })
                missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("✅ No missing data detected")
        
        with col2:
            # Data types overview
            st.write("**Data Types:**")
            dtype_summary = df.dtypes.value_counts()
            # Convert dtype objects to strings for JSON serialization
            fig = px.pie(
                values=dtype_summary.values,
                names=[str(dtype) for dtype in dtype_summary.index],  # Convert to strings
                title="Data Types Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Dataset preview
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
    
    with tab2:
        st.subheader("🌐 Network Traffic Patterns")
        
        # Protocol analysis
        if 'protocol_type' in df.columns:
            st.subheader("Protocol Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                protocol_counts = df['protocol_type'].value_counts()
                fig = px.bar(
                    x=protocol_counts.index,
                    y=protocol_counts.values,
                    title="Protocol Frequency",
                    labels={'x': 'Protocol', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.pie(
                    values=protocol_counts.values,
                    names=protocol_counts.index,
                    title="Protocol Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Port analysis - Use actual Arkime/Wireshark column names
        tcp_src_col = None
        tcp_dst_col = None
        udp_src_col = None
        udp_dst_col = None
        
        # Find the actual port columns in the data
        for col in df.columns:
            if 'tcp_srcport' in col.lower() or 'tcp.srcport' in col.lower():
                tcp_src_col = col
            elif 'tcp_dstport' in col.lower() or 'tcp.dstport' in col.lower():
                tcp_dst_col = col
            elif 'udp_srcport' in col.lower() or 'udp.srcport' in col.lower():
                udp_src_col = col
            elif 'udp_dstport' in col.lower() or 'udp.dstport' in col.lower():
                udp_dst_col = col
        
        if tcp_src_col or tcp_dst_col or udp_src_col or udp_dst_col:
            st.subheader("Port Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top Source Ports:**")
                # Combine TCP and UDP source ports
                src_ports_data = []
                if tcp_src_col and tcp_src_col in df.columns:
                    tcp_src = df[tcp_src_col].dropna()
                    src_ports_data.extend(tcp_src.tolist())
                if udp_src_col and udp_src_col in df.columns:
                    udp_src = df[udp_src_col].dropna()
                    src_ports_data.extend(udp_src.tolist())
                
                if src_ports_data:
                    src_ports = pd.Series(src_ports_data).value_counts().head(10)
                    fig = px.bar(
                        x=src_ports.values,
                        y=src_ports.index,
                        orientation='h',
                        title="Top Source Ports (TCP + UDP)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No source port data found")
            
            with col2:
                st.write("**Top Destination Ports:**")
                # Combine TCP and UDP destination ports
                dst_ports_data = []
                if tcp_dst_col and tcp_dst_col in df.columns:
                    tcp_dst = df[tcp_dst_col].dropna()
                    dst_ports_data.extend(tcp_dst.tolist())
                if udp_dst_col and udp_dst_col in df.columns:
                    udp_dst = df[udp_dst_col].dropna()
                    dst_ports_data.extend(udp_dst.tolist())
                
                if dst_ports_data:
                    dst_ports = pd.Series(dst_ports_data).value_counts().head(10)
                    fig = px.bar(
                        x=dst_ports.values,
                        y=dst_ports.index,
                        orientation='h',
                        title="Top Destination Ports (TCP + UDP)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No destination port data found")
        
        # Packet size analysis - Use actual Arkime column name
        frame_len_col = None
        for col in df.columns:
            if 'frame_len' in col.lower() or 'frame.len' in col.lower():
                frame_len_col = col
                break
        
        if frame_len_col and frame_len_col in df.columns:
            st.subheader("Packet Size Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    df,
                    x=frame_len_col,
                    title="Packet Size Distribution (Frame Length)",
                    nbins=50
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot for packet sizes by protocol
                if 'protocol_type' in df.columns:
                    fig = px.box(
                        df,
                        x='protocol_type',
                        y=frame_len_col,
                        title="Packet Sizes by Protocol"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Advanced network patterns
        st.subheader("🔬 Advanced Network Patterns")
        
        try:
            # Communication patterns
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🌐 Communication Patterns**")
                
                # Check for actual IP columns in Arkime/Wireshark format
                ip_src_cols = [col for col in df.columns if any(x in col.lower() for x in ['ip.src', 'ip_src', 'srcip'])]
                ip_dst_cols = [col for col in df.columns if any(x in col.lower() for x in ['ip.dst', 'ip_dst', 'dstip'])]
                
                if ip_src_cols:
                    src_col = ip_src_cols[0]
                    src_data = df[src_col].value_counts().head(10)
                    st.write(f"Top Source IPs ({src_col}):")
                    st.dataframe(src_data, use_container_width=True)
                else:
                    st.info("No source IP columns found in data")
                
                if ip_dst_cols:
                    dst_col = ip_dst_cols[0]
                    dst_data = df[dst_col].value_counts().head(10)
                    st.write(f"Top Destination IPs ({dst_col}):")
                    st.dataframe(dst_data, use_container_width=True)
                else:
                    st.info("No destination IP columns found in data")
            
            with col2:
                st.write("**⚡ Traffic Patterns**")
                
                # Protocol analysis
                protocol_cols = [col for col in df.columns if 'protocol' in col.lower()]
                if protocol_cols:
                    protocol_col = protocol_cols[0]
                    protocol_dist = df[protocol_col].value_counts()
                    st.write(f"Protocol Distribution ({protocol_col}):")
                    st.dataframe(protocol_dist.head(10), use_container_width=True)
                    
                    # Protocol pie chart
                    fig = px.pie(
                        values=protocol_dist.values[:5],
                        names=protocol_dist.index[:5],
                        title="Top 5 Protocols"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No protocol columns found for traffic analysis")
            
            # Port analysis
            st.write("**🔌 Port Analysis**")
            
            # Use actual Arkime/Wireshark port columns
            port_cols = []
            for col in df.columns:
                if any(x in col.lower() for x in ['tcp_srcport', 'tcp_dstport', 'udp_srcport', 'udp_dstport', 'tcp.srcport', 'tcp.dstport', 'udp.srcport', 'udp.dstport']):
                    port_cols.append(col)
            
            if port_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Show TCP ports
                    tcp_ports = [col for col in port_cols if 'tcp' in col.lower()]
                    for port_col in tcp_ports[:2]:  # Show first 2 TCP port columns
                        if port_col in df.columns:
                            port_data = df[port_col].dropna().value_counts().head(10)
                            if not port_data.empty:
                                st.write(f"Top {port_col}:")
                                st.dataframe(port_data, use_container_width=True)
                
                with col2:
                    # Show UDP ports
                    udp_ports = [col for col in port_cols if 'udp' in col.lower()]
                    for port_col in udp_ports[:2]:  # Show first 2 UDP port columns
                        if port_col in df.columns:
                            port_data = df[port_col].dropna().value_counts().head(10)
                            if not port_data.empty:
                                st.write(f"Top {port_col}:")
                                st.dataframe(port_data, use_container_width=True)
                
                # Port range analysis using the first available port column
                if len(port_cols) > 0:
                    port_col = port_cols[0]
                    try:
                        port_values = pd.to_numeric(df[port_col], errors='coerce').dropna()
                        if len(port_values) > 0:
                            common_ports = port_values[port_values <= 1024].count()
                            high_ports = port_values[port_values > 1024].count()
                            
                            if common_ports > 0 or high_ports > 0:
                                port_summary = pd.DataFrame({
                                    'Port Range': ['Well-known (≤1024)', 'High Ports (>1024)'],
                                    'Count': [common_ports, high_ports]
                                })
                                
                                fig = px.bar(port_summary, x='Port Range', y='Count', title="Port Range Distribution")
                                st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.write(f"Port analysis unavailable for {port_col}")
            else:
                st.info("No port columns found (looking for tcp_srcport, tcp_dstport, udp_srcport, udp_dstport)")
                
            # Advanced pattern detection
            st.write("**🎯 Anomaly Patterns**")
            
            # Check if we have anomaly data
            if 'anomaly_indices' in st.session_state and st.session_state.anomaly_indices is not None:
                anomaly_indices = st.session_state.anomaly_indices
                
                if len(anomaly_indices) > 0:
                    anomalies = df.iloc[anomaly_indices]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Total Anomalies", len(anomalies))
                        anomaly_rate = (len(anomalies) / len(df)) * 100
                        st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
                    
                    with col2:
                        # Show anomaly characteristics
                        if protocol_cols and len(protocol_cols) > 0:
                            anomaly_protocols = anomalies[protocol_cols[0]].value_counts().head(5)
                            st.write("Top Anomaly Protocols:")
                            st.dataframe(anomaly_protocols, use_container_width=True)
                else:
                    st.info("No anomalies detected in current dataset")
            else:
                st.info("Run anomaly detection first to see pattern analysis")
                
        except Exception as e:
            logger.error(f"Error in network patterns analysis: {str(e)}")
            st.error("Error analyzing network patterns. Please check the data format.")
    
    with tab3:
        st.subheader("⚠️ Anomaly Analysis")
        
        if 'anomaly_indices' not in st.session_state:
            st.info("No anomalies detected yet. Please run anomaly detection first.")
            return
        
        anomaly_indices = st.session_state.anomaly_indices
        anomalies = df.iloc[anomaly_indices]
        normal_data = df.drop(anomaly_indices)
        
        # Anomaly overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Anomalies", f"{len(anomalies):,}")
        
        with col2:
            anomaly_rate = (len(anomalies) / len(df)) * 100
            st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
        
        with col3:
            if 'anomaly_scores' in st.session_state:
                avg_score = np.mean([st.session_state.anomaly_scores[i] for i in anomaly_indices])
                st.metric("Avg Anomaly Score", f"{avg_score:.3f}")
        
        # Anomaly characteristics
        st.subheader("🔍 Anomaly Characteristics")
        
        if 'protocol_type' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Protocol Distribution - Anomalies vs Normal:**")
                
                anomaly_protocols = anomalies['protocol_type'].value_counts()
                normal_protocols = normal_data['protocol_type'].value_counts()
                
                comparison_df = pd.DataFrame({
                    'Anomalies': anomaly_protocols,
                    'Normal': normal_protocols
                }).fillna(0)
                
                fig = px.bar(
                    comparison_df,
                    title="Protocol Distribution Comparison",
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'frame_len' in df.columns or any('frame.len' in col.lower() for col in df.columns):
                    frame_len_col = 'frame_len' if 'frame_len' in df.columns else next(col for col in df.columns if 'frame.len' in col.lower())
                    st.write("**Packet Size Distribution:**")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=anomalies[frame_len_col],
                        name='Anomalies',
                        opacity=0.7,
                        nbinsx=30
                    ))
                    fig.add_trace(go.Histogram(
                        x=normal_data[frame_len_col],
                        name='Normal',
                        opacity=0.7,
                        nbinsx=30
                    ))
                    fig.update_layout(
                        title="Packet Size: Anomalies vs Normal",
                        barmode='overlay'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No frame length column found for packet size analysis")
        
        # Anomaly patterns
        st.subheader("📈 Anomaly Patterns")
        
        # Use actual Arkime column names for port analysis
        tcp_src_col = next((col for col in df.columns if 'tcp_srcport' in col.lower() or 'tcp.srcport' in col.lower()), None)
        tcp_dst_col = next((col for col in df.columns if 'tcp_dstport' in col.lower() or 'tcp.dstport' in col.lower()), None)
        udp_src_col = next((col for col in df.columns if 'udp_srcport' in col.lower() or 'udp.srcport' in col.lower()), None)
        udp_dst_col = next((col for col in df.columns if 'udp_dstport' in col.lower() or 'udp.dstport' in col.lower()), None)
        
        if tcp_src_col or tcp_dst_col or udp_src_col or udp_dst_col:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Anomalous Source Ports:**")
                src_ports_data = []
                if tcp_src_col and tcp_src_col in anomalies.columns:
                    src_ports_data.extend(anomalies[tcp_src_col].dropna().tolist())
                if udp_src_col and udp_src_col in anomalies.columns:
                    src_ports_data.extend(anomalies[udp_src_col].dropna().tolist())
                
                if src_ports_data:
                    anomaly_src_ports = pd.Series(src_ports_data).value_counts().head(10)
                    st.dataframe(anomaly_src_ports, use_container_width=True)
                else:
                    st.info("No anomalous source port data")
            
            with col2:
                st.write("**Anomalous Destination Ports:**")
                dst_ports_data = []
                if tcp_dst_col and tcp_dst_col in anomalies.columns:
                    dst_ports_data.extend(anomalies[tcp_dst_col].dropna().tolist())
                if udp_dst_col and udp_dst_col in anomalies.columns:
                    dst_ports_data.extend(anomalies[udp_dst_col].dropna().tolist())
                
                if dst_ports_data:
                    anomaly_dst_ports = pd.Series(dst_ports_data).value_counts().head(10)
                    st.dataframe(anomaly_dst_ports, use_container_width=True)
                else:
                    st.info("No anomalous destination port data")
        else:
            st.info("No port columns found for anomaly analysis")
        
        # Display auto-calculated risk scores if available
        if 'risk_scores' in st.session_state and st.session_state.risk_scores:
            st.subheader("🎯 Risk Score Analysis")
            
            risk_results = st.session_state.risk_scores
            
            if 'individual_scores' in risk_results:
                individual_scores = risk_results['individual_scores']
                
                # Create risk distribution
                risk_levels = [item['risk_level'] for item in individual_scores]
                risk_counts = pd.Series(risk_levels).value_counts()
                
                # Display risk metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("🔴 Critical", risk_counts.get('Critical', 0))
                with col2:
                    st.metric("🟠 High", risk_counts.get('High', 0))
                with col3:
                    st.metric("🟡 Medium", risk_counts.get('Medium', 0))
                with col4:
                    st.metric("🟢 Low", risk_counts.get('Low', 0))
                with col5:
                    st.metric("⚪ Minimal", risk_counts.get('Minimal', 0))
                
                # Risk score distribution chart
                if len(individual_scores) > 0:
                    scores_df = pd.DataFrame(individual_scores)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Risk level pie chart
                        fig_pie = px.pie(
                            values=list(risk_counts.values),
                            names=list(risk_counts.index),
                            title="Risk Level Distribution",
                            color_discrete_map={
                                'Critical': '#FF0000',
                                'High': '#FF8C00', 
                                'Medium': '#FFD700',
                                'Low': '#32CD32',
                                'Minimal': '#CCCCCC'
                            }
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col2:
                        # Risk score histogram
                        fig_hist = px.histogram(
                            scores_df, 
                            x='risk_score',
                            nbins=20,
                            title="Risk Score Distribution",
                            labels={'risk_score': 'Risk Score', 'count': 'Count'}
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
            
            # Display summary if available
            if 'summary' in risk_results:
                summary = risk_results['summary']
                st.subheader("📊 Risk Summary")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Risk", f"{summary.get('avg_risk_score', 0):.3f}")
                with col2:
                    st.metric("Max Risk", f"{summary.get('max_risk_score', 0):.3f}")
                with col3:
                    st.metric("High Risk Count", summary.get('high_risk_count', 0))
                    
        else:
            st.info("💡 **Tip:** Risk scores are automatically calculated after anomaly detection. Go to the Anomaly Detection page and run detection with auto-analysis enabled.")
    
    with tab4:
        st.subheader("🔍 Deep Insights & Correlations")
        
        # Feature correlation analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            st.subheader("🔗 Feature Correlations")
            
            correlation_matrix = df[numeric_cols].corr()
            
            # Create correlation heatmap
            fig = px.imshow(
                correlation_matrix,
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Find strong correlations
            strong_corr = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:  # Strong correlation threshold
                        strong_corr.append({
                            'Feature 1': correlation_matrix.columns[i],
                            'Feature 2': correlation_matrix.columns[j],
                            'Correlation': corr_val
                        })
            
            if strong_corr:
                st.subheader("🔥 Strong Correlations (|r| > 0.7)")
                strong_corr_df = pd.DataFrame(strong_corr)
                st.dataframe(strong_corr_df, use_container_width=True)
        
        # Statistical insights
        st.subheader("📊 Statistical Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Numerical Features Summary:**")
            if len(numeric_cols) > 0:
                stats_summary = df[numeric_cols].describe()
                st.dataframe(stats_summary.transpose(), use_container_width=True)
        
        with col2:
            st.write("**Categorical Features Summary:**")
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                cat_summary = []
                for col in categorical_cols:
                    unique_count = df[col].nunique()
                    most_common = df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A'
                    cat_summary.append({
                        'Column': col,
                        'Unique Values': unique_count,
                        'Most Common': most_common
                    })
                
                cat_df = pd.DataFrame(cat_summary)
                st.dataframe(cat_df, use_container_width=True)
        
        # Advanced pattern detection
        st.subheader("🎯 Pattern Detection")
        
        # Detect potential patterns using search engine
        if st.button("🔍 Detect Suspicious Patterns"):
            with st.spinner("Analyzing patterns..."):
                suspicious_patterns = []
                
                # Check for common attack patterns using actual column names
                attack_patterns = []
                
                # Check what columns we actually have
                available_cols = df.columns.tolist()
                
                # Add patterns based on available columns
                if any('tcp_dstport' in col.lower() for col in available_cols):
                    attack_patterns.append(("SSH Activity", "tcp_dstport == 22"))
                if any('frame_len' in col.lower() for col in available_cols):
                    attack_patterns.append(("Large Packets", "frame_len > 1500"))
                if any('tcp_srcport' in col.lower() for col in available_cols):
                    attack_patterns.append(("High Source Ports", "tcp_srcport > 1024"))
                if any('udp_dstport' in col.lower() for col in available_cols):
                    attack_patterns.append(("DNS Activity", "udp_dstport == 53"))
                
                st.write(f"**Available columns for analysis:** {', '.join(available_cols[:10])}{'...' if len(available_cols) > 10 else ''}")
                
                for pattern_name, pattern_query in attack_patterns:
                    try:
                        # Simple pattern matching based on available columns
                        matches_found = False
                        
                        if "tcp_dstport == 22" in pattern_query and any('tcp_dstport' in col.lower() for col in available_cols):
                            tcp_dst_col = next(col for col in available_cols if 'tcp_dstport' in col.lower())
                            matches = df[df[tcp_dst_col] == 22]
                            if not matches.empty:
                                matches_found = True
                                suspicious_patterns.append({
                                    'Pattern': pattern_name,
                                    'Matches': len(matches),
                                    'Percentage': (len(matches) / len(df)) * 100
                                })
                        
                        elif "frame_len > 1500" in pattern_query and any('frame_len' in col.lower() for col in available_cols):
                            frame_col = next(col for col in available_cols if 'frame_len' in col.lower())
                            matches = df[df[frame_col] > 1500]
                            if not matches.empty:
                                matches_found = True
                                suspicious_patterns.append({
                                    'Pattern': pattern_name,
                                    'Matches': len(matches),
                                    'Percentage': (len(matches) / len(df)) * 100
                                })
                        
                        elif "tcp_srcport > 1024" in pattern_query and any('tcp_srcport' in col.lower() for col in available_cols):
                            tcp_src_col = next(col for col in available_cols if 'tcp_srcport' in col.lower())
                            matches = df[df[tcp_src_col] > 1024]
                            if not matches.empty:
                                matches_found = True
                                suspicious_patterns.append({
                                    'Pattern': pattern_name,
                                    'Matches': len(matches),
                                    'Percentage': (len(matches) / len(df)) * 100
                                })
                        
                        elif "udp_dstport == 53" in pattern_query and any('udp_dstport' in col.lower() for col in available_cols):
                            udp_dst_col = next(col for col in available_cols if 'udp_dstport' in col.lower())
                            matches = df[df[udp_dst_col] == 53]
                            if not matches.empty:
                                matches_found = True
                                suspicious_patterns.append({
                                    'Pattern': pattern_name,
                                    'Matches': len(matches),
                                    'Percentage': (len(matches) / len(df)) * 100
                                })
                        
                    except Exception as e:
                        logger.warning(f"Pattern detection failed for {pattern_name}: {e}")
                        st.write(f"Pattern '{pattern_name}' analysis failed: {str(e)}")
                
                if suspicious_patterns:
                    st.success("✅ Suspicious patterns detected!")
                    pattern_df = pd.DataFrame(suspicious_patterns)
                    st.dataframe(pattern_df, use_container_width=True)
                else:
                    st.info("No suspicious patterns detected.")
    
    with tab5:
        st.subheader("📋 Comprehensive Reports")
        
        # Generate comprehensive report
        if st.button("📄 Generate Analytics Report"):
            with st.spinner("Generating comprehensive report..."):
                
                report_sections = []
                
                # Executive Summary
                report_sections.append("# Network Traffic Analytics Report")
                report_sections.append(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                report_sections.append("")
                
                # Data Overview
                report_sections.append("## Executive Summary")
                report_sections.append(f"- **Total Records Analyzed:** {len(df):,}")
                
                if 'anomaly_indices' in st.session_state:
                    anomaly_count = len(st.session_state.anomaly_indices)
                    anomaly_rate = (anomaly_count / len(df)) * 100
                    report_sections.append(f"- **Anomalies Detected:** {anomaly_count:,} ({anomaly_rate:.2f}%)")
                
                if 'protocol_type' in df.columns:
                    protocols = df['protocol_type'].nunique()
                    top_protocol = df['protocol_type'].mode().iloc[0]
                    report_sections.append(f"- **Protocols Observed:** {protocols}")
                    report_sections.append(f"- **Most Common Protocol:** {top_protocol}")
                
                report_sections.append("")
                
                # Protocol Analysis
                if 'protocol_type' in df.columns:
                    report_sections.append("## Protocol Distribution")
                    protocol_counts = df['protocol_type'].value_counts()
                    for protocol, count in protocol_counts.head().items():
                        percentage = (count / len(df)) * 100
                        report_sections.append(f"- **{protocol}:** {count:,} ({percentage:.1f}%)")
                    report_sections.append("")
                
                # Anomaly Analysis
                if 'anomaly_indices' in st.session_state:
                    report_sections.append("## Anomaly Analysis")
                    anomalies = df.iloc[st.session_state.anomaly_indices]
                    
                    if 'protocol_type' in df.columns:
                        anomaly_protocols = anomalies['protocol_type'].value_counts()
                        report_sections.append("**Anomalous Protocols:**")
                        for protocol, count in anomaly_protocols.head().items():
                            report_sections.append(f"- {protocol}: {count}")
                    
                    report_sections.append("")
                
                # Recommendations
                report_sections.append("## Recommendations")
                
                if 'anomaly_indices' in st.session_state:
                    anomaly_rate = (len(st.session_state.anomaly_indices) / len(df)) * 100
                    if anomaly_rate > 10:
                        report_sections.append("- ⚠️ **High anomaly rate detected** - Consider reviewing detection thresholds")
                    elif anomaly_rate < 1:
                        report_sections.append("- ℹ️ **Low anomaly rate** - Detection sensitivity may need adjustment")
                    else:
                        report_sections.append("- ✅ **Normal anomaly rate** - Detection appears well-calibrated")
                
                if 'protocol_type' in df.columns:
                    unusual_protocols = df['protocol_type'].value_counts()
                    low_count_protocols = unusual_protocols[unusual_protocols < 10]
                    if not low_count_protocols.empty:
                        report_sections.append(f"- 🔍 **Investigate rare protocols:** {', '.join(low_count_protocols.index[:5])}")
                
                report_sections.append("- 📊 **Continue monitoring** - Regular analysis recommended")
                report_sections.append("- 🤖 **Consider auto-labeling** - Use AI to speed up analysis")
                
                # Join all sections
                full_report = "\n".join(report_sections)
                
                st.success("✅ Report generated successfully!")
                
                # Display report
                st.markdown(full_report)
                
                # Download option
                st.download_button(
                    "📥 Download Report",
                    data=full_report,
                    file_name=f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
        
        # Export options
        st.subheader("📤 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Full Dataset"):
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download CSV",
                    data=csv,
                    file_name=f"full_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if 'anomaly_indices' in st.session_state:
                if st.button("⚠️ Export Anomalies Only"):
                    anomalies = df.iloc[st.session_state.anomaly_indices]
                    csv = anomalies.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Anomalies CSV",
                        data=csv,
                        file_name=f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        with col3:
            if st.button("📈 Export Summary Stats"):
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    stats = df[numeric_cols].describe()
                    csv = stats.to_csv().encode('utf-8')
                    st.download_button(
                        "Download Statistics CSV",
                        data=csv,
                        file_name=f"statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
