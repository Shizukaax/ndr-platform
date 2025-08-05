"""
Visualization components for the Network Anomaly Detection Platform.
Provides various plotting functions for data and anomaly visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import networkx as nx
from pyvis.network import Network
import tempfile
import os
import json
import logging
from typing import Tuple, Dict, List, Any, Optional, Union
import re

logger = logging.getLogger("streamlit_app")

def get_anomaly_mask(scores: np.ndarray, threshold: float) -> np.ndarray:
    """
    Get boolean mask for anomalies based on the current algorithm.
    
    Args:
        scores: Array of anomaly scores
        threshold: Anomaly detection threshold
        
    Returns:
        Boolean array where True indicates an anomaly
    """
    # Get current algorithm from session state
    algorithm = st.session_state.get('selected_model', 'Unknown')
    
    # For Isolation Forest: lower scores = more anomalous
    if algorithm in ['Isolation Forest', 'IsolationForest']:
        return scores < threshold
    else:
        # For other algorithms: higher scores = more anomalous
        return scores > threshold

def find_timestamp_column(df: pd.DataFrame) -> Tuple[Optional[str], Optional[pd.Series]]:
    """
    Find a suitable timestamp column in the dataframe.
    
    Args:
        df: Input dataframe
        
    Returns:
        Tuple of (column_name, processed_timestamp_series)
    """
    # Check if the dataframe has any columns
    if df.empty or len(df.columns) == 0:
        return None, None
    
    # First, check for columns with datetime dtype
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    if datetime_cols:
        return datetime_cols[0], df[datetime_cols[0]]
    
    # List of possible timestamp column names
    timestamp_cols = [
        'timestamp', 'time', 'date', 'datetime', 'frame.time', 
        'frame.time_epoch', 'frame_time', 'frame.time_relative',
        'sniff_timestamp', 'time_epoch', 'datetime', 'date_time'
    ]
    
    # Check for standard timestamp column names
    for col in timestamp_cols:
        if col in df.columns:
            try:
                # Try to convert to datetime
                # Explicitly specify error handling to avoid dateutil warnings
                time_series = pd.to_datetime(df[col], errors='coerce', format='mixed')
                
                # Check if conversion was successful
                if not time_series.isna().all():
                    return col, time_series
            except Exception as e:
                logger.debug(f"Failed to convert {col} to datetime: {str(e)}")
                continue
    
    # If no standard columns are found, look for columns with timestamp-like values
    for col in df.columns:
        # Skip obvious non-datetime columns
        if col in ['id', 'index', 'ip_src', 'ip_dst', 'ip.src', 'ip.dst', 
                  'tcp.srcport', 'tcp.dstport', 'udp.srcport', 'udp.dstport',
                  'protocol', 'length', 'source', 'destination']:
            continue
        
        # Only check string columns
        if df[col].dtype == 'object':
            # Get the first non-null value
            sample = df[col].dropna().astype(str).iloc[0] if not df[col].dropna().empty else ""
            
            # Check if it looks like a timestamp
            if re.search(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', sample) or \
               re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{4}', sample) or \
               re.search(r'\d{1,2}:\d{1,2}:\d{1,2}', sample) or \
               re.search(r'\w{3}\s+\d{1,2},\s+\d{4}', sample):  # Jul 22, 2025
                try:
                    # Try to convert to datetime using explicit format detection
                    if 'Jul' in sample or 'Aug' in sample or 'Sep' in sample:
                        # Handle the specific format seen in your data
                        time_series = pd.to_datetime(df[col], errors='coerce', 
                                                    format='%b %d, %Y %H:%M:%S.%f %z')
                    else:
                        # Use coerce to handle errors and create NaT for unparseable values
                        time_series = pd.to_datetime(df[col], errors='coerce', format='mixed')
                    
                    # Check if conversion was successful
                    if not time_series.isna().all():
                        return col, time_series
                except Exception as e:
                    logger.debug(f"Failed to convert {col} to datetime: {str(e)}")
                    continue
    
    # If still not found, return None
    return None, None

def plot_anomaly_scores(df: pd.DataFrame, scores: np.ndarray, threshold: float) -> go.Figure:
    """
    Plot anomaly scores distribution with threshold line.
    
    Args:
        df: Input dataframe
        scores: Anomaly scores array
        threshold: Anomaly threshold value
        
    Returns:
        Plotly figure object
    """
    # Create a histogram of anomaly scores
    fig = go.Figure()
    
    # Add histogram trace
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=50,
        name="Score Distribution",
        marker_color='royalblue',
        opacity=0.7
    ))
    
    # Add threshold line
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"Threshold: {threshold:.4f}",
        annotation_position="top right"
    )
    
    # Calculate statistics using correct anomaly detection logic
    total_points = len(scores)
    anomaly_mask = get_anomaly_mask(scores, threshold)
    anomaly_count = np.sum(anomaly_mask)
    anomaly_percent = (anomaly_count / total_points) * 100 if total_points > 0 else 0
    
    # Add annotations with statistics
    fig.add_annotation(
        x=0.05,
        y=0.95,
        xref="paper",
        yref="paper",
        text=f"Total records: {total_points}<br>Anomalies: {anomaly_count} ({anomaly_percent:.2f}%)",
        showarrow=False,
        font=dict(size=12),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4
    )
    
    # Update layout
    fig.update_layout(
        title="Anomaly Score Distribution",
        xaxis_title="Anomaly Score",
        yaxis_title="Count",
        hovermode="closest",
        template="plotly_white",
        margin=dict(l=50, r=50, b=50, t=50, pad=4)
    )
    
    return fig

def plot_anomaly_scatter(df: pd.DataFrame, scores: np.ndarray, threshold: float, 
                        x_feature: str, y_feature: str) -> go.Figure:
    """
    Plot scatter plot of two features with anomalies highlighted.
    
    Args:
        df: Input dataframe
        scores: Anomaly scores array
        threshold: Anomaly threshold value
        x_feature: Feature name for x-axis
        y_feature: Feature name for y-axis
        
    Returns:
        Plotly figure object
    """
    # Create a copy of the dataframe with anomaly scores
    df_with_scores = df.copy()
    df_with_scores['anomaly_score'] = scores
    df_with_scores['is_anomaly'] = get_anomaly_mask(scores, threshold)
    
    # Convert anomaly status to string for better display
    df_with_scores['anomaly_status'] = df_with_scores['is_anomaly'].apply(
        lambda x: 'Anomaly' if x else 'Normal'
    )
    
    # Create scatter plot
    fig = px.scatter(
        df_with_scores, 
        x=x_feature, 
        y=y_feature,
        color='anomaly_status',
        color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'},
        hover_data=['anomaly_score'],
        opacity=0.7,
        title=f"Anomaly Detection: {x_feature} vs {y_feature}"
    )
    
    # FIX: Instead of using size parameter with potentially negative values,
    # use a fixed size for markers based on status
    fig.update_traces(
        selector=dict(name='Normal'),
        marker=dict(size=8)
    )
    
    fig.update_traces(
        selector=dict(name='Anomaly'),
        marker=dict(size=12)
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=x_feature,
        yaxis_title=y_feature,
        legend_title="Status",
        template="plotly_white",
        margin=dict(l=50, r=50, b=50, t=50, pad=4)
    )
    
    return fig

def plot_anomaly_timeline(df: pd.DataFrame, scores: np.ndarray, threshold: float, 
                         time_col: str = None, window: str = "1h") -> Optional[go.Figure]:
    """
    Plot time series of anomaly scores with threshold line.
    
    Args:
        df: Input dataframe
        scores: Anomaly scores array
        threshold: Anomaly threshold value
        time_col: Column containing timestamps (if None, will try to detect)
        window: Time window for aggregation
        
    Returns:
        Plotly figure object or None if no timestamp column found
    """
    # Find timestamp column if not provided
    if time_col is None:
        time_col, time_series = find_timestamp_column(df)
        if time_col is None:
            return None
    else:
        # Convert to datetime if not already
        try:
            # First, handle special case of date format seen in your data
            if df[time_col].dtype == 'object':
                sample = df[time_col].dropna().astype(str).iloc[0] if not df[time_col].dropna().empty else ""
                if 'Jul' in sample or 'Aug' in sample or 'Sep' in sample:
                    try:
                        # Try the specific format
                        time_series = pd.to_datetime(df[time_col], errors='coerce', 
                                                   format='%b %d, %Y %H:%M:%S.%f %z')
                    except:
                        # Fall back to generic parsing
                        time_series = pd.to_datetime(df[time_col], errors='coerce')
                else:
                    time_series = pd.to_datetime(df[time_col], errors='coerce')
            else:
                time_series = pd.to_datetime(df[time_col], errors='coerce')
                
            # Check if conversion was successful
            if time_series.isna().all():
                return None
        except Exception as e:
            logger.error(f"Error converting {time_col} to datetime: {str(e)}")
            return None
    
    # Create a copy of the dataframe with anomaly scores
    df_with_scores = df.copy()
    df_with_scores['anomaly_score'] = scores
    df_with_scores['is_anomaly'] = get_anomaly_mask(scores, threshold)
    df_with_scores['timestamp'] = time_series
    
    # Sort by timestamp
    df_with_scores = df_with_scores.sort_values('timestamp')
    
    # Resample data by time window
    try:
        # Set timestamp as index
        df_resampled = df_with_scores.set_index('timestamp')
        
        # Resample and count anomalies
        anomaly_count = df_resampled['is_anomaly'].resample(window).sum()
        total_count = df_resampled['is_anomaly'].resample(window).count()
        
        # Calculate anomaly percentage
        anomaly_percent = (anomaly_count / total_count) * 100
        anomaly_percent = anomaly_percent.fillna(0)
        
        # Get average anomaly score
        avg_score = df_resampled['anomaly_score'].resample(window).mean()
        
        # Reset index for plotting
        timeline_df = pd.DataFrame({
            'timestamp': anomaly_count.index,
            'anomaly_count': anomaly_count.values,
            'total_count': total_count.values,
            'anomaly_percent': anomaly_percent.values,
            'avg_score': avg_score.values
        })
    except Exception as e:
        logger.error(f"Error resampling data: {str(e)}")
        # Fall back to not resampling
        timeline_df = df_with_scores
    
    # Create timeline plot
    fig = go.Figure()
    
    # Add trace for anomaly percentage
    fig.add_trace(go.Scatter(
        x=timeline_df['timestamp'],
        y=timeline_df['anomaly_percent'],
        mode='lines+markers',
        name='Anomaly %',
        line=dict(color='red', width=2),
        marker=dict(size=6),
        yaxis='y'
    ))
    
    # Add trace for total count
    fig.add_trace(go.Scatter(
        x=timeline_df['timestamp'],
        y=timeline_df['total_count'],
        mode='lines',
        name='Total Records',
        line=dict(color='blue', width=1, dash='dot'),
        yaxis='y2'
    ))
    
    # Update layout with dual y-axes
    fig.update_layout(
        title=f"Anomaly Timeline (Window: {window})",
        xaxis=dict(title="Time"),
        yaxis=dict(
            title="Anomaly %",
            side="left",
            range=[0, max(100, timeline_df['anomaly_percent'].max() * 1.1)],
            gridcolor="lightgray"
        ),
        yaxis2=dict(
            title="Record Count",
            side="right",
            overlaying="y",
            range=[0, timeline_df['total_count'].max() * 1.1],
            gridcolor="lightgray"
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        template="plotly_white",
        margin=dict(l=50, r=50, b=50, t=50, pad=4)
    )
    
    return fig

def plot_feature_importance(df: pd.DataFrame, scores: np.ndarray, threshold: float) -> Optional[go.Figure]:
    """
    Plot feature importance based on correlation with anomaly scores.
    
    Args:
        df: Input dataframe
        scores: Anomaly scores array
        threshold: Anomaly threshold value
        
    Returns:
        Plotly figure object or None if no numeric features
    """
    # Create a copy of the dataframe with anomaly scores
    df_with_scores = df.copy()
    df_with_scores['anomaly_score'] = scores
    df_with_scores['is_anomaly'] = get_anomaly_mask(scores, threshold)
    
    # Get numeric columns except the added ones
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['anomaly_score', 'is_anomaly']]
    
    if not numeric_cols:
        return None
    
    # Calculate correlation with anomaly score
    feature_importance = {}
    for col in numeric_cols:
        # Handle potential NaN values
        valid_indices = ~np.isnan(df_with_scores[col])
        if np.sum(valid_indices) > 10:  # Require at least 10 valid values
            correlation = np.corrcoef(
                df_with_scores[col][valid_indices], 
                df_with_scores['anomaly_score'][valid_indices]
            )[0, 1]
            # Use absolute correlation as importance
            feature_importance[col] = abs(correlation)
    
    # If no valid correlations, return None
    if not feature_importance:
        return None
    
    # Convert to dataframe and sort
    importance_df = pd.DataFrame({
        'Feature': list(feature_importance.keys()),
        'Importance': list(feature_importance.values())
    }).sort_values('Importance', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        importance_df, 
        x='Feature', 
        y='Importance',
        color='Importance',
        color_continuous_scale='viridis',
        title="Feature Importance (Correlation with Anomaly Score)",
        template="plotly_white"
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Feature",
        yaxis_title="Importance (|Correlation|)",
        coloraxis_showscale=False,
        margin=dict(l=50, r=50, b=100, t=50, pad=4)
    )
    
    # Rotate x-axis labels
    fig.update_xaxes(tickangle=45)
    
    return fig

def plot_protocol_pie(df: pd.DataFrame, protocol_col: str = 'protocol') -> Optional[go.Figure]:
    """
    Plot pie chart of protocol distribution.
    
    Args:
        df: Input dataframe
        protocol_col: Column containing protocol information
        
    Returns:
        Plotly figure object or None if protocol column not found
    """
    if protocol_col not in df.columns:
        return None
    
    # Count protocols
    protocol_counts = df[protocol_col].value_counts()
    
    # Remove empty or NaN values
    protocol_counts = protocol_counts.dropna()
    
    # Filter out empty string
    if '' in protocol_counts.index:
        protocol_counts = protocol_counts.drop('')
    
    # If no valid protocols, return None
    if len(protocol_counts) == 0:
        return None
    
    # Create pie chart
    fig = px.pie(
        values=protocol_counts.values,
        names=protocol_counts.index,
        title="Protocol Distribution",
        hole=0.4,
        template="plotly_white"
    )
    
    # Update layout
    fig.update_layout(
        legend_title="Protocol",
        margin=dict(l=50, r=50, b=50, t=50, pad=4)
    )
    
    # Update traces
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        marker=dict(line=dict(color='#FFFFFF', width=2))
    )
    
    return fig

def plot_network_graph(df: pd.DataFrame, src_ip_col: str = 'ip_src', dst_ip_col: str = 'ip_dst',
                      scores: Optional[np.ndarray] = None, threshold: float = None,
                      max_nodes: int = 50) -> Optional[str]:
    """
    Create interactive network graph visualization.
    
    Args:
        df: Input dataframe
        src_ip_col: Column containing source IP addresses
        dst_ip_col: Column containing destination IP addresses
        scores: Anomaly scores array
        threshold: Anomaly threshold value
        max_nodes: Maximum number of nodes to display
        
    Returns:
        HTML string of the interactive graph or None if IP columns not found
    """
    if src_ip_col not in df.columns or dst_ip_col not in df.columns:
        return None
    
    # Create a copy of the dataframe
    network_df = df.copy()
    
    # Add anomaly scores if provided
    if scores is not None and threshold is not None:
        network_df['anomaly_score'] = scores
        network_df['is_anomaly'] = get_anomaly_mask(scores, threshold)
    
    # Count connections between IPs
    connection_counts = network_df.groupby([src_ip_col, dst_ip_col]).size().reset_index(name='count')
    
    # If too many connections, filter to keep only the most frequent ones
    if len(connection_counts) > max_nodes * 2:
        connection_counts = connection_counts.sort_values('count', ascending=False).head(max_nodes * 2)
    
    # Get unique IPs
    unique_src_ips = connection_counts[src_ip_col].unique()
    unique_dst_ips = connection_counts[dst_ip_col].unique()
    all_ips = list(set(unique_src_ips) | set(unique_dst_ips))
    
    # If too many nodes, filter to keep only the most connected ones
    if len(all_ips) > max_nodes:
        # Count connections for each IP
        ip_connections = {}
        for ip in all_ips:
            count_as_src = connection_counts[connection_counts[src_ip_col] == ip]['count'].sum()
            count_as_dst = connection_counts[connection_counts[dst_ip_col] == ip]['count'].sum()
            ip_connections[ip] = count_as_src + count_as_dst
        
        # Sort IPs by connection count and keep top max_nodes
        sorted_ips = sorted(ip_connections.items(), key=lambda x: x[1], reverse=True)
        top_ips = [ip for ip, _ in sorted_ips[:max_nodes]]
        
        # Filter connections to include only top IPs
        connection_counts = connection_counts[
            (connection_counts[src_ip_col].isin(top_ips)) & 
            (connection_counts[dst_ip_col].isin(top_ips))
        ]
        
        # Update unique IPs list
        unique_src_ips = connection_counts[src_ip_col].unique()
        unique_dst_ips = connection_counts[dst_ip_col].unique()
        all_ips = list(set(unique_src_ips) | set(unique_dst_ips))
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes
    for ip in all_ips:
        # Check if this IP is associated with anomalies
        if scores is not None and threshold is not None:
            # Find all rows where this IP is source or destination
            ip_rows = network_df[
                (network_df[src_ip_col] == ip) | 
                (network_df[dst_ip_col] == ip)
            ]
            
            # Check if any of these rows are anomalies
            has_anomaly = ip_rows['is_anomaly'].any() if 'is_anomaly' in ip_rows.columns else False
            anomaly_count = ip_rows['is_anomaly'].sum() if 'is_anomaly' in ip_rows.columns else 0
            
            # Set node color based on anomaly status
            if has_anomaly:
                node_color = 'red'
                title = f"{ip} (Anomalies: {anomaly_count})"
            else:
                node_color = 'blue'
                title = ip
        else:
            node_color = 'blue'
            title = ip
        
        # Add node to graph
        G.add_node(ip, color=node_color, title=title)
    
    # Add edges
    for _, row in connection_counts.iterrows():
        src = row[src_ip_col]
        dst = row[dst_ip_col]
        count = row['count']
        
        # Check if this connection is associated with anomalies
        if scores is not None and threshold is not None:
            # Find all rows with this source-destination pair
            conn_rows = network_df[
                (network_df[src_ip_col] == src) & 
                (network_df[dst_ip_col] == dst)
            ]
            
            # Check if any of these rows are anomalies
            has_anomaly = conn_rows['is_anomaly'].any() if 'is_anomaly' in conn_rows.columns else False
            anomaly_count = conn_rows['is_anomaly'].sum() if 'is_anomaly' in conn_rows.columns else 0
            
            # Set edge color based on anomaly status
            if has_anomaly:
                edge_color = 'red'
                title = f"{src} → {dst} ({count} connections, {anomaly_count} anomalies)"
            else:
                edge_color = 'gray'
                title = f"{src} → {dst} ({count} connections)"
        else:
            edge_color = 'gray'
            title = f"{src} → {dst} ({count} connections)"
        
        # Add edge to graph with width proportional to count
        width = 1 + min(5, count / 5)  # Cap width at 6
        G.add_edge(src, dst, color=edge_color, width=width, title=title, value=count)
    
    # Create PyVis network
    net = Network(height='600px', width='100%', directed=True, notebook=False)
    
    # Copy nodes and edges from NetworkX graph
    net.from_nx(G)
    
    # Set physics options for better layout
    net.toggle_physics(True)
    net.set_options("""
    {
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -10000,
                "centralGravity": 0.3,
                "springLength": 150,
                "springConstant": 0.01,
                "damping": 0.09,
                "avoidOverlap": 0.1
            },
            "stabilization": {
                "enabled": true,
                "iterations": 1000,
                "updateInterval": 100
            }
        },
        "interaction": {
            "hover": true,
            "multiselect": true,
            "navigationButtons": true,
            "tooltipDelay": 100
        },
        "edges": {
            "arrows": {
                "to": {
                    "enabled": true,
                    "scaleFactor": 0.5
                }
            },
            "smooth": {
                "type": "continuous",
                "forceDirection": "none"
            }
        }
    }
    """)
    
    # FIX: Handle the file access issue
    try:
        # Create temporary file to save HTML
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
            temp_path = tmp.name
            net.save_graph(temp_path)
            
            # Read the HTML file
            with open(temp_path, 'r', encoding='utf-8') as f:
                html = f.read()
            
        # Try to delete the file, but don't fail if we can't
        try:
            os.unlink(temp_path)
        except Exception as e:
            logger.warning(f"Could not delete temporary file {temp_path}: {str(e)}")
            # We'll let the OS clean this up later
            pass
            
        return html
    except Exception as e:
        logger.error(f"Error generating network graph: {str(e)}")
        return None

def plot_geo_map(df: pd.DataFrame, ip_col: str = 'ip_src', 
                lat_col: str = None, lon_col: str = None,
                scores: Optional[np.ndarray] = None, threshold: float = None) -> Optional[go.Figure]:
    """
    Create a geographical map of IP addresses.
    
    Args:
        df: Input dataframe
        ip_col: Column containing IP addresses
        lat_col: Column containing latitude (if None, will try to detect)
        lon_col: Column containing longitude (if None, will try to detect)
        scores: Anomaly scores array
        threshold: Anomaly threshold value
        
    Returns:
        Plotly figure object or None if coordinates not available
    """
    # Check if IP column exists
    if ip_col not in df.columns:
        return None
    
    # If latitude and longitude columns are not provided, try to find them
    if lat_col is None or lon_col is None:
        # Try common column names
        lat_candidates = ['latitude', 'lat', 'ip_latitude', 'src_latitude', 'dst_latitude']
        lon_candidates = ['longitude', 'lon', 'long', 'ip_longitude', 'src_longitude', 'dst_longitude']
        
        for lat in lat_candidates:
            if lat in df.columns:
                lat_col = lat
                break
        
        for lon in lon_candidates:
            if lon in df.columns:
                lon_col = lon
                break
        
        # If still not found, return None
        if lat_col is None or lon_col is None:
            return None
    
    # Check if coordinate columns exist
    if lat_col not in df.columns or lon_col not in df.columns:
        return None
    
    # Create a copy of the dataframe with coordinates
    geo_df = df.copy()
    
    # Add anomaly scores if provided
    if scores is not None and threshold is not None:
        geo_df['anomaly_score'] = scores
        geo_df['is_anomaly'] = get_anomaly_mask(scores, threshold)
        geo_df['status'] = geo_df['is_anomaly'].apply(lambda x: 'Anomaly' if x else 'Normal')
    else:
        geo_df['status'] = 'Normal'
    
    # Drop rows with missing coordinates
    geo_df = geo_df.dropna(subset=[lat_col, lon_col])
    
    # If no valid coordinates, return None
    if len(geo_df) == 0:
        return None
    
    # Create map
    fig = px.scatter_geo(
        geo_df,
        lat=lat_col,
        lon=lon_col,
        color='status',
        color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'},
        hover_name=ip_col,
        hover_data=[ip_col, lat_col, lon_col],
        title="Geographic Distribution of Network Traffic",
        projection="natural earth"
    )
    
    # Add connection lines if multiple IPs are present
    if len(geo_df[ip_col].unique()) > 1 and 'ip_dst' in geo_df.columns:
        # Group by source-destination pairs
        connections = geo_df.groupby([ip_col, 'ip_dst']).size().reset_index(name='count')
        connections = connections.merge(
            geo_df[[ip_col, lat_col, lon_col]].drop_duplicates(),
            on=ip_col, how='left'
        )
        connections = connections.merge(
            geo_df[['ip_dst', lat_col, lon_col]].drop_duplicates().rename(
                columns={'ip_dst': 'ip_dst', lat_col: f'{lat_col}_dst', lon_col: f'{lon_col}_dst'}
            ),
            on='ip_dst', how='left'
        )
        
        # Drop connections with missing coordinates
        connections = connections.dropna(subset=[lat_col, lon_col, f'{lat_col}_dst', f'{lon_col}_dst'])
        
        # If valid connections exist, add lines
        if len(connections) > 0:
            for _, conn in connections.iterrows():
                fig.add_trace(go.Scattergeo(
                    lon=[conn[lon_col], conn[f'{lon_col}_dst']],
                    lat=[conn[lat_col], conn[f'{lat_col}_dst']],
                    mode='lines',
                    line=dict(width=1, color='gray'),
                    opacity=0.5,
                    showlegend=False
                ))
    
    # Update layout
    fig.update_layout(
        legend_title="Status",
        margin=dict(l=0, r=0, b=0, t=30),
        height=600
    )
    
    return fig