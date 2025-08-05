"""
Search and filtering utilities for the Network Anomaly Detection Platform.
Provides UI elements and backend logic for searching and filtering data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta

def create_filter_ui(df, key_prefix="filter"):
    """
    Create a UI for filtering a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to filter
        key_prefix (str): Prefix for session state keys
    
    Returns:
        tuple: (filtered_df, filter_description)
    """
    st.subheader("Filter Data")
    
    # Create filter UI
    filter_container = st.container()
    
    with filter_container:
        # Initialize filters
        active_filters = {}
        filter_description = []
        
        # Text search across all columns
        col1, col2 = st.columns([3, 1])
        with col1:
            search_text = st.text_input("Search across all columns", key=f"{key_prefix}_search")
            if search_text:
                filter_description.append(f"Text: '{search_text}'")
        
        with col2:
            case_sensitive = st.checkbox("Case sensitive", key=f"{key_prefix}_case")
        
        # Advanced filters
        with st.expander("Advanced Filters"):
            # Get column types
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
            categorical_cols = [col for col in df.columns if col not in numeric_cols and col not in datetime_cols
                                and df[col].nunique() < 100]  # Limit to columns with fewer unique values
            
            # Select columns to filter
            selected_cols = st.multiselect(
                "Select columns to filter",
                options=numeric_cols + datetime_cols + categorical_cols,
                default=[],
                key=f"{key_prefix}_cols"
            )
            
            # Create filters for selected columns
            for col in selected_cols:
                st.markdown(f"**Filter: {col}**")
                
                if col in numeric_cols:
                    # Numeric filter
                    min_val = float(df[col].min())
                    max_val = float(df[col].max())
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        lower = st.number_input(
                            f"Minimum {col}",
                            min_value=min_val,
                            max_value=max_val,
                            value=min_val,
                            key=f"{key_prefix}_{col}_min"
                        )
                    with col2:
                        upper = st.number_input(
                            f"Maximum {col}",
                            min_value=min_val,
                            max_value=max_val,
                            value=max_val,
                            key=f"{key_prefix}_{col}_max"
                        )
                    
                    if lower > min_val or upper < max_val:
                        active_filters[col] = (lower, upper)
                        filter_description.append(f"{col}: {lower} to {upper}")
                
                elif col in datetime_cols:
                    # Datetime filter
                    min_date = df[col].min()
                    max_date = df[col].max()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        start_date = st.date_input(
                            f"Start date for {col}",
                            min_value=min_date,
                            max_value=max_date,
                            value=min_date,
                            key=f"{key_prefix}_{col}_start"
                        )
                    with col2:
                        end_date = st.date_input(
                            f"End date for {col}",
                            min_value=min_date,
                            max_value=max_date,
                            value=max_date,
                            key=f"{key_prefix}_{col}_end"
                        )
                    
                    # Convert to datetime with time component
                    start_datetime = datetime.combine(start_date, datetime.min.time())
                    end_datetime = datetime.combine(end_date, datetime.max.time())
                    
                    if start_datetime > min_date or end_datetime < max_date:
                        active_filters[col] = (start_datetime, end_datetime)
                        filter_description.append(f"{col}: {start_date} to {end_date}")
                
                else:
                    # Categorical filter
                    unique_values = df[col].dropna().unique()
                    selected_values = st.multiselect(
                        f"Select values for {col}",
                        options=sorted(unique_values),
                        default=list(unique_values),
                        key=f"{key_prefix}_{col}_values"
                    )
                    
                    if len(selected_values) < len(unique_values):
                        active_filters[col] = selected_values
                        filter_description.append(f"{col}: {', '.join(map(str, selected_values))}")
    
    # Apply filters
    filtered_df = df.copy()
    
    # Apply text search
    if search_text:
        mask = pd.Series(False, index=filtered_df.index)
        for col in filtered_df.columns:
            if case_sensitive:
                col_mask = filtered_df[col].astype(str).str.contains(search_text, regex=True, na=False)
            else:
                col_mask = filtered_df[col].astype(str).str.contains(search_text, regex=True, case=False, na=False)
            mask = mask | col_mask
        filtered_df = filtered_df.loc[mask]
    
    # Apply column filters
    for col, filter_val in active_filters.items():
        if col in numeric_cols:
            # Numeric range filter
            lower, upper = filter_val
            filtered_df = filtered_df[(filtered_df[col] >= lower) & (filtered_df[col] <= upper)]
        
        elif col in datetime_cols:
            # Datetime range filter
            start_datetime, end_datetime = filter_val
            filtered_df = filtered_df[(filtered_df[col] >= start_datetime) & (filtered_df[col] <= end_datetime)]
        
        else:
            # Categorical filter
            filtered_df = filtered_df[filtered_df[col].isin(filter_val)]
    
    # Create filter summary
    filter_summary = ", ".join(filter_description) if filter_description else "No filters applied"
    
    return filtered_df, filter_summary

def create_ip_filter(df, src_col="ip_src", dst_col="ip_dst", key_prefix="ip_filter"):
    """
    Create a specialized filter for IP addresses.
    
    Args:
        df (pd.DataFrame): DataFrame to filter
        src_col (str): Column name for source IP
        dst_col (str): Column name for destination IP
        key_prefix (str): Prefix for session state keys
    
    Returns:
        tuple: (filtered_df, filter_description)
    """
    # Check if IP columns exist
    if src_col not in df.columns and dst_col not in df.columns:
        return df, "IP address columns not found"
    
    st.subheader("IP Address Filtering")
    
    # Get unique IP addresses
    src_ips = df[src_col].dropna().unique() if src_col in df.columns else []
    dst_ips = df[dst_col].dropna().unique() if dst_col in df.columns else []
    all_ips = sorted(set(list(src_ips) + list(dst_ips)))
    
    # Create filter UI
    filter_type = st.radio(
        "Filter type",
        options=["IP Selection", "IP Range", "IP Pattern"],
        key=f"{key_prefix}_type"
    )
    
    filtered_df = df.copy()
    filter_description = ""
    
    if filter_type == "IP Selection":
        # Filter by selecting specific IPs
        selected_ips = st.multiselect(
            "Select IP addresses",
            options=all_ips,
            key=f"{key_prefix}_select"
        )
        
        if selected_ips:
            mask = pd.Series(False, index=filtered_df.index)
            
            if src_col in df.columns:
                mask = mask | filtered_df[src_col].isin(selected_ips)
                
            if dst_col in df.columns:
                mask = mask | filtered_df[dst_col].isin(selected_ips)
                
            filtered_df = filtered_df.loc[mask]
            filter_description = f"Selected IPs: {', '.join(selected_ips)}"
    
    elif filter_type == "IP Range":
        # Filter by IP range
        st.markdown("Enter IP range (CIDR notation)")
        ip_range = st.text_input("IP range (e.g. 192.168.1.0/24)", key=f"{key_prefix}_range")
        
        if ip_range:
            try:
                # Parse CIDR
                import ipaddress
                network = ipaddress.ip_network(ip_range, strict=False)
                
                # Function to check if IP is in network
                def ip_in_network(ip):
                    try:
                        return ipaddress.ip_address(ip) in network
                    except:
                        return False
                
                # Apply filter
                mask = pd.Series(False, index=filtered_df.index)
                
                if src_col in df.columns:
                    mask = mask | filtered_df[src_col].apply(ip_in_network)
                    
                if dst_col in df.columns:
                    mask = mask | filtered_df[dst_col].apply(ip_in_network)
                    
                filtered_df = filtered_df.loc[mask]
                filter_description = f"IP range: {ip_range}"
                
            except Exception as e:
                st.error(f"Invalid IP range: {str(e)}")
    
    elif filter_type == "IP Pattern":
        # Filter by regex pattern
        ip_pattern = st.text_input("IP pattern (e.g. 192.168.*)", key=f"{key_prefix}_pattern")
        
        if ip_pattern:
            try:
                # Convert wildcard to regex
                regex_pattern = ip_pattern.replace('.', '\.').replace('*', '.*')
                
                # Apply filter
                mask = pd.Series(False, index=filtered_df.index)
                
                if src_col in df.columns:
                    mask = mask | filtered_df[src_col].str.match(regex_pattern, na=False)
                    
                if dst_col in df.columns:
                    mask = mask | filtered_df[dst_col].str.match(regex_pattern, na=False)
                    
                filtered_df = filtered_df.loc[mask]
                filter_description = f"IP pattern: {ip_pattern}"
                
            except Exception as e:
                st.error(f"Invalid pattern: {str(e)}")
    
    return filtered_df, filter_description

def create_time_filter(df, time_col="timestamp", key_prefix="time_filter"):
    """
    Create a specialized filter for timestamps.
    
    Args:
        df (pd.DataFrame): DataFrame to filter
        time_col (str): Column name for timestamp
        key_prefix (str): Prefix for session state keys
    
    Returns:
        tuple: (filtered_df, filter_description)
    """
    # Check if timestamp column exists
    if time_col not in df.columns:
        return df, "Timestamp column not found"
    
    # Ensure timestamp column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        try:
            df[time_col] = pd.to_datetime(df[time_col])
        except:
            return df, "Could not convert timestamp column to datetime"
    
    st.subheader("Time Filtering")
    
    # Get min and max timestamps
    min_time = df[time_col].min()
    max_time = df[time_col].max()
    
    # Create filter UI
    filter_type = st.radio(
        "Time filter type",
        options=["Time Range", "Relative Time", "Time of Day"],
        key=f"{key_prefix}_type"
    )
    
    filtered_df = df.copy()
    filter_description = ""
    
    if filter_type == "Time Range":
        # Filter by specific time range
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start date",
                value=min_time.date(),
                min_value=min_time.date(),
                max_value=max_time.date(),
                key=f"{key_prefix}_start_date"
            )
            
            start_time = st.time_input(
                "Start time",
                value=datetime.min.time(),
                key=f"{key_prefix}_start_time"
            )
        
        with col2:
            end_date = st.date_input(
                "End date",
                value=max_time.date(),
                min_value=min_time.date(),
                max_value=max_time.date(),
                key=f"{key_prefix}_end_date"
            )
            
            end_time = st.time_input(
                "End time",
                value=datetime.max.time(),
                key=f"{key_prefix}_end_time"
            )
        
        # Combine date and time
        start_datetime = datetime.combine(start_date, start_time)
        end_datetime = datetime.combine(end_date, end_time)
        
        # Apply filter
        filtered_df = filtered_df[(filtered_df[time_col] >= start_datetime) & 
                                  (filtered_df[time_col] <= end_datetime)]
        
        filter_description = f"Time range: {start_datetime} to {end_datetime}"
    
    elif filter_type == "Relative Time":
        # Filter by relative time (e.g. last X hours)
        time_unit = st.selectbox(
            "Time unit",
            options=["minutes", "hours", "days"],
            index=1,
            key=f"{key_prefix}_unit"
        )
        
        time_value = st.number_input(
            f"Last N {time_unit}",
            min_value=1,
            max_value=1000,
            value=24 if time_unit == "hours" else 7 if time_unit == "days" else 60,
            key=f"{key_prefix}_value"
        )
        
        # Calculate relative time
        if time_unit == "minutes":
            cutoff_time = max_time - timedelta(minutes=time_value)
        elif time_unit == "hours":
            cutoff_time = max_time - timedelta(hours=time_value)
        else:  # days
            cutoff_time = max_time - timedelta(days=time_value)
        
        # Apply filter
        filtered_df = filtered_df[filtered_df[time_col] >= cutoff_time]
        
        filter_description = f"Last {time_value} {time_unit}"
    
    elif filter_type == "Time of Day":
        # Filter by time of day (e.g. only between 9 AM and 5 PM)
        col1, col2 = st.columns(2)
        
        with col1:
            start_hour = st.slider(
                "Start hour",
                min_value=0,
                max_value=23,
                value=9,
                key=f"{key_prefix}_start_hour"
            )
        
        with col2:
            end_hour = st.slider(
                "End hour",
                min_value=0,
                max_value=23,
                value=17,
                key=f"{key_prefix}_end_hour"
            )
        
        # Apply filter
        filtered_df = filtered_df[(filtered_df[time_col].dt.hour >= start_hour) & 
                                  (filtered_df[time_col].dt.hour <= end_hour)]
        
        filter_description = f"Time of day: {start_hour}:00 to {end_hour}:59"
    
    return filtered_df, filter_description

def create_protocol_filter(df, protocol_col="_ws_col_Protocol", key_prefix="protocol_filter"):
    """
    Create a specialized filter for network protocols.
    
    Args:
        df (pd.DataFrame): DataFrame to filter
        protocol_col (str): Column name for protocol
        key_prefix (str): Prefix for session state keys
    
    Returns:
        tuple: (filtered_df, filter_description)
    """
    # Check if protocol column exists
    if protocol_col not in df.columns:
        return df, "Protocol column not found"
    
    st.subheader("Protocol Filtering")
    
    # Get unique protocols
    protocols = sorted(df[protocol_col].dropna().unique())
    
    # Create filter UI
    selected_protocols = st.multiselect(
        "Select protocols",
        options=protocols,
        default=protocols,
        key=f"{key_prefix}_select"
    )
    
    # Apply filter
    if selected_protocols and len(selected_protocols) < len(protocols):
        filtered_df = df[df[protocol_col].isin(selected_protocols)]
        filter_description = f"Protocols: {', '.join(selected_protocols)}"
    else:
        filtered_df = df.copy()
        filter_description = "All protocols"
    
    return filtered_df, filter_description