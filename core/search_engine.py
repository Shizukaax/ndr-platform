"""
Search engine module for the Network Anomaly Detection Platform.
Provides advanced search capabilities across network data and anomalies.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import ipaddress

class SearchEngine:
    """Search engine for finding patterns in network data and anomalies."""
    
    def __init__(self):
        """Initialize the search engine."""
        self.search_cache = {}
        self.max_cache_size = 100
    
    def search(self, df, query, case_sensitive=False, regex=False, limit=None):
        """
        Search for a query across all columns in a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame to search within
            query (str): Search query
            case_sensitive (bool): Whether to perform case-sensitive search
            regex (bool): Whether to interpret query as a regular expression
            limit (int, optional): Maximum number of results to return
        
        Returns:
            pd.DataFrame: Filtered DataFrame with search results
        """
        if df is None or df.empty or not query:
            return df
        
        # Generate cache key
        cache_key = f"{hash(tuple(df.columns))}-{query}-{case_sensitive}-{regex}-{limit}"
        
        # Check if results are cached
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
        
        # Initialize mask as all False
        mask = pd.Series(False, index=df.index)
        
        # Search across all columns
        for col in df.columns:
            # Convert column to string for searching
            col_str = df[col].astype(str)
            
            if regex:
                # Regular expression search
                try:
                    if case_sensitive:
                        col_mask = col_str.str.contains(query, regex=True, na=False)
                    else:
                        col_mask = col_str.str.contains(query, regex=True, case=False, na=False)
                    mask = mask | col_mask
                except re.error:
                    # Invalid regex, treat as literal string
                    if case_sensitive:
                        col_mask = col_str.str.contains(re.escape(query), regex=True, na=False)
                    else:
                        col_mask = col_str.str.contains(re.escape(query), regex=True, case=False, na=False)
                    mask = mask | col_mask
            else:
                # Plain text search
                if case_sensitive:
                    col_mask = col_str.str.contains(re.escape(query), regex=True, na=False)
                else:
                    col_mask = col_str.str.contains(re.escape(query), regex=True, case=False, na=False)
                mask = mask | col_mask
        
        # Apply filter
        results = df[mask]
        
        # Apply limit if specified
        if limit is not None and limit > 0:
            results = results.head(limit)
        
        # Cache results
        if len(self.search_cache) >= self.max_cache_size:
            # Remove oldest item if cache is full
            oldest_key = next(iter(self.search_cache))
            del self.search_cache[oldest_key]
        
        self.search_cache[cache_key] = results
        
        return results
    
    def advanced_search(self, df, filters, limit=None):
        """
        Perform advanced search with multiple filters.
        
        Args:
            df (pd.DataFrame): DataFrame to search within
            filters (dict): Dictionary of filters with format {column: {operator: value}}
            limit (int, optional): Maximum number of results to return
        
        Returns:
            pd.DataFrame: Filtered DataFrame with search results
        """
        if df is None or df.empty or not filters:
            return df
        
        # Generate cache key
        cache_key = f"adv-{hash(tuple(df.columns))}-{hash(str(filters))}-{limit}"
        
        # Check if results are cached
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
        
        # Start with all rows
        filtered_df = df.copy()
        
        # Apply each filter
        for column, conditions in filters.items():
            if column not in df.columns:
                continue
                
            for operator, value in conditions.items():
                filtered_df = self._apply_filter(filtered_df, column, operator, value)
        
        # Apply limit if specified
        if limit is not None and limit > 0:
            filtered_df = filtered_df.head(limit)
        
        # Cache results
        if len(self.search_cache) >= self.max_cache_size:
            # Remove oldest item if cache is full
            oldest_key = next(iter(self.search_cache))
            del self.search_cache[oldest_key]
        
        self.search_cache[cache_key] = filtered_df
        
        return filtered_df
    
    def _apply_filter(self, df, column, operator, value):
        """
        Apply a single filter to a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame to filter
            column (str): Column name to filter on
            operator (str): Operator to apply (e.g., 'eq', 'gt', 'contains')
            value: Value to compare against
        
        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        # Get column data
        col_data = df[column]
        
        # Apply operator
        if operator == 'eq' or operator == '==':
            return df[col_data == value]
        elif operator == 'neq' or operator == '!=':
            return df[col_data != value]
        elif operator == 'gt' or operator == '>':
            return df[col_data > value]
        elif operator == 'lt' or operator == '<':
            return df[col_data < value]
        elif operator == 'gte' or operator == '>=':
            return df[col_data >= value]
        elif operator == 'lte' or operator == '<=':
            return df[col_data <= value]
        elif operator == 'contains':
            return df[col_data.astype(str).str.contains(str(value), case=False, na=False)]
        elif operator == 'not_contains':
            return df[~col_data.astype(str).str.contains(str(value), case=False, na=False)]
        elif operator == 'startswith':
            return df[col_data.astype(str).str.startswith(str(value), na=False)]
        elif operator == 'endswith':
            return df[col_data.astype(str).str.endswith(str(value), na=False)]
        elif operator == 'between':
            if isinstance(value, list) and len(value) == 2:
                return df[(col_data >= value[0]) & (col_data <= value[1])]
            return df
        elif operator == 'in':
            if isinstance(value, list):
                return df[col_data.isin(value)]
            return df
        elif operator == 'not_in':
            if isinstance(value, list):
                return df[~col_data.isin(value)]
            return df
        elif operator == 'is_null':
            return df[col_data.isna()]
        elif operator == 'is_not_null':
            return df[~col_data.isna()]
        elif operator == 'ip_in_subnet':
            # Check if IP is in subnet
            try:
                subnet = ipaddress.ip_network(value, strict=False)
                mask = col_data.apply(lambda ip: self._ip_in_subnet(ip, subnet))
                return df[mask]
            except:
                return df
        else:
            # Unknown operator
            return df
    
    def _ip_in_subnet(self, ip, subnet):
        """Check if an IP address is in a subnet."""
        try:
            return ipaddress.ip_address(ip) in subnet
        except:
            return False
    
    def search_time_range(self, df, time_column, start_time, end_time):
        """
        Search for records within a specific time range.
        
        Args:
            df (pd.DataFrame): DataFrame to search within
            time_column (str): Name of the column containing timestamps
            start_time: Start timestamp (datetime or string)
            end_time: End timestamp (datetime or string)
        
        Returns:
            pd.DataFrame: Filtered DataFrame with records in time range
        """
        if df is None or df.empty or time_column not in df.columns:
            return df
        
        # Convert column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
            try:
                time_data = pd.to_datetime(df[time_column])
            except:
                return df  # Return original if conversion fails
        else:
            time_data = df[time_column]
        
        # Convert input times to datetime if needed
        if not isinstance(start_time, (datetime, pd.Timestamp)):
            try:
                start_time = pd.to_datetime(start_time)
            except:
                return df
        
        if not isinstance(end_time, (datetime, pd.Timestamp)):
            try:
                end_time = pd.to_datetime(end_time)
            except:
                return df
        
        # Filter by time range
        return df[(time_data >= start_time) & (time_data <= end_time)]
    
    def search_ip_traffic(self, df, src_ip=None, dst_ip=None, src_col='ip_src', dst_col='ip_dst'):
        """
        Search for traffic involving specific IP addresses.
        
        Args:
            df (pd.DataFrame): DataFrame to search within
            src_ip (str, optional): Source IP address
            dst_ip (str, optional): Destination IP address
            src_col (str): Name of the source IP column
            dst_col (str): Name of the destination IP column
        
        Returns:
            pd.DataFrame: Filtered DataFrame with matching traffic
        """
        if df is None or df.empty:
            return df
        
        # Check if IP columns exist
        src_exists = src_col in df.columns
        dst_exists = dst_col in df.columns
        
        if not src_exists and not dst_exists:
            return df
        
        # Initialize mask as all False
        mask = pd.Series(False, index=df.index)
        
        # Filter by source IP
        if src_ip and src_exists:
            src_mask = df[src_col] == src_ip
            mask = mask | src_mask
        
        # Filter by destination IP
        if dst_ip and dst_exists:
            dst_mask = df[dst_col] == dst_ip
            mask = mask | dst_mask
        
        # If no IP filters specified, return original DataFrame
        if not src_ip and not dst_ip:
            return df
        
        # Apply filter
        return df[mask]
    
    def search_top_talkers(self, df, n=10, src_col='ip_src', dst_col='ip_dst'):
        """
        Find top talkers (most active IPs) in the dataset.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            n (int): Number of top talkers to return
            src_col (str): Name of the source IP column
            dst_col (str): Name of the destination IP column
        
        Returns:
            pd.DataFrame: DataFrame with top talkers and counts
        """
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Check if IP columns exist
        src_exists = src_col in df.columns
        dst_exists = dst_col in df.columns
        
        if not src_exists and not dst_exists:
            return pd.DataFrame()
        
        # Count occurrences of each IP
        ip_counts = {}
        
        if src_exists:
            src_counts = df[src_col].value_counts()
            for ip, count in src_counts.items():
                ip_counts[ip] = ip_counts.get(ip, 0) + count
        
        if dst_exists:
            dst_counts = df[dst_col].value_counts()
            for ip, count in dst_counts.items():
                ip_counts[ip] = ip_counts.get(ip, 0) + count
        
        # Convert to DataFrame
        results = pd.DataFrame({
            'IP Address': list(ip_counts.keys()),
            'Count': list(ip_counts.values())
        }).sort_values('Count', ascending=False).head(n)
        
        return results
    
    def clear_cache(self):
        """Clear the search cache."""
        self.search_cache = {}