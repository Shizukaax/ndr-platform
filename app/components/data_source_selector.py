"""
Data Source Selector Component
Provides a unified interface for switching between data sources
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Optional

from core.data_manager import get_data_manager

def show_data_source_selector():
    """Display data source selector and status information"""
    
    data_manager = get_data_manager()
    current_info = data_manager.get_data_source_info()
    available_sources = current_info['available_sources']
    
    st.markdown("### 📊 Data Source Control")
    
    # Current status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        source_type = current_info['source_type']
        if source_type == 'auto_loaded':
            st.success("📁 **Auto-loaded Data**")
        elif source_type == 'realtime':
            st.success("🔴 **Real-time Data**")
        elif source_type == 'manual':
            st.success("📤 **Manual Upload**")
        else:
            st.warning("⚠️ **No Data Loaded**")
    
    with col2:
        record_count = current_info['record_count']
        st.metric("Records", f"{record_count:,}" if record_count > 0 else "0")
    
    with col3:
        load_time = current_info.get('load_time')
        if load_time:
            time_str = load_time.strftime("%H:%M:%S")
            st.write(f"⏰ **Last Updated:** {time_str}")
        else:
            st.write("⏰ **Not loaded**")
    
    # Data source switcher
    st.markdown("#### 🔄 Switch Data Source")
    
    source_options = []
    source_mapping = {}
    
    for source in available_sources:
        if source['available']:
            display_name = f"{source['name']} ({source['description']})"
            source_options.append(display_name)
            source_mapping[display_name] = source['type']
    
    if source_options:
        # Current selection
        current_display = None
        for display_name, source_type in source_mapping.items():
            if source_type == current_info['source_type']:
                current_display = display_name
                break
        
        # Source selector
        selected_display = st.selectbox(
            "Choose data source:",
            options=source_options,
            index=source_options.index(current_display) if current_display in source_options else 0,
            help="Select which data source to use for analysis"
        )
        
        selected_type = source_mapping[selected_display]
        
        # Switch button
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔄 Switch", type="primary"):
                with st.spinner(f"Switching to {selected_display}..."):
                    success = data_manager.switch_data_source(selected_type)
                    if success:
                        st.success(f"✅ Switched to {selected_display}")
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to switch to {selected_display}")
        
        with col2:
            if st.button("🔄 Refresh Current Source"):
                with st.spinner("Refreshing data..."):
                    success = data_manager.switch_data_source(current_info['source_type'])
                    if success:
                        st.success("✅ Data refreshed")
                        st.rerun()
                    else:
                        st.error("❌ Failed to refresh data")
    
    else:
        st.warning("⚠️ No data sources available. Please check your configuration.")
    
    # Detailed information
    with st.expander("📋 Data Source Details"):
        if current_info['source_type'] == 'auto_loaded':
            st.write(f"**Directory:** `{current_info.get('source_directory', 'Unknown')}`")
            source_files = current_info.get('source_files', [])
            if source_files:
                st.write(f"**Files loaded:** {len(source_files)}")
                with st.expander(f"View {len(source_files)} loaded files"):
                    for file_path in source_files[:20]:  # Show first 20
                        st.write(f"- {file_path}")
                    if len(source_files) > 20:
                        st.write(f"... and {len(source_files) - 20} more files")
        
        elif current_info['source_type'] == 'realtime':
            st.write(f"**Source Directory:** `{current_info.get('source_directory', 'Unknown')}`")
            monitoring_active = current_info.get('monitoring_active', False)
            st.write(f"**Monitoring Status:** {'🟢 Active' if monitoring_active else '🔴 Inactive'}")
        
        elif current_info['source_type'] == 'manual':
            source_files = current_info.get('source_files', [])
            if source_files:
                st.write(f"**Uploaded files:** {len(source_files)}")
                for file_path in source_files:
                    st.write(f"- {file_path}")
        
        # Show available sources
        st.write("**Available Sources:**")
        for source in available_sources:
            status = "✅" if source['available'] else "❌"
            st.write(f"{status} {source['name']}: {source['description']}")

def show_compact_data_status():
    """Show a compact data status for sidebar or top of pages"""
    
    data_manager = get_data_manager()
    current_info = data_manager.get_data_source_info()
    
    source_type = current_info['source_type']
    record_count = current_info['record_count']
    
    if source_type == 'auto_loaded':
        st.success(f"📁 Auto-loaded: {record_count:,} records")
    elif source_type == 'realtime':
        st.success(f"🔴 Real-time: {record_count:,} records")
    elif source_type == 'manual':
        st.success(f"📤 Manual: {record_count:,} records")
    else:
        st.warning("⚠️ No data loaded")
    
    return current_info

def ensure_data_available() -> bool:
    """Ensure data is available, show selector if not"""
    
    if st.session_state.get('combined_data') is None:
        st.warning("⚠️ No data is currently loaded.")
        
        # Show data source selector
        show_data_source_selector()
        
        # Check if data is now available
        if st.session_state.get('combined_data') is None:
            st.info("👆 Please select a data source above to proceed.")
            return False
    
    return True

def get_current_data() -> Optional[pd.DataFrame]:
    """Get the current data from session state"""
    return st.session_state.get('combined_data')

def get_current_data_source() -> str:
    """Get the current data source type"""
    return st.session_state.get('data_source', 'none')
