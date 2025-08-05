"""
Settings page for the Network Anomaly Detection Platform.
Allows users to configure application settings and parameters.
"""

import streamlit as st

def show_settings():
    """Display the Settings page."""
    
    st.header("Application Settings")
    
    # Initialize settings in session state if not present
    if 'settings' not in st.session_state:
        st.session_state.settings = {
            'anomaly_threshold': 0.8,
            'max_anomalies': 100,
            'auto_refresh': False,
            'dark_mode': False
        }
    
    # Create a form for settings
    with st.form("settings_form"):
        st.subheader("Detection Settings")
        
        # Anomaly detection threshold
        anomaly_threshold = st.slider(
            "Anomaly Detection Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.settings.get('anomaly_threshold', 0.8),
            step=0.05,
            help="Higher values mean fewer but more confident anomalies"
        )
        
        # Maximum number of anomalies to display
        max_anomalies = st.number_input(
            "Maximum Anomalies to Display",
            min_value=10,
            max_value=1000,
            value=st.session_state.settings.get('max_anomalies', 100),
            step=10,
            help="Limit the number of anomalies shown in results"
        )
        
        st.subheader("Interface Settings")
        
        # Auto-refresh option
        auto_refresh = st.checkbox(
            "Auto-refresh Data",
            value=st.session_state.settings.get('auto_refresh', False),
            help="Automatically refresh data files list periodically"
        )
        
        # Dark mode toggle
        dark_mode = st.checkbox(
            "Dark Mode",
            value=st.session_state.settings.get('dark_mode', False),
            help="Enable dark mode interface"
        )
        
        # Advanced settings expander
        with st.expander("Advanced Settings"):
            # Cache timeout
            cache_timeout = st.number_input(
                "Cache Timeout (minutes)",
                min_value=5,
                max_value=120,
                value=st.session_state.settings.get('cache_timeout', 30),
                step=5,
                help="Time before cached data is refreshed"
            )
            
            # Debug mode
            debug_mode = st.checkbox(
                "Debug Mode",
                value=st.session_state.settings.get('debug_mode', False),
                help="Enable additional logging and debugging information"
            )
        
        # Submit button
        submitted = st.form_submit_button("Save Settings")
        
        if submitted:
            # Update session state with new settings
            st.session_state.settings = {
                'anomaly_threshold': anomaly_threshold,
                'max_anomalies': max_anomalies,
                'auto_refresh': auto_refresh,
                'dark_mode': dark_mode,
                'cache_timeout': cache_timeout,
                'debug_mode': debug_mode if 'debug_mode' in locals() else False
            }
            
            st.success("Settings saved successfully!")
    
    # Display current settings
    st.subheader("Current Settings")
    st.json(st.session_state.settings)
    
    # Reset button (outside the form)
    if st.button("Reset to Defaults"):
        st.session_state.settings = {
            'anomaly_threshold': 0.8,
            'max_anomalies': 100,
            'auto_refresh': False,
            'dark_mode': False,
            'cache_timeout': 30,
            'debug_mode': False
        }
        st.success("Settings reset to defaults!")
        st.rerun()  # Rerun the app to apply changes