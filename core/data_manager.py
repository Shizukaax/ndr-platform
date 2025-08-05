"""
Data Manager for NDR Solution
Handles auto-loading, real-time data, and data source switching
"""

import os
import pandas as pd
import streamlit as st
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from core.config_loader import load_config
from app.state.session_state import process_json_data, load_json_file

logger = logging.getLogger("data_manager")

class DataManager:
    """Manages data sources for the NDR platform"""
    
    def __init__(self):
        self.config = load_config()
        # Use simplified single data source configuration
        self.data_config = self.config.get('data_source', {})
        self.monitoring_config = self.config.get('monitoring', {})
        
    def auto_load_data(self) -> bool:
        """Auto-load data from configured directory"""
        try:
            directory = self.data_config.get('directory')
            if not directory:
                logger.info("No data directory configured")
                return False
            
            file_pattern = self.data_config.get('file_pattern', '*.json')
            max_files = self.data_config.get('max_files', 100)
            recursive = self.data_config.get('recursive', True)
            
            logger.info(f"Auto-loading data from {directory}")
            
            # Find JSON files
            data_path = Path(directory)
            if not data_path.exists():
                logger.warning(f"Data directory does not exist: {directory}")
                return False
            
            # Get files based on pattern
            if recursive:
                json_files = list(data_path.rglob(file_pattern))
            else:
                json_files = list(data_path.glob(file_pattern))
            
            if not json_files:
                logger.info(f"No files found matching pattern {file_pattern} in {directory}")
                return False
            
            # Limit number of files
            json_files = json_files[:max_files]
            logger.info(f"Found {len(json_files)} files to process")
            
            # Load and combine data
            combined_data = []
            loaded_files = []
            
            for file_path in json_files:
                try:
                    file_data = load_json_file(str(file_path))
                    if file_data is not None and not file_data.empty:
                        # Add metadata
                        file_data['source_file'] = file_path.name
                        file_data['load_time'] = datetime.now()
                        file_data['data_source'] = 'auto_loaded'
                        
                        combined_data.append(file_data)
                        loaded_files.append(str(file_path))
                        
                        if len(combined_data) >= max_files:
                            break
                            
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {str(e)}")
                    continue
            
            if combined_data:
                # Combine all data
                final_df = pd.concat(combined_data, ignore_index=True)
                
                # Store in session state
                st.session_state.combined_data = final_df
                st.session_state.loaded_files = loaded_files
                st.session_state.data_source = 'auto_loaded'
                st.session_state.data_load_time = datetime.now()
                
                logger.info(f"Auto-loaded {len(final_df)} records from {len(loaded_files)} files")
                return True
            else:
                logger.warning("No valid data found in any files")
                return False
                
        except Exception as e:
            logger.error(f"Error in auto-load: {str(e)}")
            return False
    
    def get_realtime_data(self) -> Optional[pd.DataFrame]:
        """Get current real-time data from file monitor"""
        try:
            if 'file_monitor' not in st.session_state:
                return None
            
            file_monitor = st.session_state.file_monitor
            if not hasattr(file_monitor, 'recent_data') or not file_monitor.recent_data:
                return None
            
            # Convert recent data to DataFrame
            realtime_records = []
            for data_point in file_monitor.recent_data:
                record = data_point['data'].copy()
                record['source_file'] = data_point['source_file']
                record['ingestion_time'] = data_point['timestamp']
                record['data_source'] = 'realtime'
                realtime_records.append(record)
            
            if realtime_records:
                return pd.DataFrame(realtime_records)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting real-time data: {str(e)}")
            return None
    
    def switch_data_source(self, source_type: str) -> bool:
        """Switch between data sources"""
        try:
            if source_type == 'auto_loaded':
                return self.auto_load_data()
            
            elif source_type == 'realtime':
                realtime_data = self.get_realtime_data()
                if realtime_data is not None:
                    st.session_state.combined_data = realtime_data
                    st.session_state.data_source = 'realtime'
                    st.session_state.data_load_time = datetime.now()
                    logger.info(f"Switched to real-time data: {len(realtime_data)} records")
                    return True
                else:
                    logger.warning("No real-time data available")
                    return False
            
            elif source_type == 'manual':
                # Keep existing manually uploaded data
                if st.session_state.get('combined_data') is not None:
                    st.session_state.data_source = 'manual'
                    return True
                else:
                    logger.warning("No manually uploaded data available")
                    return False
            
            else:
                logger.error(f"Unknown data source type: {source_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error switching data source: {str(e)}")
            return False
    
    def get_data_source_info(self) -> Dict:
        """Get information about current data source"""
        try:
            current_source = st.session_state.get('data_source', 'none')
            data = st.session_state.get('combined_data')
            load_time = st.session_state.get('data_load_time')
            
            info = {
                'source_type': current_source,
                'record_count': len(data) if data is not None else 0,
                'load_time': load_time,
                'available_sources': self._get_available_sources()
            }
            
            if current_source == 'auto_loaded':
                info['source_files'] = st.session_state.get('loaded_files', [])
                info['source_directory'] = self.data_config.get('directory', 'Unknown')
            
            elif current_source == 'realtime':
                info['source_directory'] = st.session_state.get('file_monitor', {}).watch_directory if st.session_state.get('file_monitor') else 'Unknown'
                info['monitoring_active'] = st.session_state.get('file_monitor', {}).is_monitoring if st.session_state.get('file_monitor') else False
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting data source info: {str(e)}")
            return {'source_type': 'error', 'record_count': 0}
    
    def _get_available_sources(self) -> List[Dict]:
        """Get list of available data sources"""
        sources = []
        
        # Auto-load source
        if self.data_config.get('directory'):
            auto_dir = self.data_config.get('directory', 'data')
            sources.append({
                'type': 'auto_loaded',
                'name': '📁 Auto-loaded Data',
                'description': f'Automatically loaded from {auto_dir}',
                'available': Path(auto_dir).exists()
            })
        
        # Real-time source
        sources.append({
            'type': 'realtime',
            'name': '🔴 Real-time Data',
            'description': 'Live data from file monitoring',
            'available': st.session_state.get('file_monitor') is not None
        })
        
        return sources
    
    def initialize_default_data(self):
        """Initialize with default data source on startup"""
        try:
            # Check if data is already loaded
            if st.session_state.get('combined_data') is not None:
                # Only log this once per session
                if not hasattr(self, '_data_loaded_logged'):
                    logger.info("Data already loaded, skipping initialization")
                    self._data_loaded_logged = True
                return
            
            # Try auto-load first
            if self.data_config.get('directory'):
                if self.auto_load_data():
                    logger.info("Successfully initialized with auto-loaded data")
                    return
            
            # If auto-load fails, check for real-time data
            realtime_data = self.get_realtime_data()
            if realtime_data is not None:
                st.session_state.combined_data = realtime_data
                st.session_state.data_source = 'realtime'
                st.session_state.data_load_time = datetime.now()
                logger.info("Successfully initialized with real-time data")
                return
            
            logger.info("No data sources available, waiting for real-time data")
            
        except Exception as e:
            logger.error(f"Error initializing default data: {str(e)}")

# Global data manager instance
data_manager = DataManager()

def get_data_manager() -> DataManager:
    """Get the global data manager instance"""
    return data_manager
