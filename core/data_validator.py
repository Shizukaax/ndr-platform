"""
Data validation service for the Network Anomaly Detection Platform.
Provides data quality checks and validation utilities.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
import streamlit as st

logger = logging.getLogger("streamlit_app")

class DataValidationService:
    """Service for validating and checking data quality."""
    
    def __init__(self):
        """Initialize data validation service."""
        self.required_columns = {
            'network_basic': ['ip_src', 'ip_dst', 'proto'],
            'network_extended': ['ip.src', 'ip.dst', 'ip.proto'],
            'arkime_format': ['srcIp', 'dstIp', 'protocol']
        }
    
    def validate_data_for_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data for anomaly analysis.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dict with validation results
        """
        results = {
            'is_valid': False,
            'errors': [],
            'warnings': [],
            'recommendations': [],
            'data_summary': {},
            'column_analysis': {}
        }
        
        try:
            # Basic checks
            if data is None or data.empty:
                results['errors'].append("Data is empty or None")
                return results
            
            # Data summary
            results['data_summary'] = {
                'total_rows': len(data),
                'total_columns': len(data.columns),
                'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
                'null_percentage': (data.isnull().sum().sum() / data.size) * 100
            }
            
            # Column analysis
            results['column_analysis'] = self._analyze_columns(data)
            
            # Check for minimum requirements
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                results['errors'].append("At least 2 numeric columns required for anomaly detection")
            
            # Check for extreme null values
            null_percentage = results['data_summary']['null_percentage']
            if null_percentage > 50:
                results['errors'].append(f"Too many null values: {null_percentage:.1f}%")
            elif null_percentage > 20:
                results['warnings'].append(f"High null values: {null_percentage:.1f}%")
            
            # Check data types
            object_cols = data.select_dtypes(include=['object']).columns.tolist()
            if len(object_cols) > len(data.columns) * 0.8:
                results['warnings'].append("Many columns are non-numeric, may need preprocessing")
            
            # Check for network data patterns
            self._check_network_patterns(data, results)
            
            # Check for sufficient data volume
            if len(data) < 100:
                results['warnings'].append("Small dataset may affect anomaly detection quality")
            
            # Generate recommendations
            self._generate_recommendations(data, results)
            
            # Determine if valid
            results['is_valid'] = len(results['errors']) == 0
            
            logger.info(f"Data validation complete: {'Valid' if results['is_valid'] else 'Invalid'}")
            
        except Exception as e:
            error_msg = f"Error during validation: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        return results
    
    def _analyze_columns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze columns in the dataset."""
        analysis = {
            'numeric_columns': [],
            'categorical_columns': [],
            'datetime_columns': [],
            'high_cardinality_columns': [],
            'potential_id_columns': []
        }
        
        for col in data.columns:
            col_data = data[col]
            
            # Basic type categorization
            if col_data.dtype in ['int64', 'float64', 'int32', 'float32']:
                analysis['numeric_columns'].append(col)
            elif col_data.dtype == 'object':
                analysis['categorical_columns'].append(col)
            elif 'datetime' in str(col_data.dtype):
                analysis['datetime_columns'].append(col)
            
            # Check cardinality
            unique_count = col_data.nunique()
            if unique_count > len(data) * 0.8:
                analysis['high_cardinality_columns'].append(col)
            
            # Check for potential ID columns
            if (col.lower().endswith('id') or 
                col.lower().endswith('_id') or 
                unique_count == len(data)):
                analysis['potential_id_columns'].append(col)
        
        return analysis
    
    def _check_network_patterns(self, data: pd.DataFrame, results: Dict[str, Any]):
        """Check for network-specific data patterns."""
        columns = [col.lower() for col in data.columns]
        
        # Check for IP addresses
        ip_patterns = ['ip', 'src', 'dst', 'source', 'destination']
        has_ip_cols = any(pattern in ' '.join(columns) for pattern in ip_patterns)
        
        if has_ip_cols:
            results['data_summary']['appears_to_be_network_data'] = True
            results['recommendations'].append("Network data detected - consider using network-specific features")
        
        # Check for ports
        port_patterns = ['port', 'srcport', 'dstport', 'source_port', 'dest_port']
        has_port_cols = any(pattern in ' '.join(columns) for pattern in port_patterns)
        
        if has_port_cols:
            results['data_summary']['has_port_data'] = True
        
        # Check for protocols
        proto_patterns = ['proto', 'protocol', 'tcp', 'udp', 'icmp']
        has_proto_cols = any(pattern in ' '.join(columns) for pattern in proto_patterns)
        
        if has_proto_cols:
            results['data_summary']['has_protocol_data'] = True
    
    def _generate_recommendations(self, data: pd.DataFrame, results: Dict[str, Any]):
        """Generate data preprocessing recommendations."""
        column_analysis = results['column_analysis']
        
        # Recommendations based on column analysis
        if len(column_analysis['high_cardinality_columns']) > 0:
            results['recommendations'].append(
                f"Consider encoding or grouping high-cardinality columns: {', '.join(column_analysis['high_cardinality_columns'][:3])}"
            )
        
        if len(column_analysis['categorical_columns']) > 5:
            results['recommendations'].append(
                "Many categorical columns detected - consider feature selection or encoding"
            )
        
        if len(column_analysis['potential_id_columns']) > 0:
            results['recommendations'].append(
                f"Consider excluding ID columns from analysis: {', '.join(column_analysis['potential_id_columns'])}"
            )
        
        # Recommendations based on data summary
        if results['data_summary']['null_percentage'] > 5:
            results['recommendations'].append("Consider handling missing values before analysis")
        
        if len(data) > 100000:
            results['recommendations'].append("Large dataset detected - consider sampling for faster analysis")
    
    def prepare_features_for_analysis(self, data: pd.DataFrame, 
                                    exclude_columns: List[str] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for anomaly analysis.
        
        Args:
            data: Input DataFrame
            exclude_columns: Columns to exclude from analysis
            
        Returns:
            Tuple of (prepared_features, excluded_columns)
        """
        if exclude_columns is None:
            exclude_columns = []
        
        # Start with a copy
        features_df = data.copy()
        excluded = []
        
        # Auto-exclude common non-feature columns
        auto_exclude_patterns = [
            'id', '_id', 'index', 'row_id', 'session_id',
            'timestamp', 'time', 'date', 'datetime',
            'filename', 'file_name', 'source_file'
        ]
        
        for col in features_df.columns:
            col_lower = col.lower()
            
            # Check exclude patterns
            if any(pattern in col_lower for pattern in auto_exclude_patterns):
                if col not in exclude_columns:
                    exclude_columns.append(col)
                    excluded.append(f"Auto-excluded {col} (appears to be non-feature column)")
            
            # Check for high cardinality
            elif features_df[col].nunique() > len(features_df) * 0.9:
                if col not in exclude_columns:
                    exclude_columns.append(col)
                    excluded.append(f"Auto-excluded {col} (high cardinality)")
        
        # Remove excluded columns
        features_df = features_df.drop(columns=exclude_columns, errors='ignore')
        
        # Select only numeric columns for now
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        features_df = features_df[numeric_cols]
        
        # Handle missing values
        if features_df.isnull().any().any():
            features_df = features_df.fillna(0)
            excluded.append("Filled missing values with 0")
        
        return features_df, excluded
    
    def check_model_compatibility(self, data: pd.DataFrame, model_features: List[str]) -> Dict[str, Any]:
        """
        Check if data is compatible with a trained model.
        
        Args:
            data: Data to check
            model_features: Features the model was trained on
            
        Returns:
            Compatibility analysis
        """
        result = {
            'compatible': False,
            'missing_features': [],
            'extra_features': [],
            'feature_summary': {}
        }
        
        try:
            data_features = data.columns.tolist()
            
            # Check for missing features
            result['missing_features'] = [f for f in model_features if f not in data_features]
            
            # Check for extra features (not necessarily a problem)
            result['extra_features'] = [f for f in data_features if f not in model_features]
            
            # Feature summary
            result['feature_summary'] = {
                'model_expects': len(model_features),
                'data_has': len(data_features),
                'common_features': len(set(model_features) & set(data_features))
            }
            
            # Determine compatibility
            result['compatible'] = len(result['missing_features']) == 0
            
        except Exception as e:
            logger.error(f"Error checking model compatibility: {str(e)}")
        
        return result
    
    def show_validation_results(self, validation_results: Dict[str, Any]):
        """Display validation results in Streamlit UI."""
        if validation_results['is_valid']:
            st.success("✅ **Data validation passed!**")
        else:
            st.error("❌ **Data validation failed!**")
        
        # Show summary
        summary = validation_results['data_summary']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Rows", f"{summary.get('total_rows', 0):,}")
        with col2:
            st.metric("📋 Columns", summary.get('total_columns', 0))
        with col3:
            st.metric("💾 Memory (MB)", f"{summary.get('memory_usage_mb', 0):.1f}")
        with col4:
            st.metric("❓ Null %", f"{summary.get('null_percentage', 0):.1f}%")
        
        # Show errors
        if validation_results['errors']:
            st.error("**Errors found:**")
            for error in validation_results['errors']:
                st.error(f"• {error}")
        
        # Show warnings
        if validation_results['warnings']:
            st.warning("**Warnings:**")
            for warning in validation_results['warnings']:
                st.warning(f"• {warning}")
        
        # Show recommendations
        if validation_results['recommendations']:
            st.info("💡 **Recommendations:**")
            for rec in validation_results['recommendations']:
                st.info(f"• {rec}")

# Singleton instance
data_validator = DataValidationService()
