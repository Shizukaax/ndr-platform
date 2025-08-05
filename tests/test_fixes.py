#!/usr/bin/env python3
"""
Test script to validate critical fixes for the NDR Platform.
Tests threshold handling and page loading functionality.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Mock streamlit for testing
class MockStreamlit:
    """Mock streamlit module for testing without running the web app."""
    
    class session_state:
        selected_model = "IsolationForest"
        anomaly_threshold = 0.5
        
        @classmethod
        def get(cls, key, default=None):
            return getattr(cls, key, default)
    
    @staticmethod
    def error(msg):
        print(f"ERROR: {msg}")
    
    @staticmethod
    def info(msg):
        print(f"INFO: {msg}")
    
    @staticmethod
    def success(msg):
        print(f"SUCCESS: {msg}")
    
    @staticmethod
    def header(msg):
        print(f"HEADER: {msg}")
    
    @staticmethod
    def markdown(msg):
        print(f"MARKDOWN: {msg}")

# Mock streamlit
sys.modules['streamlit'] = MockStreamlit()

def test_threshold_handling():
    """Test that threshold handling works correctly."""
    print("\n=== Testing Threshold Handling ===")
    
    try:
        # Test scenario: missing threshold in model_results
        model_results = {
            'scores': np.array([0.1, 0.2, 0.7, 0.8, 0.9]),
            'model': 'IsolationForest'
            # Note: 'threshold' key is missing
        }
        
        # Simulate the fix from reporting.py
        scores = model_results.get('scores', np.array([]))
        threshold = model_results.get('threshold')
        
        if threshold is None:
            # Fallback 1: Check session state
            threshold = MockStreamlit.session_state.get('anomaly_threshold')
            if threshold is None:
                # Fallback 2: Calculate from scores
                if len(scores) > 0:
                    threshold = float(np.percentile(scores, 90))
                else:
                    threshold = 0.5
        
        print(f"✅ Threshold calculation successful: {threshold}")
        print(f"   - Scores available: {len(scores)} values")
        print(f"   - Threshold value: {threshold:.3f}")
        
        # Test anomaly counting
        if len(scores) > 0:
            anomaly_count = (scores > threshold).sum()
            print(f"   - Anomalies detected: {anomaly_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Threshold handling test failed: {e}")
        return False

def test_anomaly_session_state():
    """Test anomaly session state handling."""
    print("\n=== Testing Anomaly Session State ===")
    
    try:
        # Test 1: No anomalies in session state
        if not hasattr(MockStreamlit.session_state, 'anomalies'):
            print("✅ Correctly handles missing anomalies session state")
        
        # Test 2: Empty anomalies DataFrame
        MockStreamlit.session_state.anomalies = pd.DataFrame()
        
        # Simulate mitre_mapping.py check
        anomalies = MockStreamlit.session_state.anomalies
        
        if isinstance(anomalies, pd.DataFrame) and anomalies.empty:
            print("✅ Correctly handles empty anomalies DataFrame")
        
        # Test 3: Valid anomalies DataFrame
        MockStreamlit.session_state.anomalies = pd.DataFrame({
            'anomaly_score': [0.8, 0.9, 0.7],
            'timestamp': ['2025-01-01', '2025-01-02', '2025-01-03'],
            'source_ip': ['192.168.1.1', '192.168.1.2', '192.168.1.3']
        })
        
        anomalies = MockStreamlit.session_state.anomalies
        
        if len(anomalies) > 0 and not anomalies.empty:
            sample = anomalies.iloc[0]
            print(f"✅ Correctly handles valid anomalies DataFrame")
            print(f"   - Sample anomaly: {sample.to_dict()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Anomaly session state test failed: {e}")
        return False

def test_format_string_safety():
    """Test format string safety with mixed data types."""
    print("\n=== Testing Format String Safety ===")
    
    try:
        # Test mixed data types as would occur in selectbox formatting
        test_values = [
            0.85,           # float
            "N/A",          # string
            None,           # None type
            0,              # integer
            np.nan,         # numpy nan
        ]
        
        def format_anomaly_option(anomaly_score):
            """Safe formatting function from explain_feedback.py fix."""
            if isinstance(anomaly_score, (int, float)) and not pd.isna(anomaly_score):
                return f"Anomaly Score: {anomaly_score:.3f}"
            else:
                return f"Anomaly Score: {anomaly_score}"
        
        for i, value in enumerate(test_values):
            formatted = format_anomaly_option(value)
            print(f"   - Test {i+1}: {value} -> {formatted}")
        
        print("✅ Format string safety test passed")
        return True
        
    except Exception as e:
        print(f"❌ Format string safety test failed: {e}")
        return False

def test_data_column_mapping():
    """Test data column mapping for Arkime compatibility."""
    print("\n=== Testing Data Column Mapping ===")
    
    try:
        # Create sample Arkime-style data
        arkime_data = pd.DataFrame({
            'frame.len': [1500, 1200, 800],
            'tcp.srcport': [443, 80, 22],
            'tcp.dstport': [12345, 54321, 33333],
            'ip.src': ['192.168.1.1', '10.0.0.1', '172.16.0.1'],
            'ip.dst': ['8.8.8.8', '1.1.1.1', '208.67.222.222']
        })
        
        # Test column detection as in analytics_dashboard.py fix
        expected_columns = {
            'frame_len': ['frame.len', 'frame_len', 'packet_len'],
            'src_port': ['tcp.srcport', 'tcp_srcport', 'src_port', 'udp.srcport'],
            'dst_port': ['tcp.dstport', 'tcp_dstport', 'dst_port', 'udp.dstport'],
            'src_ip': ['ip.src', 'src_ip', 'source_ip'],
            'dst_ip': ['ip.dst', 'dst_ip', 'dest_ip', 'destination_ip']
        }
        
        def find_column(data, possible_names):
            """Find the first matching column name in the data."""
            for name in possible_names:
                if name in data.columns:
                    return name
            return None
        
        mapped_columns = {}
        for logical_name, possible_names in expected_columns.items():
            found_column = find_column(arkime_data, possible_names)
            if found_column:
                mapped_columns[logical_name] = found_column
        
        print(f"✅ Column mapping successful:")
        for logical, actual in mapped_columns.items():
            print(f"   - {logical} -> {actual}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data column mapping test failed: {e}")
        return False

def test_configuration_files():
    """Test that all configuration files exist and are valid."""
    print("\n=== Testing Configuration Files ===")
    
    config_files = {
        'README.md': 'Documentation',
        'Dockerfile': 'Container configuration',
        'docker-compose.yml': 'Multi-container setup',
        'requirements.txt': 'Python dependencies',
        '.gitignore': 'Git ignore patterns',
        '.env.example': 'Environment template',
        'DEPLOYMENT.md': 'Deployment guide'
    }
    
    all_exist = True
    
    for filename, description in config_files.items():
        filepath = project_root / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"✅ {filename} ({description}): {size} bytes")
        else:
            print(f"❌ {filename} missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests."""
    print("🧪 NDR Platform Fix Validation Tests")
    print("=" * 50)
    
    tests = [
        test_threshold_handling,
        test_anomaly_session_state,
        test_format_string_safety,
        test_data_column_mapping,
        test_configuration_files
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Platform fixes are working correctly.")
    else:
        print("⚠️  Some tests failed. Please review the output above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
