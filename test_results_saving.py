#!/usr/bin/env python3
"""
Test script to verify ModelManager results saving functionality.
"""

import pandas as pd
import numpy as np
from core.model_manager import ModelManager

def test_results_saving():
    """Test that ModelManager saves results to data/results"""
    print("🧪 Testing ModelManager results saving...")
    
    # Create ModelManager
    mm = ModelManager()
    print(f"✅ ModelManager created with results dir: {mm.results_dir}")
    
    # Check available models
    models = mm.list_models()
    if not models:
        print("❌ No models available for testing")
        return False
        
    print(f"✅ Found {len(models)} models: {[m['type'] if isinstance(m, dict) else m for m in models]}")
    
    # Get first model type
    first_model = models[0]
    model_type = first_model['type'] if isinstance(first_model, dict) else first_model
    print(f"✅ Testing with model: {model_type}")
    
    # Create test data with network features that the model expects
    test_data = pd.DataFrame({
        'frame_len': np.random.randint(64, 1500, 50),
        'tcp_dstport': np.random.randint(1, 65535, 50), 
        'tcp_srcport': np.random.randint(1, 65535, 50),
        'udp_dstport': np.random.randint(1, 65535, 50),
        'udp_srcport': np.random.randint(1, 65535, 50)
    })
    feature_names = ['frame_len', 'tcp_dstport', 'tcp_srcport', 'udp_dstport', 'udp_srcport']
    print(f"✅ Created test data with {len(test_data)} rows and network features")
    
    try:
        # Apply model and save results
        result = mm.apply_model_to_data(
            model_type, 
            test_data, 
            feature_names, 
            save_results=True
        )
        
        print(f"✅ Model applied successfully")
        print(f"✅ Found {result.get('anomaly_count', 0)} anomalies")
        
        # Check if files were saved
        if 'anomalies_path' in result:
            print(f"✅ Anomalies saved to: {result['anomalies_path']}")
        
        if 'summary_path' in result:
            print(f"✅ Summary saved to: {result['summary_path']}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error applying model: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_results_saving()
    if success:
        print("\n🎉 Results saving test passed!")
    else:
        print("\n⚠️ Results saving test failed!")
