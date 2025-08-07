#!/usr/bin/env python3
"""
Test script to verify ModelManager results saving functionality.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model_application():
    """Test applying a model and saving results."""
    try:
        from core.model_manager import ModelManager
        print("✅ ModelManager imported successfully")
        
        # Create ModelManager instance
        mm = ModelManager()
        print(f"✅ ModelManager created, results dir: {mm.results_dir}")
        
        # Check available models
        models = mm.list_models()
        print(f"✅ Available models: {models}")
        
        if not models:
            print("❌ No models found. Please train a model first.")
            return False
        
        # Create some test data
        np.random.seed(42)
        test_data = np.random.randn(100, 5)  # 100 samples, 5 features
        feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
        
        print(f"✅ Created test data: {test_data.shape}")
        
        # Apply the first available model with save_results=True
        model_type = models[0]
        print(f"✅ Testing with model: {model_type}")
        
        results = mm.apply_model_to_data(
            model_type=model_type,
            data=test_data,
            feature_names=feature_names,
            save_results=True  # This should create files
        )
        
        print(f"✅ Model applied successfully!")
        print(f"   - Total samples: {results['total_samples']}")
        print(f"   - Anomaly count: {results['anomaly_count']}")
        print(f"   - Anomaly percentage: {results['anomaly_percentage']:.2f}%")
        print(f"   - Threshold: {results['threshold']}")
        
        # Check if files were created
        results_dir = mm.results_dir
        if os.path.exists(results_dir):
            files = os.listdir(results_dir)
            print(f"✅ Files in results directory: {files}")
            
            if files:
                print("🎉 SUCCESS: Files were created in data/results!")
                return True
            else:
                print("⚠️ WARNING: No files found in results directory")
                return False
        else:
            print(f"❌ Results directory does not exist: {results_dir}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing model application: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing ModelManager Results Saving")
    print("=" * 50)
    
    success = test_model_application()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! ModelManager is saving results correctly.")
    else:
        print("⚠️ Some tests failed. Please check the output above.")
