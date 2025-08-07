#!/usr/bin/env python3
"""
Test script to verify that anomaly detection now saves results properly
and handles NaN values correctly.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.append('.')

def test_results_saving():
    """Test that anomaly detection saves results to data/results"""
    print("🔧 Testing anomaly detection results saving...")
    
    try:
        from core.model_manager import ModelManager
        from core.models.isolation_forest import IsolationForestDetector
        
        # Create test data with some NaN values
        np.random.seed(42)
        n_samples = 1000
        n_features = 5
        
        # Create normal data
        normal_data = np.random.randn(n_samples, n_features)
        
        # Add some anomalies
        anomaly_indices = np.random.choice(n_samples, size=50, replace=False)
        normal_data[anomaly_indices] += np.random.randn(50, n_features) * 3
        
        # Add some NaN values to test NaN handling
        nan_indices = np.random.choice(n_samples, size=20, replace=False)
        nan_feature_indices = np.random.choice(n_features, size=20, replace=True)
        for i, feat_idx in zip(nan_indices, nan_feature_indices):
            normal_data[i, feat_idx] = np.nan
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(normal_data, columns=feature_names)
        
        print(f"✅ Created test data: {df.shape} with {df.isnull().sum().sum()} NaN values")
        
        # Initialize ModelManager
        mm = ModelManager()
        
        # Test apply_model_to_data which should save results
        print("🔧 Testing apply_model_to_data with results saving...")
        
        # First train a model
        model = IsolationForestDetector(contamination=0.1, random_state=42)
        model.feature_names = feature_names
        
        # Fill NaN values for training (models can't train on NaN)
        df_clean = df.fillna(df.median())
        model.fit(df_clean)
        
        # Save the model
        model_path = mm.save_model(model, model_type="IsolationForest")
        print(f"✅ Model saved to: {model_path}")
        
        # Now test apply_model_to_data (this should handle NaN and save results)
        results = mm.apply_model_to_data(
            model_type="IsolationForest",
            data=df,  # Data with NaN values
            feature_names=feature_names,
            save_results=True
        )
        
        print("✅ apply_model_to_data completed successfully!")
        print(f"   - Found {len(results['anomalies'])} anomalies")
        print(f"   - Threshold: {results['threshold']:.4f}")
        print(f"   - Results saved to: {results.get('results_path', 'data/results')}")
        
        # Check if files were actually created
        from pathlib import Path
        results_dir = Path("data/results/IsolationForest")
        
        if results_dir.exists():
            csv_files = list(results_dir.glob("anomalies_*.csv"))
            json_files = list(results_dir.glob("summary_*.json"))
            
            print(f"✅ Found {len(csv_files)} CSV files and {len(json_files)} JSON files in results directory")
            
            if csv_files:
                latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
                print(f"   Latest CSV: {latest_csv}")
                
                # Check the CSV content
                anomalies_df = pd.read_csv(latest_csv)
                print(f"   CSV contains {len(anomalies_df)} anomaly records")
                
            if json_files:
                latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
                print(f"   Latest JSON: {latest_json}")
        else:
            print("❌ Results directory not found!")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_shap_nan_handling():
    """Test that SHAP explainer handles NaN values properly"""
    print("\n🔧 Testing SHAP explainer NaN handling...")
    
    try:
        from core.explainers.shap_explainer import ShapExplainer
        from core.models.isolation_forest import IsolationForestDetector
        
        # Create test data with NaN values
        np.random.seed(42)
        data = np.random.randn(100, 4)
        
        # Add NaN values
        data[10:15, 1] = np.nan
        data[20:25, 2] = np.nan
        
        df = pd.DataFrame(data, columns=['f1', 'f2', 'f3', 'f4'])
        print(f"✅ Created test data with {df.isnull().sum().sum()} NaN values")
        
        # Train a model on clean data
        df_clean = df.fillna(df.median())
        model = IsolationForestDetector(contamination=0.1, random_state=42)
        model.feature_names = df.columns.tolist()
        model.fit(df_clean)
        
        # Test SHAP explainer with NaN data
        explainer = ShapExplainer(model=model, feature_names=df.columns.tolist())
        
        # This should not crash due to NaN values
        explanation = explainer.explain(df.head(10))  # Include some rows with NaN
        
        print("✅ SHAP explanation completed successfully with NaN data!")
        print(f"   - Explanation type: {type(explanation)}")
        
        if isinstance(explanation, dict):
            print(f"   - Keys: {list(explanation.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ SHAP test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🎯 Testing anomaly detection fixes...")
    print("=" * 50)
    
    test1_passed = test_results_saving()
    test2_passed = test_shap_nan_handling()
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results:")
    print(f"   Results Saving: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   SHAP NaN Handling: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("🎉 All tests passed! Your anomaly detection should now work properly.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return test1_passed and test2_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
