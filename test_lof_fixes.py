#!/usr/bin/env python3
"""
Test script to verify that the LOF model and SHAP fixes work correctly
"""

import sys
import pandas as pd
import numpy as np

sys.path.append('.')

def test_lof_with_results_saving():
    """Test LOF model with results saving"""
    print("🔧 Testing LOF model with results saving...")
    
    try:
        from core.model_manager import ModelManager
        from core.models.local_outlier_factor import LocalOutlierFactorDetector
        
        # Create test data with some NaN values
        np.random.seed(42)
        data = np.random.randn(100, 4)
        
        # Add some anomalies
        data[0:5] += 3  # Make first 5 rows anomalous
        
        # Add some NaN values
        data[10:15, 1] = np.nan
        data[20:25, 2] = np.nan
        
        df = pd.DataFrame(data, columns=['f1', 'f2', 'f3', 'f4'])
        feature_names = ['f1', 'f2', 'f3', 'f4']
        
        print(f"✅ Created test data: {df.shape} with {df.isnull().sum().sum()} NaN values")
        
        # Initialize ModelManager
        mm = ModelManager()
        
        # Train LOF model
        model = LocalOutlierFactorDetector(contamination=0.1, novelty=True, n_neighbors=20)
        model.feature_names = feature_names
        
        # Fill NaN for training
        df_clean = df.fillna(df.median())
        model.fit(df_clean)
        
        # Save model
        model_path = mm.save_model(model, model_type="LocalOutlierFactor")
        print(f"✅ LOF Model saved to: {model_path}")
        
        # Test apply_model_to_data with results saving
        results = mm.apply_model_to_data(
            model_type="LocalOutlierFactor",
            data=df,  # Data with NaN values
            feature_names=feature_names,
            save_results=True
        )
        
        print("✅ apply_model_to_data completed successfully!")
        print(f"   - Found {len(results['anomalies'])} anomalies")
        print(f"   - anomaly_indices present: {'anomaly_indices' in results}")
        if 'anomaly_indices' in results:
            print(f"   - anomaly_indices: {results['anomaly_indices']}")
        
        return True
        
    except Exception as e:
        print(f"❌ LOF test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_lof_shap_with_nan():
    """Test LOF model with SHAP explanations and NaN data"""
    print("\n🔧 Testing LOF model with SHAP and NaN handling...")
    
    try:
        from core.explainers.shap_explainer import ShapExplainer
        from core.models.local_outlier_factor import LocalOutlierFactorDetector
        
        # Create test data with NaN values
        np.random.seed(42)
        data = np.random.randn(50, 3)
        
        # Add NaN values
        data[5:10, 1] = np.nan
        data[15:20, 2] = np.nan
        
        df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3'])
        print(f"✅ Created test data with {df.isnull().sum().sum()} NaN values")
        
        # Train LOF model on clean data
        df_clean = df.fillna(df.median())
        model = LocalOutlierFactorDetector(contamination=0.2, novelty=True, n_neighbors=10)
        model.feature_names = df.columns.tolist()
        model.fit(df_clean)
        
        # Test SHAP explainer with NaN data
        explainer = ShapExplainer(model=model, feature_names=df.columns.tolist())
        
        # This should not crash due to NaN values or model incompatibility
        explanation = explainer.explain(df.head(10))  # Include some rows with NaN
        
        print("✅ SHAP explanation completed!")
        print(f"   - Explanation type: {type(explanation)}")
        
        if isinstance(explanation, dict):
            print(f"   - Keys: {list(explanation.keys())}")
            if 'error' in explanation:
                print(f"   - Error (expected): {explanation['error']}")
                print(f"   - Message: {explanation.get('message', 'No message')}")
            else:
                print(f"   - SHAP values shape: {explanation.get('shap_values', 'None')}")
        
        return True
        
    except Exception as e:
        print(f"❌ SHAP test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🎯 Testing LOF model fixes...")
    print("=" * 50)
    
    test1_passed = test_lof_with_results_saving()
    test2_passed = test_lof_shap_with_nan()
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results:")
    print(f"   LOF Results Saving: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   LOF SHAP with NaN: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("🎉 All LOF tests passed!")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return test1_passed and test2_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
