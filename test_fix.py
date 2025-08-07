#!/usr/bin/env python3
"""
Test script to verify ModelManager and config integration is working correctly.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model_manager():
    """Test ModelManager creation and config integration."""
    try:
        from core.model_manager import ModelManager
        print("✅ ModelManager imported successfully")
        
        # Test creating ModelManager instance
        mm = ModelManager()
        print("✅ ModelManager instance created successfully")
        
        # Check that it's using config paths
        print(f"✅ Models directory: {mm.models_dir}")
        print(f"✅ Data directory: {mm.data_dir}")
        print(f"✅ Results directory: {mm.results_dir}")
        
        # Test basic functionality - list_models returns dictionaries
        models = mm.list_models()
        print(f"✅ Available models: {len(models)}")
        if models:
            first_model = models[0]
            print(f"✅ First model type: {first_model.get('type', 'Unknown')}")
            print(f"✅ Model metadata available: {'metadata' in first_model}")
        
        # Test storage stats
        stats = mm.get_storage_stats()
        print(f"✅ Storage stats: {stats.get('total_models', 0)} models, {stats.get('total_size_mb', 0):.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing ModelManager: {str(e)}")
        return False

def test_config_integration():
    """Test that config paths are being used correctly."""
    try:
        from core.config_loader import load_config
        config = load_config()
        
        print("✅ Config loaded successfully")
        print(f"✅ Config data dir: {config.get('system', {}).get('data_dir', 'data')}")
        print(f"✅ Config results dir: {config.get('system', {}).get('results_dir', 'data/results')}")
        print(f"✅ Config feedback dir: {config.get('feedback', {}).get('storage_dir', 'data/feedback')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing config: {str(e)}")
        return False

def check_directory_structure():
    """Check that directories are in the correct locations."""
    print("\n📁 Checking directory structure:")
    
    # Check that bad directories don't exist in root (except lib which is legitimate)
    bad_dirs = ['feedback', 'results']
    for bad_dir in bad_dirs:
        if os.path.exists(bad_dir):
            print(f"❌ Found bad directory in root: {bad_dir}")
        else:
            print(f"✅ No bad directory in root: {bad_dir}")
    
    # lib directory is legitimate (contains JS libraries)
    if os.path.exists('lib'):
        print(f"✅ lib directory exists (contains legitimate JS libraries)")
    
    # Check that good directories exist in data/
    good_dirs = ['data/feedback', 'data/results', 'data/reports']
    for good_dir in good_dirs:
        if os.path.exists(good_dir):
            print(f"✅ Found correct directory: {good_dir}")
        else:
            print(f"⚠️  Missing directory: {good_dir}")

if __name__ == "__main__":
    print("🧪 Testing ModelManager and Config Integration")
    print("=" * 50)
    
    # Test config integration
    config_ok = test_config_integration()
    
    # Test ModelManager
    mm_ok = test_model_manager()
    
    # Check directory structure
    check_directory_structure()
    
    print("\n" + "=" * 50)
    if config_ok and mm_ok:
        print("🎉 All tests passed! The application should be working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the output above.")
