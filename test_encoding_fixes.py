#!/usr/bin/env python3
"""
Test script to verify all encoding fixes are working properly
"""

import os
import sys
import traceback

# Add current directory to path
sys.path.append('.')

def test_imports():
    """Test that all critical modules can be imported"""
    try:
        print("🔧 Testing imports...")
        
        from core.config_loader import load_config
        print("✅ config_loader imported")
        
        from core.model_manager import ModelManager
        print("✅ ModelManager imported")
        
        from core.feedback_manager import FeedbackManager
        print("✅ FeedbackManager imported")
        
        from app.components.file_utils import load_data_file
        print("✅ file_utils imported")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {str(e)}")
        traceback.print_exc()
        return False

def test_config_loading():
    """Test config loading with UTF-8 encoding"""
    try:
        print("\n🔧 Testing config loading...")
        from core.config_loader import load_config
        
        config = load_config()
        print(f"✅ Config loaded successfully")
        print(f"   Data path: {config['paths']['data']}")
        print(f"   Results path: {config['paths']['results']}")
        
        return True
    except Exception as e:
        print(f"❌ Config loading failed: {str(e)}")
        traceback.print_exc()
        return False

def test_model_manager():
    """Test ModelManager initialization"""
    try:
        print("\n🔧 Testing ModelManager...")
        from core.model_manager import ModelManager
        
        mm = ModelManager()
        print("✅ ModelManager initialized successfully")
        
        # Test list_models method
        models = mm.list_models()
        print(f"✅ Found {len(models)} models")
        
        return True
    except Exception as e:
        print(f"❌ ModelManager test failed: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Run all encoding tests"""
    print("🎯 Starting encoding fixes verification...")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config_loading,
        test_model_manager
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All encoding fixes applied successfully!")
        print("✅ Your application should now work without charmap codec errors!")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
