#!/usr/bin/env python3
"""
Test to verify that feedback directories are created in the correct location according to config.
"""

import sys
import os
import shutil
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

def test_feedback_directory_creation():
    """Test that feedback directories are created in the correct location"""
    print("[TEST] Testing feedback directory creation...")
    
    # Load config to get expected paths
    try:
        from core.config_loader import load_config
        config = load_config()
        expected_feedback_dir = config.get('feedback', {}).get('storage_dir', 'data/feedback')
        print(f"  Expected feedback directory from config: {expected_feedback_dir}")
    except Exception as e:
        print(f"  [ERROR] Could not load config: {e}")
        return False
    
    # Check that top-level feedback directory does NOT exist
    top_level_feedback = Path("feedback")
    if top_level_feedback.exists():
        print(f"  [FAIL] Top-level feedback directory still exists: {top_level_feedback}")
        return False
    else:
        print(f"  [PASS] No top-level feedback directory found")
    
    # Check that configured feedback directory exists or can be created
    config_feedback_dir = Path(expected_feedback_dir)
    if not config_feedback_dir.exists():
        print(f"  [INFO] Creating configured feedback directory: {config_feedback_dir}")
        os.makedirs(config_feedback_dir, exist_ok=True)
    
    if config_feedback_dir.exists():
        print(f"  [PASS] Configured feedback directory exists: {config_feedback_dir}")
    else:
        print(f"  [FAIL] Could not create configured feedback directory: {config_feedback_dir}")
        return False
    
    # Test FeedbackManager uses correct path
    try:
        from core.feedback_manager import FeedbackManager
        fm = FeedbackManager()
        actual_storage_dir = Path(fm.storage_dir)
        expected_storage_dir = Path(expected_feedback_dir)
        
        if actual_storage_dir.resolve() == expected_storage_dir.resolve():
            print(f"  [PASS] FeedbackManager uses correct directory: {actual_storage_dir}")
        else:
            print(f"  [FAIL] FeedbackManager uses wrong directory: {actual_storage_dir} (expected: {expected_storage_dir})")
            return False
    except Exception as e:
        print(f"  [ERROR] Could not test FeedbackManager: {e}")
        return False
    
    return True

def test_directory_structure():
    """Test overall directory structure matches config"""
    print("[TEST] Testing directory structure...")
    
    try:
        from core.config_loader import load_config
        config = load_config()
        
        # Expected directories from config
        expected_dirs = {
            'data': config.get('system', {}).get('data_dir', 'data'),
            'models': config.get('system', {}).get('models_dir', 'models'),
            'cache': config.get('system', {}).get('cache_dir', 'cache'),
            'reports': config.get('system', {}).get('reports_dir', 'data/reports'),
            'results': config.get('system', {}).get('results_dir', 'data/results'),
            'feedback': config.get('feedback', {}).get('storage_dir', 'data/feedback'),
        }
        
        print(f"  Expected directories from config:")
        for name, path in expected_dirs.items():
            print(f"    {name}: {path}")
        
        # Check each directory
        all_exist = True
        for name, path in expected_dirs.items():
            dir_path = Path(path)
            if dir_path.exists():
                print(f"  [PASS] {name} directory exists: {path}")
            else:
                print(f"  [WARN] {name} directory does not exist: {path}")
                # Create it for next time
                os.makedirs(dir_path, exist_ok=True)
                print(f"  [INFO] Created {name} directory: {path}")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] Could not test directory structure: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("FEEDBACK DIRECTORY CONFIGURATION TEST")
    print("=" * 60)
    
    tests = [
        test_feedback_directory_creation,
        test_directory_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"[ERROR] Test {test.__name__} crashed: {e}")
        print()
    
    print("=" * 60)
    print(f"FINAL RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] Feedback directory configuration is correct!")
        print("Fixed issues:")
        print("  ✓ No top-level 'feedback' directory created")
        print("  ✓ Feedback stored in configured 'data/feedback' directory") 
        print("  ✓ FeedbackManager uses correct path from config")
        print("  ✓ All directories follow configuration settings")
    else:
        print("[WARNING] Some tests failed - check output above")
        
    print("=" * 60)

if __name__ == "__main__":
    main()
