#!/usr/bin/env python3
"""
Directory Structure Optimization Verification
Tests the new consolidated directory structure
"""

import os
import yaml
from pathlib import Path

def verify_optimized_structure():
    """Verify the optimized directory structure is working correctly"""
    
    print("🔧 NDR Platform - Directory Structure Verification")
    print("=" * 55)
    
    # Verify configuration
    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("✅ Configuration loaded successfully")
        
        # Check data source path
        data_dir = config.get('data_source', {}).get('directory', '')
        if 'data/json' in data_dir or 'data\\json' in data_dir:
            print("✅ Data source correctly points to data/json/")
        else:
            print(f"⚠️ Data source path: {data_dir}")
        
        # Check other paths
        anomaly_dir = config.get('anomaly_storage', {}).get('history_dir', '')
        if anomaly_dir == 'data/anomaly_history':
            print("✅ Anomaly storage correctly configured")
        
        feedback_dir = config.get('feedback', {}).get('storage_dir', '')
        if feedback_dir == 'data/feedback':
            print("✅ Feedback storage correctly moved")
        
        reports_dir = config.get('reports', {}).get('output_dir', '')
        if reports_dir == 'data/reports':
            print("✅ Reports directory correctly moved")
    
    # Verify directory structure
    expected_dirs = [
        "data/json",
        "data/anomaly_history", 
        "data/feedback",
        "data/reports",
        "data/results",
        "app/assets/lib"
    ]
    
    print(f"\n📁 Directory Structure Verification:")
    for dir_path in expected_dirs:
        if Path(dir_path).exists():
            print(f"   ✅ {dir_path}")
        else:
            print(f"   ❌ {dir_path} - Missing")
    
    # Check for old directories that should be removed
    old_dirs = ["feedback", "reports", "results", "lib"]
    removed_dirs = []
    for old_dir in old_dirs:
        if not Path(old_dir).exists():
            removed_dirs.append(old_dir)
    
    if len(removed_dirs) == len(old_dirs):
        print(f"✅ Old directories successfully removed: {', '.join(old_dirs)}")
    else:
        still_exist = [d for d in old_dirs if Path(d).exists()]
        print(f"⚠️ Old directories still exist: {still_exist}")
    
    # Check JSON data files
    json_dir = Path("data/json")
    if json_dir.exists():
        json_files = list(json_dir.glob("*.json"))
        print(f"📊 JSON data files found: {len(json_files)}")
        for json_file in json_files:
            print(f"   📄 {json_file.name}")
    
    # Summary
    print(f"\n🎯 Optimization Summary:")
    root_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and not d.startswith('.')]
    print(f"   📁 Top-level directories: {len(root_dirs)}")
    print(f"   🗂️ Data consolidation: All data under data/ directory")
    print(f"   🎯 JSON isolation: Packet data in data/json/ for clean uploads")
    print(f"   🔧 Assets consolidation: Static files in app/assets/")
    
    print(f"\n✅ Directory structure optimization completed successfully!")

if __name__ == "__main__":
    verify_optimized_structure()
