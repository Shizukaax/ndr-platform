#!/usr/bin/env python3
"""
Anomaly Storage Migration Script
Migrates anomaly data from cache/anomalies to data/anomaly_history
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime

def migrate_anomaly_storage():
    """Migrate anomaly data from cache to persistent storage"""
    
    # Paths
    old_cache_dir = Path("cache/anomalies")
    new_persistent_dir = Path("data/anomaly_history")
    backup_dir = Path("data/anomaly_history_backup")
    
    print("🔄 NDR Platform - Anomaly Storage Migration")
    print("=" * 50)
    
    # Check if old cache directory exists
    if not old_cache_dir.exists():
        print("✅ No migration needed - cache/anomalies directory not found")
        return
    
    # Create new directory
    new_persistent_dir.mkdir(parents=True, exist_ok=True)
    
    # Create backup of existing data if any
    if new_persistent_dir.exists() and any(new_persistent_dir.iterdir()):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"backup_{timestamp}"
        backup_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📦 Creating backup at: {backup_path}")
        for file in new_persistent_dir.glob("*.json"):
            shutil.copy2(file, backup_path / file.name)
    
    # Migrate files
    files_migrated = 0
    for file in old_cache_dir.glob("*.json"):
        dest_file = new_persistent_dir / file.name
        
        # If destination exists, merge the data
        if dest_file.exists():
            print(f"🔄 Merging data from {file.name}")
            try:
                # Load existing data
                with open(dest_file, 'r') as f:
                    existing_data = json.load(f)
                
                # Load migration data
                with open(file, 'r') as f:
                    migration_data = json.load(f)
                
                # Merge based on file type
                if file.name == "anomaly_history.json" and isinstance(existing_data, list):
                    # Merge anomaly history lists, avoiding duplicates
                    existing_ids = {record.get("detection_id") for record in existing_data}
                    for record in migration_data:
                        if record.get("detection_id") not in existing_ids:
                            existing_data.append(record)
                
                # Save merged data
                with open(dest_file, 'w') as f:
                    json.dump(existing_data, f, indent=2)
                
            except Exception as e:
                print(f"⚠️ Error merging {file.name}: {e}")
                print(f"   Copying as backup instead...")
                shutil.copy2(file, new_persistent_dir / f"migrated_{file.name}")
        else:
            print(f"📁 Migrating {file.name}")
            shutil.copy2(file, dest_file)
        
        files_migrated += 1
    
    print(f"\n✅ Migration completed!")
    print(f"   📁 Files migrated: {files_migrated}")
    print(f"   📂 New location: {new_persistent_dir.absolute()}")
    print(f"   🔒 Data is now in persistent storage (safe from cache clearing)")
    
    # Show cleanup option
    if files_migrated > 0:
        print(f"\n⚠️ Cleanup Option:")
        print(f"   You can now safely delete: {old_cache_dir.absolute()}")
        print(f"   (Recommended after verifying migration was successful)")

if __name__ == "__main__":
    migrate_anomaly_storage()
