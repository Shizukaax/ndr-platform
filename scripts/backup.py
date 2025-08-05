#!/usr/bin/env python3
"""
NDR Platform Backup Script
Automated backup and restore functionality for the platform.
"""

import os
import sys
import json
import shutil
import tarfile
import zipfile
from pathlib import Path
from datetime import datetime

def create_backup(backup_type='full', output_dir='backups'):
    """Create platform backup."""
    print(f"💾 Creating {backup_type} backup...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"ndr_platform_{backup_type}_{timestamp}"
    backup_path = Path(output_dir) / backup_name
    backup_path.mkdir(parents=True, exist_ok=True)
    
    if backup_type == 'full':
        return create_full_backup(backup_path)
    elif backup_type == 'data':
        return create_data_backup(backup_path)
    elif backup_type == 'models':
        return create_models_backup(backup_path)
    elif backup_type == 'config':
        return create_config_backup(backup_path)

def create_full_backup(backup_path):
    """Create full platform backup."""
    backup_items = {
        'data': ['data/'],
        'models': ['models/'],
        'config': ['config/', '.env', 'docker-compose.yml', 'requirements.txt'],
        'logs': ['logs/'],
        'reports': ['reports/'],
        'results': ['results/'],
        'feedback': ['feedback/']
    }
    
    for category, paths in backup_items.items():
        category_path = backup_path / category
        category_path.mkdir(exist_ok=True)
        
        for path in paths:
            src_path = Path(path)
            if src_path.exists():
                if src_path.is_file():
                    shutil.copy2(src_path, category_path / src_path.name)
                    print(f"   ✅ Backed up file: {src_path}")
                elif src_path.is_dir():
                    dst_path = category_path / src_path.name
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                    print(f"   ✅ Backed up directory: {src_path}")
            else:
                print(f"   ⚠️  Not found: {src_path}")
    
    # Create backup manifest
    manifest = {
        'backup_type': 'full',
        'timestamp': datetime.now().isoformat(),
        'platform_version': get_platform_version(),
        'items_backed_up': list(backup_items.keys())
    }
    
    with open(backup_path / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Compress backup
    archive_path = f"{backup_path}.tar.gz"
    with tarfile.open(archive_path, 'w:gz') as tar:
        tar.add(backup_path, arcname=backup_path.name)
    
    # Clean up temporary directory
    shutil.rmtree(backup_path)
    
    print(f"   ✅ Full backup created: {archive_path}")
    return archive_path

def create_data_backup(backup_path):
    """Create data-only backup."""
    data_sources = ['data/', 'results/', 'feedback/']
    
    for source in data_sources:
        src_path = Path(source)
        if src_path.exists() and src_path.is_dir():
            dst_path = backup_path / src_path.name
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            print(f"   ✅ Backed up: {src_path}")
    
    # Create manifest
    manifest = {
        'backup_type': 'data',
        'timestamp': datetime.now().isoformat(),
        'data_sources': data_sources
    }
    
    with open(backup_path / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    archive_path = f"{backup_path}.zip"
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(backup_path):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(backup_path)
                zipf.write(file_path, arcname)
    
    shutil.rmtree(backup_path)
    print(f"   ✅ Data backup created: {archive_path}")
    return archive_path

def create_models_backup(backup_path):
    """Create models-only backup."""
    models_dir = Path('models')
    if models_dir.exists():
        shutil.copytree(models_dir, backup_path / 'models', dirs_exist_ok=True)
        print(f"   ✅ Backed up models directory")
    
    # Create manifest with model information
    models_info = []
    for model_file in (backup_path / 'models').glob('*.pkl'):
        metadata_file = model_file.with_suffix('.json')
        metadata_file = metadata_file.with_name(metadata_file.stem + '_metadata.json')
        
        info = {
            'model_file': model_file.name,
            'size_mb': model_file.stat().st_size / (1024 * 1024),
            'created': datetime.fromtimestamp(model_file.stat().st_ctime).isoformat()
        }
        
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                info['metadata'] = metadata
            except:
                pass
        
        models_info.append(info)
    
    manifest = {
        'backup_type': 'models',
        'timestamp': datetime.now().isoformat(),
        'models_count': len(models_info),
        'models': models_info
    }
    
    with open(backup_path / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    archive_path = f"{backup_path}.tar.gz"
    with tarfile.open(archive_path, 'w:gz') as tar:
        tar.add(backup_path, arcname=backup_path.name)
    
    shutil.rmtree(backup_path)
    print(f"   ✅ Models backup created: {archive_path}")
    return archive_path

def create_config_backup(backup_path):
    """Create configuration-only backup."""
    config_files = [
        'config/',
        '.env',
        'docker-compose.yml',
        'requirements.txt',
        'nginx.conf',
        'Dockerfile'
    ]
    
    for config_file in config_files:
        src_path = Path(config_file)
        if src_path.exists():
            if src_path.is_file():
                shutil.copy2(src_path, backup_path / src_path.name)
                print(f"   ✅ Backed up: {src_path}")
            elif src_path.is_dir():
                dst_path = backup_path / src_path.name
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                print(f"   ✅ Backed up: {src_path}")
    
    manifest = {
        'backup_type': 'config',
        'timestamp': datetime.now().isoformat(),
        'config_files': config_files
    }
    
    with open(backup_path / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    archive_path = f"{backup_path}.zip"
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(backup_path):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(backup_path)
                zipf.write(file_path, arcname)
    
    shutil.rmtree(backup_path)
    print(f"   ✅ Config backup created: {archive_path}")
    return archive_path

def list_backups(backup_dir='backups'):
    """List available backups."""
    print(f"📋 Available backups in {backup_dir}:")
    
    backup_path = Path(backup_dir)
    if not backup_path.exists():
        print("   No backups directory found")
        return []
    
    backups = []
    for backup_file in backup_path.glob('ndr_platform_*'):
        if backup_file.suffix in ['.tar.gz', '.zip']:
            backup_info = {
                'file': backup_file,
                'name': backup_file.stem,
                'size_mb': backup_file.stat().st_size / (1024 * 1024),
                'created': datetime.fromtimestamp(backup_file.stat().st_ctime)
            }
            backups.append(backup_info)
    
    # Sort by creation time (newest first)
    backups.sort(key=lambda x: x['created'], reverse=True)
    
    for backup in backups:
        print(f"   📦 {backup['name']}")
        print(f"      Size: {backup['size_mb']:.1f} MB")
        print(f"      Created: {backup['created'].strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    return backups

def restore_backup(backup_file, restore_type='full'):
    """Restore from backup."""
    print(f"🔄 Restoring from: {backup_file}")
    
    backup_path = Path(backup_file)
    if not backup_path.exists():
        print(f"   ❌ Backup file not found: {backup_file}")
        return False
    
    # Create temporary extraction directory
    extract_dir = Path('temp_restore')
    extract_dir.mkdir(exist_ok=True)
    
    try:
        # Extract backup
        if backup_path.suffix == '.gz':
            with tarfile.open(backup_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
        elif backup_path.suffix == '.zip':
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                zipf.extractall(extract_dir)
        
        # Find the extracted directory
        extracted_dirs = [d for d in extract_dir.iterdir() if d.is_dir()]
        if not extracted_dirs:
            print("   ❌ No directory found in backup")
            return False
        
        backup_content = extracted_dirs[0]
        
        # Read manifest
        manifest_file = backup_content / 'manifest.json'
        if manifest_file.exists():
            with open(manifest_file) as f:
                manifest = json.load(f)
            print(f"   ℹ️  Backup type: {manifest.get('backup_type', 'unknown')}")
            print(f"   ℹ️  Created: {manifest.get('timestamp', 'unknown')}")
        
        # Restore based on type
        if restore_type == 'full' or restore_type == 'all':
            restore_full_backup(backup_content)
        elif restore_type == 'data':
            restore_data_backup(backup_content)
        elif restore_type == 'models':
            restore_models_backup(backup_content)
        elif restore_type == 'config':
            restore_config_backup(backup_content)
        
        print("   ✅ Restore completed successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Restore failed: {e}")
        return False
    finally:
        # Clean up
        if extract_dir.exists():
            shutil.rmtree(extract_dir)

def restore_full_backup(backup_content):
    """Restore full backup."""
    for item in backup_content.iterdir():
        if item.name == 'manifest.json':
            continue
            
        if item.is_dir():
            # Restore directory
            target_path = Path(item.name)
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.copytree(item, target_path)
            print(f"   ✅ Restored directory: {target_path}")

def restore_data_backup(backup_content):
    """Restore data backup."""
    data_dirs = ['data', 'results', 'feedback']
    for dir_name in data_dirs:
        src_dir = backup_content / dir_name
        if src_dir.exists():
            target_path = Path(dir_name)
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.copytree(src_dir, target_path)
            print(f"   ✅ Restored: {target_path}")

def restore_models_backup(backup_content):
    """Restore models backup."""
    models_dir = backup_content / 'models'
    if models_dir.exists():
        target_path = Path('models')
        if target_path.exists():
            shutil.rmtree(target_path)
        shutil.copytree(models_dir, target_path)
        print(f"   ✅ Restored models directory")

def restore_config_backup(backup_content):
    """Restore configuration backup."""
    for item in backup_content.iterdir():
        if item.name == 'manifest.json':
            continue
        
        target_path = Path(item.name)
        if item.is_file():
            shutil.copy2(item, target_path)
            print(f"   ✅ Restored file: {target_path}")
        elif item.is_dir():
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.copytree(item, target_path)
            print(f"   ✅ Restored directory: {target_path}")

def get_platform_version():
    """Get platform version from version file or git."""
    version_file = Path('VERSION')
    if version_file.exists():
        return version_file.read_text().strip()
    
    try:
        import subprocess
        result = subprocess.run(['git', 'describe', '--tags', '--always'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except:
        return 'unknown'

def cleanup_old_backups(backup_dir='backups', keep_count=10):
    """Clean up old backups, keeping only the most recent ones."""
    print(f"🧹 Cleaning up old backups (keeping {keep_count} most recent)...")
    
    backups = list_backups(backup_dir)
    if len(backups) <= keep_count:
        print(f"   ℹ️  Only {len(backups)} backups found, no cleanup needed")
        return
    
    backups_to_delete = backups[keep_count:]
    for backup in backups_to_delete:
        backup['file'].unlink()
        print(f"   🗑️  Deleted: {backup['name']}")
    
    print(f"   ✅ Cleaned up {len(backups_to_delete)} old backups")

def main():
    """Main backup function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='NDR Platform Backup Tool')
    parser.add_argument('action', choices=['create', 'list', 'restore', 'cleanup'],
                       help='Action to perform')
    parser.add_argument('--type', choices=['full', 'data', 'models', 'config'],
                       default='full', help='Backup type')
    parser.add_argument('--file', help='Backup file for restore')
    parser.add_argument('--output', default='backups', help='Output directory')
    parser.add_argument('--keep', type=int, default=10, 
                       help='Number of backups to keep during cleanup')
    
    args = parser.parse_args()
    
    print("💾 NDR Platform Backup Tool")
    print("=" * 50)
    
    # Change to project directory
    os.chdir(Path(__file__).parent.parent)
    
    try:
        if args.action == 'create':
            create_backup(args.type, args.output)
        elif args.action == 'list':
            list_backups(args.output)
        elif args.action == 'restore':
            if not args.file:
                print("❌ --file argument required for restore")
                sys.exit(1)
            restore_backup(args.file, args.type)
        elif args.action == 'cleanup':
            cleanup_old_backups(args.output, args.keep)
            
    except Exception as e:
        print(f"❌ Operation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
