#!/usr/bin/env python3
"""
NDR Platform Model Management Script
Tools for model training, evaluation, deployment, and lifecycle management.
"""

import os
import json
import pickle
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import hashlib
import numpy as np
from typing import Dict, List, Any

class ModelManager:
    """Model lifecycle management and operations."""
    
    def __init__(self, models_dir='models'):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Model registry file
        self.registry_file = self.models_dir / 'model_registry.json'
        self.registry = self._load_registry()
    
    def _load_registry(self):
        """Load model registry from file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("⚠️  Warning: Model registry corrupted, creating new one")
        
        return {
            'models': {},
            'created': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
    
    def _save_registry(self):
        """Save model registry to file."""
        self.registry['last_updated'] = datetime.now().isoformat()
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def list_models(self, status_filter=None):
        """List all models with their information."""
        print("📋 Model Inventory")
        print("=" * 70)
        
        if not self.registry['models']:
            print("   ❌ No models found in registry")
            return
        
        # Filter models by status if specified
        models_to_show = self.registry['models']
        if status_filter:
            models_to_show = {
                name: info for name, info in models_to_show.items()
                if info.get('status') == status_filter
            }
        
        if not models_to_show:
            print(f"   ❌ No models found with status: {status_filter}")
            return
        
        # Sort by creation date (newest first)
        sorted_models = sorted(
            models_to_show.items(),
            key=lambda x: x[1].get('created', ''),
            reverse=True
        )
        
        for model_name, model_info in sorted_models:
            self._print_model_info(model_name, model_info)
        
        print(f"\n📊 Total models: {len(models_to_show)}")
    
    def _print_model_info(self, name, info):
        """Print formatted model information."""
        status = info.get('status', 'unknown')
        status_emoji = {
            'active': '🟢',
            'archived': '📦',
            'deprecated': '🔴',
            'training': '🔄',
            'testing': '🧪'
        }.get(status, '❓')
        
        print(f"\n   {status_emoji} {name}")
        print(f"      Type: {info.get('type', 'Unknown')}")
        print(f"      Status: {status}")
        print(f"      Created: {info.get('created', 'Unknown')}")
        print(f"      Size: {info.get('size_mb', 0):.2f} MB")
        print(f"      Performance: {info.get('performance', {})}")
        
        if info.get('description'):
            print(f"      Description: {info['description']}")
    
    def evaluate_models(self, model_names=None):
        """Evaluate model performance and health."""
        print("🔬 Model Evaluation")
        print("=" * 50)
        
        models_to_eval = model_names or list(self.registry['models'].keys())
        if not models_to_eval:
            print("   ❌ No models to evaluate")
            return
        
        evaluation_results = {}
        
        for model_name in models_to_eval:
            if model_name not in self.registry['models']:
                print(f"   ❌ Model not found: {model_name}")
                continue
            
            print(f"\n   🔍 Evaluating: {model_name}")
            result = self._evaluate_single_model(model_name)
            evaluation_results[model_name] = result
            
            # Print results
            if result['status'] == 'healthy':
                print(f"      ✅ Status: Healthy")
            else:
                print(f"      ❌ Status: {result['status']}")
            
            for metric, value in result.get('metrics', {}).items():
                print(f"      📊 {metric}: {value}")
            
            if result.get('issues'):
                print(f"      ⚠️  Issues:")
                for issue in result['issues']:
                    print(f"         - {issue}")
        
        # Save evaluation report
        report_file = self.models_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"\n💾 Evaluation report saved: {report_file}")
        return evaluation_results
    
    def _evaluate_single_model(self, model_name):
        """Evaluate a single model."""
        model_info = self.registry['models'][model_name]
        result = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'metrics': {},
            'issues': []
        }
        
        # Check file existence
        model_file = self.models_dir / f"{model_name}.pkl"
        metadata_file = self.models_dir / f"{model_name}_metadata.json"
        
        if not model_file.exists():
            result['status'] = 'missing_file'
            result['issues'].append(f"Model file not found: {model_file}")
            return result
        
        # Check file integrity
        try:
            # Calculate current hash
            current_hash = self._calculate_file_hash(model_file)
            stored_hash = model_info.get('file_hash')
            
            if stored_hash and current_hash != stored_hash:
                result['issues'].append("File integrity check failed - hash mismatch")
            
            result['metrics']['file_hash'] = current_hash
            
        except Exception as e:
            result['issues'].append(f"Hash calculation failed: {e}")
        
        # Check model loadability
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            result['metrics']['loadable'] = True
            
            # Check if model has expected attributes
            if hasattr(model, 'predict'):
                result['metrics']['has_predict'] = True
            else:
                result['issues'].append("Model missing predict method")
            
        except Exception as e:
            result['issues'].append(f"Model loading failed: {e}")
            result['metrics']['loadable'] = False
        
        # Check metadata
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                result['metrics']['metadata_available'] = True
                result['metrics']['training_date'] = metadata.get('training_date')
                result['metrics']['model_version'] = metadata.get('version')
                
            except Exception as e:
                result['issues'].append(f"Metadata reading failed: {e}")
        else:
            result['issues'].append("Metadata file missing")
        
        # Check age
        created_date = model_info.get('created')
        if created_date:
            try:
                created_dt = datetime.fromisoformat(created_date.replace('Z', '+00:00'))
                age_days = (datetime.now() - created_dt).days
                result['metrics']['age_days'] = age_days
                
                if age_days > 90:  # 3 months
                    result['issues'].append(f"Model is {age_days} days old - consider retraining")
                
            except Exception as e:
                result['issues'].append(f"Date parsing failed: {e}")
        
        # Set overall status
        if result['issues']:
            result['status'] = 'issues_found'
        
        return result
    
    def cleanup_models(self, keep_latest=3, dry_run=True):
        """Clean up old model versions."""
        print(f"🧹 Model Cleanup (keep latest {keep_latest} versions)")
        print("=" * 50)
        
        if dry_run:
            print("   ℹ️  DRY RUN - No files will be deleted")
        
        # Group models by base name (without timestamp)
        model_groups = {}
        for model_name in self.registry['models']:
            base_name = self._extract_base_name(model_name)
            if base_name not in model_groups:
                model_groups[base_name] = []
            model_groups[base_name].append(model_name)
        
        cleanup_summary = {
            'groups_processed': 0,
            'models_removed': 0,
            'space_freed_mb': 0
        }
        
        for base_name, models in model_groups.items():
            if len(models) <= keep_latest:
                continue
            
            # Sort by creation date (newest first)
            sorted_models = sorted(
                models,
                key=lambda x: self.registry['models'][x].get('created', ''),
                reverse=True
            )
            
            # Models to remove (keep the latest ones)
            models_to_remove = sorted_models[keep_latest:]
            
            print(f"\n   📦 {base_name}:")
            print(f"      Total versions: {len(models)}")
            print(f"      Keeping: {keep_latest}")
            print(f"      Removing: {len(models_to_remove)}")
            
            for model_name in models_to_remove:
                model_info = self.registry['models'][model_name]
                size_mb = model_info.get('size_mb', 0)
                
                print(f"      - {model_name} ({size_mb:.2f} MB)")
                
                if not dry_run:
                    # Remove model files
                    model_file = self.models_dir / f"{model_name}.pkl"
                    metadata_file = self.models_dir / f"{model_name}_metadata.json"
                    
                    try:
                        if model_file.exists():
                            model_file.unlink()
                        if metadata_file.exists():
                            metadata_file.unlink()
                        
                        # Remove from registry
                        del self.registry['models'][model_name]
                        
                        cleanup_summary['models_removed'] += 1
                        cleanup_summary['space_freed_mb'] += size_mb
                        
                    except Exception as e:
                        print(f"        ❌ Failed to remove: {e}")
            
            cleanup_summary['groups_processed'] += 1
        
        if not dry_run:
            self._save_registry()
        
        print(f"\n📊 Cleanup Summary:")
        print(f"   Model groups processed: {cleanup_summary['groups_processed']}")
        print(f"   Models removed: {cleanup_summary['models_removed']}")
        print(f"   Space freed: {cleanup_summary['space_freed_mb']:.2f} MB")
        
        if dry_run:
            print(f"   ℹ️  Use --no-dry-run to actually remove these models")
    
    def backup_models(self, backup_dir=None):
        """Create backup of all models and registry."""
        print("💾 Creating Model Backup")
        print("=" * 50)
        
        if backup_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = Path(f"model_backup_{timestamp}")
        else:
            backup_dir = Path(backup_dir)
        
        backup_dir.mkdir(exist_ok=True)
        
        # Copy model files
        model_files = list(self.models_dir.glob("*.pkl"))
        metadata_files = list(self.models_dir.glob("*_metadata.json"))
        
        total_files = len(model_files) + len(metadata_files) + 1  # +1 for registry
        copied_files = 0
        total_size = 0
        
        # Copy model files
        for model_file in model_files:
            try:
                dest_file = backup_dir / model_file.name
                shutil.copy2(model_file, dest_file)
                copied_files += 1
                total_size += model_file.stat().st_size
                print(f"   ✅ {model_file.name}")
            except Exception as e:
                print(f"   ❌ Failed to copy {model_file}: {e}")
        
        # Copy metadata files
        for metadata_file in metadata_files:
            try:
                dest_file = backup_dir / metadata_file.name
                shutil.copy2(metadata_file, dest_file)
                copied_files += 1
                total_size += metadata_file.stat().st_size
                print(f"   ✅ {metadata_file.name}")
            except Exception as e:
                print(f"   ❌ Failed to copy {metadata_file}: {e}")
        
        # Copy registry
        try:
            dest_registry = backup_dir / "model_registry.json"
            shutil.copy2(self.registry_file, dest_registry)
            copied_files += 1
            total_size += self.registry_file.stat().st_size
            print(f"   ✅ model_registry.json")
        except Exception as e:
            print(f"   ❌ Failed to copy registry: {e}")
        
        # Create backup manifest
        manifest = {
            'backup_created': datetime.now().isoformat(),
            'source_directory': str(self.models_dir),
            'total_files': copied_files,
            'total_size_mb': total_size / (1024 * 1024),
            'models_backed_up': list(self.registry['models'].keys())
        }
        
        manifest_file = backup_dir / "backup_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n📊 Backup Summary:")
        print(f"   Files copied: {copied_files}/{total_files}")
        print(f"   Total size: {total_size / (1024 * 1024):.2f} MB")
        print(f"   Backup location: {backup_dir}")
        print(f"   Manifest: {manifest_file}")
        
        return backup_dir
    
    def restore_models(self, backup_dir):
        """Restore models from backup."""
        print(f"🔄 Restoring Models from Backup")
        print("=" * 50)
        
        backup_path = Path(backup_dir)
        if not backup_path.exists():
            print(f"   ❌ Backup directory not found: {backup_path}")
            return False
        
        # Check manifest
        manifest_file = backup_path / "backup_manifest.json"
        if manifest_file.exists():
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
            print(f"   📋 Backup created: {manifest['backup_created']}")
            print(f"   📁 Source: {manifest['source_directory']}")
            print(f"   📊 Files: {manifest['total_files']}")
        else:
            print("   ⚠️  No manifest found, proceeding anyway")
        
        # Backup current models before restore
        current_backup = self.backup_models(f"pre_restore_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        print(f"   💾 Current models backed up to: {current_backup}")
        
        # Restore files
        restored_files = 0
        for backup_file in backup_path.glob("*"):
            if backup_file.name in ["backup_manifest.json"]:
                continue
            
            try:
                dest_file = self.models_dir / backup_file.name
                shutil.copy2(backup_file, dest_file)
                restored_files += 1
                print(f"   ✅ Restored: {backup_file.name}")
            except Exception as e:
                print(f"   ❌ Failed to restore {backup_file}: {e}")
        
        # Reload registry
        self.registry = self._load_registry()
        
        print(f"\n📊 Restore Summary:")
        print(f"   Files restored: {restored_files}")
        print(f"   Registry reloaded: ✅")
        
        return True
    
    def register_model(self, model_path, model_type, description="", performance_metrics=None):
        """Register a new model in the registry."""
        model_file = Path(model_path)
        if not model_file.exists():
            print(f"❌ Model file not found: {model_file}")
            return False
        
        model_name = model_file.stem
        
        # Calculate file hash
        file_hash = self._calculate_file_hash(model_file)
        
        # Model information
        model_info = {
            'type': model_type,
            'description': description,
            'created': datetime.now().isoformat(),
            'file_path': str(model_file),
            'file_hash': file_hash,
            'size_mb': model_file.stat().st_size / (1024 * 1024),
            'status': 'active',
            'performance': performance_metrics or {}
        }
        
        # Add to registry
        self.registry['models'][model_name] = model_info
        self._save_registry()
        
        print(f"✅ Model registered: {model_name}")
        return True
    
    def _extract_base_name(self, model_name):
        """Extract base name from model (remove timestamp suffix)."""
        # Common patterns: model_YYYYMMDD_HHMMSS, model_comparison_YYYYMMDD_HHMMSS
        import re
        pattern = r'(.+?)_\d{8}_\d{6}$'
        match = re.match(pattern, model_name)
        return match.group(1) if match else model_name
    
    def _calculate_file_hash(self, file_path):
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

def main():
    """Main function for model management tool."""
    parser = argparse.ArgumentParser(description='NDR Platform Model Manager')
    parser.add_argument('command', 
                       choices=['list', 'evaluate', 'cleanup', 'backup', 'restore', 'register'],
                       help='Model management command to run')
    parser.add_argument('--models-dir', default='models',
                       help='Models directory path (default: models)')
    parser.add_argument('--status', choices=['active', 'archived', 'deprecated', 'training', 'testing'],
                       help='Filter models by status')
    parser.add_argument('--models', nargs='+',
                       help='Specific model names to operate on')
    parser.add_argument('--keep-latest', type=int, default=3,
                       help='Number of latest versions to keep during cleanup (default: 3)')
    parser.add_argument('--no-dry-run', action='store_true',
                       help='Actually perform destructive operations')
    parser.add_argument('--backup-dir',
                       help='Backup directory path')
    parser.add_argument('--model-path',
                       help='Path to model file for registration')
    parser.add_argument('--model-type',
                       help='Type of model for registration')
    parser.add_argument('--description',
                       help='Description for model registration')
    
    args = parser.parse_args()
    
    print("🤖 NDR Platform Model Manager")
    print("=" * 50)
    
    # Change to project directory
    os.chdir(Path(__file__).parent.parent)
    
    manager = ModelManager(args.models_dir)
    
    try:
        if args.command == 'list':
            manager.list_models(args.status)
        elif args.command == 'evaluate':
            manager.evaluate_models(args.models)
        elif args.command == 'cleanup':
            manager.cleanup_models(args.keep_latest, not args.no_dry_run)
        elif args.command == 'backup':
            manager.backup_models(args.backup_dir)
        elif args.command == 'restore':
            if not args.backup_dir:
                print("❌ --backup-dir required for restore command")
                return
            manager.restore_models(args.backup_dir)
        elif args.command == 'register':
            if not args.model_path or not args.model_type:
                print("❌ --model-path and --model-type required for register command")
                return
            manager.register_model(args.model_path, args.model_type, args.description or "")
            
    except Exception as e:
        print(f"❌ Model management error: {e}")

if __name__ == "__main__":
    main()
