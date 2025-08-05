#!/usr/bin/env python3
"""
NDR Platform Data Management Script
Tools for data validation, cleaning, transformation, and migration.
"""

import os
import json
import csv
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import hashlib
import gzip

class DataManager:
    """Data management and validation tools."""
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.examples_dir = self.data_dir / 'examples'
        self.realtime_dir = self.data_dir / 'realtime'
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.examples_dir.mkdir(exist_ok=True)
        self.realtime_dir.mkdir(exist_ok=True)
    
    def validate_json_files(self, directory=None):
        """Validate JSON files for format and content."""
        print("🔍 Validating JSON Files")
        print("=" * 50)
        
        target_dir = Path(directory) if directory else self.data_dir
        json_files = list(target_dir.rglob('*.json'))
        
        if not json_files:
            print(f"   ❌ No JSON files found in {target_dir}")
            return
        
        valid_files = 0
        invalid_files = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Validate structure for Arkime data
                if self._is_arkime_file(json_file):
                    if self._validate_arkime_structure(data):
                        valid_files += 1
                        print(f"   ✅ {json_file.name}")
                    else:
                        invalid_files.append((json_file, "Invalid Arkime structure"))
                        print(f"   ❌ {json_file.name} - Invalid structure")
                else:
                    # General JSON validation
                    valid_files += 1
                    print(f"   ✅ {json_file.name}")
                    
            except json.JSONDecodeError as e:
                invalid_files.append((json_file, f"JSON decode error: {e}"))
                print(f"   ❌ {json_file.name} - JSON decode error")
            except Exception as e:
                invalid_files.append((json_file, f"Error: {e}"))
                print(f"   ❌ {json_file.name} - {e}")
        
        # Summary
        print(f"\n📊 Validation Summary:")
        print(f"   Total files: {len(json_files)}")
        print(f"   Valid files: {valid_files}")
        print(f"   Invalid files: {len(invalid_files)}")
        
        if invalid_files:
            print(f"\n❌ Invalid Files:")
            for file_path, error in invalid_files:
                print(f"   {file_path}: {error}")
    
    def clean_old_data(self, days=30, dry_run=True):
        """Clean old data files based on age."""
        print(f"🧹 Cleaning Old Data (older than {days} days)")
        print("=" * 50)
        
        if dry_run:
            print("   ℹ️  DRY RUN - No files will be deleted")
        
        cutoff_date = datetime.now() - timedelta(days=days)
        old_files = []
        total_size = 0
        
        # Find old files
        for data_file in self.data_dir.rglob('*'):
            if data_file.is_file():
                modified_time = datetime.fromtimestamp(data_file.stat().st_mtime)
                if modified_time < cutoff_date:
                    file_size = data_file.stat().st_size
                    old_files.append((data_file, modified_time, file_size))
                    total_size += file_size
        
        if not old_files:
            print("   ✅ No old files found")
            return
        
        print(f"   📁 Found {len(old_files)} old files")
        print(f"   💾 Total size: {total_size / (1024 * 1024):.2f} MB")
        
        # Group by directory
        by_directory = {}
        for file_path, mod_time, size in old_files:
            parent = file_path.parent
            if parent not in by_directory:
                by_directory[parent] = []
            by_directory[parent].append((file_path, mod_time, size))
        
        # Display by directory
        for directory, files in by_directory.items():
            dir_size = sum(size for _, _, size in files)
            print(f"\n   📂 {directory}:")
            print(f"      Files: {len(files)}")
            print(f"      Size: {dir_size / (1024 * 1024):.2f} MB")
            
            for file_path, mod_time, size in files[:5]:  # Show first 5
                print(f"      - {file_path.name} ({mod_time.strftime('%Y-%m-%d')})")
            
            if len(files) > 5:
                print(f"      ... and {len(files) - 5} more files")
        
        # Delete files if not dry run
        if not dry_run:
            print(f"\n🗑️  Deleting {len(old_files)} old files...")
            for file_path, _, _ in old_files:
                try:
                    file_path.unlink()
                    print(f"   ✅ Deleted: {file_path}")
                except Exception as e:
                    print(f"   ❌ Failed to delete {file_path}: {e}")
        else:
            print(f"\n   ℹ️  Use --no-dry-run to actually delete these files")
    
    def compress_data(self, directory=None, extension='.json'):
        """Compress data files to save space."""
        print(f"🗜️  Compressing Data Files ({extension})")
        print("=" * 50)
        
        target_dir = Path(directory) if directory else self.data_dir
        files_to_compress = list(target_dir.rglob(f'*{extension}'))
        
        if not files_to_compress:
            print(f"   ❌ No {extension} files found")
            return
        
        compressed_count = 0
        total_saved = 0
        
        for file_path in files_to_compress:
            try:
                # Skip already compressed files
                if file_path.suffix == '.gz':
                    continue
                
                original_size = file_path.stat().st_size
                compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
                
                # Compress file
                with open(file_path, 'rb') as f_in:
                    with gzip.open(compressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                compressed_size = compressed_path.stat().st_size
                saved_space = original_size - compressed_size
                total_saved += saved_space
                
                # Remove original if compression successful
                file_path.unlink()
                compressed_count += 1
                
                compression_ratio = (saved_space / original_size) * 100
                print(f"   ✅ {file_path.name} → {compressed_path.name} "
                      f"({compression_ratio:.1f}% saved)")
                
            except Exception as e:
                print(f"   ❌ Failed to compress {file_path}: {e}")
        
        print(f"\n📊 Compression Summary:")
        print(f"   Files compressed: {compressed_count}")
        print(f"   Total space saved: {total_saved / (1024 * 1024):.2f} MB")
    
    def migrate_data_structure(self, old_format='arkime', new_format='standard'):
        """Migrate data from old format to new format."""
        print(f"🔄 Migrating Data Structure ({old_format} → {new_format})")
        print("=" * 50)
        
        migration_dir = self.data_dir / 'migration'
        migration_dir.mkdir(exist_ok=True)
        
        if old_format == 'arkime' and new_format == 'standard':
            self._migrate_arkime_to_standard(migration_dir)
        else:
            print(f"   ❌ Migration {old_format} → {new_format} not supported")
    
    def generate_data_report(self):
        """Generate comprehensive data report."""
        print("📊 Data Report")
        print("=" * 50)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'directories': {},
            'file_types': {},
            'total_size': 0,
            'total_files': 0
        }
        
        # Analyze directories
        for directory in [self.data_dir, self.examples_dir, self.realtime_dir]:
            if directory.exists():
                dir_info = self._analyze_directory(directory)
                report['directories'][str(directory)] = dir_info
                report['total_size'] += dir_info['total_size']
                report['total_files'] += dir_info['file_count']
                
                # Update file types
                for file_type, count in dir_info['file_types'].items():
                    report['file_types'][file_type] = report['file_types'].get(file_type, 0) + count
        
        # Display report
        print(f"\n📁 Directory Analysis:")
        for dir_path, info in report['directories'].items():
            print(f"   {Path(dir_path).name}:")
            print(f"     Files: {info['file_count']}")
            print(f"     Size: {info['total_size'] / (1024 * 1024):.2f} MB")
            print(f"     Last modified: {info['last_modified']}")
        
        print(f"\n📄 File Types:")
        for file_type, count in sorted(report['file_types'].items()):
            print(f"   {file_type}: {count} files")
        
        print(f"\n📈 Summary:")
        print(f"   Total files: {report['total_files']}")
        print(f"   Total size: {report['total_size'] / (1024 * 1024):.2f} MB")
        
        # Save report
        report_file = Path('data_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Report saved to: {report_file}")
        
        return report
    
    def export_data(self, format_type='csv', output_dir='exports'):
        """Export data in different formats."""
        print(f"📤 Exporting Data to {format_type.upper()}")
        print("=" * 50)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        json_files = list(self.data_dir.rglob('*.json'))
        if not json_files:
            print("   ❌ No JSON files found to export")
            return
        
        exported_count = 0
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if format_type == 'csv':
                    csv_file = output_path / f"{json_file.stem}.csv"
                    self._export_to_csv(data, csv_file)
                    exported_count += 1
                    print(f"   ✅ {json_file.name} → {csv_file.name}")
                
                elif format_type == 'txt':
                    txt_file = output_path / f"{json_file.stem}.txt"
                    self._export_to_txt(data, txt_file)
                    exported_count += 1
                    print(f"   ✅ {json_file.name} → {txt_file.name}")
                
            except Exception as e:
                print(f"   ❌ Failed to export {json_file}: {e}")
        
        print(f"\n📊 Export Summary:")
        print(f"   Files exported: {exported_count}")
        print(f"   Output directory: {output_path}")
    
    def _is_arkime_file(self, file_path):
        """Check if file is an Arkime data file."""
        return 'arkime' in file_path.name.lower() or 'pcap' in file_path.name.lower()
    
    def _validate_arkime_structure(self, data):
        """Validate Arkime data structure."""
        if not isinstance(data, list):
            return False
        
        for record in data[:5]:  # Check first 5 records
            if not isinstance(record, dict):
                return False
            
            # Check for common Arkime fields
            required_fields = ['timestamp', 'source', 'destination']
            if not any(field in record for field in required_fields):
                return False
        
        return True
    
    def _migrate_arkime_to_standard(self, migration_dir):
        """Migrate Arkime data to standard format."""
        arkime_files = [f for f in self.data_dir.rglob('*.json') if self._is_arkime_file(f)]
        
        if not arkime_files:
            print("   ❌ No Arkime files found")
            return
        
        migrated_count = 0
        
        for arkime_file in arkime_files:
            try:
                with open(arkime_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Transform data structure
                transformed_data = self._transform_arkime_data(data)
                
                # Save to migration directory
                new_file = migration_dir / f"standard_{arkime_file.name}"
                with open(new_file, 'w', encoding='utf-8') as f:
                    json.dump(transformed_data, f, indent=2)
                
                migrated_count += 1
                print(f"   ✅ Migrated: {arkime_file.name}")
                
            except Exception as e:
                print(f"   ❌ Failed to migrate {arkime_file}: {e}")
        
        print(f"\n📊 Migration Summary:")
        print(f"   Files migrated: {migrated_count}")
        print(f"   Output directory: {migration_dir}")
    
    def _transform_arkime_data(self, data):
        """Transform Arkime data to standard format."""
        transformed = []
        
        for record in data:
            if isinstance(record, dict):
                # Create standardized record
                standard_record = {
                    'id': record.get('_id', ''),
                    'timestamp': record.get('@timestamp', record.get('timestamp', '')),
                    'source_ip': record.get('source', {}).get('ip', ''),
                    'dest_ip': record.get('destination', {}).get('ip', ''),
                    'protocol': record.get('network', {}).get('protocol', ''),
                    'bytes': record.get('network', {}).get('bytes', 0),
                    'original_data': record  # Keep original for reference
                }
                transformed.append(standard_record)
        
        return transformed
    
    def _analyze_directory(self, directory):
        """Analyze directory contents."""
        info = {
            'file_count': 0,
            'total_size': 0,
            'file_types': {},
            'last_modified': None
        }
        
        latest_time = datetime.min
        
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                info['file_count'] += 1
                file_size = file_path.stat().st_size
                info['total_size'] += file_size
                
                # File type
                extension = file_path.suffix.lower() or 'no_extension'
                info['file_types'][extension] = info['file_types'].get(extension, 0) + 1
                
                # Last modified
                mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if mod_time > latest_time:
                    latest_time = mod_time
        
        info['last_modified'] = latest_time.isoformat() if latest_time != datetime.min else None
        return info
    
    def _export_to_csv(self, data, output_file):
        """Export JSON data to CSV format."""
        if not data:
            return
        
        # Flatten data if it's a list of objects
        if isinstance(data, list) and data and isinstance(data[0], dict):
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
        else:
            # Simple JSON structure
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['key', 'value'])
                if isinstance(data, dict):
                    for key, value in data.items():
                        writer.writerow([key, str(value)])
    
    def _export_to_txt(self, data, output_file):
        """Export JSON data to text format."""
        with open(output_file, 'w', encoding='utf-8') as f:
            if isinstance(data, (dict, list)):
                f.write(json.dumps(data, indent=2))
            else:
                f.write(str(data))

def main():
    """Main function for data management tool."""
    parser = argparse.ArgumentParser(description='NDR Platform Data Management Tool')
    parser.add_argument('command', choices=['validate', 'clean', 'compress', 'migrate', 'report', 'export'],
                       help='Data management command to run')
    parser.add_argument('--data-dir', default='data',
                       help='Data directory path (default: data)')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days for cleaning (default: 30)')
    parser.add_argument('--no-dry-run', action='store_true',
                       help='Actually perform destructive operations')
    parser.add_argument('--format', choices=['csv', 'txt'], default='csv',
                       help='Export format (default: csv)')
    parser.add_argument('--extension', default='.json',
                       help='File extension for compression (default: .json)')
    parser.add_argument('--directory',
                       help='Specific directory to operate on')
    
    args = parser.parse_args()
    
    print("🗂️  NDR Platform Data Manager")
    print("=" * 50)
    
    # Change to project directory
    os.chdir(Path(__file__).parent.parent)
    
    manager = DataManager(args.data_dir)
    
    try:
        if args.command == 'validate':
            manager.validate_json_files(args.directory)
        elif args.command == 'clean':
            manager.clean_old_data(args.days, not args.no_dry_run)
        elif args.command == 'compress':
            manager.compress_data(args.directory, args.extension)
        elif args.command == 'migrate':
            manager.migrate_data_structure()
        elif args.command == 'report':
            manager.generate_data_report()
        elif args.command == 'export':
            manager.export_data(args.format)
            
    except Exception as e:
        print(f"❌ Data management error: {e}")

if __name__ == "__main__":
    main()
