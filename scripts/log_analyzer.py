#!/usr/bin/env python3
"""
NDR Platform Log Analysis Script
Analyze and monitor application logs for issues, performance, and trends.
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import argparse

class LogAnalyzer:
    """Log analysis and monitoring tool."""
    
    def __init__(self, log_dir='logs'):
        self.log_dir = Path(log_dir)
        self.log_files = self._discover_log_files()
        
    def _discover_log_files(self):
        """Discover all log files in the logs directory."""
        if not self.log_dir.exists():
            print(f"❌ Log directory not found: {self.log_dir}")
            return []
        
        log_files = []
        for log_file in self.log_dir.glob('*.log*'):
            if log_file.is_file():
                log_files.append(log_file)
        
        return sorted(log_files)
    
    def analyze_errors(self, hours=24):
        """Analyze error patterns in logs."""
        print(f"🔍 Analyzing Errors (last {hours} hours)")
        print("=" * 50)
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        error_patterns = defaultdict(list)
        error_counts = Counter()
        
        for log_file in self.log_files:
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        # Parse timestamp
                        timestamp = self._extract_timestamp(line)
                        if timestamp and timestamp < cutoff_time:
                            continue
                        
                        # Check for error patterns
                        if self._is_error_line(line):
                            error_type = self._classify_error(line)
                            error_patterns[error_type].append({
                                'file': log_file.name,
                                'line': line_num,
                                'timestamp': timestamp,
                                'message': line.strip()
                            })
                            error_counts[error_type] += 1
            
            except Exception as e:
                print(f"   ⚠️  Error reading {log_file}: {e}")
        
        # Display results
        if error_counts:
            print("\n📊 Error Summary:")
            for error_type, count in error_counts.most_common():
                print(f"   {error_type}: {count} occurrences")
                
                # Show recent examples
                recent_errors = error_patterns[error_type][-3:]
                for error in recent_errors:
                    timestamp = error['timestamp'].strftime('%H:%M:%S') if error['timestamp'] else 'N/A'
                    print(f"     [{timestamp}] {error['file']}:{error['line']}")
                    print(f"       {error['message'][:100]}...")
                print()
        else:
            print("   ✅ No errors found in the specified time period")
    
    def analyze_performance(self, hours=24):
        """Analyze performance metrics from logs."""
        print(f"⚡ Performance Analysis (last {hours} hours)")
        print("=" * 50)
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        performance_data = {
            'response_times': [],
            'memory_usage': [],
            'processing_times': [],
            'model_load_times': []
        }
        
        for log_file in self.log_files:
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        timestamp = self._extract_timestamp(line)
                        if timestamp and timestamp < cutoff_time:
                            continue
                        
                        # Extract performance metrics
                        self._extract_performance_metrics(line, performance_data)
            
            except Exception as e:
                print(f"   ⚠️  Error reading {log_file}: {e}")
        
        # Display performance summary
        self._display_performance_summary(performance_data)
    
    def analyze_usage_patterns(self, hours=24):
        """Analyze usage patterns and trends."""
        print(f"📈 Usage Pattern Analysis (last {hours} hours)")
        print("=" * 50)
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        usage_data = {
            'page_views': Counter(),
            'feature_usage': Counter(),
            'user_sessions': set(),
            'hourly_activity': defaultdict(int)
        }
        
        for log_file in self.log_files:
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        timestamp = self._extract_timestamp(line)
                        if timestamp and timestamp < cutoff_time:
                            continue
                        
                        # Extract usage patterns
                        self._extract_usage_patterns(line, usage_data, timestamp)
            
            except Exception as e:
                print(f"   ⚠️  Error reading {log_file}: {e}")
        
        # Display usage summary
        self._display_usage_summary(usage_data)
    
    def check_log_health(self):
        """Check overall log health and disk usage."""
        print("🏥 Log Health Check")
        print("=" * 50)
        
        total_size = 0
        file_info = []
        
        for log_file in self.log_files:
            try:
                size = log_file.stat().st_size
                total_size += size
                modified = datetime.fromtimestamp(log_file.stat().st_mtime)
                
                file_info.append({
                    'name': log_file.name,
                    'size': size,
                    'size_mb': size / (1024 * 1024),
                    'modified': modified,
                    'age_hours': (datetime.now() - modified).total_seconds() / 3600
                })
            
            except Exception as e:
                print(f"   ⚠️  Error checking {log_file}: {e}")
        
        # Display health summary
        print(f"📁 Total log files: {len(file_info)}")
        print(f"💾 Total size: {total_size / (1024 * 1024):.2f} MB")
        print(f"📊 Average file size: {(total_size / len(file_info)) / (1024 * 1024):.2f} MB")
        
        # Check for issues
        print("\n🔍 Health Issues:")
        issues_found = False
        
        # Large files
        large_files = [f for f in file_info if f['size_mb'] > 100]
        if large_files:
            issues_found = True
            print("   ⚠️  Large log files (>100MB):")
            for f in large_files:
                print(f"     {f['name']}: {f['size_mb']:.1f} MB")
        
        # Old files
        old_files = [f for f in file_info if f['age_hours'] > 72]
        if old_files:
            issues_found = True
            print("   ⚠️  Old log files (>72 hours):")
            for f in old_files:
                print(f"     {f['name']}: {f['age_hours']:.1f} hours old")
        
        # Disk space warning
        if total_size > 1024 * 1024 * 1024:  # 1GB
            issues_found = True
            print(f"   ⚠️  High disk usage: {total_size / (1024 * 1024 * 1024):.2f} GB")
        
        if not issues_found:
            print("   ✅ No health issues detected")
    
    def tail_logs(self, lines=50, follow=False):
        """Display recent log entries."""
        print(f"📜 Recent Log Entries (last {lines} lines)")
        print("=" * 50)
        
        # Combine and sort log entries
        all_entries = []
        
        for log_file in self.log_files:
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    file_lines = f.readlines()
                    for line in file_lines[-lines:]:
                        timestamp = self._extract_timestamp(line)
                        all_entries.append({
                            'timestamp': timestamp or datetime.min,
                            'file': log_file.name,
                            'line': line.strip()
                        })
            
            except Exception as e:
                print(f"   ⚠️  Error reading {log_file}: {e}")
        
        # Sort by timestamp and display
        all_entries.sort(key=lambda x: x['timestamp'])
        for entry in all_entries[-lines:]:
            timestamp_str = entry['timestamp'].strftime('%H:%M:%S') if entry['timestamp'] != datetime.min else 'N/A'
            print(f"[{timestamp_str}] {entry['file']}: {entry['line']}")
        
        # Follow mode (simplified)
        if follow:
            print("\n👀 Following logs... (Ctrl+C to stop)")
            try:
                import time
                while True:
                    time.sleep(2)
                    # In a real implementation, you'd monitor file changes
                    # This is a placeholder
                    print(".", end="", flush=True)
            except KeyboardInterrupt:
                print("\n   Stopped following logs")
    
    def _extract_timestamp(self, line):
        """Extract timestamp from log line."""
        # Common timestamp patterns
        patterns = [
            r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})',  # YYYY-MM-DD HH:MM:SS
            r'(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})',  # MM/DD/YYYY HH:MM:SS
            r'(\d{2}:\d{2}:\d{2})',  # HH:MM:SS only
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                try:
                    timestamp_str = match.group(1)
                    if len(timestamp_str) == 8:  # HH:MM:SS only
                        # Assume today
                        today = datetime.now().strftime('%Y-%m-%d')
                        timestamp_str = f"{today} {timestamp_str}"
                    
                    # Try different formats
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S']:
                        try:
                            return datetime.strptime(timestamp_str, fmt)
                        except ValueError:
                            continue
                except:
                    pass
        
        return None
    
    def _is_error_line(self, line):
        """Check if line contains an error."""
        error_keywords = ['ERROR', 'CRITICAL', 'FATAL', 'Exception', 'Traceback', 'Failed']
        return any(keyword in line.upper() for keyword in error_keywords)
    
    def _classify_error(self, line):
        """Classify the type of error."""
        line_upper = line.upper()
        
        if 'FILENOTFOUND' in line_upper or 'NO SUCH FILE' in line_upper:
            return 'File Not Found'
        elif 'PERMISSION' in line_upper or 'ACCESS DENIED' in line_upper:
            return 'Permission Error'
        elif 'CONNECTION' in line_upper or 'TIMEOUT' in line_upper:
            return 'Network Error'
        elif 'MEMORY' in line_upper or 'OUT OF MEMORY' in line_upper:
            return 'Memory Error'
        elif 'KEYERROR' in line_upper:
            return 'Key Error'
        elif 'VALUEERROR' in line_upper:
            return 'Value Error'
        elif 'INDEXERROR' in line_upper or 'OUT OF BOUNDS' in line_upper:
            return 'Index Error'
        elif 'DATABASE' in line_upper or 'SQL' in line_upper:
            return 'Database Error'
        else:
            return 'General Error'
    
    def _extract_performance_metrics(self, line, data):
        """Extract performance metrics from log line."""
        # Response time patterns
        response_match = re.search(r'response_time[:\s]+(\d+\.?\d*)', line, re.IGNORECASE)
        if response_match:
            data['response_times'].append(float(response_match.group(1)))
        
        # Memory usage patterns
        memory_match = re.search(r'memory[:\s]+(\d+\.?\d*)\s*MB', line, re.IGNORECASE)
        if memory_match:
            data['memory_usage'].append(float(memory_match.group(1)))
        
        # Processing time patterns
        process_match = re.search(r'processing.*?(\d+\.?\d*)\s*sec', line, re.IGNORECASE)
        if process_match:
            data['processing_times'].append(float(process_match.group(1)))
        
        # Model load time patterns
        model_match = re.search(r'model.*?load.*?(\d+\.?\d*)\s*sec', line, re.IGNORECASE)
        if model_match:
            data['model_load_times'].append(float(model_match.group(1)))
    
    def _extract_usage_patterns(self, line, data, timestamp):
        """Extract usage patterns from log line."""
        # Page views
        page_match = re.search(r'page[:\s]+(\w+)', line, re.IGNORECASE)
        if page_match:
            data['page_views'][page_match.group(1)] += 1
        
        # Feature usage
        feature_keywords = ['anomaly_detection', 'visualization', 'analysis', 'export', 'model_training']
        for feature in feature_keywords:
            if feature in line.lower():
                data['feature_usage'][feature] += 1
        
        # Session tracking
        session_match = re.search(r'session[:\s]+([a-f0-9-]+)', line, re.IGNORECASE)
        if session_match:
            data['user_sessions'].add(session_match.group(1))
        
        # Hourly activity
        if timestamp:
            hour = timestamp.hour
            data['hourly_activity'][hour] += 1
    
    def _display_performance_summary(self, data):
        """Display performance analysis summary."""
        print("\n⚡ Performance Metrics:")
        
        for metric_name, values in data.items():
            if values:
                avg_val = sum(values) / len(values)
                max_val = max(values)
                min_val = min(values)
                
                print(f"   {metric_name.replace('_', ' ').title()}:")
                print(f"     Average: {avg_val:.2f}")
                print(f"     Min: {min_val:.2f}")
                print(f"     Max: {max_val:.2f}")
                print(f"     Count: {len(values)}")
                print()
    
    def _display_usage_summary(self, data):
        """Display usage analysis summary."""
        print("\n📊 Usage Statistics:")
        
        # Page views
        if data['page_views']:
            print("   Top Pages:")
            for page, count in data['page_views'].most_common(5):
                print(f"     {page}: {count} views")
            print()
        
        # Feature usage
        if data['feature_usage']:
            print("   Feature Usage:")
            for feature, count in data['feature_usage'].most_common():
                print(f"     {feature}: {count} times")
            print()
        
        # Session count
        print(f"   Unique Sessions: {len(data['user_sessions'])}")
        
        # Hourly activity
        if data['hourly_activity']:
            print("\n   Activity by Hour:")
            for hour in range(24):
                count = data['hourly_activity'].get(hour, 0)
                bar = '█' * min(count, 20)
                print(f"     {hour:02d}:00 {bar} ({count})")

def main():
    """Main function for log analysis tool."""
    parser = argparse.ArgumentParser(description='NDR Platform Log Analysis Tool')
    parser.add_argument('command', choices=['errors', 'performance', 'usage', 'health', 'tail'],
                       help='Analysis command to run')
    parser.add_argument('--hours', type=int, default=24,
                       help='Number of hours to analyze (default: 24)')
    parser.add_argument('--lines', type=int, default=50,
                       help='Number of lines to show for tail command (default: 50)')
    parser.add_argument('--follow', action='store_true',
                       help='Follow logs in real-time (tail command only)')
    parser.add_argument('--log-dir', default='logs',
                       help='Log directory path (default: logs)')
    
    args = parser.parse_args()
    
    print("🔍 NDR Platform Log Analyzer")
    print("=" * 50)
    
    # Change to project directory
    os.chdir(Path(__file__).parent.parent)
    
    analyzer = LogAnalyzer(args.log_dir)
    
    if not analyzer.log_files:
        print("❌ No log files found")
        return
    
    print(f"📁 Found {len(analyzer.log_files)} log files")
    print()
    
    try:
        if args.command == 'errors':
            analyzer.analyze_errors(args.hours)
        elif args.command == 'performance':
            analyzer.analyze_performance(args.hours)
        elif args.command == 'usage':
            analyzer.analyze_usage_patterns(args.hours)
        elif args.command == 'health':
            analyzer.check_log_health()
        elif args.command == 'tail':
            analyzer.tail_logs(args.lines, args.follow)
            
    except Exception as e:
        print(f"❌ Analysis error: {e}")

if __name__ == "__main__":
    main()
