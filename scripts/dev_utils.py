#!/usr/bin/env python3
"""
NDR Platform Development Utilities
Helper scripts for development, testing, and debugging.
"""

import os
import sys
import json
import time
import shutil
import requests
import subprocess
from pathlib import Path
from datetime import datetime

def start_dev_server():
    """Start the development server with hot reloading."""
    print("🚀 Starting NDR Platform Development Server...")
    
    # Check if already running
    try:
        response = requests.get('http://localhost:8501/_stcore/health', timeout=2)
        if response.status_code == 200:
            print("   ⚠️  Server already running at http://localhost:8501")
            return
    except:
        pass
    
    # Start Streamlit server
    try:
        cmd = [sys.executable, '-m', 'streamlit', 'run', 'run.py', 
               '--server.address', '0.0.0.0',
               '--server.port', '8501',
               '--server.headless', 'false',
               '--browser.gatherUsageStats', 'false',
               '--server.fileWatcherType', 'auto']
        
        print("   🌐 Starting Streamlit server...")
        print("   📍 URL: http://localhost:8501")
        
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n   ⏹️  Server stopped")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to start server: {e}")

def run_tests():
    """Run the test suite."""
    print("🧪 Running Test Suite...")
    
    test_commands = [
        ['python', '-m', 'pytest', 'tests/', '-v'],
        ['python', '-m', 'pytest', 'tests/', '--cov=app', '--cov=core'],
        ['python', '-c', 'import app.main; print("✅ App imports successfully")'],
        ['python', '-c', 'import core.data_manager; print("✅ Core modules import successfully")']
    ]
    
    for cmd in test_commands:
        try:
            print(f"\n   Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"   ✅ Success")
            if result.stdout:
                print(f"   Output: {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed: {e}")
            if e.stdout:
                print(f"   Output: {e.stdout}")
            if e.stderr:
                print(f"   Error: {e.stderr}")

def lint_code():
    """Run code linting and formatting checks."""
    print("🔍 Running Code Quality Checks...")
    
    # Python files to check
    python_paths = ['app/', 'core/', 'scripts/', 'tests/']
    
    lint_commands = [
        ['python', '-m', 'flake8', '--max-line-length=100'] + python_paths,
        ['python', '-m', 'black', '--check', '--diff'] + python_paths,
        ['python', '-m', 'isort', '--check-only', '--diff'] + python_paths
    ]
    
    for cmd in lint_commands:
        try:
            print(f"\n   Running: {cmd[2]} {cmd[3] if len(cmd) > 3 else ''}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"   ✅ Passed")
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️  Issues found")
            if e.stdout:
                print(f"   Output: {e.stdout[:500]}...")
        except FileNotFoundError:
            print(f"   ⚠️  {cmd[2]} not installed")

def format_code():
    """Auto-format code using black and isort."""
    print("✨ Formatting Code...")
    
    python_paths = ['app/', 'core/', 'scripts/', 'tests/']
    
    format_commands = [
        ['python', '-m', 'black'] + python_paths,
        ['python', '-m', 'isort'] + python_paths
    ]
    
    for cmd in format_commands:
        try:
            print(f"\n   Running: {cmd[2]}")
            subprocess.run(cmd, check=True)
            print(f"   ✅ Formatted")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed: {e}")
        except FileNotFoundError:
            print(f"   ⚠️  {cmd[2]} not installed")

def generate_sample_data(count=1000):
    """Generate sample network data for testing."""
    print(f"📊 Generating {count} sample network records...")
    
    import random
    from datetime import datetime, timedelta
    
    # Sample data templates
    protocols = ['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS', 'DNS']
    internal_ips = [f"192.168.1.{i}" for i in range(1, 255)]
    external_ips = [
        '8.8.8.8', '1.1.1.1', '185.199.108.133', '151.101.193.140',
        '172.217.164.142', '13.107.42.14', '23.185.0.2'
    ]
    
    sample_data = []
    base_time = datetime.now() - timedelta(hours=24)
    
    for i in range(count):
        timestamp = base_time + timedelta(seconds=random.randint(0, 86400))
        
        record = {
            'timestamp': timestamp.isoformat() + 'Z',
            'src_ip': random.choice(internal_ips),
            'dst_ip': random.choice(external_ips) if random.random() > 0.3 else random.choice(internal_ips),
            'src_port': random.randint(1024, 65535),
            'dst_port': random.choice([80, 443, 53, 22, 3389, 8080, 8443]),
            'protocol': random.choice(protocols),
            'bytes': random.randint(64, 65536),
            'packets': random.randint(1, 100),
            'duration': random.randint(1, 3600)
        }
        
        # Add some anomalous records
        if random.random() < 0.05:  # 5% anomalies
            record['bytes'] = random.randint(100000, 1000000)  # Large transfer
            record['packets'] = random.randint(1000, 10000)   # Many packets
            record['dst_port'] = random.randint(1, 1023)      # Unusual port
    
        sample_data.append(record)
    
    # Save to file
    data_dir = Path('data/examples')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = data_dir / f'sample_data_{count}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"   ✅ Sample data saved: {output_file}")
    print(f"   📈 Records: {len(sample_data)}")
    print(f"   🎯 Anomalies: ~{int(count * 0.05)}")

def profile_performance():
    """Profile application performance."""
    print("⚡ Profiling Application Performance...")
    
    try:
        import cProfile
        import pstats
        from io import StringIO
        
        # Profile imports
        pr = cProfile.Profile()
        pr.enable()
        
        # Import main modules (simulates startup)
        import app.main
        import core.data_manager
        import core.model_manager
        
        pr.disable()
        
        # Generate report
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        profile_output = s.getvalue()
        print("   📊 Performance Profile (Top 20 functions):")
        print(profile_output)
        
        # Save detailed report
        reports_dir = Path('reports')
        reports_dir.mkdir(exist_ok=True)
        
        profile_file = reports_dir / f'performance_profile_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(profile_file, 'w') as f:
            f.write(profile_output)
        
        print(f"   ✅ Detailed profile saved: {profile_file}")
        
    except ImportError:
        print("   ⚠️  cProfile not available")

def monitor_resources(duration=60):
    """Monitor system resources during operation."""
    print(f"📈 Monitoring Resources for {duration} seconds...")
    
    try:
        import psutil
        
        # Start monitoring
        start_time = time.time()
        measurements = []
        
        while time.time() - start_time < duration:
            measurement = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_used_gb': psutil.virtual_memory().used / (1024**3),
                'disk_percent': psutil.disk_usage('.').percent
            }
            measurements.append(measurement)
            print(f"   📊 CPU: {measurement['cpu_percent']:5.1f}% | "
                  f"RAM: {measurement['memory_percent']:5.1f}% ({measurement['memory_used_gb']:.1f}GB) | "
                  f"Disk: {measurement['disk_percent']:5.1f}%")
            
            time.sleep(5)
        
        # Calculate averages
        avg_cpu = sum(m['cpu_percent'] for m in measurements) / len(measurements)
        avg_memory = sum(m['memory_percent'] for m in measurements) / len(measurements)
        
        print(f"\n   📊 Averages over {duration}s:")
        print(f"   CPU: {avg_cpu:.1f}%")
        print(f"   Memory: {avg_memory:.1f}%")
        
        # Save detailed measurements
        reports_dir = Path('reports')
        reports_dir.mkdir(exist_ok=True)
        
        monitor_file = reports_dir / f'resource_monitor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(monitor_file, 'w') as f:
            json.dump({
                'duration': duration,
                'measurements': measurements,
                'averages': {'cpu': avg_cpu, 'memory': avg_memory}
            }, f, indent=2)
        
        print(f"   ✅ Monitoring data saved: {monitor_file}")
        
    except ImportError:
        print("   ⚠️  psutil not installed - install with: pip install psutil")

def reset_platform():
    """Reset platform to clean state (for development)."""
    print("🔄 Resetting Platform to Clean State...")
    print("⚠️  This will delete all data, models, and results!")
    
    response = input("Are you sure? (yes/no): ")
    if response.lower() != 'yes':
        print("   ❌ Reset cancelled")
        return
    
    # Directories to clean
    clean_dirs = [
        'cache/',
        'logs/',
        'models/',
        'reports/',
        'results/',
        'feedback/'
    ]
    
    # Files to clean  
    clean_files = [
        'data/*.json',
        'data/*.pcap'
    ]
    
    for clean_dir in clean_dirs:
        dir_path = Path(clean_dir)
        if dir_path.exists():
            shutil.rmtree(dir_path)
            dir_path.mkdir()
            (dir_path / '.gitkeep').touch()
            print(f"   🧹 Cleaned: {clean_dir}")
    
    import glob
    for pattern in clean_files:
        for file_path in glob.glob(pattern):
            Path(file_path).unlink()
            print(f"   🗑️  Deleted: {file_path}")
    
    print("   ✅ Platform reset complete")

def main():
    """Main development utilities function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='NDR Platform Development Utilities')
    parser.add_argument('action', choices=[
        'serve', 'test', 'lint', 'format', 'sample-data', 
        'profile', 'monitor', 'reset'
    ], help='Action to perform')
    parser.add_argument('--count', type=int, default=1000, 
                       help='Number of sample records to generate')
    parser.add_argument('--duration', type=int, default=60,
                       help='Duration for monitoring (seconds)')
    
    args = parser.parse_args()
    
    print("🛠️  NDR Platform Development Utilities")
    print("=" * 50)
    
    # Change to project directory
    os.chdir(Path(__file__).parent.parent)
    
    try:
        if args.action == 'serve':
            start_dev_server()
        elif args.action == 'test':
            run_tests()
        elif args.action == 'lint':
            lint_code()
        elif args.action == 'format':
            format_code()
        elif args.action == 'sample-data':
            generate_sample_data(args.count)
        elif args.action == 'profile':
            profile_performance()
        elif args.action == 'monitor':
            monitor_resources(args.duration)
        elif args.action == 'reset':
            reset_platform()
            
    except Exception as e:
        print(f"❌ Operation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
