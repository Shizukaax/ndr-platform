#!/usr/bin/env python3
"""
NDR Platform Health Check Script
Comprehensive health monitoring for all platform components.
"""

import os
import sys
import json
import time
import requests
import subprocess
from pathlib import Path
from datetime import datetime

def check_system_resources():
    """Check system resource usage."""
    print("💻 System Resources:")
    
    try:
        import psutil
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"   CPU Usage: {cpu_percent}%")
        
        # Memory usage
        memory = psutil.virtual_memory()
        print(f"   Memory Usage: {memory.percent}% ({memory.used // 1024**3}GB / {memory.total // 1024**3}GB)")
        
        # Disk usage
        disk = psutil.disk_usage('.')
        print(f"   Disk Usage: {disk.percent}% ({disk.used // 1024**3}GB / {disk.total // 1024**3}GB)")
        
        return True
    except ImportError:
        print("   ⚠️  psutil not installed - install with: pip install psutil")
        return False

def check_application_health():
    """Check if the Streamlit application is running and healthy."""
    print("\n🌐 Application Health:")
    
    health_endpoints = [
        "http://localhost:8501/_stcore/health",
        "http://localhost:8501"
    ]
    
    for endpoint in health_endpoints:
        try:
            response = requests.get(endpoint, timeout=5)
            if response.status_code == 200:
                print(f"   ✅ {endpoint} - Healthy")
                return True
            else:
                print(f"   ❌ {endpoint} - Status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"   ❌ {endpoint} - Error: {e}")
    
    print("   ⚠️  Application may not be running")
    return False

def check_data_sources():
    """Check data source availability and integrity."""
    print("\n📁 Data Sources:")
    
    data_dir = Path("data")
    if not data_dir.exists():
        print("   ❌ Data directory not found")
        return False
    
    # Count data files
    json_files = list(data_dir.glob("*.json"))
    pcap_files = list(data_dir.glob("*.pcap"))
    
    print(f"   JSON files: {len(json_files)}")
    print(f"   PCAP files: {len(pcap_files)}")
    
    if len(json_files) == 0 and len(pcap_files) == 0:
        print("   ⚠️  No data files found")
        return False
    
    # Check sample data file
    sample_file = data_dir / "examples" / "sample_network_data.json"
    if sample_file.exists():
        try:
            with open(sample_file) as f:
                data = json.load(f)
            print(f"   ✅ Sample data: {len(data)} records")
        except json.JSONDecodeError:
            print("   ❌ Sample data file corrupted")
            return False
    
    return True

def check_models():
    """Check trained models and their status."""
    print("\n🤖 Machine Learning Models:")
    
    models_dir = Path("models")
    if not models_dir.exists():
        print("   ❌ Models directory not found")
        return False
    
    model_files = list(models_dir.glob("*.pkl"))
    metadata_files = list(models_dir.glob("*_metadata.json"))
    
    print(f"   Trained models: {len(model_files)}")
    print(f"   Metadata files: {len(metadata_files)}")
    
    # Check for recent models (last 7 days)
    recent_models = []
    for model_file in model_files:
        model_age = time.time() - model_file.stat().st_mtime
        if model_age < 7 * 24 * 3600:  # 7 days
            recent_models.append(model_file)
    
    print(f"   Recent models (7 days): {len(recent_models)}")
    
    if len(model_files) == 0:
        print("   ⚠️  No trained models found")
        return False
    
    return True

def check_logs():
    """Check log files and recent activity."""
    print("\n📝 Log Files:")
    
    logs_dir = Path("logs")
    if not logs_dir.exists():
        print("   ❌ Logs directory not found")
        return False
    
    log_files = list(logs_dir.glob("*.log"))
    print(f"   Log files: {len(log_files)}")
    
    # Check for recent log activity
    recent_activity = False
    for log_file in log_files:
        if log_file.stat().st_size > 0:
            # Check if modified in last hour
            log_age = time.time() - log_file.stat().st_mtime
            if log_age < 3600:  # 1 hour
                recent_activity = True
                print(f"   ✅ Recent activity in: {log_file.name}")
    
    if not recent_activity:
        print("   ⚠️  No recent log activity")
    
    # Check for errors in recent logs
    error_count = 0
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()[-100:]  # Last 100 lines
                for line in lines:
                    if 'ERROR' in line.upper():
                        error_count += 1
        except Exception:
            continue
    
    if error_count > 0:
        print(f"   ⚠️  Found {error_count} errors in recent logs")
    else:
        print("   ✅ No recent errors found")
    
    return True

def check_docker_containers():
    """Check Docker container status if running in Docker."""
    print("\n🐳 Docker Containers:")
    
    try:
        result = subprocess.run(['docker', 'ps', '--format', 'table {{.Names}}\t{{.Status}}'], 
                              capture_output=True, text=True, check=True)
        
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:  # Header + containers
            print("   Running containers:")
            for line in lines[1:]:  # Skip header
                if 'ndr' in line.lower():
                    print(f"   ✅ {line}")
        else:
            print("   ℹ️  No Docker containers running")
        
        return True
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("   ℹ️  Docker not available or not running")
        return True

def check_configuration():
    """Check configuration files."""
    print("\n⚙️ Configuration:")
    
    config_files = [
        '.env',
        'config/config.yaml',
        'guides/deployment/docker-compose.yml',
        'requirements.txt'
    ]
    
    all_good = True
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"   ✅ {config_file}")
        else:
            print(f"   ❌ {config_file} missing")
            all_good = False
    
    return all_good

def check_network_connectivity():
    """Check external network connectivity."""
    print("\n🌍 Network Connectivity:")
    
    test_urls = [
        "https://api.github.com",
        "https://pypi.org", 
        "https://hub.docker.com"
    ]
    
    connected = True
    for url in test_urls:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"   ✅ {url}")
            else:
                print(f"   ⚠️  {url} - Status: {response.status_code}")
        except requests.exceptions.RequestException:
            print(f"   ❌ {url} - Connection failed")
            connected = False
    
    return connected

def generate_health_report():
    """Generate a comprehensive health report."""
    print("\n📊 Generating Health Report...")
    
    timestamp = datetime.now().isoformat()
    
    # Run all checks
    checks = {
        'system_resources': check_system_resources(),
        'application_health': check_application_health(),
        'data_sources': check_data_sources(),
        'models': check_models(),
        'logs': check_logs(),
        'docker_containers': check_docker_containers(),
        'configuration': check_configuration(),
        'network_connectivity': check_network_connectivity()
    }
    
    # Calculate overall health score
    passed_checks = sum(checks.values())
    total_checks = len(checks)
    health_score = (passed_checks / total_checks) * 100
    
    # Create health report
    report = {
        'timestamp': timestamp,
        'health_score': health_score,
        'checks': checks,
        'summary': {
            'passed': passed_checks,
            'total': total_checks,
            'status': 'healthy' if health_score >= 80 else 'degraded' if health_score >= 60 else 'critical'
        }
    }
    
    # Save report
    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)
    
    report_file = reports_dir / f'health_check_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"   ✅ Report saved: {report_file}")
    
    return report

def main():
    """Main health check function."""
    print("🏥 NDR Platform Health Check")
    print("=" * 50)
    
    # Change to project directory
    os.chdir(Path(__file__).parent.parent)
    
    try:
        report = generate_health_report()
        
        print("\n" + "=" * 50)
        print("📋 Health Summary:")
        print(f"   Overall Health: {report['health_score']:.1f}%")
        print(f"   Status: {report['summary']['status'].upper()}")
        print(f"   Checks Passed: {report['summary']['passed']}/{report['summary']['total']}")
        
        if report['health_score'] >= 80:
            print("\n🎉 Platform is healthy!")
        elif report['health_score'] >= 60:
            print("\n⚠️  Platform has some issues but is operational")
        else:
            print("\n🚨 Platform has critical issues!")
            
    except Exception as e:
        print(f"\n❌ Health check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
