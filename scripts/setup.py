#!/usr/bin/env python3
"""
NDR Platform Setup Script
Automates initial platform setup and configuration.
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

def create_directories():
    """Create necessary directories for the platform."""
    directories = [
        'data/examples',
        'data/realtime',
        'logs',
        'models/backups',
        'reports',
        'results',
        'feedback',
        'cache'
    ]
    
    print("📁 Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {directory}")

def setup_github_templates():
    """Setup GitHub templates and workflows if they don't exist."""
    print("\n🐙 Setting up GitHub configuration...")
    
    github_dir = Path('.github')
    if github_dir.exists():
        print("   ✅ GitHub directory already exists")
    else:
        print("   ℹ️  Run the platform setup to create GitHub templates")

def setup_deployment_structure():
    """Setup deployment directory structure."""
    print("\n🚀 Setting up deployment structure...")
    
    deployment_dir = Path('deployment')
    if deployment_dir.exists():
        print("   ✅ Deployment directory already exists")
        
        # Check for key deployment files
        docker_files = ['Dockerfile', 'docker-compose.yml', 'nginx.conf']
        for file_name in docker_files:
            file_path = deployment_dir / file_name
            if file_path.exists():
                print(f"   ✅ Found: {file_name}")
            else:
                print(f"   ⚠️  Missing: {file_name}")
    else:
        print("   ℹ️  Deployment directory will be created automatically")

def setup_environment():
    """Setup environment file from template."""
    print("\n🔧 Setting up environment configuration...")
    
    env_example = Path('.env.example')
    env_file = Path('.env')
    
    if env_example.exists() and not env_file.exists():
        shutil.copy(env_example, env_file)
        print("   ✅ Created .env from .env.example")
        print("   ⚠️  Please edit .env with your specific configuration")
    elif env_file.exists():
        print("   ℹ️  .env file already exists")
    else:
        # Create basic .env file
        env_content = """# NDR Platform Environment Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
DATA_DIRECTORY=data/
LOG_LEVEL=INFO
DEFAULT_MODEL=IsolationForest
ALERT_THRESHOLD=0.8
"""
        env_file.write_text(env_content)
        print("   ✅ Created basic .env file")

def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n🔍 Checking dependencies...")
    
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'scikit-learn',
        'plotly',
        'pyyaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Run: pip install -r requirements.txt")
        return False
    else:
        print("\n🎉 All dependencies satisfied!")
        return True

def download_sample_data():
    """Download or create sample data for testing."""
    print("\n📦 Setting up sample data...")
    
    sample_data_dir = Path('data/examples')
    sample_file = sample_data_dir / 'sample_network_data.json'
    
    if not sample_file.exists():
        # Create sample network data
        sample_data = [
            {
                "timestamp": "2025-08-04T10:00:00Z",
                "src_ip": "192.168.1.100",
                "dst_ip": "8.8.8.8",
                "src_port": 54321,
                "dst_port": 53,
                "protocol": "UDP",
                "bytes": 64,
                "packets": 1
            },
            {
                "timestamp": "2025-08-04T10:01:00Z", 
                "src_ip": "192.168.1.100",
                "dst_ip": "185.199.108.133",
                "src_port": 54322,
                "dst_port": 443,
                "protocol": "TCP",
                "bytes": 1500,
                "packets": 10
            }
        ]
        
        with open(sample_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        print(f"   ✅ Created sample data: {sample_file}")
    else:
        print(f"   ℹ️  Sample data already exists: {sample_file}")

def setup_docker():
    """Check Docker setup and create development compose file."""
    print("\n🐳 Checking Docker setup...")
    
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"   ✅ {result.stdout.strip()}")
        
        # Check docker-compose
        result = subprocess.run(['docker-compose', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"   ✅ {result.stdout.strip()}")
        
        # Check if deployment files exist
        deployment_dir = Path('deployment')
        if deployment_dir.exists():
            compose_file = deployment_dir / 'docker-compose.yml'
            if compose_file.exists():
                print("   ✅ Docker Compose file found in deployment/")
            else:
                print("   ⚠️  Docker Compose file not found in deployment/")
        
        return True
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("   ❌ Docker not found or not running")
        print("   Install Docker Desktop: https://www.docker.com/products/docker-desktop")
        return False

def main():
    """Main setup function."""
    print("🛠️  NDR Platform Setup")
    print("=" * 50)
    
    # Change to project directory
    os.chdir(Path(__file__).parent.parent)
    
    try:
        create_directories()
        setup_github_templates()
        setup_deployment_structure()
        setup_environment()
        deps_ok = check_dependencies()
        download_sample_data()
        docker_ok = setup_docker()
        
        print("\n" + "=" * 50)
        print("📋 Setup Summary:")
        print(f"   Dependencies: {'✅' if deps_ok else '❌'}")
        print(f"   Docker: {'✅' if docker_ok else '❌'}")
        print("\n🎉 NDR Platform setup complete!")
        
        if deps_ok:
            print("\n🚀 You can now run the platform:")
            print("   streamlit run run.py")
        else:
            print("\n⚠️  Install missing dependencies first:")
            print("   pip install -r requirements.txt")
            
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
