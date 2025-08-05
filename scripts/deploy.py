#!/usr/bin/env python3
"""
NDR Platform Deployment Script
Automated deployment for development, staging, and production environments.
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

def deploy_development():
    """Deploy for development environment."""
    print("🚀 Deploying for Development Environment...")
    
    # Check prerequisites
    if not check_prerequisites():
        return False
    
    # Setup environment
    setup_dev_environment()
    
    # Install dependencies
    install_dependencies()
    
    # Setup data directories
    setup_directories()
    
    # Start development server
    print("\n   🌐 Starting development server...")
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'run.py',
            '--server.port', '8501',
            '--server.headless', 'false'
        ], check=True)
    except KeyboardInterrupt:
        print("\n   ⏹️  Development server stopped")
    
    return True

def deploy_docker():
    """Deploy using Docker containers."""
    print("🐳 Deploying with Docker...")
    
    # Check Docker
    if not check_docker():
        return False
    
    # Build and start containers
    try:
        print("   🔨 Building Docker images...")
        subprocess.run(['docker-compose', 'build'], cwd='deployment', check=True)
        
        print("   🚀 Starting containers...")
        subprocess.run(['docker-compose', 'up', '-d'], cwd='deployment', check=True)
        
        print("   ✅ Docker deployment complete!")
        print("   🌐 Application available at: http://localhost:8501")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Docker deployment failed: {e}")
        return False

def deploy_production():
    """Deploy for production environment."""
    print("🏭 Deploying for Production Environment...")
    
    # Check prerequisites
    if not check_prerequisites():
        return False
    
    # Production checks
    if not run_production_checks():
        return False
    
    # Backup current deployment
    backup_current_deployment()
    
    # Deploy new version
    try:
        # Setup production environment
        setup_prod_environment()
        
        # Install dependencies
        install_dependencies(production=True)
        
        # Setup directories with proper permissions
        setup_directories(production=True)
        
        # Run database migrations if needed
        run_migrations()
        
        # Start production services
        start_production_services()
        
        # Health check
        if not health_check():
            print("   ❌ Health check failed, rolling back...")
            rollback_deployment()
            return False
        
        print("   ✅ Production deployment complete!")
        return True
        
    except Exception as e:
        print(f"   ❌ Production deployment failed: {e}")
        rollback_deployment()
        return False

def check_prerequisites():
    """Check deployment prerequisites."""
    print("   🔍 Checking prerequisites...")
    
    # Check Python version
    if sys.version_info < (3, 11):
        print(f"   ❌ Python 3.11+ required, found {sys.version}")
        return False
    print(f"   ✅ Python {sys.version.split()[0]}")
    
    # Check required files
    required_files = [
        'requirements.txt',
        'run.py',
        'app/main.py',
        'core/data_manager.py'
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"   ❌ Required file missing: {file_path}")
            return False
    print("   ✅ Required files present")
    
    return True

def check_docker():
    """Check Docker installation and status."""
    print("   🐳 Checking Docker...")
    
    try:
        # Check Docker
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"   ✅ {result.stdout.strip()}")
        
        # Check Docker Compose
        result = subprocess.run(['docker-compose', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"   ✅ {result.stdout.strip()}")
        
        # Check if Docker is running
        subprocess.run(['docker', 'info'], 
                      capture_output=True, check=True)
        print("   ✅ Docker daemon running")
        
        return True
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("   ❌ Docker not available or not running")
        return False

def setup_dev_environment():
    """Setup development environment."""
    print("   🔧 Setting up development environment...")
    
    # Create .env for development
    env_file = Path('.env')
    if not env_file.exists():
        env_content = """# Development Environment
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
STREAMLIT_SERVER_HEADLESS=false
DATA_DIRECTORY=data/
"""
        env_file.write_text(env_content)
        print("   ✅ Created development .env file")

def setup_prod_environment():
    """Setup production environment."""
    print("   🏭 Setting up production environment...")
    
    # Create production .env
    env_file = Path('.env')
    env_content = """# Production Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
STREAMLIT_SERVER_HEADLESS=true
DATA_DIRECTORY=/opt/arkime/json
ALERT_THRESHOLD=0.9
"""
    env_file.write_text(env_content)
    print("   ✅ Created production .env file")

def install_dependencies(production=False):
    """Install Python dependencies."""
    print("   📦 Installing dependencies...")
    
    try:
        cmd = [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt']
        if production:
            cmd.extend(['--no-dev', '--only=main'])
        
        subprocess.run(cmd, check=True)
        print("   ✅ Dependencies installed")
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to install dependencies: {e}")
        raise

def setup_directories(production=False):
    """Setup required directories."""
    print("   📁 Setting up directories...")
    
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
    
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set production permissions
        if production and os.name != 'nt':  # Not Windows
            os.chmod(dir_path, 0o755)
        
        # Create .gitkeep
        (dir_path / '.gitkeep').touch()
    
    print(f"   ✅ Created {len(directories)} directories")

def run_production_checks():
    """Run production readiness checks."""
    print("   🔒 Running production checks...")
    
    checks = []
    
    # Check environment variables
    required_env = ['DATA_DIRECTORY', 'LOG_LEVEL']
    for env_var in required_env:
        if env_var in os.environ:
            checks.append(f"✅ {env_var}")
        else:
            checks.append(f"❌ {env_var} missing")
    
    # Check data directory permissions
    data_dir = Path(os.getenv('DATA_DIRECTORY', 'data/'))
    if data_dir.exists() and os.access(data_dir, os.R_OK):
        checks.append("✅ Data directory accessible")
    else:
        checks.append("❌ Data directory not accessible")
    
    # Check disk space
    import shutil
    total, used, free = shutil.disk_usage('.')
    free_gb = free // (1024**3)
    if free_gb >= 10:  # At least 10GB free
        checks.append(f"✅ Disk space: {free_gb}GB free")
    else:
        checks.append(f"❌ Low disk space: {free_gb}GB free")
    
    for check in checks:
        print(f"   {check}")
    
    failed_checks = [c for c in checks if c.startswith("❌")]
    if failed_checks:
        print(f"   ❌ {len(failed_checks)} production checks failed")
        return False
    
    print("   ✅ All production checks passed")
    return True

def backup_current_deployment():
    """Backup current deployment before update."""
    print("   💾 Creating deployment backup...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f'deployment_backup_{timestamp}')
    backup_dir.mkdir(exist_ok=True)
    
    # Backup key directories
    backup_items = ['models/', 'config/', 'logs/']
    for item in backup_items:
        src_path = Path(item)
        if src_path.exists():
            dst_path = backup_dir / item
            if src_path.is_file():
                shutil.copy2(src_path, dst_path)
            else:
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    
    print(f"   ✅ Backup created: {backup_dir}")

def run_migrations():
    """Run database migrations or data updates."""
    print("   🔄 Running migrations...")
    
    # Check if migrations are needed
    # This could check version files, database schema, etc.
    
    # Example: Update configuration format
    config_file = Path('config/config.yaml')
    if config_file.exists():
        print("   ✅ Configuration up to date")
    else:
        print("   ℹ️  No migrations needed")

def start_production_services():
    """Start production services."""
    print("   🚀 Starting production services...")
    
    try:
        # Use production configuration
        cmd = [
            sys.executable, '-m', 'streamlit', 'run', 'run.py',
            '--server.port', '8501',
            '--server.address', '0.0.0.0',
            '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false'
        ]
        
        # Start as background process
        process = subprocess.Popen(cmd)
        
        # Give it time to start
        import time
        time.sleep(10)
        
        if process.poll() is None:  # Still running
            print("   ✅ Production services started")
            return True
        else:
            print("   ❌ Services failed to start")
            return False
            
    except Exception as e:
        print(f"   ❌ Failed to start services: {e}")
        return False

def health_check():
    """Perform post-deployment health check."""
    print("   🏥 Running health check...")
    
    import time
    import requests
    
    # Wait for service to be ready
    max_attempts = 12  # 60 seconds
    for attempt in range(max_attempts):
        try:
            response = requests.get('http://localhost:8501/_stcore/health', timeout=5)
            if response.status_code == 200:
                print("   ✅ Health check passed")
                return True
        except requests.exceptions.RequestException:
            pass
        
        print(f"   ⏳ Waiting for service... ({attempt + 1}/{max_attempts})")
        time.sleep(5)
    
    print("   ❌ Health check failed")
    return False

def rollback_deployment():
    """Rollback to previous deployment."""
    print("   🔄 Rolling back deployment...")
    
    # Find most recent backup
    backup_dirs = [d for d in Path('.').glob('deployment_backup_*') if d.is_dir()]
    if not backup_dirs:
        print("   ❌ No backup found for rollback")
        return
    
    latest_backup = max(backup_dirs, key=lambda d: d.stat().st_mtime)
    print(f"   📂 Restoring from: {latest_backup}")
    
    # Restore from backup
    for item in latest_backup.iterdir():
        dst_path = Path(item.name)
        if dst_path.exists():
            if dst_path.is_file():
                dst_path.unlink()
            else:
                shutil.rmtree(dst_path)
        
        if item.is_file():
            shutil.copy2(item, dst_path)
        else:
            shutil.copytree(item, dst_path)
    
    print("   ✅ Rollback completed")

def stop_services():
    """Stop running services."""
    print("🛑 Stopping Services...")
    
    try:
        # Stop Docker containers
        subprocess.run(['docker-compose', 'down'], cwd='deployment', check=False)
        print("   ✅ Docker containers stopped")
    except:
        pass
    
    # Stop any running Streamlit processes
    if os.name != 'nt':  # Unix/Linux
        try:
            subprocess.run(['pkill', '-f', 'streamlit'], check=False)
            print("   ✅ Streamlit processes stopped")
        except:
            pass

def main():
    """Main deployment function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='NDR Platform Deployment Tool')
    parser.add_argument('environment', choices=['dev', 'docker', 'production', 'stop'],
                       help='Deployment environment')
    parser.add_argument('--force', action='store_true',
                       help='Force deployment without checks')
    
    args = parser.parse_args()
    
    print("🚀 NDR Platform Deployment Tool")
    print("=" * 50)
    
    # Change to project directory
    os.chdir(Path(__file__).parent.parent)
    
    try:
        if args.environment == 'dev':
            success = deploy_development()
        elif args.environment == 'docker':
            success = deploy_docker()
        elif args.environment == 'production':
            success = deploy_production()
        elif args.environment == 'stop':
            stop_services()
            success = True
        
        if success:
            print("\n🎉 Deployment completed successfully!")
        else:
            print("\n❌ Deployment failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Deployment error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
