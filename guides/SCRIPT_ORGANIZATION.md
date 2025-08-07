# 🛠️ NDR Platform v2.1.0 - Script Organization Guide

## 🚀 **Production-Ready Scripts** *(August 2025)*

Following recent critical fixes, all management scripts have been updated to work with the enhanced NDR Platform v2.1.0 configuration system.

---

## 📋 **Script Directory Structure**

```
📦 scripts/
├── 📄 backup.py                 # ✅ Data backup utilities
├── 📄 data_manager.py           # ✅ Data processing operations
├── 📄 dev_utils.py              # Development utilities
├── 📄 health_check.py           # ✅ System monitoring
├── 📄 log_analyzer.py           # Log analysis tools
├── 📄 migrate_anomaly_storage.py # Data migration
├── 📄 model_manager.py          # ✅ ML model operations
├── 📄 security_scanner.py       # Security validation
├── 📄 verify_structure.py       # ✅ Directory validation
├── 📄 README.md                 # Script documentation
├── 📁 linux/                    # Linux deployment scripts
│   ├── 📄 setup.sh              # Linux environment setup
│   └── 📄 deploy.sh             # Linux deployment
└── 📁 windows/                  # Windows deployment scripts
    ├── 📄 setup.bat             # Windows environment setup
    └── 📄 deploy.bat            # Windows deployment
```

---

## 🚀 **Quick Start Guide**

### **🪟 Windows Deployment**

```batch
@echo off
REM NDR Platform v2.1.0 Windows Setup

echo ===================================
echo NDR Platform v2.1.0 Setup
echo ===================================

REM Step 1: Environment Setup (One-time)
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

REM Step 2: ✅ Create directories using config
echo Creating required directories...
python run.py --create-dirs-only

echo Setup complete! Run deploy.bat to start the platform.
pause
```

### **� Linux/macOS Deployment**

```bash
#!/bin/bash
# NDR Platform v2.1.0 Linux Setup

echo "==================================="
echo "NDR Platform v2.1.0 Setup"  
echo "==================================="

# Step 1: Environment Setup
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

# Step 2: ✅ Create directories using config
echo "Creating required directories..."
python run.py --create-dirs-only

echo "Setup complete! Run ./deploy.sh to start the platform."
```

---

## 🔧 **Core Management Scripts**

### **📊 Health Check Script** *(Enhanced)*

```python
# scripts/health_check.py - Enhanced for v2.1.0
import os
import sys
import requests
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from core.config_loader import load_config

def comprehensive_health_check():
    """Enhanced health check for NDR Platform v2.1.0"""
    
    print("🔍 NDR Platform v2.1.0 Health Check")
    print("=" * 50)
    
    results = {}
    
    # ✅ Configuration validation
    try:
        config = load_config()
        results['config'] = True
        print("✅ Configuration: VALID")
    except Exception as e:
        results['config'] = False
        print(f"❌ Configuration: FAILED ({e})")
    
    # ✅ Directory structure validation
    config = load_config()
    required_dirs = [
        config.get('system.data_dir', 'data'),
        config.get('system.results_dir', 'data/results'),
        config.get('feedback.storage_dir', 'data/feedback'),
        config.get('system.models_dir', 'models'),
        config.get('system.logs_dir', 'logs')
    ]
    
    dir_status = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ Directory: {dir_path}")
        else:
            print(f"❌ Directory missing: {dir_path}")
            dir_status = False
    
    results['directories'] = dir_status
    
    # ✅ Application health check
    try:
        response = requests.get('http://localhost:8501/_stcore/health', timeout=10)
        if response.status_code == 200:
            results['app'] = True
            print("✅ Application: RUNNING")
        else:
            results['app'] = False
            print("❌ Application: NOT RESPONDING")
    except:
        results['app'] = False
        print("❌ Application: NOT ACCESSIBLE")
    
    # ✅ Model availability check
    models_dir = config.get('system.models_dir', 'models')
    model_files = list(Path(models_dir).glob('*.pkl')) if os.path.exists(models_dir) else []
    
    if model_files:
        results['models'] = True
        print(f"✅ Models: {len(model_files)} available")
    else:
        results['models'] = False
        print("❌ Models: None found")
    
    # Summary
    print("\n" + "=" * 50)
    overall_health = all(results.values())
    status = "✅ HEALTHY" if overall_health else "❌ ISSUES DETECTED"
    print(f"Overall Status: {status}")
    
    return overall_health, results

if __name__ == "__main__":
    healthy, details = comprehensive_health_check()
    sys.exit(0 if healthy else 1)
```

### **💾 Backup Script** *(Enhanced)*

```python
# scripts/backup.py - Enhanced for v2.1.0
import os
import shutil
import tarfile
import datetime
from pathlib import Path

def create_backup():
    """Create comprehensive backup of NDR Platform data"""
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"ndr_backup_{timestamp}"
    backup_path = Path("backups") / backup_name
    
    # ✅ Create backup directory
    backup_path.mkdir(parents=True, exist_ok=True)
    
    # ✅ Critical directories to backup (config-driven)
    from core.config_loader import load_config
    config = load_config()
    
    backup_targets = {
        'models': config.get('system.models_dir', 'models'),
        'config': 'config',
        'feedback': config.get('feedback.storage_dir', 'data/feedback'),
        'results': config.get('system.results_dir', 'data/results'),
        'anomaly_history': 'data/anomaly_history'
    }
    
    print(f"🔄 Creating backup: {backup_name}")
    
    for name, source in backup_targets.items():
        if os.path.exists(source):
            dest = backup_path / name
            if os.path.isdir(source):
                shutil.copytree(source, dest)
            else:
                shutil.copy2(source, dest)
            print(f"✅ Backed up: {source}")
        else:
            print(f"⚠️ Skipped (not found): {source}")
    
    # ✅ Create compressed archive
    with tarfile.open(f"{backup_path}.tar.gz", "w:gz") as tar:
        tar.add(backup_path, arcname=backup_name)
    
    # Clean up temporary directory
    shutil.rmtree(backup_path)
    
    print(f"✅ Backup created: {backup_path}.tar.gz")
    return f"{backup_path}.tar.gz"

if __name__ == "__main__":
    create_backup()
```

### **� Structure Verification** *(Enhanced)*

```python
# scripts/verify_structure.py - Enhanced for v2.1.0
import os
from pathlib import Path
from core.config_loader import load_config

def verify_platform_structure():
    """Verify NDR Platform v2.1.0 directory structure"""
    
    print("🔍 Verifying NDR Platform v2.1.0 Structure")
    print("=" * 50)
    
    # ✅ Load configuration
    config = load_config()
    
    # ✅ Required directories from config
    required_structure = {
        'System Directories': [
            config.get('system.data_dir', 'data'),
            config.get('system.results_dir', 'data/results'),
            config.get('system.models_dir', 'models'),
            config.get('system.logs_dir', 'logs'),
            'cache'
        ],
        'Data Directories': [
            'data/json',
            config.get('feedback.storage_dir', 'data/feedback'),
            'data/reports',
            'data/anomaly_history'
        ],
        'Application Structure': [
            'app',
            'app/pages',
            'app/components',
            'core',
            'core/models',
            'core/explainers',
            'guides'
        ],
        'Configuration': [
            'config',
            'config/config.yaml'
        ]
    }
    
    all_good = True
    
    for category, paths in required_structure.items():
        print(f"\n📂 {category}:")
        for path in paths:
            if os.path.exists(path):
                if os.path.isfile(path):
                    print(f"  ✅ File: {path}")
                else:
                    print(f"  ✅ Directory: {path}/")
            else:
                print(f"  ❌ Missing: {path}")
                all_good = False
    
    print("\n" + "=" * 50)
    if all_good:
        print("✅ Platform structure is COMPLETE")
    else:
        print("❌ Platform structure has ISSUES")
        print("💡 Run 'python run.py --create-dirs-only' to fix missing directories")
    
    return all_good

if __name__ == "__main__":
    verify_platform_structure()
```

---

## 🎯 **Deployment Scripts**

### **🚀 Windows Deployment**

```batch
@echo off
REM deploy.bat - Windows deployment for NDR Platform v2.1.0

echo ===================================
echo NDR Platform v2.1.0 Deployment
echo ===================================

REM Check if setup was run
if not exist "venv" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Starting NDR Platform...
echo Platform will be available at: http://localhost:8501
echo Press Ctrl+C to stop the platform

python run.py
```

### **🐧 Linux/macOS Deployment**

```bash
#!/bin/bash
# deploy.sh - Linux deployment for NDR Platform v2.1.0

echo "==================================="
echo "NDR Platform v2.1.0 Deployment"
echo "==================================="

# Check if setup was run
if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please run setup.sh first."
    exit 1
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Starting NDR Platform..."
echo "Platform will be available at: http://localhost:8501"
echo "Press Ctrl+C to stop the platform"

python run.py
```

---

## 🔧 **Development Utilities**

### **🛠️ Developer Setup**

```python
# scripts/dev_utils.py - Development utilities
import subprocess
import sys
from pathlib import Path

def setup_dev_environment():
    """Setup development environment with additional tools"""
    
    dev_requirements = [
        'black',      # Code formatting
        'flake8',     # Linting
        'pytest',     # Testing
        'jupyter',    # Notebooks
        'pre-commit'  # Git hooks
    ]
    
    print("🛠️ Setting up development environment...")
    
    for package in dev_requirements:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ Installed: {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install: {package}")

def run_quality_checks():
    """Run code quality checks"""
    
    print("🔍 Running code quality checks...")
    
    # Black formatting
    subprocess.run(['black', '--check', '.'])
    
    # Flake8 linting
    subprocess.run(['flake8', '.'])
    
    # Tests
    subprocess.run(['pytest', 'tests/'])

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'setup':
        setup_dev_environment()
    elif len(sys.argv) > 1 and sys.argv[1] == 'check':
        run_quality_checks()
    else:
        print("Usage: python dev_utils.py [setup|check]")
```

---

## 📊 **Script Usage Examples**

### **Daily Operations**

```bash
# Check platform health
python scripts/health_check.py

# Create backup before major changes
python scripts/backup.py

# Verify structure after updates
python scripts/verify_structure.py

# Analyze recent logs
python scripts/log_analyzer.py --hours 24
```

### **Maintenance Tasks**

```bash
# Clean old logs
python scripts/maintenance.py --clean-logs

# Update models
python scripts/model_manager.py --update-all

# Security scan
python scripts/security_scanner.py

# Performance analysis
python scripts/performance_analyzer.py
```

---

## � **Related Documentation**

For detailed information:
- **Configuration:** See `CONFIGURATION_GUIDE.md`
- **Deployment:** See `DEPLOYMENT_GUIDE.md`
- **User Guide:** See `USER_GUIDE.md`
- **Project Structure:** See `PROJECT_ORGANIZATION.md`

**Script Status:** Production Ready v2.1.0 with enhanced configuration support.

### 🚀 **Root Level - Core Operations**
```
├── deploy.sh              # 🐳 Docker deployment (Linux/macOS)
├── deploy.bat             # 🐳 Docker deployment (Windows)  
├── health-check.sh        # 🏥 System health verification
└── run.py                 # 🎯 Application entry point
```

### 📁 **scripts/ - Platform Management**
```
├── linux/
│   ├── setup.sh           # 🔧 One-time platform setup
│   └── cleanup.sh         # 🧹 System cleanup
├── windows/
│   ├── setup.bat          # 🔧 One-time platform setup  
│   └── cleanup.bat        # 🧹 System cleanup
├── backup.py              # 💾 Data backup/restore
├── data_manager.py        # 📊 Data processing utilities
├── dev_utils.py           # 🛠️ Development utilities
├── health_check.py        # 🏥 Detailed health analysis
├── log_analyzer.py        # 📋 Log analysis tools
├── model_manager.py       # 🤖 ML model management
└── security_scanner.py    # 🔒 Security scanning
```

## 🎯 **WHEN TO USE EACH SCRIPT**

### **🔧 FIRST TIME SETUP**
```bash
# Linux/macOS
./scripts/linux/setup.sh

# Windows  
scripts\windows\setup.bat
```
**What it does:**
- ✅ Creates directories (data, logs, models, etc.)
- ✅ Sets up Python virtual environment
- ✅ Installs dependencies from requirements.txt
- ✅ Creates config.yaml from example
- ✅ Sets proper permissions
- ✅ Checks system dependencies

### **🚀 DEPLOYMENT (Every Time You Start)**
```bash
# Docker deployment (Production)
./deploy.sh                 # Smart deployment
./deploy.sh rebuild         # Force rebuild

# Development mode (Alternative)
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows
streamlit run run.py
```
**What deploy.sh does:**
- 🔍 Checks if rebuild needed (dependencies changed)
- 🐳 Builds/rebuilds Docker containers
- ▶️ Starts services
- ✅ Verifies deployment health

### **🧹 MAINTENANCE**
```bash
# Cleanup temporary files
./scripts/linux/cleanup.sh
./scripts/linux/cleanup.sh --deep    # Deep clean with prompts

# Health check
./health-check.sh                    # Quick health check
python scripts/health_check.py       # Detailed analysis

# Backup
python scripts/backup.py --type full
```

## ❌ **REMOVED REDUNDANT SCRIPTS**

**Deleted (Functionality moved to shell scripts):**
- ❌ `scripts/cleanup.py` → ✅ `scripts/linux/cleanup.sh` + `scripts/windows/cleanup.bat`
- ❌ `scripts/deploy.py` → ✅ `deploy.sh` + `deploy.bat` 
- ❌ `scripts/clean_restart.py` → ✅ Functionality in `cleanup.sh` + `deploy.sh restart`

## 🎯 **SIMPLIFIED WORKFLOW**

### **For New Users:**
```bash
1. ./scripts/linux/setup.sh        # One-time setup
2. ./deploy.sh                     # Start platform
3. Open http://localhost:8501       # Use platform
```

### **For Developers:**
```bash
1. ./scripts/linux/setup.sh        # One-time setup
2. source venv/bin/activate        # Activate environment
3. streamlit run run.py            # Development mode
```

### **For Updates:**
```bash
1. git pull                        # Get latest code
2. ./deploy.sh                     # Smart deployment (auto-detects changes)
```

## 💡 **KEY DISTINCTIONS**

| Script Type | Purpose | Frequency | Example |
|-------------|---------|-----------|---------|
| **Setup** | Initial environment preparation | Once | `setup.sh` |
| **Deploy** | Start/update running services | Every startup | `deploy.sh` |
| **Utilities** | Maintenance & troubleshooting | As needed | `backup.py` |

## 🎉 **BENEFITS OF THIS ORGANIZATION**

✅ **Clear separation of concerns**
✅ **No more redundant scripts**  
✅ **Cross-platform support** (Linux/Windows)
✅ **Simple workflow** for new users
✅ **Flexible deployment** options (Docker vs Development)
✅ **Maintenance scripts** for ongoing operations

**Bottom Line:** Setup once, deploy many times, maintain as needed! 🚀
