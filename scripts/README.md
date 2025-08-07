# 🛠️ NDR Platform - Management Scripts (Cleaned Up)

**Author:** [Shizukaax](https://github.com/Shizukaax) | **Repository:** [ndr-platform](https://github.com/Shizukaax/ndr-platform)

## 📋 Overview

This directory contains **specialized management scripts** for the NDR Platform. 

⚠️ **NOTE:** For basic operations, use the **root-level scripts**:
- `./deploy.sh` or `deploy.bat` - Smart deployment
- `./scripts/linux/setup.sh` or `scripts\windows\setup.bat` - One-time setup
- `./health-check.sh` - Health verification

## �️ **Current Script Organization**

### 📁 **Platform-Specific Setup/Cleanup**
```
├── linux/
│   ├── setup.sh           # One-time platform setup (Linux/macOS)
│   └── cleanup.sh         # System cleanup (Linux/macOS)
└── windows/
    ├── setup.bat          # One-time platform setup (Windows)
    └── cleanup.bat        # System cleanup (Windows)
```

### 🐍 **Specialized Python Scripts** (Complex logic & integrations)
```
├── backup.py              # Data backup/restore with compression
├── data_manager.py        # Advanced data processing & validation
├── dev_utils.py           # Development utilities & debugging
├── health_check.py        # Detailed health analysis (uses psutil)
├── log_analyzer.py        # Log parsing & analysis
├── model_manager.py       # ML model training & management
└── security_scanner.py    # Security scanning & threat analysis
```

## 🎯 **Why Keep Python Scripts?**

These scripts use **complex Python libraries** and **application integration** that would be difficult to replicate in shell scripts:

- **`backup.py`** - Uses tarfile, zipfile, complex directory traversal
- **`health_check.py`** - Uses psutil for system metrics, requests for API testing
- **`model_manager.py`** - Integrates with scikit-learn, pandas, pickle
- **`security_scanner.py`** - Uses requests, JSON parsing, threat intelligence APIs
- **`log_analyzer.py`** - Complex regex patterns, pandas analysis
- **`data_manager.py`** - DataFrame operations, data validation
- **`dev_utils.py`** - IDE integration, debugging utilities

## 🚀 **Quick Start for New PC Setup**

### **Option 1: Docker Deployment (Recommended)**
```bash
# Windows
1. scripts\windows\setup.bat     # One-time setup
2. deploy.bat                    # Docker deployment
3. Open http://localhost:8501    # Access platform

# Linux/macOS
1. ./scripts/linux/setup.sh     # One-time setup
2. ./deploy.sh                   # Docker deployment
3. Open http://localhost:8501    # Access platform
```

### **Option 2: Development Mode**
```bash
# Windows
1. scripts\windows\setup.bat     # One-time setup
2. venv\Scripts\activate         # Activate environment
3. streamlit run run.py          # Start development server

# Linux/macOS
1. ./scripts/linux/setup.sh     # One-time setup
2. source venv/bin/activate      # Activate environment
3. streamlit run run.py          # Start development server
```

## 📜 **Specialized Python Scripts**

### 🔧 **System Management**

#### `health_check.py` - Detailed Health Analysis
**Purpose:** Comprehensive system health validation with detailed metrics
```bash
python scripts/health_check.py [--detailed] [--export-report]
```
- ✅ Service availability checks
- ✅ Database connectivity validation
- ✅ System resource monitoring (CPU, Memory, Disk)
- ✅ Model performance validation
- ✅ API endpoint testing

#### `backup.py` - Backup & Restore Operations
**Purpose:** Complete backup and disaster recovery
```bash
python scripts/backup.py [--action backup|restore] [--type full|incremental]
```
- ✅ Full and incremental backups
- ✅ Model and configuration backup
- ✅ Data archive and restore
- ✅ Point-in-time recovery
- ✅ Backup validation and testing

### 🤖 **Machine Learning & Data**

#### `model_manager.py` - ML Model Lifecycle Management
**Purpose:** Complete machine learning model management
```bash
python scripts/model_manager.py [--action train|evaluate|deploy] [--model-type isolation|dbscan|knn]
```
- ✅ Model training and retraining
- ✅ Performance evaluation and metrics
- ✅ Model versioning and deployment
- ✅ A/B testing support
- ✅ Model rollback capabilities

#### `data_manager.py` - Data Operations & Validation
**Purpose:** Data pipeline management and validation
```bash
python scripts/data_manager.py [--action validate|clean|transform] [--source path/to/data]
```
- ✅ Data validation and quality checks
- ✅ Data cleaning and preprocessing
- ✅ Schema validation
- ✅ Data transformation pipelines
- ✅ Export and backup operations

### � **Security & Monitoring**

#### `security_scanner.py` - Security Auditing
**Purpose:** Comprehensive security scanning and vulnerability assessment
```bash
python scripts/security_scanner.py [--scan-type all|deps|files] [--report-format json|html]
```
- ✅ Dependency vulnerability scanning
- ✅ File permission auditing
- ✅ Configuration security validation
- ✅ Compliance reporting
- ✅ Security recommendations

#### `log_analyzer.py` - Log Analysis & Monitoring
**Purpose:** Comprehensive log analysis and monitoring
```bash
python scripts/log_analyzer.py [--timeframe 24h|7d|30d] [--severity error|warning|info]
```
- ✅ Log aggregation and analysis
- ✅ Error pattern detection
- ✅ Performance metrics extraction
- ✅ Alert generation
- ✅ Report generation

### 🛠️ **Development Tools**

#### `dev_utils.py` - Development Utilities
**Purpose:** Development and debugging tools
```bash
python scripts/dev_utils.py [--action profile|debug|test] [--component all|models|api]
```
- ✅ Performance profiling
- ✅ Debug utilities and helpers
- ✅ Test data generation
- ✅ Code quality checks
- ✅ Development environment setup

---

## 🔄 Common Workflows

### **🆕 Fresh Installation on New PC**
```bash
# Windows Users
1. scripts\windows\setup.bat     # Install dependencies & setup environment
2. deploy.bat                    # Deploy to Docker containers
3. Open http://localhost:8501    # Access NDR Platform

# Linux/macOS Users  
1. ./scripts/linux/setup.sh     # Install dependencies & setup environment
2. ./deploy.sh                   # Deploy to Docker containers
3. Open http://localhost:8501    # Access NDR Platform
```

### **🔄 Daily Operations**
```bash
# Start/Restart Platform
deploy.bat                       # Windows
./deploy.sh                      # Linux/macOS

# Health Check
./health-check.sh               # Check platform status

# View Logs
docker-compose logs -f          # Follow logs
```

### **🛠️ Development Mode (Alternative to Docker)**
```bash
# Windows
1. scripts\windows\setup.bat     # One-time setup
2. venv\Scripts\activate         # Activate Python environment
3. streamlit run run.py          # Start development server

# Linux/macOS
1. ./scripts/linux/setup.sh     # One-time setup
2. source venv/bin/activate      # Activate Python environment
3. streamlit run run.py          # Start development server
```

### **🔍 Advanced Operations**
```bash
# Security Audit
python scripts/security_scanner.py --scan-type all

# Model Training
python scripts/model_manager.py --action train --model-type isolation

# Data Backup
python scripts/backup.py --action backup --type full

# Log Analysis
python scripts/log_analyzer.py --timeframe 24h --severity error
```

## 📊 Script Dependencies

| Script | Python Packages | External Tools | Notes |
|--------|------------------|----------------|-------|
| `setup.py` | `pip`, `venv` | Docker (optional) | Core setup |
| `deploy.py` | `docker`, `requests` | Docker, Docker Compose | Deployment |
| `health_check.py` | `psutil`, `requests` | - | Monitoring |
| `security_scanner.py` | `safety`, `bandit` | - | Security |
| `model_manager.py` | `scikit-learn`, `joblib` | - | ML Operations |
| `data_manager.py` | `pandas`, `jsonschema` | - | Data Processing |
| `backup.py` | `shutil`, `zipfile` | - | Backup Operations |
| `log_analyzer.py` | `regex`, `datetime` | - | Log Analysis |
| `dev_utils.py` | `pytest`, `black` | - | Development |

## 🚨 Important Notes

### **Security Considerations**
- All scripts validate user permissions before execution
- Sensitive operations require explicit confirmation
- Logs are sanitized to prevent information leakage
- Backup files are encrypted by default

### **Error Handling**
- Comprehensive error logging and reporting
- Graceful degradation for non-critical failures
- Rollback capabilities for deployment scripts
- Detailed error messages with suggested solutions

### **Performance**
- Scripts are optimized for large datasets
- Background processing for long-running operations
- Progress indicators for time-consuming tasks
- Resource usage monitoring and limits

## 🆘 Getting Help

### **Script-specific Help**
```bash
python scripts/<script_name>.py --help
```

### **Common Issues**
1. **Permission Errors**: Ensure scripts have execute permissions
2. **Missing Dependencies**: Run `pip install -r requirements.txt`
3. **Configuration Issues**: Check `.env` file configuration
4. **Docker Issues**: Ensure Docker daemon is running

### **Support Channels**
- **🐛 Bug Reports**: [GitHub Issues](https://github.com/Shizukaax/ndr-platform/issues)
- **💬 Questions**: [GitHub Discussions](https://github.com/Shizukaax/ndr-platform/discussions)
- **🔒 Security Issues**: See [SECURITY.md](../.github/SECURITY.md)

---

**📅 Last Updated**: August 5, 2025 | **✍️ Maintained by**: [Shizukaax](https://github.com/Shizukaax)
