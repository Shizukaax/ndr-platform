# 🛠️ NDR Platform - Management Scripts

**Author:** [Shizukaax](https://github.com/Shizukaax) | **Repository:** [ndr-platform](https://github.com/Shizukaax/ndr-platform)

## 📋 Overview

This directory contains comprehensive management and utility scripts for the NDR Platform. These scripts automate common tasks including setup, deployment, monitoring, and maintenance operations.

## 🚀 Quick Start

```bash
# Make scripts executable (Linux/Mac)
chmod +x scripts/*.py

# Run any script with help
python scripts/<script_name>.py --help

# Example: Check system health
python scripts/health_check.py
```

## 📜 Available Scripts

### 🔧 **Setup & Configuration**

#### `setup.py` - Platform Initialization
**Purpose:** Complete platform setup and environment configuration
```bash
python scripts/setup.py [--environment dev|prod] [--force]
```
- ✅ Environment setup and validation
- ✅ Dependency installation and verification
- ✅ Configuration file generation
- ✅ Database initialization
- ✅ Directory structure creation

---

### 🚀 **Deployment & Operations**

#### `deploy.py` - Deployment Automation
**Purpose:** Automated deployment to various environments
```bash
python scripts/deploy.py [--env staging|production] [--rollback]
```
- ✅ Multi-environment deployment (dev/staging/prod)
- ✅ Docker container orchestration
- ✅ Health checks and validation
- ✅ Rollback capabilities
- ✅ Blue-green deployment support

#### `health_check.py` - System Health Monitoring
**Purpose:** Comprehensive system health validation
```bash
python scripts/health_check.py [--detailed] [--export-report]
```
- ✅ Service availability checks
- ✅ Database connectivity validation
- ✅ Disk space and memory monitoring
- ✅ Model performance validation
- ✅ API endpoint testing

---

### 🔒 **Security & Auditing**

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

---

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

---

### 💾 **Backup & Recovery**

#### `backup.py` - Backup & Restore Operations
**Purpose:** Complete backup and disaster recovery
```bash
python scripts/backup.py [--action backup|restore] [--type full|incremental]
```
- ✅ Full and incremental backups
- ✅ Model and configuration backup
- ✅ Database backup and restore
- ✅ Point-in-time recovery
- ✅ Backup validation and testing

---

### 📊 **Monitoring & Analysis**

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

### **Fresh Installation**
```bash
# 1. Initialize platform
python scripts/setup.py --environment dev

# 2. Validate installation
python scripts/health_check.py --detailed

# 3. Run security scan
python scripts/security_scanner.py --scan-type all
```

### **Production Deployment**
```bash
# 1. Deploy to production
python scripts/deploy.py --env production

# 2. Validate deployment
python scripts/health_check.py

# 3. Create backup
python scripts/backup.py --action backup --type full
```

### **Maintenance Routine**
```bash
# 1. Analyze logs
python scripts/log_analyzer.py --timeframe 24h

# 2. Health check
python scripts/health_check.py

# 3. Security scan
python scripts/security_scanner.py

# 4. Backup
python scripts/backup.py --action backup --type incremental
```

### **Model Updates**
```bash
# 1. Train new model
python scripts/model_manager.py --action train --model-type isolation

# 2. Evaluate performance
python scripts/model_manager.py --action evaluate

# 3. Deploy if satisfactory
python scripts/model_manager.py --action deploy
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
