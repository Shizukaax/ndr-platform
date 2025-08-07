# 🛡️ NDR Platform v2.1.0 - Network Detection & Response

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-brightgreen.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Production](https://img.shields.io/badge/status-production%20ready-brightgreen.svg)](https://github.com/Shizukaax/ndr-platform/releases)
[![Enhanced](https://img.shields.io/badge/version-v2.1.0-orange.svg)](https://github.com/Shizukaax/ndr-platform/releases)

**🚀 Production-Ready Network Security Analytics Platform with Enhanced ML-Powered Anomaly Detection**

---

## 🎯 **Platform Overview** *(Production v2.1.0)*

The NDR Platform is a **production-ready Network Detection and Response system** with **comprehensive anomaly detection capabilities**. Following recent critical fixes, it provides **enterprise-grade network security analytics** with **robust error handling** and **reliable data processing**.

---

## ✅ **Recent Critical Fixes Applied**

### **🔧 Data Integrity Fixes**
- **✅ NaN Port Handling:** Port values now display "N/A" instead of corrupted "nan"
- **✅ Arrow Serialization:** Fixed dataframe display issues in Streamlit interface
- **✅ Results Saving:** Resolved missing anomaly indices in ModelManager
- **✅ Directory Structure:** Fixed feedback storage to use config-driven paths

### **🚀 System Enhancements**
- **✅ Configuration Management:** All directories now use config.yaml settings
- **✅ Error Handling:** Comprehensive exception handling and logging
- **✅ Data Validation:** Enhanced input validation and cleaning
- **✅ Performance:** Optimized data processing and display

---

## 🌟 **Core Features**

### **🤖 Machine Learning Engine**
- **Multi-Algorithm Support:** 6 ML algorithms (Isolation Forest, LOF, One-Class SVM, KNN, HDBSCAN, Ensemble)
- **✅ Fixed Results Saving:** Proper anomaly indices and model metadata storage
- **Enhanced Error Handling:** Robust exception handling for all ML operations
- **Configuration-Driven:** YAML-based model selection and tuning

### **📊 Data Processing**
- **✅ Arrow Compatibility:** Clean dataframe displays without serialization errors
- **✅ NaN Handling:** Proper handling of missing values in network data
- **Real-time Processing:** Live network packet analysis
- **Data Validation:** Enhanced input validation and cleaning

### **🎯 Security Intelligence**
- **MITRE ATT&CK Integration:** Automatic mapping to threat techniques
- **Severity Classification:** Automated risk assessment
- **Real-time Monitoring:** Live network analysis with configurable refresh
- **Historical Analysis:** Trend analysis and pattern recognition

### **🔧 Production Features**
- **✅ Configuration Management:** Robust YAML-driven configuration
- **✅ Directory Compliance:** All paths follow config settings
- **Docker Ready:** Container deployment for scalability
- **Comprehensive Logging:** Enhanced error tracking and debugging

---

## 🚀 **Quick Start Guide**

### **📋 Prerequisites**
- **Python 3.11+**
- **Docker & Docker Compose** (for containerized deployment)
- **8GB+ RAM** (recommended for large datasets)

### **🔧 Installation**

```bash
# Clone repository
git clone https://github.com/your-org/ndr-platform.git
cd ndr-platform

# Install dependencies
pip install -r requirements.txt

# Start platform (creates directories per config)
python run.py

# Access web interface
# Open browser: http://localhost:8501
```

### **🐳 Docker Deployment**

```bash
# Build and start containers
docker-compose -f docker-compose.prod.yml up -d

# Verify deployment
docker-compose -f docker-compose.prod.yml ps

# Access platform
open http://localhost:8501
```

---

## 🏗️ **Architecture Overview**

```
📦 NDR Platform v2.1.0
├── 🎯 Frontend Layer (Streamlit Web UI)
│   ├── ✅ Enhanced Anomaly Detection
│   ├── ✅ Fixed Explain & Feedback
│   ├── Real-time Monitoring
│   └── Analytics Dashboard
├── 🧠 Core Engine (Python Backend)
│   ├── ✅ Fixed ModelManager
│   ├── ✅ Enhanced DataManager
│   ├── Configuration Loader
│   └── SHAP Explainers
├── 📊 Data Layer
│   ├── ✅ Config-driven Storage
│   ├── Results Persistence
│   ├── Feedback Management
│   └── Historical Data
└── ⚙️ Infrastructure
    ├── Docker Containers
    ├── Configuration Management
    └── Logging System
```

---

## 📊 **Key Capabilities**

### **🔍 Anomaly Detection**
- **Multi-Algorithm Analysis:** Choose from 6 different ML algorithms
- **Ensemble Models:** Combine multiple algorithms for better accuracy
- **Real-time Processing:** Live network packet analysis
- **Historical Trending:** Pattern recognition and baseline learning

### **🎯 Threat Intelligence**
- **MITRE ATT&CK Mapping:** Automatic technique classification
- **Risk Scoring:** Automated severity assessment (Low/Medium/High/Critical)
- **Context Analysis:** Network behavior pattern analysis
- **Alert Generation:** Configurable notification system

### **📈 Analytics & Reporting**
- **Interactive Dashboards:** Real-time visualization and analysis
- **Comprehensive Reports:** Detailed anomaly reports with export capabilities
- **Historical Analysis:** Trend analysis with time-based filtering
- **Performance Metrics:** System performance and accuracy tracking

---

## 🔧 **Configuration Management**

### **📋 Main Configuration (config/config.yaml)**
```yaml
# System paths (✅ Fixed)
system:
  data_dir: "data"
  results_dir: "data/results"
  models_dir: "models"

# Feedback management (✅ Fixed)
feedback:
  storage_dir: "data/feedback"

# ML models configuration
anomaly_detection:
  save_results: true
  models:
    ensemble:
      enabled: true
```

### **🚀 Environment Variables**
```bash
# Production settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
PYTHONPATH=/app
```

---

## 📚 **Documentation**

### **📖 Complete Guide Suite**
- **[User Guide](guides/USER_GUIDE.md)** - Complete platform usage
- **[Configuration Guide](guides/CONFIGURATION_GUIDE.md)** - System configuration
- **[Deployment Guide](guides/DEPLOYMENT_GUIDE.md)** - Production deployment
- **[API Documentation](guides/API_DOCUMENTATION.md)** - Technical reference
- **[Project Organization](guides/PROJECT_ORGANIZATION.md)** - Codebase structure

### **🛠️ Development Resources**
- **[Script Organization](guides/SCRIPT_ORGANIZATION.md)** - Management utilities
- **[Logging Guide](guides/LOGGING_GUIDE.md)** - Monitoring and debugging

---

## 🚨 **Troubleshooting**

### **Common Issues & Solutions**

**Issue:** Port values showing "nan"
- **Status:** ✅ **FIXED** - Now displays "N/A" for missing values

**Issue:** Arrow serialization errors in dataframes
- **Status:** ✅ **FIXED** - Comprehensive data type cleaning implemented

**Issue:** Results not saving to configured directories
- **Status:** ✅ **FIXED** - Proper config.yaml integration

**Issue:** Feedback stored in wrong directory
- **Status:** ✅ **FIXED** - Uses config-specified paths

### **Health Check**
```bash
# Verify platform health
python scripts/health_check.py

# Check logs for errors
tail -f logs/app.log

# Validate configuration
python -c "from core.config_loader import load_config; print('Config valid:', load_config())"
```

---

## 🤝 **Contributing**

### **Development Setup**
```bash
# Setup development environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Install development tools
pip install black flake8 pytest

# Run tests
pytest tests/
```

### **Code Quality**
- **Black:** Code formatting
- **Flake8:** Linting and style checks
- **Pytest:** Comprehensive testing framework
- **Pre-commit:** Git hooks for quality assurance

---

## 📞 **Support**

### **Getting Help**
1. **Check Documentation:** Comprehensive guides in `guides/` directory
2. **Review Logs:** Detailed logging in `logs/app.log`
3. **GitHub Issues:** Report bugs and request features
4. **Health Check:** Use built-in diagnostic tools

### **Reporting Issues**
- **Bug Reports:** Use GitHub issue templates
- **Feature Requests:** Provide detailed use cases
- **Documentation:** Help improve guides and examples
- **Security:** Follow responsible disclosure practices

---

## 📊 **Platform Status**

**Current Version:** v2.1.0 - Production Ready  
**Status:** ✅ All Critical Fixes Applied  
**Documentation:** ✅ Complete and Updated  
**Testing:** ✅ Comprehensive Test Suite  
**Deployment:** ✅ Production-Ready Containers  

**Last Updated:** August 2025
