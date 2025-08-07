# 🛡️ NDR Platform v2.1.0 (Network Detection & Response)

**👨‍💻 Author:** [Shizukaax](https://github.com/Shizukaax) | **📧 Contact:** justinchua@tunglok.com

> **🎯 Enterprise-Grade Real-time Network Monitoring Platform**  
> Specialized for **Arkime JSON packet analysis** with **ML-powered anomaly detection**, **comprehensive anomaly tracking**, and **predictive security intelligence**.

---

## 🌟 **Platform Overview**

The NDR Platform is a comprehensive network security solution that combines advanced machine learning algorithms with real-time monitoring capabilities to detect, analyze, and respond to network anomalies. Built specifically for processing Arkime JSON packet data, it provides enterprise-grade security intelligence with an intuitive web interface.

### **� Key Capabilities:**
- **🤖 Multi-Model ML Detection** - 6 algorithms including ensemble methods
- **📊 Real-time Analytics** - Live monitoring with interactive dashboards  
- **🎯 MITRE ATT&CK Mapping** - Automated threat classification
- **🏷️ AI-Powered Auto-Labeling** - Intelligent anomaly classification
- **� Predictive Security** - Forecasting and trend analysis
- **💬 Feedback-Driven Learning** - Continuous model improvement

---

## 🚀 **Recent Updates** *(August 2025)*

### **✅ Critical Fixes Applied:**
- **Results Saving** - Now properly saves to configured `data/results` directory
- **SHAP Explanations** - Enhanced with comprehensive NaN value handling
- **Data Display** - Fixed Arrow serialization errors and NaN port values
- **Directory Structure** - Corrected feedback storage to follow config settings
- **Model Integration** - Seamless ensemble model creation and error handling

### **🆕 Latest Features:**
- **Enhanced Anomaly Tracking** - Persistent storage with historical analysis
- **Advanced Configuration** - Configurable ML models via YAML
- **Improved Visualizations** - Clean data display without encoding issues
- **Robust Error Handling** - Comprehensive exception management

---

## 📖 **Documentation Suite**

### **📋 Core Documentation**
| Document | Purpose | Status |
|----------|---------|--------|
| **[📖 Project Organization](guides/PROJECT_ORGANIZATION.md)** | Architecture & structure | ✅ Updated |
| **[� User Guide](guides/USER_GUIDE.md)** | Platform usage workflows | ✅ Updated |  
| **[⚙️ Configuration Guide](guides/CONFIGURATION_GUIDE.md)** | Setup & ML configuration | ✅ Updated |
| **[🚀 Deployment Guide](guides/DEPLOYMENT_GUIDE.md)** | Production deployment | ✅ Updated |

### **📚 Technical Documentation**  
| Document | Purpose | Status |
|----------|---------|--------|
| **[📋 API Documentation](guides/API_DOCUMENTATION.md)** | API endpoints & integration | ✅ Updated |
| **[📝 Logging Guide](guides/LOGGING_GUIDE.md)** | Logging configuration | ✅ Updated |
| **[� Script Organization](guides/SCRIPT_ORGANIZATION.md)** | Utility scripts guide | ✅ Updated |

---

## 🚀 **Quick Start**

### **⚡ Launch Platform**
```powershell
# Start the NDR Platform
python run.py
# Access at: http://localhost:8501
```

### **📊 Essential Configuration**
```yaml
# config/config.yaml - Core settings
system:
  data_dir: "data"
  results_dir: "data/results" 
  models_dir: "models"

data_source:
  directory: "./data/json/"
  file_pattern: "*.json"

anomaly_detection:
  default_threshold: 0.8
  models:
    ensemble:
      enabled: true
      combination_method: "weighted_average"
```

---

## 🏗️ **Platform Architecture**

### **📁 Core Components**
```
NDR Platform/
├── 🎮 app/           # Streamlit web interface
├── 🧠 core/          # ML engine & data processing  
├── ⚙️ config/        # Configuration files
├── 📊 data/          # Data storage & results
├── 🤖 models/        # Trained ML models
├── 📚 guides/        # Documentation suite
└── 🛠️ scripts/       # Management utilities
```

### **🔄 Workflow Pipeline**
1. **📥 Data Ingestion** → JSON packet processing
2. **🤖 ML Analysis** → Multi-model anomaly detection  
3. **📊 Visualization** → Interactive dashboards
4. **🎯 Investigation** → SHAP explanations & feedback
5. **📈 Intelligence** → Predictive analytics & reporting

---

## 🛠️ **Development Status**

### **✅ Fully Implemented**
- Multi-algorithm anomaly detection (6 models + ensemble)
- Real-time monitoring with auto-refresh capabilities
- MITRE ATT&CK threat mapping and classification
- AI-powered auto-labeling with confidence scoring
- Comprehensive analytics dashboard with trend analysis
- SHAP-based model explanations with feedback integration
- Predictive security intelligence and forecasting
- Professional reporting system with multiple formats

### **� Configuration-Driven**
- All paths and settings managed via `config/config.yaml`
- Flexible ML model selection and tuning
- Customizable monitoring intervals and thresholds
- Adaptable data source configuration

---

## 🤝 **Contributing**

This platform is actively maintained and enhanced. For issues, improvements, or feature requests, please contact the development team.

---

## 📄 **License**

Enterprise Network Detection & Response Platform - All rights reserved.
  directory: "C:\\Users\\justinchua\\Desktop\\newnewapp\\data"
  file_pattern: "*.json"  # Arkime JSON files
  max_files: 100
  recursive: true
```

### **🆕 New PC Setup (Alternative)**
```powershell
# Windows PowerShell
1. scripts\windows\setup.bat     # One-time setup (installs dependencies)
2. scripts\windows\deploy.bat    # Deploy to Docker containers (or use deploy.bat in root)
3. Open http://localhost:8501    # Access NDR Platform

# Linux/macOS
1. ./scripts/linux/setup.sh     # One-time setup (installs dependencies)
2. ./scripts/linux/deploy.sh    # Deploy to Docker containers
3. Open http://localhost:8501    # Access NDR Platform
```

## ✨ **Key Features**

### **🔍 Enhanced Real-time Network Monitoring**
- **Live Arkime JSON Processing**: Direct integration with packet capture data
- **Multi-protocol Analysis**: TCP, UDP, ICMP, and custom protocol detection
- **Real-time Threat Detection**: Advanced ML-powered anomaly detection
- **Configurable Auto-refresh**: 30s to 30min intervals via config.yaml
- **Comprehensive Alerting**: Multi-severity anomaly classification

### **🤖 Advanced ML-Powered Security**
- **6 ML Algorithms**: Isolation Forest, LOF, One-Class SVM, KNN, HDBSCAN, Ensemble
- **Ensemble Detection**: Weighted combination of multiple models
- **Baseline Learning**: System learns normal patterns automatically
- **Anomaly Tracking**: Persistent storage and historical analysis
- **MITRE ATT&CK Integration**: Threat framework correlation

### **📊 Comprehensive Anomaly Management**
- **📈 Anomaly History Dashboard**: Time-filtered historical analysis
- **🎯 Severity Classification**: Automated risk assessment levels
- **📋 Detailed Tracking**: Every detection stored with full context
- **⚡ Acknowledgment System**: Mark anomalies as reviewed
- **📊 Trend Analysis**: Baseline deviation and pattern recognition

### **� Predictive Security Intelligence**
- **Threat Intelligence**: Advanced behavioral analytics
- **Risk Scoring**: Dynamic threat assessment algorithms
- **Geographic Intelligence**: IP geolocation and threat correlation
- **Security Metrics**: Real-time performance monitoring

## 🎯 **Current Platform Status**

### **Your Enhanced Arkime Data Integration:**
- **📁 Data Directory**: `C:\Users\justinchua\Desktop\newnewapp\data`
- **📊 Files Detected**: 2 JSON files  
- **📦 Total Records**: 21,281+ network packets
- **🔄 Auto-refresh**: Configurable (30s to 30min intervals)
- **✅ Real Data**: No mock data - authentic packet analysis with ML enhancement

### **🤖 Enhanced ML Capabilities:**
- **✅ Anomaly Tracking**: Complete historical tracking with baseline learning
- **✅ Model Configuration**: 6 algorithms configurable via config.yaml
- **✅ Ensemble Detection**: Weighted model combinations for accuracy
- **✅ Severity Classification**: Automated risk assessment and alerting
- **✅ Real-time Integration**: Seamless ML detection with persistent storage

### **📊 Dashboard Enhancements:**
- **📈 Anomaly History Tab**: Time-filtered historical analysis
- **🎯 Detailed Tracking**: Every detection stored with full context
- **⚡ Quick Actions**: Acknowledge, export, and analyze anomalies
- **📊 Baseline Status**: Real-time deviation from learned baselines
- **🔧 Configuration UI**: Easy model selection and threshold adjustment

## 🛠️ **Recent Major Enhancements** *(August 2025)*

### **✅ Enhanced Anomaly Tracking System:**
1. **📊 AnomalyTracker Component** - Enterprise-grade anomaly lifecycle management
2. **📈 Anomaly History Dashboard** - Complete historical analysis with filtering
3. **🎯 Baseline Learning** - System learns normal patterns and detects deviations  
4. **📋 Persistent Storage** - JSON-based storage for all anomaly detections
5. **⚡ Real-time Integration** - Seamless tracking of all ML detections

### **🤖 Advanced ML Configuration:**
1. **🔧 Configurable Models** - Choose from 6 ML algorithms via config.yaml
2. **⚖️ Ensemble Detection** - Weighted ensemble for improved accuracy
3. **🎛️ Dynamic Thresholds** - Environment-specific confidence levels
4. **📊 Model Performance** - Detailed analysis and comparison tools

### **🚀 Platform Optimization:**
- **Network-Centric Architecture**: Optimized for Arkime packet analysis
- **Real-time Processing**: Direct file monitoring with ML integration
- **Enhanced Interface**: Clean, professional dashboard with advanced features
- **Performance Optimized**: Efficient data loading with comprehensive tracking

---

**🔧 Need Help?** Check the [Enhanced User Guide](guides/USER_GUIDE.md) or [ML Configuration Guide](guides/CONFIGURATION_GUIDE.md)  
**🐛 Found Issues?** Review our comprehensive testing suite with `python test_tracking_system.py`
