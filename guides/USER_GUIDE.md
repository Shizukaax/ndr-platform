# NDR Platform v2.1.0 - User Guide

## 👥 Complete User Documentation *(Production Ready - August 2025)*

### 🎯 Platform Overview

The **Network Detection and Response (NDR) Platform v2.1.0** is a production-ready cybersecurity analytics system powered by machine learning. Following recent critical fixes and enhancements, the platform provides enterprise-grade anomaly detection with comprehensive data integrity and reliable results storage.

---

## ⚡ **Quick Start Guide**

### **1️⃣ Launch Platform**
```powershell
# Start the application (uses config-driven directory creation)
python run.py
```

### **2️⃣ Access Web Interface**
- Open browser: `http://localhost:8501`
- Platform automatically creates required directories per configuration
- All results properly saved to configured paths

### **3️⃣ Core Features**
- **Anomaly Detection:** Multi-algorithm ML analysis with proper results saving
- **Explain & Feedback:** Enhanced SHAP explanations with NaN handling
- **Real-time Monitoring:** Live network analysis
- **Model Management:** ML lifecycle with ensemble capabilities

---

## 🔧 **Recent Critical Fixes Applied**

### **✅ Data Integrity Fixes**
- **Port Display:** NaN values now show "N/A" instead of corrupted "nan"
- **Arrow Compatibility:** All dataframes cleaned for proper Streamlit display
- **Results Saving:** Fixed missing anomaly indices in ModelManager
- **Directory Structure:** Feedback stored in `data/feedback` per config

### **✅ Configuration Management**
- **Path Resolution:** All directories created using `config.yaml` settings
- **Error Handling:** Robust fallback mechanisms for missing configurations
- **Encoding Support:** Documentation and fixes for Unicode issues

---

## 📊 **Core Modules**

### **🔍 Anomaly Detection Module**
**Location:** Main Dashboard → Anomaly Detection

**Features:**
- ✅ **Fixed Results Saving:** All analysis results properly stored
- ✅ **Enhanced Error Handling:** Graceful handling of missing data
- **Multi-Algorithm Support:** 6 ML algorithms available
- **Ensemble Detection:** Combined model predictions
- **Real-time Processing:** Live analysis capabilities

**Usage Steps:**
1. **Data Upload:** Upload JSON network packet files
2. **Model Selection:** Choose from available algorithms
3. **Analysis Execution:** Run detection with proper result persistence
4. **Results Review:** View anomalies with corrected data display

### **🎯 Explain & Feedback Module**
**Location:** Main Dashboard → Explain & Feedback

**Recent Enhancements:**
- ✅ **NaN Port Handling:** Clean display of network port data
- ✅ **Arrow Serialization:** Fixed dataframe display issues
- ✅ **SHAP Integration:** Comprehensive model explanations
- **User Feedback:** Learning system for model improvement

**Features:**
- **Model Explanations:** SHAP-based feature importance
- **Interactive Feedback:** User validation of predictions
- **Data Visualization:** Clean, Arrow-compatible displays
- **Feature Analysis:** Detailed network packet analysis

### **📈 Analytics Dashboard**
**Location:** Main Dashboard → Analytics

**Capabilities:**
- **Performance Metrics:** Model accuracy and precision
- **Historical Trends:** Time-series anomaly analysis
- **Comparative Analysis:** Multi-model performance
- **Risk Assessment:** Threat severity scoring

### **⚙️ Model Management**
**Location:** Main Dashboard → Model Management

**Functions:**
- **Lifecycle Management:** Train, deploy, retire models
- **Version Control:** Model backup and rollback
- **Performance Monitoring:** Continuous accuracy tracking
- **Configuration:** YAML-driven model settings

---

## 🗂️ **Data Management**

### **📁 Directory Structure** *(Post-Fix)*

```
📦 Data Organization/
├── 📁 data/json/              # Source network packet data
├── 📁 data/results/           # ✅ ML analysis results (config-driven)
├── 📁 data/feedback/          # ✅ User feedback (config-driven)
├── 📁 data/reports/           # Generated analysis reports
├── 📁 data/anomaly_history/   # Historical anomaly tracking
├── 📁 models/                 # Trained ML models and metadata
└── 📁 logs/                   # Application and error logs
```

### **✅ Configuration-Driven Paths**
All data storage locations are managed via `config/config.yaml`:

```yaml
system:
  data_dir: "data"
  results_dir: "data/results"
  models_dir: "models"

feedback:
  storage_dir: "data/feedback"  # Fixed: Proper directory usage

anomaly_detection:
  save_results: true            # Fixed: Consistent results saving
```

---

## 🔍 **Troubleshooting Guide**

### **✅ Common Issues Resolved**

**Issue:** Port values showing "nan"
- **Status:** ✅ **FIXED** - Now displays "N/A" for missing values
- **Solution:** Enhanced data cleaning in `explain_feedback.py`

**Issue:** Arrow serialization errors in dataframes
- **Status:** ✅ **FIXED** - Comprehensive data type cleaning
- **Solution:** `clean_dataframe_for_arrow()` function implemented

**Issue:** Results not saving to configured directories
- **Status:** ✅ **FIXED** - Proper config.yaml integration
- **Solution:** Updated `run.py` and `ModelManager` for config compliance

**Issue:** Feedback stored in wrong directory
- **Status:** ✅ **FIXED** - Uses config-specified paths
- **Solution:** Directory creation follows `config.yaml` settings

### **📞 Error Recovery**
If you encounter issues:

1. **Check Configuration:** Verify `config/config.yaml` settings
2. **Review Logs:** Check `logs/app.log` for detailed errors
3. **Restart Platform:** Use `python run.py` for clean restart
4. **Verify Data:** Ensure JSON files are properly formatted
5. **Contact Support:** Reference error logs for assistance

---

## 🔒 **Data Security & Persistence**

### **🚨 Critical Directories (Never Delete)**
```
📦 PERSISTENT DATA:
├── 📁 models/                 # Trained ML models
├── 📁 config/                 # System configuration
├── 📁 data/feedback/          # User feedback history
├── 📁 data/anomaly_history/   # Historical anomaly data
└── 📁 logs/                   # Security logs
```

### **🔄 Safe to Clear**
```
📦 TEMPORARY DATA:
└── 📁 cache/                  # Processing cache
```

---

## 📖 **Additional Resources**

For detailed technical information:
- **Configuration:** See `CONFIGURATION_GUIDE.md`
- **Deployment:** See `DEPLOYMENT_GUIDE.md`
- **API Reference:** See `API_DOCUMENTATION.md`
- **Project Structure:** See `PROJECT_ORGANIZATION.md`

**Platform Status:** Production Ready v2.1.0 with all critical fixes applied.

1. **Open your web browser** and navigate to the platform URL
   - Development: `http://localhost:8501`
   - Production: `https://your-domain.com`

2. **Initial Setup** (First-time users)
   - No authentication required for local deployment
   - For production, contact your administrator for access credentials

### 🏠 Enhanced Platform Interface

The NDR Platform features an intuitive sidebar navigation with **enhanced sections**:

```
📊 Main Dashboard          # Real-time overview and metrics
🔍 Real-time Monitoring   # 🆕 ENHANCED: Live monitoring with ML anomaly detection
  ├── 🚀 Quick Start      # Platform overview and status
  ├── 📡 Live Dashboard   # Real-time metrics and monitoring
  ├── 📊 Security Metrics # Performance tracking
  ├── 🔮 Predictive Security # Threat forecasting
  └── 📈 Anomaly History  # 🆕 NEW: Historical analysis dashboard
🔍 Anomaly Detection      # AI-powered threat detection
📈 Advanced Analytics     # Deep-dive analysis tools
🎯 MITRE ATT&CK Mapping   # Threat technique mapping  
📋 Reporting             # Generate and export reports
⚙️ Settings              # Enhanced configuration and ML model preferences
ℹ️ About                 # Platform information
```

## 📊 Enhanced Main Dashboard

### **Real-time Monitoring Section** *(Primary Interface)*
The enhanced dashboard provides **comprehensive real-time insights** with ML-powered anomaly detection:

- **🚀 Quick Start Tab**
  - Platform status and data source overview
  - Monitoring status (active/ready/no data)
  - Data file detection and statistics
  - Real-time file counting with change notifications

- **📡 Live Dashboard Tab**  
  - **Configurable auto-refresh** (30 seconds to 30 minutes)
  - Real-time network metrics from actual Arkime JSON data
  - **Enhanced threat level indicators** with dynamic scoring
  - **Advanced ML-based anomaly detection** (configurable models)
  - Live packet stream analysis with threat scoring

- **📊 Security Metrics Tab**
  - SIEM performance metrics (marked as simulated)
  - System health monitoring
  - Compliance tracking

- **🔮 Predictive Security Tab**
  - 24-hour threat probability forecasting
  - Attack progression modeling
  - Risk assessment algorithms

- **📈 Anomaly History Tab** *(NEW)*
  - **Time-filtered historical analysis** (24 hours, 7 days, 30 days)
  - **Summary metrics** (total detections, average rates, severity distribution)
  - **Detailed anomaly records** with full forensic context
  - **Trend analysis** and baseline comparison

### **Real-time Monitoring**
Monitor your network activity with live updates:

1. **Event Stream**: Real-time feed of network events
2. **Anomaly Alerts**: Immediate notifications of detected threats
3. **Performance Metrics**: System resource usage and model accuracy
4. **Health Status**: Platform component status indicators

## 🔍 Anomaly Detection

### **Starting Analysis**

#### **Step 1: Data Source Selection**
1. Navigate to the **Anomaly Detection** page
2. Choose your data source:
   - **Automatic**: Use default data directory
   - **Manual Upload**: Upload specific PCAP/JSON files
   - **Real-time Stream**: Connect to live data feed

#### **Step 2: Model Configuration**
Select and configure your detection model:

**Available Algorithms:**
- **Isolation Forest** (Recommended for general use)
  - Best for: Large datasets with mixed anomaly types
  - Contamination: 0.1 (10% expected anomalies)
  - Estimators: 100 (number of trees)

- **Local Outlier Factor**
  - Best for: Local density-based anomalies
  - Neighbors: 20 (comparison group size)
  - Contamination: 0.1

- **One-Class SVM**
  - Best for: High-dimensional data
  - Kernel: RBF (Radial Basis Function)
  - Nu: 0.1 (outlier fraction)

**Configuration Options:**
```
Model Parameters:
├── Contamination: Expected anomaly percentage (0.01-0.5)
├── Training Size: Number of samples for model training
├── Feature Selection: Automatic or manual feature engineering
└── Validation: Cross-validation and performance metrics
```

#### **Step 3: Feature Engineering**
Customize features for analysis:

- **Network Features**
  - Source/Destination IPs and ports
  - Protocol types and flags
  - Packet sizes and timing
  - Flow duration and byte counts

- **Behavioral Features**
  - Communication patterns
  - Frequency analysis
  - Geographic information
  - Time-based patterns

#### **Step 4: Model Training**
1. Click **"Train Model"** to start the process
2. Monitor training progress in real-time
3. Review model performance metrics:
   - **Accuracy**: Overall prediction correctness
   - **Precision**: True positive rate
   - **Recall**: Sensitivity to anomalies
   - **F1-Score**: Balanced performance metric

#### **Step 5: Anomaly Detection**
1. Apply the trained model to detect anomalies
2. Review detected anomalies in the results table
3. Analyze anomaly scores and confidence levels
4. Export results for further investigation

### **Results Interpretation**

#### **Anomaly Scores**
- **0.0 - 0.3**: Normal behavior (low risk)
- **0.3 - 0.7**: Suspicious activity (medium risk)
- **0.7 - 1.0**: Highly anomalous (high risk)

#### **Confidence Levels**
- **High (>0.8)**: Strong evidence of anomaly
- **Medium (0.5-0.8)**: Moderate confidence
- **Low (<0.5)**: Weak evidence, may be false positive

## 📈 Advanced Analytics

### **Comparative Analysis**
Compare multiple models and datasets:

1. **Model Comparison**
   - Run multiple algorithms simultaneously
   - Compare performance metrics
   - Identify best-performing models
   - Ensemble model creation

2. **Time-series Analysis**
   - Trend analysis over time
   - Seasonal pattern detection
   - Anomaly frequency analysis
   - Baseline establishment

3. **Feature Importance**
   - Identify key indicators
   - Feature ranking and selection
   - Correlation analysis
   - Dimensionality reduction

### **Interactive Visualizations**

#### **Scatter Plots**
- 2D/3D anomaly visualization
- Feature correlation plots
- Cluster identification
- Interactive data exploration

#### **Time Series Charts**
- Anomaly timeline visualization
- Trend analysis
- Seasonal pattern identification
- Real-time monitoring

#### **Network Topology**
- Source-destination mapping
- Geographic visualization
- Protocol distribution
- Traffic flow analysis

### **Statistical Analysis**
Deep-dive into your data with advanced statistics:

- **Descriptive Statistics**: Mean, median, standard deviation
- **Distribution Analysis**: Histograms and probability density
- **Correlation Analysis**: Feature relationships
- **Outlier Detection**: Statistical anomaly identification

## 🎯 MITRE ATT&CK Mapping

### **Threat Technique Identification**

#### **Automatic Mapping**
The platform automatically maps detected anomalies to MITRE ATT&CK techniques:

1. **Anomaly Analysis**: Extract behavior patterns
2. **Pattern Matching**: Compare against MITRE database
3. **Technique Assignment**: Map to specific ATT&CK techniques
4. **Confidence Scoring**: Assess mapping reliability

#### **Manual Review**
Review and validate automatic mappings:

1. **Technique Details**
   - Technique ID and name
   - Tactic categories
   - Detailed descriptions
   - Mitigation strategies

2. **Evidence Review**
   - Supporting network evidence
   - Behavior indicators
   - Confidence assessment
   - False positive evaluation

### **Risk Assessment**

#### **Risk Scoring Algorithm**
The platform calculates comprehensive risk scores:

```
Risk Score = (Technique Severity × 0.4) + 
             (Anomaly Score × 0.3) + 
             (Prevalence × 0.2) + 
             (Context × 0.1)
```

**Risk Categories:**
- **Critical (0.9-1.0)**: Immediate action required
- **High (0.7-0.9)**: Priority investigation
- **Medium (0.4-0.7)**: Monitor and analyze
- **Low (0.0-0.4)**: Background monitoring

#### **Threat Intelligence Integration**
Enhance analysis with external threat intelligence:

- **IOC Matching**: Compare against known indicators
- **Reputation Scoring**: IP/domain reputation checks
- **Geolocation Analysis**: Geographic risk assessment
- **Temporal Analysis**: Attack timing patterns

## 📋 Reporting

### **Report Generation**

#### **Standard Reports**
Generate comprehensive security reports:

1. **Executive Summary**
   - High-level security overview
   - Key findings and recommendations
   - Risk trend analysis
   - Executive dashboard

2. **Technical Analysis**
   - Detailed anomaly findings
   - Model performance metrics
   - MITRE technique mappings
   - Technical recommendations

3. **Incident Response**
   - Specific incident details
   - Timeline reconstruction
   - Impact assessment
   - Response recommendations

#### **Custom Reports**
Create tailored reports for specific needs:

1. **Report Builder**
   - Drag-and-drop interface
   - Custom visualizations
   - Flexible data filtering
   - Template creation

2. **Scheduled Reports**
   - Automated generation
   - Email delivery
   - Custom schedules
   - Report archives

### **Export Options**

#### **Format Support**
- **PDF**: Professional formatted reports
- **Excel**: Data analysis and manipulation
- **CSV**: Raw data export
- **JSON**: API integration and automation

#### **Data Filtering**
Customize report content:
- Time range selection
- Risk level filtering
- Technique category focus
- Model-specific results

## ⚙️ Settings & Configuration

### **Data Source Configuration**

#### **Connection Settings**
Configure data source connections:

1. **Local Files**
   - Directory path configuration
   - File format settings
   - Auto-refresh intervals
   - Backup configurations

2. **Network Streams**
   - Real-time data feeds
   - Protocol configurations
   - Buffer settings
   - Connection monitoring

#### **Data Validation**
Ensure data quality and consistency:

- **Required Fields**: Specify mandatory data fields
- **Format Validation**: Check data structure
- **Quality Metrics**: Monitor data completeness
- **Error Handling**: Configure error responses

### **Model Management**

#### **Model Configuration**
Manage machine learning models:

1. **Default Parameters**
   - Set default algorithm settings
   - Performance thresholds
   - Training configurations
   - Validation rules

2. **Model Storage**
   - Save trained models
   - Version management
   - Performance tracking
   - Backup strategies

#### **Performance Monitoring**
Track model performance over time:

- **Accuracy Tracking**: Monitor prediction accuracy
- **Drift Detection**: Identify model degradation
- **Retraining Triggers**: Automatic model updates
- **Performance Alerts**: Notification thresholds

### **Notification Settings**

#### **Alert Configuration**
Configure security alerts and notifications:

1. **Alert Thresholds**
   - Risk score thresholds
   - Anomaly count limits
   - Performance degradation alerts
   - System health monitoring

2. **Notification Channels**
   - In-app notifications
   - Email alerts
   - Webhook integrations
   - Log file notifications

#### **User Preferences**
Customize your platform experience:

- **Dashboard Layout**: Customize widget arrangement
- **Default Views**: Set preferred starting pages
- **Data Refresh**: Configure update intervals
- **Theme Settings**: Light/dark mode preferences

## 🛠️ Advanced Features

### **API Integration**

#### **REST API**
Integrate with external systems:

```python
# Example API usage
import requests

# Get anomalies
response = requests.get('http://localhost:8501/api/anomalies')
anomalies = response.json()

# Submit new data
data = {"events": [...]}
response = requests.post('http://localhost:8501/api/analyze', json=data)
```

#### **Webhook Support**
Real-time notifications and integrations:

- **Alert Webhooks**: Send alerts to external systems
- **Data Webhooks**: Stream results to analytics platforms
- **Status Webhooks**: Monitor platform health
- **Custom Webhooks**: User-defined integrations

### **Automation Features**

#### **Automated Analysis**
Set up automated security analysis:

1. **Scheduled Scans**
   - Periodic anomaly detection
   - Automated model training
   - Report generation
   - Performance monitoring

2. **Rule-based Actions**
   - Automatic alert escalation
   - Custom response actions
   - Integration triggers
   - Workflow automation

#### **Machine Learning Pipelines**
Advanced ML workflow automation:

- **Data Preprocessing**: Automated feature engineering
- **Model Selection**: Automated algorithm selection
- **Hyperparameter Tuning**: Optimization automation
- **Ensemble Methods**: Combined model approaches

## 🆘 Troubleshooting

### **Common Issues**

#### **Performance Issues**
- **Slow Loading**: Check data source connectivity
- **High Memory Usage**: Reduce dataset size or increase resources
- **Model Training Failures**: Verify data quality and format

#### **Data Issues**
- **Missing Data**: Check file paths and permissions
- **Format Errors**: Validate JSON structure and required fields
- **Timestamp Issues**: Verify time format and timezone settings

#### **Connectivity Issues**
- **Network Errors**: Check firewall and network settings
- **API Failures**: Verify endpoint URLs and authentication
- **Database Connections**: Check credentials and connectivity

### **Error Resolution**

#### **Log Analysis**
Check platform logs for detailed error information:

```bash
# View application logs
docker-compose logs ndr-app

# View specific error logs
tail -f logs/errors.log

# View model training logs
tail -f logs/models.log
```

#### **Support Resources**
- **Documentation**: Comprehensive guides and references
- **Community Forum**: User community and discussions
- **Technical Support**: Contact system administrators
- **Issue Tracking**: Report bugs and feature requests

## 📞 Getting Help

### **Support Channels**
- **Documentation**: This user guide and technical documentation
- **In-app Help**: Built-in help tooltips and guides
- **Administrator**: Contact your system administrator
- **Community**: User forums and community resources

### **Best Practices**
- **Regular Monitoring**: Check platform daily for new alerts
- **Model Maintenance**: Retrain models monthly or when performance degrades
- **Data Quality**: Ensure clean, consistent data inputs
- **Security Awareness**: Stay informed about new threat techniques

This comprehensive user guide ensures all users can effectively utilize the NDR Platform for network security monitoring and threat detection.
