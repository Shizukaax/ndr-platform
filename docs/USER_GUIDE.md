# NDR Platform - User Guide

## 👥 Complete User Documentation

### 🎯 Overview

The **Network Detection and Response (NDR) Platform** is an advanced cybersecurity analytics system that uses machine learning to detect network anomalies and map them to MITRE ATT&CK techniques. This guide provides comprehensive instructions for all user types.

## 🚀 Getting Started

### 🔐 Accessing the Platform

1. **Open your web browser** and navigate to the platform URL
   - Development: `http://localhost:8501`
   - Production: `https://your-domain.com`

2. **Initial Setup** (First-time users)
   - No authentication required for local deployment
   - For production, contact your administrator for access credentials

### 🏠 Platform Interface

The NDR Platform features an intuitive sidebar navigation with the following sections:

```
📊 Main Dashboard          # Real-time overview and metrics
🔍 Anomaly Detection      # AI-powered threat detection
📈 Advanced Analytics     # Deep-dive analysis tools
🎯 MITRE ATT&CK Mapping   # Threat technique mapping  
📋 Reporting             # Generate and export reports
⚙️ Settings              # Configuration and preferences
ℹ️ About                 # Platform information
```

## 📊 Main Dashboard

### **Overview Section**
The dashboard provides real-time insights into your network security posture:

- **Live Metrics**
  - Total events processed
  - Active anomalies detected
  - Model performance scores
  - System health status

- **Quick Actions**
  - Start new analysis
  - View recent alerts
  - Access saved models
  - Generate quick reports

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
