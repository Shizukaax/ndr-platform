# NDR Platform - Project Organization

**Author:** [Shizukaax](https://github.com/Shizukaax) | **Repository:** [ndr-platform](https://github.com/Shizukaax/ndr-platform)

## 🏗️ Architecture Overview

The NDR Platform follows a modular, layered architecture designed for scalability, maintainability, and security.

## 📁 Complete Directory Structure

```
ndr-platform/                          # 🏠 Root Project Directory
│
├── 📁 .github/                         # 🐙 GitHub Configuration & Workflows
│   ├── workflows/
│   │   └── ci.yml                      # Automated CI/CD pipeline
│   ├── README.md                       # Main project documentation
│   ├── CONTRIBUTING.md                 # Contribution guidelines
│   ├── SECURITY.md                     # Security policy & disclosure
│   ├── ISSUE_TEMPLATE.md               # Bug report template
│   └── PULL_REQUEST_TEMPLATE.md        # Pull request template
│
├── 📁 deployment/                      # 🚀 Deployment Configuration
│   ├── docker-compose.yml              # Container orchestration
│   ├── Dockerfile                      # Application container definition
│   ├── nginx.conf                      # Reverse proxy configuration
│   └── README.md                       # Deployment documentation
│
├── 📁 scripts/                         # 🛠️ Management & Utility Scripts
│   ├── setup.py                        # Platform initialization & setup
│   ├── deploy.py                       # Deployment automation
│   ├── health_check.py                 # System health monitoring
│   ├── security_scanner.py             # Security auditing & scanning
│   ├── model_manager.py                # ML model lifecycle management
│   ├── data_manager.py                 # Data validation & operations
│   ├── backup.py                       # Backup & restore utilities
│   ├── log_analyzer.py                 # Log analysis & monitoring
│   └── dev_utils.py                    # Development tools & utilities
│
├── 📁 app/                             # 🎯 Streamlit Application Layer
│   ├── main.py                         # Main application entry point
│   ├── about.py                        # About page & information
│   ├── settings.py                     # Application settings
│   │
│   ├── 📁 assets/                      # 🎨 Static Resources
│   │   └── network_logo.png           # Application branding
│   │
│   ├── 📁 components/                  # 🧩 Reusable UI Components
│   │   ├── chart_factory.py           # Chart creation utilities
│   │   ├── data_source_selector.py    # Data source selection
│   │   ├── download_utils.py           # File download utilities
│   │   ├── error_handler.py           # Error handling components
│   │   ├── explainers.py              # Model explanation UI
│   │   ├── file_utils.py               # File manipulation tools
│   │   ├── model_cards.py             # Model information display
│   │   ├── model_comparison.py        # Model comparison widgets
│   │   ├── model_retraining.py        # Model retraining interface
│   │   ├── report_generator.py        # Report generation UI
│   │   ├── search_filter.py           # Data filtering components
│   │   └── visualization.py           # Data visualization tools
│   │
│   ├── 📁 pages/                       # 📄 Streamlit Page Modules
│   │   ├── analytics_dashboard.py     # Advanced analytics dashboard
│   │   ├── anomaly_detection.py       # Main detection interface
│   │   ├── auto_labeling.py           # Automatic data labeling
│   │   ├── data_upload.py             # Data ingestion interface
│   │   ├── explain_feedback.py        # Explanation & feedback
│   │   ├── file_diagnostics.py        # File analysis tools
│   │   ├── mitre_mapping.py           # MITRE ATT&CK mapping
│   │   ├── model_comparison.py        # Model performance comparison
│   │   ├── model_management.py        # Model lifecycle management
│   │   ├── real_time_monitoring.py    # Real-time monitoring
│   │   └── reporting.py               # Report generation
│   │
│   └── 📁 state/                       # 🔄 Session State Management
│       └── session_state.py           # Session utilities
│
├── 📁 core/                            # ⚙️ Core Business Logic Layer
│   ├── 📁 models/                      # 🤖 ML Model Implementations
│   │   ├── base_model.py               # Abstract base model
│   │   ├── isolation_forest.py        # Isolation Forest detector
│   │   ├── dbscan.py                   # DBSCAN clustering
│   │   ├── knn.py                      # K-Nearest Neighbors
│   │   ├── local_outlier_factor.py    # LOF anomaly detector
│   │   ├── one_class_svm.py           # One-Class SVM
│   │   ├── ensemble.py                 # Ensemble methods
│   │   └── hdbscan_detector.py        # HDBSCAN clustering
│   │
│   ├── 📁 explainers/                  # 🔍 Model Interpretability
│   │   ├── base_explainer.py          # Abstract explainer base
│   │   ├── shap_explainer.py          # SHAP explanations
│   │   ├── lime_explainer.py          # LIME explanations
│   │   └── explainer_factory.py       # Explainer factory pattern
│   │
│   ├── data_manager.py                 # Data processing pipeline
│   ├── model_manager.py                # ML model lifecycle
│   ├── mitre_mapper.py                 # MITRE ATT&CK integration
│   ├── security_intelligence.py       # Security analytics
│   ├── advanced_analytics.py          # Advanced analysis
│   ├── auto_analysis.py               # Automated analysis
│   ├── auto_labeler.py                # Automatic labeling
│   ├── config_loader.py               # Configuration management
│   ├── data_validator.py              # Data validation
│   ├── feedback_manager.py            # User feedback system
│   ├── file_watcher.py                # File monitoring
│   ├── logging_config.py              # Logging configuration
│   ├── notification_service.py        # Notification system
│   ├── risk_scorer.py                 # Risk assessment
│   ├── search_engine.py               # Search functionality
│   └── session_manager.py             # Session management
│
├── 📁 docs/                            # 📚 Comprehensive Documentation
│   ├── README.md                       # Documentation index
│   ├── USER_GUIDE.md                   # Complete user manual
│   ├── CONFIGURATION_GUIDE.md          # Setup & configuration
│   ├── DEPLOYMENT_GUIDE.md             # Production deployment
│   ├── API_DOCUMENTATION.md            # API reference guide
│   ├── PROJECT_ORGANIZATION.md         # This architecture document
│   ├── LOGGING_GUIDE.md                # Logging configuration
│   └── FIXES_SUMMARY.md                # Bug fixes & updates
│
├── 📁 examples/                        # 📝 Usage Examples & Tutorials
│   ├── basic_usage.py                  # Simple usage example
│   └── README.md                       # Examples documentation
│
├── 📁 tools/                           # 🔧 Utility Modules
│   ├── data_saver.py                   # Data persistence utilities
│   ├── file_diagnostics.py            # File validation tools
│   └── README.md                       # Tools documentation
│
├── 📁 tests/                           # 🧪 Test Suite
│   ├── test_*.py                       # Unit tests
│   └── README.md                       # Testing documentation
│
├── 📁 data/                            # 💾 Data Storage
│   ├── examples/                       # Sample datasets
│   │   ├── sample_network_data.json   # Basic network data
│   │   ├── mitre_test_data.json       # MITRE testing data
│   │   └── synthetic_packets.json     # Synthetic packet data
│   └── realtime/                       # Live data ingestion
│
├── 📁 logs/                            # 📋 Application Logs
│   ├── app.log                         # Main application log
│   ├── anomaly_detection.log          # Detection system log
│   ├── model_manager.log              # Model management log
│   └── errors.log                      # Error tracking
│
├── 📁 models/                          # 🎯 Trained ML Models
│   ├── *.pkl                          # Serialized models
│   ├── *_metadata.json                # Model metadata
│   └── backups/                        # Model backups
│
├── 📁 reports/                         # 📊 Generated Reports
├── 📁 results/                         # 📈 Analysis Results
├── 📁 feedback/                        # 💬 User Feedback Data
├── 📁 cache/                           # ⚡ Application Cache
├── 📁 config/                          # ⚙️ Configuration Files
│   ├── config.yaml                     # Main configuration
│   └── mitre_attack_data.json         # MITRE ATT&CK data
│
├── CHANGELOG.md                        # 📝 Version history
├── requirements.txt                    # 📦 Python dependencies
├── .gitignore                          # 🚫 Git ignore rules
├── .env.example                        # 🔧 Environment template
└── run.py                              # 🚀 Application launcher
```

## 🏛️ Architectural Layers

### 1. 🎯 **Presentation Layer** (`app/`)
- **Purpose**: User interface and interaction
- **Technology**: Streamlit, Plotly, HTML/CSS
- **Components**: Pages, Components, Assets, State Management

### 2. ⚙️ **Business Logic Layer** (`core/`)
- **Purpose**: Core application functionality
- **Technology**: Python, Scikit-learn, Custom algorithms
- **Components**: ML Models, Data Processing, Analytics, MITRE Integration

### 3. 🛠️ **Utility Layer** (`scripts/`, `tools/`)
- **Purpose**: System management and utilities
- **Technology**: Python, Shell scripts, Automation tools
- **Components**: Setup, Deployment, Monitoring, Development tools

### 4. 💾 **Data Layer** (`data/`, `models/`, `results/`)
- **Purpose**: Data storage and persistence
- **Technology**: JSON, Pickle, File system
- **Components**: Raw data, Trained models, Analysis results

### 5. 🚀 **Infrastructure Layer** (`deployment/`, `.github/`)
- **Purpose**: Deployment and CI/CD
- **Technology**: Docker, GitHub Actions, Nginx
- **Components**: Containers, Workflows, Configuration

## 🔄 Data Flow Architecture

```
📥 Data Ingestion → 🔍 Processing → 🤖 ML Analysis → 📊 Visualization → 📋 Reporting
    ↓                    ↓              ↓              ↓              ↓
data_manager.py   →  core/models/  →  explainers/  →  components/  →  reporting.py
```

## 🔧 Key Design Patterns

### 1. **Factory Pattern**
- `explainer_factory.py` - Creates appropriate explainer instances
- `chart_factory.py` - Standardized chart creation

### 2. **Repository Pattern**
- `data_manager.py` - Data access abstraction
- `model_manager.py` - Model persistence management

### 3. **Observer Pattern**
- `session_state.py` - State change notifications
- `file_watcher.py` - File system monitoring

### 4. **Strategy Pattern**
- `base_model.py` - Interchangeable ML algorithms
- `base_explainer.py` - Different explanation strategies

## 🛡️ Security Architecture

### **Input Validation**
- File upload validation
- Data format verification
- Parameter sanitization

### **Access Control**
- Session-based authentication
- Role-based permissions
- Secure file handling

### **Data Protection**
- Encrypted data transmission
- Secure temporary file handling
- Audit logging

## 📈 Scalability Considerations

### **Horizontal Scaling**
- Containerized deployment with Docker
- Load balancing with Nginx
- Microservices architecture readiness

### **Vertical Scaling**
- Optimized memory usage
- Efficient data processing
- Model caching strategies

### **Performance Optimization**
- Lazy loading of models
- Caching frequently accessed data
- Asynchronous processing where possible

## 🔮 Future Architecture Plans

### **Microservices Migration**
- API-first design
- Service decomposition
- Event-driven architecture

### **Cloud-Native Features**
- Kubernetes deployment
- Auto-scaling capabilities
- Cloud storage integration

### **Enhanced Security**
- OAuth integration
- Certificate-based authentication
- Advanced audit logging

---

**📝 Note**: This architecture document is regularly updated to reflect the current state of the NDR Platform. For the latest version, please refer to the [GitHub repository](https://github.com/Shizukaax/ndr-platform).
