# NDR Platform v2.1.0 - API Documentation

## 🚀 Core API Reference *(Production Ready - August 2025)*

### 📋 **API Status Summary**

Following recent critical fixes, all APIs have been enhanced for production reliability:

- **✅ ModelManager API:** Fixed results saving with proper anomaly indices
- **✅ DataManager API:** Enhanced NaN handling and Arrow compatibility  
- **✅ ConfigLoader API:** Robust configuration management
- **✅ FeedbackManager API:** Fixed directory structure compliance
- **✅ Explainer APIs:** Comprehensive error handling for missing values

---

## 🧠 **Core Model Management API**

### **ModelManager Class** *(Enhanced)*

```python
from core.model_manager import ModelManager

# Initialize model manager
model_manager = ModelManager()

# ✅ Fixed: Train model with proper results saving
results = model_manager.train_model(
    data=dataframe,
    model_type="isolation_forest",
    save_results=True  # Now properly saves with anomaly_indices
)

# ✅ Enhanced: Get available models
models = model_manager.get_available_models()

# ✅ Fixed: Load trained model
model = model_manager.load_model("IsolationForest")

# ✅ Enhanced: Ensemble prediction
ensemble_results = model_manager.predict_ensemble(data)
```

**Methods:**
- `train_model(data, model_type, **kwargs)` → `dict` - ✅ Fixed results saving
- `get_available_models()` → `List[str]` - List trained models
- `load_model(model_name)` → `object` - Load specific model
- `predict_ensemble(data)` → `dict` - Multi-model predictions
- `save_results(results, filename)` → `bool` - ✅ Fixed directory compliance

**Enhanced Return Format:**
```python
{
    'anomalies': DataFrame,           # Detected anomalies
    'anomaly_indices': List[int],     # ✅ Fixed: Now included
    'model_scores': List[float],      # Anomaly scores
    'model_metadata': dict,           # Model information
    'execution_time': float,          # Processing time
    'total_anomalies': int           # Count of anomalies
}
```

---

## 📊 **Data Management API**

### **DataManager Class** *(Enhanced)*

```python
from core.data_manager import DataManager

# Initialize with config-driven paths
data_manager = DataManager()

# ✅ Enhanced: Auto-load with validation
success = data_manager.auto_load_data()

# ✅ Fixed: Get sources from configured directory
sources = data_manager.get_available_sources()

# ✅ Enhanced: Load with NaN handling
data = data_manager.load_json_file(file_path)
```

**Methods:**
- `auto_load_data()` → `bool` - Load from configured data directory
- `get_available_sources()` → `List[str]` - Available data files
- `load_json_file(path)` → `DataFrame` - Load with error handling
- `validate_data(data)` → `bool` - Data integrity checks
- `clean_dataframe_for_arrow(df)` → `DataFrame` - ✅ Arrow compatibility

---

## 🔍 **Explainer API** *(Enhanced)*

### **SHAP Explainer** *(Fixed)*

```python
from core.explainers.shap_explainer import SHAPExplainer

# Initialize explainer
explainer = SHAPExplainer()

# ✅ Fixed: Generate explanations with NaN handling
explanations = explainer.explain_predictions(
    model=trained_model,
    data=feature_data,
    predictions=anomaly_predictions,
    handle_missing=True  # ✅ Enhanced NaN handling
)

# ✅ Enhanced: Feature importance with clean display
feature_importance = explainer.get_feature_importance(
    explanations,
    clean_display=True  # ✅ Proper port value formatting
)
```

**Methods:**
- `explain_predictions(model, data, predictions)` → `dict` - ✅ Enhanced explanations
- `get_feature_importance(explanations)` → `DataFrame` - ✅ Clean formatting
- `plot_explanations(explanations)` → `Figure` - Visualization
- `handle_missing_values(data)` → `DataFrame` - ✅ NaN processing

---

## ⚙️ **Configuration API**

### **ConfigLoader Class** *(Enhanced)*

```python
from core.config_loader import load_config, ConfigLoader

# ✅ Enhanced: Load with validation
config = load_config()

# ✅ Fixed: Get configured paths
data_dir = config.get('system.data_dir', 'data')
results_dir = config.get('system.results_dir', 'data/results')
feedback_dir = config.get('feedback.storage_dir', 'data/feedback')

# ✅ Enhanced: Validate configuration
config_loader = ConfigLoader()
is_valid = config_loader.validate_config(config)
```

**Methods:**
- `load_config(config_path=None)` → `dict` - Load configuration
- `validate_config(config)` → `bool` - ✅ Configuration validation
- `get_config_value(key, default=None)` → `Any` - Nested key access
- `reload_config()` → `dict` - Hot reload configuration

---

## 💬 **Feedback Management API** *(Fixed)*

### **FeedbackManager Class** *(Enhanced)*

```python
from core.feedback_manager import FeedbackManager

# ✅ Initialize with config-driven paths
feedback_manager = FeedbackManager()

# ✅ Fixed: Save to configured directory
success = feedback_manager.save_feedback(
    prediction_id="anomaly_123",
    user_feedback="true_positive",
    confidence=0.95,
    notes="Confirmed network intrusion"
)

# ✅ Enhanced: Load feedback with validation
feedback_data = feedback_manager.load_feedback_history()
```

**Methods:**
- `save_feedback(prediction_id, feedback, **kwargs)` → `bool` - ✅ Config-compliant saving
- `load_feedback_history()` → `DataFrame` - Historical feedback
- `get_feedback_stats()` → `dict` - Feedback statistics
- `export_feedback(format='json')` → `str` - Export functionality

---

## 🔧 **Utility APIs**

### **Data Cleaning Utilities** *(New)*

```python
from app.pages.explain_feedback import clean_dataframe_for_arrow

# ✅ Clean DataFrame for Streamlit display
clean_df = clean_dataframe_for_arrow(original_df)

# ✅ Handle port values specifically
def clean_port_values(df, port_column='dest_port'):
    """Clean port values for display"""
    df[port_column] = df[port_column].fillna('N/A')
    df[port_column] = df[port_column].replace([np.nan, 'nan'], 'N/A')
    return df
```

### **Directory Management** *(Enhanced)*

```python
from run import create_required_directories

# ✅ Create directories using config.yaml
success = create_required_directories()

# Manual directory creation with config
import os
from core.config_loader import load_config

config = load_config()
required_dirs = [
    config.get('system.data_dir'),
    config.get('system.results_dir'),
    config.get('feedback.storage_dir'),
    config.get('system.logs_dir')
]

for dir_path in required_dirs:
    os.makedirs(dir_path, exist_ok=True)
```

---

## 🚨 **Error Handling API**

### **Enhanced Error Management**

```python
from core.logging_config import setup_logging
import logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# ✅ Comprehensive error handling pattern
try:
    result = model_manager.train_model(data, "isolation_forest")
    if 'anomaly_indices' not in result:
        logger.error("Missing anomaly_indices in results")
        result['anomaly_indices'] = []
except Exception as e:
    logger.error(f"Model training failed: {e}")
    return {"error": str(e), "success": False}
```

### **Data Validation Patterns**

```python
# ✅ Validate data before processing
def validate_input_data(df):
    """Validate DataFrame for ML processing"""
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")
    
    # Check for required columns
    required_cols = ['source_ip', 'dest_ip', 'timestamp']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # ✅ Handle NaN values
    df = df.dropna(subset=required_cols)
    
    return df
```

---

## 📊 **API Response Formats**

### **Standard Response Structure**

```python
# Success Response
{
    "success": True,
    "data": {...},
    "message": "Operation completed successfully",
    "timestamp": "2025-08-XX 12:00:00",
    "execution_time": 1.23
}

# Error Response
{
    "success": False,
    "error": "Error description",
    "error_code": "DATA_VALIDATION_ERROR",
    "timestamp": "2025-08-XX 12:00:00",
    "details": {...}
}
```

### **Model Training Response** *(Enhanced)*

```python
{
    "success": True,
    "model_type": "isolation_forest",
    "anomalies": DataFrame,
    "anomaly_indices": [1, 5, 23, 45],  # ✅ Fixed: Always included
    "total_anomalies": 4,
    "model_scores": [0.95, 0.87, 0.92, 0.89],
    "execution_time": 2.34,
    "model_metadata": {
        "n_estimators": 100,
        "contamination": 0.1,
        "training_samples": 1000
    }
}
```

---

## 🔗 **Integration Examples**

### **Complete Workflow Example**

```python
from core.model_manager import ModelManager
from core.data_manager import DataManager
from core.feedback_manager import FeedbackManager
from core.explainers.shap_explainer import SHAPExplainer

# ✅ Complete workflow with fixed components
def run_anomaly_detection_workflow():
    # 1. Load data with config-driven paths
    data_manager = DataManager()
    data = data_manager.auto_load_data()
    
    # 2. Train model with proper results saving
    model_manager = ModelManager()
    results = model_manager.train_model(data, "isolation_forest")
    
    # 3. Generate explanations with NaN handling
    explainer = SHAPExplainer()
    explanations = explainer.explain_predictions(
        model=results['model'],
        data=data,
        predictions=results['anomaly_indices']
    )
    
    # 4. Save feedback to configured directory
    feedback_manager = FeedbackManager()
    feedback_manager.save_feedback(
        prediction_id="batch_001",
        feedback="processed",
        results_summary=results
    )
    
    return {
        "anomalies": results['anomalies'],
        "explanations": explanations,
        "success": True
    }
```

---

## 📖 **Related Documentation**

For implementation details:
- **User Guide:** See `USER_GUIDE.md`
- **Configuration:** See `CONFIGURATION_GUIDE.md`
- **Deployment:** See `DEPLOYMENT_GUIDE.md`
- **Project Structure:** See `PROJECT_ORGANIZATION.md`

**API Status:** Production Ready v2.1.0 with all critical fixes applied.

#### AnomalyTracker Class
```python
from core.anomaly_tracker import AnomalyTracker

# Initialize anomaly tracker
tracker = AnomalyTracker(storage_dir="cache/anomalies")

# Record new anomaly detection
detection_record = tracker.record_anomaly_detection(
    anomalies=anomaly_dataframe,
    model_type="ensemble",
    confidence_threshold=0.7,
    source_file="packet_capture.json",
    total_packets=1000
)

# Get recent anomalies
recent = tracker.get_recent_anomalies(hours=24)

# Get baseline deviation analysis
baseline_info = tracker.get_baseline_deviation(current_rate=5.2)

# Acknowledge an anomaly
tracker.acknowledge_anomaly(
    detection_id="20250806_120303_ensemble",
    note="Acknowledged - false positive"
)

# Get anomaly trends
trends = tracker.get_anomaly_trends(days=7)
```

**Key Methods:**
- `record_anomaly_detection()` → `Dict` - Record new anomaly detection with full context
- `get_recent_anomalies(hours)` → `List[Dict]` - Retrieve anomalies within time range
- `get_baseline_deviation(rate)` → `Dict` - Compare current rate to learned baseline
- `acknowledge_anomaly(id, note)` → `bool` - Mark anomaly as reviewed
- `get_anomaly_trends(days)` → `Dict` - Get trend analysis and statistics

**Detection Record Structure:**
```python
detection_record = {
    "detection_id": "20250806_120303_ensemble",
    "timestamp": "2025-08-06T12:03:03.123456",
    "model_type": "ensemble",
    "anomaly_count": 5,
    "total_packets": 1000,
    "anomaly_rate": 0.5,
    "confidence_threshold": 0.7,
    "severity": "medium",  # low, medium, high, critical
    "source_file": "packet_capture.json",
    "status": "new",  # new, acknowledged, resolved
    "details": {
        "anomaly_types": ["size_anomaly", "external_connection"],
        "affected_ports": [80, 443, 22],
        "affected_ips": ["192.168.1.10", "10.0.0.5"],
        "size_anomalies": {"count": 3, "max_size": 15000},
        "external_connections": 2
    }
}
```

### 🤖 Enhanced Model Management API

#### ModelManager Class
```python
from core.model_manager import ModelManager

model_manager = ModelManager()

# Train model
results = model_manager.train_model(
    data=dataframe,
    model_type="IsolationForest",
    parameters={"contamination": 0.1}
)

# Get available models
models = model_manager.get_available_models()

# Load trained model
model = model_manager.load_model(model_name)
```

**Key Methods:**
- `train_model(data, model_type, params)` → `dict` - Train ML model
- `predict_anomalies(model, data)` → `np.array` - Detect anomalies
- `save_model(model, name)` → `str` - Save trained model
- `get_model_metadata(name)` → `dict` - Get model information

#### Available Model Types
- `IsolationForest` - Isolation Forest anomaly detection
- `LocalOutlierFactor` - Local Outlier Factor
- `OneClassSVM` - One-Class Support Vector Machine
- `DBSCAN` - Density-based clustering
- `HDBSCAN` - Hierarchical DBSCAN
- `KNN` - K-Nearest Neighbors
- `Ensemble` - Multiple model ensemble

### 🛡️ MITRE ATT&CK Integration API

#### MitreMapper Class
```python
from core.mitre_mapper import MitreMapper

mitre_mapper = MitreMapper()

# Map anomalies to MITRE techniques
mappings = mitre_mapper.map_anomalies_to_techniques(anomalies)

# Get technique information
technique_info = mitre_mapper.get_technique_info(technique_id)

# Generate risk scores
risk_scores = mitre_mapper.calculate_risk_scores(mappings)
```

#### AutoAnalysisService Class
```python
from core.auto_analysis import AutoAnalysisService

auto_analysis = AutoAnalysisService()

# Run complete automatic analysis
results = auto_analysis.run_automatic_analysis(anomaly_data)

# Get analysis summary
summary = auto_analysis.get_analysis_summary(results)
```

### 📊 Analytics & Visualization API

#### ChartFactory Class
```python
from app.components.chart_factory import ChartFactory

factory = ChartFactory()

# Create anomaly score distribution
chart = factory.create_anomaly_score_distribution(scores)

# Create feature importance plot
importance_chart = factory.create_feature_importance(features, scores)

# Create time series analysis
timeline = factory.create_anomaly_timeline(data, time_column)
```

#### Visualization Components
```python
from app.components.visualization import (
    plot_anomaly_scores,
    plot_network_graph,
    plot_protocol_pie
)

# Plot anomaly score distribution
fig = plot_anomaly_scores(data, scores, threshold)

# Create network topology visualization
network_fig = plot_network_graph(connections)

# Generate protocol distribution pie chart
protocol_fig = plot_protocol_pie(data)
```

### 🔔 Notification & Alerting API

#### NotificationService Class
```python
from core.notification_service import NotificationService

notification_service = NotificationService()

# Show critical alert
notification_service.show_critical_alert(
    title="Security Threat Detected",
    message="High-risk anomaly detected in network traffic",
    details=threat_details
)

# Display analysis results
notification_service.show_auto_analysis_results(analysis_data)

# Add notification to queue
notification_service.add_notification(
    level="warning",
    message="Model performance degraded",
    category="model_performance"
)
```

### 📝 Logging API

#### Logging Configuration
```python
from core.logging_config import setup_logging, get_logger

# Initialize logging
logger = setup_logging(log_dir="logs/", log_level=logging.INFO)

# Get module-specific logger
data_logger = get_logger("data_processing")
model_logger = get_logger("model_manager")

# Log messages
data_logger.info("Processing 1000 records")
model_logger.error("Model training failed", exc_info=True)
```

#### Available Loggers
- `data_manager` - Data management operations
- `model_manager` - ML model operations
- `anomaly_detection` - Detection processes
- `visualization` - Chart generation
- `streamlit_app` - Application events
- `app_startup` - Startup messages

### 🔍 Search & Filtering API

#### SearchEngine Class
```python
from core.search_engine import SearchEngine

search_engine = SearchEngine()

# Search anomalies
results = search_engine.search_anomalies(
    query="high_severity AND network_traffic",
    filters={"score_range": (0.8, 1.0)}
)

# Advanced filtering
filtered_data = search_engine.filter_data(
    data=dataframe,
    conditions={"protocol": "TCP", "port": [80, 443]}
)
```

### 💾 Session State Management API

#### SessionStateManager Class
```python
from core.session_manager import SessionStateManager

session_manager = SessionStateManager()

# Store analysis results
session_manager.set_analysis_results(results)

# Get session data
data = session_manager.get_session_data("model_results")

# Clear session
session_manager.clear_session()
```

## 🔧 Configuration API

### Config Loader
```python
from core.config_loader import load_config

# Load main configuration
config = load_config()

# Get specific section
data_config = config.get('data_source', {})
model_config = config.get('models', {})

# Environment-specific config
prod_config = load_config(environment='production')
```

### Environment Variables
```python
import os

# Application settings
STREAMLIT_PORT = os.getenv('STREAMLIT_SERVER_PORT', '8501')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
DATA_DIR = os.getenv('DATA_DIRECTORY', '/app/data')

# Security settings
ALERT_THRESHOLD = float(os.getenv('ALERT_THRESHOLD', '0.8'))
MITRE_ENABLED = os.getenv('MITRE_MAPPING_ENABLED', 'true').lower() == 'true'
```

## 📄 Report Generation API

### Report Generator
```python
from app.components.report_generator import (
    generate_anomaly_report,
    generate_comparison_report,
    generate_executive_summary
)

# Generate anomaly report
html_report = generate_anomaly_report(
    data=dataframe,
    anomalies=anomaly_df,
    scores=scores,
    threshold=threshold,
    model_name="IsolationForest",
    features=feature_list,
    plots=chart_list
)

# Create executive summary
summary = generate_executive_summary(
    data=dataframe,
    anomalies=anomaly_df,
    model_name="IsolationForest",
    threshold=threshold
)
```

## 🧪 Testing API

### Test Utilities
```python
# Run platform validation
from tests.test_fixes import test_threshold_handling
result = test_threshold_handling()

# Test logging configuration
from tests.test_logging import test_logging_initialization
logging_ok = test_logging_initialization()
```

## 🚨 Error Handling Patterns

### Standard Error Handling
```python
try:
    result = risky_operation()
except SpecificException as e:
    logger.error(f"Operation failed: {e}", exc_info=True)
    notification_service.show_error(f"Error: {e}")
    return None
except Exception as e:
    logger.critical(f"Unexpected error: {e}", exc_info=True)
    raise
```

### Graceful Degradation
```python
def robust_function(data):
    try:
        return primary_method(data)
    except PrimaryException:
        logger.warning("Primary method failed, using fallback")
        return fallback_method(data)
    except Exception as e:
        logger.error(f"All methods failed: {e}")
        return default_value
```

## 📈 Performance Guidelines

### Efficient Data Processing
```python
# Use chunking for large datasets
def process_large_dataset(data, chunk_size=1000):
    for chunk in pd.read_csv(data, chunksize=chunk_size):
        yield process_chunk(chunk)

# Cache expensive operations
@lru_cache(maxsize=128)
def expensive_calculation(params):
    return complex_computation(params)
```

### Memory Management
```python
# Clear large objects when done
del large_dataframe
gc.collect()

# Use generators for large sequences
def data_generator(source):
    for item in source:
        yield process_item(item)
```

This API documentation provides comprehensive coverage of the NDR Platform's programmatic interfaces, enabling developers to integrate, extend, and maintain the system effectively.
