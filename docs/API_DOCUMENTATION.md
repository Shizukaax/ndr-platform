# NDR Platform - API Documentation

## 🚀 Core API Reference

### 📊 Data Management API

#### DataManager Class
```python
from core.data_manager import DataManager

# Initialize data manager
data_manager = DataManager()

# Auto-load data from configured directory
success = data_manager.auto_load_data()

# Get available data sources
sources = data_manager.get_available_sources()

# Load specific JSON file
data = data_manager.load_json_file(file_path)
```

**Methods:**
- `auto_load_data()` → `bool` - Load data from default directory
- `get_available_sources()` → `List[str]` - Get available data sources
- `load_json_file(path)` → `DataFrame` - Load specific JSON file
- `initialize_default_data()` → `None` - Initialize with default data

#### Data Validation
```python
from core.data_validator import DataValidationService

validator = DataValidationService()

# Validate data for analysis
is_valid = validator.validate_data_for_analysis(dataframe)

# Get validation report
report = validator.get_validation_report(dataframe)
```

### 🤖 Model Management API

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
