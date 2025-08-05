# NDR Platform Tests

This directory contains test files for validating platform fixes and configuration.

## 🧪 Test Files

### test_fixes.py
Comprehensive validation of critical platform fixes:
- Threshold handling with fallbacks
- Anomaly session state management  
- Format string safety
- Data column mapping
- Configuration file validation

**Usage:**
```bash
cd newnewapp
python tests/test_fixes.py
```

### test_logging.py
Logging configuration validation:
- Duplication prevention
- Handler management
- Module-specific loggers
- Streamlit reload compatibility

**Usage:**
```bash
cd newnewapp
python tests/test_logging.py
```

## 🎯 Test Results

Both test suites should pass with all green checkmarks:
- ✅ 5/5 tests passed (test_fixes.py)
- ✅ All logging tests passed (test_logging.py)

## 📋 Running Tests

From the project root directory:
```bash
# Run platform fixes validation
python tests/test_fixes.py

# Run logging configuration test  
python tests/test_logging.py

# Run all tests (if pytest is installed)
pytest tests/
```

## 🔍 Test Coverage

The tests validate:
- Critical error fixes
- Session state handling
- Data compatibility
- Logging system integrity
- Configuration completeness

These tests ensure the platform is stable and production-ready.
