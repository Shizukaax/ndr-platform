# Tools & Utilities

This directory contains utility modules and helper functions used throughout the NDR Platform.

## Available Tools

### 📊 `data_saver.py`
- Data persistence utilities
- Export functions
- Format conversion tools

### 🔍 `file_diagnostics.py`
- File validation and diagnostics
- Health checking utilities
- Content analysis tools

## Usage

These tools are used internally by the platform but can also be used standalone:

```python
from tools.data_saver import save_results
from tools.file_diagnostics import validate_file

# Save analysis results
save_results(data, "output.json")

# Validate file
is_valid = validate_file("data.json")
```

## Development

When adding new tools:
1. Follow the existing patterns
2. Add proper documentation
3. Include error handling
4. Add tests in `tests/`
