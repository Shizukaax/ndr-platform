# Contributing to NDR Platform v2.1.0

We love your input! We want to make contributing to the production-ready NDR Platform as easy and transparent as possible.

## Development Process

We use GitHub to sync code, to track issues and feature requests, as well as accept pull requests. The platform is currently in production status with critical fixes applied.

## Pull Requests

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add comprehensive tests.
3. If you've changed APIs, update both inline documentation and guides/.
4. Ensure the complete test suite passes including encoding and data validation tests.
5. Make sure your code lints and follows security best practices.
6. Verify no NaN values or Arrow serialization issues are introduced.
7. Issue that pull request!

## Development Setup

```bash
# Clone your fork
git clone https://github.com/Shizukaax/ndr-platform.git
cd ndr-platform

# Setup development environment (v2.1.0)
python scripts/setup.py

# Install production dependencies
pip install -r requirements.txt

# Run comprehensive test suite (includes encoding/NaN fixes)
python scripts/dev_utils.py test

# Run specific fix validation tests
python test_encoding_fixes.py
python test_results_saving.py
python test_feedback_dirs.py

# Start development server
python scripts/dev_utils.py dev-server
```

## Code Style & Standards

* We use Python Black for code formatting
* Follow PEP 8 guidelines  
* Use type hints for all functions
* Write descriptive commit messages
* **Critical**: Validate all numeric inputs for NaN values
* **Critical**: Use proper Unicode/UTF-8 encoding for all file operations
* **Critical**: Test Arrow serialization compatibility

## Testing Requirements

* Write comprehensive tests for new features including edge cases
* Ensure all tests pass: `python scripts/dev_utils.py test`
* Run encoding validation: `python test_encoding_fixes.py`  
* Run results saving tests: `python test_results_saving.py`
* Run security scans: `python scripts/security_scanner.py all`
* Validate directory structure: `python scripts/verify_structure.py`

## Bug Reports

We use GitHub issues to track public bugs. Report a bug by opening a new issue using our bug report template.

## Feature Requests

We welcome feature requests! Please provide:
- Clear description of the feature
- Why it would be useful
- How it should work

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
