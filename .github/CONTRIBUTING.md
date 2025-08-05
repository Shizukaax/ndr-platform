# Contributing to NDR Platform

We love your input! We want to make contributing to the NDR Platform as easy and transparent as possible.

## Development Process

We use GitHub to sync code, to track issues and feature requests, as well as accept pull requests.

## Pull Requests

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Development Setup

```bash
# Clone your fork
git clone https://github.com/Shizukaax/ndr-platform.git
cd ndr-platform

# Setup development environment
python scripts/setup.py

# Install dependencies
pip install -r requirements.txt

# Run tests
python scripts/dev_utils.py test

# Start development server
python scripts/dev_utils.py dev-server
```

## Code Style

* We use Python Black for code formatting
* Follow PEP 8 guidelines
* Use type hints where possible
* Write descriptive commit messages

## Testing

* Write tests for new features
* Ensure all tests pass: `python scripts/dev_utils.py test`
* Run security scans: `python scripts/security_scanner.py all`

## Bug Reports

We use GitHub issues to track public bugs. Report a bug by opening a new issue using our bug report template.

## Feature Requests

We welcome feature requests! Please provide:
- Clear description of the feature
- Why it would be useful
- How it should work

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
