---
name: Bug Report - NDR Platform v2.1.0
about: Create a report to help us improve the production-ready platform
title: '[BUG] '
labels: bug
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment (please complete the following information):**
 - OS: [e.g. Windows 11, Ubuntu 22.04]
 - Python Version: [e.g. 3.11.0]
 - Browser [e.g. Chrome, Firefox]
 - **NDR Platform Version: v2.1.0 (Production)**

**Diagnostic Information**
Please run the following and include output:
```bash
# Health check
python scripts/health_check.py

# Structure validation
python scripts/verify_structure.py

# Critical fixes validation (if applicable)
python test_encoding_fixes.py
python test_results_saving.py
python test_feedback_dirs.py
```

**Logs**
Please check `logs/` directory and include relevant entries:
```
paste log entries here
```

**Additional context**
- Have you verified critical fixes are applied? (Run diagnostic commands above)
- Does this issue involve NaN values, encoding, or Arrow serialization?
- Add any other context about the problem here.
