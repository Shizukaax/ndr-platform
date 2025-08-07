# 🆘 Getting Support - NDR Platform v2.1.0

**Author:** [Shizukaax](https://github.com/Shizukaax) | **Contact:** justinchua@tunglok.com

## 📞 How to Get Help

### 🐛 **Bug Reports (v2.1.0 Production)**
If you've found a bug in the production-ready platform, please:
1. **Check existing issues** first: [GitHub Issues](https://github.com/Shizukaax/ndr-platform/issues)
2. **Create a new issue** using our [bug report template](https://github.com/Shizukaax/ndr-platform/issues/new)
3. **Include details**: Error messages, screenshots, steps to reproduce
4. **Run diagnostics**: `python scripts/health_check.py` and include output
5. **Check for known fixes**: Review recent critical fixes applied in v2.1.0

### 💡 **Feature Requests**
For new features or enhancements:
1. **Search existing requests**: [GitHub Issues](https://github.com/Shizukaax/ndr-platform/issues?q=is%3Aissue+label%3Aenhancement)
2. **Submit a feature request** with detailed use case
3. **Explain the problem** your feature would solve

### ❓ **Questions & Discussions**
For general questions, usage help, or discussions:
1. **GitHub Discussions**: [Start a discussion](https://github.com/Shizukaax/ndr-platform/discussions)
2. **Documentation**: Check our [comprehensive guides](../guides/)
3. **Examples**: Review [usage examples](../examples/)

### 🚨 **Security Issues**
For security vulnerabilities:
1. **DO NOT** create a public issue
2. **Email directly**: justinchua@tunglok.com
3. **Follow our** [Security Policy](SECURITY.md)

## 📚 **Self-Help Resources**

### **Documentation**
- **[User Guide](../guides/USER_GUIDE.md)** - Complete user manual
- **[Configuration Guide](../guides/CONFIGURATION_GUIDE.md)** - Setup instructions
- **[Deployment Guide](../guides/DEPLOYMENT_GUIDE.md)** - Production deployment
- **[API Documentation](../guides/API_DOCUMENTATION.md)** - API reference
- **[Troubleshooting](../guides/USER_GUIDE.md#troubleshooting)** - Common issues

### **Quick Fixes & Diagnostics (v2.1.0)**
| Problem | Solution | Diagnostic Tool |
|---------|----------|----------------|
| Installation issues | Check [Configuration Guide](../guides/CONFIGURATION_GUIDE.md) | `python scripts/health_check.py` |
| Docker problems | Verify Docker and check [Deployment Guide](../guides/DEPLOYMENT_GUIDE.md) | `python scripts/verify_structure.py` |
| Data loading errors | Review formats in [User Guide](../guides/USER_GUIDE.md) | `python test_encoding_fixes.py` |
| Model training issues | Check [Model Management](../guides/USER_GUIDE.md#model-training) | `python test_results_saving.py` |
| Permission errors | Ensure proper file permissions and paths | `python test_feedback_dirs.py` |
| NaN value errors | Check numeric data validation | `python test_final_verification.py` |
| Unicode/encoding issues | Verify text processing | `python test_encoding_fixes.py` |
| Arrow serialization problems | Check data format compatibility | `python test_results_saving.py` |

### **Logs & Debugging (Enhanced)**
1. **Check application logs**: `logs/` directory (now with UTF-8 encoding fixes)
2. **Enable debug mode**: Set `LOG_LEVEL=DEBUG` in `.env`  
3. **Health check**: Run `python scripts/health_check.py`
4. **System validation**: Use `python scripts/verify_structure.py`
5. **Critical fixes validation**: Run comprehensive test suite:
   ```bash
   python test_encoding_fixes.py
   python test_results_saving.py  
   python test_feedback_dirs.py
   python test_final_verification.py
   ```

## ⏱️ **Response Times**

| Issue Type | Expected Response |
|------------|-------------------|
| Security vulnerabilities | Within 24 hours |
| Critical bugs | 1-3 business days |
| Feature requests | 1-2 weeks |
| General questions | 3-5 business days |
| Documentation issues | 1-2 weeks |

## 🤝 **Community Guidelines**

- **Be respectful** and constructive
- **Provide context** and details
- **Search before posting** to avoid duplicates
- **Use clear titles** that describe the issue
- **Follow up** on your issues and discussions

## 📧 **Direct Contact**

For urgent issues, complex problems, or business inquiries:

**Email:** justinchua@tunglok.com  
**GitHub:** [@Shizukaax](https://github.com/Shizukaax)

Please include:
- Your operating system and version
- Python version  
- **NDR Platform v2.1.0** (production version)
- Error messages (if any)
- Steps you've already tried
- Output from diagnostic tools (`python scripts/health_check.py`)
- Results from critical fix validation tests

---

**Thank you for using the NDR Platform v2.1.0!** 🛡️

We're committed to providing excellent support for our production-ready platform and making network security accessible to everyone.
