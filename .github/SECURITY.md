# Security Policy

**👨‍💻 Author:** [Shizukaax](https://github.com/Shizukaax) | **📧 Contact:** justinchua@tunglok.com

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          | End of Life |
| ------- | ------------------ | ----------- |
| 2.x.x   | :white_check_mark: | Active |
| 1.x.x   | :white_check_mark: | Dec 2025 |
| < 1.0   | :x:                | Unsupported |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please follow these steps:

### 🔒 **Private Disclosure**

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please report security issues privately:

1. **Email**: Send details to **justinchua@tunglok.com**
2. **Subject**: "SECURITY: NDR Platform Vulnerability Report"
3. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact assessment
   - Affected versions
   - Suggested fix (if any)
   - Your contact information for follow-up

### ⏱️ **Response Timeline**

- **24 hours**: Initial acknowledgment
- **72 hours**: Initial assessment and severity classification
- **7 days**: Detailed response with timeline for fix
- **30 days**: Security patch released (for high/critical issues)

### 🛡️ **Security Best Practices**

When deploying NDR Platform:

1. **Environment Variables**: Never commit `.env` files
2. **Data Security**: Ensure network data is properly encrypted
3. **Access Control**: Use proper authentication mechanisms
4. **Updates**: Keep dependencies updated regularly
5. **Monitoring**: Enable security logging and monitoring

### 🔍 **Security Scanning**

We provide built-in security scanning:

```bash
# Run comprehensive security scan
python scripts/security_scanner.py all

# Check for vulnerabilities
python scripts/security_scanner.py dependencies

# Audit code security
python scripts/security_scanner.py code
```

### 🏆 **Responsible Disclosure**

We appreciate responsible disclosure and may recognize researchers who help improve our security:

- Public acknowledgment (with permission)
- Security contributors section in documentation
- LinkedIn recommendation for significant findings
- Potential collaboration opportunities

### 📋 **Security Checklist**

Before deployment, ensure:

- [ ] All dependencies are up to date
- [ ] Security scanner shows no HIGH severity issues
- [ ] Environment variables are properly configured
- [ ] Data access is restricted and logged
- [ ] Network communications are encrypted
- [ ] Authentication is properly implemented
- [ ] Docker containers run as non-root user
- [ ] File permissions are properly set
- [ ] Logs don't contain sensitive information

### 🔐 **Security Features**

The NDR Platform includes:

- **Data Encryption**: TLS 1.3 for data in transit
- **Access Control**: Role-based permissions
- **Audit Logging**: Complete activity tracking  
- **Network Isolation**: Containerized deployment
- **Secure Defaults**: Hardened configuration templates
- **Input Validation**: Comprehensive data sanitization
- **Session Management**: Secure session handling

## Contact

For security-related questions or concerns:
- **Security Contact**: justinchua@tunglok.com
- **GitHub Issues**: For non-security topics only
- **Repository**: [NDR Platform](https://github.com/Shizukaax/ndr-platform)

---

**Last Updated**: August 5, 2025  
**Maintained by**: [Shizukaax](https://github.com/Shizukaax)
