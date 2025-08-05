# NDR Platform - Documentation Index

**👨‍💻 Author:** [Shizukaax](https://github.com/Shizukaax) | **🔗 Repository:** [ndr-platform](https://github.com/Shizukaax/ndr-platform)

## 📚 Complete Documentation Suite

Welcome to the **Network Detection and Response (NDR) Platform** documentation. This comprehensive guide covers all aspects of the platform from basic usage to advanced deployment and development.

---

## 🗂️ Documentation Structure

### **📖 Core Documentation**

#### **1. [Project Organization](PROJECT_ORGANIZATION.md)**
*Complete project structure and architecture overview*
- **Purpose**: Understand the entire codebase organization
- **Audience**: Developers, architects, and new team members
- **Content**: Directory structure, file purposes, architecture principles
- **Key Sections**: App layer, core services, data management, deployment files

#### **2. [User Guide](USER_GUIDE.md)**
*Comprehensive user manual for all platform features*
- **Purpose**: End-to-end user documentation for all roles
- **Audience**: Security analysts, administrators, and end users
- **Content**: Feature tutorials, workflows, troubleshooting
- **Key Sections**: Getting started, anomaly detection, MITRE mapping, reporting

#### **3. [API Documentation](API_DOCUMENTATION.md)**
*Complete API reference and integration guide*
- **Purpose**: Technical reference for developers and integrators
- **Audience**: Developers, API consumers, system integrators
- **Content**: API endpoints, examples, authentication, SDKs
- **Key Sections**: Core APIs, data models, authentication, examples

---

### **⚙️ Configuration & Deployment**

#### **4. [Configuration Guide](CONFIGURATION_GUIDE.md)**
*Complete configuration management documentation*
- **Purpose**: Comprehensive configuration reference
- **Audience**: System administrators, DevOps engineers
- **Content**: Config files, environment variables, validation
- **Key Sections**: Main config, environment setup, best practices

#### **5. [Deployment Guide](DEPLOYMENT_GUIDE.md)**
*Production deployment and infrastructure documentation*
- **Purpose**: Production-ready deployment instructions
- **Audience**: DevOps engineers, system administrators
- **Content**: Docker, Kubernetes, monitoring, maintenance
- **Key Sections**: Container deployment, production setup, monitoring

---

### **🔧 Technical References**

#### **6. [Fixes Summary](FIXES_SUMMARY.md)**
*Complete summary of all critical fixes and improvements*
- **Purpose**: Track platform improvements and bug fixes
- **Audience**: Developers, administrators
- **Content**: Critical fixes, improvements, validation results

#### **7. [Logging Guide](LOGGING_GUIDE.md)**
*Comprehensive logging configuration and best practices*
- **Purpose**: Logging system setup and troubleshooting
- **Audience**: Developers, system administrators
- **Content**: Configuration, troubleshooting, best practices

---

## 🎯 Quick Reference by Role

### **🔍 Security Analysts**
*Primary users performing threat detection and analysis*

**Essential Reading:**
1. **[User Guide](USER_GUIDE.md)** - Complete platform usage
2. **[API Documentation](API_DOCUMENTATION.md)** - Integration capabilities

**Key Workflows:**
- Anomaly detection and analysis
- MITRE ATT&CK technique mapping
- Report generation and export
- Real-time monitoring setup

---

### **👨‍💻 Developers**
*Platform developers and contributors*

**Essential Reading:**
1. **[Project Organization](PROJECT_ORGANIZATION.md)** - Codebase structure
2. **[API Documentation](API_DOCUMENTATION.md)** - API development
3. **[Fixes Summary](FIXES_SUMMARY.md)** - Recent improvements

**Key Resources:**
- Architecture patterns and principles
- API design and implementation
- Testing strategies and frameworks
- Code organization standards

---

### **⚙️ System Administrators**
*Platform deployment and maintenance*

**Essential Reading:**
1. **[Configuration Guide](CONFIGURATION_GUIDE.md)** - System configuration
2. **[Deployment Guide](DEPLOYMENT_GUIDE.md)** - Production deployment
3. **[Logging Guide](LOGGING_GUIDE.md)** - Logging setup

**Key Responsibilities:**
- Production deployment and scaling
- Configuration management
- Performance monitoring
- Security hardening

---

### **🏢 DevOps Engineers**
*Infrastructure automation and CI/CD*

**Essential Reading:**
1. **[Deployment Guide](DEPLOYMENT_GUIDE.md)** - Infrastructure setup
2. **[Configuration Guide](CONFIGURATION_GUIDE.md)** - Environment management
3. **[Project Organization](PROJECT_ORGANIZATION.md)** - Build understanding

---

## 📊 Feature Coverage Matrix

| Feature | User Guide | API Docs | Config Guide | Deployment |
|---------|------------|----------|--------------|------------|
| Anomaly Detection | ✅ Complete | ✅ Complete | ✅ Covered | ✅ Covered |
| MITRE Mapping | ✅ Complete | ✅ Complete | ✅ Covered | ✅ Covered |
| Real-time Monitoring | ✅ Complete | ✅ Complete | ✅ Covered | ✅ Covered |
| Model Management | ✅ Complete | ✅ Complete | ✅ Covered | ✅ Covered |
| Reporting | ✅ Complete | ✅ Complete | ✅ Covered | ✅ Covered |
| Data Management | ✅ Complete | ✅ Complete | ✅ Covered | ✅ Covered |
| Security Features | ✅ Complete | ✅ Complete | ✅ Covered | ✅ Covered |
| Integration APIs | ✅ Covered | ✅ Complete | ✅ Covered | ✅ Covered |

---

## � Documentation Maintenance

### **Update Schedule**
- **Monthly**: Review user guide for new features
- **Quarterly**: Update API documentation for changes
- **Release-based**: Update deployment and configuration guides
- **Continuous**: Project organization updates with code changes

### **Version Control**
All documentation follows semantic versioning aligned with platform releases:
- **Major Changes**: Complete section rewrites or new documents
- **Minor Changes**: New features or significant updates
- **Patch Changes**: Bug fixes, clarifications, and minor improvements

### **Quality Assurance**
- **Technical Review**: All changes reviewed by technical team
- **User Testing**: User guide validated with actual users
- **Integration Testing**: API documentation validated with examples
- **Deployment Testing**: Deployment guides tested in clean environments

---

## 📞 Support & Feedback

### **Documentation Issues**
- **Report Errors**: Create issue tickets for documentation bugs
- **Request Clarifications**: Submit requests for unclear sections
- **Suggest Improvements**: Propose documentation enhancements
- **Contribute Updates**: Submit pull requests for improvements

### **Platform Support**
- **Technical Issues**: Use platform error reporting features
- **Feature Requests**: Submit enhancement requests
- **User Questions**: Access community forums or support channels
- **Training Needs**: Contact administrators for training resources

---

## 🎉 Documentation Summary

The NDR Platform documentation suite provides comprehensive coverage for all user types and use cases:

- **📖 7 Complete Guides**: Covering every aspect from usage to deployment
- **🎯 Role-based Paths**: Tailored documentation journeys for each user type
- **📊 100% Feature Coverage**: All platform features documented across guides
- **🔄 Maintained Standards**: Regular updates and quality assurance processes

**Total Documentation**: 7 comprehensive guides with 75+ sections covering all platform capabilities, deployment scenarios, and user workflows.

This complete documentation suite ensures the NDR Platform is accessible, maintainable, and scalable for all users and deployment scenarios.

### 📖 Project Documentation
- **../README.md** - Main project documentation and setup guide
- **../DEPLOYMENT.md** - Deployment instructions for development and production

## 📋 Quick Reference

### Recent Critical Fixes
1. ✅ Format string errors in explain_feedback.py
2. ✅ Threshold KeyError in reporting page  
3. ✅ MITRE mapping index errors
4. ✅ Column compatibility with Arkime data
5. ✅ Logging duplication and spam prevention

### Configuration Files
- All project configuration files (Dockerfile, docker-compose.yml, etc.) are in the root directory
- Environment templates and examples are properly configured
- Complete dependency management in requirements.txt

### Testing
- Test files are located in `../tests/` directory
- Run tests to validate fixes and configuration

For detailed information, see the specific documentation files in this directory.
