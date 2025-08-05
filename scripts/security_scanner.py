#!/usr/bin/env python3
"""
NDR Platform Security Scanner
Security audit and vulnerability scanning for the NDR platform.
"""

import os
import re
import json
import subprocess
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
from typing import Dict, List, Any

class SecurityScanner:
    """Security scanning and audit tools."""
    
    def __init__(self, project_dir='.'):
        self.project_dir = Path(project_dir)
        self.scan_results = {
            'timestamp': datetime.now().isoformat(),
            'vulnerabilities': [],
            'security_issues': [],
            'recommendations': [],
            'summary': {}
        }
    
    def scan_dependencies(self):
        """Scan dependencies for known vulnerabilities."""
        print("🔍 Scanning Dependencies for Vulnerabilities")
        print("=" * 50)
        
        requirements_file = self.project_dir / 'requirements.txt'
        if not requirements_file.exists():
            print("   ❌ requirements.txt not found")
            return
        
        vulnerabilities = []
        
        # Read requirements
        with open(requirements_file, 'r') as f:
            requirements = f.read().splitlines()
        
        # Check each package (simplified check)
        vulnerable_packages = self._get_vulnerable_packages()
        
        for requirement in requirements:
            if requirement.strip() and not requirement.strip().startswith('#'):
                package_name = self._extract_package_name(requirement)
                if package_name in vulnerable_packages:
                    vuln_info = vulnerable_packages[package_name]
                    vulnerability = {
                        'type': 'dependency_vulnerability',
                        'severity': vuln_info['severity'],
                        'package': package_name,
                        'description': vuln_info['description'],
                        'recommendation': vuln_info['recommendation']
                    }
                    vulnerabilities.append(vulnerability)
                    print(f"   ❌ {package_name}: {vuln_info['description']}")
                else:
                    print(f"   ✅ {package_name}")
        
        self.scan_results['vulnerabilities'].extend(vulnerabilities)
        
        if not vulnerabilities:
            print("   ✅ No known vulnerabilities found in dependencies")
    
    def scan_code_security(self):
        """Scan code for security issues."""
        print("\n🔐 Scanning Code for Security Issues")
        print("=" * 50)
        
        security_patterns = {
            'hardcoded_secrets': {
                'patterns': [
                    r'password\s*=\s*["\'][^"\']+["\']',
                    r'api_key\s*=\s*["\'][^"\']+["\']',
                    r'secret\s*=\s*["\'][^"\']+["\']',
                    r'token\s*=\s*["\'][^"\']+["\']',
                ],
                'severity': 'HIGH'
            },
            'sql_injection': {
                'patterns': [
                    r'execute\s*\(\s*["\'].*%.*["\']',
                    r'cursor\.execute\s*\(\s*f["\']',
                    r'\.format\s*\(.*query',
                ],
                'severity': 'HIGH'
            },
            'command_injection': {
                'patterns': [
                    r'os\.system\s*\(',
                    r'subprocess\.(call|run|Popen)\s*\([^)]*shell\s*=\s*True',
                    r'exec\s*\(',
                    r'eval\s*\(',
                ],
                'severity': 'MEDIUM'
            },
            'insecure_random': {
                'patterns': [
                    r'random\.random\(',
                    r'random\.randint\(',
                    r'random\.choice\(',
                ],
                'severity': 'LOW'
            },
            'debug_code': {
                'patterns': [
                    r'print\s*\(.*password',
                    r'print\s*\(.*secret',
                    r'print\s*\(.*token',
                    r'DEBUG\s*=\s*True',
                ],
                'severity': 'MEDIUM'
            }
        }
        
        python_files = list(self.project_dir.rglob('*.py'))
        security_issues = []
        
        for py_file in python_files:
            # Skip virtual environment and cache directories
            if any(part in str(py_file) for part in ['venv', '__pycache__', '.git']):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                file_issues = self._scan_file_content(py_file, content, security_patterns)
                security_issues.extend(file_issues)
                
            except Exception as e:
                print(f"   ⚠️  Error scanning {py_file}: {e}")
        
        self.scan_results['security_issues'].extend(security_issues)
        
        # Display results
        if security_issues:
            print(f"   ❌ Found {len(security_issues)} security issues:")
            for issue in security_issues[:10]:  # Show first 10
                print(f"      {issue['severity']} - {issue['file']}:{issue['line']}")
                print(f"        {issue['type']}: {issue['description']}")
        else:
            print("   ✅ No security issues found in code")
    
    def scan_configuration(self):
        """Scan configuration files for security issues."""
        print("\n⚙️  Scanning Configuration Security")
        print("=" * 50)
        
        config_issues = []
        
        # Check .env files
        env_files = list(self.project_dir.glob('.env*'))
        for env_file in env_files:
            if env_file.is_file():
                issues = self._scan_env_file(env_file)
                config_issues.extend(issues)
        
        # Check YAML configuration files
        yaml_files = list(self.project_dir.rglob('*.yaml')) + list(self.project_dir.rglob('*.yml'))
        for yaml_file in yaml_files:
            issues = self._scan_yaml_file(yaml_file)
            config_issues.extend(issues)
        
        # Check Docker files
        docker_files = list(self.project_dir.glob('Dockerfile*')) + list(self.project_dir.glob('docker-compose*'))
        for docker_file in docker_files:
            if docker_file.is_file():
                issues = self._scan_docker_file(docker_file)
                config_issues.extend(issues)
        
        self.scan_results['security_issues'].extend(config_issues)
        
        if config_issues:
            print(f"   ❌ Found {len(config_issues)} configuration issues:")
            for issue in config_issues:
                print(f"      {issue['severity']} - {issue['file']}")
                print(f"        {issue['description']}")
        else:
            print("   ✅ No configuration security issues found")
    
    def scan_permissions(self):
        """Scan file and directory permissions."""
        print("\n🔒 Scanning File Permissions")
        print("=" * 50)
        
        if os.name == 'nt':  # Windows
            print("   ℹ️  Permission scanning not available on Windows")
            return
        
        permission_issues = []
        
        # Check sensitive files
        sensitive_files = [
            '.env', '.env.local', '.env.production',
            'config/config.yaml', 'config/secrets.yaml',
            'private.key', 'id_rsa', 'certificate.pem'
        ]
        
        for file_pattern in sensitive_files:
            files = list(self.project_dir.rglob(file_pattern))
            for file_path in files:
                if file_path.is_file():
                    perms = oct(file_path.stat().st_mode)[-3:]
                    if perms != '600' and perms != '644':  # Too permissive
                        issue = {
                            'type': 'file_permissions',
                            'severity': 'MEDIUM',
                            'file': str(file_path),
                            'current_perms': perms,
                            'description': f"Sensitive file has permissive permissions: {perms}",
                            'recommendation': 'Change permissions to 600 (owner read/write only)'
                        }
                        permission_issues.append(issue)
                        print(f"   ❌ {file_path}: {perms} (too permissive)")
                    else:
                        print(f"   ✅ {file_path}: {perms}")
        
        # Check script files
        script_files = list(self.project_dir.rglob('*.sh')) + list(self.project_dir.rglob('scripts/*.py'))
        for script_file in script_files:
            if script_file.is_file():
                perms = oct(script_file.stat().st_mode)[-3:]
                if not perms.startswith('7'):  # Not executable by owner
                    issue = {
                        'type': 'script_permissions',
                        'severity': 'LOW',
                        'file': str(script_file),
                        'current_perms': perms,
                        'description': f"Script file not executable: {perms}",
                        'recommendation': 'Add execute permission for owner (chmod +x)'
                    }
                    permission_issues.append(issue)
                    print(f"   ⚠️  {script_file}: {perms} (not executable)")
        
        self.scan_results['security_issues'].extend(permission_issues)
        
        if not permission_issues:
            print("   ✅ No permission issues found")
    
    def scan_network_security(self):
        """Scan for network security configurations."""
        print("\n🌐 Scanning Network Security")
        print("=" * 50)
        
        network_issues = []
        
        # Check Streamlit configuration
        config_files = list(self.project_dir.rglob('*.toml')) + [Path('.streamlit/config.toml')]
        for config_file in config_files:
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        content = f.read()
                    
                    # Check for insecure configurations
                    if 'headless = false' in content:
                        issue = {
                            'type': 'streamlit_config',
                            'severity': 'LOW',
                            'file': str(config_file),
                            'description': 'Streamlit running in non-headless mode',
                            'recommendation': 'Set headless = true for production'
                        }
                        network_issues.append(issue)
                    
                    if 'enableCORS = false' in content:
                        issue = {
                            'type': 'streamlit_config',
                            'severity': 'MEDIUM',
                            'file': str(config_file),
                            'description': 'CORS disabled in Streamlit',
                            'recommendation': 'Enable CORS for production security'
                        }
                        network_issues.append(issue)
                
                except Exception as e:
                    print(f"   ⚠️  Error reading {config_file}: {e}")
        
        # Check for hardcoded URLs and IPs
        python_files = list(self.project_dir.rglob('*.py'))
        for py_file in python_files:
            if any(part in str(py_file) for part in ['venv', '__pycache__']):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Check for localhost/127.0.0.1 in production code
                if re.search(r'localhost|127\.0\.0\.1', content):
                    issue = {
                        'type': 'hardcoded_localhost',
                        'severity': 'LOW',
                        'file': str(py_file),
                        'description': 'Hardcoded localhost/127.0.0.1 found',
                        'recommendation': 'Use configuration variables for hostnames'
                    }
                    network_issues.append(issue)
            
            except Exception:
                pass
        
        self.scan_results['security_issues'].extend(network_issues)
        
        if network_issues:
            print(f"   ❌ Found {len(network_issues)} network security issues")
            for issue in network_issues:
                print(f"      {issue['severity']} - {issue['description']}")
        else:
            print("   ✅ No network security issues found")
    
    def generate_security_report(self):
        """Generate comprehensive security report."""
        print("\n📊 Generating Security Report")
        print("=" * 50)
        
        # Calculate summary statistics
        total_issues = len(self.scan_results['vulnerabilities']) + len(self.scan_results['security_issues'])
        
        severity_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        all_issues = self.scan_results['vulnerabilities'] + self.scan_results['security_issues']
        
        for issue in all_issues:
            severity = issue.get('severity', 'UNKNOWN')
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        self.scan_results['summary'] = {
            'total_issues': total_issues,
            'severity_breakdown': severity_counts,
            'scan_date': datetime.now().isoformat(),
            'files_scanned': len(list(self.project_dir.rglob('*.py')))
        }
        
        # Generate recommendations
        recommendations = []
        
        if severity_counts['HIGH'] > 0:
            recommendations.append("🚨 Address HIGH severity issues immediately")
        
        if any('hardcoded_secrets' in str(issue) for issue in all_issues):
            recommendations.append("🔐 Use environment variables for secrets")
        
        if any('sql_injection' in str(issue) for issue in all_issues):
            recommendations.append("💉 Use parameterized queries to prevent SQL injection")
        
        if any('command_injection' in str(issue) for issue in all_issues):
            recommendations.append("⚡ Avoid shell=True in subprocess calls")
        
        recommendations.extend([
            "🔄 Regularly update dependencies",
            "🔍 Implement automated security scanning in CI/CD",
            "📝 Conduct regular security code reviews",
            "🛡️  Implement proper input validation",
            "🔒 Use HTTPS in production"
        ])
        
        self.scan_results['recommendations'] = recommendations
        
        # Save report
        report_file = Path(f"security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_file, 'w') as f:
            json.dump(self.scan_results, f, indent=2)
        
        # Display summary
        print(f"   📋 Total Issues Found: {total_issues}")
        print(f"   🔴 High Severity: {severity_counts['HIGH']}")
        print(f"   🟡 Medium Severity: {severity_counts['MEDIUM']}")
        print(f"   🟢 Low Severity: {severity_counts['LOW']}")
        print(f"   📁 Files Scanned: {self.scan_results['summary']['files_scanned']}")
        
        print(f"\n💡 Recommendations:")
        for rec in recommendations[:5]:  # Show top 5
            print(f"   {rec}")
        
        print(f"\n💾 Detailed report saved: {report_file}")
        
        return report_file
    
    def _get_vulnerable_packages(self):
        """Get list of known vulnerable packages (simplified)."""
        # In a real implementation, this would query vulnerability databases
        return {
            'tensorflow': {
                'severity': 'HIGH',
                'description': 'Known vulnerability in versions < 2.8.0',
                'recommendation': 'Update to tensorflow >= 2.8.0'
            },
            'pillow': {
                'severity': 'MEDIUM', 
                'description': 'Image processing vulnerability',
                'recommendation': 'Update to latest version'
            }
        }
    
    def _extract_package_name(self, requirement):
        """Extract package name from requirement string."""
        # Remove version specifiers
        package = re.split(r'[<>=!]', requirement)[0].strip()
        return package
    
    def _scan_file_content(self, file_path, content, patterns):
        """Scan file content for security patterns."""
        issues = []
        lines = content.split('\n')
        
        for pattern_name, pattern_info in patterns.items():
            for pattern in pattern_info['patterns']:
                for line_num, line in enumerate(lines, 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        issue = {
                            'type': pattern_name,
                            'severity': pattern_info['severity'],
                            'file': str(file_path),
                            'line': line_num,
                            'description': f"Potential {pattern_name.replace('_', ' ')} detected",
                            'code_snippet': line.strip()
                        }
                        issues.append(issue)
        
        return issues
    
    def _scan_env_file(self, env_file):
        """Scan environment file for security issues."""
        issues = []
        
        try:
            with open(env_file, 'r') as f:
                content = f.read()
            
            # Check for weak passwords
            if re.search(r'PASSWORD=123|PASSWORD=admin|PASSWORD=password', content, re.IGNORECASE):
                issue = {
                    'type': 'weak_password',
                    'severity': 'HIGH',
                    'file': str(env_file),
                    'description': 'Weak default password found',
                    'recommendation': 'Use strong, randomly generated passwords'
                }
                issues.append(issue)
            
            # Check if file is committed (simplified check)
            gitignore_file = self.project_dir / '.gitignore'
            if gitignore_file.exists():
                with open(gitignore_file, 'r') as f:
                    gitignore_content = f.read()
                
                if env_file.name not in gitignore_content:
                    issue = {
                        'type': 'env_not_ignored',
                        'severity': 'HIGH',
                        'file': str(env_file),
                        'description': 'Environment file not in .gitignore',
                        'recommendation': 'Add environment files to .gitignore'
                    }
                    issues.append(issue)
        
        except Exception as e:
            pass
        
        return issues
    
    def _scan_yaml_file(self, yaml_file):
        """Scan YAML configuration file."""
        issues = []
        
        try:
            with open(yaml_file, 'r') as f:
                content = f.read()
            
            # Check for hardcoded secrets
            if re.search(r'(password|secret|key):\s*["\']?[a-zA-Z0-9]+["\']?', content, re.IGNORECASE):
                issue = {
                    'type': 'yaml_hardcoded_secret',
                    'severity': 'MEDIUM',
                    'file': str(yaml_file),
                    'description': 'Potential hardcoded secret in YAML',
                    'recommendation': 'Use environment variables for secrets'
                }
                issues.append(issue)
        
        except Exception:
            pass
        
        return issues
    
    def _scan_docker_file(self, docker_file):
        """Scan Docker file for security issues."""
        issues = []
        
        try:
            with open(docker_file, 'r') as f:
                content = f.read()
            
            # Check for running as root
            if 'USER root' in content or not re.search(r'USER \w+', content):
                issue = {
                    'type': 'docker_root_user',
                    'severity': 'MEDIUM',
                    'file': str(docker_file),
                    'description': 'Docker container may run as root',
                    'recommendation': 'Create and use non-root user in Docker'
                }
                issues.append(issue)
            
            # Check for exposed ports
            exposed_ports = re.findall(r'EXPOSE\s+(\d+)', content)
            for port in exposed_ports:
                if port in ['22', '23', '3389']:  # Common admin ports
                    issue = {
                        'type': 'docker_admin_port',
                        'severity': 'HIGH',
                        'file': str(docker_file),
                        'description': f'Administrative port {port} exposed',
                        'recommendation': 'Avoid exposing administrative ports'
                    }
                    issues.append(issue)
        
        except Exception:
            pass
        
        return issues

def main():
    """Main function for security scanner."""
    parser = argparse.ArgumentParser(description='NDR Platform Security Scanner')
    parser.add_argument('scans', nargs='*', 
                       choices=['dependencies', 'code', 'config', 'permissions', 'network', 'all'],
                       default=['all'],
                       help='Security scans to run (default: all)')
    parser.add_argument('--project-dir', default='.',
                       help='Project directory to scan (default: current directory)')
    
    args = parser.parse_args()
    
    print("🛡️  NDR Platform Security Scanner")
    print("=" * 50)
    
    # Change to project directory
    os.chdir(Path(__file__).parent.parent)
    
    scanner = SecurityScanner(args.project_dir)
    
    scans_to_run = args.scans
    if 'all' in scans_to_run:
        scans_to_run = ['dependencies', 'code', 'config', 'permissions', 'network']
    
    try:
        if 'dependencies' in scans_to_run:
            scanner.scan_dependencies()
        
        if 'code' in scans_to_run:
            scanner.scan_code_security()
        
        if 'config' in scans_to_run:
            scanner.scan_configuration()
        
        if 'permissions' in scans_to_run:
            scanner.scan_permissions()
        
        if 'network' in scans_to_run:
            scanner.scan_network_security()
        
        # Always generate report
        report_file = scanner.generate_security_report()
        
        # Exit with error code if high severity issues found
        high_severity_count = scanner.scan_results['summary']['severity_breakdown']['HIGH']
        if high_severity_count > 0:
            print(f"\n🚨 Security scan completed with {high_severity_count} HIGH severity issues!")
            exit(1)
        else:
            print(f"\n✅ Security scan completed successfully!")
            
    except Exception as e:
        print(f"❌ Security scan error: {e}")
        exit(1)

if __name__ == "__main__":
    main()
