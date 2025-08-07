"""
Security Metrics Component
Provides advanced security metrics including MTTD, MTTR, and performance tracking
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Any, Optional

class SecurityMetrics:
    """Component for advanced security metrics and performance tracking"""
    
    def __init__(self):
        self.metric_categories = {
            'detection': ['MTTD', 'Detection Rate', 'False Positive Rate'],
            'response': ['MTTR', 'Escalation Time', 'Resolution Rate'],
            'coverage': ['Network Coverage', 'Endpoint Coverage', 'Asset Visibility']
        }
    
    def render_performance_metrics(self):
        """Render the performance metrics dashboard"""
        st.subheader("📊 Security Performance Metrics")
        
        # Key performance indicators
        self._render_kpi_section()
        
        # Detailed metrics
        self._render_detailed_metrics()
        
        # Trend analysis
        self._render_trend_analysis()
        
        # Compliance metrics
        self._render_compliance_metrics()
    
    def _render_kpi_section(self):
        """Render key performance indicators"""
        st.markdown("### 🎯 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            mttd = self._calculate_mttd()
            st.metric(
                "Mean Time to Detection (MTTD)",
                f"{mttd:.1f} min",
                delta=f"-{np.random.uniform(0.5, 2.0):.1f} min",
                delta_color="inverse"
            )
        
        with col2:
            mttr = self._calculate_mttr()
            st.metric(
                "Mean Time to Response (MTTR)",
                f"{mttr:.1f} min",
                delta=f"-{np.random.uniform(1.0, 5.0):.1f} min",
                delta_color="inverse"
            )
        
        with col3:
            detection_rate = np.random.uniform(92, 98)
            st.metric(
                "Detection Rate",
                f"{detection_rate:.1f}%",
                delta=f"+{np.random.uniform(0.1, 1.5):.1f}%"
            )
        
        with col4:
            fp_rate = np.random.uniform(2, 8)
            st.metric(
                "False Positive Rate",
                f"{fp_rate:.1f}%",
                delta=f"-{np.random.uniform(0.2, 1.0):.1f}%",
                delta_color="inverse"
            )
    
    def _render_detailed_metrics(self):
        """Render detailed security metrics"""
        st.markdown("### 📈 Detailed Security Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Alert volume and resolution trends
            alert_data = self._generate_alert_trends()
            
            fig_alerts = go.Figure()
            
            fig_alerts.add_trace(go.Scatter(
                x=alert_data['timestamp'],
                y=alert_data['alerts_generated'],
                name='Alerts Generated',
                line=dict(color='red', width=2)
            ))
            
            fig_alerts.add_trace(go.Scatter(
                x=alert_data['timestamp'],
                y=alert_data['alerts_resolved'],
                name='Alerts Resolved',
                line=dict(color='green', width=2)
            ))
            
            fig_alerts.update_layout(
                title="Alert Generation vs Resolution",
                xaxis_title="Time",
                yaxis_title="Number of Alerts",
                height=400
            )
            
            st.plotly_chart(fig_alerts, use_container_width=True)
        
        with col2:
            # Security coverage metrics
            coverage_data = self._generate_coverage_data()
            
            fig_coverage = go.Figure()
            
            fig_coverage.add_trace(go.Scatterpolar(
                r=coverage_data['values'],
                theta=coverage_data['categories'],
                fill='toself',
                name='Current Coverage'
            ))
            
            fig_coverage.add_trace(go.Scatterpolar(
                r=[100] * len(coverage_data['categories']),
                theta=coverage_data['categories'],
                fill='toself',
                name='Target Coverage',
                opacity=0.3
            ))
            
            fig_coverage.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                title="Security Coverage Assessment",
                height=400
            )
            
            st.plotly_chart(fig_coverage, use_container_width=True)
    
    def _render_trend_analysis(self):
        """Render trend analysis section"""
        st.markdown("### 📊 Performance Trends")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # MTTD trend
            mttd_data = self._generate_mttd_trend()
            
            fig_mttd = px.line(
                mttd_data,
                x='date',
                y='mttd',
                title="MTTD Trend (Last 30 Days)",
                labels={'mttd': 'MTTD (minutes)', 'date': 'Date'}
            )
            
            # Add target line
            fig_mttd.add_hline(
                y=15,
                line_dash="dash",
                line_color="red",
                annotation_text="Target: 15 min"
            )
            
            st.plotly_chart(fig_mttd, use_container_width=True)
        
        with col2:
            # Threat landscape
            threat_data = self._generate_threat_landscape()
            
            fig_threats = px.bar(
                threat_data,
                x='threat_type',
                y='count',
                color='severity',
                title="Threat Landscape (Last 7 Days)",
                color_discrete_map={
                    'Critical': 'red',
                    'High': 'orange',
                    'Medium': 'yellow',
                    'Low': 'green'
                }
            )
            
            st.plotly_chart(fig_threats, use_container_width=True)
        
        with col3:
            # Analyst efficiency
            efficiency_data = self._generate_efficiency_data()
            
            fig_efficiency = px.scatter(
                efficiency_data,
                x='analyst',
                y='resolution_time',
                size='cases_handled',
                color='efficiency_score',
                title="Analyst Performance",
                labels={
                    'resolution_time': 'Avg Resolution Time (min)',
                    'efficiency_score': 'Efficiency Score'
                }
            )
            
            st.plotly_chart(fig_efficiency, use_container_width=True)
    
    def _render_compliance_metrics(self):
        """Render compliance and governance metrics"""
        st.markdown("### 📋 Compliance & Governance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Regulatory Compliance**")
            compliance_frameworks = [
                {"framework": "SOC 2", "compliance": 94, "status": "Good"},
                {"framework": "ISO 27001", "compliance": 89, "status": "Good"},
                {"framework": "NIST CSF", "compliance": 87, "status": "Needs Work"},
                {"framework": "GDPR", "compliance": 96, "status": "Excellent"},
                {"framework": "HIPAA", "compliance": 91, "status": "Good"}
            ]
            
            for framework in compliance_frameworks:
                compliance = framework['compliance']
                if compliance >= 95:
                    st.success(f"✅ {framework['framework']}: {compliance}%")
                elif compliance >= 85:
                    st.warning(f"⚠️ {framework['framework']}: {compliance}%")
                else:
                    st.error(f"❌ {framework['framework']}: {compliance}%")
        
        with col2:
            st.markdown("**Audit Findings**")
            audit_data = [
                {"finding": "Missing encryption", "severity": "High", "days_open": 5},
                {"finding": "Weak passwords", "severity": "Medium", "days_open": 12},
                {"finding": "Outdated patches", "severity": "Critical", "days_open": 2},
                {"finding": "Access controls", "severity": "Low", "days_open": 25}
            ]
            
            for finding in audit_data:
                severity = finding['severity']
                days = finding['days_open']
                
                if severity == "Critical":
                    st.error(f"🔴 {finding['finding']} ({days} days)")
                elif severity == "High":
                    st.warning(f"🟡 {finding['finding']} ({days} days)")
                else:
                    st.info(f"🟢 {finding['finding']} ({days} days)")
        
        with col3:
            st.markdown("**Risk Posture**")
            risk_metrics = {
                "Overall Risk Score": 6.8,
                "Critical Assets Protected": 94,
                "Vulnerabilities Patched": 87,
                "Security Training Completion": 92
            }
            
            for metric, value in risk_metrics.items():
                if isinstance(value, float):
                    if value <= 7.0:
                        st.success(f"✅ {metric}: {value}/10")
                    else:
                        st.warning(f"⚠️ {metric}: {value}/10")
                else:
                    if value >= 90:
                        st.success(f"✅ {metric}: {value}%")
                    elif value >= 80:
                        st.warning(f"⚠️ {metric}: {value}%")
                    else:
                        st.error(f"❌ {metric}: {value}%")
    
    def _calculate_mttd(self) -> float:
        """Calculate Mean Time to Detection"""
        # Simulate MTTD calculation based on recent detection times
        detection_times = np.random.exponential(12, 100)  # Average 12 minutes
        return np.mean(detection_times)
    
    def _calculate_mttr(self) -> float:
        """Calculate Mean Time to Response"""
        # Simulate MTTR calculation based on recent response times
        response_times = np.random.exponential(25, 100)  # Average 25 minutes
        return np.mean(response_times)
    
    def _generate_alert_trends(self) -> pd.DataFrame:
        """Generate alert trend data for the last 30 days"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=30),
            end=datetime.now(),
            freq='D'
        )
        
        # Generate realistic alert patterns
        base_alerts = 50
        trend = np.linspace(0, 10, len(dates))  # Slight upward trend
        noise = np.random.normal(0, 5, len(dates))
        
        alerts_generated = np.maximum(0, base_alerts + trend + noise + 
                                    np.random.poisson(10, len(dates)))
        
        # Resolution typically lags behind generation
        alerts_resolved = np.maximum(0, alerts_generated - 
                                   np.random.poisson(3, len(dates)) - 2)
        
        return pd.DataFrame({
            'timestamp': dates,
            'alerts_generated': alerts_generated.astype(int),
            'alerts_resolved': alerts_resolved.astype(int)
        })
    
    def _generate_coverage_data(self) -> Dict[str, List]:
        """Generate security coverage assessment data"""
        categories = [
            'Network Monitoring',
            'Endpoint Protection',
            'Email Security',
            'Web Security',
            'Identity Management',
            'Data Protection',
            'Cloud Security',
            'Mobile Security'
        ]
        
        # Generate coverage percentages with some realistic variation
        values = [
            95, 88, 92, 85, 90, 93, 78, 82
        ]
        
        return {
            'categories': categories,
            'values': values
        }
    
    def _generate_mttd_trend(self) -> pd.DataFrame:
        """Generate MTTD trend data"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=30),
            end=datetime.now(),
            freq='D'
        )
        
        # Generate MTTD with improving trend
        base_mttd = 20
        improvement = np.linspace(5, 0, len(dates))  # Gradual improvement
        noise = np.random.normal(0, 2, len(dates))
        
        mttd_values = np.maximum(8, base_mttd - improvement + noise)
        
        return pd.DataFrame({
            'date': dates,
            'mttd': mttd_values
        })
    
    def _generate_threat_landscape(self) -> pd.DataFrame:
        """Generate threat landscape data"""
        threat_types = ['Malware', 'Phishing', 'DDoS', 'Insider Threat', 'APT', 'Ransomware']
        data = []
        
        for threat in threat_types:
            for severity in ['Critical', 'High', 'Medium', 'Low']:
                if severity == 'Critical':
                    count = np.random.poisson(2)
                elif severity == 'High':
                    count = np.random.poisson(5)
                elif severity == 'Medium':
                    count = np.random.poisson(8)
                else:
                    count = np.random.poisson(12)
                
                data.append({
                    'threat_type': threat,
                    'severity': severity,
                    'count': count
                })
        
        return pd.DataFrame(data)
    
    def _generate_efficiency_data(self) -> pd.DataFrame:
        """Generate analyst efficiency data"""
        analysts = ['Alice S.', 'Bob J.', 'Carol W.', 'David B.', 'Eve M.']
        data = []
        
        for analyst in analysts:
            resolution_time = np.random.uniform(15, 45)
            cases_handled = np.random.randint(20, 80)
            efficiency_score = np.random.uniform(70, 95)
            
            data.append({
                'analyst': analyst,
                'resolution_time': resolution_time,
                'cases_handled': cases_handled,
                'efficiency_score': efficiency_score
            })
        
        return pd.DataFrame(data)
