"""
Automatic analysis service for the Network Anomaly Detection Platform.
Handles automatic MITRE mapping and other post-detection analysis.
"""

import streamlit as st
import logging
from typing import Dict, Any, Optional
import pandas as pd

from core.mitre_mapper import MitreMapper
from core.risk_scorer import RiskScorer

logger = logging.getLogger("streamlit_app")

class AutoAnalysisService:
    """Service for automatically analyzing detected anomalies."""
    
    def __init__(self):
        """Initialize the auto analysis service."""
        self.mitre_mapper = MitreMapper()
        self.risk_scorer = RiskScorer()
    
    def run_automatic_analysis(self, anomalies: pd.DataFrame, confidence_threshold: float = 0.6) -> Dict[str, Any]:
        """
        Run automatic analysis on detected anomalies.
        
        Args:
            anomalies (pd.DataFrame): Detected anomalies
            confidence_threshold (float): Minimum confidence for MITRE mappings
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        results = {
            'mitre_mappings': None,
            'risk_scores': None,
            'analysis_summary': {},
            'success': False,
            'errors': []
        }
        
        try:
            logger.info(f"Starting automatic analysis for {len(anomalies)} anomalies")
            
            # 1. Automatic MITRE ATT&CK Mapping
            mitre_results = self._run_mitre_mapping(anomalies, confidence_threshold)
            results['mitre_mappings'] = mitre_results
            
            # 2. Automatic Risk Scoring
            risk_results = self._run_risk_scoring(anomalies)
            results['risk_scores'] = risk_results
            
            # 3. Generate analysis summary
            summary = self._generate_analysis_summary(anomalies, mitre_results, risk_results)
            results['analysis_summary'] = summary
            
            results['success'] = True
            logger.info("Automatic analysis completed successfully")
            
        except Exception as e:
            error_msg = f"Error in automatic analysis: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        return results
    
    def _run_mitre_mapping(self, anomalies: pd.DataFrame, confidence_threshold: float) -> Optional[Dict]:
        """Run automatic MITRE ATT&CK mapping."""
        try:
            logger.info("Running automatic MITRE ATT&CK mapping")
            mapping_results = self.mitre_mapper.map_anomalies(anomalies, confidence_threshold)
            
            if mapping_results:
                logger.info(f"MITRE mapping successful: {len(mapping_results)} anomalies mapped")
                # Store in session state for UI access
                st.session_state.mitre_mappings = mapping_results
                st.session_state.mitre_auto_mapped = True
            else:
                logger.warning("No MITRE mappings found with current confidence threshold")
            
            return mapping_results
            
        except Exception as e:
            logger.error(f"Error in MITRE mapping: {str(e)}")
            return None
    
    def _run_risk_scoring(self, anomalies: pd.DataFrame) -> Optional[Dict]:
        """Run automatic risk scoring."""
        try:
            logger.info("Running automatic risk scoring")
            
            # Calculate risk scores for each anomaly
            risk_scores = []
            for idx, anomaly in anomalies.iterrows():
                score = self.risk_scorer.calculate_risk_score(anomaly.to_dict())
                risk_scores.append({
                    'anomaly_id': idx,
                    'risk_score': score,
                    'risk_level': self._get_risk_level(score)
                })
            
            risk_results = {
                'individual_scores': risk_scores,
                'summary': self._summarize_risk_scores(risk_scores)
            }
            
            # Store in session state
            st.session_state.risk_scores = risk_results
            st.session_state.risk_auto_calculated = True
            
            logger.info(f"Risk scoring completed for {len(risk_scores)} anomalies")
            return risk_results
            
        except Exception as e:
            logger.error(f"Error in risk scoring: {str(e)}")
            return None
    
    def _get_risk_level(self, score: float) -> str:
        """Convert risk score to risk level."""
        if score >= 0.8:
            return "Critical"
        elif score >= 0.6:
            return "High" 
        elif score >= 0.4:
            return "Medium"
        elif score >= 0.2:
            return "Low"
        else:
            return "Minimal"
    
    def _summarize_risk_scores(self, risk_scores: list) -> Dict[str, Any]:
        """Generate summary of risk scores."""
        if not risk_scores:
            return {}
        
        scores = [item['risk_score'] for item in risk_scores]
        levels = [item['risk_level'] for item in risk_scores]
        
        from collections import Counter
        level_counts = Counter(levels)
        
        return {
            'total_anomalies': len(risk_scores),
            'avg_risk_score': sum(scores) / len(scores),
            'max_risk_score': max(scores),
            'min_risk_score': min(scores),
            'risk_level_distribution': dict(level_counts),
            'high_risk_count': level_counts.get('Critical', 0) + level_counts.get('High', 0)
        }
    
    def _generate_analysis_summary(self, anomalies: pd.DataFrame, mitre_results: Optional[Dict], 
                                 risk_results: Optional[Dict]) -> Dict[str, Any]:
        """Generate comprehensive analysis summary."""
        summary = {
            'total_anomalies': len(anomalies),
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'mitre_analysis': {},
            'risk_analysis': {},
            'recommendations': []
        }
        
        # MITRE analysis summary
        if mitre_results:
            technique_counts = self.mitre_mapper.get_technique_counts(mitre_results)
            tactic_counts = self.mitre_mapper.get_tactic_counts(mitre_results)
            
            summary['mitre_analysis'] = {
                'mapped_anomalies': len(mitre_results),
                'unique_techniques': len(technique_counts),
                'unique_tactics': len(tactic_counts),
                'top_techniques': list(technique_counts.keys())[:5],
                'top_tactics': list(tactic_counts.keys())[:3]
            }
        
        # Risk analysis summary
        if risk_results:
            summary['risk_analysis'] = risk_results.get('summary', {})
        
        # Generate recommendations
        summary['recommendations'] = self._generate_recommendations(mitre_results, risk_results)
        
        return summary
    
    def _generate_recommendations(self, mitre_results: Optional[Dict], 
                                risk_results: Optional[Dict]) -> list:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # MITRE-based recommendations
        if mitre_results:
            technique_counts = self.mitre_mapper.get_technique_counts(mitre_results)
            
            if 'T1046' in technique_counts:  # Network Service Discovery
                recommendations.append({
                    'priority': 'High',
                    'category': 'Network Security',
                    'action': 'Review network segmentation and access controls',
                    'reason': 'Network service discovery techniques detected'
                })
            
            if 'T1071' in technique_counts:  # Application Layer Protocol
                recommendations.append({
                    'priority': 'Medium',
                    'category': 'Traffic Analysis',
                    'action': 'Implement deep packet inspection for application protocols',
                    'reason': 'Suspicious application layer protocol usage detected'
                })
        
        # Risk-based recommendations
        if risk_results and 'summary' in risk_results:
            summary = risk_results['summary']
            high_risk_count = summary.get('high_risk_count', 0)
            
            if high_risk_count > 0:
                recommendations.append({
                    'priority': 'Critical',
                    'category': 'Incident Response',
                    'action': f'Investigate {high_risk_count} high-risk anomalies immediately',
                    'reason': 'Critical or high-risk threats detected'
                })
        
        # Default recommendations
        if not recommendations:
            recommendations.append({
                'priority': 'Low',
                'category': 'Monitoring',
                'action': 'Continue monitoring for anomalous patterns',
                'reason': 'No immediate threats detected, maintain vigilance'
            })
        
        return recommendations

# Singleton instance
auto_analysis_service = AutoAnalysisService()
