"""
Incident Correlation Engine
Core module for correlating security incidents and building attack timelines
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import json

@dataclass
class SecurityIncident:
    """Data class representing a security incident"""
    id: str
    timestamp: datetime
    severity: str
    incident_type: str
    source_ip: str
    dest_ip: Optional[str]
    protocol: str
    description: str
    indicators: List[str]
    mitre_tactics: List[str]
    confidence_score: float
    raw_data: Dict[str, Any]

class IncidentCorrelationEngine:
    """Engine for correlating security incidents and identifying attack patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.correlation_window_hours = 24
        self.confidence_threshold = 0.7
        
        # MITRE ATT&CK tactic progression patterns
        self.attack_progression_patterns = {
            'typical_progression': [
                'Initial Access',
                'Execution', 
                'Persistence',
                'Privilege Escalation',
                'Defense Evasion',
                'Credential Access',
                'Discovery',
                'Lateral Movement',
                'Collection',
                'Command and Control',
                'Exfiltration',
                'Impact'
            ],
            'fast_attack': [
                'Initial Access',
                'Execution',
                'Lateral Movement',
                'Impact'
            ],
            'stealth_attack': [
                'Initial Access',
                'Defense Evasion',
                'Persistence',
                'Discovery',
                'Credential Access',
                'Lateral Movement',
                'Collection',
                'Exfiltration'
            ]
        }
        
        # Correlation rules
        self.correlation_rules = [
            {
                'name': 'Failed Login + Successful Login',
                'pattern': ['Authentication Failure', 'Authentication Success'],
                'time_window_minutes': 30,
                'confidence_boost': 0.3,
                'description': 'Potential credential stuffing or brute force attack'
            },
            {
                'name': 'Network Scan + Lateral Movement',
                'pattern': ['Port Scan', 'Lateral Movement'],
                'time_window_minutes': 60,
                'confidence_boost': 0.4,
                'description': 'Reconnaissance followed by lateral movement'
            },
            {
                'name': 'Malware + C2 Communication',
                'pattern': ['Malware Detection', 'Command and Control'],
                'time_window_minutes': 120,
                'confidence_boost': 0.5,
                'description': 'Malware establishing command and control'
            }
        ]
    
    def correlate_incidents(self, incidents: List[SecurityIncident]) -> Dict[str, Any]:
        """
        Correlate security incidents to identify attack campaigns
        
        Args:
            incidents: List of security incidents to correlate
            
        Returns:
            Correlation analysis results including attack chains and campaigns
        """
        try:
            if not incidents:
                return {'attack_chains': [], 'campaigns': [], 'statistics': {}}
            
            # Sort incidents by timestamp
            sorted_incidents = sorted(incidents, key=lambda x: x.timestamp)
            
            # Identify attack chains
            attack_chains = self._identify_attack_chains(sorted_incidents)
            
            # Group into campaigns
            campaigns = self._group_into_campaigns(attack_chains)
            
            # Calculate correlation statistics
            statistics = self._calculate_correlation_statistics(incidents, attack_chains, campaigns)
            
            # Generate insights
            insights = self._generate_correlation_insights(attack_chains, campaigns)
            
            return {
                'attack_chains': attack_chains,
                'campaigns': campaigns,
                'statistics': statistics,
                'insights': insights,
                'correlation_timestamp': datetime.now(),
                'total_incidents_analyzed': len(incidents)
            }
            
        except Exception as e:
            self.logger.error(f"Error in incident correlation: {e}")
            return {'error': str(e)}
    
    def _identify_attack_chains(self, incidents: List[SecurityIncident]) -> List[Dict[str, Any]]:
        """Identify sequences of related incidents forming attack chains"""
        attack_chains = []
        processed_incidents = set()
        
        for i, incident in enumerate(incidents):
            if incident.id in processed_incidents:
                continue
            
            # Start a new potential attack chain
            chain = {
                'chain_id': f"chain_{len(attack_chains) + 1}",
                'start_time': incident.timestamp,
                'incidents': [incident],
                'source_ips': {incident.source_ip},
                'dest_ips': {incident.dest_ip} if incident.dest_ip else set(),
                'tactics': set(incident.mitre_tactics),
                'confidence_score': incident.confidence_score,
                'severity': incident.severity
            }
            
            processed_incidents.add(incident.id)
            
            # Look for related incidents within the correlation window
            for j in range(i + 1, len(incidents)):
                candidate = incidents[j]
                
                if candidate.id in processed_incidents:
                    continue
                
                # Check if candidate should be added to this chain
                if self._should_correlate_incidents(chain, candidate):
                    chain['incidents'].append(candidate)
                    chain['source_ips'].add(candidate.source_ip)
                    if candidate.dest_ip:
                        chain['dest_ips'].add(candidate.dest_ip)
                    chain['tactics'].update(candidate.mitre_tactics)
                    chain['end_time'] = candidate.timestamp
                    
                    # Update confidence based on correlation strength
                    correlation_strength = self._calculate_correlation_strength(chain, candidate)
                    chain['confidence_score'] = min(0.95, 
                        chain['confidence_score'] + correlation_strength * 0.1)
                    
                    # Update severity (take highest)
                    severity_weights = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
                    if severity_weights.get(candidate.severity, 0) > severity_weights.get(chain['severity'], 0):
                        chain['severity'] = candidate.severity
                    
                    processed_incidents.add(candidate.id)
            
            # Only keep chains with multiple incidents or high-confidence single incidents
            if len(chain['incidents']) > 1 or chain['confidence_score'] > 0.8:
                # Calculate chain metrics
                chain['duration_minutes'] = (
                    (chain.get('end_time', chain['start_time']) - chain['start_time']).total_seconds() / 60
                )
                chain['incident_count'] = len(chain['incidents'])
                chain['unique_ips'] = len(chain['source_ips'].union(chain['dest_ips']))
                chain['tactic_progression'] = self._analyze_tactic_progression(chain['incidents'])
                chain['attack_pattern'] = self._classify_attack_pattern(chain)
                
                attack_chains.append(chain)
        
        return attack_chains
    
    def _should_correlate_incidents(self, chain: Dict[str, Any], candidate: SecurityIncident) -> bool:
        """Determine if a candidate incident should be correlated with an existing chain"""
        
        # Time window check
        time_diff = (candidate.timestamp - chain['start_time']).total_seconds() / 3600
        if time_diff > self.correlation_window_hours:
            return False
        
        # IP address correlation
        ip_overlap = (
            candidate.source_ip in chain['source_ips'] or
            candidate.dest_ip in chain['dest_ips'] or
            (candidate.dest_ip and candidate.dest_ip in chain['source_ips']) or
            candidate.source_ip in chain['dest_ips']
        )
        
        # MITRE tactic progression correlation
        tactic_correlation = bool(set(candidate.mitre_tactics).intersection(chain['tactics']))
        
        # Apply correlation rules
        rule_match = self._check_correlation_rules(chain, candidate)
        
        # Correlation decision
        correlation_score = 0
        if ip_overlap:
            correlation_score += 0.4
        if tactic_correlation:
            correlation_score += 0.3
        if rule_match:
            correlation_score += 0.4
        
        return correlation_score >= 0.5
    
    def _check_correlation_rules(self, chain: Dict[str, Any], candidate: SecurityIncident) -> bool:
        """Check if incidents match predefined correlation rules"""
        
        for rule in self.correlation_rules:
            pattern = rule['pattern']
            time_window = rule['time_window_minutes']
            
            # Check if we have the pattern elements
            chain_types = [inc.incident_type for inc in chain['incidents']]
            candidate_type = candidate.incident_type
            
            # Look for pattern matches
            for i, pattern_type in enumerate(pattern[:-1]):
                if pattern_type in chain_types:
                    next_pattern_type = pattern[i + 1]
                    if candidate_type == next_pattern_type:
                        # Check time window
                        for chain_incident in chain['incidents']:
                            if chain_incident.incident_type == pattern_type:
                                time_diff = (candidate.timestamp - chain_incident.timestamp).total_seconds() / 60
                                if 0 <= time_diff <= time_window:
                                    return True
        
        return False
    
    def _calculate_correlation_strength(self, chain: Dict[str, Any], candidate: SecurityIncident) -> float:
        """Calculate the strength of correlation between chain and candidate"""
        strength = 0.0
        
        # IP address overlap strength
        if candidate.source_ip in chain['source_ips']:
            strength += 0.3
        if candidate.dest_ip in chain['dest_ips']:
            strength += 0.3
        
        # Temporal proximity (closer in time = stronger correlation)
        time_diff_hours = (candidate.timestamp - chain['start_time']).total_seconds() / 3600
        temporal_strength = max(0, 1 - (time_diff_hours / self.correlation_window_hours))
        strength += temporal_strength * 0.2
        
        # MITRE tactic progression strength
        tactic_overlap = len(set(candidate.mitre_tactics).intersection(chain['tactics']))
        if tactic_overlap > 0:
            strength += 0.2
        
        return min(1.0, strength)
    
    def _analyze_tactic_progression(self, incidents: List[SecurityIncident]) -> Dict[str, Any]:
        """Analyze the progression of MITRE ATT&CK tactics in the incident chain"""
        
        # Extract tactic timeline
        tactic_timeline = []
        for incident in sorted(incidents, key=lambda x: x.timestamp):
            for tactic in incident.mitre_tactics:
                tactic_timeline.append({
                    'timestamp': incident.timestamp,
                    'tactic': tactic,
                    'incident_id': incident.id
                })
        
        # Identify progression pattern
        unique_tactics = []
        for item in tactic_timeline:
            if item['tactic'] not in [t['tactic'] for t in unique_tactics]:
                unique_tactics.append(item)
        
        # Match against known attack patterns
        pattern_match = self._match_attack_pattern([t['tactic'] for t in unique_tactics])
        
        return {
            'tactic_timeline': tactic_timeline,
            'unique_tactics_sequence': [t['tactic'] for t in unique_tactics],
            'pattern_match': pattern_match,
            'progression_speed': self._calculate_progression_speed(unique_tactics),
            'completeness': self._calculate_attack_completeness(unique_tactics)
        }
    
    def _match_attack_pattern(self, tactic_sequence: List[str]) -> Dict[str, Any]:
        """Match observed tactic sequence against known attack patterns"""
        
        best_match = {'pattern_name': 'unknown', 'similarity': 0.0, 'coverage': 0.0}
        
        for pattern_name, pattern_tactics in self.attack_progression_patterns.items():
            # Calculate similarity using sequence alignment
            similarity = self._calculate_sequence_similarity(tactic_sequence, pattern_tactics)
            coverage = len(set(tactic_sequence).intersection(set(pattern_tactics))) / len(pattern_tactics)
            
            combined_score = (similarity + coverage) / 2
            
            if combined_score > best_match['similarity']:
                best_match = {
                    'pattern_name': pattern_name,
                    'similarity': similarity,
                    'coverage': coverage,
                    'combined_score': combined_score
                }
        
        return best_match
    
    def _calculate_sequence_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """Calculate similarity between two sequences using longest common subsequence"""
        
        if not seq1 or not seq2:
            return 0.0
        
        # Simple LCS-based similarity
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        return lcs_length / max(m, n)
    
    def _calculate_progression_speed(self, tactics_timeline: List[Dict]) -> str:
        """Calculate the speed of attack progression"""
        
        if len(tactics_timeline) < 2:
            return "unknown"
        
        total_duration = (tactics_timeline[-1]['timestamp'] - tactics_timeline[0]['timestamp']).total_seconds() / 3600
        tactics_per_hour = len(tactics_timeline) / max(total_duration, 0.1)
        
        if tactics_per_hour > 2:
            return "fast"
        elif tactics_per_hour > 0.5:
            return "medium"
        else:
            return "slow"
    
    def _calculate_attack_completeness(self, tactics_timeline: List[Dict]) -> float:
        """Calculate how complete the attack chain is compared to typical patterns"""
        
        observed_tactics = set(t['tactic'] for t in tactics_timeline)
        typical_tactics = set(self.attack_progression_patterns['typical_progression'])
        
        return len(observed_tactics.intersection(typical_tactics)) / len(typical_tactics)
    
    def _classify_attack_pattern(self, chain: Dict[str, Any]) -> str:
        """Classify the type of attack pattern based on chain characteristics"""
        
        duration = chain.get('duration_minutes', 0)
        incident_count = chain.get('incident_count', 0)
        unique_ips = chain.get('unique_ips', 0)
        
        if duration < 60 and incident_count > 5:
            return "Automated Attack"
        elif duration > 1440 and incident_count < 10:  # > 24 hours, few incidents
            return "Advanced Persistent Threat"
        elif unique_ips > 5:
            return "Distributed Attack"
        elif 'Lateral Movement' in chain['tactics'] and 'Persistence' in chain['tactics']:
            return "Insider Threat / Lateral Movement"
        else:
            return "Standard Attack"
    
    def _group_into_campaigns(self, attack_chains: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group related attack chains into campaigns"""
        
        campaigns = []
        processed_chains = set()
        
        for i, chain in enumerate(attack_chains):
            if i in processed_chains:
                continue
            
            campaign = {
                'campaign_id': f"campaign_{len(campaigns) + 1}",
                'chains': [chain],
                'start_time': chain['start_time'],
                'end_time': chain.get('end_time', chain['start_time']),
                'source_ips': chain['source_ips'].copy(),
                'dest_ips': chain['dest_ips'].copy(),
                'all_tactics': chain['tactics'].copy(),
                'severity': chain['severity']
            }
            
            processed_chains.add(i)
            
            # Look for related chains
            for j, candidate_chain in enumerate(attack_chains[i+1:], i+1):
                if j in processed_chains:
                    continue
                
                # Check if chains should be grouped into same campaign
                if self._should_group_chains(campaign, candidate_chain):
                    campaign['chains'].append(candidate_chain)
                    campaign['source_ips'].update(candidate_chain['source_ips'])
                    campaign['dest_ips'].update(candidate_chain['dest_ips'])
                    campaign['all_tactics'].update(candidate_chain['tactics'])
                    
                    if candidate_chain.get('end_time', candidate_chain['start_time']) > campaign['end_time']:
                        campaign['end_time'] = candidate_chain.get('end_time', candidate_chain['start_time'])
                    
                    processed_chains.add(j)
            
            # Calculate campaign metrics
            campaign['duration_hours'] = (campaign['end_time'] - campaign['start_time']).total_seconds() / 3600
            campaign['total_incidents'] = sum(len(chain['incidents']) for chain in campaign['chains'])
            campaign['chain_count'] = len(campaign['chains'])
            campaign['campaign_type'] = self._classify_campaign(campaign)
            
            campaigns.append(campaign)
        
        return campaigns
    
    def _should_group_chains(self, campaign: Dict[str, Any], candidate_chain: Dict[str, Any]) -> bool:
        """Determine if a chain should be grouped into an existing campaign"""
        
        # IP overlap check
        ip_overlap = bool(
            campaign['source_ips'].intersection(candidate_chain['source_ips']) or
            campaign['dest_ips'].intersection(candidate_chain['dest_ips']) or
            campaign['source_ips'].intersection(candidate_chain['dest_ips']) or
            campaign['dest_ips'].intersection(candidate_chain['source_ips'])
        )
        
        # Temporal proximity (within 7 days)
        time_gap = abs((candidate_chain['start_time'] - campaign['end_time']).total_seconds()) / 3600
        temporal_proximity = time_gap < 168  # 7 days
        
        # Tactical similarity
        tactic_overlap = len(campaign['all_tactics'].intersection(candidate_chain['tactics'])) > 0
        
        # Attack pattern similarity
        campaign_patterns = [chain.get('attack_pattern', '') for chain in campaign['chains']]
        pattern_similarity = candidate_chain.get('attack_pattern', '') in campaign_patterns
        
        # Grouping decision
        if ip_overlap and temporal_proximity:
            return True
        elif tactic_overlap and temporal_proximity and pattern_similarity:
            return True
        else:
            return False
    
    def _classify_campaign(self, campaign: Dict[str, Any]) -> str:
        """Classify the type of campaign based on characteristics"""
        
        duration_hours = campaign['duration_hours']
        chain_count = campaign['chain_count']
        total_incidents = campaign['total_incidents']
        unique_source_ips = len(campaign['source_ips'])
        
        if duration_hours > 168 and chain_count > 3:  # > 1 week, multiple chains
            return "Advanced Persistent Threat Campaign"
        elif unique_source_ips > 10:
            return "Distributed Campaign"
        elif total_incidents > 50 and duration_hours < 24:
            return "Intensive Attack Campaign"
        elif 'Command and Control' in campaign['all_tactics'] and 'Persistence' in campaign['all_tactics']:
            return "Persistent Access Campaign"
        else:
            return "Standard Attack Campaign"
    
    def _calculate_correlation_statistics(self, incidents: List[SecurityIncident], 
                                        attack_chains: List[Dict], campaigns: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive correlation statistics"""
        
        total_incidents = len(incidents)
        correlated_incidents = sum(len(chain['incidents']) for chain in attack_chains)
        correlation_rate = correlated_incidents / max(total_incidents, 1)
        
        # Severity distribution
        severity_counts = {}
        for incident in incidents:
            severity_counts[incident.severity] = severity_counts.get(incident.severity, 0) + 1
        
        # Tactic frequency
        tactic_counts = {}
        for incident in incidents:
            for tactic in incident.mitre_tactics:
                tactic_counts[tactic] = tactic_counts.get(tactic, 0) + 1
        
        # Time distribution
        if incidents:
            time_span = (max(inc.timestamp for inc in incidents) - 
                        min(inc.timestamp for inc in incidents)).total_seconds() / 3600
        else:
            time_span = 0
        
        return {
            'total_incidents': total_incidents,
            'correlated_incidents': correlated_incidents,
            'uncorrelated_incidents': total_incidents - correlated_incidents,
            'correlation_rate': correlation_rate,
            'attack_chain_count': len(attack_chains),
            'campaign_count': len(campaigns),
            'severity_distribution': severity_counts,
            'top_tactics': sorted(tactic_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            'analysis_time_span_hours': time_span,
            'avg_chain_size': np.mean([len(chain['incidents']) for chain in attack_chains]) if attack_chains else 0,
            'confidence_scores': {
                'avg_chain_confidence': np.mean([chain['confidence_score'] for chain in attack_chains]) if attack_chains else 0,
                'high_confidence_chains': len([chain for chain in attack_chains if chain['confidence_score'] > 0.8])
            }
        }
    
    def _generate_correlation_insights(self, attack_chains: List[Dict], campaigns: List[Dict]) -> List[Dict[str, Any]]:
        """Generate actionable insights from correlation analysis"""
        
        insights = []
        
        # High-confidence attack chains
        high_conf_chains = [chain for chain in attack_chains if chain['confidence_score'] > 0.8]
        if high_conf_chains:
            insights.append({
                'type': 'alert',
                'title': 'High-Confidence Attack Chains Detected',
                'description': f"Found {len(high_conf_chains)} attack chains with >80% confidence",
                'severity': 'high',
                'recommendation': 'Prioritize investigation of these correlated incidents'
            })
        
        # Fast progression attacks
        fast_attacks = [chain for chain in attack_chains 
                       if chain.get('tactic_progression', {}).get('progression_speed') == 'fast']
        if fast_attacks:
            insights.append({
                'type': 'warning',
                'title': 'Fast-Moving Attacks Detected',
                'description': f"Found {len(fast_attacks)} rapidly progressing attack chains",
                'severity': 'medium',
                'recommendation': 'Implement immediate containment measures'
            })
        
        # APT-style campaigns
        apt_campaigns = [camp for camp in campaigns if 'Persistent' in camp.get('campaign_type', '')]
        if apt_campaigns:
            insights.append({
                'type': 'alert',
                'title': 'Potential APT Activity',
                'description': f"Detected {len(apt_campaigns)} campaigns with APT characteristics",
                'severity': 'critical',
                'recommendation': 'Initiate advanced threat hunting procedures'
            })
        
        # Lateral movement patterns
        lateral_chains = [chain for chain in attack_chains if 'Lateral Movement' in chain['tactics']]
        if lateral_chains:
            insights.append({
                'type': 'warning',
                'title': 'Lateral Movement Activity',
                'description': f"Found {len(lateral_chains)} chains involving lateral movement",
                'severity': 'high',
                'recommendation': 'Review network segmentation and access controls'
            })
        
        return insights
