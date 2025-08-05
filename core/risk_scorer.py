"""
Risk scoring module for the Network Anomaly Detection Platform.
Calculates risk scores for anomalies based on various factors.
"""

import numpy as np
import pandas as pd

class RiskScorer:
    """Risk scoring engine for anomalies."""
    
    def __init__(self, config=None):
        """
        Initialize the risk scorer.
        
        Args:
            config (dict, optional): Configuration dictionary with scoring weights
        """
        # Default weights if not provided in config
        self.weights = {
            'anomaly_score': 0.6,          # Weight for the raw anomaly score
            'protocol_risk': 0.2,          # Weight for protocol risk
            'port_risk': 0.1,              # Weight for port risk
            'geoip_risk': 0.1,             # Weight for geographical risk
            'temporal_risk': 0.05,         # Weight for time-based risk
            'volume_risk': 0.05            # Weight for volume-based risk
        }
        
        # Protocol risk levels (higher = riskier)
        self.protocol_risk = {
            'SSH': 0.6,
            'TELNET': 0.8,
            'FTP': 0.7,
            'HTTP': 0.4,
            'HTTPS': 0.3,
            'DNS': 0.5,
            'ICMP': 0.6,
            'SMB': 0.7,
            'RDP': 0.8,
            'SMTP': 0.6,
            'IRC': 0.7
        }
        
        # Port risk levels (higher = riskier)
        self.high_risk_ports = [22, 23, 3389, 445, 135, 137, 138, 139, 1433, 3306, 5432]
        
        # Update weights from config if provided
        if config and 'risk_scoring' in config:
            for key, value in config['risk_scoring'].items():
                if key in self.weights:
                    self.weights[key] = value
    
    def calculate_risk(self, df, anomaly_scores, threshold=None):
        """
        Calculate risk scores for each row in the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing network data
            anomaly_scores (np.ndarray): Array of anomaly scores
            threshold (float, optional): Threshold for anomaly detection
        
        Returns:
            np.ndarray: Array of risk scores (0-100)
        """
        # Initialize risk score with normalized anomaly score
        normalized_anomaly_scores = self._normalize_scores(anomaly_scores)
        risk_scores = normalized_anomaly_scores * self.weights['anomaly_score']
        
        # Add protocol risk if available
        if '_ws_col_Protocol' in df.columns:
            protocol_risks = self._calculate_protocol_risk(df['_ws_col_Protocol'])
            risk_scores += protocol_risks * self.weights['protocol_risk']
        
        # Add port risk if available
        port_risks = np.zeros(len(df))
        if 'tcp_dstport' in df.columns:
            tcp_port_risks = self._calculate_port_risk(df['tcp_dstport'])
            port_risks = np.maximum(port_risks, tcp_port_risks)
        if 'udp_dstport' in df.columns:
            udp_port_risks = self._calculate_port_risk(df['udp_dstport'])
            port_risks = np.maximum(port_risks, udp_port_risks)
        risk_scores += port_risks * self.weights['port_risk']
        
        # Scale to 0-100 range
        final_risk_scores = np.clip(risk_scores * 100, 0, 100)
        
        return final_risk_scores
    
    def calculate_risk_score(self, anomaly_data):
        """
        Calculate risk score for a single anomaly.
        
        Args:
            anomaly_data (dict): Dictionary containing anomaly information
        
        Returns:
            float: Risk score (0-1)
        """
        risk_score = 0.0
        
        # Base risk from anomaly score
        anomaly_score = anomaly_data.get('anomaly_score', 0)
        if anomaly_score > 0:
            # Normalize anomaly score to 0-1 range (assuming max reasonable score is 1.0)
            normalized_score = min(abs(anomaly_score), 1.0)
            risk_score += normalized_score * self.weights['anomaly_score']
        
        # Protocol-based risk
        protocol = anomaly_data.get('_ws_col_Protocol', '').upper()
        if protocol in self.protocol_risk:
            risk_score += self.protocol_risk[protocol] * self.weights['protocol_risk']
        
        # Port-based risk
        tcp_port = anomaly_data.get('tcp_dstport')
        udp_port = anomaly_data.get('udp_dstport')
        
        port_risk = 0.0
        if tcp_port and tcp_port in self.high_risk_ports:
            port_risk = 0.8
        if udp_port and udp_port in self.high_risk_ports:
            port_risk = max(port_risk, 0.8)
        
        risk_score += port_risk * self.weights['port_risk']
        
        # Ensure risk score is between 0 and 1
        return min(max(risk_score, 0.0), 1.0)
    
    def _normalize_scores(self, scores):
        """
        Normalize scores to 0-1 range.
        
        Args:
            scores (np.ndarray): Array of scores
        
        Returns:
            np.ndarray: Normalized scores
        """
        if len(scores) == 0:
            return scores
        
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score == min_score:
            return np.ones_like(scores) * 0.5
        
        return (scores - min_score) / (max_score - min_score)
    
    def _calculate_protocol_risk(self, protocols):
        """
        Calculate risk based on protocol.
        
        Args:
            protocols (pd.Series): Series of protocol values
        
        Returns:
            np.ndarray: Protocol risk scores
        """
        # Default risk for unknown protocols
        default_risk = 0.5
        
        # Calculate risk for each protocol
        risks = np.zeros(len(protocols))
        
        for i, protocol in enumerate(protocols):
            if pd.isna(protocol):
                risks[i] = default_risk
            else:
                # Convert to uppercase for case-insensitive matching
                protocol_upper = str(protocol).upper()
                
                # Check for exact matches
                if protocol_upper in self.protocol_risk:
                    risks[i] = self.protocol_risk[protocol_upper]
                else:
                    # Check for partial matches
                    for known_protocol, risk in self.protocol_risk.items():
                        if known_protocol in protocol_upper:
                            risks[i] = risk
                            break
                    else:
                        # No match found
                        risks[i] = default_risk
        
        return risks
    
    def _calculate_port_risk(self, ports):
        """
        Calculate risk based on port numbers.
        
        Args:
            ports (pd.Series): Series of port values
        
        Returns:
            np.ndarray: Port risk scores
        """
        # Default risk for unknown or missing ports
        default_risk = 0.3
        
        # Calculate risk for each port
        risks = np.ones(len(ports)) * default_risk
        
        for i, port in enumerate(ports):
            if pd.isna(port):
                continue
                
            try:
                port_num = int(port)
                
                # Check if it's a high risk port
                if port_num in self.high_risk_ports:
                    risks[i] = 0.8
                # Check if it's a privileged port
                elif port_num < 1024:
                    risks[i] = 0.6
                # Check if it's an ephemeral port
                elif port_num >= 49152 and port_num <= 65535:
                    risks[i] = 0.4
                # Otherwise it's a registered port
                else:
                    risks[i] = 0.5
            except (ValueError, TypeError):
                # If port can't be converted to int
                continue
        
        return risks
    
    def categorize_risk(self, risk_scores):
        """
        Categorize risk scores into text categories.
        
        Args:
            risk_scores (np.ndarray): Array of risk scores (0-100)
        
        Returns:
            list: List of risk categories
        """
        categories = []
        
        for score in risk_scores:
            if score >= 80:
                categories.append("Critical")
            elif score >= 60:
                categories.append("High")
            elif score >= 40:
                categories.append("Medium")
            elif score >= 20:
                categories.append("Low")
            else:
                categories.append("Minimal")
        
        return categories