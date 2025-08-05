"""
MITRE ATT&CK mapping module for the Network Anomaly Detection Platform.
Maps network anomalies to MITRE ATT&CK techniques and tactics.
"""

import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict

class MitreMapper:
    """Maps anomalies to MITRE ATT&CK techniques and tactics."""
    
    def __init__(self, mitre_data_path="config/mitre_attack_data.json"):
        """
        Initialize the MITRE mapper.
        
        Args:
            mitre_data_path (str): Path to the MITRE ATT&CK data file
        """
        self.mitre_data_path = mitre_data_path
        self.mitre_data = self._load_mitre_data()
        self.mapping_rules = self._initialize_mapping_rules()
    
    def _load_mitre_data(self):
        """
        Load MITRE ATT&CK data from JSON file.
        
        Returns:
            dict: MITRE ATT&CK data
        """
        try:
            if os.path.exists(self.mitre_data_path):
                with open(self.mitre_data_path, 'r') as f:
                    return json.load(f)
            else:
                print(f"MITRE data file not found: {self.mitre_data_path}")
                return self._initialize_default_mitre_data()
        except Exception as e:
            print(f"Error loading MITRE data: {str(e)}")
            return self._initialize_default_mitre_data()
    
    def _initialize_default_mitre_data(self):
        """
        Initialize default MITRE ATT&CK data if file not found.
        
        Returns:
            dict: Default MITRE ATT&CK data
        """
        # Create a minimal set of MITRE ATT&CK data
        return {
            "techniques": [
                {
                    "technique_id": "T1046",
                    "name": "Network Service Discovery",
                    "tactic": "Discovery",
                    "description": "Adversaries may attempt to get a listing of services running on remote hosts."
                },
                {
                    "technique_id": "T1048",
                    "name": "Exfiltration Over Alternative Protocol",
                    "tactic": "Exfiltration",
                    "description": "Adversaries may steal data by exfiltrating it over a different protocol than that of the existing command and control channel."
                },
                {
                    "technique_id": "T1071",
                    "name": "Application Layer Protocol",
                    "tactic": "Command and Control",
                    "description": "Adversaries may communicate using application layer protocols to avoid detection."
                },
                {
                    "technique_id": "T1095",
                    "name": "Non-Application Layer Protocol",
                    "tactic": "Command and Control",
                    "description": "Adversaries may use a non-application layer protocol for communication between host and C2 server."
                },
                {
                    "technique_id": "T1059",
                    "name": "Command and Scripting Interpreter",
                    "tactic": "Execution",
                    "description": "Adversaries may abuse command and script interpreters to execute commands, scripts, or binaries."
                }
            ],
            "tactics": [
                {
                    "tactic_id": "TA0007",
                    "name": "Discovery",
                    "description": "The adversary is trying to figure out your environment."
                },
                {
                    "tactic_id": "TA0010",
                    "name": "Exfiltration",
                    "description": "The adversary is trying to steal data."
                },
                {
                    "tactic_id": "TA0011",
                    "name": "Command and Control",
                    "description": "The adversary is trying to communicate with compromised systems to control them."
                },
                {
                    "tactic_id": "TA0002",
                    "name": "Execution",
                    "description": "The adversary is trying to run malicious code."
                }
            ]
        }
    
    def _initialize_mapping_rules(self):
        """
        Initialize rules for mapping anomalies to MITRE techniques.
        
        Returns:
            dict: Mapping rules
        """
        # Create rules based on network protocols, ports, and patterns
        rules = {
            "protocol_rules": {
                "SSH": ["T1021.004"],  # Remote Services: SSH
                "TELNET": ["T1021.001"],  # Remote Services: Telnet
                "FTP": ["T1048.003"],  # Exfiltration Over Unencrypted/Obfuscated Non-C2 Protocol
                "SFTP": ["T1048.002"],  # Exfiltration Over Encrypted Non-C2 Protocol
                "HTTP": ["T1071.001"],  # Application Layer Protocol: Web Protocols
                "HTTPS": ["T1071.001"],  # Application Layer Protocol: Web Protocols
                "DNS": ["T1071.004"],  # Application Layer Protocol: DNS
                "ICMP": ["T1095"],  # Non-Application Layer Protocol
                "ICMPv6": ["T1095"],  # Non-Application Layer Protocol
                "SMB": ["T1021.002"],  # Remote Services: SMB/Windows Admin Shares
                "SMB2": ["T1021.002"],  # Remote Services: SMB/Windows Admin Shares
                "RDP": ["T1021.001"],  # Remote Services: Remote Desktop Protocol
                "SMTP": ["T1048.003"],  # Exfiltration Over Alternative Protocol
                "IRC": ["T1071.001"],  # Application Layer Protocol: Web Protocols
                "TCP": ["T1095"],  # Non-Application Layer Protocol
                "UDP": ["T1095"],  # Non-Application Layer Protocol
                "TLS": ["T1071.001"],  # Application Layer Protocol: Web Protocols
                "SSL": ["T1071.001"],  # Application Layer Protocol: Web Protocols
                "SNMP": ["T1046"],  # Network Service Discovery
                "DHCP": ["T1046"],  # Network Service Discovery
                "ARP": ["T1046"],  # Network Service Discovery
                "NETBIOS": ["T1046"],  # Network Service Discovery
            },
            "port_rules": {
                22: ["T1021.004"],  # SSH
                23: ["T1021.001"],  # Telnet
                21: ["T1048.003"],  # FTP
                20: ["T1048.003"],  # FTP Data
                80: ["T1071.001"],  # HTTP
                443: ["T1071.001"],  # HTTPS
                53: ["T1071.004"],  # DNS
                445: ["T1021.002"],  # SMB
                139: ["T1021.002"],  # NetBIOS
                3389: ["T1021.001"],  # RDP
                25: ["T1048.003"],  # SMTP
                110: ["T1048.003"],  # POP3
                143: ["T1048.003"],  # IMAP
                993: ["T1048.002"],  # IMAPS
                995: ["T1048.002"],  # POP3S
                6667: ["T1071.001"],  # IRC
                6666: ["T1071.001"],  # IRC
                6668: ["T1071.001"],  # IRC
                6669: ["T1071.001"],  # IRC
                4444: ["T1571"],  # Non-Standard Port
                8080: ["T1071.001"],  # Alternative HTTP
                8443: ["T1071.001"],  # Alternative HTTPS
                1433: ["T1046"],  # MSSQL
                3306: ["T1046"],  # MySQL
                5432: ["T1046"],  # PostgreSQL
                161: ["T1046"],  # SNMP
                162: ["T1046"],  # SNMP Trap
                135: ["T1021.002"],  # RPC
                636: ["T1021.002"],  # LDAPS
                389: ["T1021.002"],  # LDAP
                88: ["T1021.001"],   # Kerberos
                123: ["T1095"],  # NTP
                69: ["T1048.003"],   # TFTP
            },
            "pattern_rules": {
                "scan": ["T1046"],  # Network Service Discovery
                "port scan": ["T1046"],  # Network Service Discovery
                "data exfiltration": ["T1048"],  # Exfiltration Over Alternative Protocol
                "exfiltration": ["T1048"],  # Exfiltration Over Alternative Protocol
                "lateral movement": ["T1021"],  # Remote Services
                "lateral": ["T1021"],  # Remote Services
                "command control": ["T1071"],  # Application Layer Protocol
                "command and control": ["T1071"],  # Application Layer Protocol
                "c2": ["T1071"],  # Application Layer Protocol
                "c&c": ["T1071"],  # Application Layer Protocol
                "execution": ["T1059"],  # Command and Scripting Interpreter
                "execute": ["T1059"],  # Command and Scripting Interpreter
                "shell": ["T1059"],  # Command and Scripting Interpreter
                "cmd": ["T1059"],  # Command and Scripting Interpreter
                "powershell": ["T1059.001"],  # PowerShell
                "bash": ["T1059.004"],  # Unix Shell
                "reconnaissance": ["T1046"],  # Network Service Discovery
                "recon": ["T1046"],  # Network Service Discovery
                "discovery": ["T1046"],  # Network Service Discovery
                "brute force": ["T1110"],  # Brute Force
                "bruteforce": ["T1110"],  # Brute Force
                "credential": ["T1110"],  # Brute Force
                "login": ["T1110"],  # Brute Force
                "authentication": ["T1110"],  # Brute Force
                "tunnel": ["T1572"],  # Protocol Tunneling
                "tunneling": ["T1572"],  # Protocol Tunneling
                "proxy": ["T1090"],  # Proxy
                "backdoor": ["T1071"],  # Application Layer Protocol
                "malware": ["T1071"],  # Application Layer Protocol
                "suspicious": ["T1071"],  # Application Layer Protocol
                "anomaly": ["T1071"],  # Application Layer Protocol
            }
        }
        
        return rules
    
    def map_anomaly(self, anomaly, confidence_threshold=0.6):
        """
        Map an anomaly to MITRE ATT&CK techniques.
        
        Args:
            anomaly (dict): Anomaly data
            confidence_threshold (float): Minimum confidence threshold
        
        Returns:
            dict: Mapping results
        """
        mappings = []
        
        # Extract relevant fields from the anomaly - try multiple column name formats
        protocol = (
            anomaly.get('_ws.col.Protocol') or 
            anomaly.get('_ws_col_Protocol') or 
            anomaly.get('protocol') or 
            ''
        )
        
        # Try different port field names
        src_port = (
            anomaly.get('tcp.srcport') or
            anomaly.get('tcp_srcport') or 
            anomaly.get('udp.srcport') or
            anomaly.get('udp_srcport') or 
            anomaly.get('src_port') or
            ''
        )
        
        dst_port = (
            anomaly.get('tcp.dstport') or
            anomaly.get('tcp_dstport') or 
            anomaly.get('udp.dstport') or
            anomaly.get('udp_dstport') or 
            anomaly.get('dst_port') or
            ''
        )
        
        info = (
            anomaly.get('_ws.col.Info') or
            anomaly.get('_ws_col_Info') or 
            anomaly.get('info') or
            ''
        )
        
        # Convert ports to integers if possible
        try:
            src_port = int(float(src_port)) if src_port else None
        except (ValueError, TypeError):
            src_port = None
            
        try:
            dst_port = int(float(dst_port)) if dst_port else None
        except (ValueError, TypeError):
            dst_port = None
        
        # Debug information
        print(f"Mapping anomaly: protocol={protocol}, src_port={src_port}, dst_port={dst_port}, info={info[:50]}...")
        
        # Map based on protocol
        if protocol and protocol.upper() in self.mapping_rules["protocol_rules"]:
            for technique_id in self.mapping_rules["protocol_rules"][protocol.upper()]:
                technique = self._get_technique_by_id(technique_id)
                if technique:
                    mappings.append({
                        "technique": technique,
                        "confidence": 0.8,
                        "reason": f"Protocol match: {protocol}"
                    })
                    print(f"Protocol mapping: {protocol} -> {technique_id}")
        
        # Map based on destination port
        if dst_port and dst_port in self.mapping_rules["port_rules"]:
            for technique_id in self.mapping_rules["port_rules"][dst_port]:
                technique = self._get_technique_by_id(technique_id)
                if technique:
                    mappings.append({
                        "technique": technique,
                        "confidence": 0.7,
                        "reason": f"Destination port match: {dst_port}"
                    })
                    print(f"Port mapping: {dst_port} -> {technique_id}")
        
        # Map based on source port (for outbound connections)
        if src_port and src_port in self.mapping_rules["port_rules"]:
            for technique_id in self.mapping_rules["port_rules"][src_port]:
                technique = self._get_technique_by_id(technique_id)
                if technique:
                    mappings.append({
                        "technique": technique,
                        "confidence": 0.6,
                        "reason": f"Source port match: {src_port}"
                    })
                    print(f"Source port mapping: {src_port} -> {technique_id}")
        
        # Map based on patterns in info
        if info:
            for pattern, technique_ids in self.mapping_rules["pattern_rules"].items():
                if pattern.lower() in info.lower():
                    for technique_id in technique_ids:
                        technique = self._get_technique_by_id(technique_id)
                        if technique:
                            mappings.append({
                                "technique": technique,
                                "confidence": 0.6,
                                "reason": f"Pattern match in info: {pattern}"
                            })
                            print(f"Pattern mapping: {pattern} -> {technique_id}")
        
        # Add some heuristic mappings based on anomaly score or other features
        anomaly_score = anomaly.get('anomaly_score', 0)
        if anomaly_score > 0.8:  # High anomaly score
            # High anomaly scores might indicate data exfiltration or C2 communication
            for technique_id in ["T1048", "T1071"]:
                technique = self._get_technique_by_id(technique_id)
                if technique:
                    mappings.append({
                        "technique": technique,
                        "confidence": 0.5,
                        "reason": f"High anomaly score: {anomaly_score:.3f}"
                    })
        
        # Filter out low confidence mappings
        mappings = [m for m in mappings if m["confidence"] >= confidence_threshold]
        
        # Remove duplicates (keep highest confidence)
        unique_mappings = {}
        for mapping in mappings:
            technique_id = mapping["technique"]["technique_id"]
            if technique_id not in unique_mappings or mapping["confidence"] > unique_mappings[technique_id]["confidence"]:
                unique_mappings[technique_id] = mapping
        
        result = list(unique_mappings.values())
        print(f"Final mappings: {len(result)} techniques mapped")
        return result
    
    def map_anomalies(self, anomalies, confidence_threshold=0.6):
        """
        Map multiple anomalies to MITRE ATT&CK techniques.
        
        Args:
            anomalies (pd.DataFrame): DataFrame of anomalies
            confidence_threshold (float): Minimum confidence threshold
        
        Returns:
            dict: Mapping results for each anomaly
        """
        results = {}
        
        for idx, anomaly in anomalies.iterrows():
            anomaly_dict = anomaly.to_dict()
            mappings = self.map_anomaly(anomaly_dict, confidence_threshold)
            
            if mappings:
                results[idx] = mappings
        
        return results
    
    def _get_technique_by_id(self, technique_id):
        """
        Get technique details by ID.
        
        Args:
            technique_id (str): Technique ID
        
        Returns:
            dict: Technique details
        """
        for technique in self.mitre_data.get("techniques", []):
            if technique.get("technique_id") == technique_id or technique_id in technique.get("technique_id", ""):
                return technique
        
        # Handle sub-techniques by looking for parent technique
        if "." in technique_id:
            parent_id = technique_id.split(".")[0]
            for technique in self.mitre_data.get("techniques", []):
                if technique.get("technique_id") == parent_id:
                    # Create a copy of the parent with modified ID
                    sub_technique = technique.copy()
                    sub_technique["technique_id"] = technique_id
                    sub_technique["name"] = f"{technique['name']} (Specific Variant)"
                    return sub_technique
        
        return None
    
    def get_technique_details(self, technique_id):
        """
        Get detailed information about a technique.
        
        Args:
            technique_id (str): Technique ID
        
        Returns:
            dict: Technique details
        """
        technique = self._get_technique_by_id(technique_id)
        
        if not technique:
            return None
        
        # Get tactic details
        tactic_name = technique.get("tactic")
        tactic_details = None
        
        for tactic in self.mitre_data.get("tactics", []):
            if tactic.get("name") == tactic_name:
                tactic_details = tactic
                break
        
        # Combine technique and tactic details
        details = {
            "technique_id": technique.get("technique_id"),
            "technique_name": technique.get("name"),
            "technique_description": technique.get("description"),
            "tactic": tactic_name,
            "tactic_id": tactic_details.get("tactic_id") if tactic_details else None,
            "tactic_description": tactic_details.get("description") if tactic_details else None
        }
        
        return details
    
    def get_all_techniques(self):
        """
        Get all techniques in the MITRE ATT&CK data.
        
        Returns:
            list: All techniques
        """
        return self.mitre_data.get("techniques", [])
    
    def get_all_tactics(self):
        """
        Get all tactics in the MITRE ATT&CK data.
        
        Returns:
            list: All tactics
        """
        return self.mitre_data.get("tactics", [])
    
    def add_custom_rule(self, rule_type, key, technique_ids):
        """
        Add a custom mapping rule.
        
        Args:
            rule_type (str): Type of rule ('protocol_rules', 'port_rules', 'pattern_rules')
            key (str or int): Key for the rule
            technique_ids (list): List of technique IDs
        
        Returns:
            bool: True if successful, False otherwise
        """
        if rule_type not in self.mapping_rules:
            return False
        
        # Convert key to appropriate type for port rules
        if rule_type == "port_rules" and isinstance(key, str):
            try:
                key = int(key)
            except (ValueError, TypeError):
                return False
        
        # Add or update the rule
        self.mapping_rules[rule_type][key] = technique_ids
        
        return True
    
    def get_tactic_counts(self, mapping_results):
        """
        Count the number of anomalies mapped to each tactic.
        
        Args:
            mapping_results (dict): Results from map_anomalies()
        
        Returns:
            dict: Counts for each tactic
        """
        tactic_counts = defaultdict(int)
        
        for anomaly_idx, mappings in mapping_results.items():
            for mapping in mappings:
                tactic = mapping["technique"].get("tactic")
                if tactic:
                    tactic_counts[tactic] += 1
        
        return dict(tactic_counts)
    
    def get_technique_counts(self, mapping_results):
        """
        Count the number of anomalies mapped to each technique.
        
        Args:
            mapping_results (dict): Results from map_anomalies()
        
        Returns:
            dict: Counts for each technique
        """
        technique_counts = defaultdict(int)
        
        for anomaly_idx, mappings in mapping_results.items():
            for mapping in mappings:
                technique_id = mapping["technique"].get("technique_id")
                technique_name = mapping["technique"].get("name")
                if technique_id and technique_name:
                    key = f"{technique_id}: {technique_name}"
                    technique_counts[key] += 1
        
        return dict(technique_counts)
    
    def save_custom_mappings(self, output_path=None):
        """
        Save custom mapping rules to a JSON file.
        
        Args:
            output_path (str, optional): Output file path
        
        Returns:
            bool: True if successful, False otherwise
        """
        if output_path is None:
            output_path = os.path.join(os.path.dirname(self.mitre_data_path), "custom_mappings.json")
        
        try:
            with open(output_path, 'w') as f:
                json.dump(self.mapping_rules, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving custom mappings: {str(e)}")
            return False
    
    def load_custom_mappings(self, input_path=None):
        """
        Load custom mapping rules from a JSON file.
        
        Args:
            input_path (str, optional): Input file path
        
        Returns:
            bool: True if successful, False otherwise
        """
        if input_path is None:
            input_path = os.path.join(os.path.dirname(self.mitre_data_path), "custom_mappings.json")
        
        try:
            if os.path.exists(input_path):
                with open(input_path, 'r') as f:
                    custom_mappings = json.load(f)
                
                # Update mapping rules
                for rule_type, rules in custom_mappings.items():
                    if rule_type in self.mapping_rules:
                        self.mapping_rules[rule_type].update(rules)
                
                return True
            else:
                return False
        except Exception as e:
            print(f"Error loading custom mappings: {str(e)}")
            return False