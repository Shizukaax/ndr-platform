"""
Security Intelligence Module for Network Anomaly Detection Platform.
Provides threat feed integration, IOC correlation, and security enrichment.
"""

import pandas as pd
import numpy as np
import requests
import json
import logging
import hashlib
import ipaddress
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
import streamlit as st
from dataclasses import dataclass, asdict
import re

logger = logging.getLogger("security_intelligence")

@dataclass
class ThreatIndicator:
    """Data class for threat indicators."""
    ioc_type: str  # ip, domain, hash, url, etc.
    value: str
    threat_type: str  # malware, phishing, c2, etc.
    confidence: float  # 0.0 to 1.0
    source: str
    first_seen: datetime
    last_seen: datetime
    description: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class SecurityEnrichment:
    """Data class for security enrichment results."""
    original_value: str
    enrichment_type: str
    threat_indicators: List[ThreatIndicator]
    risk_score: float  # 0.0 to 1.0
    reputation_score: float  # 0.0 to 1.0
    geolocation: Dict[str, Any] = None
    whois_info: Dict[str, Any] = None
    
class ThreatIntelligenceManager:
    """Manages threat intelligence feeds and IOC correlation."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.indicators = {}  # Store threat indicators
        self.feeds = {}  # Store feed configurations
        self.cache = {}  # Cache enrichment results
        self.cache_ttl = 3600  # 1 hour cache TTL
        
        # Load configuration
        if config_path:
            self.load_config(config_path)
        else:
            self.setup_default_feeds()
    
    def setup_default_feeds(self):
        """Setup default threat intelligence feeds."""
        self.feeds = {
            "abuse_ch": {
                "name": "Abuse.ch Malware Bazaar",
                "url": "https://bazaar.abuse.ch/export/csv/recent/",
                "type": "malware_hashes",
                "enabled": True,
                "format": "csv"
            },
            "alienvault": {
                "name": "AlienVault OTX",
                "url": "https://otx.alienvault.com/api/v1/indicators/export",
                "type": "mixed",
                "enabled": False,  # Requires API key
                "format": "json"
            },
            "emergingthreats": {
                "name": "Emerging Threats",
                "url": "https://rules.emergingthreats.net/fwrules/emerging-Block-IPs.txt",
                "type": "ip_addresses",
                "enabled": True,
                "format": "text"
            }
        }
    
    def load_config(self, config_path: str):
        """Load threat intelligence configuration from file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.feeds = config.get('threat_feeds', {})
                self.cache_ttl = config.get('cache_ttl', 3600)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self.setup_default_feeds()
    
    def update_threat_feeds(self) -> Dict[str, Any]:
        """Update threat intelligence feeds."""
        results = {
            "updated_feeds": [],
            "failed_feeds": [],
            "total_indicators": 0
        }
        
        for feed_id, feed_config in self.feeds.items():
            if not feed_config.get("enabled", False):
                continue
                
            try:
                indicators = self._fetch_feed(feed_id, feed_config)
                self.indicators[feed_id] = indicators
                results["updated_feeds"].append(feed_id)
                results["total_indicators"] += len(indicators)
                logger.info(f"Updated feed {feed_id}: {len(indicators)} indicators")
                
            except Exception as e:
                logger.error(f"Failed to update feed {feed_id}: {e}")
                results["failed_feeds"].append({"feed": feed_id, "error": str(e)})
        
        return results
    
    def _fetch_feed(self, feed_id: str, feed_config: Dict[str, Any]) -> List[ThreatIndicator]:
        """Fetch indicators from a specific threat feed."""
        indicators = []
        
        try:
            # Add headers to mimic legitimate requests
            headers = {
                'User-Agent': 'Network-Anomaly-Detection-Platform/1.0',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            }
            
            response = requests.get(feed_config["url"], headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parse based on format
            if feed_config["format"] == "csv":
                indicators = self._parse_csv_feed(response.text, feed_id, feed_config)
            elif feed_config["format"] == "json":
                indicators = self._parse_json_feed(response.json(), feed_id, feed_config)
            elif feed_config["format"] == "text":
                indicators = self._parse_text_feed(response.text, feed_id, feed_config)
                
        except requests.RequestException as e:
            logger.error(f"Network error fetching {feed_id}: {e}")
        except Exception as e:
            logger.error(f"Error parsing {feed_id}: {e}")
        
        return indicators
    
    def _parse_csv_feed(self, content: str, feed_id: str, feed_config: Dict[str, Any]) -> List[ThreatIndicator]:
        """Parse CSV format threat feed."""
        indicators = []
        lines = content.strip().split('\n')
        
        # Skip header if present
        if lines and not lines[0].strip().startswith('#'):
            lines = lines[1:]
        
        for line in lines:
            if line.strip() and not line.startswith('#'):
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        indicator = ThreatIndicator(
                            ioc_type="hash" if feed_config["type"] == "malware_hashes" else "unknown",
                            value=parts[0].strip(),
                            threat_type=parts[1].strip() if len(parts) > 1 else "unknown",
                            confidence=0.7,  # Default confidence
                            source=feed_config["name"],
                            first_seen=datetime.now(),
                            last_seen=datetime.now(),
                            description=parts[2].strip() if len(parts) > 2 else ""
                        )
                        indicators.append(indicator)
                    except Exception as e:
                        logger.warning(f"Failed to parse line in {feed_id}: {line[:50]}...")
        
        return indicators
    
    def _parse_text_feed(self, content: str, feed_id: str, feed_config: Dict[str, Any]) -> List[ThreatIndicator]:
        """Parse text format threat feed (e.g., IP lists)."""
        indicators = []
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    # Validate IP address
                    ipaddress.ip_address(line)
                    indicator = ThreatIndicator(
                        ioc_type="ip",
                        value=line,
                        threat_type="malicious_ip",
                        confidence=0.8,
                        source=feed_config["name"],
                        first_seen=datetime.now(),
                        last_seen=datetime.now(),
                        description="Malicious IP from threat feed"
                    )
                    indicators.append(indicator)
                except ValueError:
                    # Not a valid IP, skip
                    continue
                except Exception as e:
                    logger.warning(f"Failed to parse IP in {feed_id}: {line}")
        
        return indicators
    
    def _parse_json_feed(self, data: Dict[str, Any], feed_id: str, feed_config: Dict[str, Any]) -> List[ThreatIndicator]:
        """Parse JSON format threat feed."""
        indicators = []
        
        # This would need to be customized based on the specific JSON structure
        # of each threat feed. For now, we'll implement a generic parser.
        
        try:
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict) and 'results' in data:
                items = data['results']
            else:
                items = []
            
            for item in items:
                if isinstance(item, dict):
                    indicator = ThreatIndicator(
                        ioc_type=item.get('type', 'unknown'),
                        value=item.get('indicator', item.get('value', '')),
                        threat_type=item.get('threat_type', 'unknown'),
                        confidence=float(item.get('confidence', 0.5)),
                        source=feed_config["name"],
                        first_seen=datetime.now(),
                        last_seen=datetime.now(),
                        description=item.get('description', '')
                    )
                    indicators.append(indicator)
        except Exception as e:
            logger.error(f"Error parsing JSON feed {feed_id}: {e}")
        
        return indicators
    
    def correlate_with_threats(self, data: pd.DataFrame) -> pd.DataFrame:
        """Correlate network data with threat intelligence."""
        enriched_data = data.copy()
        
        # Initialize threat correlation columns
        enriched_data['threat_indicators'] = None
        enriched_data['threat_score'] = 0.0
        enriched_data['threat_types'] = None
        enriched_data['ioc_sources'] = None
        
        # Find IP address columns
        ip_columns = self._find_ip_columns(data)
        
        for idx, row in data.iterrows():
            threats_found = []
            threat_score = 0.0
            threat_types = set()
            ioc_sources = set()
            
            # Check IP addresses
            for col in ip_columns:
                ip_value = str(row[col])
                if ip_value and ip_value != 'nan':
                    threat_info = self._check_ip_threats(ip_value)
                    if threat_info:
                        threats_found.extend(threat_info)
                        for threat in threat_info:
                            threat_score = max(threat_score, threat.confidence)
                            threat_types.add(threat.threat_type)
                            ioc_sources.add(threat.source)
            
            # Update enriched data
            if threats_found:
                enriched_data.at[idx, 'threat_indicators'] = len(threats_found)
                enriched_data.at[idx, 'threat_score'] = threat_score
                enriched_data.at[idx, 'threat_types'] = ', '.join(threat_types)
                enriched_data.at[idx, 'ioc_sources'] = ', '.join(ioc_sources)
        
        return enriched_data
    
    def _find_ip_columns(self, data: pd.DataFrame) -> List[str]:
        """Find columns that likely contain IP addresses."""
        ip_columns = []
        
        # Common IP column names
        ip_patterns = ['ip', 'addr', 'src', 'dst', 'source', 'destination', 'host']
        
        for col in data.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in ip_patterns):
                # Verify it actually contains IP-like data
                sample_values = data[col].dropna().head(10)
                ip_like_count = 0
                
                for value in sample_values:
                    try:
                        ipaddress.ip_address(str(value))
                        ip_like_count += 1
                    except ValueError:
                        continue
                
                if ip_like_count > len(sample_values) * 0.5:  # At least 50% are IP addresses
                    ip_columns.append(col)
        
        return ip_columns
    
    def _check_ip_threats(self, ip_value: str) -> List[ThreatIndicator]:
        """Check if an IP address matches known threat indicators."""
        threats = []
        
        # Check cache first
        cache_key = f"ip_{ip_value}"
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if (datetime.now() - cache_entry['timestamp']).seconds < self.cache_ttl:
                return cache_entry['data']
        
        # Check against all loaded indicators
        for feed_id, indicators in self.indicators.items():
            for indicator in indicators:
                if indicator.ioc_type == "ip" and indicator.value == ip_value:
                    threats.append(indicator)
        
        # Cache the result
        self.cache[cache_key] = {
            'data': threats,
            'timestamp': datetime.now()
        }
        
        return threats
    
    def enrich_anomalies(self, anomalies: pd.DataFrame) -> pd.DataFrame:
        """Enrich anomaly data with security intelligence."""
        if anomalies.empty:
            return anomalies
        
        # Correlate with threat intelligence
        enriched = self.correlate_with_threats(anomalies)
        
        # Add risk scoring based on multiple factors
        enriched['security_risk_score'] = self._calculate_security_risk_score(enriched)
        
        # Add geolocation enrichment for IPs
        enriched = self._add_geolocation_enrichment(enriched)
        
        return enriched
    
    def _calculate_security_risk_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate comprehensive security risk score."""
        risk_scores = pd.Series(0.0, index=data.index)
        
        # Factor 1: Threat intelligence matches
        if 'threat_score' in data.columns:
            risk_scores += data['threat_score'] * 0.4
        
        # Factor 2: Anomaly score from ML model
        if 'anomaly_score' in data.columns:
            risk_scores += data['anomaly_score'] * 0.3
        
        # Factor 3: Port-based risk (uncommon ports = higher risk)
        port_columns = [col for col in data.columns if 'port' in col.lower()]
        if port_columns:
            for col in port_columns:
                port_risk = self._calculate_port_risk(data[col])
                risk_scores += port_risk * 0.1
        
        # Factor 4: Volume-based risk (unusual volumes = higher risk)
        volume_columns = [col for col in data.columns if any(term in col.lower() for term in ['bytes', 'size', 'length'])]
        if volume_columns:
            for col in volume_columns:
                volume_risk = self._calculate_volume_risk(data[col])
                risk_scores += volume_risk * 0.1
        
        # Factor 5: Temporal risk (unusual times = higher risk)
        time_columns = [col for col in data.columns if any(term in col.lower() for term in ['time', 'timestamp'])]
        if time_columns:
            time_risk = self._calculate_temporal_risk(data[time_columns[0]])
            risk_scores += time_risk * 0.1
        
        # Normalize to 0-1 range
        risk_scores = np.clip(risk_scores, 0.0, 1.0)
        
        return risk_scores
    
    def _calculate_port_risk(self, port_series: pd.Series) -> pd.Series:
        """Calculate risk score based on port usage."""
        common_ports = {20, 21, 22, 23, 25, 53, 67, 68, 69, 80, 110, 143, 443, 993, 995}
        
        def port_risk(port):
            try:
                port_num = int(port)
                if port_num in common_ports:
                    return 0.0
                elif port_num < 1024:
                    return 0.3  # System ports but not common
                elif port_num > 49152:
                    return 0.4  # Dynamic/private ports
                else:
                    return 0.2  # Registered ports
            except (ValueError, TypeError):
                return 0.0
        
        return port_series.apply(port_risk)
    
    def _calculate_volume_risk(self, volume_series: pd.Series) -> pd.Series:
        """Calculate risk score based on data volume."""
        if volume_series.empty:
            return pd.Series(0.0, index=volume_series.index)
        
        # Use statistical outlier detection
        q75 = volume_series.quantile(0.75)
        q25 = volume_series.quantile(0.25)
        iqr = q75 - q25
        upper_bound = q75 + 1.5 * iqr
        
        def volume_risk(volume):
            try:
                vol = float(volume)
                if vol > upper_bound:
                    return 0.3
                elif vol > q75:
                    return 0.1
                else:
                    return 0.0
            except (ValueError, TypeError):
                return 0.0
        
        return volume_series.apply(volume_risk)
    
    def _calculate_temporal_risk(self, time_series: pd.Series) -> pd.Series:
        """Calculate risk score based on timing patterns."""
        if time_series.empty:
            return pd.Series(0.0, index=time_series.index)
        
        def temporal_risk(timestamp):
            try:
                if pd.isna(timestamp):
                    return 0.0
                
                # Convert to datetime if needed
                if isinstance(timestamp, str):
                    dt = pd.to_datetime(timestamp)
                else:
                    dt = timestamp
                
                # Check if outside business hours (higher risk)
                hour = dt.hour
                if hour < 6 or hour > 22:  # Night hours
                    return 0.2
                elif hour < 8 or hour > 18:  # Extended hours
                    return 0.1
                else:
                    return 0.0
            except Exception:
                return 0.0
        
        return time_series.apply(temporal_risk)
    
    def _add_geolocation_enrichment(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add geolocation information for IP addresses."""
        # This would integrate with geolocation services like MaxMind
        # For now, we'll add placeholder columns
        
        data['geo_country'] = None
        data['geo_region'] = None
        data['geo_city'] = None
        data['geo_risk_score'] = 0.0
        
        # High-risk countries (this should be configurable)
        high_risk_countries = {'CN', 'RU', 'KP', 'IR'}
        
        # In a real implementation, you would use a geolocation service
        # For demo purposes, we'll assign random countries to some IPs
        ip_columns = self._find_ip_columns(data)
        
        for col in ip_columns:
            for idx, ip_value in data[col].items():
                if pd.notna(ip_value):
                    # Mock geolocation (replace with real service)
                    country = self._mock_geolocation(str(ip_value))
                    data.at[idx, 'geo_country'] = country
                    data.at[idx, 'geo_risk_score'] = 0.3 if country in high_risk_countries else 0.0
        
        return data
    
    def _mock_geolocation(self, ip: str) -> str:
        """Mock geolocation function (replace with real service)."""
        # Simple hash-based country assignment for demo
        hash_val = hash(ip) % 10
        countries = ['US', 'CN', 'RU', 'DE', 'GB', 'FR', 'JP', 'CA', 'AU', 'BR']
        return countries[hash_val]
    
    def get_threat_summary(self) -> Dict[str, Any]:
        """Get summary of threat intelligence status."""
        total_indicators = sum(len(indicators) for indicators in self.indicators.values())
        
        summary = {
            'total_indicators': total_indicators,
            'active_feeds': len([f for f in self.feeds.values() if f.get('enabled', False)]),
            'total_feeds': len(self.feeds),
            'cache_size': len(self.cache),
            'last_updated': datetime.now().isoformat(),
            'feeds_status': {}
        }
        
        for feed_id, feed_config in self.feeds.items():
            summary['feeds_status'][feed_id] = {
                'name': feed_config['name'],
                'enabled': feed_config.get('enabled', False),
                'indicators': len(self.indicators.get(feed_id, [])),
                'type': feed_config['type']
            }
        
        return summary
    
    def search_threats(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search threat indicators by query."""
        results = []
        query_lower = query.lower()
        
        for feed_id, indicators in self.indicators.items():
            for indicator in indicators:
                if (query_lower in indicator.value.lower() or 
                    query_lower in indicator.threat_type.lower() or
                    query_lower in indicator.description.lower()):
                    
                    results.append({
                        'feed': feed_id,
                        'indicator': asdict(indicator),
                        'relevance_score': self._calculate_relevance(query_lower, indicator)
                    })
                    
                    if len(results) >= limit:
                        break
            
            if len(results) >= limit:
                break
        
        # Sort by relevance
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return results[:limit]
    
    def _calculate_relevance(self, query: str, indicator: ThreatIndicator) -> float:
        """Calculate relevance score for search results."""
        score = 0.0
        
        # Exact matches get highest score
        if query == indicator.value.lower():
            score += 1.0
        elif query in indicator.value.lower():
            score += 0.8
        
        if query in indicator.threat_type.lower():
            score += 0.6
        
        if query in indicator.description.lower():
            score += 0.4
        
        # Boost recent indicators
        days_old = (datetime.now() - indicator.last_seen).days
        if days_old < 7:
            score += 0.2
        elif days_old < 30:
            score += 0.1
        
        # Boost high confidence indicators
        score += indicator.confidence * 0.3
        
        return score

# Utility functions for Streamlit integration
def display_threat_intelligence_dashboard(ti_manager: ThreatIntelligenceManager, st_container=None):
    """Display threat intelligence dashboard in Streamlit."""
    if st_container is None:
        st_container = st
    
    st_container.header("🛡️ Threat Intelligence Dashboard")
    
    # Get summary
    summary = ti_manager.get_threat_summary()
    
    # Key metrics
    col1, col2, col3, col4 = st_container.columns(4)
    
    with col1:
        st_container.metric("Total Indicators", summary['total_indicators'])
    with col2:
        st_container.metric("Active Feeds", f"{summary['active_feeds']}/{summary['total_feeds']}")
    with col3:
        st_container.metric("Cache Entries", summary['cache_size'])
    with col4:
        if st_container.button("🔄 Update Feeds"):
            with st_container.spinner("Updating threat feeds..."):
                results = ti_manager.update_threat_feeds()
                if results['failed_feeds']:
                    st_container.warning(f"Failed to update {len(results['failed_feeds'])} feeds")
                else:
                    st_container.success(f"Updated {len(results['updated_feeds'])} feeds successfully")
    
    # Feed status
    st_container.subheader("Feed Status")
    feed_data = []
    for feed_id, status in summary['feeds_status'].items():
        feed_data.append({
            'Feed': status['name'],
            'Status': '✅ Active' if status['enabled'] else '❌ Disabled',
            'Type': status['type'],
            'Indicators': status['indicators']
        })
    
    st_container.dataframe(pd.DataFrame(feed_data), use_container_width=True)
    
    # Search functionality
    st_container.subheader("🔍 Threat Search")
    search_query = st_container.text_input("Search threat indicators:")
    
    if search_query:
        results = ti_manager.search_threats(search_query, limit=20)
        
        if results:
            st_container.write(f"Found {len(results)} matching indicators:")
            
            search_data = []
            for result in results:
                indicator = result['indicator']
                search_data.append({
                    'Value': indicator['value'],
                    'Type': indicator['ioc_type'],
                    'Threat': indicator['threat_type'],
                    'Confidence': f"{indicator['confidence']:.2f}",
                    'Source': indicator['source'],
                    'Last Seen': indicator['last_seen'][:10]  # Date only
                })
            
            st_container.dataframe(pd.DataFrame(search_data), use_container_width=True)
        else:
            st_container.info("No matching threat indicators found.")

def create_security_risk_visualization(enriched_data: pd.DataFrame) -> Dict[str, Any]:
    """Create visualizations for security risk analysis."""
    import plotly.express as px
    import plotly.graph_objects as go
    
    figures = {}
    
    if 'security_risk_score' in enriched_data.columns:
        # Risk score distribution
        fig_risk = px.histogram(
            enriched_data,
            x='security_risk_score',
            nbins=20,
            title="Security Risk Score Distribution",
            labels={'security_risk_score': 'Risk Score', 'count': 'Number of Events'}
        )
        figures['risk_distribution'] = fig_risk
        
        # Risk vs Anomaly Score
        if 'anomaly_score' in enriched_data.columns:
            fig_scatter = px.scatter(
                enriched_data,
                x='anomaly_score',
                y='security_risk_score',
                color='threat_score' if 'threat_score' in enriched_data.columns else None,
                title="Security Risk vs Anomaly Score",
                labels={
                    'anomaly_score': 'ML Anomaly Score',
                    'security_risk_score': 'Security Risk Score'
                }
            )
            figures['risk_vs_anomaly'] = fig_scatter
    
    # Threat type distribution
    if 'threat_types' in enriched_data.columns:
        threat_counts = enriched_data['threat_types'].value_counts().head(10)
        if not threat_counts.empty:
            fig_threats = px.bar(
                x=threat_counts.values,
                y=threat_counts.index,
                orientation='h',
                title="Top Threat Types",
                labels={'x': 'Count', 'y': 'Threat Type'}
            )
            figures['threat_types'] = fig_threats
    
    return figures
