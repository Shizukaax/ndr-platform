"""
Real-time Anomaly Tracking and Storage System
Stores anomaly history, trends, and provides enhanced analysis
"""

import json
import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import streamlit as st
import yaml

class AnomalyTracker:
    """Track and store real-time anomaly detection results"""
    
    def __init__(self, storage_dir: str = None):
        # Load configuration for storage directory
        if storage_dir is None:
            try:
                with open("config/config.yaml", 'r') as f:
                    config = yaml.safe_load(f)
                storage_dir = config.get("anomaly_storage", {}).get("history_dir", "data/anomaly_history")
            except:
                storage_dir = "data/anomaly_history"  # Fallback
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.anomaly_history_file = self.storage_dir / "anomaly_history.json"
        self.daily_summary_file = self.storage_dir / "daily_summary.json"
        self.baseline_file = self.storage_dir / "baseline_metrics.json"
        
        # Load existing data
        self.anomaly_history = self._load_anomaly_history()
        self.daily_summaries = self._load_daily_summaries()
        self.baseline_metrics = self._load_baseline_metrics()
        
    def _load_anomaly_history(self) -> List[Dict]:
        """Load anomaly history from file"""
        if self.anomaly_history_file.exists():
            try:
                with open(self.anomaly_history_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _load_daily_summaries(self) -> Dict:
        """Load daily summary data"""
        if self.daily_summary_file.exists():
            try:
                with open(self.daily_summary_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _load_baseline_metrics(self) -> Dict:
        """Load baseline metrics"""
        if self.baseline_file.exists():
            try:
                with open(self.baseline_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def record_anomaly_detection(self, 
                                anomalies: pd.DataFrame, 
                                model_type: str,
                                confidence_threshold: float,
                                source_file: str,
                                total_packets: int) -> Dict[str, Any]:
        """Record a new anomaly detection event"""
        
        timestamp = datetime.now()
        detection_id = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{model_type}"
        
        # Check for duplicates within the last 30 seconds
        if self._is_duplicate_detection(anomalies, model_type, timestamp, total_packets):
            return None  # Skip duplicate detection
        
        # Analyze anomalies for detailed insights
        anomaly_details = self._analyze_anomalies(anomalies) if anomalies is not None and len(anomalies) > 0 else {}
        
        # Create detection record
        detection_record = {
            "detection_id": detection_id,
            "timestamp": timestamp.isoformat(),
            "model_type": model_type,
            "confidence_threshold": confidence_threshold,
            "source_file": source_file,
            "total_packets": int(total_packets),  # Ensure native int
            "anomaly_count": int(len(anomalies)) if anomalies is not None else 0,
            "anomaly_rate": float((len(anomalies) / total_packets * 100)) if anomalies is not None and total_packets > 0 else 0.0,
            "details": anomaly_details,
            "severity": self._calculate_severity(anomalies, total_packets),
            "status": "new"  # new, acknowledged, investigating, resolved
        }
        
        # Add to history
        self.anomaly_history.append(detection_record)
        
        # Update daily summary
        self._update_daily_summary(detection_record)
        
        # Update baseline if needed
        self._update_baseline_metrics(total_packets, len(anomalies) if anomalies is not None else 0)
        
        # Save to disk
        self._save_all_data()
        
        return detection_record
    
    def _is_duplicate_detection(self, anomalies, model_type: str, timestamp: datetime, total_packets: int) -> bool:
        """Check if this detection is a duplicate of a recent one"""
        if not self.anomaly_history:
            return False
        
        # Check last few detections for duplicates within 30 seconds
        cutoff_time = timestamp - timedelta(seconds=30)
        anomaly_count = len(anomalies) if anomalies is not None else 0
        
        for record in self.anomaly_history[-5:]:  # Check last 5 records
            record_time = datetime.fromisoformat(record["timestamp"])
            
            # If within 30 seconds and same model type and similar anomaly count
            if (record_time >= cutoff_time and 
                record["model_type"] == model_type and
                abs(record["anomaly_count"] - anomaly_count) <= 2 and
                abs(record["total_packets"] - total_packets) <= 10):
                return True
        
        return False
    
    def _analyze_anomalies(self, anomalies: pd.DataFrame) -> Dict[str, Any]:
        """Analyze anomalies for detailed insights"""
        if anomalies is None or len(anomalies) == 0:
            return {}
        
        details = {
            "anomaly_types": {},
            "affected_ports": [],
            "affected_ips": [],
            "size_anomalies": [],
            "protocol_anomalies": []
        }
        
        # Analyze frame sizes
        if 'frame_len' in anomalies.columns:
            large_frames = anomalies[anomalies['frame_len'] > 5000]
            if len(large_frames) > 0:
                details["size_anomalies"] = {
                    "count": int(len(large_frames)),
                    "max_size": int(large_frames['frame_len'].max()),
                    "avg_size": int(large_frames['frame_len'].mean())
                }
        
        # Analyze ports
        if 'src_port' in anomalies.columns:
            unique_src_ports = anomalies['src_port'].unique()
            details["affected_ports"] = [int(port) for port in unique_src_ports[:10]]  # Top 10
        
        # Analyze external connections
        if 'external_conn' in anomalies.columns:
            external_anomalies = anomalies[anomalies['external_conn'] == 1]
            details["external_connections"] = int(len(external_anomalies))
        
        return details
    
    def _calculate_severity(self, anomalies, total_packets) -> str:
        """Calculate severity level based on anomaly characteristics"""
        if anomalies is None or len(anomalies) == 0:
            return "none"
        
        anomaly_rate = len(anomalies) / total_packets * 100 if total_packets > 0 else 0
        
        if anomaly_rate > 10:
            return "critical"
        elif anomaly_rate > 5:
            return "high"
        elif anomaly_rate > 1:
            return "medium"
        else:
            return "low"
    
    def _update_daily_summary(self, detection_record: Dict):
        """Update daily summary statistics"""
        date_key = detection_record["timestamp"][:10]  # YYYY-MM-DD
        
        if date_key not in self.daily_summaries:
            self.daily_summaries[date_key] = {
                "date": date_key,
                "total_detections": 0,
                "total_anomalies": 0,
                "severity_counts": {"low": 0, "medium": 0, "high": 0, "critical": 0},
                "model_usage": {}
            }
        
        summary = self.daily_summaries[date_key]
        summary["total_detections"] += 1
        summary["total_anomalies"] += detection_record["anomaly_count"]
        summary["severity_counts"][detection_record["severity"]] += 1
        
        model_type = detection_record["model_type"]
        if model_type not in summary["model_usage"]:
            summary["model_usage"][model_type] = 0
        summary["model_usage"][model_type] += 1
    
    def _update_baseline_metrics(self, total_packets: int, anomaly_count: int):
        """Update baseline metrics for normal behavior"""
        if "normal_packet_count" not in self.baseline_metrics:
            self.baseline_metrics["normal_packet_count"] = []
        if "normal_anomaly_rate" not in self.baseline_metrics:
            self.baseline_metrics["normal_anomaly_rate"] = []
        
        # Keep rolling window of last 50 measurements
        self.baseline_metrics["normal_packet_count"].append(total_packets)
        if len(self.baseline_metrics["normal_packet_count"]) > 50:
            self.baseline_metrics["normal_packet_count"] = self.baseline_metrics["normal_packet_count"][-50:]
        
        anomaly_rate = (anomaly_count / total_packets * 100) if total_packets > 0 else 0
        self.baseline_metrics["normal_anomaly_rate"].append(anomaly_rate)
        if len(self.baseline_metrics["normal_anomaly_rate"]) > 50:
            self.baseline_metrics["normal_anomaly_rate"] = self.baseline_metrics["normal_anomaly_rate"][-50:]
        
        # Calculate baseline statistics
        if len(self.baseline_metrics["normal_anomaly_rate"]) >= 10:
            rates = self.baseline_metrics["normal_anomaly_rate"]
            self.baseline_metrics["baseline_anomaly_rate"] = sum(rates) / len(rates)
            self.baseline_metrics["baseline_updated"] = datetime.now().isoformat()
    
    def _convert_numpy_types(self, obj):
        """Recursively convert numpy/pandas types to native Python types for JSON serialization"""
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    def _save_all_data(self):
        """Save all data to disk"""
        try:
            # Convert numpy types to JSON-serializable types
            history_data = self._convert_numpy_types(self.anomaly_history)
            summary_data = self._convert_numpy_types(self.daily_summaries)
            baseline_data = self._convert_numpy_types(self.baseline_metrics)
            
            with open(self.anomaly_history_file, 'w') as f:
                json.dump(history_data, f, indent=2, default=str)
            
            with open(self.daily_summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
            
            with open(self.baseline_file, 'w') as f:
                json.dump(baseline_data, f, indent=2, default=str)
        except Exception as e:
            st.error(f"Error saving anomaly data: {e}")
    
    def get_recent_anomalies(self, hours: int = 24) -> List[Dict]:
        """Get anomalies from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent = []
        for record in self.anomaly_history:
            record_time = datetime.fromisoformat(record["timestamp"])
            if record_time >= cutoff_time:
                recent.append(record)
        
        return sorted(recent, key=lambda x: x["timestamp"], reverse=True)
    
    def get_anomaly_trends(self, days: int = 7) -> Dict[str, Any]:
        """Get anomaly trends over the last N days"""
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        trends = {
            "daily_counts": {},
            "severity_trends": {"low": [], "medium": [], "high": [], "critical": []},
            "total_anomalies": 0,
            "avg_anomaly_rate": 0
        }
        
        for date, summary in self.daily_summaries.items():
            if date >= cutoff_date:
                trends["daily_counts"][date] = summary["total_anomalies"]
                trends["total_anomalies"] += summary["total_anomalies"]
                
                for severity, count in summary["severity_counts"].items():
                    trends["severity_trends"][severity].append({"date": date, "count": count})
        
        return trends
    
    def get_baseline_deviation(self, current_anomaly_rate: float) -> Dict[str, Any]:
        """Calculate how much current rate deviates from baseline"""
        baseline_rate = self.baseline_metrics.get("baseline_anomaly_rate", 0)
        
        if baseline_rate == 0:
            return {"deviation": 0, "status": "no_baseline", "message": "Building baseline..."}
        
        deviation_percent = ((current_anomaly_rate - baseline_rate) / baseline_rate) * 100
        
        if deviation_percent > 500:  # 5x normal
            status = "critical"
            message = f"Anomaly rate is {deviation_percent:.0f}% above baseline"
        elif deviation_percent > 200:  # 3x normal
            status = "high"
            message = f"Anomaly rate is {deviation_percent:.0f}% above baseline"
        elif deviation_percent > 50:  # 1.5x normal
            status = "medium"
            message = f"Anomaly rate is {deviation_percent:.0f}% above baseline"
        else:
            status = "normal"
            message = "Anomaly rate is within normal range"
        
        return {
            "deviation": deviation_percent,
            "status": status,
            "message": message,
            "baseline_rate": baseline_rate,
            "current_rate": current_anomaly_rate
        }
    
    def acknowledge_anomaly(self, detection_id: str, notes: str = ""):
        """Mark an anomaly as acknowledged"""
        for record in self.anomaly_history:
            if record["detection_id"] == detection_id:
                record["status"] = "acknowledged"
                record["acknowledgment_time"] = datetime.now().isoformat()
                record["acknowledgment_notes"] = notes
                self._save_all_data()
                return True
        return False
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Remove old anomaly records to prevent storage bloat"""
        cutoff_time = datetime.now() - timedelta(days=days_to_keep)
        
        self.anomaly_history = [
            record for record in self.anomaly_history
            if datetime.fromisoformat(record["timestamp"]) >= cutoff_time
        ]
        
        cutoff_date = cutoff_time.strftime("%Y-%m-%d")
        self.daily_summaries = {
            date: summary for date, summary in self.daily_summaries.items()
            if date >= cutoff_date
        }
        
        self._save_all_data()
