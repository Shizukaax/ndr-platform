"""
Feedback manager for the Network Anomaly Detection Platform.
Stores and retrieves analyst feedback on detected anomalies.
"""

import os
import json
import pandas as pd
from datetime import datetime
import uuid
import logging

# Setup logger
logger = logging.getLogger(__name__)

class FeedbackManager:
    """Manages collection and storage of user feedback on anomalies."""
    
    def __init__(self, storage_dir=None):
        """
        Initialize the feedback manager with config-aware paths.
        
        Args:
            storage_dir (str, optional): Directory to store feedback data
        """
        # Load config to get proper paths
        if storage_dir is None:
            try:
                from core.config_loader import load_config
                config = load_config()
                storage_dir = config.get('feedback', {}).get('storage_dir', 'data/feedback')
            except Exception as e:
                logger.warning(f"Could not load config: {e}. Using default path.")
                storage_dir = 'data/feedback'
        
        self.storage_dir = storage_dir
        self.feedback_file = os.path.join(storage_dir, "feedback.json")
        self.session_feedback = {}
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize feedback store if it doesn't exist
        if not os.path.exists(self.feedback_file):
            self._save_feedback({})
    
    def add_feedback(self, anomaly_id, feedback_data):
        """
        Add feedback for an anomaly.
        
        Args:
            anomaly_id (str): Unique identifier for the anomaly
            feedback_data (dict): Feedback data
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load existing feedback
            all_feedback = self._load_feedback()
            
            # Create feedback entry
            feedback_entry = {
                "timestamp": datetime.now().isoformat(),
                "data": feedback_data
            }
            
            # Add to feedback store
            all_feedback[anomaly_id] = feedback_entry
            
            # Save updated feedback
            self._save_feedback(all_feedback)
            
            # Update session feedback
            self.session_feedback[anomaly_id] = feedback_entry
            
            return True
        except Exception as e:
            print(f"Error adding feedback: {str(e)}")
            return False
    
    def get_feedback(self, anomaly_id=None):
        """
        Get feedback for a specific anomaly or all feedback.
        
        Args:
            anomaly_id (str, optional): Unique identifier for the anomaly
        
        Returns:
            dict: Feedback data
        """
        # Load all feedback
        all_feedback = self._load_feedback()
        
        if anomaly_id is not None:
            # Return feedback for specific anomaly
            return all_feedback.get(anomaly_id)
        else:
            # Return all feedback
            return all_feedback
    
    def get_feedback_dataframe(self):
        """
        Get all feedback as a pandas DataFrame.
        Consolidates feedback from main file and dated backup files.
        
        Returns:
            pd.DataFrame: Feedback data
        """
        # Load feedback from main file
        all_feedback = self._load_feedback()
        
        # Convert main feedback to DataFrame
        records = []
        
        for anomaly_id, feedback in all_feedback.items():
            record = {
                "anomaly_id": anomaly_id,
                "feedback_time": feedback["timestamp"]
            }
            
            # Add feedback data
            for key, value in feedback["data"].items():
                record[key] = value
            
            records.append(record)
        
        # Load feedback from dated files
        import glob
        dated_files = glob.glob(os.path.join(self.storage_dir, "feedback_20*.json"))
        
        for dated_file in dated_files:
            try:
                with open(dated_file, 'r', encoding='utf-8') as f:
                    dated_data = json.load(f)
                    
                # Handle different formats
                if "feedback" in dated_data and isinstance(dated_data["feedback"], list):
                    for feedback_entry in dated_data["feedback"]:
                        record = {
                            "anomaly_id": feedback_entry.get("anomaly_id", f"dated_{len(records)}"),
                            "feedback_time": feedback_entry.get("timestamp", ""),
                            "classification": feedback_entry.get("classification", ""),
                            "priority": feedback_entry.get("priority", ""),
                            "technique": feedback_entry.get("technique", ""),
                            "action_taken": feedback_entry.get("action_taken", ""),
                            "comments": feedback_entry.get("comments", ""),
                            "analyst": feedback_entry.get("analyst", ""),
                            "anomaly_score": feedback_entry.get("anomaly_score", 0),
                            "risk_score": feedback_entry.get("risk_score", 0),
                            "risk_level": feedback_entry.get("risk_level", "")
                        }
                        records.append(record)
            except Exception as e:
                logger.warning(f"Error loading dated feedback file {dated_file}: {e}")
        
        # Create DataFrame
        if records:
            return pd.DataFrame(records)
        else:
            return pd.DataFrame()
    
    def update_feedback(self, anomaly_id, feedback_data):
        """
        Update existing feedback for an anomaly.
        
        Args:
            anomaly_id (str): Unique identifier for the anomaly
            feedback_data (dict): Updated feedback data
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load existing feedback
            all_feedback = self._load_feedback()
            
            # Check if feedback exists
            if anomaly_id not in all_feedback:
                return False
            
            # Update feedback data
            all_feedback[anomaly_id]["data"].update(feedback_data)
            all_feedback[anomaly_id]["updated"] = datetime.now().isoformat()
            
            # Save updated feedback
            self._save_feedback(all_feedback)
            
            # Update session feedback
            self.session_feedback[anomaly_id] = all_feedback[anomaly_id]
            
            return True
        except Exception as e:
            print(f"Error updating feedback: {str(e)}")
            return False
    
    def delete_feedback(self, anomaly_id):
        """
        Delete feedback for an anomaly.
        
        Args:
            anomaly_id (str): Unique identifier for the anomaly
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load existing feedback
            all_feedback = self._load_feedback()
            
            # Check if feedback exists
            if anomaly_id not in all_feedback:
                return False
            
            # Remove feedback
            del all_feedback[anomaly_id]
            
            # Save updated feedback
            self._save_feedback(all_feedback)
            
            # Update session feedback
            if anomaly_id in self.session_feedback:
                del self.session_feedback[anomaly_id]
            
            return True
        except Exception as e:
            print(f"Error deleting feedback: {str(e)}")
            return False
    
    def generate_anomaly_id(self, anomaly_data):
        """
        Generate a unique identifier for an anomaly.
        
        Args:
            anomaly_data (dict): Anomaly data
        
        Returns:
            str: Unique anomaly identifier
        """
        # Create a unique ID based on anomaly data
        # Use key attributes if available
        id_components = []
        
        for key in ["ip_src", "ip_dst", "timestamp", "_ws_col_Protocol"]:
            if key in anomaly_data:
                id_components.append(str(anomaly_data[key]))
        
        if id_components:
            # Create deterministic ID from components
            return "_".join(id_components)
        else:
            # Fallback to random UUID
            return str(uuid.uuid4())
    
    def _load_feedback(self):
        """
        Load feedback from storage.
        
        Returns:
            dict: Feedback data
        """
        try:
            if not os.path.exists(self.feedback_file):
                return {}
                
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                data = f.read().strip()
                if not data:
                    return {}
                return json.loads(data)
        except json.JSONDecodeError as e:
            print(f"JSON decode error in feedback file: {str(e)}")
            # Try to backup corrupted file
            try:
                backup_file = f"{self.feedback_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.rename(self.feedback_file, backup_file)
                print(f"Corrupted feedback file backed up to: {backup_file}")
            except:
                pass
            return {}
        except Exception as e:
            print(f"Error loading feedback: {str(e)}")
            return {}
    
    def _save_feedback(self, feedback_data):
        """
        Save feedback to storage.
        
        Args:
            feedback_data (dict): Feedback data to save
        """
        with open(self.feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, indent=2)
    
    def get_feedback_stats(self):
        """
        Get statistics about collected feedback.
        
        Returns:
            dict: Feedback statistics
        """
        # Load all feedback
        all_feedback = self._load_feedback()
        
        # Initialize statistics
        stats = {
            "total_feedback": len(all_feedback),
            "true_positive": 0,
            "false_positive": 0,
            "feedback_by_category": {},
            "feedback_by_date": {}
        }
        
        # Process feedback
        for anomaly_id, feedback in all_feedback.items():
            # Check true/false positive
            if "is_true_positive" in feedback["data"]:
                if feedback["data"]["is_true_positive"]:
                    stats["true_positive"] += 1
                else:
                    stats["false_positive"] += 1
            
            # Count by category
            if "category" in feedback["data"]:
                category = feedback["data"]["category"]
                stats["feedback_by_category"][category] = stats["feedback_by_category"].get(category, 0) + 1
            
            # Count by date
            date = feedback["timestamp"].split("T")[0]
            stats["feedback_by_date"][date] = stats["feedback_by_date"].get(date, 0) + 1
        
        return stats