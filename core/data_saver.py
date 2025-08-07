"""
Robust data saving utility for ensuring persistence across different environments.
Provides multiple fallback methods for saving data and models.
"""

import os
import sys
import pickle
import json
import logging
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("data_saver")

class DataSaver:
    """Utility class for robustly saving data and models."""
    
    def __init__(self, base_dir=None):
        """
        Initialize the data saver.
        
        Args:
            base_dir (str): Base directory for saving data. Defaults to current working directory.
        """
        self.base_dir = Path(base_dir if base_dir else os.getcwd())
        
        # Ensure required directories exist
        self.directories = {
            "data": self.base_dir / "data",
            "models": self.base_dir / "models",
            "cache": self.base_dir / "cache",
            "feedback": self.base_dir / "feedback"
        }
        
        for name, path in self.directories.items():
            path.mkdir(exist_ok=True, parents=True)
            logger.info(f"Ensured {name} directory exists: {path}")
    
    def save_model(self, model, name, metadata=None):
        """
        Save a model with multiple fallback methods.
        
        Args:
            model: The model object to save
            name (str): Name for the saved model
            metadata (dict, optional): Additional metadata to save with the model
        
        Returns:
            dict: Results of the save operation including all paths tried
        """
        results = {"success": False, "paths_tried": [], "final_path": None}
        
        # Add timestamp to name if not already present
        if not any(c.isdigit() for c in name):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"{name}_{timestamp}"
        
        # Make sure name has .pkl extension
        if not name.endswith(".pkl"):
            name = f"{name}.pkl"
        
        # Try multiple save methods
        methods = [
            self._save_direct,
            self._save_temp_copy,
            self._save_absolute_path,
            self._save_alternative_dir
        ]
        
        # Try each method until one succeeds
        for i, method in enumerate(methods):
            try:
                path = method(model, name, i)
                if path:
                    results["paths_tried"].append(str(path))
                    results["success"] = True
                    results["final_path"] = str(path)
                    logger.info(f"Successfully saved model to {path} (method {i+1})")
                    
                    # Save metadata if provided
                    if metadata:
                        meta_path = Path(str(path).replace(".pkl", "_metadata.json"))
                        with open(meta_path, "w") as f:
                            json.dump(metadata, f, indent=2)
                        logger.info(f"Saved metadata to {meta_path}")
                    
                    break
            except Exception as e:
                logger.error(f"Save method {i+1} failed: {str(e)}")
        
        return results
    
    def _save_direct(self, model, name, attempt_num):
        """Direct save to models directory."""
        path = self.directories["models"] / name
        with open(path, "wb") as f:
            pickle.dump(model, f)
        return path
    
    def _save_temp_copy(self, model, name, attempt_num):
        """Save to temp file then copy to final location."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            pickle.dump(model, temp_file)
            temp_path = temp_file.name
        
        # Copy from temp to final location
        path = self.directories["models"] / f"{name.replace('.pkl', '')}_attempt{attempt_num+1}.pkl"
        shutil.copy2(temp_path, path)
        os.unlink(temp_path)
        return path
    
    def _save_absolute_path(self, model, name, attempt_num):
        """Save using absolute path."""
        path = self.directories["models"].absolute() / f"{name.replace('.pkl', '')}_absolute.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        return path
    
    def _save_alternative_dir(self, model, name, attempt_num):
        """Save to alternative directory (data)."""
        path = self.directories["data"] / f"{name.replace('.pkl', '')}_data_dir.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        return path
    
    def save_data(self, data, name, format="json"):
        """
        Save data in specified format.
        
        Args:
            data: The data to save
            name (str): Name for the saved file
            format (str): Format to save as ('json', 'csv', 'pickle')
        
        Returns:
            str: Path to the saved file or None if failed
        """
        # Add timestamp to name if not already present
        if not any(c.isdigit() for c in name):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"{name}_{timestamp}"
        
        # Add extension if not present
        if format == "json" and not name.endswith(".json"):
            name = f"{name}.json"
        elif format == "csv" and not name.endswith(".csv"):
            name = f"{name}.csv"
        elif format == "pickle" and not name.endswith(".pkl"):
            name = f"{name}.pkl"
        
        # Choose save directory
        save_dir = self.directories["data"]
        save_path = save_dir / name
        
        try:
            if format == "json":
                with open(save_path, "w") as f:
                    if hasattr(data, "to_json"):
                        # Handle pandas DataFrame
                        json_str = data.to_json(orient="records", date_format="iso")
                        f.write(json_str)
                    else:
                        # Handle regular dict/list
                        json.dump(data, f, indent=2, default=str)
            
            elif format == "csv":
                if hasattr(data, "to_csv"):
                    # Handle pandas DataFrame
                    data.to_csv(save_path, index=False)
                else:
                    # Handle other data types
                    import csv
                    with open(save_path, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(data)
            
            elif format == "pickle":
                with open(save_path, "wb") as f:
                    pickle.dump(data, f)
            
            logger.info(f"Successfully saved data to {save_path}")
            return str(save_path)
        
        except Exception as e:
            logger.error(f"Error saving data to {save_path}: {str(e)}")
            return None
    
    def load_model(self, path):
        """
        Load a model from disk.
        
        Args:
            path (str): Path to the model file
        
        Returns:
            The loaded model or None if failed
        """
        try:
            path = Path(path)
            if not path.is_absolute():
                # Try models directory first
                model_path = self.directories["models"] / path
                if not model_path.exists():
                    # Try data directory
                    model_path = self.directories["data"] / path
            else:
                model_path = path
            
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return None
            
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            
            logger.info(f"Successfully loaded model from {model_path}")
            return model
        
        except Exception as e:
            logger.error(f"Error loading model from {path}: {str(e)}")
            return None