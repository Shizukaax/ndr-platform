"""
Diagnostic utility for troubleshooting file I/O issues.
Checks directory permissions, paths, and attempts test writes.
"""

import os
import sys
import logging
import tempfile
import json
import pickle
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("file_diagnostics")

def diagnose_directories():
    """Check all project directories for existence and write permissions."""
    app_root = Path(os.getcwd())
    logger.info(f"Application root directory: {app_root.absolute()}")
    
    # Check common directories
    directories = ["data", "models", "logs", "feedback", "cache", "config", "app/assets"]
    results = {}
    
    for directory in directories:
        dir_path = app_root / directory
        exists = dir_path.exists()
        
        # Try to create if doesn't exist
        if not exists:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                exists = True
                logger.info(f"Created directory: {dir_path}")
            except Exception as e:
                logger.error(f"Failed to create directory {dir_path}: {str(e)}")
        
        # Check write permissions by attempting to write a test file
        writable = False
        test_file = None
        if exists:
            try:
                test_file = dir_path / f"test_write_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tmp"
                with open(test_file, 'w') as f:
                    f.write("Test write")
                writable = True
                logger.info(f"Directory {dir_path} is writable")
            except Exception as e:
                logger.error(f"Directory {dir_path} is not writable: {str(e)}")
            finally:
                # Clean up test file
                if test_file and test_file.exists():
                    try:
                        os.remove(test_file)
                    except:
                        pass
        
        # Store results
        results[directory] = {
            "path": str(dir_path.absolute()),
            "exists": exists,
            "writable": writable,
            "abs_path": str(dir_path.resolve())
        }
    
    return results

def diagnose_pickle_save():
    """Test pickle serialization and saving."""
    try:
        # Create a simple test object
        test_obj = {
            "name": "test_model",
            "created": datetime.now().isoformat(),
            "metadata": {"type": "test", "version": "1.0"}
        }
        
        # Test serialization in memory
        serialized = pickle.dumps(test_obj)
        logger.info(f"Successfully serialized object in memory ({len(serialized)} bytes)")
        
        # Test writing to temp file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            pickle.dump(test_obj, temp_file)
            temp_path = temp_file.name
        
        logger.info(f"Successfully wrote object to temporary file: {temp_path}")
        
        # Test writing to data directory
        data_dir = Path(os.getcwd()) / "data"
        data_dir.mkdir(exist_ok=True)
        
        test_file = data_dir / f"test_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(test_file, "wb") as f:
            pickle.dump(test_obj, f)
        
        logger.info(f"Successfully wrote test model to {test_file}")
        
        # Check the saved file
        file_size = os.path.getsize(test_file)
        logger.info(f"Saved file size: {file_size} bytes")
        
        # Clean up test file
        os.remove(test_file)
        logger.info(f"Cleaned up test file")
        
        return {"success": True, "message": "Pickle serialization test passed"}
    except Exception as e:
        logger.exception(f"Pickle serialization test failed: {str(e)}")
        return {"success": False, "message": f"Pickle serialization test failed: {str(e)}"}

def force_save_model(model, model_name):
    """
    Force-save a model to disk using multiple methods.
    
    Args:
        model: The model object to save
        model_name: Name for the saved model file
    
    Returns:
        dict: Results of the save attempts
    """
    results = {"success": False, "paths_tried": []}
    
    try:
        # Ensure model directory exists
        model_dir = Path(os.getcwd()) / "models"
        model_dir.mkdir(exist_ok=True, parents=True)
        
        # Method 1: Direct pickle dump
        try:
            path1 = model_dir / f"{model_name}_method1.pkl"
            with open(path1, "wb") as f:
                pickle.dump(model, f)
            results["paths_tried"].append(str(path1))
            logger.info(f"Method 1 - Successfully saved model to {path1}")
            results["success"] = True
        except Exception as e:
            logger.error(f"Method 1 failed: {str(e)}")
        
        # Method 2: Save to temp file then move
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                pickle.dump(model, temp_file)
                temp_path = temp_file.name
            
            path2 = model_dir / f"{model_name}_method2.pkl"
            import shutil
            shutil.copy2(temp_path, path2)
            os.unlink(temp_path)
            results["paths_tried"].append(str(path2))
            logger.info(f"Method 2 - Successfully saved model to {path2}")
            results["success"] = True
        except Exception as e:
            logger.error(f"Method 2 failed: {str(e)}")
        
        # Method 3: Save with absolute path
        try:
            path3 = model_dir.absolute() / f"{model_name}_method3.pkl"
            with open(path3, "wb") as f:
                pickle.dump(model, f)
            results["paths_tried"].append(str(path3))
            logger.info(f"Method 3 - Successfully saved model to {path3}")
            results["success"] = True
        except Exception as e:
            logger.error(f"Method 3 failed: {str(e)}")
        
        # Also try data directory
        data_dir = Path(os.getcwd()) / "data"
        data_dir.mkdir(exist_ok=True, parents=True)
        
        try:
            path4 = data_dir / f"{model_name}_data_dir.pkl"
            with open(path4, "wb") as f:
                pickle.dump(model, f)
            results["paths_tried"].append(str(path4))
            logger.info(f"Data dir - Successfully saved model to {path4}")
            results["success"] = True
        except Exception as e:
            logger.error(f"Data dir save failed: {str(e)}")
        
        return results
    except Exception as e:
        logger.exception(f"Force save failed: {str(e)}")
        results["error"] = str(e)
        return results

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    logger.info("Running file diagnostics")
    
    # Run diagnostics
    dir_results = diagnose_directories()
    
    # Print results
    print("\n=== Directory Diagnostics ===")
    for directory, info in dir_results.items():
        status = "✅" if info["exists"] and info["writable"] else "❌"
        print(f"{status} {directory}: Exists: {info['exists']}, Writable: {info['writable']}")
        print(f"   Path: {info['path']}")
    
    # Test pickle serialization
    print("\n=== Pickle Serialization Test ===")
    pickle_results = diagnose_pickle_save()
    if pickle_results["success"]:
        print("✅ Pickle serialization test passed")
    else:
        print(f"❌ Pickle serialization test failed: {pickle_results['message']}")
    
    print("\nDiagnostics completed.")