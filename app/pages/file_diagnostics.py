"""
File Diagnostics page for the Network Anomaly Detection Platform.
Provides UI for testing file saving and diagnosing persistence issues.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime

from utils.file_diagnostics import diagnose_directories, diagnose_pickle_save, force_save_model
from utils.data_saver import DataSaver

# Get logger
logger = logging.getLogger("streamlit_app")

def show_file_diagnostics():
    """Display the File Diagnostics page."""
    
    st.header("File System Diagnostics")
    st.write("""
    This page helps diagnose issues with file saving and directory permissions.
    Use the tools below to troubleshoot data persistence problems.
    """)
    
    # Run diagnostics
    if st.button("Run Directory Diagnostics", type="primary"):
        st.subheader("Directory Diagnostics Results")
        
        with st.spinner("Checking directories..."):
            results = diagnose_directories()
            
            # Show results in a table
            results_data = []
            for directory, info in results.items():
                results_data.append({
                    "Directory": directory,
                    "Path": info["path"],
                    "Exists": "✅" if info["exists"] else "❌",
                    "Writable": "✅" if info["writable"] else "❌",
                })
            
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
            
            # Show extra diagnostic info
            st.subheader("Current Working Directory")
            st.info(os.getcwd())
            
            st.subheader("Python Path")
            st.info(os.path.dirname(os.__file__))
    
    # Test data saving
    st.subheader("Test Data Saving")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create a simple DataFrame for testing
        if st.button("Test CSV Save"):
            with st.spinner("Saving test CSV..."):
                test_df = pd.DataFrame({
                    "timestamp": [datetime.now() for _ in range(5)],
                    "value": np.random.rand(5),
                    "category": ["test"] * 5
                })
                
                # Save using direct pandas method
                data_dir = Path(os.getcwd()) / "data"
                data_dir.mkdir(exist_ok=True)
                csv_path = data_dir / f"test_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                
                try:
                    test_df.to_csv(csv_path, index=False)
                    st.success(f"Successfully saved CSV to: {csv_path}")
                    
                    # Display file info
                    file_exists = os.path.exists(csv_path)
                    file_size = os.path.getsize(csv_path) if file_exists else 0
                    st.info(f"File exists: {file_exists}, Size: {file_size} bytes")
                except Exception as e:
                    st.error(f"Error saving CSV: {str(e)}")
                
                # Try with DataSaver
                saver = DataSaver()
                save_path = saver.save_data(test_df, "test_data_saver", format="csv")
                if save_path:
                    st.success(f"Successfully saved CSV using DataSaver to: {save_path}")
                else:
                    st.error("Failed to save CSV using DataSaver")
    
    with col2:
        # Test pickle save
        if st.button("Test Pickle Save"):
            with st.spinner("Testing pickle serialization..."):
                results = diagnose_pickle_save()
                
                if results["success"]:
                    st.success("Pickle serialization test passed!")
                else:
                    st.error(f"Pickle test failed: {results['message']}")
    
    # Test model saving
    st.subheader("Test Model Saving")
    
    # Create a dummy model
    model_type = st.selectbox("Select model type to test", ["Simple Dict", "Isolation Forest", "Random Array"])
    
    if st.button("Test Model Save"):
        with st.spinner("Creating and saving test model..."):
            # Create test model
            if model_type == "Simple Dict":
                test_model = {
                    "name": "test_model",
                    "created": datetime.now().isoformat(),
                    "metadata": {"type": "test", "version": "1.0"},
                    "params": {"n_estimators": 100, "contamination": 0.1}
                }
            elif model_type == "Isolation Forest":
                from sklearn.ensemble import IsolationForest
                test_model = IsolationForest(n_estimators=10, contamination=0.1, random_state=42)
                # Fit with small random data
                X = np.random.rand(10, 2)
                test_model.fit(X)
            else:
                # Random array
                test_model = np.random.rand(100, 5)
            
            # Test direct save
            try:
                models_dir = Path(os.getcwd()) / "models"
                models_dir.mkdir(exist_ok=True)
                model_path = models_dir / f"test_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                
                with open(model_path, "wb") as f:
                    pickle.dump(test_model, f)
                
                st.success(f"Successfully saved model to: {model_path}")
                
                # Display file info
                file_exists = os.path.exists(model_path)
                file_size = os.path.getsize(model_path) if file_exists else 0
                st.info(f"File exists: {file_exists}, Size: {file_size} bytes")
            except Exception as e:
                st.error(f"Error saving model: {str(e)}")
            
            # Test with force save
            try:
                results = force_save_model(test_model, f"test_model_force_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                
                if results["success"]:
                    st.success("Force save succeeded!")
                    st.json(results)
                else:
                    st.error("Force save failed")
                    st.json(results)
            except Exception as e:
                st.error(f"Error during force save: {str(e)}")
            
            # Try with DataSaver
            saver = DataSaver()
            results = saver.save_model(
                test_model, 
                f"test_model_saver_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
                metadata={"created": datetime.now().isoformat(), "type": model_type}
            )
            
            if results["success"]:
                st.success("DataSaver save succeeded!")
                st.json(results)
            else:
                st.error("DataSaver save failed")
                st.json(results)
    
    # Show a list of files in the data and models directories
    st.subheader("Files in Data Directory")
    data_dir = Path(os.getcwd()) / "data"
    data_files = list(data_dir.glob("*"))
    
    if data_files:
        file_info = []
        for file in data_files:
            file_info.append({
                "Name": file.name,
                "Size (KB)": f"{file.stat().st_size / 1024:.2f}",
                "Modified": datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            })
        
        st.dataframe(pd.DataFrame(file_info), use_container_width=True)
    else:
        st.info("No files found in data directory")
    
    st.subheader("Files in Models Directory")
    models_dir = Path(os.getcwd()) / "models"
    model_files = list(models_dir.glob("*"))
    
    if model_files:
        file_info = []
        for file in model_files:
            file_info.append({
                "Name": file.name,
                "Size (KB)": f"{file.stat().st_size / 1024:.2f}",
                "Modified": datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            })
        
        st.dataframe(pd.DataFrame(file_info), use_container_width=True)
    else:
        st.info("No files found in models directory")