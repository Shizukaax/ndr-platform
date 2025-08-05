"""
About page for the Network Anomaly Detection Platform.
Provides information about the project, features, and credits.
"""

import streamlit as st

def show_about():
    """Display the About page with project information."""
    
    st.header("About This Project")
    
    st.markdown("""
    ## Network Anomaly Detection Platform
    
    This application is designed to help network security analysts detect and investigate 
    anomalous network traffic patterns using machine learning algorithms.
    
    ### Key Features
    
    - **Data Import**: Load and preprocess network traffic data from JSON files
    - **Anomaly Detection**: Apply various machine learning models to detect outliers
    - **Model Comparison**: Compare results from different detection algorithms
    - **Explainability**: Understand why specific traffic was flagged as anomalous
    - **MITRE ATT&CK Mapping**: Map detected anomalies to known attack techniques
    - **Reporting**: Generate comprehensive reports for further analysis
    
    ### Technology Stack
    
    - **Frontend**: Streamlit
    - **Data Processing**: Pandas, NumPy
    - **Machine Learning**: Scikit-learn, Isolation Forest, LOF, One-Class SVM
    - **Visualization**: Plotly, Seaborn, NetworkX, PyVis
    - **Explainability**: SHAP, LIME
    - **Deployment**: Docker
    
    ### Credits
    
    This project was developed as a modular, Docker-ready application for network security analysis.
    
    Special thanks to:
    - The Streamlit team for their excellent framework
    - The open-source ML community for the algorithms and tools
    - The MITRE ATT&CK framework for security knowledge base
    """)
    
    # Display version and GitHub link
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Version 1.0.0\n\n"
        "This is an open-source project."
    )