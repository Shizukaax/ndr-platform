"""
Report generation utilities for the Network Anomaly Detection Platform.
Provides functions for creating and formatting reports.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from datetime import datetime
import json
import tempfile
import base64
import matplotlib.pyplot as plt
from core.config_loader import load_config
from io import BytesIO
import jinja2
import uuid

from app.components.download_utils import create_html_report, save_html_report

def generate_anomaly_report(df, anomalies, scores, threshold, model_name, features_used, plots=None):
    """
    Generate a comprehensive anomaly detection report.
    
    Args:
        df (pd.DataFrame): Original data
        anomalies (pd.DataFrame): Detected anomalies
        scores (np.ndarray): Anomaly scores
        threshold (float): Anomaly threshold
        model_name (str): Name of the model used
        features_used (list): Features used for detection
        plots (list): List of plotly figures to include
    
    Returns:
        str: HTML content of the report
    """
    # Create report title and description
    title = f"Network Anomaly Detection Report - {model_name}"
    
    description = f"""
    This report contains the results of network anomaly detection using {model_name}.
    
    **Detection Summary:**
    - Total records analyzed: {len(df)}
    - Anomalies detected: {len(anomalies)} ({len(anomalies)/len(df)*100:.2f}%)
    - Anomaly score threshold: {threshold:.4f}
    - Features used: {', '.join(features_used)}
    - Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    """
    
    # Create HTML report
    html_content = create_html_report(
        anomalies,
        title,
        description,
        plots
    )
    
    return html_content

def generate_comparison_report(df, model_results, features_used, plots=None):
    """
    Generate a report comparing multiple anomaly detection models.
    
    Args:
        df (pd.DataFrame): Original data
        model_results (dict): Dictionary of model results
        features_used (list): Features used for detection
        plots (list): List of plotly figures to include
    
    Returns:
        str: HTML content of the report
    """
    # Create report title and description
    title = "Model Comparison Report"
    
    # Create summary of models
    model_summary = []
    for model_name, results in model_results.items():
        anomaly_count = (results["scores"] > results["threshold"]).sum()
        model_summary.append(
            f"- **{model_name}**: {anomaly_count} anomalies "
            f"({anomaly_count/len(df)*100:.2f}%), "
            f"threshold: {results['threshold']:.4f}, "
            f"training time: {results['training_time']:.4f}s"
        )
    
    description = f"""
    This report compares the results of multiple anomaly detection models.
    
    **Analysis Summary:**
    - Total records analyzed: {len(df)}
    - Features used: {', '.join(features_used)}
    - Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    **Model Summary:**
    {"".join(model_summary)}
    """
    
    # Create a dataframe with anomaly flags for each model
    comparison_df = df.copy()
    
    for model_name, results in model_results.items():
        comparison_df[f"{model_name}_score"] = results["scores"]
        comparison_df[f"{model_name}_anomaly"] = results["scores"] > results["threshold"]
    
    # Count models flagging each record as anomalous
    anomaly_cols = [f"{model_name}_anomaly" for model_name in model_results.keys()]
    comparison_df["models_flagged"] = comparison_df[anomaly_cols].sum(axis=1)
    
    # Filter to show only records flagged by at least one model
    anomalies_df = comparison_df[comparison_df["models_flagged"] > 0]
    
    # Create HTML report
    html_content = create_html_report(
        anomalies_df,
        title,
        description,
        plots
    )
    
    return html_content

def generate_pdf_report(html_content, output_path=None):
    """
    Convert HTML report to PDF.
    
    Args:
        html_content (str): HTML content to convert
        output_path (str, optional): Path to save the PDF file
    
    Returns:
        str or bytes: Path to the saved PDF file or PDF bytes
    """
    try:
        # Check if wkhtmltopdf is installed
        import pdfkit
        
        # Set options for PDF generation
        options = {
            'page-size': 'Letter',
            'margin-top': '0.75in',
            'margin-right': '0.75in',
            'margin-bottom': '0.75in',
            'margin-left': '0.75in',
            'encoding': "UTF-8",
            'no-outline': None,
            'enable-local-file-access': None
        }
        
        if output_path is None:
            # Get reports directory from config
            config = load_config()
            reports_dir = config.get('system', {}).get('reports_dir', 'data/reports')
            os.makedirs(reports_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{reports_dir}/report_{timestamp}.pdf"
        
        # Generate PDF
        pdfkit.from_string(html_content, output_path, options=options)
        
        return output_path
    
    except ImportError:
        st.error("PDF generation requires pdfkit and wkhtmltopdf. Please install them to enable PDF reports.")
        return None
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None

def create_report_ui(df, title="Network Anomaly Report", key_prefix="report"):
    """
    Create a UI for customizing and generating reports.
    
    Args:
        df (pd.DataFrame): DataFrame to include in the report
        title (str): Default report title
        key_prefix (str): Prefix for session state keys
    
    Returns:
        tuple: (generate_report, report_config)
    """
    st.subheader("Report Configuration")
    
    # Create a form for report configuration
    with st.form(f"{key_prefix}_form"):
        # Report title and description
        report_title = st.text_input("Report Title", value=title)
        
        report_description = st.text_area(
            "Report Description",
            value="This report contains the results of network anomaly detection."
        )
        
        # Report content options
        col1, col2 = st.columns(2)
        
        with col1:
            include_charts = st.checkbox("Include Charts", value=True)
            include_data = st.checkbox("Include Data Table", value=True)
            include_metadata = st.checkbox("Include Metadata", value=True)
        
        with col2:
            max_records = st.number_input(
                "Maximum Records to Include",
                min_value=10,
                max_value=10000,
                value=1000
            )
            
            report_format = st.selectbox(
                "Report Format",
                options=["HTML", "PDF", "CSV", "Excel", "JSON"],
                index=0
            )
        
        # Advanced options
        with st.expander("Advanced Options"):
            # Select columns to include
            selected_columns = st.multiselect(
                "Select Columns to Include",
                options=df.columns.tolist(),
                default=df.columns.tolist()
            )
            
            # Color theme
            color_theme = st.selectbox(
                "Color Theme",
                options=["Default", "Light", "Dark", "Colorful"],
                index=0
            )
            
            # Page orientation (for PDF)
            page_orientation = st.radio(
                "Page Orientation (PDF only)",
                options=["Portrait", "Landscape"],
                index=0
            )
        
        # Submit button
        generate_report = st.form_submit_button("Generate Report")
    
    # Create config dictionary
    report_config = {
        "title": report_title,
        "description": report_description,
        "include_charts": include_charts,
        "include_data": include_data,
        "include_metadata": include_metadata,
        "max_records": max_records,
        "format": report_format,
        "columns": selected_columns,
        "color_theme": color_theme,
        "orientation": page_orientation
    }
    
    return generate_report, report_config

def generate_executive_summary(df, anomalies, model_name, threshold):
    """
    Generate an executive summary of anomaly detection results.
    
    Args:
        df (pd.DataFrame): Original data
        anomalies (pd.DataFrame): Detected anomalies
        model_name (str): Name of the model used
        threshold (float): Anomaly threshold
    
    Returns:
        str: Markdown content for executive summary
    """
    # Calculate basic statistics
    total_records = len(df)
    anomaly_count = len(anomalies)
    anomaly_percent = (anomaly_count / total_records) * 100 if total_records > 0 else 0
    
    # Get time range if available
    time_range = ""
    if 'timestamp' in df.columns:
        start_time = df['timestamp'].min()
        end_time = df['timestamp'].max()
        time_range = f"Time range: {start_time} to {end_time}\n"
    
    # Protocol statistics if available
    protocol_stats = ""
    if '_ws_col_Protocol' in df.columns:
        protocols = df['_ws_col_Protocol'].value_counts().head(5)
        protocol_stats = "Top Protocols:\n" + "\n".join([f"- {p}: {c} records" for p, c in protocols.items()])
    
    # Anomaly statistics
    anomaly_protocols = ""
    if '_ws_col_Protocol' in anomalies.columns:
        anomaly_proto = anomalies['_ws_col_Protocol'].value_counts().head(5)
        anomaly_protocols = "Top Anomalous Protocols:\n" + "\n".join([f"- {p}: {c} anomalies" for p, c in anomaly_proto.items()])
    
    # Create executive summary
    summary = f"""
    # Executive Summary - Network Anomaly Detection
    
    ## Overview
    
    - **Analysis Date:** {datetime.now().strftime("%Y-%m-%d")}
    - **Model Used:** {model_name}
    - **Threshold:** {threshold:.4f}
    - **Total Records Analyzed:** {total_records:,}
    - **Anomalies Detected:** {anomaly_count:,} ({anomaly_percent:.2f}%)
    {time_range}
    
    ## Key Findings
    
    {protocol_stats}
    
    {anomaly_protocols}
    
    ## Recommendations
    
    1. Investigate the top anomalous protocols for potential security issues
    2. Review source/destination pairs with high anomaly scores
    3. Consider tuning the anomaly threshold based on these results
    4. Schedule regular anomaly detection to establish baseline patterns
    
    """
    
    return summary

def create_report_download_links(report_content, base_filename="report", formats=None):
    """
    Create download links for a report in multiple formats.
    
    Args:
        report_content (str or pd.DataFrame): Report content or data
        base_filename (str): Base filename without extension
        formats (list): List of formats to provide
    
    Returns:
        dict: Dictionary of download links
    """
    if formats is None:
        formats = ["html", "csv", "json"]
    
    download_links = {}
    
    for fmt in formats:
        if fmt == "html" and isinstance(report_content, str):
            # HTML content
            b64 = base64.b64encode(report_content.encode()).decode()
            href = f'<a href="data:text/html;base64,{b64}" download="{base_filename}.html">Download HTML</a>'
            download_links["html"] = href
            
        elif fmt == "pdf" and isinstance(report_content, str):
            try:
                # Try to generate PDF
                pdf_path = generate_pdf_report(report_content)
                if pdf_path:
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    b64 = base64.b64encode(pdf_bytes).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="{base_filename}.pdf">Download PDF</a>'
                    download_links["pdf"] = href
            except:
                pass
                
        elif isinstance(report_content, pd.DataFrame):
            # DataFrame formats
            if fmt == "csv":
                csv = report_content.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:text/csv;base64,{b64}" download="{base_filename}.csv">Download CSV</a>'
                download_links["csv"] = href
                
            elif fmt == "excel":
                # Create Excel file in memory
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    report_content.to_excel(writer, index=False, sheet_name='Report')
                excel_data = output.getvalue()
                b64 = base64.b64encode(excel_data).decode()
                href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{base_filename}.xlsx">Download Excel</a>'
                download_links["excel"] = href
                
            elif fmt == "json":
                json_str = report_content.to_json(orient='records', date_format='iso')
                b64 = base64.b64encode(json_str.encode()).decode()
                href = f'<a href="data:application/json;base64,{b64}" download="{base_filename}.json">Download JSON</a>'
                download_links["json"] = href
    
    return download_links

def render_html_template(template_str, context):
    """
    Render an HTML template with the given context.
    
    Args:
        template_str (str): Jinja2 template string
        context (dict): Context variables for the template
    
    Returns:
        str: Rendered HTML
    """
    # Create Jinja2 environment
    template = jinja2.Template(template_str)
    
    # Render template
    return template.render(**context)

# Default HTML template for reports
DEFAULT_REPORT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
            text-align: left;
        }
        th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .container {
            margin-bottom: 30px;
        }
        .metadata {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .footer {
            margin-top: 30px;
            text-align: center;
            font-size: 0.8em;
            color: #7f8c8d;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ title }}</h1>
        <div class="metadata">
            <p>{{ description }}</p>
            <p><em>Generated on {{ timestamp }}</em></p>
        </div>
        
        {% if plots %}
        <h2>Visualizations</h2>
        {% for plot in plots %}
        <div class="plot">{{ plot }}</div>
        {% endfor %}
        {% endif %}
        
        {% if include_data %}
        <h2>Data</h2>
        {{ table_html }}
        {% endif %}
    </div>
    
    <div class="footer">
        <p>Generated by Network Anomaly Detection Platform</p>
    </div>
</body>
</html>
"""