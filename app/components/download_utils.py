"""
Download utilities for the Network Anomaly Detection Platform.
Provides functions for generating downloadable reports and data exports.
"""

import streamlit as st
import pandas as pd
import json
import csv
import io
import base64
import os
from datetime import datetime
import tempfile
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from core.config_loader import load_config

def get_csv_download_link(df, filename="data.csv", link_text="Download CSV"):
    """
    Generate a download link for a DataFrame as CSV.
    
    Args:
        df (pd.DataFrame): DataFrame to download
        filename (str): Name of the file to download
        link_text (str): Text to display for the download link
    
    Returns:
        str: HTML download link
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def get_json_download_link(df, filename="data.json", link_text="Download JSON"):
    """
    Generate a download link for a DataFrame as JSON.
    
    Args:
        df (pd.DataFrame): DataFrame to download
        filename (str): Name of the file to download
        link_text (str): Text to display for the download link
    
    Returns:
        str: HTML download link
    """
    json_str = df.to_json(orient='records', date_format='iso')
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def download_button(object_to_download, download_filename, button_text):
    """
    Create a download button for any object that can be serialized to JSON or string.
    
    Args:
        object_to_download: The object to be downloaded
        download_filename (str): Name of the file to download
        button_text (str): Text to display on the button
    
    Returns:
        bool: True if button was clicked, False otherwise
    """
    # Serialize object based on type
    if isinstance(object_to_download, pd.DataFrame):
        # If it's a DataFrame, convert to CSV
        object_to_download = object_to_download.to_csv(index=False)
        file_type = "text/csv"
    elif isinstance(object_to_download, dict) or isinstance(object_to_download, list):
        # If it's a dict or list, convert to JSON
        object_to_download = json.dumps(object_to_download, indent=2)
        file_type = "application/json"
    else:
        # Otherwise, convert to string
        object_to_download = str(object_to_download)
        file_type = "text/plain"
    
    # Create download button
    button = st.download_button(
        label=button_text,
        data=object_to_download,
        file_name=download_filename,
        mime=file_type
    )
    
    return button

def download_plotly_figure(fig, filename="plot.html"):
    """
    Create a download link for a Plotly figure as HTML.
    
    Args:
        fig (plotly.graph_objects.Figure): Plotly figure to download
        filename (str): Name of the file to download
    
    Returns:
        str: HTML download link
    """
    buffer = io.StringIO()
    fig.write_html(buffer)
    html_bytes = buffer.getvalue().encode()
    
    b64 = base64.b64encode(html_bytes).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}">Download Plot as HTML</a>'
    return href

def download_matplotlib_figure(fig, filename="plot.png"):
    """
    Create a download link for a Matplotlib figure as PNG.
    
    Args:
        fig (matplotlib.figure.Figure): Matplotlib figure to download
        filename (str): Name of the file to download
    
    Returns:
        str: HTML download link
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    img_bytes = buffer.getvalue()
    
    b64 = base64.b64encode(img_bytes).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">Download Plot as PNG</a>'
    return href

def export_to_csv(df, filepath=None):
    """
    Export a DataFrame to CSV.
    
    Args:
        df (pd.DataFrame): DataFrame to export
        filepath (str, optional): Path to save the CSV file. If None, saves to reports directory
        
    Returns:
        str: Path to the saved file
    """
    if filepath is None:
        # Get reports directory from config
        config = load_config()
        reports_dir = config.get('system', {}).get('reports_dir', 'data/reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"{reports_dir}/export_{timestamp}.csv"
    
    # Export to CSV
    df.to_csv(filepath, index=False)
    
    return filepath

def export_to_excel(df, filepath=None):
    """
    Export a DataFrame to Excel.
    
    Args:
        df (pd.DataFrame): DataFrame to export
        filepath (str, optional): Path to save the Excel file. If None, saves to reports directory
        
    Returns:
        str: Path to the saved file
    """
    if filepath is None:
        # Get reports directory from config
        config = load_config()
        reports_dir = config.get('system', {}).get('reports_dir', 'data/reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"{reports_dir}/export_{timestamp}.xlsx"
    
    # Export to Excel
    df.to_excel(filepath, index=False)
    
    return filepath

def create_html_report(df, title, description, plots=None):
    """
    Create an HTML report with data and plots.
    
    Args:
        df (pd.DataFrame): DataFrame to include in the report
        title (str): Report title
        description (str): Report description
        plots (list, optional): List of Plotly figures to include
        
    Returns:
        str: HTML content of the report
    """
    # Start HTML content
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #3498db; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .footer {{ margin-top: 30px; color: #7f8c8d; font-size: 0.8em; }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <p>{description}</p>
        <p><em>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</em></p>
    """
    
    # Add plots if provided
    if plots:
        html += "<h2>Visualizations</h2>"
        for i, plot in enumerate(plots):
            # Create a temporary file for the plot
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
                plot_path = tmp.name
                plot.write_html(plot_path)
            
            # Read the plot HTML
            with open(plot_path, "r", encoding='utf-8') as f:
                plot_html = f.read()
            
            # Extract the plot content (remove html, head, body tags)
            plot_content = plot_html.split("<body>")[1].split("</body>")[0]
            
            # Add to report
            html += f"<div class='plot'>{plot_content}</div>"
            
            # Clean up temp file
            os.unlink(plot_path)
    
    # Add data table
    html += "<h2>Data</h2>"
    html += df.to_html(index=False, classes="dataframe")
    
    # Add footer and close tags
    html += """
        <div class="footer">
            <p>Generated by Network Anomaly Detection Platform</p>
        </div>
    </body>
    </html>
    """
    
    return html

def save_html_report(html_content, filepath=None):
    """
    Save HTML report to a file.
    
    Args:
        html_content (str): HTML content to save
        filepath (str, optional): Path to save the HTML file. If None, saves to reports directory
        
    Returns:
        str: Path to the saved file
    """
    if filepath is None:
        # Get reports directory from config
        config = load_config()
        reports_dir = config.get('system', {}).get('reports_dir', 'data/reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"{reports_dir}/report_{timestamp}.html"
    
    # Save HTML content
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return filepath