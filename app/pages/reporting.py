"""
Reporting page for the Network Anomaly Detection Platform.
Provides UI for generating and exporting reports of analysis results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime
import json
import base64
import io

from app.components.report_generator import (
    generate_anomaly_report, generate_comparison_report, 
    generate_pdf_report, create_report_ui, 
    generate_executive_summary, create_report_download_links
)
from app.components.visualization import (
    plot_anomaly_scores, plot_anomaly_scatter, plot_anomaly_timeline,
    plot_feature_importance, plot_network_graph, plot_protocol_pie
)


def show_reporting():
    """Display the Reporting page."""
    
    st.header("Analysis Reports")
    
    # Initialize session-state keys if missing
    if "combined_data" not in st.session_state:
        st.session_state.combined_data = None
    if "anomalies" not in st.session_state:
        st.session_state.anomalies = pd.DataFrame()
    if "model_results" not in st.session_state:
        st.session_state.model_results = {}
    if "model_comparison_results" not in st.session_state:
        st.session_state.model_comparison_results = {}
    if "model_comparison_features" not in st.session_state:
        st.session_state.model_comparison_features = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None

    # Check if data and results are available
    if st.session_state.combined_data is None:
        st.info("No data loaded. Please go to the Data Upload page to select JSON files.")
        return
    
    # Get data
    df = st.session_state.combined_data
    
    # Create tabs for different report types
    tab1, tab2, tab3, tab4 = st.tabs([
        "Anomaly Report", "Model Comparison Report", "Executive Summary", "Custom Report"
    ])
    
    # Anomaly Report tab
    with tab1:
        st.subheader("Generate Anomaly Detection Report")
        
        # Check if anomalies are detected
        if st.session_state.anomalies.empty:
            st.warning("No anomalies detected. Please run anomaly detection first.")
        else:
            anomalies = st.session_state.anomalies
            
            # Build model_results from session state data
            model_results = {
                'scores': getattr(st.session_state, 'anomaly_scores', []),
                'threshold': getattr(st.session_state, 'anomaly_threshold', 0.5),
                'features': getattr(st.session_state, 'anomaly_features', []),
                'algorithm': getattr(st.session_state, 'selected_model', 'Unknown'),
                'model': getattr(st.session_state, 'anomaly_model', None)
            }
            
            # Also update session_state.model_results for compatibility
            st.session_state.model_results = model_results
            
            # Show report configuration
            col1, col2 = st.columns(2)
            
            with col1:
                report_format = st.selectbox(
                    "Report format",
                    options=["HTML", "PDF", "CSV", "Excel", "JSON"],
                    index=0,
                    key="anomaly_report_format"
                )
                include_charts = st.checkbox("Include visualizations", value=True, key="anomaly_include_charts")
                include_raw_data = st.checkbox("Include raw data", value=True, key="anomaly_include_raw")
            
            with col2:
                max_records = st.number_input(
                    "Maximum anomalies to include",
                    min_value=1,
                    max_value=len(anomalies),
                    value=min(100, len(anomalies)),
                    key="anomaly_max_records"
                )
                sort_by = st.selectbox(
                    "Sort anomalies by",
                    options=["anomaly_score", "timestamp"] + 
                            [col for col in anomalies.columns if col not in ["anomaly_score", "is_anomaly"]],
                    index=0,
                    key="anomaly_sort_by"
                )
                sort_ascending = st.checkbox("Sort ascending", value=False, key="anomaly_sort_ascending")
            
            if st.button("Generate Anomaly Report", key="generate_anomaly_report"):
                with st.spinner("Generating report..."):
                    try:
                        # Check if model_results has required data
                        if not model_results or 'scores' not in model_results:
                            st.error("Model results are incomplete. Please run anomaly detection first.")
                            return
                        
                        # Get threshold from session state or calculate from model results
                        threshold = model_results.get('threshold', st.session_state.get('anomaly_threshold', 0.5))
                        if threshold is None or threshold == 0:
                            # Calculate a reasonable threshold if none exists
                            scores = model_results.get('scores', [])
                            if len(scores) > 0:
                                threshold = np.percentile(scores, 90)  # 90th percentile as default
                            else:
                                threshold = 0.5  # Fallback default
                        
                        sorted_anomalies = anomalies.sort_values(
                            sort_by, 
                            ascending=sort_ascending
                        ).head(max_records)
                        
                        plots = []
                        if include_charts:
                            score_plot = plot_anomaly_scores(
                                df, 
                                model_results['scores'], 
                                threshold,
                                title="Anomaly Score Distribution"
                            )
                            plots.append(score_plot)
                            
                            if len(model_results.get('features', [])) > 0:
                                feature_plot = plot_feature_importance(
                                    df, 
                                    model_results['scores'], 
                                    threshold
                                )
                                if feature_plot:
                                    plots.append(feature_plot)
                        
                        if report_format in ["HTML", "PDF"]:
                            html_report = generate_anomaly_report(
                                df,
                                sorted_anomalies,
                                model_results['scores'],
                                threshold,
                                st.session_state.selected_model,
                                model_results.get('features', []),
                                plots
                            )
                            if report_format == "PDF":
                                try:
                                    pdf_path = generate_pdf_report(html_report)
                                    if pdf_path:
                                        with open(pdf_path, "rb") as f:
                                            pdf_bytes = f.read()
                                        st.download_button(
                                            "Download PDF Report",
                                            data=pdf_bytes,
                                            file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                            mime="application/pdf"
                                        )
                                    else:
                                        st.error("Failed to generate PDF report.")
                                except Exception as e:
                                    st.error(f"Error generating PDF: {str(e)}")
                            else:
                                st.markdown("### HTML Report Preview")
                                st.components.v1.html(html_report, height=500, scrolling=True)
                                st.download_button(
                                    "Download HTML Report",
                                    data=html_report,
                                    file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                    mime="text/html"
                                )
                        elif report_format == "CSV":
                            csv_data = sorted_anomalies.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "Download CSV Report",
                                data=csv_data,
                                file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        elif report_format == "Excel":
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                sorted_anomalies.to_excel(writer, sheet_name='Anomalies', index=False)
                                metadata = pd.DataFrame([
                                    {"Key": "Report Date", "Value": datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                                    {"Key": "Total Records", "Value": len(df)},
                                    {"Key": "Anomalies Detected", "Value": len(anomalies)},
                                    {"Key": "Model", "Value": st.session_state.selected_model},
                                    {"Key": "Threshold", "Value": f"{threshold:.3f}"}
                                ])
                                metadata.to_excel(writer, sheet_name='Metadata', index=False)
                            excel_data = output.getvalue()
                            st.download_button(
                                "Download Excel Report",
                                data=excel_data,
                                file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        elif report_format == "JSON":
                            csv_data = sorted_anomalies.to_csv(index=False).encode('utf-8')
                            json_data = sorted_anomalies.to_json(orient='records', date_format='iso')
                            json_obj = {
                                "metadata": {
                                    "report_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    "total_records": len(df),
                                    "anomalies_detected": len(anomalies),
                                    "model": st.session_state.selected_model,
                                    "threshold": threshold
                                },
                                "anomalies": sorted_anomalies.to_dict(orient='records')
                            }
                            st.download_button(
                                "Download JSON Report",
                                data=json.dumps(json_obj, indent=2),
                                file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                        
                        st.success("Report generated successfully!")
                        
                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")
                        st.error("Please ensure you have run anomaly detection and have valid model results.")
    
    # Model Comparison Report tab
    with tab2:
        st.subheader("Generate Model Comparison Report")
        
        if not st.session_state.model_comparison_results:
            st.warning("No model comparison results available. Please run model comparison first.")
        else:
            model_results = st.session_state.model_comparison_results
            features = st.session_state.model_comparison_features
            
            col1, col2 = st.columns(2)
            
            with col1:
                report_format = st.selectbox(
                    "Report format",
                    options=["HTML", "PDF", "CSV", "Excel", "JSON"],
                    index=0,
                    key="comparison_report_format"
                )
                include_charts = st.checkbox("Include visualizations", value=True, key="comparison_include_charts")
            
            with col2:
                available_models = list(model_results.keys())
                selected_models = st.multiselect(
                    "Include models",
                    options=available_models,
                    default=available_models,
                    key="comparison_selected_models"
                )
                min_confidence = st.slider(
                    "Minimum anomaly score threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.8,
                    step=0.05,
                    key="comparison_min_confidence"
                )
            
            if st.button("Generate Comparison Report", key="generate_comparison_report"):
                if not selected_models:
                    st.error("Please select at least one model to include in the report.")
                else:
                    with st.spinner("Generating report..."):
                        filtered_results = {m: r for m, r in model_results.items() if m in selected_models}
                        plots = []
                        if include_charts:
                            from plotly.subplots import make_subplots
                            fig = make_subplots(
                                rows=1, 
                                cols=len(filtered_results),
                                subplot_titles=list(filtered_results.keys())
                            )
                            for i, (model_name, results) in enumerate(filtered_results.items(), 1):
                                fig.add_trace(
                                    go.Histogram(
                                        x=results["scores"],
                                        nbinsx=50,
                                        name=model_name,
                                        opacity=0.7
                                    ), row=1, col=i
                                )
                                fig.add_vline(
                                    x=results.get("threshold", 0.5),
                                    line_dash="dash",
                                    line_color="red",
                                    row=1, col=i
                                )
                            fig.update_layout(
                                title="Anomaly Score Distributions by Model",
                                height=400,
                                showlegend=False
                            )
                            plots.append(fig)
                        comparison_data = []
                        for model_name, results in filtered_results.items():
                            scores = results["scores"]
                            threshold = results.get("threshold", 0.5)
                            anomaly_count = (scores > threshold).sum()
                            anomaly_percent = anomaly_count / len(scores) * 100
                            comparison_data.append({
                                "Model": model_name,
                                "Anomalies Detected": anomaly_count,
                                "Percentage (%)": f"{anomaly_percent:.2f}%",
                                "Threshold": threshold,
                                "Training Time (s)": results.get("training_time", 0),
                                "Prediction Time (s)": results.get("prediction_time", 0)
                            })
                        comparison_df = pd.DataFrame(comparison_data)
                        if report_format in ["HTML", "PDF"]:
                            html_report = generate_comparison_report(
                                df,
                                filtered_results,
                                features,
                                plots
                            )
                            if report_format == "PDF":
                                try:
                                    pdf_path = generate_pdf_report(html_report)
                                    if pdf_path:
                                        with open(pdf_path, "rb") as f:
                                            pdf_bytes = f.read()
                                        st.download_button(
                                            "Download PDF Report",
                                            data=pdf_bytes,
                                            file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                            mime="application/pdf"
                                        )
                                    else:
                                        st.error("Failed to generate PDF report.")
                                except Exception as e:
                                    st.error(f"Error generating PDF: {str(e)}")
                            else:
                                st.markdown("### HTML Report Preview")
                                st.components.v1.html(html_report, height=500, scrolling=True)
                                st.download_button(
                                    "Download HTML Report",
                                    data=html_report,
                                    file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                    mime="text/html"
                                )
                        elif report_format == "CSV":
                            csv_data = comparison_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "Download CSV Report",
                                data=csv_data,
                                file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        elif report_format == "Excel":
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                comparison_df.to_excel(writer, sheet_name='Model Comparison', index=False)
                                for model_name, results in filtered_results.items():
                                    model_anomalies = df.copy()
                                    model_anomalies['anomaly_score'] = results["scores"]
                                    model_threshold = results.get("threshold", 0.5)
                                    model_anomalies['is_anomaly'] = results["scores"] > model_threshold
                                    model_anomalies = model_anomalies[model_anomalies['is_anomaly']]
                                    if not model_anomalies.empty:
                                        sheet_name = model_name[:31]
                                        model_anomalies.to_excel(writer, sheet_name=sheet_name, index=False)
                            excel_data = output.getvalue()
                            st.download_button(
                                "Download Excel Report",
                                data=excel_data,
                                file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        elif report_format == "JSON":
                            json_obj = {
                                "metadata": {
                                    "report_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    "total_records": len(df),
                                    "features_used": features
                                },
                                "model_comparison": json.loads(comparison_df.to_json(orient='records'))
                            }
                            json_obj["models"] = {}
                            for model_name, results in filtered_results.items():
                                model_anomalies = df.copy()
                                model_anomalies['anomaly_score'] = results["scores"]
                                model_threshold = results.get("threshold", 0.5)
                                model_anomalies['is_anomaly'] = results["scores"] > model_threshold
                                model_anomalies = model_anomalies[model_anomalies['is_anomaly']]
                                if not model_anomalies.empty:
                                    json_obj["models"][model_name] = {
                                        "threshold": float(model_threshold),
                                        "training_time": float(results["training_time"]),
                                        "prediction_time": float(results["prediction_time"]),
                                        "anomalies": json.loads(model_anomalies.to_json(orient='records', date_format='iso'))
                                    }
                            json_str = json.dumps(json_obj, indent=2)
                            st.download_button(
                                "Download JSON Report",
                                data=json_str,
                                file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
    # Executive Summary tab
    with tab3:
        st.subheader("Executive Summary")
        if st.session_state.anomalies.empty:
            st.warning("No anomalies detected. Please run anomaly detection first.")
        else:
            anomalies = st.session_state.anomalies
            model_results = st.session_state.model_results
            
            # Get threshold with fallback logic
            threshold = model_results.get('threshold', st.session_state.get('anomaly_threshold', 0.5))
            if threshold is None or threshold == 0:
                scores = model_results.get('scores', [])
                if len(scores) > 0:
                    threshold = np.percentile(scores, 90)
                else:
                    threshold = 0.5
            
            summary = generate_executive_summary(
                df,
                anomalies,
                st.session_state.selected_model,
                threshold
            )
            st.markdown(summary)
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download as Markdown",
                    data=summary,
                    file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )
            with col2:
                if st.button("Generate PDF Summary"):
                    with st.spinner("Generating PDF..."):
                        import markdown
                        html_content = markdown.markdown(summary)
                        html_report = f"""
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <meta charset="UTF-8">
                            <title>Executive Summary</title>
                            <style>
                                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                                h1, h2, h3 {{ color: #2c3e50; }}
                            </style>
                        </head>
                        <body>
                            {html_content}
                        </body>
                        </html>
                        """
                        try:
                            pdf_path = generate_pdf_report(html_report)
                            if pdf_path:
                                with open(pdf_path, "rb") as f:
                                    pdf_bytes = f.read()
                                st.download_button(
                                    "Download PDF Summary",
                                    data=pdf_bytes,
                                    file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d')}.pdf",
                                    mime="application/pdf"
                                )
                            else:
                                st.error("Failed to generate PDF summary.")
                        except Exception as e:
                            st.error(f"Error generating PDF: {str(e)}")
    # Custom Report tab
    with tab4:
        st.subheader("Custom Report Builder")
        if st.session_state.combined_data is None:
            st.warning("No data loaded. Please go to the Data Upload page first.")
            return
        df = st.session_state.combined_data
        generate_report, report_config = create_report_ui(df)
        if generate_report:
            with st.spinner("Generating custom report..."):
                if report_config["columns"]:
                    report_df = df[report_config["columns"]].copy()
                else:
                    report_df = df.copy()
                report_df = report_df.head(report_config["max_records"])
                plots = []
                if report_config["include_charts"]:
                    if '_ws_col_Protocol' in report_df.columns:
                        protocol_chart = plot_protocol_pie(report_df)
                        if protocol_chart:
                            plots.append(protocol_chart)
                    if 'timestamp' in report_df.columns:
                        timeline_chart = plot_anomaly_timeline(
                            report_df, 
                            np.zeros(len(report_df)),
                            time_col='timestamp',
                            window='1H'
                        )
                        if timeline_chart:
                            plots.append(timeline_chart)
                if report_config["format"] == "HTML":
                    from app.components.report_generator import render_html_template, DEFAULT_REPORT_TEMPLATE
                    context = {
                        "title": report_config["title"],
                        "description": report_config["description"],
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "plots": plots,
                        "include_data": report_config["include_data"],
                        "table_html": report_df.to_html(index=False) if report_config["include_data"] else ""
                    }
                    html_report = render_html_template(DEFAULT_REPORT_TEMPLATE, context)
                    st.markdown("### HTML Report Preview")
                    st.components.v1.html(html_report, height=500, scrolling=True)
                    st.download_button(
                        "Download HTML Report",
                        data=html_report,
                        file_name=f"custom_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )
                elif report_config["format"] == "PDF":
                    from app.components.report_generator import render_html_template, DEFAULT_REPORT_TEMPLATE
                    context = {
                        "title": report_config["title"],
                        "description": report_config["description"],
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "plots": plots,
                        "include_data": report_config["include_data"],
                        "table_html": report_df.to_html(index=False) if report_config["include_data"] else ""
                    }
                    html_report = render_html_template(DEFAULT_REPORT_TEMPLATE, context)
                    try:
                        pdf_path = generate_pdf_report(html_report)
                        if pdf_path:
                            with open(pdf_path, "rb") as f:
                                pdf_bytes = f.read()
                            st.download_button(
                                "Download PDF Report",
                                data=pdf_bytes,
                                file_name=f"custom_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf"
                            )
                        else:
                            st.error("Failed to generate PDF report.")
                    except Exception as e:
                        st.error(f"Error generating PDF: {str(e)}")
                elif report_config["format"] == "CSV":
                    csv_data = report_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download CSV Report",
                        data=csv_data,
                        file_name=f"custom_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                elif report_config["format"] == "Excel":
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        report_df.to_excel(writer, sheet_name='Data', index=False)
                        metadata = pd.DataFrame([
                            {"Key": "Report Title", "Value": report_config["title"]},
                            {"Key": "Report Date", "Value": datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                            {"Key": "Total Records", "Value": len(report_df)},
                            {"Key": "Description", "Value": report_config["description"]}
                        ])
                        metadata.to_excel(writer, sheet_name='Metadata', index=False)
                    excel_data = output.getvalue()
                    st.download_button(
                        "Download Excel Report",
                        data=excel_data,
                        file_name=f"custom_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                elif report_config["format"] == "JSON":
                    json_obj = {
                        "metadata": {
                            "title": report_config["title"],
                            "description": report_config["description"],
                            "report_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "total_records": len(report_df)
                        },
                        "data": json.loads(report_df.to_json(orient='records', date_format='iso'))
                    }
                    json_str = json.dumps(json_obj, indent=2)
                    st.download_button(
                        "Download JSON Report",
                        data=json_str,
                        file_name=f"custom_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
