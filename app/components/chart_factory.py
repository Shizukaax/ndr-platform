"""
Chart factory for the Network Anomaly Detection Platform.
Provides standardized chart creation with consistent styling.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger("streamlit_app")

class ChartFactory:
    """Factory for creating standardized charts across the application."""
    
    def __init__(self):
        """Initialize chart factory with default styling."""
        self.default_colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#ff7f0e',
            'danger': '#d62728',
            'info': '#17a2b8',
            'anomaly': '#d62728',
            'normal': '#2ca02c'
        }
        
        self.default_layout = {
            'font': {'family': 'Arial, sans-serif', 'size': 12},
            'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50},
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white'
        }
    
    def create_anomaly_score_distribution(self, scores: np.ndarray, threshold: float = None, 
                                        title: str = "Anomaly Score Distribution") -> go.Figure:
        """
        Create anomaly score distribution chart.
        
        Args:
            scores: Array of anomaly scores
            threshold: Anomaly threshold line
            title: Chart title
            
        Returns:
            Plotly figure
        """
        try:
            fig = go.Figure()
            
            # Add histogram
            fig.add_trace(go.Histogram(
                x=scores,
                nbinsx=50,
                name="Score Distribution",
                marker_color=self.default_colors['primary'],
                opacity=0.7
            ))
            
            # Add threshold line if provided
            if threshold is not None:
                fig.add_vline(
                    x=threshold,
                    line_dash="dash",
                    line_color=self.default_colors['danger'],
                    annotation_text=f"Threshold: {threshold:.3f}",
                    annotation_position="top"
                )
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Anomaly Score",
                yaxis_title="Frequency",
                **self.default_layout
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating anomaly score distribution: {str(e)}")
            return self._create_error_chart("Error creating score distribution")
    
    def create_anomaly_scatter(self, data: pd.DataFrame, x_col: str, y_col: str,
                             anomaly_col: str = 'is_anomaly', 
                             title: str = "Anomaly Scatter Plot") -> go.Figure:
        """
        Create scatter plot highlighting anomalies.
        
        Args:
            data: DataFrame with data points
            x_col: X-axis column name
            y_col: Y-axis column name
            anomaly_col: Column indicating anomaly status
            title: Chart title
            
        Returns:
            Plotly figure
        """
        try:
            fig = go.Figure()
            
            # Separate normal and anomalous points
            normal_data = data[~data[anomaly_col]]
            anomaly_data = data[data[anomaly_col]]
            
            # Add normal points
            if not normal_data.empty:
                fig.add_trace(go.Scatter(
                    x=normal_data[x_col],
                    y=normal_data[y_col],
                    mode='markers',
                    name='Normal',
                    marker=dict(
                        color=self.default_colors['normal'],
                        size=6,
                        opacity=0.6
                    )
                ))
            
            # Add anomaly points
            if not anomaly_data.empty:
                fig.add_trace(go.Scatter(
                    x=anomaly_data[x_col],
                    y=anomaly_data[y_col],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(
                        color=self.default_colors['anomaly'],
                        size=8,
                        symbol='x',
                        opacity=0.8
                    )
                ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title=x_col.replace('_', ' ').title(),
                yaxis_title=y_col.replace('_', ' ').title(),
                **self.default_layout
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating anomaly scatter plot: {str(e)}")
            return self._create_error_chart("Error creating scatter plot")
    
    def create_timeline_chart(self, data: pd.DataFrame, time_col: str, value_col: str,
                            anomaly_col: str = None, title: str = "Timeline Chart") -> go.Figure:
        """
        Create timeline chart with optional anomaly highlighting.
        
        Args:
            data: DataFrame with time series data
            time_col: Time column name
            value_col: Value column name
            anomaly_col: Optional anomaly indicator column
            title: Chart title
            
        Returns:
            Plotly figure
        """
        try:
            fig = go.Figure()
            
            # Add main time series
            fig.add_trace(go.Scatter(
                x=data[time_col],
                y=data[value_col],
                mode='lines+markers',
                name='Values',
                line=dict(color=self.default_colors['primary'])
            ))
            
            # Highlight anomalies if specified
            if anomaly_col and anomaly_col in data.columns:
                anomalies = data[data[anomaly_col]]
                if not anomalies.empty:
                    fig.add_trace(go.Scatter(
                        x=anomalies[time_col],
                        y=anomalies[value_col],
                        mode='markers',
                        name='Anomalies',
                        marker=dict(
                            color=self.default_colors['anomaly'],
                            size=10,
                            symbol='x'
                        )
                    ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Time",
                yaxis_title=value_col.replace('_', ' ').title(),
                **self.default_layout
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating timeline chart: {str(e)}")
            return self._create_error_chart("Error creating timeline chart")
    
    def create_mitre_technique_chart(self, technique_counts: Dict[str, int],
                                   title: str = "MITRE ATT&CK Techniques") -> go.Figure:
        """
        Create chart for MITRE technique distribution.
        
        Args:
            technique_counts: Dictionary of technique counts
            title: Chart title
            
        Returns:
            Plotly figure
        """
        try:
            if not technique_counts:
                return self._create_empty_chart("No MITRE techniques found")
            
            # Prepare data
            techniques = list(technique_counts.keys())
            counts = list(technique_counts.values())
            
            # Sort by count (descending)
            sorted_data = sorted(zip(techniques, counts), key=lambda x: x[1], reverse=True)
            techniques, counts = zip(*sorted_data)
            
            # Take top 10
            techniques = techniques[:10]
            counts = counts[:10]
            
            # Create horizontal bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=counts,
                y=techniques,
                orientation='h',
                marker_color=self.default_colors['danger'],
                text=counts,
                textposition='auto'
            ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Frequency",
                yaxis_title="Technique",
                **self.default_layout
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating MITRE technique chart: {str(e)}")
            return self._create_error_chart("Error creating MITRE chart")
    
    def create_risk_distribution_chart(self, risk_data: List[Dict[str, Any]],
                                     title: str = "Risk Score Distribution") -> go.Figure:
        """
        Create risk score distribution chart.
        
        Args:
            risk_data: List of risk score dictionaries
            title: Chart title
            
        Returns:
            Plotly figure
        """
        try:
            if not risk_data:
                return self._create_empty_chart("No risk data available")
            
            # Extract risk levels
            risk_levels = [item.get('risk_level', 'Unknown') for item in risk_data]
            
            # Count risk levels
            from collections import Counter
            level_counts = Counter(risk_levels)
            
            # Define colors for risk levels
            risk_colors = {
                'Critical': '#8B0000',
                'High': '#DC143C',
                'Medium': '#FF8C00',
                'Low': '#FFD700',
                'Minimal': '#90EE90'
            }
            
            labels = list(level_counts.keys())
            values = list(level_counts.values())
            colors = [risk_colors.get(label, self.default_colors['primary']) for label in labels]
            
            # Create pie chart
            fig = go.Figure()
            
            fig.add_trace(go.Pie(
                labels=labels,
                values=values,
                marker_colors=colors,
                textinfo='label+percent+value',
                hole=0.3
            ))
            
            # Update layout
            fig.update_layout(
                title=title,
                **self.default_layout
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating risk distribution chart: {str(e)}")
            return self._create_error_chart("Error creating risk chart")
    
    def create_comparison_chart(self, comparison_data: Dict[str, Any],
                              title: str = "Model Comparison") -> go.Figure:
        """
        Create model comparison chart.
        
        Args:
            comparison_data: Dictionary with comparison data
            title: Chart title
            
        Returns:
            Plotly figure
        """
        try:
            if not comparison_data:
                return self._create_empty_chart("No comparison data available")
            
            # Extract model names and metrics
            models = list(comparison_data.keys())
            anomaly_counts = [comparison_data[model].get('n_anomalies', 0) for model in models]
            
            # Create bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=models,
                y=anomaly_counts,
                marker_color=self.default_colors['primary'],
                text=anomaly_counts,
                textposition='auto'
            ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Model",
                yaxis_title="Anomalies Detected",
                **self.default_layout
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating comparison chart: {str(e)}")
            return self._create_error_chart("Error creating comparison chart")
    
    def _create_error_chart(self, message: str) -> go.Figure:
        """Create an error message chart."""
        fig = go.Figure()
        
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            font=dict(size=16, color="red")
        )
        
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            **self.default_layout
        )
        
        return fig
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create an empty data message chart."""
        fig = go.Figure()
        
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            font=dict(size=14, color="gray")
        )
        
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            **self.default_layout
        )
        
        return fig

# Singleton instance
chart_factory = ChartFactory()
