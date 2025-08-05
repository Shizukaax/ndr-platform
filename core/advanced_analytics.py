"""
Advanced Analytics Module for Network Anomaly Detection Platform.
Provides behavioral baselines, predictive analytics, and advanced pattern recognition.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import streamlit as st
from dataclasses import dataclass, asdict
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger("advanced_analytics")

@dataclass
class BehavioralBaseline:
    """Data class for behavioral baselines."""
    entity_id: str  # IP, user, etc.
    entity_type: str  # ip, user, service, etc.
    baseline_period: Tuple[datetime, datetime]
    features: Dict[str, Any]  # Statistical features
    patterns: Dict[str, Any]  # Behavioral patterns
    risk_level: str  # low, medium, high
    confidence: float
    last_updated: datetime

@dataclass
class PredictiveInsight:
    """Data class for predictive insights."""
    prediction_type: str  # trend, anomaly_forecast, risk_escalation
    confidence: float
    time_horizon: timedelta
    predicted_value: Any
    factors: List[str]  # Contributing factors
    recommendation: str
    created_at: datetime

class BehavioralAnalytics:
    """Advanced behavioral analytics engine."""
    
    def __init__(self):
        self.baselines = {}  # Entity ID -> BehavioralBaseline
        self.temporal_patterns = {}
        self.cluster_models = {}
        self.scaler = StandardScaler()
    
    def build_behavioral_baselines(self, data: pd.DataFrame, 
                                 entity_column: str = 'ip_src',
                                 time_column: str = None,
                                 baseline_days: int = 30) -> Dict[str, BehavioralBaseline]:
        """Build behavioral baselines for entities."""
        
        if time_column:
            # Filter to baseline period
            end_date = pd.to_datetime(data[time_column]).max()
            start_date = end_date - timedelta(days=baseline_days)
            baseline_data = data[pd.to_datetime(data[time_column]) >= start_date]
        else:
            baseline_data = data
        
        baselines = {}
        
        # Group by entity
        for entity_id, entity_data in baseline_data.groupby(entity_column):
            if len(entity_data) < 5:  # Need minimum data points
                continue
            
            baseline = self._calculate_entity_baseline(entity_id, entity_data, time_column)
            baselines[entity_id] = baseline
            
        self.baselines.update(baselines)
        logger.info(f"Built baselines for {len(baselines)} entities")
        
        return baselines
    
    def _calculate_entity_baseline(self, entity_id: str, data: pd.DataFrame, 
                                 time_column: Optional[str]) -> BehavioralBaseline:
        """Calculate baseline for a single entity."""
        
        # Extract numerical features
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        features = {}
        patterns = {}
        
        # Statistical features
        for col in numeric_cols:
            if col != entity_id and not data[col].isna().all():
                values = data[col].dropna()
                features[f"{col}_mean"] = float(values.mean())
                features[f"{col}_std"] = float(values.std())
                features[f"{col}_median"] = float(values.median())
                features[f"{col}_q75"] = float(values.quantile(0.75))
                features[f"{col}_q25"] = float(values.quantile(0.25))
                features[f"{col}_max"] = float(values.max())
                features[f"{col}_min"] = float(values.min())
        
        # Temporal patterns if time column available
        if time_column and time_column in data.columns:
            patterns.update(self._extract_temporal_patterns(data, time_column))
        
        # Communication patterns
        patterns.update(self._extract_communication_patterns(data))
        
        # Volume patterns
        patterns.update(self._extract_volume_patterns(data))
        
        # Calculate risk level
        risk_level = self._assess_baseline_risk(features, patterns)
        
        # Calculate confidence based on data quantity and consistency
        confidence = self._calculate_baseline_confidence(data, features)
        
        return BehavioralBaseline(
            entity_id=str(entity_id),
            entity_type="ip",  # Could be detected automatically
            baseline_period=(data.index.min(), data.index.max()),
            features=features,
            patterns=patterns,
            risk_level=risk_level,
            confidence=confidence,
            last_updated=datetime.now()
        )
    
    def _extract_temporal_patterns(self, data: pd.DataFrame, time_column: str) -> Dict[str, Any]:
        """Extract temporal behavioral patterns."""
        patterns = {}
        
        try:
            timestamps = pd.to_datetime(data[time_column])
            
            # Hour of day patterns
            hour_counts = timestamps.dt.hour.value_counts().sort_index()
            patterns['active_hours'] = hour_counts.to_dict()
            patterns['peak_hour'] = int(hour_counts.idxmax())
            patterns['activity_variance'] = float(hour_counts.std())
            
            # Day of week patterns
            dow_counts = timestamps.dt.day_of_week.value_counts().sort_index()
            patterns['weekday_activity'] = dow_counts.to_dict()
            
            # Activity frequency
            daily_counts = timestamps.dt.date.value_counts()
            patterns['avg_daily_events'] = float(daily_counts.mean())
            patterns['daily_variance'] = float(daily_counts.std())
            
        except Exception as e:
            logger.warning(f"Error extracting temporal patterns: {e}")
        
        return patterns
    
    def _extract_communication_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract communication behavioral patterns."""
        patterns = {}
        
        # Destination diversity
        dst_cols = [col for col in data.columns if 'dst' in col.lower() or 'destination' in col.lower()]
        for col in dst_cols:
            if col in data.columns:
                unique_dsts = data[col].nunique()
                patterns[f'{col}_diversity'] = unique_dsts
        
        # Port usage patterns
        port_cols = [col for col in data.columns if 'port' in col.lower()]
        for col in port_cols:
            if col in data.columns and data[col].dtype in ['int64', 'float64']:
                unique_ports = data[col].nunique()
                common_ports = data[col].value_counts().head(5).to_dict()
                patterns[f'{col}_diversity'] = unique_ports
                patterns[f'{col}_common'] = {str(k): v for k, v in common_ports.items()}
        
        # Protocol patterns
        protocol_cols = [col for col in data.columns if 'protocol' in col.lower()]
        for col in protocol_cols:
            if col in data.columns:
                protocol_dist = data[col].value_counts().to_dict()
                patterns[f'{col}_distribution'] = {str(k): v for k, v in protocol_dist.items()}
        
        return patterns
    
    def _extract_volume_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract volume and size behavioral patterns."""
        patterns = {}
        
        # Data volume patterns
        volume_cols = [col for col in data.columns if any(term in col.lower() 
                      for term in ['bytes', 'size', 'length', 'volume'])]
        
        for col in volume_cols:
            if col in data.columns and data[col].dtype in ['int64', 'float64']:
                values = data[col].dropna()
                if len(values) > 0:
                    patterns[f'{col}_total'] = float(values.sum())
                    patterns[f'{col}_avg_per_event'] = float(values.mean())
                    patterns[f'{col}_peak'] = float(values.max())
        
        return patterns
    
    def _assess_baseline_risk(self, features: Dict[str, Any], patterns: Dict[str, Any]) -> str:
        """Assess risk level based on baseline characteristics."""
        risk_score = 0.0
        
        # High diversity in destinations = higher risk
        for key, value in patterns.items():
            if 'diversity' in key and isinstance(value, (int, float)):
                if value > 100:
                    risk_score += 0.3
                elif value > 50:
                    risk_score += 0.1
        
        # Unusual port usage = higher risk
        for key, value in patterns.items():
            if 'port_diversity' in key and isinstance(value, (int, float)):
                if value > 20:
                    risk_score += 0.2
        
        # High variance in activity = higher risk
        for key, value in patterns.items():
            if 'variance' in key and isinstance(value, (int, float)):
                if value > 10:
                    risk_score += 0.1
        
        # Classify risk level
        if risk_score >= 0.5:
            return "high"
        elif risk_score >= 0.25:
            return "medium"
        else:
            return "low"
    
    def _calculate_baseline_confidence(self, data: pd.DataFrame, features: Dict[str, Any]) -> float:
        """Calculate confidence in the baseline."""
        confidence = 0.0
        
        # More data points = higher confidence
        data_points = len(data)
        if data_points >= 100:
            confidence += 0.4
        elif data_points >= 50:
            confidence += 0.3
        elif data_points >= 20:
            confidence += 0.2
        else:
            confidence += 0.1
        
        # Lower variance = higher confidence
        variance_scores = [v for k, v in features.items() if 'std' in k and isinstance(v, (int, float))]
        if variance_scores:
            avg_variance = np.mean(variance_scores)
            if avg_variance < 1.0:
                confidence += 0.3
            elif avg_variance < 5.0:
                confidence += 0.2
            else:
                confidence += 0.1
        
        # Time span coverage
        confidence += 0.3  # Base confidence for having any baseline
        
        return min(confidence, 1.0)
    
    def detect_baseline_deviations(self, new_data: pd.DataFrame, 
                                 entity_column: str = 'ip_src',
                                 threshold_std: float = 2.0) -> pd.DataFrame:
        """Detect deviations from established baselines."""
        
        deviations = []
        
        for entity_id, entity_data in new_data.groupby(entity_column):
            if str(entity_id) not in self.baselines:
                continue
                
            baseline = self.baselines[str(entity_id)]
            
            # Calculate current behavior
            current_features = self._calculate_current_features(entity_data)
            
            # Compare with baseline
            deviation_score = self._calculate_deviation_score(
                current_features, baseline.features, threshold_std
            )
            
            if deviation_score > 0.5:  # Significant deviation
                deviations.append({
                    'entity_id': entity_id,
                    'deviation_score': deviation_score,
                    'baseline_risk': baseline.risk_level,
                    'baseline_confidence': baseline.confidence,
                    'deviation_type': self._classify_deviation_type(current_features, baseline.features),
                    'event_count': len(entity_data)
                })
        
        return pd.DataFrame(deviations)
    
    def _calculate_current_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate current features for comparison with baseline."""
        features = {}
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if not data[col].isna().all():
                values = data[col].dropna()
                if len(values) > 0:
                    features[f"{col}_mean"] = float(values.mean())
                    features[f"{col}_std"] = float(values.std())
                    features[f"{col}_median"] = float(values.median())
        
        return features
    
    def _calculate_deviation_score(self, current: Dict[str, Any], 
                                 baseline: Dict[str, Any], 
                                 threshold_std: float) -> float:
        """Calculate deviation score between current and baseline features."""
        
        deviations = []
        
        for feature_name in current.keys():
            if feature_name in baseline:
                current_val = current[feature_name]
                baseline_mean = baseline[feature_name]
                baseline_std_key = feature_name.replace('_mean', '_std')
                
                if baseline_std_key in baseline:
                    baseline_std = baseline[baseline_std_key]
                    if baseline_std > 0:
                        z_score = abs(current_val - baseline_mean) / baseline_std
                        if z_score > threshold_std:
                            deviations.append(min(z_score / threshold_std, 3.0))  # Cap at 3x threshold
        
        return np.mean(deviations) if deviations else 0.0
    
    def _classify_deviation_type(self, current: Dict[str, Any], baseline: Dict[str, Any]) -> str:
        """Classify the type of deviation observed."""
        
        volume_deviations = []
        frequency_deviations = []
        
        for feature_name, current_val in current.items():
            if feature_name in baseline:
                baseline_val = baseline[feature_name]
                ratio = current_val / baseline_val if baseline_val > 0 else float('inf')
                
                if 'bytes' in feature_name or 'size' in feature_name:
                    if ratio > 2.0:
                        volume_deviations.append('high_volume')
                    elif ratio < 0.5:
                        volume_deviations.append('low_volume')
                
                elif 'count' in feature_name or 'freq' in feature_name:
                    if ratio > 2.0:
                        frequency_deviations.append('high_frequency')
                    elif ratio < 0.5:
                        frequency_deviations.append('low_frequency')
        
        # Determine primary deviation type
        if volume_deviations:
            return max(set(volume_deviations), key=volume_deviations.count)
        elif frequency_deviations:
            return max(set(frequency_deviations), key=frequency_deviations.count)
        else:
            return 'general_deviation'

class PredictiveAnalytics:
    """Predictive analytics for anomaly forecasting."""
    
    def __init__(self):
        self.time_series_models = {}
        self.trend_models = {}
    
    def forecast_anomaly_trends(self, historical_data: pd.DataFrame,
                              time_column: str,
                              forecast_hours: int = 24) -> Dict[str, Any]:
        """Forecast anomaly trends for the next period."""
        
        try:
            # Prepare time series data
            ts_data = self._prepare_time_series(historical_data, time_column)
            
            if len(ts_data) < 10:
                return {"error": "Insufficient historical data for forecasting"}
            
            # Simple trend analysis (could be enhanced with advanced time series models)
            forecast = self._simple_trend_forecast(ts_data, forecast_hours)
            
            # Identify patterns
            patterns = self._identify_patterns(ts_data)
            
            # Generate insights
            insights = self._generate_predictive_insights(forecast, patterns)
            
            return {
                "forecast": forecast,
                "patterns": patterns,
                "insights": insights,
                "confidence": self._calculate_forecast_confidence(ts_data)
            }
            
        except Exception as e:
            logger.error(f"Error in anomaly forecasting: {e}")
            return {"error": str(e)}
    
    def _prepare_time_series(self, data: pd.DataFrame, time_column: str) -> pd.DataFrame:
        """Prepare time series data for forecasting."""
        
        try:
            # Check if time column exists
            if time_column not in data.columns:
                raise ValueError(f"Time column '{time_column}' not found in data")
            
            # Convert to datetime
            data[time_column] = pd.to_datetime(data[time_column])
            
            # Group by hour and count anomalies
            hourly_counts = data.groupby(data[time_column].dt.floor('H')).size()
            
            # Create complete time range
            start_time = hourly_counts.index.min()
            end_time = hourly_counts.index.max()
            complete_range = pd.date_range(start=start_time, end=end_time, freq='H')
            
            # Reindex to fill missing hours with 0
            ts_data = hourly_counts.reindex(complete_range, fill_value=0)
            
            return pd.DataFrame({
                'timestamp': ts_data.index,
                'anomaly_count': ts_data.values
            })
            
        except Exception as e:
            logger.error(f"Error preparing time series data: {e}")
            raise
    
    def _simple_trend_forecast(self, ts_data: pd.DataFrame, forecast_hours: int) -> Dict[str, Any]:
        """Simple trend-based forecasting."""
        
        values = ts_data['anomaly_count'].values
        timestamps = ts_data['timestamp'].values
        
        # Calculate trend using simple linear regression
        x = np.arange(len(values))
        trend_coef = np.polyfit(x, values, 1)[0]
        
        # Generate forecast
        last_timestamp = timestamps[-1]
        forecast_timestamps = pd.date_range(
            start=last_timestamp + pd.Timedelta(hours=1),
            periods=forecast_hours,
            freq='H'
        )
        
        # Simple extrapolation with trend
        base_value = np.mean(values[-24:]) if len(values) >= 24 else np.mean(values)
        forecast_values = []
        
        for i in range(forecast_hours):
            # Add trend and some cyclical pattern (24-hour cycle)
            hour = forecast_timestamps[i].hour
            cyclical_factor = 1 + 0.3 * np.sin(2 * np.pi * hour / 24)  # Daily cycle
            trend_component = trend_coef * i
            predicted_value = max(0, base_value + trend_component * cyclical_factor)
            forecast_values.append(predicted_value)
        
        return {
            "timestamps": forecast_timestamps.tolist(),
            "values": forecast_values,
            "trend": "increasing" if trend_coef > 0.1 else "decreasing" if trend_coef < -0.1 else "stable"
        }
    
    def _identify_patterns(self, ts_data: pd.DataFrame) -> Dict[str, Any]:
        """Identify patterns in time series data."""
        
        patterns = {}
        values = ts_data['anomaly_count'].values
        
        # Daily patterns
        if len(ts_data) >= 24:
            daily_pattern = []
            for hour in range(24):
                hour_mask = ts_data['timestamp'].dt.hour == hour
                if hour_mask.any():
                    avg_count = ts_data[hour_mask]['anomaly_count'].mean()
                    daily_pattern.append(avg_count)
                else:
                    daily_pattern.append(0)
            
            patterns['daily_cycle'] = daily_pattern
            patterns['peak_hour'] = int(np.argmax(daily_pattern))
            patterns['quiet_hour'] = int(np.argmin(daily_pattern))
        
        # Weekly patterns
        if len(ts_data) >= 168:  # At least one week
            weekly_pattern = []
            for day in range(7):
                day_mask = ts_data['timestamp'].dt.dayofweek == day
                if day_mask.any():
                    avg_count = ts_data[day_mask]['anomaly_count'].mean()
                    weekly_pattern.append(avg_count)
                else:
                    weekly_pattern.append(0)
            
            patterns['weekly_cycle'] = weekly_pattern
            patterns['busiest_day'] = int(np.argmax(weekly_pattern))
        
        # Anomaly bursts (periods of high activity)
        threshold = np.percentile(values, 90)
        burst_periods = []
        in_burst = False
        burst_start = None
        
        for i, (timestamp, count) in enumerate(zip(ts_data['timestamp'], values)):
            if count > threshold and not in_burst:
                in_burst = True
                burst_start = timestamp
            elif count <= threshold and in_burst:
                in_burst = False
                burst_periods.append({
                    'start': burst_start,
                    'end': timestamp,
                    'duration_hours': (timestamp - burst_start).total_seconds() / 3600
                })
        
        patterns['burst_periods'] = len(burst_periods)
        patterns['avg_burst_duration'] = np.mean([b['duration_hours'] for b in burst_periods]) if burst_periods else 0
        
        return patterns
    
    def _generate_predictive_insights(self, forecast: Dict[str, Any], patterns: Dict[str, Any]) -> List[PredictiveInsight]:
        """Generate actionable predictive insights."""
        
        insights = []
        
        # Trend insights
        if forecast.get('trend') == 'increasing':
            insights.append(PredictiveInsight(
                prediction_type="trend",
                confidence=0.7,
                time_horizon=timedelta(hours=24),
                predicted_value="increasing_anomalies",
                factors=["historical_trend", "pattern_analysis"],
                recommendation="Increase monitoring capacity and prepare incident response",
                created_at=datetime.now()
            ))
        
        # Peak time insights
        if 'peak_hour' in patterns:
            peak_hour = patterns['peak_hour']
            insights.append(PredictiveInsight(
                prediction_type="pattern_forecast",
                confidence=0.8,
                time_horizon=timedelta(hours=1),
                predicted_value=f"peak_activity_at_hour_{peak_hour}",
                factors=["daily_pattern"],
                recommendation=f"Expect increased anomaly activity around {peak_hour}:00",
                created_at=datetime.now()
            ))
        
        # Burst pattern insights
        if patterns.get('burst_periods', 0) > 0:
            insights.append(PredictiveInsight(
                prediction_type="anomaly_forecast",
                confidence=0.6,
                time_horizon=timedelta(hours=6),
                predicted_value="potential_anomaly_burst",
                factors=["historical_burst_patterns"],
                recommendation="Monitor for anomaly clusters based on historical burst patterns",
                created_at=datetime.now()
            ))
        
        return insights
    
    def _calculate_forecast_confidence(self, ts_data: pd.DataFrame) -> float:
        """Calculate confidence in the forecast."""
        
        confidence = 0.0
        
        # More data = higher confidence
        data_points = len(ts_data)
        if data_points >= 168:  # One week
            confidence += 0.4
        elif data_points >= 72:  # Three days
            confidence += 0.3
        elif data_points >= 24:  # One day
            confidence += 0.2
        else:
            confidence += 0.1
        
        # Lower variance = higher confidence
        variance = ts_data['anomaly_count'].var()
        if variance < 1.0:
            confidence += 0.3
        elif variance < 5.0:
            confidence += 0.2
        else:
            confidence += 0.1
        
        # Pattern consistency
        if len(ts_data) >= 48:  # Two days for pattern detection
            recent_pattern = ts_data['anomaly_count'].tail(24).values
            older_pattern = ts_data['anomaly_count'].head(24).values
            correlation = np.corrcoef(recent_pattern, older_pattern)[0, 1]
            
            if not np.isnan(correlation) and correlation > 0.5:
                confidence += 0.3
            elif not np.isnan(correlation) and correlation > 0.2:
                confidence += 0.2
            else:
                confidence += 0.1
        
        return min(confidence, 1.0)

# Visualization functions
def create_behavioral_analysis_plots(baselines: Dict[str, BehavioralBaseline]) -> Dict[str, Any]:
    """Create visualizations for behavioral analysis."""
    
    plots = {}
    
    # Risk level distribution
    risk_levels = [baseline.risk_level for baseline in baselines.values()]
    risk_counts = pd.Series(risk_levels).value_counts()
    
    fig_risk = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title="Entity Risk Level Distribution",
        color_discrete_map={'low': 'green', 'medium': 'orange', 'high': 'red'}
    )
    plots['risk_distribution'] = fig_risk
    
    # Confidence distribution
    confidences = [baseline.confidence for baseline in baselines.values()]
    fig_confidence = px.histogram(
        x=confidences,
        nbins=20,
        title="Baseline Confidence Distribution",
        labels={'x': 'Confidence Score', 'y': 'Number of Entities'}
    )
    plots['confidence_distribution'] = fig_confidence
    
    # Activity patterns (if available)
    if baselines:
        sample_baseline = next(iter(baselines.values()))
        if 'active_hours' in sample_baseline.patterns:
            # Aggregate hourly activity across all entities
            hourly_activity = np.zeros(24)
            entity_count = 0
            
            for baseline in baselines.values():
                if 'active_hours' in baseline.patterns:
                    entity_count += 1
                    for hour, count in baseline.patterns['active_hours'].items():
                        hourly_activity[int(hour)] += count
            
            if entity_count > 0:
                hourly_activity /= entity_count  # Average across entities
                
                fig_hourly = px.bar(
                    x=list(range(24)),
                    y=hourly_activity,
                    title="Average Hourly Activity Pattern",
                    labels={'x': 'Hour of Day', 'y': 'Average Activity'}
                )
                plots['hourly_patterns'] = fig_hourly
    
    return plots

def create_predictive_analytics_plots(forecast_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create visualizations for predictive analytics."""
    
    plots = {}
    
    if 'forecast' in forecast_data and 'error' not in forecast_data:
        forecast = forecast_data['forecast']
        
        # Forecast plot
        fig_forecast = go.Figure()
        
        fig_forecast.add_trace(go.Scatter(
            x=forecast['timestamps'],
            y=forecast['values'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='blue', dash='dash')
        ))
        
        fig_forecast.update_layout(
            title="Anomaly Count Forecast",
            xaxis_title="Time",
            yaxis_title="Predicted Anomaly Count"
        )
        plots['forecast'] = fig_forecast
        
        # Daily pattern if available
        if 'patterns' in forecast_data and 'daily_cycle' in forecast_data['patterns']:
            daily_pattern = forecast_data['patterns']['daily_cycle']
            
            fig_daily = px.bar(
                x=list(range(24)),
                y=daily_pattern,
                title="Daily Activity Pattern",
                labels={'x': 'Hour of Day', 'y': 'Average Anomaly Count'}
            )
            plots['daily_pattern'] = fig_daily
    
    return plots

def display_advanced_analytics_dashboard(behavioral_analytics: BehavioralAnalytics,
                                       predictive_analytics: PredictiveAnalytics,
                                       data: pd.DataFrame,
                                       st_container=None):
    """Display advanced analytics dashboard in Streamlit."""
    
    if st_container is None:
        st_container = st
    
    st_container.header("🔬 Advanced Analytics Dashboard")
    
    # Behavioral Analytics Section
    st_container.subheader("📊 Behavioral Analytics")
    
    col1, col2 = st_container.columns(2)
    
    with col1:
        if st_container.button("🔍 Build Behavioral Baselines"):
            with st_container.spinner("Building behavioral baselines..."):
                baselines = behavioral_analytics.build_behavioral_baselines(data)
                st_container.success(f"Built baselines for {len(baselines)} entities")
                
                # Display visualizations
                plots = create_behavioral_analysis_plots(baselines)
                for plot_name, fig in plots.items():
                    st_container.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if st_container.button("🚨 Detect Baseline Deviations"):
            if behavioral_analytics.baselines:
                with st_container.spinner("Detecting deviations..."):
                    deviations = behavioral_analytics.detect_baseline_deviations(data)
                    
                    if not deviations.empty:
                        st_container.warning(f"Found {len(deviations)} entities with significant deviations")
                        st_container.dataframe(deviations, use_container_width=True)
                    else:
                        st_container.success("No significant baseline deviations detected")
            else:
                st_container.warning("Please build baselines first")
    
    # Predictive Analytics Section
    st_container.subheader("🔮 Predictive Analytics")
    
    time_columns = [col for col in data.columns if any(term in col.lower() for term in ['time', 'timestamp', 'date'])]
    
    if time_columns:
        selected_time_col = st_container.selectbox("Select time column:", time_columns)
        forecast_hours = st_container.slider("Forecast horizon (hours):", 1, 72, 24)
        
        if st_container.button("📈 Generate Forecast"):
            with st_container.spinner("Generating predictive analytics..."):
                forecast_data = predictive_analytics.forecast_anomaly_trends(
                    data, selected_time_col, forecast_hours
                )
                
                if 'error' in forecast_data:
                    st_container.error(f"Forecasting error: {forecast_data['error']}")
                else:
                    # Display forecast plots
                    plots = create_predictive_analytics_plots(forecast_data)
                    for plot_name, fig in plots.items():
                        st_container.plotly_chart(fig, use_container_width=True)
                    
                    # Display insights
                    if 'insights' in forecast_data:
                        st_container.subheader("🎯 Predictive Insights")
                        for insight in forecast_data['insights']:
                            st_container.info(f"**{insight.prediction_type.title()}**: {insight.recommendation}")
                    
                    # Display confidence
                    confidence = forecast_data.get('confidence', 0.0)
                    st_container.metric("Forecast Confidence", f"{confidence:.1%}")
    else:
        st_container.warning("No time columns found in the data for predictive analytics")
