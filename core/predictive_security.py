"""
Predictive Security Analytics Engine
Core module for threat prediction, risk assessment, and intelligent alerting
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import logging

class PredictiveSecurityEngine:
    """Core engine for predictive security analytics and threat forecasting"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.prediction_models = {
            'threat_probability': self._threat_probability_model,
            'attack_progression': self._attack_progression_model,
            'risk_escalation': self._risk_escalation_model,
            'resource_demand': self._resource_demand_model
        }
        
        # Model parameters
        self.threat_baseline = 0.15
        self.seasonal_patterns = {
            'hourly': np.array([0.8, 0.6, 0.4, 0.3, 0.3, 0.4, 0.7, 1.0, 
                               1.2, 1.3, 1.4, 1.5, 1.4, 1.3, 1.2, 1.1,
                               1.0, 0.9, 0.8, 0.7, 0.8, 0.9, 1.0, 0.9]),
            'daily': np.array([1.0, 0.9, 0.8, 0.7, 0.8, 1.2, 1.3])  # Mon-Sun
        }
    
    def predict_threat_probability(self, time_horizon_hours: int = 24) -> List[Dict[str, Any]]:
        """
        Predict threat probability over specified time horizon
        
        Args:
            time_horizon_hours: Number of hours to predict ahead
            
        Returns:
            List of hourly predictions with probability, confidence, and risk factors
        """
        try:
            predictions = []
            current_time = datetime.now()
            
            for hour in range(time_horizon_hours):
                prediction_time = current_time + timedelta(hours=hour)
                
                # Apply time-based patterns
                hour_factor = self.seasonal_patterns['hourly'][prediction_time.hour]
                day_factor = self.seasonal_patterns['daily'][prediction_time.weekday()]
                
                # Calculate base probability
                base_prob = self.threat_baseline * hour_factor * day_factor
                
                # Add environmental factors
                environmental_risk = self._calculate_environmental_risk(prediction_time)
                
                # Add trend analysis
                trend_factor = self._calculate_trend_factor(prediction_time)
                
                # Combine factors
                final_probability = min(0.95, base_prob + environmental_risk + trend_factor)
                
                # Calculate confidence based on data quality and model stability
                confidence = self._calculate_prediction_confidence(prediction_time)
                
                # Identify primary risk factors
                risk_factors = self._identify_risk_factors(prediction_time, final_probability)
                
                predictions.append({
                    'timestamp': prediction_time,
                    'hour_offset': hour,
                    'threat_probability': final_probability,
                    'confidence': confidence,
                    'risk_factors': risk_factors,
                    'severity': self._categorize_risk(final_probability),
                    'recommended_actions': self._generate_recommendations(final_probability, risk_factors)
                })
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error in threat probability prediction: {e}")
            return []
    
    def predict_attack_progression(self, initial_indicators: List[str]) -> Dict[str, Any]:
        """
        Predict potential attack progression based on initial indicators
        
        Args:
            initial_indicators: List of observed attack indicators
            
        Returns:
            Attack progression prediction with stages and probabilities
        """
        try:
            # Define attack stages (MITRE ATT&CK inspired)
            attack_stages = [
                'Initial Access',
                'Execution',
                'Persistence',
                'Privilege Escalation',
                'Defense Evasion',
                'Credential Access',
                'Discovery',
                'Lateral Movement',
                'Collection',
                'Command and Control',
                'Exfiltration',
                'Impact'
            ]
            
            # Map indicators to likely attack stages
            stage_probabilities = self._map_indicators_to_stages(initial_indicators, attack_stages)
            
            # Predict progression likelihood
            progression_model = self._build_progression_model(stage_probabilities)
            
            # Calculate time estimates for each stage
            time_estimates = self._estimate_stage_timing(progression_model)
            
            return {
                'attack_stages': attack_stages,
                'stage_probabilities': stage_probabilities,
                'progression_likelihood': progression_model,
                'estimated_timing': time_estimates,
                'critical_stages': self._identify_critical_stages(stage_probabilities),
                'mitigation_opportunities': self._suggest_mitigations(progression_model)
            }
            
        except Exception as e:
            self.logger.error(f"Error in attack progression prediction: {e}")
            return {}
    
    def predict_risk_escalation(self, current_incidents: List[Dict]) -> List[Dict[str, Any]]:
        """
        Predict which current incidents are likely to escalate
        
        Args:
            current_incidents: List of current security incidents
            
        Returns:
            Escalation predictions for each incident
        """
        try:
            escalation_predictions = []
            
            for incident in current_incidents:
                # Extract incident features
                features = self._extract_incident_features(incident)
                
                # Calculate escalation probability
                escalation_prob = self._calculate_escalation_probability(features)
                
                # Estimate time to escalation
                time_to_escalation = self._estimate_escalation_time(features, escalation_prob)
                
                # Identify escalation factors
                escalation_factors = self._identify_escalation_factors(features)
                
                escalation_predictions.append({
                    'incident_id': incident.get('id', 'unknown'),
                    'current_severity': incident.get('severity', 'unknown'),
                    'escalation_probability': escalation_prob,
                    'estimated_time_to_escalation': time_to_escalation,
                    'escalation_factors': escalation_factors,
                    'recommended_priority': self._calculate_priority(escalation_prob, features),
                    'intervention_suggestions': self._suggest_interventions(features, escalation_prob)
                })
            
            # Sort by escalation probability
            escalation_predictions.sort(key=lambda x: x['escalation_probability'], reverse=True)
            
            return escalation_predictions
            
        except Exception as e:
            self.logger.error(f"Error in risk escalation prediction: {e}")
            return []
    
    def predict_resource_demand(self, prediction_window_hours: int = 8) -> Dict[str, Any]:
        """
        Predict security resource demand (analyst time, system load, etc.)
        
        Args:
            prediction_window_hours: Hours ahead to predict
            
        Returns:
            Resource demand predictions
        """
        try:
            current_time = datetime.now()
            predictions = []
            
            for hour in range(prediction_window_hours):
                prediction_time = current_time + timedelta(hours=hour)
                
                # Predict analyst workload
                analyst_demand = self._predict_analyst_demand(prediction_time)
                
                # Predict system resource usage
                system_load = self._predict_system_load(prediction_time)
                
                # Predict alert volume
                alert_volume = self._predict_alert_volume(prediction_time)
                
                predictions.append({
                    'timestamp': prediction_time,
                    'hour_offset': hour,
                    'analyst_demand': analyst_demand,
                    'system_load_prediction': system_load,
                    'expected_alert_volume': alert_volume,
                    'resource_stress_level': self._calculate_stress_level(
                        analyst_demand, system_load, alert_volume
                    )
                })
            
            return {
                'hourly_predictions': predictions,
                'peak_demand_periods': self._identify_peak_periods(predictions),
                'resource_recommendations': self._generate_resource_recommendations(predictions),
                'capacity_warnings': self._check_capacity_warnings(predictions)
            }
            
        except Exception as e:
            self.logger.error(f"Error in resource demand prediction: {e}")
            return {}
    
    def _threat_probability_model(self, time_features: Dict) -> float:
        """Calculate threat probability using time-based features"""
        base_prob = 0.15
        
        # Time-based adjustments
        hour_multiplier = self.seasonal_patterns['hourly'][time_features['hour']]
        day_multiplier = self.seasonal_patterns['daily'][time_features['weekday']]
        
        # Environmental factors
        if time_features.get('is_business_hours', False):
            base_prob *= 1.2
        
        if time_features.get('is_weekend', False):
            base_prob *= 0.8
        
        return min(0.95, base_prob * hour_multiplier * day_multiplier)
    
    def _attack_progression_model(self, indicators: List[str]) -> Dict[str, float]:
        """Model attack progression based on indicators"""
        # Simplified progression model
        stage_weights = {
            'reconnaissance': 0.9,
            'weaponization': 0.7,
            'delivery': 0.8,
            'exploitation': 0.85,
            'installation': 0.75,
            'command_control': 0.8,
            'actions_objectives': 0.6
        }
        
        progression = {}
        for stage, weight in stage_weights.items():
            # Calculate based on indicator patterns
            indicator_match = len([i for i in indicators if stage[:3] in i.lower()]) / max(len(indicators), 1)
            progression[stage] = min(0.95, weight * indicator_match + np.random.uniform(0.1, 0.3))
        
        return progression
    
    def _risk_escalation_model(self, incident_features: Dict) -> float:
        """Calculate risk escalation probability"""
        base_risk = 0.2
        
        # Feature-based adjustments
        severity_multipliers = {'low': 0.5, 'medium': 1.0, 'high': 1.5, 'critical': 2.0}
        severity = incident_features.get('severity', 'medium').lower()
        base_risk *= severity_multipliers.get(severity, 1.0)
        
        # Time factor (older incidents more likely to escalate)
        age_hours = incident_features.get('age_hours', 1)
        time_factor = min(2.0, 1.0 + (age_hours / 24) * 0.5)
        base_risk *= time_factor
        
        # Asset criticality factor
        if incident_features.get('affects_critical_asset', False):
            base_risk *= 1.5
        
        return min(0.95, base_risk)
    
    def _resource_demand_model(self, time_features: Dict) -> Dict[str, float]:
        """Model resource demand based on time features"""
        # Base demand levels
        base_analyst_demand = 0.6
        base_system_load = 0.4
        
        # Time-based adjustments
        hour = time_features['hour']
        if 8 <= hour <= 18:  # Business hours
            base_analyst_demand *= 1.4
            base_system_load *= 1.2
        elif 18 <= hour <= 22:  # Evening
            base_analyst_demand *= 1.1
            base_system_load *= 1.0
        else:  # Night/early morning
            base_analyst_demand *= 0.7
            base_system_load *= 0.8
        
        return {
            'analyst_demand': min(1.0, base_analyst_demand),
            'system_load': min(1.0, base_system_load)
        }
    
    def _calculate_environmental_risk(self, prediction_time: datetime) -> float:
        """Calculate environmental risk factors"""
        risk = 0.0
        
        # Simulated external threat intelligence
        if np.random.random() > 0.8:  # 20% chance of elevated threat
            risk += 0.1
        
        # Simulated vulnerability exposure
        if np.random.random() > 0.9:  # 10% chance of new vulnerabilities
            risk += 0.15
        
        # Time-based patterns (e.g., end of month, holidays)
        if prediction_time.day >= 28:  # End of month
            risk += 0.05
        
        return risk
    
    def _calculate_trend_factor(self, prediction_time: datetime) -> float:
        """Calculate trend-based risk adjustments"""
        # Simulate trend analysis based on historical data
        base_trend = np.sin(prediction_time.timestamp() / 86400) * 0.05  # Daily cycle
        weekly_trend = np.cos(prediction_time.timestamp() / 604800) * 0.03  # Weekly cycle
        
        return base_trend + weekly_trend
    
    def _calculate_prediction_confidence(self, prediction_time: datetime) -> float:
        """Calculate confidence in prediction based on data quality"""
        base_confidence = 0.85
        
        # Reduce confidence for predictions further in the future
        hours_ahead = (prediction_time - datetime.now()).total_seconds() / 3600
        time_decay = max(0.1, 1.0 - (hours_ahead / 72))  # Confidence decays over 72 hours
        
        # Simulated data quality factors
        data_quality = np.random.uniform(0.8, 1.0)
        
        return min(0.95, base_confidence * time_decay * data_quality)
    
    def _identify_risk_factors(self, prediction_time: datetime, probability: float) -> List[str]:
        """Identify primary risk factors for given prediction"""
        factors = []
        
        if prediction_time.hour in [14, 15, 16]:  # Peak business hours
            factors.append("Peak business hours")
        
        if prediction_time.weekday() in [0, 1]:  # Monday, Tuesday
            factors.append("High activity weekday")
        
        if probability > 0.6:
            factors.append("Elevated threat landscape")
        
        if np.random.random() > 0.7:
            factors.append("External threat intelligence")
        
        return factors
    
    def _categorize_risk(self, probability: float) -> str:
        """Categorize risk level based on probability"""
        if probability >= 0.8:
            return "Critical"
        elif probability >= 0.6:
            return "High"
        elif probability >= 0.4:
            return "Medium"
        else:
            return "Low"
    
    def _generate_recommendations(self, probability: float, risk_factors: List[str]) -> List[str]:
        """Generate actionable recommendations based on prediction"""
        recommendations = []
        
        if probability >= 0.7:
            recommendations.append("Increase monitoring alert sensitivity")
            recommendations.append("Deploy additional analysts")
            recommendations.append("Review critical asset protections")
        
        if "Peak business hours" in risk_factors:
            recommendations.append("Monitor user authentication carefully")
        
        if "External threat intelligence" in risk_factors:
            recommendations.append("Update threat hunting queries")
        
        return recommendations
    
    # Additional helper methods would go here...
    def _map_indicators_to_stages(self, indicators: List[str], stages: List[str]) -> Dict[str, float]:
        """Map indicators to attack stages"""
        # Simplified mapping - in practice this would use ML models
        stage_probs = {}
        for stage in stages:
            # Random probability for demonstration
            stage_probs[stage] = np.random.uniform(0.1, 0.9)
        return stage_probs
    
    def _build_progression_model(self, stage_probs: Dict[str, float]) -> Dict[str, float]:
        """Build attack progression model"""
        return stage_probs  # Simplified for demo
    
    def _estimate_stage_timing(self, progression: Dict[str, float]) -> Dict[str, str]:
        """Estimate timing for each attack stage"""
        timing = {}
        for stage in progression:
            # Random timing for demonstration
            hours = np.random.uniform(1, 24)
            timing[stage] = f"{hours:.1f} hours"
        return timing
    
    def _identify_critical_stages(self, stage_probs: Dict[str, float]) -> List[str]:
        """Identify critical attack stages"""
        return [stage for stage, prob in stage_probs.items() if prob > 0.7]
    
    def _suggest_mitigations(self, progression: Dict[str, float]) -> List[str]:
        """Suggest mitigation strategies"""
        return ["Deploy additional monitoring", "Update access controls", "Review network segmentation"]
    
    def _extract_incident_features(self, incident: Dict) -> Dict[str, Any]:
        """Extract features from incident for escalation prediction"""
        return {
            'severity': incident.get('severity', 'medium'),
            'age_hours': incident.get('age_hours', 1),
            'affects_critical_asset': incident.get('critical_asset', False),
            'user_count': incident.get('affected_users', 1),
            'system_count': incident.get('affected_systems', 1)
        }
    
    def _calculate_escalation_probability(self, features: Dict[str, Any]) -> float:
        """Calculate incident escalation probability"""
        base_prob = 0.2
        
        # Severity adjustment
        severity_multipliers = {'low': 0.5, 'medium': 1.0, 'high': 1.5, 'critical': 2.0}
        severity = features.get('severity', 'medium').lower()
        base_prob *= severity_multipliers.get(severity, 1.0)
        
        # Age factor
        age_hours = features.get('age_hours', 1)
        if age_hours > 24:
            base_prob *= 1.5
        
        return min(0.95, base_prob)
    
    def _estimate_escalation_time(self, features: Dict[str, Any], probability: float) -> str:
        """Estimate time until escalation"""
        base_hours = 12
        
        if probability > 0.8:
            base_hours = 4
        elif probability > 0.6:
            base_hours = 8
        
        return f"{base_hours} hours"
    
    def _identify_escalation_factors(self, features: Dict[str, Any]) -> List[str]:
        """Identify factors contributing to escalation risk"""
        factors = []
        
        if features.get('affects_critical_asset', False):
            factors.append("Critical asset involvement")
        
        if features.get('age_hours', 0) > 24:
            factors.append("Extended incident duration")
        
        if features.get('user_count', 0) > 10:
            factors.append("Multiple users affected")
        
        return factors
    
    def _calculate_priority(self, escalation_prob: float, features: Dict[str, Any]) -> str:
        """Calculate incident priority"""
        if escalation_prob > 0.8:
            return "P1 - Critical"
        elif escalation_prob > 0.6:
            return "P2 - High"
        elif escalation_prob > 0.4:
            return "P3 - Medium"
        else:
            return "P4 - Low"
    
    def _suggest_interventions(self, features: Dict[str, Any], escalation_prob: float) -> List[str]:
        """Suggest intervention strategies"""
        interventions = []
        
        if escalation_prob > 0.7:
            interventions.append("Immediate senior analyst assignment")
            interventions.append("Activate incident response team")
        
        if features.get('affects_critical_asset', False):
            interventions.append("Isolate affected systems")
        
        return interventions
    
    def _predict_analyst_demand(self, prediction_time: datetime) -> float:
        """Predict analyst workload demand"""
        base_demand = 0.6
        
        # Business hours adjustment
        if 8 <= prediction_time.hour <= 18:
            base_demand *= 1.4
        elif 18 <= prediction_time.hour <= 22:
            base_demand *= 1.1
        else:
            base_demand *= 0.7
        
        return min(1.0, base_demand)
    
    def _predict_system_load(self, prediction_time: datetime) -> float:
        """Predict system resource load"""
        base_load = 0.4
        
        # Add daily patterns
        hour_factor = self.seasonal_patterns['hourly'][prediction_time.hour]
        base_load *= hour_factor
        
        return min(1.0, base_load)
    
    def _predict_alert_volume(self, prediction_time: datetime) -> int:
        """Predict expected alert volume"""
        base_volume = 20
        
        # Time-based adjustments
        hour_factor = self.seasonal_patterns['hourly'][prediction_time.hour]
        base_volume *= hour_factor
        
        # Add randomness
        volume = int(base_volume + np.random.poisson(5))
        
        return max(0, volume)
    
    def _calculate_stress_level(self, analyst_demand: float, system_load: float, alert_volume: int) -> str:
        """Calculate overall resource stress level"""
        stress_score = (analyst_demand + system_load) / 2 + (alert_volume / 100)
        
        if stress_score > 0.8:
            return "High Stress"
        elif stress_score > 0.6:
            return "Medium Stress"
        else:
            return "Low Stress"
    
    def _identify_peak_periods(self, predictions: List[Dict]) -> List[Dict]:
        """Identify peak demand periods"""
        peaks = []
        for pred in predictions:
            if (pred['analyst_demand'] > 0.8 or 
                pred['system_load_prediction'] > 0.8 or 
                pred['expected_alert_volume'] > 50):
                peaks.append(pred)
        return peaks
    
    def _generate_resource_recommendations(self, predictions: List[Dict]) -> List[str]:
        """Generate resource allocation recommendations"""
        recommendations = []
        
        high_demand_periods = [p for p in predictions if p['analyst_demand'] > 0.8]
        if high_demand_periods:
            recommendations.append("Schedule additional analysts for high-demand periods")
        
        high_load_periods = [p for p in predictions if p['system_load_prediction'] > 0.8]
        if high_load_periods:
            recommendations.append("Consider scaling system resources")
        
        return recommendations
    
    def _check_capacity_warnings(self, predictions: List[Dict]) -> List[str]:
        """Check for capacity warnings"""
        warnings = []
        
        critical_periods = [p for p in predictions if p['resource_stress_level'] == "High Stress"]
        if critical_periods:
            warnings.append(f"High stress predicted for {len(critical_periods)} hour(s)")
        
        return warnings
