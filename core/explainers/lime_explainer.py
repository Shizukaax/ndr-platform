"""
LIME-based model explainer for anomaly detection models.
Provides local interpretable model-agnostic explanations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Any, Union, Optional
import lime
import lime.lime_tabular
from lime.explanation import Explanation
from lime.lime_tabular import LimeTabularExplainer

from .base_explainer import BaseExplainer

# Get logger for this module
logger = logging.getLogger(__name__)

class LimeExplainer(BaseExplainer):
    """
    LIME-based explainer for interpreting anomaly detection models.
    Uses LIME to explain individual predictions with a locally interpretable model.
    """
    
    def __init__(self, model=None, feature_names=None, class_names=None, training_data=None, **kwargs):
        """
        Initialize the LIME explainer.
        
        Args:
            model: The trained anomaly detection model to explain
            feature_names (list, optional): Names of the features
            class_names (list, optional): Names of the classes (for classification)
            training_data: Training data for the LIME explainer
            **kwargs: Additional parameters for the LIME explainer
        """
        super().__init__(model, feature_names, class_names)
        self.training_data = training_data
        self.kwargs = kwargs
        self.explainer = None
        self.categorical_features = kwargs.get('categorical_features', [])
        
        # Try to initialize the explainer right away if model and training data are provided
        if model is not None and training_data is not None and feature_names is not None:
            self._initialize_explainer(training_data)
    
    def predict_fn(self, instances):
        """
        Prediction function for LIME to call the model.
        
        Args:
            instances: Array of instances to predict
            
        Returns:
            Array of prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model is not initialized")
        
        try:
            # Convert to numpy array if needed
            if isinstance(instances, pd.DataFrame):
                instances = instances.values
            
            # Ensure 2D array
            if len(instances.shape) == 1:
                instances = instances.reshape(1, -1)
            
            # Get anomaly scores from the model
            scores = self.model.predict(instances)
            
            # Convert scores to probabilities for LIME
            # For anomaly detection, we treat it as binary classification
            # Higher scores = more anomalous = higher probability of being anomaly
            if hasattr(self.model, 'metadata') and 'anomaly_threshold' in self.model.metadata:
                threshold = self.model.metadata['anomaly_threshold']
            else:
                # Use default threshold based on scores distribution
                threshold = np.percentile(scores, 90)  # Top 10% as anomalies
            
            # Convert to probabilities
            probabilities = np.zeros((len(scores), 2))
            for i, score in enumerate(scores):
                if score > threshold:
                    # Anomaly: higher probability for class 1 (anomaly)
                    prob_anomaly = min(0.99, 0.5 + (score - threshold) / (2 * abs(threshold)))
                    probabilities[i] = [1 - prob_anomaly, prob_anomaly]
                else:
                    # Normal: higher probability for class 0 (normal)
                    prob_normal = min(0.99, 0.5 + (threshold - score) / (2 * abs(threshold)))
                    probabilities[i] = [prob_normal, 1 - prob_normal]
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Error in predict_fn: {str(e)}")
            # Return default probabilities if prediction fails
            num_instances = len(instances) if hasattr(instances, '__len__') else 1
            return np.array([[0.5, 0.5]] * num_instances)
    
    def _initialize_explainer(self, training_data):
        """Initialize the LIME tabular explainer."""
        try:
            # Robust data preprocessing for LIME
            processed_data = training_data.copy()
            
            # Check for data variance issues and fix them
            if hasattr(processed_data, 'var'):
                variance = processed_data.var()
                zero_variance_features = variance[variance == 0].index.tolist()
                near_zero_variance_features = variance[variance < 1e-8].index.tolist()
                
                if zero_variance_features:
                    logger.warning(f"Found zero variance features: {zero_variance_features}")
                    # Add small noise to zero variance features
                    for feature in zero_variance_features:
                        processed_data[feature] += np.random.normal(0, 1e-6, len(processed_data))
                
                if near_zero_variance_features:
                    logger.warning(f"Found near-zero variance features: {near_zero_variance_features}")
                    # Add small noise to near-zero variance features
                    for feature in near_zero_variance_features:
                        processed_data[feature] += np.random.normal(0, 1e-6, len(processed_data))
            
            # Ensure all data is finite and not too extreme
            processed_data = processed_data.replace([np.inf, -np.inf], np.nan)
            processed_data = processed_data.fillna(processed_data.median())
            
            # Clip extreme values
            for col in processed_data.columns:
                Q1 = processed_data[col].quantile(0.01)
                Q99 = processed_data[col].quantile(0.99)
                processed_data[col] = processed_data[col].clip(Q1, Q99)
            
            # Initialize LIME with very conservative settings
            self.explainer = LimeTabularExplainer(
                processed_data.values,
                feature_names=self.feature_names,
                class_names=['Normal', 'Anomaly'],
                mode='classification',
                verbose=False,  # Reduce verbosity
                sample_around_instance=False,  # Use simpler sampling
                discretize_continuous=False,  # Disable discretization completely
                random_state=42,
                kernel_width=None,  # Let LIME determine kernel width
                feature_selection='auto'  # Use automatic feature selection
            )
            
            logger.info("LIME explainer initialized successfully with robust settings")
            
        except Exception as e:
            logger.error(f"Error initializing LIME explainer: {str(e)}")
            # Try a minimal fallback initialization
            try:
                logger.info("Attempting fallback LIME initialization...")
                simple_data = training_data.select_dtypes(include=[np.number]).fillna(0)
                self.explainer = LimeTabularExplainer(
                    simple_data.values,
                    feature_names=list(simple_data.columns),
                    mode='classification',
                    discretize_continuous=False,
                    random_state=42
                )
                self.feature_names = list(simple_data.columns)
                logger.info("Fallback LIME explainer initialized")
            except Exception as fallback_error:
                logger.error(f"Fallback LIME initialization also failed: {str(fallback_error)}")
                raise
    
    def explain(self, X, instance_index=None):
        """
        Generate LIME explanation for the given instance.
        
        Args:
            X: Input data to explain
            instance_index (int, optional): Index of the instance to explain
            
        Returns:
            dict: Explanation results
        """
        if self.explainer is None:
            self.training_data = X
            self._initialize_explainer(X)
        
        if instance_index is None:
            instance_index = 0
        
        if isinstance(X, pd.DataFrame):
            instance = X.iloc[instance_index].values
            instance_df = X.iloc[instance_index:instance_index+1]
        else:
            instance = X[instance_index]
            instance_df = pd.DataFrame([instance], columns=self.feature_names)
        
        # Ensure instance data is clean
        instance = np.nan_to_num(instance.astype(float), nan=0.0, posinf=1e6, neginf=-1e6)
        
        num_features = min(self.kwargs.get('num_features', 10), len(self.feature_names))
        
        try:
            lime_explanation = self.explainer.explain_instance(
                instance,
                self.predict_fn,
                num_features=num_features,
                top_labels=1,
                num_samples=self.kwargs.get('num_samples', 1000)  # Reduce samples to avoid issues
            )
            
            feature_importance = {}
            exp = lime_explanation.as_list()
            for feature_desc, importance in exp:
                for feature_name in self.feature_names:
                    if feature_name in feature_desc:
                        feature_importance[feature_name] = importance
                        break
            
            return {
                "lime_explanation": lime_explanation,
                "feature_names": self.feature_names,
                "instance_index": instance_index,
                "instance": instance_df,
                "score": self.model.predict(instance.reshape(1, -1))[0],
                "feature_importance": feature_importance
            }
            
        except Exception as e:
            # Return error if LIME fails
            return {
                "error": str(e),
                "feature_names": self.feature_names,
                "instance_index": instance_index,
                "instance": instance_df,
                "score": self.model.predict(instance.reshape(1, -1))[0] if hasattr(self, 'model') else None,
                "feature_importance": {}
            }
    
    def plot_explanation(self, explanation):
        """
        Plot the LIME explanation.
        
        Args:
            explanation (dict): Explanation from the explain method
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        lime_explanation = explanation.get("lime_explanation")
        if lime_explanation is None:
            raise ValueError("Explanation must contain 'lime_explanation'")
        
        plt.figure(figsize=(10, 6))  # Set figure size here
        lime_explanation.as_pyplot_figure()  # Do NOT pass figsize!
        fig = plt.gcf()
        plt.tight_layout()
        return fig
    
    def get_feature_importance(self, explanation=None):
        """
        Get feature importance from the explanation.
        
        Args:
            explanation (dict, optional): Explanation from the explain method
            
        Returns:
            dict: Feature importance scores
        """
        if explanation is None:
            raise ValueError("Explanation must be provided")
        
        return explanation.get("feature_importance", {})