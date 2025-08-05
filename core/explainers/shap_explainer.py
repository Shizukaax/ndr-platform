"""
SHAP-based model explainer for anomaly detection models.
Provides SHAP value calculation and visualization for feature importance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from typing import List, Dict, Any, Union, Optional
import streamlit as st

from .base_explainer import BaseExplainer

class ShapExplainer(BaseExplainer):
    """
    SHAP-based explainer for interpreting anomaly detection models.
    Uses SHAP to calculate feature importance and explain individual predictions.
    """
    
    def __init__(self, model=None, feature_names=None, class_names=None, background_data=None, **kwargs):
        """
        Initialize the SHAP explainer.
        
        Args:
            model: The trained anomaly detection model to explain
            feature_names (list, optional): Names of the features
            class_names (list, optional): Names of the classes (for classification)
            background_data: Background data for SHAP explainer
            **kwargs: Additional parameters for the SHAP explainer
        """
        super().__init__(model, feature_names, class_names)
        self.background_data = background_data
        self.kwargs = kwargs
        self.explainer = None
        
        # Determine the correct explainer to use
        self.explainer_type = kwargs.get('explainer_type', 'auto')
        
        # Try to initialize the explainer right away if model and background data are provided
        if model is not None and background_data is not None:
            try:
                self._initialize_explainer()
            except:
                # We'll initialize it later when needed
                pass
    
    def _extract_sklearn_model_from_pipeline(self, pipeline):
        """
        Extract the final estimator from an sklearn pipeline.
        
        Args:
            pipeline: The pipeline to extract the final estimator from
            
        Returns:
            The final estimator from the pipeline
        """
        # Check if this is a Pipeline
        if hasattr(pipeline, 'steps'):
            # Return the final step's estimator
            return pipeline.steps[-1][1]
        # Check if this is a ColumnTransformer
        elif hasattr(pipeline, 'transformers'):
            # This is more complex, but we'll just return the pipeline itself
            return pipeline
        # If it's not a pipeline, return it as is
        return pipeline

    def _get_final_model_type(self, model):
        """
        Get the type of the final model in a pipeline or the model itself.
        
        Args:
            model: The model or pipeline
            
        Returns:
            str: The type of the final model
        """
        # Extract the final model from a pipeline if needed
        if hasattr(model, 'steps'):
            final_model = model.steps[-1][1]
        else:
            final_model = model
            
        # Get the class name
        return final_model.__class__.__name__
    
    def _initialize_explainer(self):
        """Initialize the appropriate SHAP explainer based on the model type."""
        if self.model is None:
            raise ValueError("Model must be provided before initializing the explainer")
        
        if self.background_data is None:
            raise ValueError("Background data must be provided for SHAP explanation")
        
        # Extract the final model from a pipeline if needed
        sklearn_model = self._extract_sklearn_model_from_pipeline(self.model.model)
        
        # Determine model type
        model_type = self._get_final_model_type(self.model.model)
        
        if self.explainer_type == 'auto':
            # Automatically determine the explainer to use based on model type
            if model_type in ['RandomForestClassifier', 'RandomForestRegressor', 
                             'IsolationForest', 'GradientBoostingClassifier', 
                             'GradientBoostingRegressor', 'DecisionTreeClassifier', 
                             'DecisionTreeRegressor', 'ExtraTreesClassifier', 
                             'ExtraTreesRegressor']:
                # Tree-based models can use TreeExplainer
                try:
                    self.explainer = shap.TreeExplainer(sklearn_model, **self.kwargs)
                    return
                except Exception as e:
                    print(f"TreeExplainer failed: {str(e)}")
                    # Fall back to KernelExplainer
                    pass
            
            # For unsupported models or if TreeExplainer fails, use KernelExplainer
            try:
                # Check if the model has a predict_proba method
                if hasattr(sklearn_model, 'predict_proba'):
                    self.explainer = shap.KernelExplainer(sklearn_model.predict_proba, 
                                                         shap.sample(self.background_data, 100))
                else:
                    # Otherwise use the predict method
                    self.explainer = shap.KernelExplainer(sklearn_model.predict, 
                                                         shap.sample(self.background_data, 100))
            except Exception as e:
                # As a last resort, use a simple wrapper function
                print(f"KernelExplainer failed: {str(e)}")
                def predict_fn(x):
                    return self.model.predict(x)
                self.explainer = shap.KernelExplainer(predict_fn, 
                                                     shap.sample(self.background_data, 100))
        
        elif self.explainer_type == 'kernel':
            # Create a prediction function
            def predict_fn(x):
                return self.model.predict(x)
            self.explainer = shap.KernelExplainer(predict_fn, 
                                                 shap.sample(self.background_data, 100))
        
        elif self.explainer_type == 'tree':
            # Use TreeExplainer for tree-based models
            self.explainer = shap.TreeExplainer(sklearn_model, **self.kwargs)
        
        elif self.explainer_type == 'deep':
            # DeepExplainer for deep learning models
            self.explainer = shap.DeepExplainer(self.model.model, self.background_data)
        
        else:
            raise ValueError(f"Unsupported explainer type: {self.explainer_type}")
    
    def explain(self, X, instance_index=None):
        """
        Generate SHAP explanation for the given instance.
        
        Args:
            X: Input data to explain
            instance_index (int, optional): Index of the instance to explain
            
        Returns:
            dict: Explanation results including SHAP values and visualizations
        """
        # Initialize explainer if not done yet
        if self.explainer is None:
            self.background_data = X
            self._initialize_explainer()
        
        # If no specific instance is provided, explain all instances
        if instance_index is None:
            shap_values = self.explainer.shap_values(X)
            
            # Handle different return types based on explainer
            if isinstance(shap_values, list):
                # For multi-output models, use the first output
                shap_values = shap_values[0]
        else:
            # Select the specific instance
            instance = X.iloc[instance_index].values.reshape(1, -1) if hasattr(X, 'iloc') else X[instance_index].reshape(1, -1)
            shap_values = self.explainer.shap_values(instance)
            
            # Handle different return types
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
        
        # Create feature names if not provided
        if self.feature_names is None:
            if hasattr(X, 'columns'):
                self.feature_names = X.columns.tolist()
            else:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Return explanation
        return {
            "shap_values": shap_values,
            "feature_names": self.feature_names,
            "instance_index": instance_index,
            "base_value": self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
            "data": X.values if hasattr(X, 'values') else X  # Include the actual data for plotting
        }
    
    def plot_summary(self, explanation=None, max_display=10, plot_type="bar"):
        """
        Plot a summary of the SHAP values.
        
        Args:
            explanation (dict, optional): Explanation from the explain method
            max_display (int): Maximum number of features to display
            plot_type (str): Type of plot ('bar', 'beeswarm', 'heatmap')
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        if explanation is None:
            raise ValueError("Explanation must be provided")
        
        shap_values = explanation["shap_values"]
        feature_names = explanation["feature_names"]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Different plot types
        if plot_type == "bar":
            # Create bar plot of feature importance
            shap.summary_plot(
                shap_values, 
                features=explanation.get("data", None),
                feature_names=feature_names,
                max_display=max_display,
                plot_type="bar",
                show=False
            )
        elif plot_type == "beeswarm":
            # Create beeswarm plot
            shap.summary_plot(
                shap_values, 
                features=explanation.get("data", None),
                feature_names=feature_names,
                max_display=max_display,
                show=False
            )
        elif plot_type == "heatmap":
            # Create heatmap
            if len(shap_values.shape) == 2:
                shap.summary_plot(
                    shap_values, 
                    features=explanation.get("data", None),
                    feature_names=feature_names,
                    max_display=max_display,
                    plot_type="compact_dot",
                    show=False
                )
            else:
                print("Heatmap plot requires 2D SHAP values array")
                return None
        
        # Return the current figure
        fig = plt.gcf()
        plt.tight_layout()
        return fig
    
    def plot_dependence(self, explanation, feature_idx, interaction_idx=None):
        """
        Plot a dependence plot for a specific feature.
        
        Args:
            explanation (dict): Explanation from the explain method
            feature_idx (int or str): Index or name of the feature to plot
            interaction_idx (int or str, optional): Index or name of the interaction feature
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        if explanation is None:
            raise ValueError("Explanation must be provided")
        
        shap_values = explanation["shap_values"]
        feature_names = explanation["feature_names"]
        
        # Convert feature name to index if needed
        if isinstance(feature_idx, str):
            if feature_idx in feature_names:
                feature_idx = feature_names.index(feature_idx)
            else:
                raise ValueError(f"Feature name '{feature_idx}' not found in feature names")
        
        # Convert interaction feature name to index if needed
        if interaction_idx is not None and isinstance(interaction_idx, str):
            if interaction_idx in feature_names:
                interaction_idx = feature_names.index(interaction_idx)
            else:
                raise ValueError(f"Feature name '{interaction_idx}' not found in feature names")
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create dependence plot
        shap.dependence_plot(
            feature_idx, 
            shap_values, 
            features=explanation.get("data", None),
            feature_names=feature_names,
            interaction_index=interaction_idx,
            show=False
        )
        
        # Return the current figure
        fig = plt.gcf()
        plt.tight_layout()
        return fig
    
    def plot_force(self, explanation, instance_index=None):
        """
        Create a force plot for a specific instance.
        
        Args:
            explanation (dict): Explanation from the explain method
            instance_index (int, optional): Index of the instance to plot
            
        Returns:
            shap.plots._force.AdditiveForceVisualizer: Force plot object
        """
        if explanation is None:
            raise ValueError("Explanation must be provided")
        
        shap_values = explanation["shap_values"]
        feature_names = explanation["feature_names"]
        base_value = explanation.get("base_value", 0)
        
        # Get data for the instance
        if instance_index is not None:
            if instance_index >= len(shap_values):
                raise ValueError(f"Instance index {instance_index} out of range")
            
            # Get single instance
            instance_shap = shap_values[instance_index]
            instance_data = explanation.get("data", None)
            if instance_data is not None:
                instance_data = instance_data[instance_index]
            
            # Create force plot
            force_plot = shap.force_plot(
                base_value=base_value,
                shap_values=instance_shap,
                features=instance_data,
                feature_names=feature_names
            )
        else:
            # Create force plot for all instances
            force_plot = shap.force_plot(
                base_value=base_value,
                shap_values=shap_values,
                features=explanation.get("data", None),
                feature_names=feature_names
            )
        
        return force_plot
    
    def get_feature_importance(self, explanation=None):
        """
        Get global feature importance based on SHAP values.
        
        Args:
            explanation (dict, optional): Explanation from the explain method
            
        Returns:
            dict: Feature importance scores
        """
        if explanation is None:
            raise ValueError("Explanation must be provided")
        
        shap_values = explanation["shap_values"]
        feature_names = explanation["feature_names"]
        
        # Calculate mean absolute SHAP values for each feature
        importance_values = np.mean(np.abs(shap_values), axis=0) if len(shap_values.shape) > 1 else np.abs(shap_values)
        
        # Create dictionary mapping feature names to importance scores
        importance_dict = {name: float(value) for name, value in zip(feature_names, importance_values)}
        
        # Sort by importance
        importance_dict = {k: v for k, v in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)}
        
        return importance_dict