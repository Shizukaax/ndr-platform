"""
Factory for creating model explainers.
Provides a way to create the appropriate explainer based on the model type.
"""

import inspect
from typing import Dict, Any, Optional, Union, Type

from .base_explainer import BaseExplainer
from .shap_explainer import ShapExplainer
from .lime_explainer import LimeExplainer

class ExplainerFactory:
    """Factory for creating model explainers."""
    
    @staticmethod
    def create_explainer(explainer_type: str, model=None, **kwargs) -> BaseExplainer:
        """
        Create an explainer of the specified type.
        
        Args:
            explainer_type (str): Type of explainer to create ('shap' or 'lime')
            model: The trained model to explain
            **kwargs: Additional parameters for the explainer
            
        Returns:
            BaseExplainer: The created explainer
        """
        if explainer_type.lower() == 'shap':
            return ShapExplainer(model=model, **kwargs)
        elif explainer_type.lower() == 'lime':
            return LimeExplainer(model=model, **kwargs)
        else:
            raise ValueError(f"Unsupported explainer type: {explainer_type}")
    
    @staticmethod
    def create_best_explainer(model, X, **kwargs) -> BaseExplainer:
        """
        Create the best explainer for the given model based on model type.
        
        Args:
            model: The trained model to explain
            X: Training data to use for the explainer
            **kwargs: Additional parameters for the explainer
            
        Returns:
            BaseExplainer: The created explainer
        """
        # Extract model class name
        model_class = model.__class__.__name__
        
        # Use feature names if provided in kwargs or extract from X
        feature_names = kwargs.get('feature_names')
        if feature_names is None and hasattr(X, 'columns'):
            feature_names = X.columns.tolist()
        
        # For tree-based models, prefer SHAP
        if model_class in ['IsolationForestDetector', 'RandomForestRegressor', 'RandomForestClassifier',
                          'GradientBoostingRegressor', 'GradientBoostingClassifier',
                          'XGBRegressor', 'XGBClassifier', 'LGBMRegressor', 'LGBMClassifier']:
            try:
                return ShapExplainer(model=model, feature_names=feature_names, background_data=X, **kwargs)
            except:
                # Fall back to LIME if SHAP fails
                return LimeExplainer(model=model, feature_names=feature_names, training_data=X, **kwargs)
        
        # For neural network models, use SHAP with DeepExplainer
        elif 'keras' in str(model.__class__) or 'torch' in str(model.__class__) or 'tensorflow' in str(model.__class__):
            return ShapExplainer(model=model, feature_names=feature_names, background_data=X, explainer_type='deep', **kwargs)
        
        # For kernel-based or distance-based models, prefer LIME
        elif model_class in ['LocalOutlierFactorDetector', 'OneClassSVMDetector', 'KNNDetector', 'DBSCANDetector']:
            return LimeExplainer(model=model, feature_names=feature_names, training_data=X, **kwargs)
        
        # For ensemble models, try SHAP first then fall back to LIME
        elif model_class in ['EnsembleDetector', 'VotingRegressor', 'VotingClassifier', 'StackingRegressor', 'StackingClassifier']:
            try:
                return ShapExplainer(model=model, feature_names=feature_names, background_data=X, **kwargs)
            except:
                return LimeExplainer(model=model, feature_names=feature_names, training_data=X, **kwargs)
        
        # For other models, use LIME as it's more model-agnostic
        else:
            return LimeExplainer(model=model, feature_names=feature_names, training_data=X, **kwargs)


def get_explainer(explainer_type: str = 'auto', model=None, X=None, **kwargs) -> BaseExplainer:
    """
    Convenience function to get an explainer instance.
    
    Args:
        explainer_type (str): Type of explainer ('shap', 'lime', or 'auto')
        model: The trained model to explain
        X: Training data to use for the explainer
        **kwargs: Additional parameters for the explainer
        
    Returns:
        BaseExplainer: The created explainer
    """
    factory = ExplainerFactory()
    
    if explainer_type == 'auto':
        if model is None or X is None:
            raise ValueError("Model and training data (X) are required for automatic explainer selection")
        return factory.create_best_explainer(model, X, **kwargs)
    else:
        # For specific explainer types, pass the training data as background_data or training_data
        if explainer_type.lower() == 'shap':
            kwargs['background_data'] = X
        elif explainer_type.lower() == 'lime':
            kwargs['training_data'] = X
        return factory.create_explainer(explainer_type, model, **kwargs)