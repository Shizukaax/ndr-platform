"""
Base explainer class for model interpretability.
Provides a common interface for all explainers.
"""

from abc import ABC, abstractmethod

class BaseExplainer(ABC):
    """Base class for all explainers."""
    
    def __init__(self, model=None, feature_names=None, class_names=None):
        """
        Initialize the base explainer.
        
        Args:
            model: The trained model to explain
            feature_names (list, optional): Names of the features
            class_names (list, optional): Names of the classes (for classification)
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
    
    @abstractmethod
    def explain(self, X, instance_index=None):
        """
        Generate an explanation for the given instances.
        
        Args:
            X: Input data to explain
            instance_index (int, optional): Index of the instance to explain
            
        Returns:
            dict: Explanation results
        """
        pass
    
    def explain_instance(self, X, instance_index):
        """
        Generate an explanation for a specific instance.
        
        Args:
            X: Input data that contains the instance
            instance_index (int): Index of the instance to explain
            
        Returns:
            dict: Explanation results for the instance
        """
        # Get explanation for the instance
        explanation = self.explain(X, instance_index)
        
        return explanation
    
    def explain_global(self, X):
        """
        Generate a global explanation for the model.
        
        Args:
            X: Input data to explain
            
        Returns:
            dict: Global explanation results
        """
        # Get explanation for all instances
        explanation = self.explain(X)
        
        return explanation
    
    def get_feature_importance(self, explanation=None):
        """
        Get feature importance from the explanation.
        
        Args:
            explanation (dict, optional): Explanation from the explain method
            
        Returns:
            dict: Feature importance scores
        """
        raise NotImplementedError("Feature importance not implemented for this explainer")