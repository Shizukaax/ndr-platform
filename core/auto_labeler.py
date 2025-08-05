"""
Auto-labeling module for the Network Anomaly Detection Platform.
Suggests labels for anomalies based on historical feedback.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from core.feedback_manager import FeedbackManager

class AutoLabeler:
    """Suggests labels for anomalies based on historical feedback."""
    
    def __init__(self, feedback_manager=None):
        """
        Initialize the auto labeler.
        
        Args:
            feedback_manager (FeedbackManager, optional): Feedback manager instance
        """
        self.feedback_manager = feedback_manager or FeedbackManager()
        self.models = {}
        self.feature_columns = None
        self.scaler = StandardScaler()
        self.min_samples = 10  # Minimum samples required for training
    
    def train_models(self, force_retrain=False):
        """
        Train classification models based on feedback data.
        
        Args:
            force_retrain (bool): Force retraining even if models exist
        
        Returns:
            dict: Training results including accuracy metrics
        """
        # Get feedback data
        feedback_df = self.feedback_manager.get_feedback_dataframe()
        
        # If no feedback data, return empty results
        if feedback_df.empty:
            return {"status": "error", "message": "No feedback data available for training"}
        
        # Convert feedback data structure to usable format
        training_data = []
        for _, row in feedback_df.iterrows():
            # Extract feedback information
            feedback_info = {
                'classification': row.get('classification'),
                'priority': row.get('priority'),
                'technique': row.get('technique'),
                'action_taken': row.get('action_taken'),
                'anomaly_score': row.get('anomaly_score', 0)
            }
            
            # Create binary labels for different classification types
            feedback_info['is_true_positive'] = row.get('classification') == 'True Positive'
            feedback_info['is_high_priority'] = row.get('priority') in ['Critical', 'High']
            
            training_data.append(feedback_info)
        
        # Convert to DataFrame for training
        training_df = pd.DataFrame(training_data)
        
        # Check if we have enough samples
        if len(training_df) < self.min_samples:
            return {
                "status": "error", 
                "message": f"Not enough feedback samples for training. Need at least {str(self.min_samples)}, have {str(len(training_df))}"
            }
        
        # Use anomaly scores as features for now (can be enhanced)
        X = training_df[['anomaly_score']].values.reshape(-1, 1)
        self.feature_columns = ['anomaly_score']
        
        # Train models for different classification tasks
        training_results = {}
        
        # 1. True/False Positive Classification
        if 'is_true_positive' in training_df.columns:
            y_tp = training_df['is_true_positive'].astype(bool)
            if len(np.unique(y_tp)) > 1:  # Need both positive and negative examples
                tp_results = self._train_classifier('true_positive', X, y_tp, force_retrain)
                training_results['true_positive'] = tp_results
        
        # 2. Priority Classification
        if 'is_high_priority' in training_df.columns:
            y_priority = training_df['is_high_priority'].astype(bool)
            if len(np.unique(y_priority)) > 1:
                priority_results = self._train_classifier('priority', X, y_priority, force_retrain)
                training_results['priority'] = priority_results
        
        # 3. Classification Type
        if 'classification' in training_df.columns:
            valid_classifications = training_df['classification'].notna()
            if valid_classifications.sum() > self.min_samples:
                y_class = training_df.loc[valid_classifications, 'classification']
                X_class = X[valid_classifications.values]
                if len(np.unique(y_class)) > 1:
                    class_results = self._train_classifier('classification', X_class, y_class, force_retrain)
                    training_results['classification'] = class_results
        
        return {
            "status": "success",
            "results": training_results
        }
        
        # Check if we have enough samples
        if len(X) < self.min_samples:
            return {
                "status": "error", 
                "message": f"Not enough samples for training. Need at least {self.min_samples}, have {len(X)}"
            }
        
        # Extract labels from feedback
        # We'll train models for various label types
        training_results = {}
        
        # 1. True/False Positive Classification
        if 'is_true_positive' in feedback_df.columns:
            y_tp = feedback_df['is_true_positive'].astype(bool)
            if len(np.unique(y_tp)) > 1:  # Need both positive and negative examples
                tp_results = self._train_classifier('true_positive', X, y_tp, force_retrain)
                training_results['true_positive'] = tp_results
        
        # 2. Category Classification
        if 'category' in feedback_df.columns:
            # Remove None/NaN values
            valid_categories = feedback_df['category'].notna()
            if valid_categories.sum() > self.min_samples:
                y_cat = feedback_df.loc[valid_categories, 'category']
                if len(np.unique(y_cat)) > 1:  # Need multiple categories
                    cat_results = self._train_classifier('category', X.loc[valid_categories], y_cat, force_retrain)
                    training_results['category'] = cat_results
        
        # 3. Severity Classification
        if 'severity' in feedback_df.columns:
            valid_severity = feedback_df['severity'].notna()
            if valid_severity.sum() > self.min_samples:
                y_sev = feedback_df.loc[valid_severity, 'severity']
                if len(np.unique(y_sev)) > 1:  # Need multiple severity levels
                    sev_results = self._train_classifier('severity', X.loc[valid_severity], y_sev, force_retrain)
                    training_results['severity'] = sev_results
        
        # Return training results
        if training_results:
            return {"status": "success", "results": training_results}
        else:
            return {"status": "error", "message": "Could not train any models"}
    
    def _train_classifier(self, label_type, X, y, force_retrain):
        """
        Train a classifier for a specific label type.
        
        Args:
            label_type (str): Type of label ('true_positive', 'category', 'severity')
            X (pd.DataFrame): Feature data
            y (pd.Series): Labels
            force_retrain (bool): Force retraining even if model exists
            
        Returns:
            dict: Training results
        """
        # Check if model already exists and we're not forcing a retrain
        if label_type in self.models and not force_retrain:
            # Update existing model with new data instead of creating from scratch
            existing_model_info = self.models[label_type]
            
            # Get existing training count
            training_count = existing_model_info.get('training_count', 0) + 1
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data for training and testing
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Retrain existing models with new data
            rf_model = existing_model_info['rf_model']
            knn_model = existing_model_info['knn_model']
            
            # Retrain models
            rf_model.fit(X_train, y_train)
            knn_model.fit(X_train, y_train)
            
            # Evaluate updated models
            rf_pred = rf_model.predict(X_test)
            knn_pred = knn_model.predict(X_test)
            
            # Calculate accuracies
            rf_accuracy = accuracy_score(y_test, rf_pred)
            knn_accuracy = accuracy_score(y_test, knn_pred)
            
            # Choose the better model
            if rf_accuracy >= knn_accuracy:
                best_model = rf_model
                best_accuracy = rf_accuracy
                model_type = "RandomForest"
            else:
                best_model = knn_model
                best_accuracy = knn_accuracy
                model_type = "KNeighbors"
            
            # Update the stored model info
            self.models[label_type] = {
                'rf_model': rf_model,
                'knn_model': knn_model,
                'best_model': best_model,
                'model_type': model_type,
                'accuracy': best_accuracy,
                'features': list(X.columns),
                'training_samples': len(X),
                'training_count': training_count,
                'last_updated': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return {
                "model_type": model_type,
                "accuracy": best_accuracy,
                "features": list(X.columns),
                "training_samples": len(X),
                "training_count": training_count,
                "status": "updated"
            }
        
        # Create new model if none exists or force_retrain is True
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train RandomForestClassifier
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        # Train KNeighborsClassifier
        knn_model = KNeighborsClassifier()
        knn_model.fit(X_train, y_train)
        knn_pred = knn_model.predict(X_test)
        
        # Calculate metrics
        rf_accuracy = accuracy_score(y_test, rf_pred)
        knn_accuracy = accuracy_score(y_test, knn_pred)
        
        # Choose best model
        if rf_accuracy >= knn_accuracy:
            best_model = rf_model
            best_accuracy = rf_accuracy
            model_type = "RandomForest"
        else:
            best_model = knn_model
            best_accuracy = knn_accuracy
            model_type = "KNeighbors"
        
        # Store the model
        self.models[label_type] = {
            "model": best_model,
            "accuracy": best_accuracy,
            "type": model_type,
            "feature_columns": self.feature_columns
        }
        
        # Return training results
        return {
            "accuracy": best_accuracy,
            "model_type": model_type,
            "features": self.feature_columns,
            "training_samples": len(X_train)
        }
    
    def predict_labels(self, anomalies):
        """
        Predict labels for anomalies based on trained models.
        
        Args:
            anomalies (pd.DataFrame): Anomalies to label
        
        Returns:
            pd.DataFrame: Anomalies with predicted labels
        """
        # If no models trained, return original anomalies
        if not self.models:
            training_result = self.train_models()
            if training_result["status"] == "error":
                return anomalies
        
        # Make a copy of the anomalies DataFrame
        labeled_anomalies = anomalies.copy()
        
        # Check if we have the required feature columns
        if self.feature_columns is None or not all(col in anomalies.columns for col in self.feature_columns):
            return anomalies
        
        # Extract features
        X = anomalies[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions for each label type
        for label_type, model_info in self.models.items():
            model = model_info["model"]
            
            # Make predictions
            predictions = model.predict(X_scaled)
            
            # Add predictions to DataFrame
            if label_type == 'true_positive':
                labeled_anomalies['predicted_true_positive'] = predictions
            elif label_type == 'category':
                labeled_anomalies['predicted_category'] = predictions
            elif label_type == 'severity':
                labeled_anomalies['predicted_severity'] = predictions
            
            # Add prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(X_scaled)
                # Get the probability of the predicted class
                if len(probas.shape) > 1 and probas.shape[1] > 1:
                    # For multiclass, get probability of predicted class
                    max_proba = np.max(probas, axis=1)
                    labeled_anomalies[f'predicted_{label_type}_confidence'] = max_proba
                else:
                    # For binary, get probability of positive class
                    labeled_anomalies[f'predicted_{label_type}_confidence'] = probas[:, 1]
        
        return labeled_anomalies
    
    def get_model_info(self):
        """
        Get information about trained models.
        
        Returns:
            dict: Model information
        """
        model_info = {}
        
        for label_type, model_data in self.models.items():
            model_info[label_type] = {
                "accuracy": model_data["accuracy"],
                "model_type": model_data["type"],
                "features": model_data["feature_columns"]
            }
        
        return model_info
    
    def suggest_labels(self, anomaly):
        """
        Suggest labels for a single anomaly.
        
        Args:
            anomaly (pd.Series or dict): Anomaly data
        
        Returns:
            dict: Suggested labels with confidence scores
        """
        # Convert to DataFrame if needed
        if isinstance(anomaly, pd.Series):
            anomaly_df = pd.DataFrame([anomaly])
        elif isinstance(anomaly, dict):
            anomaly_df = pd.DataFrame([anomaly])
        else:
            return {}
        
        # Get predictions
        predictions = self.predict_labels(anomaly_df)
        
        # Extract predictions for the single anomaly
        suggestions = {}
        
        if 'predicted_true_positive' in predictions.columns:
            suggestions['is_true_positive'] = bool(predictions['predicted_true_positive'].iloc[0])
            if 'predicted_true_positive_confidence' in predictions.columns:
                suggestions['is_true_positive_confidence'] = float(predictions['predicted_true_positive_confidence'].iloc[0])
        
        if 'predicted_category' in predictions.columns:
            suggestions['category'] = predictions['predicted_category'].iloc[0]
            if 'predicted_category_confidence' in predictions.columns:
                suggestions['category_confidence'] = float(predictions['predicted_category_confidence'].iloc[0])
        
        if 'predicted_severity' in predictions.columns:
            suggestions['severity'] = predictions['predicted_severity'].iloc[0]
            if 'predicted_severity_confidence' in predictions.columns:
                suggestions['severity_confidence'] = float(predictions['predicted_severity_confidence'].iloc[0])
        
        return suggestions