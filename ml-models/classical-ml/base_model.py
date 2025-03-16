import os
import time
import logging
import joblib
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Union, Optional, Any, List, Tuple
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec

logger = logging.getLogger(__name__)

class MLModel(ABC):
    """Base class for all ML models in the pipeline."""
    
    def __init__(self, 
                 model_name: str, 
                 model_type: str,
                 tracking_uri: Optional[str] = None,
                 registry_uri: Optional[str] = None,
                 experiment_name: str = "ml-models"):
        """
        Initialize base model class.
        
        Args:
            model_name: Name of the model
            model_type: Type of model (regression, classification)
            tracking_uri: MLflow tracking server URI
            registry_uri: MLflow model registry URI
            experiment_name: MLflow experiment name
        """
        self.model_name = model_name
        self.model_type = model_type
        self.model: Optional[BaseEstimator] = None
        self.is_trained = False
        self.training_time = 0
        self.feature_importances = None
        self.metrics = {}
        self.params = {}
        self.experiment_name = experiment_name
        
        # Configure MLflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)
            
        # Get or create experiment
        try:
            self.experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=os.environ.get("MLFLOW_ARTIFACT_PATH", None)
            )
        except:
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, float]:
        """
        Train the model on the provided data.
        
        Args:
            X: Features dataframe
            y: Target series
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for the input data.
        
        Args:
            X: Features dataframe
            
        Returns:
            Predictions as numpy array
        """
        pass
    
    def _validate_model(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, float]:
        """
        Validate model performance using cross-validation.
        
        Args:
            X: Features dataframe
            y: Target series
            **kwargs: Additional validation parameters
            
        Returns:
            Dictionary of validation metrics
        """
        if not self.model:
            raise ValueError("Model must be trained before validation")
            
        cv = kwargs.get('cv', 5)
        scoring = kwargs.get('scoring', ['neg_mean_squared_error', 'r2'] if self.model_type == 'regression' 
                                      else ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
        
        cv_results = cross_validate(self.model, X, y, cv=cv, scoring=scoring, return_train_score=True)
        
        # Extract and format metrics
        metrics = {}
        for metric in scoring:
            metrics[f"val_{metric}"] = float(np.mean(cv_results[f'test_{metric}']))
            metrics[f"train_{metric}"] = float(np.mean(cv_results[f'train_{metric}']))
        
        return metrics
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            X: Features dataframe
            y: Target series
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.model:
            raise ValueError("Model must be trained before evaluation")
            
        y_pred = self.predict(X)
        
        metrics = {}
        if self.model_type == 'regression':
            metrics['mse'] = mean_squared_error(y, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
        else:  # classification
            metrics['accuracy'] = accuracy_score(y, y_pred)
            
            # For binary classification
            if len(np.unique(y)) == 2:
                metrics['precision'] = precision_score(y, y_pred, average='binary')
                metrics['recall'] = recall_score(y, y_pred, average='binary')
                metrics['f1'] = f1_score(y, y_pred, average='binary')
                
                # If predict_proba is available
                if hasattr(self.model, "predict_proba"):
                    y_proba = self.model.predict_proba(X)[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y, y_proba)
            else:
                # Multi-class
                metrics['precision'] = precision_score(y, y_pred, average='weighted')
                metrics['recall'] = recall_score(y, y_pred, average='weighted')
                metrics['f1'] = f1_score(y, y_pred, average='weighted')
        
        self.metrics.update(metrics)
        return metrics
    
    def save(self, path: str) -> str:
        """
        Save model to disk.
        
        Args:
            path: Directory path to save the model
            
        Returns:
            Full path to saved model
        """
        if not self.model:
            raise ValueError("Model must be trained before saving")
            
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, f"{self.model_name}.joblib")
        joblib.dump(self.model, model_path)
        
        # Save metadata
        metadata = {
            "name": self.model_name,
            "type": self.model_type,
            "is_trained": self.is_trained,
            "training_time": self.training_time,
            "metrics": self.metrics,
            "params": self.params,
            "feature_importances": self.feature_importances
        }
        
        metadata_path = os.path.join(path, f"{self.model_name}_metadata.joblib")
        joblib.dump(metadata, metadata_path)
        
        return model_path
    
    def load(self, path: str) -> None:
        """
        Load model from disk.
        
        Args:
            path: Path to model file or directory
        """
        if os.path.isdir(path):
            model_path = os.path.join(path, f"{self.model_name}.joblib")
            metadata_path = os.path.join(path, f"{self.model_name}_metadata.joblib")
        else:
            model_path = path
            metadata_path = f"{os.path.splitext(path)[0]}_metadata.joblib"
        
        self.model = joblib.load(model_path)
        
        # Load metadata if available
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.is_trained = metadata.get("is_trained", True)
            self.training_time = metadata.get("training_time", 0)
            self.metrics = metadata.get("metrics", {})
            self.params = metadata.get("params", {})
            self.feature_importances = metadata.get("feature_importances", None)
    
    def log_to_mlflow(self, X: pd.DataFrame, run_name: Optional[str] = None) -> str:
        """
        Log model and metrics to MLflow.
        
        Args:
            X: Sample dataset for signature inference
            run_name: Custom run name
            
        Returns:
            MLflow run ID
        """
        if not self.model:
            raise ValueError("Model must be trained before logging to MLflow")
        
        # Start MLflow run
        run_name = run_name or f"{self.model_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name, experiment_id=self.experiment_id) as run:
            # Log model parameters
            mlflow.log_params(self.params)
            
            # Log metrics
            mlflow.log_metrics(self.metrics)
            
            # Log feature importances if available
            if self.feature_importances is not None:
                for i, importance in enumerate(self.feature_importances):
                    feature_name = X.columns[i] if i < len(X.columns) else f"feature_{i}"
                    mlflow.log_metric(f"feature_importance_{feature_name}", importance)
            
            # Create model signature
            input_schema = Schema([ColSpec(type_string=str(X[col].dtype), name=col) for col in X.columns])
            output_schema = Schema([ColSpec(type_string="double", name="prediction")])
            signature = ModelSignature(inputs=input_schema, outputs=output_schema)
            
            # Log model
            mlflow.sklearn.log_model(
                self.model,
                "model",
                signature=signature,
                registered_model_name=self.model_name
            )
            
            # Log additional custom artifacts
            mlflow.log_metric("training_time_seconds", self.training_time)
            
            return run.info.run_id
    
    def get_feature_importances(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get feature importances if available.
        
        Args:
            feature_names: Names of features
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.model or not hasattr(self.model, "feature_importances_"):
            return {}
            
        importances = self.model.feature_importances_
        self.feature_importances = importances
        
        if feature_names is None:
            return {f"feature_{i}": imp for i, imp in enumerate(importances)}
        
        return {name: imp for name, imp in zip(feature_names, importances)}