import time
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, List, Any
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
import joblib
import json
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from base_model import MLModel

class GaussianNaiveBayesModel(MLModel):
    """Gaussian Naive Bayes model implementation."""
    
    def __init__(self,
                 model_name: str = "gaussian_naive_bayes",
                 var_smoothing: float = 1e-9,
                 priors: Optional[List[float]] = None,
                 tracking_uri: Optional[str] = None,
                 registry_uri: Optional[str] = None,
                 experiment_name: Optional[str] = None,
                 **kwargs):
        """
        Initialize Gaussian Naive Bayes model.
        
        Args:
            model_name: Name of the model
            var_smoothing: Portion of the largest variance of all features that is added to variances
            priors: Prior probabilities of the classes
            tracking_uri: MLflow tracking server URI
            registry_uri: MLflow model registry URI
            experiment_name: MLflow experiment name
            **kwargs: Additional parameters for the model
        """
        if experiment_name is None:
            experiment_name = "naive_bayes_models"
            
        super().__init__(model_name, "classification", tracking_uri, registry_uri, experiment_name)
        
        self.var_smoothing = var_smoothing
        self.priors = priors
        
        # Store parameters
        self.params = {
            "var_smoothing": var_smoothing,
            "priors": priors
        }
        
        # Add additional parameters
        self.params.update(kwargs)
        
        # Initialize model
        self.model = GaussianNB(
            var_smoothing=var_smoothing,
            priors=priors,
            **{k: v for k, v in kwargs.items() if k not in self.params}
        )
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], 
            validation_data: Optional[tuple] = None, log_to_mlflow: bool = True, **kwargs) -> Dict[str, float]:
        """
        Fit the Gaussian Naive Bayes model.
        
        Args:
            X: Training features
            y: Training target
            validation_data: Optional tuple of (X_val, y_val) for validation
            log_to_mlflow: Whether to log metrics and model to MLflow
            **kwargs: Additional parameters to pass to the fit method
            
        Returns:
            Dictionary with training metrics
        """
        self.logger.info(f"Fitting Gaussian Naive Bayes model with {X.shape[0]} samples")
        
        # Store feature names for future reference
        self.feature_names = X.columns.tolist() if hasattr(X, "columns") else None
        
        # Train model
        start_time = time.time()
        self.model.fit(X, y, **kwargs)
        train_time = time.time() - start_time
        
        self.logger.info(f"Model training completed in {train_time:.2f} seconds")
        
        # Calculate training metrics
        train_metrics = self._calculate_metrics(X, y)
        train_metrics["training_time_seconds"] = train_time
        
# Calculate validation metrics if validation data is provided
if validation_data is not None:
    X_val, y_val = validation_data
    val_metrics = self._calculate_metrics(X_val, y_val)
    train_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

# Log metrics to MLflow
if log_to_mlflow:
    self._log_to_mlflow(train_metrics)


if validation_data is not None:
    X_val, y_val = validation_data
    val_metrics = self._calculate_metrics(X_val, y_val)
    train_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

return train_metrics

from base_model import MLModel

class GaussianNaiveBayesModel(MLModel):
    """Gaussian Naive Bayes model implementation."""
    
    def __init__(self,
                 model_name: str = "gaussian_naive_bayes",
                 var_smoothing: float = 1e-9,
                 priors: Optional[List[float]] = None,
                 tracking_uri: Optional[str] = None,
                 registry_uri: Optional[str] = None,
                 experiment_name: Optional[str] = None,
                 **kwargs):
        """
        Initialize Gaussian Naive Bayes model.
        
        Args:
            model_name: Name of the model
            var_smoothing: Portion of the largest variance of all features that is added to variances
            priors: Prior probabilities of the classes
            tracking_uri: MLflow tracking server URI
            registry_uri: MLflow model registry URI
            experiment_name: MLflow experiment name
            **kwargs: Additional parameters for the model
        """
        if experiment_name is None:
            experiment_name = "naive_bayes_models"
            
        super().__init__(model_name, "classification", tracking_uri, registry_uri, experiment_name)
        
        self.var_smoothing = var_smoothing
        self.priors = priors
        
        # Store parameters
        self.params = {
            "var_smoothing": var_smoothing,
            "priors": priors
        }
        
        # Add additional parameters
        self.params.update(kwargs)
        
        # Initialize model
        self.model = GaussianNB(
            var_smoothing=var_smoothing,
            priors=priors,
            **{k: v for k, v in kwargs.items() if k not in self.params}
        )
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], 
            validation_data: Optional[tuple] = None, log_to_mlflow: bool = True, **kwargs) -> Dict[str, float]:
        """
        Fit the Gaussian Naive Bayes model.
        
        Args:
            X: Training features
            y: Training target
            validation_data: Optional tuple of (X_val, y_val) for validation
            log_to_mlflow: Whether to log metrics and model to MLflow
            **kwargs: Additional parameters to pass to the fit method
            
        Returns:
            Dictionary with training metrics
        """
        self.logger.info(f"Fitting Gaussian Naive Bayes model with {X.shape[0]} samples")
        
        # Store feature names for future reference
        self.feature_names = X.columns.tolist() if hasattr(X, "columns") else None
        
        # Train model
        start_time = time.time()
        self.model.fit(X, y, **kwargs)
        train_time = time.time() - start_time
        
        self.logger.info(f"Model training completed in {train_time:.2f} seconds")
        
        # Calculate training metrics
        train_metrics = self._calculate_metrics(X, y)
        train_metrics["training_time_seconds"] = train_time
        
        # Calculate validation metrics if validation data is provided
        if validation_