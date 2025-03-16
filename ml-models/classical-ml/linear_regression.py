import time
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, List, Any
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from base_model import MLModel

class LinearRegressionModel(MLModel):
    """Linear Regression model implementation."""
    
    def __init__(self,
                 model_name: str = "linear_regression",
                 regularization: Optional[str] = None,
                 alpha: float = 1.0,
                 l1_ratio: float = 0.5,
                 tracking_uri: Optional[str] = None,
                 registry_uri: Optional[str] = None,
                 experiment_name: str = "regression_models"):
        """
        Initialize Linear Regression model.
        
        Args:
            model_name: Name of the model
            regularization: Type of regularization (None, 'l1', 'l2', 'elasticnet')
            alpha: Regularization strength
            l1_ratio: ElasticNet mixing parameter (0 <= l1_ratio <= 1)
            tracking_uri: MLflow tracking server URI
            registry_uri: MLflow model registry URI
            experiment_name: MLflow experiment name
        """
        super().__init__(model_name, "regression", tracking_uri, registry_uri, experiment_name)
        
        self.regularization = regularization
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.params = {
            "regularization": regularization,
            "alpha": alpha,
            "l1_ratio": l1_ratio
        }
        
        # Configure linear regression model based on regularization type
        if regularization is None:
            regressor = LinearRegression()
        elif regularization == 'l1':
            regressor = Lasso(alpha=alpha, random_state=42)
        elif regularization == 'l2':
            regressor = Ridge(alpha=alpha, random_state=42)
        elif regularization == 'elasticnet':
            regressor = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        else:
            raise ValueError(f"Unsupported regularization type: {regularization}")
            
        # Create pipeline with scaling
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', regressor)
        ])
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, float]:
        """
        Train the linear regression model.
        
        Args:
            X: Features dataframe
            y: Target series
            **kwargs: Additional training parameters
                - cv: Number of cross-validation folds (default: 5)
                - tune_hyperparams: Whether to tune hyperparameters (default: False)
                - param_grid: Parameter grid for hyperparameter tuning
                
        Returns:
            Dictionary of training metrics
        """
        start_time = time.time()
        
        # Extract training parameters
        cv = kwargs.get('cv', 5)
        tune_hyperparams = kwargs.get('tune_hyperparams', False)
        
        if tune_hyperparams:
            # Define default parameter grid if not provided
            param_grid = kwargs.get('param_grid', {
                'regressor__alpha': [0.01, 0.1, 1.0, 10.0],
            })
            
            # Add l1_ratio to param_grid if using ElasticNet
            if self.regularization == 'elasticnet':
                param_grid['regressor__l1_ratio'] = [0.1, 0.3, 0.5, 0.7, 0.9]
                
            # Create grid search CV
            grid_search = GridSearchCV(
                self.model,
                param_grid,
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            # Fit grid search
            grid_search.fit(X, y)
            
            # Update model with best estimator
            self.model = grid_search.best_estimator_
            
            # Update parameters
            best_params = {k.replace('regressor__', ''): v for k, v in grid_search.best_params_.items()}
            self.params.update(best_params)
            
            # Get best score
            best_score = -grid_search.best_score_  # Convert back to MSE
            self.metrics['cv_mse'] = best_score
            self.metrics['cv_rmse'] = np.sqrt(best_score)
        else:
            # Train without hyperparameter tuning
            self.model.fit(X, y)
            
            # Validate model
            metrics = self._validate_model(X, y, cv=cv)
            self.metrics.update(metrics)
        
        # Record training time
        self.training_time = time.time() - start_time
        self.is_trained = True
        
        # Extract coefficients and intercept
        coef = self.model.named_steps['regressor'].coef_
        intercept = self.model.named_steps['regressor'].intercept_
        
        # Store feature importances as absolute coefficients
        self.feature_importances = np.abs(coef)
        
        # Add coefficient info to metrics
        self.metrics['intercept'] = float(intercept)
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the trained model.
        
        Args:
            X: Features dataframe
            
        Returns:
            Predictions as numpy array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)