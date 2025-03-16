import time
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, List, Any, Tuple
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib
import mlflow
import mlflow.sklearn

from base_model import MLModel

class RandomForestModel(MLModel):
    """Random Forest model implementation supporting both regression and classification."""
    
    def __init__(self,
                 model_name: str = "random_forest",
                 model_type: str = "classification",
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Optional[Union[str, float]] = "sqrt",
                 bootstrap: bool = True,
                 random_state: int = 42,
                 tracking_uri: Optional[str] = None,
                 registry_uri: Optional[str] = None,
                 experiment_name: Optional[str] = None):
        """
        Initialize Random Forest model.
        
        Args:
            model_name: Name of the model
            model_type: Type of task ('classification' or 'regression')
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            max_features: Number of features to consider for best split
            bootstrap: Whether to use bootstrap samples
            random_state: Random seed for reproducibility
            tracking_uri: MLflow tracking server URI
            registry_uri: MLflow model registry URI
            experiment_name: MLflow experiment name
        """
        if model_type not in ["classification", "regression"]:
            raise ValueError("model_type must be either 'classification' or 'regression'")
            
        if experiment_name is None:
            experiment_name = "classification_models" if model_type == "classification" else "regression_models"
            
        super().__init__(model_name, model_type, tracking_uri, registry_uri, experiment_name)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        
        # Store parameters
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "bootstrap": bootstrap,
            "random_state": random_state
        }
        
        # Initialize the model based on the task type
        if model_type == "classification":
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                bootstrap=bootstrap,
                random_state=random_state,
                n_jobs=-1
            )
        else:  # regression
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                bootstrap=bootstrap,
                random_state=random_state,
                n_jobs=-1
            )
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, float]:
        """
        Train the Random Forest model.
        
        Args:
            X: Features dataframe
            y: Target series
            **kwargs: Additional training parameters
                - cv: Number of cross-validation folds (default: 5)
                - tune_hyperparams: Whether to tune hyperparameters (default: False)
                - param_grid: Parameter grid for hyperparameter tuning
                - search_method: 'grid' or 'random' for hyperparameter tuning method
                - n_iter: Number of iterations for random search
                
        Returns:
            Dictionary of training metrics
        """
        start_time = time.time()
        
        # Extract training parameters
        cv = kwargs.get('cv', 5)
        tune_hyperparams = kwargs.get('tune_hyperparams', False)
        
        if tune_hyperparams:
            # Extract hyperparameter tuning parameters
            search_method = kwargs.get('search_method', 'grid')
            n_iter = kwargs.get('n_iter', 20)
            
            # Define default parameter grid if not provided
            param_grid = kwargs.get('param_grid', {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            })
            
            if search_method == 'grid':
                # Use grid search
                search = GridSearchCV(
                    self.model,
                    param_grid,
                    cv=cv,
                    scoring='accuracy' if self.model_type == 'classification' else 'neg_mean_squared_error',
                    n_jobs=-1
                )
            else:
                # Use random search
                search = RandomizedSearchCV(
                    self.model,
                    param_grid,
                    n_iter=n_iter,
                    cv=cv,
                    scoring='accuracy' if self.model_type == 'classification' else 'neg_mean_squared_error',
                    random_state=self.random_state,
                    n_jobs=-1
                )
            
            # Fit search
            search.fit(X, y)
            
            # Update model with best estimator
            self.model = search.best_estimator_
            
            # Update parameters
            self.params.update(search.best_params_)
            
            # Store best score
            if self.model_type == 'classification':
                self.metrics['best_cv_accuracy'] = search.best_score_
            else:
                best_mse = -search.best_score_  # Convert back from negative MSE
                self.metrics['best_cv_mse'] = best_mse
                self.metrics['best_cv_rmse'] = np.sqrt(best_mse)
        else:
            # Train without hyperparameter tuning
            self.model.fit(X, y)
            
            # Validate model
            metrics = self._validate_model(X, y, cv=cv)
            self.metrics.update(metrics)
        
        # Extract feature importances
        feature_names = list(X.columns)
        self.feature_importances = self.model.feature_importances_
        
        # Store top feature importances in metrics
        feature_imp_dict = self.get_feature_importances(feature_names)
        sorted_features = sorted(feature_imp_dict.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, imp) in enumerate(sorted_features[:10]):  # Store top 10 features
            self.metrics[f"feature_imp_{i+1}_{feature}"] = imp
        
        # Record training time
        self.training_time = time.time() - start_time
        self.is_trained = True
        
        # Calculate out-of-bag score if bootstrapping is enabled
        if self.bootstrap:
            self.metrics['oob_score'] = self.model.oob_score_ if hasattr(self.model, 'oob_score_') else None
        
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
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate class probabilities for classification.
        
        Args:
            X: Features dataframe
            
        Returns:
            Class probabilities as numpy array
            
        Raises:
            ValueError: If model is not a classifier or not trained
        """
        if self.model_type != "classification":
            raise ValueError("predict_proba is only available for classification models")
            
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict_proba(X)
    
    def feature_importance_analysis(self, feature_names: List[str], top_n: int = 10) -> Dict[str, float]:
        """
        Analyze and return the top feature importances.
        
        Args:
            feature_names: Names of features
            top_n: Number of top features to return
            
        Returns:
            Dictionary with top feature importances
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before analyzing feature importances")
        
        importances = self.feature_importances
        indices = np.argsort(importances)[::-1]
        
        top_features = {}
        for i in range(min(top_n, len(feature_names))):
            idx = indices[i]
            top_features[feature_names[idx]] = importances[idx]
        
        return top_features