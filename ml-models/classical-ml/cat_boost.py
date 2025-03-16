import time
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, List, Any
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib
import mlflow
import mlflow.catboost
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt

from base_model import MLModel

class CatBoostModel(MLModel):
    """CatBoost model implementation."""
    
    def __init__(self,
                 model_name: str = "catboost",
                 model_type: str = "classification",
                 iterations: int = 500,
                 learning_rate: float = 0.1,
                 depth: int = 6,
                 l2_leaf_reg: float = 3.0,
                 random_state: int = 42,
                 cat_features: Optional[List[int]] = None,
                 verbose: bool = True,
                 tracking_uri: Optional[str] = None,
                 registry_uri: Optional[str] = None,
                 experiment_name: Optional[str] = None,
                 **kwargs):
        """
        Initialize CatBoost model.
        
        Args:
            model_name: Name of the model
            model_type: Type of task ('classification' or 'regression')
            iterations: Number of boosting iterations
            learning_rate: Learning rate
            depth: Depth of the tree
            l2_leaf_reg: L2 regularization coefficient
            random_state: Random seed for reproducibility
            cat_features: List of categorical feature indices
            verbose: Whether to show progress during training
            tracking_uri: MLflow tracking server URI
            registry_uri: MLflow model registry URI
            experiment_name: MLflow experiment name
            **kwargs: Additional parameters for the model
        """
        if model_type not in ["classification", "regression"]:
            raise ValueError("model_type must be either 'classification' or 'regression'")
        
        if experiment_name is None:
            experiment_name = f"catboost_{model_type}_models"
            
        super().__init__(model_name, model_type, tracking_uri, registry_uri, experiment_name)
        
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_state = random_state
        self.cat_features = cat_features
        self.verbose = 10 if verbose else 0
        
        # Store parameters
        self.params = {
            "iterations": iterations,
            "learning_rate": learning_rate,
            "depth": depth,
            "l2_leaf_reg": l2_leaf_reg,
            "random_seed": random_state,
        }
        
        # Add additional parameters
        self.params.update(kwargs)
        
        # Initialize the model based on task type
        if model_type == "classification":
            self.model = CatBoostClassifier(
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                l2_leaf_reg=l2_leaf_reg,
                random_seed=random_state,
                verbose=self.verbose,
                **{k: v for k, v in kwargs.items() if k not in self.params}
            )
        else:  # regression
            self.model = CatBoostRegressor(
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                l2_leaf_reg=l2_leaf_reg,
                random_seed=random_state,
                verbose=self.verbose,
                **{k: v for k, v in kwargs.items() if k not in self.params}
            )

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], 
            validation_data: Optional[tuple] = None, log_to_mlflow: bool = True, 
            early_stopping_rounds: Optional[int] = 20, **kwargs) -> Dict[str, float]:
        """
        Fit the CatBoost model.
        
        Args:
            X: Training features
            y: Training target
            validation_data: Optional tuple of (X_val, y_val) for validation
            log_to_mlflow: Whether to log metrics and model to MLflow
            early_stopping_rounds: Number of rounds for early stopping
            **kwargs: Additional parameters to pass to the fit method
            
        Returns:
            Dictionary with training metrics
        """
        self.logger.info(f"Fitting CatBoost {self.model_type} model with {X.shape[0]} samples")
        
        # Store feature names for feature importance
        self.feature_names = X.columns.tolist() if hasattr(X, "columns") else None
        
        # Configure validation data for early stopping
        fit_params = {}
        
        # Create CatBoost Pool objects
        train_pool = Pool(X, y, cat_features=self.cat_features)
        
        if validation_data is not None:
            X_val, y_val = validation_data
            eval_pool = Pool(X_val, y_val, cat_features=self.cat_features)
            fit_params["eval_set"] = eval_pool
            
            if early_stopping_rounds:
                if self.model_type == "classification":
                    fit_params["early_stopping_rounds"] = early_stopping_rounds
                    fit_params["metric_period"] = kwargs.get("metric_period", 10)
                else:  # regression
                    fit_params["early_stopping_rounds"] = early_stopping_rounds
                    fit_params["metric_period"] = kwargs.get("metric_period", 10)
        
        # Merge with any additional fit parameters
        for k, v in kwargs.items():
            if k not in fit_params:
                fit_params[k] = v
        
        # Train model
        start_time = time.time()
        self.model.fit(train_pool, **fit_params)
        train_time = time.time() - start_time
        
        self.logger.info(f"Model training completed in {train_time:.2f} seconds")
        
        # Calculate training metrics
        train_metrics = self._calculate_metrics(X, y)
        train_metrics["training_time_seconds"] = train_time
        
        # Calculate validation metrics if validation data is provided
        if validation_data is not None:
            X_val, y_val = validation_data
            val_metrics = self._calculate_metrics(X_val, y_val, prefix="val_")
            train_metrics.update(val_metrics)
            
            if early_stopping_rounds:
                train_metrics["best_iteration"] = self.model.get_best_iteration()
        
        # Log to MLflow if enabled
        if log_to_mlflow:
            with mlflow.start_run(run_name=f"{self.model_name}_training") as run:
                # Log parameters
                mlflow.log_params(self.params)
                
                # Log metrics
                for metric_name, metric_value in train_metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log model
                mlflow.catboost.log_model(self.model, "model")
                
                # Log feature importance
                if self.feature_names is not None:
                    feature_importance = self.get_feature_importance(self.feature_names)
                    feature_importance.to_csv("feature_importance.csv", index=False)
                    mlflow.log_artifact("feature_importance.csv")
                    
                    # Visualize feature importance
                    self._log_feature_importance_plot(feature_importance, mlflow)
                
                self.logger.info(f"Training metrics and model logged to MLflow run: {run.info.run_id}")
        
        return train_metrics
    
    def _log_feature_importance_plot(self, feature_importance: pd.DataFrame, mlflow_client):
        """
        Create and log feature importance visualization to MLflow.
        
        Args:
            feature_importance: DataFrame with feature importance
            mlflow_client: MLflow client
        """
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['Feature'][:20], feature_importance['Importance'][:20])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")
        mlflow_client.log_artifact("feature_importance.png")
        plt.close()
    
    def _calculate_metrics(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], 
                          prefix: str = "") -> Dict[str, float]:
        """
        Calculate model performance metrics.
        
        Args:
            X: Features
            y: True labels/values
            prefix: Prefix for metric names (e.g., 'val_' for validation metrics)
            
        Returns:
            Dictionary with metrics
        """
        y_pred = self.predict(X)
        metrics = {}
        
        if self.model_type == "classification":
            y_pred_proba = self.predict_proba(X)
            
            # Classification metrics
            metrics[f"{prefix}accuracy"] = accuracy_score(y, y_pred)
            metrics[f"{prefix}precision"] = precision_score(y, y_pred, average='weighted')
            metrics[f"{prefix}recall"] = recall_score(y, y_pred, average='weighted')
            metrics[f"{prefix}f1"] = f1_score(y, y_pred, average='weighted')
            
            # For binary classification, add ROC-AUC
            if len(np.unique(y)) == 2:
                metrics[f"{prefix}roc_auc"] = roc_auc_score(y, y_pred_proba[:, 1])
        else:
            # Regression metrics
            metrics[f"{prefix}mse"] = mean_squared_error(y, y_pred)
            metrics[f"{prefix}rmse"] = np.sqrt(metrics[f"{prefix}mse"])
            metrics[f"{prefix}r2"] = r2_score(y, y_pred)
            
        return metrics
        
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        return self.model.predict(X)
        
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make probability predictions with the model.
        
        Args:
            X: Features
            
        Returns:
            Probability predictions
        """
        if self.model_type != "classification":
            raise ValueError("predict_proba is only available for classification models")
            
        return self.model.predict_proba(X)
        
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get feature importance from the model.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if feature_names is None:
            feature_names = self.feature_names
            
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.model.feature_importances_))]
            
        # Get feature importance
        importance = self.model.get_feature_importance()
        
        # Create DataFrame
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        return feature_importance
        
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
        """
        if not path.endswith('.cbm'):
            path = f"{path}.cbm"
            
        self.model.save_model(path)
        self.logger.info(f"Model saved to {path}")
        
        # Save additional metadata
        metadata = {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'params': self.params,
            'feature_names': self.feature_names,
            'cat_features': self.cat_features
        }
        
        metadata_path = path.replace('.cbm', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
            
        self.logger.info(f"Model metadata saved to {metadata_path}")
        
    @classmethod
    def load(cls, path: str) -> 'CatBoostModel':
        """
        Load model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded model
        """
        if not path.endswith('.cbm'):
            path = f"{path}.cbm"
            
        # Load metadata
        metadata_path = path.replace('.cbm', '_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Create model instance
        model_instance = cls(
            model_name=metadata['model_name'],
            model_type=metadata['model_type'],
            cat_features=metadata['cat_features'],
            **{k: v for k, v in metadata['params'].items() if k != 'random_seed'}
        )
        
        # Load the actual model
        if metadata['model_type'] == 'classification':
            model_instance.model = CatBoostClassifier()
        else:
            model_instance.model = CatBoostRegressor()
            
        model_instance.model.load_model(path)
        model_instance.feature_names = metadata['feature_names']
        
        return model_instance
        
    def tune_hyperparameters(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray],
                             param_grid: Dict[str, List[Any]], cv: int = 5, 
                             scoring: Optional[str] = None, n_jobs: int = -1,
                             search_type: str = 'grid', n_iter: int = 10,
                                                 log_to_mlflow: bool = True) -> Dict[str, Any]:
                            """
                            Tune hyperparameters for the CatBoost model.
                            
                            Args:
                                X: Training features
                                y: Training target
                                param_grid: Dictionary with parameters to search
                                cv: Number of cross-validation folds
                                scoring: Scoring metric for evaluation
                                n_jobs: Number of jobs to run in parallel
                                search_type: Type of search ('grid' or 'random')
                                n_iter: Number of iterations for random search
                                log_to_mlflow: Whether to log results to MLflow
                                
                            Returns:
                                Best parameters found
                            """
                            self.logger.info(f"Starting hyperparameter tuning with {search_type} search")
                            
                            if search_type == 'grid':
                                search = GridSearchCV(self.model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=self.verbose)
                            elif search_type == 'random':
                                search = RandomizedSearchCV(self.model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, n_iter=n_iter, verbose=self.verbose)
                            else:
                                raise ValueError("search_type must be either 'grid' or 'random'")
                            
                            search.fit(X, y)
                            
                            self.logger.info(f"Best parameters found: {search.best_params_}")
                            
                            # Update model with best parameters
                            self.model.set_params(**search.best_params_)
                            
                            # Log to MLflow if enabled
                            if log_to_mlflow:
                                with mlflow.start_run(run_name=f"{self.model_name}_hyperparameter_tuning") as run:
                                    mlflow.log_params(search.best_params_)
                                    self.logger.info(f"Hyperparameter tuning results logged to MLflow run: {run.info.run_id}")
                            
                            return search.best_params_