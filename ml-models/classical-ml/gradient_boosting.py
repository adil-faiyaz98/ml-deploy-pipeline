import time
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, List, Any
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import xgboost as xgb
import lightgbm as lgb
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, r2_score

from base_model import MLModel

class GradientBoostingModel(MLModel):
    """Gradient Boosting model implementation supporting multiple implementations."""
    
    def __init__(self,
                 model_name: str = "gradient_boosting",
                 model_type: str = "classification",
                 implementation: str = "sklearn",  # 'sklearn', 'xgboost', 'lightgbm'
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 max_depth: int = 3,
                 subsample: float = 1.0,
                 random_state: int = 42,
                 tracking_uri: Optional[str] = None,
                 registry_uri: Optional[str] = None,
                 experiment_name: Optional[str] = None,
                 **kwargs):
        """
        Initialize Gradient Boosting model.
        
        Args:
            model_name: Name of the model
            model_type: Type of task ('classification' or 'regression')
            implementation: Library to use ('sklearn', 'xgboost', 'lightgbm')
            n_estimators: Number of boosting stages
            learning_rate: Learning rate shrinks the contribution of each tree
            max_depth: Maximum depth of the individual trees
            subsample: Fraction of samples used for fitting the trees
            random_state: Random seed for reproducibility
            tracking_uri: MLflow tracking server URI
            registry_uri: MLflow model registry URI
            experiment_name: MLflow experiment name
            **kwargs: Additional parameters for the specific implementation
        """
        if model_type not in ["classification", "regression"]:
            raise ValueError("model_type must be either 'classification' or 'regression'")
            
        if implementation not in ["sklearn", "xgboost", "lightgbm"]:
            raise ValueError("implementation must be one of 'sklearn', 'xgboost', 'lightgbm'")
            
        if experiment_name is None:
            experiment_name = f"{implementation}_{model_type}_models"
            
        super().__init__(model_name, model_type, tracking_uri, registry_uri, experiment_name)
        
        self.implementation = implementation
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state
        
        # Store base parameters
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "random_state": random_state,
            "implementation": implementation
        }
        
        # Add additional parameters
        self.params.update(kwargs)
        
        # Initialize model based on implementation and task type
        if implementation == "sklearn":
            if model_type == "classification":
                self.model = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    subsample=subsample,
                    random_state=random_state,
                    **{k: v for k, v in kwargs.items() if k not in self.params}
                )
            else:  # regression
                self.model = GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    subsample=subsample,
                    random_state=random_state,
                    **{k: v for k, v in kwargs.items() if k not in self.params}
                )
                
        elif implementation == "xgboost":
            if model_type == "classification":
                self.model = xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    subsample=subsample,
                    random_state=random_state,
                    **{k: v for k, v in kwargs.items() if k not in self.params}
                )
            else:  # regression
                self.model = xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    subsample=subsample,
                    random_state=random_state,
                    **{k: v for k, v in kwargs.items() if k not in self.params}
                )
                
        elif implementation == "lightgbm":
            if model_type == "classification":
                self.model = lgb.LGBMClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    subsample=subsample,
                    random_state=random_state,
                    **{k: v for k, v in kwargs.items() if k not in self.params}
                )
            else:  # regression
                self.model = lgb.LGBMRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    subsample=subsample,
                    random_state=random_state,
                    **{k: v for k, v in kwargs.items() if k not in self.params}
                )
                
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], 
            validation_data: Optional[tuple] = None, log_to_mlflow: bool = True, 
            early_stopping_rounds: Optional[int] = None, **kwargs) -> Dict[str, float]:
        """
        Fit the Gradient Boosting model.
        
        Args:
            X: Training features
            y: Training target
            validation_data: Optional tuple of (X_val, y_val) for validation
            log_to_mlflow: Whether to log metrics and model to MLflow
            early_stopping_rounds: Number of rounds for early stopping (XGBoost and LightGBM only)
            **kwargs: Additional parameters to pass to the fit method
            
        Returns:
            Dictionary with training metrics
        """
        self.logger.info(f"Fitting {self.implementation} {self.model_type} model with {X.shape[0]} samples")
        
        # Store feature names for feature importance
        self.feature_names = X.columns.tolist() if hasattr(X, "columns") else None
        
        # Configure fit parameters
        fit_params = {}
        
        # Handle early stopping if validation data is provided
        if validation_data is not None and early_stopping_rounds is not None:
            X_val, y_val = validation_data
            
            if self.implementation == "xgboost":
                fit_params["eval_set"] = [(X, y), (X_val, y_val)]
                fit_params["early_stopping_rounds"] = early_stopping_rounds
                fit_params["verbose"] = kwargs.get("verbose", False)
                
            elif self.implementation == "lightgbm":
                fit_params["eval_set"] = [(X_val, y_val)]
                fit_params["early_stopping_rounds"] = early_stopping_rounds
                fit_params["verbose"] = kwargs.get("verbose", False)
        
        # Merge with any additional fit parameters
        for k, v in kwargs.items():
            if k not in fit_params:
                fit_params[k] = v
        
        # Train model
        start_time = time.time()
        self.model.fit(X, y, **fit_params)
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
            
            # Get best iteration for XGBoost and LightGBM
            if early_stopping_rounds is not None:
                if self.implementation == "xgboost":
                    train_metrics["best_iteration"] = self.model.best_iteration
                elif self.implementation == "lightgbm":
                    train_metrics["best_iteration"] = self.model.best_iteration_
        
        # Log to MLflow if enabled
        if log_to_mlflow:
            with mlflow.start_run(run_name=f"{self.model_name}_training") as run:
                # Log parameters
                mlflow.log_params(self.params)
                
                # Log metrics
                for metric_name, metric_value in train_metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log model based on implementation
                if self.implementation == "sklearn":
                    mlflow.sklearn.log_model(self.model, "model")
                elif self.implementation == "xgboost":
                    mlflow.xgboost.log_model(self.model, "model")
                elif self.implementation == "lightgbm":
                    mlflow.lightgbm.log_model(self.model, "model")
                
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
            if self.implementation == "sklearn":
                feature_names = [f"feature_{i}" for i in range(len(self.model.feature_importances_))]
            elif self.implementation == "xgboost":
                feature_names = [f"feature_{i}" for i in range(len(self.model.feature_importances_))]
            elif self.implementation == "lightgbm":
                feature_names = [f"feature_{i}" for i in range(len(self.model.feature_importances_))]
                
        # Get feature importance based on implementation
        if self.implementation == "sklearn":
            importance = self.model.feature_importances_
        elif self.implementation == "xgboost":
            importance = self.model.feature_importances_
        elif self.implementation == "lightgbm":
            importance = self.model.feature_importances_
            
        # Create DataFrame
        feature_importance = pd.DataFrame({
            'Feature': feature_names[:len(importance)],
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
        # Save model based on implementation
        if self.implementation == "sklearn":
            if not path.endswith('.pkl'):
                path = f"{path}.pkl"
            joblib.dump(self.model, path)
        elif self.implementation == "xgboost":
            if not path.endswith('.xgb'):
                path = f"{path}.xgb"
            self.model.save_model(path)
        elif self.implementation == "lightgbm":
            if not path.endswith('.lgb'):
                path = f"{path}.lgb"
            self.model.save_model(path)
            
        self.logger.info(f"Model saved to {path}")
        
        # Save additional metadata
        metadata = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "implementation": self.implementation,
            "params": self.params,
            "feature_names": self.feature_names
        }
        metadata_path = f"{path.rsplit('.', 1)[0]}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        self.logger.info(f"Model metadata saved to {metadata_path}")
        
    @classmethod
    def load(cls, path: str) -> "GradientBoostingModel":
        """
        Load model from disk.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded GradientBoostingModel instance
        """
        # Load metadata
        metadata_path = f"{path.rsplit('.', 1)[0]}_metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Create instance with metadata
        instance = cls(
            model_name=metadata["model_name"],
            model_type=metadata["model_type"],
            implementation=metadata["implementation"],
            **metadata["params"]
        )
        instance.feature_names = metadata.get("feature_names")
        
        # Load model based on implementation
        if metadata["implementation"] == "sklearn":
            if not path.endswith('.pkl'):
                path = f"{path}.pkl"
            instance.model = joblib.load(path)
        elif metadata["implementation"] == "xgboost":
            if not path.endswith('.xgb'):
                path = f"{path}.xgb"
            if metadata["model_type"] == "classification":
                instance.model = xgb.XGBClassifier()
            else:
                instance.model = xgb.XGBRegressor()
            instance.model.load_model(path)
        elif metadata["implementation"] == "lightgbm":
            if not path.endswith('.lgb'):
                path = f"{path}.lgb"
            instance.model = lgb.Booster(model_file=path)
            
        return instance