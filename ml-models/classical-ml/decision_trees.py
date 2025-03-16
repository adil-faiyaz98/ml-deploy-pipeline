import time
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, List, Any, Tuple
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_text
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import io

from base_model import MLModel

class DecisionTreeModel(MLModel):
    """Decision Tree model implementation."""
    
    def __init__(self,
                 model_name: str = "decision_tree",
                 model_type: str = "classification",
                 max_depth: int = 5,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 random_state: int = 42,
                 criterion: str = None,  # 'gini'/'entropy' for classification, 'mse'/'mae' for regression
                 tracking_uri: Optional[str] = None,
                 registry_uri: Optional[str] = None,
                 experiment_name: Optional[str] = None,
                 **kwargs):
        """
        Initialize Decision Tree model.
        
        Args:
            model_name: Name of the model
            model_type: Type of task ('classification' or 'regression')
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split an internal node
            min_samples_leaf: Minimum samples required to be at a leaf node
            random_state: Random seed for reproducibility
            criterion: Function to measure the quality of a split
            tracking_uri: MLflow tracking server URI
            registry_uri: MLflow model registry URI
            experiment_name: MLflow experiment name
            **kwargs: Additional parameters for the model
        """
        if model_type not in ["classification", "regression"]:
            raise ValueError("model_type must be either 'classification' or 'regression'")
        
        if experiment_name is None:
            experiment_name = f"decision_tree_{model_type}_models"
            
        super().__init__(model_name, model_type, tracking_uri, registry_uri, experiment_name)
        
        # Set default criterion based on model type if not specified
        if criterion is None:
            criterion = "gini" if model_type == "classification" else "squared_error"
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.criterion = criterion
        
        # Store parameters
        self.params = {
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state,
            "criterion": criterion
        }
        
        # Add additional parameters
        self.params.update(kwargs)
        
        # Initialize the model based on task type
        if model_type == "classification":
            self.model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                criterion=criterion,
                **kwargs
            )
        else:  # regression
            self.model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                criterion=criterion,
                **kwargs
            )
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], 
            validation_data: Optional[tuple] = None, log_to_mlflow: bool = True, **kwargs) -> Dict[str, float]:
        """
        Fit the decision tree model.
        
        Args:
            X: Training features
            y: Training target
            validation_data: Optional tuple of (X_val, y_val) for validation
            log_to_mlflow: Whether to log metrics and model to MLflow
            **kwargs: Additional parameters to pass to the fit method
            
        Returns:
            Dictionary with training metrics
        """
        self.logger.info(f"Fitting decision tree {self.model_type} model with {X.shape[0]} samples")
        
        # Store feature names for visualization
        self.feature_names = X.columns.tolist() if hasattr(X, "columns") else None
        
        # Class names for visualization
        self.class_names = [str(c) for c in np.unique(y)] if self.model_type == "classification" else None
        
        start_time = time.time()
        self.model.fit(X, y, **kwargs)
        train_time = time.time() - start_time
        
        self.logger.info(f"Model training completed in {train_time:.2f} seconds")
        
        # Calculate training metrics
        train_metrics = self._calculate_metrics(X, y)
        train_metrics["training_time_seconds"] = train_time
        train_metrics["tree_depth"] = self.model.get_depth()
        train_metrics["n_leaves"] = self.model.get_n_leaves()
        
        # Calculate validation metrics if validation data is provided
        if validation_data is not None:
            X_val, y_val = validation_data
            val_metrics = self._calculate_metrics(X_val, y_val, prefix="val_")
            train_metrics.update(val_metrics)
        
        # Log to MLflow if enabled
        if log_to_mlflow:
            with mlflow.start_run(run_name=f"{self.model_name}_training") as run:
                # Log parameters
                mlflow.log_params(self.params)
                
                # Log metrics
                for metric_name, metric_value in train_metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log model
                mlflow.sklearn.log_model(self.model, "model")
                
                # Log feature importance
                if self.feature_names is not None:
                    feature_importance = self.get_feature_importance(self.feature_names)
                    feature_importance.to_csv("feature_importance.csv", index=False)
                    mlflow.log_artifact("feature_importance.csv")
                
                # Log tree visualization
                self._log_tree_visualization(mlflow)
                
                self.logger.info(f"Training metrics and model logged to MLflow run: {run.info.run_id}")
        
        return train_metrics
    
    def _log_tree_visualization(self, mlflow_client):
        """
        Create and log tree visualization to MLflow.
        
        Args:
            mlflow_client: MLflow client
        """
        # Text representation of the tree
        tree_text = export_text(self.model, feature_names=self.feature_names)
        with open("tree_structure.txt", "w") as f:
            f.write(tree_text)
        mlflow_client.log_artifact("tree_structure.txt")
        
        # Graphical representation of the tree
        plt.figure(figsize=(20, 10))
        plot_tree(self.model, 
                  feature_names=self.feature_names,
                  class_names=self.class_names,
                  filled=True,
                  rounded=True,
                  fontsize=10)
        plt.title(f"Decision Tree - Max Depth: {self.max_depth}")
        plt.tight_layout()
        plt.savefig("tree_visualization.png", dpi=300, bbox_inches="tight")
        mlflow_client.log_artifact("tree_visualization.png")
        plt.close()
    
    def _calculate_metrics(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], 
                          prefix: str = "") -> Dict[str, float]:
        """
        Calculate model performance metrics.
        
        Args:
            X: Features
            y: True labels/values
            prefix: Prefix to add to metric names (e.g., "val_" for validation metrics)
            
        Returns:
            Dictionary with performance metrics
        """
        y_pred = self.model.predict(X)
        metrics = {}
        
        if self.model_type == "classification":
            # For classification problems
            y_pred_proba = self.model.predict_proba(X)[:, 1] if hasattr(self.model, "predict_proba") else None
            
            metrics[f"{prefix}accuracy"] = accuracy_score(y, y_pred)
            
            # For binary classification
            if len(np.unique(y)) == 2:
                metrics[f"{prefix}precision"] = precision_score(y, y_pred)
                metrics[f"{prefix}recall"] = recall_score(y, y_pred)
                metrics[f"{prefix}f1"] = f1_score(y, y_pred)
                
                if y_pred_proba is not None:
                    metrics[f"{prefix}auc"] = roc_auc_score(y, y_pred_proba)
        else:
            # For regression problems
            metrics[f"{prefix}mse"] = mean_squared_error(y, y_pred)
            metrics[f"{prefix}rmse"] = np.sqrt(metrics[f"{prefix}mse"])
            metrics[f"{prefix}r2"] = r2_score(y, y_pred)
        
        return metrics
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Args:
            X: Features to make predictions on
            
        Returns:
            Array of predictions
        """
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        self.logger.info(f"Making predictions with decision tree model on {X.shape[0]} samples")
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Get probability estimates for classification problems.
        
        Args:
            X: Features to make predictions on
            
        Returns:
            Array of class probabilities
        """
        if self.model_type != "classification":
            raise ValueError("predict_proba is only available for classification models")
            
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        self.logger.info(f"Computing prediction probabilities with decision tree model on {X.shape[0]} samples")
        return self.model.predict_proba(X)
    
    def save(self, path: str) -> str:
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
            
        Returns:
            Path where model was saved
        """
        if not path.endswith(".joblib"):
            path = f"{path}.joblib"
            
        self.logger.info(f"Saving decision tree model to {path}")
        joblib.dump(self, path)
        return path
    
    @classmethod
    def load(cls, path: str) -> "DecisionTreeModel":
        """
        Load model from# filepath: c:\Users\adilm\repositories\Go\ml-deploy-pipeline\ml-models\classical-ml\decision_trees.py
import time
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, List, Any, Tuple
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_text
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import io

from base_model import MLModel

class DecisionTreeModel(MLModel):
    
    def __init__(self,
                 model_name: str = "decision_tree",
                 model_type: str = "classification",
                 max_depth: int = 5,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 random_state: int = 42,
                 criterion: str = None,  # 'gini'/'entropy' for classification, 'mse'/'mae' for regression
                 tracking_uri: Optional[str] = None,
                 registry_uri: Optional[str] = None,
                 experiment_name: Optional[str] = None,
                 **kwargs):
        """
        

        if model_type not in ["classification", "regression"]:
            raise ValueError("model_type must be either 'classification' or 'regression'")
        
        if experiment_name is None:
            experiment_name = f"decision_tree_{model_type}_models"
            
        super().__init__(model_name, model_type, tracking_uri, registry_uri, experiment_name)
        
        # Set default criterion based on model type if not specified
        if criterion is None:
            criterion = "gini" if model_type == "classification" else "squared_error"
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.criterion = criterion
        
        # Store parameters
        self.params = {
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state,
            "criterion": criterion
        }
        
        # Add additional parameters
        self.params.update(kwargs)
        
        # Initialize the model based on task type
        if model_type == "classification":
            self.model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                criterion=criterion,
                **kwargs
            )
        else:  # regression
            self.model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                criterion=criterion,
                **kwargs
            )
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], 
            validation_data: Optional[tuple] = None, log_to_mlflow: bool = True, **kwargs) -> Dict[str, float]:
        """
        Fit the decision tree model.
        
        Args:
            X: Training features
            y: Training target
            validation_data: Optional tuple of (X_val, y_val) for validation
            log_to_mlflow: Whether to log metrics and model to MLflow
            **kwargs: Additional parameters to pass to the fit method
            
        Returns:
            Dictionary with training metrics
        """
        self.logger.info(f"Fitting decision tree {self.model_type} model with {X.shape[0]} samples")
        
        # Store feature names for visualization
        self.feature_names = X.columns.tolist() if hasattr(X, "columns") else None
        
        # Class names for visualization
        self.class_names = [str(c) for c in np.unique(y)] if self.model_type == "classification" else None
        
        start_time = time.time()
        self.model.fit(X, y, **kwargs)
        train_time = time.time() - start_time
        
        self.logger.info(f"Model training completed in {train_time:.2f} seconds")
        
        # Calculate training metrics
        train_metrics = self._calculate_metrics(X, y)
        train_metrics["training_time_seconds"] = train_time
        train_metrics["tree_depth"] = self.model.get_depth()
        train_metrics["n_leaves"] = self.model.get_n_leaves()
        
        # Calculate validation metrics if validation data is provided
        if validation_data is not None:
            X_val, y_val = validation_data
            val_metrics = self._calculate_metrics(X_val, y_val, prefix="val_")
            train_metrics.update(val_metrics)
        
        # Log to MLflow if enabled
        if log_to_mlflow:
            with mlflow.start_run(run_name=f"{self.model_name}_training") as run:
                # Log parameters
                mlflow.log_params(self.params)
                
                # Log metrics
                for metric_name, metric_value in train_metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log model
                mlflow.sklearn.log_model(self.model, "model")
                
                # Log feature importance
                if self.feature_names is not None:
                    feature_importance = self.get_feature_importance(self.feature_names)
                    feature_importance.to_csv("feature_importance.csv", index=False)
                    mlflow.log_artifact("feature_importance.csv")
                
                # Log tree visualization
                self._log_tree_visualization(mlflow)
                
                self.logger.info(f"Training metrics and model logged to MLflow run: {run.info.run_id}")
        
        return train_metrics
    
    def _log_tree_visualization(self, mlflow_client):
        """
        Create and log tree visualization to MLflow.
        
        Args:
            mlflow_client: MLflow client
        """
        # Text representation of the tree
        tree_text = export_text(self.model, feature_names=self.feature_names)
        with open("tree_structure.txt", "w") as f:
            f.write(tree_text)
        mlflow_client.log_artifact("tree_structure.txt")
        
        # Graphical representation of the tree
        plt.figure(figsize=(20, 10))
        plot_tree(self.model, 
                  feature_names=self.feature_names,
                  class_names=self.class_names,
                  filled=True,
                  rounded=True,
                  fontsize=10)
        plt.title(f"Decision Tree - Max Depth: {self.max_depth}")
        plt.tight_layout()
        plt.savefig("tree_visualization.png", dpi=300, bbox_inches="tight")
        mlflow_client.log_artifact("tree_visualization.png")
        plt.close()
    
    def _calculate_metrics(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], 
                          prefix: str = "") -> Dict[str, float]:
        """
        Calculate model performance metrics.
        
        Args:
            X: Features
            y: True labels/values
            prefix: Prefix to add to metric names (e.g., "val_" for validation metrics)
            
        Returns:
            Dictionary with performance metrics
        """
        y_pred = self.model.predict(X)
        metrics = {}
        
        if self.model_type == "classification":
            # For classification problems
            y_pred_proba = self.model.predict_proba(X)[:, 1] if hasattr(self.model, "predict_proba") else None
            
            metrics[f"{prefix}accuracy"] = accuracy_score(y, y_pred)
            
            # For binary classification
            if len(np.unique(y)) == 2:
                metrics[f"{prefix}precision"] = precision_score(y, y_pred)
                metrics[f"{prefix}recall"] = recall_score(y, y_pred)
                metrics[f"{prefix}f1"] = f1_score(y, y_pred)
                
                if y_pred_proba is not None:
                    metrics[f"{prefix}auc"] = roc_auc_score(y, y_pred_proba)
        else:
            # For regression problems
            metrics[f"{prefix}mse"] = mean_squared_error(y, y_pred)
            metrics[f"{prefix}rmse"] = np.sqrt(metrics[f"{prefix}mse"])
            metrics[f"{prefix}r2"] = r2_score(y, y_pred)
        
        return metrics
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Args:
            X: Features to make predictions on
            
        Returns:
            Array of predictions
        """
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        self.logger.info(f"Making predictions with decision tree model on {X.shape[0]} samples")
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Get probability estimates for classification problems.
        
        Args:
            X: Features to make predictions on
            
        Returns:
            Array of class probabilities
        """
        if self.model_type != "classification":
            raise ValueError("predict_proba is only available for classification models")
            
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        self.logger.info(f"Computing prediction probabilities with decision tree model on {X.shape[0]} samples")
        return self.model.predict_proba(X)
    
    def save(self, path: str) -> str:
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
            
        Returns:
            Path where model was saved
        """
        if not path.endswith(".joblib"):
            path = f"{path}.joblib"
            
        self.logger.info(f"Saving decision tree model to {path}")
        joblib.dump(self, path)
        return path
    
    @classmethod
    def load(cls, path: str) -> "DecisionTreeModel":
        """
        Load model from