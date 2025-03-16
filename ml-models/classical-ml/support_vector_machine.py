import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, Union, List, Any, Tuple
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import matplotlib.colors as mcolors
import joblib
import json
import mlflow
import mlflow.sklearn
from mpl_toolkits.mplot3d import Axes3D

from base_model import MLModel

class SupportVectorMachineModel(MLModel):
    """
    Support Vector Machine implementation supporting both classification and regression.
    
    Features:
    - Multiple kernels (linear, poly, rbf, sigmoid)
    - Hyperparameter tuning
    - Decision boundary visualization
    - Feature importance approximation
    - Grid/Random search for optimal parameters
    - MLflow integration
    """
    
    def __init__(self,
                 model_name: str = "svm",
                 model_type: str = "classification",
                 kernel: str = "rbf",
                 C: float = 1.0,
                 gamma: Union[str, float] = "scale",
                 degree: int = 3,
                 coef0: float = 0.0,
                 epsilon: float = 0.1,  # For regression only
                 shrinking: bool = True,
                 probability: bool = True,
                 cache_size: int = 200,
                 max_iter: int = -1,
                 random_state: int = 42,
                 feature_scaling: bool = True,
                 tracking_uri: Optional[str] = None,
                 registry_uri: Optional[str] = None,
                 experiment_name: Optional[str] = None):
        """
        Initialize Support Vector Machine model.
        
        Args:
            model_name: Name of the model
            model_type: Type of task ('classification' or 'regression')
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            C: Regularization parameter
            gamma: Kernel coefficient ('scale', 'auto' or float)
            degree: Polynomial degree (for 'poly' kernel)
            coef0: Independent term in kernel function
            epsilon: Epsilon in epsilon-SVR model (for regression only)
            shrinking: Whether to use the shrinking heuristic
            probability: Whether to enable probability estimates
            cache_size: Size of the kernel cache (MB)
            max_iter: Hard limit on iterations (-1 for no limit)
            random_state: Random seed for reproducibility
            feature_scaling: Whether to apply feature scaling
            tracking_uri: MLflow tracking server URI
            registry_uri: MLflow model registry URI
            experiment_name: MLflow experiment name
        """
        if model_type not in ["classification", "regression"]:
            raise ValueError("model_type must be either 'classification' or 'regression'")
            
        if kernel not in ["linear", "poly", "rbf", "sigmoid"]:
            raise ValueError("kernel must be one of 'linear', 'poly', 'rbf', 'sigmoid'")
            
        if experiment_name is None:
            experiment_name = f"svm_{model_type}_models"
            
        super().__init__(model_name, model_type, tracking_uri, registry_uri, experiment_name)
        
        # Store parameters
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.epsilon = epsilon
        self.shrinking = shrinking
        self.probability = probability
        self.cache_size = cache_size
        self.max_iter = max_iter
        self.random_state = random_state
        self.feature_scaling = feature_scaling
        
        self.params = {
            "kernel": kernel,
            "C": C,
            "gamma": gamma,
            "degree": degree,
            "coef0": coef0,
            "epsilon": epsilon,
            "shrinking": shrinking,
            "probability": probability,
            "cache_size": cache_size,
            "max_iter": max_iter,
            "random_state": random_state
        }
        
        # Initialize the model
        self._initialize_model()
        
        # Initialize feature scaler if requested
        self.scaler = StandardScaler() if feature_scaling else None
        self.feature_names = None
        
        # Feature importance (for linear kernels)
        self.feature_importances_ = None
        
    def _initialize_model(self):
        """Initialize the SVM model based on type and kernel."""
        if self.model_type == "classification":
            if self.kernel == "linear" and self.C >= 10:
                # Use LinearSVC for better performance on large datasets with linear kernel
                self.model = LinearSVC(
                    penalty='l2',
                    loss='squared_hinge',
                    dual=True,  # Better for n_samples > n_features
                    C=self.C,
                    random_state=self.random_state,
                    max_iter=self.max_iter if self.max_iter > 0 else 1000
                )
            else:
                self.model = SVC(
                    C=self.C,
                    kernel=self.kernel,
                    gamma=self.gamma,
                    degree=self.degree,
                    coef0=self.coef0,
                    shrinking=self.shrinking,
                    probability=self.probability,
                    cache_size=self.cache_size,
                    max_iter=self.max_iter,
                    random_state=self.random_state
                )
        else:  # regression
            if self.kernel == "linear":
                # Use LinearSVR for better performance on large datasets with linear kernel
                self.model = LinearSVR(
                    epsilon=self.epsilon,
                    C=self.C,
                    random_state=self.random_state,
                    max_iter=self.max_iter if self.max_iter > 0 else 1000
                )
            else:
                self.model = SVR(
                    kernel=self.kernel,
                    C=self.C,
                    gamma=self.gamma,
                    degree=self.degree,
                    coef0=self.coef0,
                    epsilon=self.epsilon,
                    shrinking=self.shrinking,
                    cache_size=self.cache_size,
                    max_iter=self.max_iter
                )

    def fit(self, 
            X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray],
            validation_data: Optional[Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]] = None,
            feature_names: Optional[List[str]] = None,
            log_to_mlflow: bool = True,
            **kwargs) -> Dict[str, float]:
        """
        Fit the SVM model to the training data.
        
        Args:
            X: Training features
            y: Target variable
            validation_data: Tuple of (X_val, y_val) for validation
            feature_names: List of feature names
            log_to_mlflow: Whether to log metrics and model to MLflow
            **kwargs: Additional parameters for training
                - tune_hyperparams: Whether to tune hyperparameters
                - param_grid: Parameter grid for hyperparameter tuning
                - cv: Number of cross-validation folds
                - search_method: 'grid' or 'random'
                - n_iter: Number of parameter settings for random search
                - scoring: Scoring metric for hyperparameter tuning
                
        Returns:
            Dictionary of training metrics
        """
        self.logger.info(f"Fitting SVM model with {X.shape[0]} samples and {X.shape[1]} features")
        
        start_time = time.time()
        
        # Store feature names if provided or extract from DataFrame
        if feature_names is not None:
            self.feature_names = feature_names
        elif isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        
        # Convert to numpy arrays if needed
        X_train = X.values if isinstance(X, pd.DataFrame) else X
        y_train = y.values if isinstance(y, pd.Series) else y

        # Apply feature scaling if enabled
        if self.feature_scaling:
            X_train = self.scaler.fit_transform(X_train)
        
        # Hyperparameter tuning
        if kwargs.get('tune_hyperparams', False):
            tune_result = self._tune_hyperparameters(X_train, y_train, **kwargs)
            self.model = tune_result['best_estimator']
            self.params.update(tune_result['best_params'])
            train_metrics = tune_result['cv_results']
        else:
            # Fit the model
            self.model.fit(X_train, y_train)
            
            # Calculate training metrics
            train_metrics = self._calculate_metrics(X_train, y_train)
        
        # Calculate validation metrics if validation data provided
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
            y_val = y_val.values if isinstance(y_val, pd.Series) else y_val
            
            if self.feature_scaling:
                X_val = self.scaler.transform(X_val)
                
            val_metrics = self._calculate_metrics(X_val, y_val, prefix="val_")
            train_metrics.update(val_metrics)
        
        # Record training time
        train_metrics['training_time_seconds'] = time.time() - start_time
        
        # Extract support vectors information
        if hasattr(self.model, 'n_support_'):
            train_metrics['num_support_vectors'] = sum(self.model.n_support_)
            train_metrics['percent_support_vectors'] = train_metrics['num_support_vectors'] / X_train.shape[0] * 100
        
        # Calculate feature importances for linear kernel
        if self.kernel == "linear":
            self._calculate_feature_importances()
            
            # Log top feature importances
            if self.feature_names is not None and self.feature_importances_ is not None:
                importance_dict = dict(zip(self.feature_names, self.feature_importances_))
                for i, (name, importance) in enumerate(sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]):
                    train_metrics[f"feature_importance_{i+1}_{name}"] = importance
        
        # Log to MLflow if requested
        if log_to_mlflow:
            self._log_to_mlflow(train_metrics)
        
        return train_metrics

    def _tune_hyperparameters(self, X, y, **kwargs):
        """Tune hyperparameters using grid or random search."""
        cv = kwargs.get('cv', 5)
        search_method = kwargs.get('search_method', 'grid')
        n_iter = kwargs.get('n_iter', 20)
        
        # Default scoring based on model type
        default_scoring = 'accuracy' if self.model_type == 'classification' else 'neg_mean_squared_error'
        scoring = kwargs.get('scoring', default_scoring)
        
        # Default parameter grid if not provided
        default_param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01, 0.001]
        }
        
        if self.kernel == 'poly':
            default_param_grid['degree'] = [2, 3, 4]
            
        param_grid = kwargs.get('param_grid', default_param_grid)
        
        # Create search CV object
        if search_method == 'grid':
            search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
                return_train_score=True
            )
        else:  # random search
            search = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state,
                return_train_score=True
            )
            
        # Fit search
        self.logger.info(f"Starting {search_method} search for hyperparameter tuning...")
        search.fit(X, y)
        self.logger.info(f"Hyperparameter tuning completed. Best score: {search.best_score_:.4f}")
        
        # Prepare CV results
        cv_results = {
            'best_score': search.best_score_,
            'best_params': search.best_params_
        }
        
        # If classification, include additional metrics
        if self.model_type == 'classification':
            # Get the predictions from the best estimator
            y_pred = search.best_estimator_.predict(X)
            cv_results.update({
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted') if len(np.unique(y)) > 2 else precision_score(y, y_pred),
                'recall': recall_score(y, y_pred, average='weighted') if len(np.unique(y)) > 2 else recall_score(y, y_pred),
                'f1_score': f1_score(y, y_pred, average='weighted') if len(np.unique(y)) > 2 else f1_score(y, y_pred)
            })
            
            # Add ROC AUC if probability is enabled
            if self.probability and hasattr(search.best_estimator_, 'predict_proba'):
                if len(np.unique(y)) == 2:
                    y_prob = search.best_estimator_.predict_proba(X)[:, 1]
                    cv_results['roc_auc'] = roc_auc_score(y, y_prob)
        else:  # regression
            y_pred = search.best_estimator_.predict(X)
            mse = mean_squared_error(y, y_pred)
            cv_results.update({
                'mse': mse,
                'rmse': np.sqrt(mse),
                'mae': mean_absolute_error(y, y_pred),
                'r2': r2_score(y, y_pred)
            })
            
        return {
            'best_estimator': search.best_estimator_,
            'best_params': search.best_params_,
            'cv_results': cv_results
        }

    def _calculate_metrics(self, X, y, prefix: str = "") -> Dict[str, float]:
        """Calculate model performance metrics based on model type."""
        metrics = {}
        
        # Get predictions
        y_pred = self.model.predict(X)
        
        if self.model_type == "classification":
            # Classification metrics
            metrics[f"{prefix}accuracy"] = accuracy_score(y, y_pred)
            
            # Multi-class vs binary classification
            avg = 'weighted' if len(np.unique(y)) > 2 else 'binary'
            metrics[f"{prefix}precision"] = precision_score(y, y_pred, average=avg, zero_division=0)
            metrics[f"{prefix}recall"] = recall_score(y, y_pred, average=avg, zero_division=0)
            metrics[f"{prefix}f1"] = f1_score(y, y_pred, average=avg, zero_division=0)
            
            # Add ROC AUC if probability is enabled
            if self.probability and hasattr(self.model, 'predict_proba'):
                if len(np.unique(y)) == 2:
                    y_prob = self.model.predict_proba(X)[:, 1]
                    metrics[f"{prefix}roc_auc"] = roc_auc_score(y, y_prob)
        else:  # regression
            # Regression metrics
            mse = mean_squared_error(y, y_pred)
            metrics[f"{prefix}mse"] = mse
            metrics[f"{prefix}rmse"] = np.sqrt(mse)
            metrics[f"{prefix}mae"] = mean_absolute_error(y, y_pred)
            metrics[f"{prefix}r2"] = r2_score(y, y_pred)
            
        return metrics
        
    def _calculate_feature_importances(self):
        """Calculate feature importances for linear kernel."""
        if not hasattr(self.model, 'coef_'):
            return
            
        # Extract feature importances (coefficients for linear kernel)
        self.feature_importances_ = np.ravel(self.model.coef_)
        
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Generate predictions using the trained model.
        
        Args:
            X: Features to predict
            
        Returns:
            Predictions as numpy array
        """
        # Convert to numpy if needed
        X_pred = X.values if isinstance(X, pd.DataFrame) else X
        
        # Apply feature scaling if used during training
        if self.feature_scaling and hasattr(self, 'scaler') and self.scaler is not None:
            X_pred = self.scaler.transform(X_pred)
            
        return self.model.predict(X_pred)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Generate class probabilities for classification.
        
        Args:
            X: Features to predict
            
        Returns:
            Class probabilities
            
        Raises:
            ValueError: If model is not a classifier or doesn't support probabilities
        """
        if self.model_type != "classification":
            raise ValueError("predict_proba is only available for classification models")
            
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model doesn't support probability predictions. Set probability=True during initialization.")
        
        # Convert to numpy if needed
        X_pred = X.values if isinstance(X, pd.DataFrame) else X
        
        # Apply feature scaling if used during training
        if self.feature_scaling and hasattr(self, 'scaler') and self.scaler is not None:
            X_pred = self.scaler.transform(X_pred)
            
        return self.model.predict_proba(X_pred)
        
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get feature importances as a DataFrame.
        
        Args:
            feature_names: List of feature names. If None, use saved feature names.
            
        Returns:
            DataFrame with feature importances
            
        Raises:
            ValueError: If model doesn't have linear kernel or hasn't been trained
        """
        if self.kernel != "linear":
            raise ValueError("Feature importances are only available for linear kernel")
            
        if self.feature_importances_ is None:
            raise ValueError("Model has not been trained yet or doesn't support feature importances")
            
        # Use provided feature names or stored ones
        if feature_names is None:
            if self.feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(self.feature_importances_))]
            else:
                feature_names = self.feature_names
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.feature_importances_
        })
        
        # Sort by absolute importance
        importance_df['AbsImportance'] = importance_df['Importance'].abs()
        importance_df = importance_df.sort_values('AbsImportance', ascending=False)
        importance_df = importance_df.drop('AbsImportance', axis=1)
        
        return importance_df
        
    def visualize_decision_boundary(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], 
                                     feature_idx: List[int] = [0, 1], resolution: int = 100) -> plt.Figure:
        """
        Visualize decision boundary for classification models (works for 2D visualization).
        
        Args:
            X: Features
            y: Target classes
            feature_idx: Indices of two features to use for visualization
            resolution: Resolution of the decision boundary grid
            
        Returns:
            Matplotlib figure with decision boundary
        """
        if self.model_type != "classification":
            raise ValueError("Decision boundary visualization is only available for classification models")
        
        if len(feature_idx) != 2:
            raise ValueError("feature_idx must contain exactly two feature indices")
            
        # Convert to numpy if needed
        X_data = X.values if isinstance(X, pd.DataFrame) else X
        y_data = y.values if isinstance(y, pd.Series) else y
        
        # Apply feature scaling if needed
        if self.feature_scaling and hasattr(self, 'scaler') and self.scaler is not None:
            X_data = self.scaler.transform(X_data)
        
        # Extract the two features for visualization
        X_viz = X_data[:, feature_idx]
        
        # Generate mesh grid for decision boundary
        x_min, x_max = X_viz[:, 0].min() - 0.1, X_viz[:, 0].max() + 0.1
        y_min, y_max = X_viz[:, 1].min() - 0.1, X_viz[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                             np.linspace(y_min, y_max, resolution))
        
        # Create full-dimensional data with default values
        if X_data.shape[1] > 2:
            # Use mean values for non-visualized dimensions
            X_default = np.tile(X_data.mean(axis=0), (resolution * resolution, 1))
            # Replace the two visualized features
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            X_default[:, feature_idx] = mesh_points
            Z = self.model.predict(X_default)
        else:
            # If only 2D data, use the mesh grid directly
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            Z = self.model.predict(mesh_points)
            
        # Reshape for plotting
        Z = Z.reshape(xx.shape)
        
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        
        # Choose colormap based on number of classes
        unique_classes = np.unique(y_data)
        n_classes = len(unique_classes)
        cmap = plt.cm.viridis if n_classes > 10 else plt.cm.tab10
        
        # Plot decision boundary
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap)
        
        # Plot training points
        markers = ['o', 's', '^', 'v', 'D', '<', '>', 'p', 'h', '8']
        for i, cls in enumerate(unique_classes):
            idx = y_data == cls
            marker = markers[i % len(markers)]
            color = cmap(i / n_classes) if n_classes > 10 else cmap(i)
            plt.scatter(X_viz[idx, 0], X_viz[idx, 1], c=[color], marker=marker, label=f'Class {cls}', edgecolors='k')
            
        # Plot support vectors if available
        if hasattr(self.model, 'support_vectors_'):
            # Get support vectors in the original feature space
            sv = self.model.support_vectors_
            if sv.shape[1] > 2:
                # Extract only the two visualized dimensions
                sv = sv[:, feature_idx]
            plt.scatter(sv[:, 0], sv[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='red', label='Support Vectors')
        
        # Add labels and legend
        feature_names = [f"Feature {idx}" for idx in feature_idx]
        if self.feature_names is not None:
            feature_names = [self.feature_names[idx] for idx in feature_idx]
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title(f'SVM Decision Boundary ({self.kernel.capitalize()} Kernel)')
        plt.legend(loc='upper right')
        plt.tight_layout()
        
        return fig
    
    def visualize_hyperplane_3d(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray],
                               feature_idx: List[int] = [0, 1, 2], resolution: int = 20) -> plt.Figure:
        """
        Visualize 3D decision hyperplane for binary classification with linear kernel.
        
        Args:
            X: Features
            y: Target classes
            feature_idx: Indices of three features to use for visualization
            resolution: Resolution of the hyperplane grid
            
        Returns:
            Matplotlib figure with 3D hyperplane visualization
        """
        if self.model_type != "classification" or self.kernel != "linear":
            raise ValueError("3D hyperplane visualization is only available for linear SVM classification")
            
        if len(feature_idx) != 3:
            raise ValueError("feature_idx must contain exactly three feature indices")
            
        # Convert to numpy if needed
        X_data = X.values if isinstance(X, pd.DataFrame) else X
        y_data = y.values if isinstance(y, pd.Series) else y
        
        # Apply feature scaling if needed
        if self.feature_scaling and hasattr(self, 'scaler') and self.scaler is not None:
            X_data = self.scaler.transform(X_data)
        
        # Extract the three features for visualization
        X_viz = X_data[:, feature_idx]
        
        # Get model coefficients and intercept
        coef = self.model.coef_[0]
        intercept = self.model.intercept_[0]
        
        # Extract coefficients for the three selected features
        a, b, c = coef[feature_idx]
        d = intercept
        
        # Generate mesh grid for the hyperplane
        x_min, x_max = X_viz[:, 0].min() - 0.5, X_viz[:, 0].max() + 0.5
        y_min, y_max = X_viz[:, 1].min() - 0.5, X_viz[:, 1].max() + 0.5
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                            np.linspace(y_min, y_max, resolution))
        
        # Calculate z from the hyperplane equation: a*x + b*y + c*z + d = 0
        zz = (-a * xx - b * yy - d) / c
        
        # Create figure and 3D axis
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the hyperplane
        surf = ax.plot_surface(xx, yy, zz, alpha=0.5, color='cyan')
        
        # Plot the data points
        classes = np.unique(y_data)
        colors = ['darkblue', 'darkred']
        markers = ['o', '^']
        
        for i, cls in enumerate(classes):
            idx = y_data == cls
            ax.scatter(X_viz[idx, 0], X_viz[idx, 1], X_viz[idx, 2], 
                      c=colors[i % len(colors)], marker=markers[i % len(markers)],
                      s=50, label=f'Class {cls}')
        
        # Add labels
        feature_names = [f"Feature {idx}" for idx in feature_idx]
        if self.feature_names is not None:
            feature_names = [self.feature_names[idx] for idx in feature_idx]
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_zlabel(feature_names[2])
        ax.set_title('SVM Decision Hyperplane (Linear Kernel)')
        ax.legend()
        plt.tight_layout()
        
        return fig

    def _log_to_mlflow(self, metrics: Dict[str, float]):
        """Log model parameters, metrics and artifacts to MLflow."""
        with mlflow.start_run(run_name=f"{self.model_name}_training") as run:
            # Log parameters
            mlflow.log_params(self.params)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                if metric_value is not None:
                    mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            mlflow.sklearn.log_model(self.model, "model")
            
            # Log scaler if used
            if self.feature_scaling and hasattr(self, 'scaler') and self.scaler is not None:
                with open("scaler.pkl", "wb") as f:
                    joblib.dump(self.scaler, f)
                mlflow.log_artifact("scaler.pkl")
            
            # Log feature importance for linear kernel
            if self.kernel == "linear" and self.feature_names is not None and self.feature_importances_ is not None:
                feature_imp_df = self.get_feature_importance(self.feature_names)
                feature_imp_df.to_csv("feature_importance.csv", index=False)
                mlflow.log_artifact("feature_importance.csv")
                
                # Create and log feature importance plot
                plt.figure(figsize=(10, 6))
                top_features = feature_imp_df.head(20)
                colors = ['green' if imp > 0 else 'red' for imp in top_features['Importance']]
                plt.barh(top_features['Feature'], top_features['Importance'], color=colors)
                plt.xlabel('Coefficient/Weight')
                plt.ylabel('Feature')
                plt.title('Top 20 Feature Importances (Linear SVM)')
                plt.tight_layout()
                plt.savefig("feature_importance_plot.png", dpi=300, bbox_inches="tight")
                mlflow.log_artifact("feature_importance_plot.png")
                plt.close()
            
            self.logger.info(f"Model, parameters, and metrics logged to MLflow run: {run.info.run_id}")
    
    def save(self, path: str) -> None:
        """
        Save model to disk with metadata.
        
        Args:
            path: Path to save the model
        """
        if not path.endswith('.pkl'):
            path = f"{path}.pkl"
            
        # Save the model
        joblib.dump(self.model, path)
        
        # Save scaler if used
        if self.feature_scaling and self.scaler is not None:
            scaler_path = f"{path.rsplit('.', 1)[0]}_scaler.pkl"
            joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "kernel": self.kernel,
            "feature_names": self.feature_names,
            "params": self.params,
            "feature_scaling": self.feature_scaling,
            "has_feature_importances": self.feature_importances_ is not None
        }
        
        metadata_path = f"{path.rsplit('.', 1)[0]}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
            
        self.logger.info(f"Model saved to {path}")
        self.logger.info(f"Metadata saved to {metadata_path}")
        
        # Save feature importances if available
        if self.feature_importances_ is not None and self.feature_names is not None:
            importance_path = f"{path.rsplit('.', 1)[0]}_feature_importance.csv"
            feature_imp_df = self.get_feature_importance(self.feature_names)
            feature_imp_df.to_csv(importance_path, index=False)
            self.logger.info(f"Feature importances saved to {importance_path}")
    
    @classmethod
    def load(cls, path: str) -> "SupportVectorMachineModel":
        """
        Load a saved SVM model.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded SupportVectorMachineModel instance
        """
        if not path.endswith('.pkl'):
            path = f"{path}.pkl"
            
        # Load metadata
        metadata_path = f"{path.rsplit('.', 1)[0]}_metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Create a new instance with the saved parameters
        instance = cls(
            model_name=metadata["model_name"],
            model_type=metadata["model_type"],
            kernel=metadata["kernel"],
            feature_scaling=metadata["feature_scaling"],
            **{k: v for k, v in metadata["params"].items() if k != "kernel"}
        )
        
        # Load the scikit-learn model
        instance.model = joblib.load(path)
        
        # Load feature names if available
        if "feature_names" in metadata and metadata["feature_names"] is not None:
            instance.feature_names = metadata["feature_names"]
        
        # Load scaler if used
        if metadata["feature_scaling"]:
            scaler_path = f"{path.rsplit('.', 1)[0]}_scaler.pkl"
            if os.path.exists(scaler_path):
                instance.scaler = joblib.load(scaler_path)
        
        # Load feature importances if available
        if metadata.get("has_feature_importances", False):
            importance_path = f"{path.rsplit('.', 1)[0]}_feature_importance.csv"
            if os.path.exists(importance_path):
                feature_imp_df = pd.read_csv(importance_path)
                if "Importance" in feature_imp_df.columns:
                    instance.feature_importances_ = feature_imp_df["Importance"].values
        
        return instance
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the model.
        
        Returns:
            Dictionary with model parameters
        """
        return self.params
    
    def evaluate(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], 
                 verbose: bool = True) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Test features
            y: True target values
            verbose: Whether to print evaluation results
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Convert to numpy if needed
        X_test = X.values if isinstance(X, pd.DataFrame) else X
        y_test = y.values if isinstance(y, pd.Series) else y
        
        # Apply feature scaling if used during training
        if self.feature_scaling and self.scaler is not None:
            X_test = self.scaler.transform(X_test)
            
        # Calculate metrics
        metrics = self._calculate_metrics(X_test, y_test, prefix="test_")
        
        if verbose:
            self.logger.info("Model Evaluation Results:")
            for name, value in metrics.items():
                self.logger.info(f"  {name}: {value:.4f}")
        
        return metrics
    
    def cross_validate(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], 
                       cv: int = 5, scoring: Optional[Union[str, List[str]]] = None) -> Dict[str, float]:
        """
        Perform cross-validation on the model.
        
        Args:
            X: Features
            y: Target values
            cv: Number of cross-validation folds
            scoring: Scoring metric(s) to use
            
        Returns:
            Dictionary of cross-validation results
        """
        # Convert to numpy if needed
        X_data = X.values if isinstance(X, pd.DataFrame) else X
        y_data = y.values if isinstance(y, pd.Series) else y
        
        # Default scoring based on model type
        if scoring is None:
            scoring = 'accuracy' if self.model_type == 'classification' else 'neg_mean_squared_error'
        
        # Create pipeline with scaling if needed
        if self.feature_scaling:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', self.model)
            ])
        else:
            pipeline = self.model
        
        # Perform cross-validation
        self.logger.info(f"Performing {cv}-fold cross-validation with scoring: {scoring}")
        if isinstance(scoring, list):
            cv_results = {}
            for score in scoring:
                scores = cross_val_score(pipeline, X_data, y_data, cv=cv, scoring=score, n_jobs=-1)
                cv_results[f"cv_{score}"] = scores.mean()
                cv_results[f"cv_{score}_std"] = scores.std()
        else:
            scores = cross_val_score(pipeline, X_data, y_data, cv=cv, scoring=scoring, n_jobs=-1)
            cv_results = {
                "cv_score": scores.mean(),
                "cv_score_std": scores.std()
            }
        
        self.logger.info(f"Cross-validation results: {cv_results}")
        return cv_results
    
    def explain_prediction(self, X: Union[pd.DataFrame, np.ndarray], threshold: float = 0.1) -> pd.DataFrame:
        """
        Explain individual predictions for linear kernel SVM.
        
        Args:
            X: Sample(s) to explain
            threshold: Threshold for feature contribution significance
            
        Returns:
            DataFrame with feature contributions
            
        Raises:
            ValueError: If not using linear kernel or model hasn't been trained
        """
        if self.kernel != "linear":
            raise ValueError("Prediction explanation is only available for linear kernel")
            
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Model has not been trained yet")
        
        # Convert to numpy if needed
        X_data = X.values if isinstance(X, pd.DataFrame) else X
        
        # Apply feature scaling if used during training
        if self.feature_scaling and self.scaler is not None:
            X_data = self.scaler.transform(X_data)
        
        # Get coefficients
        coef = np.ravel(self.model.coef_)
        
        # Get feature names
        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(coef))]
        else:
            feature_names = self.feature_names
        
        # Calculate feature contributions for each sample
        results = []
        for i in range(X_data.shape[0]):
            # Calculate contribution of each feature
            contributions = X_data[i] * coef
            
            # Create DataFrame for this sample
            sample_df = pd.DataFrame({
                'Feature': feature_names,
                'Value': X_data[i],
                'Coefficient': coef,
                'Contribution': contributions
            })
            
            # Add intercept
            intercept = self.model.intercept_[0] if hasattr(self.model, 'intercept_') else 0
            intercept_df = pd.DataFrame({
                'Feature': ['Intercept'],
                'Value': [1],
                'Coefficient': [intercept],
                'Contribution': [intercept]
            })
            
            sample_df = pd.concat([sample_df, intercept_df], ignore_index=True)
            
            # Filter insignificant contributions if requested
            if threshold > 0:
                abs_contributions = np.abs(sample_df['Contribution'])
                max_contrib = abs_contributions.max()
                significant = abs_contributions >= (threshold * max_contrib)
                sample_df = sample_df[significant]
            
            # Sort by absolute contribution
            sample_df['AbsContribution'] = sample_df['Contribution'].abs()
            sample_df = sample_df.sort_values('AbsContribution', ascending=False)
            sample_df = sample_df.drop('AbsContribution', axis=1)
            
            # Calculate decision value
            decision_value = sample_df['Contribution'].sum()
            
            # Add prediction info
            prediction = self.model.predict(X_data[i:i+1])[0]
            sample_df['Total Decision Value'] = decision_value
            sample_df['Predicted Class'] = prediction
            
            results.append(sample_df)
        
        return results