"""
Model Evaluation Framework

A comprehensive framework for evaluating machine learning models with:
- Support for classification, regression, and deep learning models
- Advanced metrics and statistical analysis
- Cross-validation strategies
- Visualization capabilities
- Interpretability features
- MLflow integration
- Statistical testing and calibration
- Report generation
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import yaml
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, 
    classification_report, confusion_matrix, 
    explained_variance_score, f1_score, log_loss,
    mean_absolute_error, mean_squared_error, precision_recall_curve,
    precision_score, r2_score, recall_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit,
    cross_val_score, learning_curve, validation_curve
)
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/model_evaluation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("model_evaluation")

# Default configuration
DEFAULT_CONFIG = {
    "evaluation": {
        "metrics": {
            "classification": ["accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"],
            "regression": ["mse", "rmse", "mae", "r2", "explained_variance"]
        },
        "cv": {
            "method": "stratified",
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42
        },
        "visualization": {
            "enabled": True,
            "plots": ["confusion_matrix", "roc_curve", "pr_curve", "feature_importance", "learning_curve"]
        },
        "interpretability": {
            "enabled": True,
            "methods": ["shap", "permutation_importance"],
            "max_display": 20
        },
        "statistical_tests": {
            "enabled": True,
            "significance_level": 0.05,
            "bootstrap_iterations": 1000
        },
        "calibration": {
            "enabled": True,
            "method": "isotonic",  # 'isotonic' or 'sigmoid'
            "bins": 10
        }
    },
    "reporting": {
        "output_dir": "evaluation_results",
        "generate_report": True,
        "format": ["json", "csv", "html"],
        "visualizations_dir": "visualizations"
    },
    "mlflow": {
        "enabled": True,
        "experiment_name": "model_evaluation",
        "tracking_uri": None
    }
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Comprehensive model evaluation framework")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model file"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the evaluation dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to evaluation configuration file"
    )
    parser.add_argument(
        "--target_col",
        type=str,
        help="Name of the target column"
    )
    parser.add_argument(
        "--problem_type",
        type=str,
        choices=["classification", "regression", "binary_classification", "multiclass_classification"],
        help="Type of machine learning problem"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save evaluation results"
    )
    
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Dict:
    """
    Load and merge configuration from file and command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Dictionary with merged configuration
    """
    config = DEFAULT_CONFIG.copy()
    
    # Load config from file if provided
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as file:
                file_config = yaml.safe_load(file)
                
            # Deep merge configuration
            for section, values in file_config.items():
                if isinstance(values, dict) and section in config:
                    for subsection, subvalues in values.items():
                        if isinstance(subvalues, dict) and subsection in config[section]:
                            config[section][subsection].update(subvalues)
                        else:
                            config[section][subsection] = subvalues
                else:
                    config[section] = values
                    
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
    
    # Override with command-line arguments
    if args.output_dir:
        config["reporting"]["output_dir"] = args.output_dir
    
    return config


def auto_detect_problem_type(y: np.ndarray) -> str:
    """
    Automatically detect the problem type based on target data.
    
    Args:
        y: Target values array
        
    Returns:
        Problem type string: 'regression', 'binary_classification', or 'multiclass_classification'
    """
    unique_values = np.unique(y)
    
    # Check if it's likely a regression problem
    if len(unique_values) > 10 and isinstance(y[0], (int, float, np.integer, np.floating)):
        if np.issubdtype(y.dtype, np.number) and len(unique_values) / len(y) > 0.05:
            return "regression"
    
    # Must be classification
    if len(unique_values) == 2:
        return "binary_classification"
    else:
        return "multiclass_classification"


class ModelEvaluator:
    """Comprehensive model evaluation framework for various ML model types."""
    
    def __init__(self, config: Dict):
        """
        Initialize model evaluator with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.evaluation_results = {}
        self.feature_names = None
        self.class_names = None
        self.metrics = {}
        self.visualizations = {}
        
        # Ensure output directories exist
        self._create_output_dirs()
        
        # Initialize MLflow if enabled
        if self.config["mlflow"]["enabled"]:
            self._setup_mlflow()
    
    def _create_output_dirs(self) -> None:
        """Create necessary output directories."""
        output_dir = self.config["reporting"]["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        viz_dir = os.path.join(output_dir, self.config["reporting"]["visualizations_dir"])
        os.makedirs(viz_dir, exist_ok=True)
        
        logger.info(f"Created output directories: {output_dir}, {viz_dir}")
    
    def _setup_mlflow(self) -> None:
        """Set up MLflow tracking."""
        try:
            tracking_uri = self.config["mlflow"]["tracking_uri"]
            experiment_name = self.config["mlflow"]["experiment_name"]
            
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
                
            # Check if experiment exists, create if not
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
                
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow tracking configured with experiment: {experiment_name}")
        except Exception as e:
            logger.error(f"Failed to set up MLflow: {str(e)}")
            self.config["mlflow"]["enabled"] = False
    
    def _get_cv_splitter(self, X: np.ndarray, y: np.ndarray, problem_type: str) -> Any:
        """
        Get the appropriate cross-validation splitter based on data and config.
        
        Args:
            X: Feature data
            y: Target data
            problem_type: Type of machine learning problem
            
        Returns:
            CV splitter object
        """
        cv_config = self.config["evaluation"]["cv"]
        n_splits = cv_config["n_splits"]
        shuffle = cv_config["shuffle"]
        random_state = cv_config["random_state"]
        
        cv_method = cv_config["method"].lower()
        
        if cv_method == "kfold":
            return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        elif cv_method == "stratified" and problem_type != "regression":
            return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        elif cv_method == "timeseries":
            return TimeSeriesSplit(n_splits=n_splits)
        else:
            # Default to KFold for regression or unknown methods
            return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    def evaluate(
        self, 
        model: Any, 
        X: np.ndarray, 
        y: np.ndarray, 
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        problem_type: Optional[str] = None
    ) -> Dict:
        """
        Evaluate model with comprehensive metrics and visualizations.
        
        Args:
            model: Trained model object
            X: Training features
            y: Training targets
            X_test: Test features (if None, will use cross-validation on X)
            y_test: Test targets
            feature_names: List of feature names
            class_names: List of class names for classification
            problem_type: Type of problem ('regression', 'binary_classification', 'multiclass_classification')
            
        Returns:
            Dictionary with evaluation results
        """
        start_time = time.time()
        
        # Store feature and class names
        self.feature_names = feature_names
        self.class_names = class_names
        
        # Auto-detect problem type if not provided
        if not problem_type:
            problem_type = auto_detect_problem_type(y)
        logger.info(f"Problem type: {problem_type}")
        
        # Start MLflow run if enabled
        if self.config["mlflow"]["enabled"]:
            with mlflow.start_run(run_name=f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
                mlflow.log_param("problem_type", problem_type)
                mlflow.log_param("features_count", X.shape[1])
                mlflow.log_param("samples_count", X.shape[0])
                
                self._run_evaluation(model, X, y, X_test, y_test, problem_type)
                self._log_to_mlflow()
                
                run_id = run.info.run_id
                self.evaluation_results["mlflow_run_id"] = run_id
                logger.info(f"MLflow run ID: {run_id}")
        else:
            self._run_evaluation(model, X, y, X_test, y_test, problem_type)
        
        # Generate reports
        if self.config["reporting"]["generate_report"]:
            self._generate_reports()
        
        execution_time = time.time() - start_time
        self.evaluation_results["execution_time"] = execution_time
        logger.info(f"Evaluation completed in {execution_time:.2f} seconds")
        
        return self.evaluation_results
    
    def _run_evaluation(
        self, 
        model: Any, 
        X: np.ndarray, 
        y: np.ndarray, 
        X_test: Optional[np.ndarray], 
        y_test: Optional[np.ndarray],
        problem_type: str
    ) -> None:
        """
        Run the complete evaluation process.
        
        Args:
            model: Trained model object
            X: Training features
            y: Training targets
            X_test: Test features
            y_test: Test targets
            problem_type: Type of machine learning problem
        """
        # Store basic information
        self.evaluation_results["model_type"] = type(model).__name__
        self.evaluation_results["problem_type"] = problem_type
        self.evaluation_results["data_shape"] = {"X": X.shape, "y": y.shape}
        self.evaluation_results["timestamp"] = datetime.now().isoformat()
        
        # If test set is not provided, we'll use cross-validation
        if X_test is None or y_test is None:
            logger.info("No test set provided, using cross-validation for evaluation")
            self._evaluate_with_cross_validation(model, X, y, problem_type)
        else:
            logger.info("Using provided test set for evaluation")
            self._evaluate_with_test_set(model, X, y, X_test, y_test, problem_type)
        
        # Evaluate model complexity and training stability (when applicable)
        self._evaluate_model_complexity(model)
            
        # Generate visualizations if enabled
        if self.config["evaluation"]["visualization"]["enabled"]:
            self._generate_visualizations(model, X, y, X_test, y_test, problem_type)
        
        # Calculate interpretability metrics if enabled
        if self.config["evaluation"]["interpretability"]["enabled"]:
            self._evaluate_interpretability(model, X, problem_type)
        
        # Run statistical tests if enabled
        if self.config["evaluation"]["statistical_tests"]["enabled"]:
            self._perform_statistical_tests(model, X, y, problem_type)
        
        # Evaluate calibration for classification problems
        if (problem_type in ["binary_classification", "multiclass_classification"] 
            and self.config["evaluation"]["calibration"]["enabled"]):
            self._evaluate_calibration(model, X, y, X_test, y_test)
    
    def _evaluate_with_cross_validation(self, model: Any, X: np.ndarray, y: np.ndarray, problem_type: str) -> None:
        """
        Evaluate model using cross-validation.
        
        Args:
            model: Trained model object
            X: Features
            y: Target values
            problem_type: Type of machine learning problem
        """
        # Get appropriate CV splitter
        cv = self._get_cv_splitter(X, y, problem_type)
        
        # Select appropriate metrics based on problem type
        if "regression" in problem_type:
            metrics_list = self.config["evaluation"]["metrics"]["regression"]
            scoring = {
                "mse": "neg_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "r2": "r2",
                "explained_variance": "explained_variance"
            }
        else:  # Classification
            metrics_list = self.config["evaluation"]["metrics"]["classification"]
            scoring = {
                "accuracy": "accuracy",
                "precision": "precision_weighted",
                "recall": "recall_weighted",
                "f1": "f1_weighted"
            }
            if problem_type == "binary_classification":
                scoring["roc_auc"] = "roc_auc"
        
        # Filter to only included metrics
        scoring = {k: v for k, v in scoring.items() if k in metrics_list}
        
        # Perform cross-validation for each metric
        cv_results = {}
        logger.info("Performing cross-validation...")
        
        for metric_name, scorer in scoring.items():
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
                
                # Convert negative scores back to positive for error metrics
                if scorer.startswith("neg_"):
                    scores = -scores
                
                cv_results[metric_name] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "values": scores.tolist()
                }
                
                logger.info(f"CV {metric_name}: Mean = {cv_results[metric_name]['mean']:.4f}, "
                            f"Std = {cv_results[metric_name]['std']:.4f}")
            except Exception as e:
                logger.error(f"Error calculating {metric_name} with CV: {str(e)}")
                cv_results[metric_name] = {"error": str(e)}
        
        # Add RMSE for regression problems
        if "regression" in problem_type and "mse" in cv_results:
            try:
                rmse_mean = np.sqrt(cv_results["mse"]["mean"])
                rmse_values = np.sqrt(cv_results["mse"]["values"])
                cv_results["rmse"] = {
                    "mean": float(rmse_mean),
                    "std": float(np.std(rmse_values)),
                    "values": rmse_values.tolist()
                }
                logger.info(f"CV RMSE: Mean = {rmse_mean:.4f}, Std = {cv_results['rmse']['std']:.4f}")
            except Exception as e:
                logger.error(f"Error calculating RMSE with CV: {str(e)}")
        
        self.metrics["cross_validation"] = cv_results
        self.evaluation_results["metrics"] = self.metrics
    
    def _evaluate_with_test_set(
        self, 
        model: Any, 
        X: np.ndarray, 
        y: np.ndarray, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        problem_type: str
    ) -> None:
        """
        Evaluate model using a separate test set.
        
        Args:
            model: Trained model object
            X: Training features
            y: Training targets
            X_test: Test features
            y_test: Test targets
            problem_type: Type of machine learning problem
        """
        # Make predictions
        try:
            y_pred = model.predict(X_test)
            
            # Get prediction probabilities for classification problems
            y_pred_proba = None
            if problem_type in ["binary_classification", "multiclass_classification"]:
                if hasattr(model, "predict_proba"):
                    y_pred_proba = model.predict_proba(X_test)
                elif hasattr(model, "decision_function"):
                    y_pred_proba = model.decision_function(X_test)
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            self.metrics["test"] = {"error": str(e)}
            self.evaluation_results["metrics"] = self.metrics
            return
        
        # Calculate metrics based on problem type
        test_metrics = {}
        
        if "regression" in problem_type:
            metrics_list = self.config["evaluation"]["metrics"]["regression"]
            
            if "mse" in metrics_list:
                test_metrics["mse"] = float(mean_squared_error(y_test, y_pred))
            
            if "rmse" in metrics_list:
                test_metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            
            if "mae" in metrics_list:
                test_metrics["mae"] = float(mean_absolute_error(y_test, y_pred))
            
            if "r2" in metrics_list:
                test_metrics["r2"] = float(r2_score(y_test, y_pred))
            
            if "explained_variance" in metrics_list:
                test_metrics["explained_variance"] = float(explained_variance_score(y_test, y_pred))
            
            # Calculate residuals
            residuals = y_test - y_pred
            test_metrics["residuals"] = {
                "mean": float(np.mean(residuals)),
                "std": float(np.std(residuals)),
                "min": float(np.min(residuals)),
                "max": float(np.max(residuals))
            }
            
        else:  # Classification
            metrics_list = self.config["evaluation"]["metrics"]["classification"]
            
            if "accuracy" in metrics_list:
                test_metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
            
            # For weighted metrics, use 'weighted' for multiclass, None for binary
            avg = 'weighted' if problem_type == "multiclass_classification" else None
            
            if "precision" in metrics_list:
                test_metrics["precision"] = float(precision_score(y_test, y_pred, average=avg))
            
            if "recall" in metrics_list:
                test_metrics["recall"] = float(recall_score(y_test, y_pred, average=avg))
            
            if "f1" in metrics_list:
                test_metrics["f1"] = float(f1_score(y_test, y_pred, average=avg))
            
            if "log_loss" in metrics_list and y_pred_proba is not None:
                test_metrics["log_loss"] = float(log_loss(y_test, y_pred_proba))
            
            # Calculate ROC AUC for binary classification or one-vs-rest for multiclass
            if "roc_auc" in metrics_list and y_pred_proba is not None:
                try:
                    if problem_type == "binary_classification":
                        if y_pred_proba.ndim > 1:  # If probabilities for both classes
                            # Use the probability of the positive class
                            test_metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba[:, 1]))
                        else:
                            test_metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba))
                    else:  # Multiclass
                        # One-vs-Rest ROC AUC
                        y_bin = label_binarize(y_test, classes=np.unique(y))
                        if y_bin.shape[1] == 1:
                            y_bin = np.hstack((1 - y_bin, y_bin))
                        
                        test_metrics["roc_auc"] = float(roc_auc_score(y_bin, y_pred_proba, multi_class='ovr'))
                except Exception as e:
                    logger.error(f"Error calculating ROC AUC: {str(e)}")
            
            # Add classification report
            try:
                report = classification_report(y_test, y_pred, output_dict=True)
                test_metrics["classification_report"] = report
            except Exception as e:
                logger.error(f"Error generating classification report: {str(e)}")
            
            # Add confusion matrix
            try:
                cm = confusion_matrix(y_test, y_pred).tolist()
                test_metrics["confusion_matrix"] = cm
            except Exception as e:
                logger.error(f"Error generating confusion matrix: {str(e)}")
        
        # Store metrics
        self.metrics["test"] = test_metrics
        
        # Add metrics for training data as well for comparison
        train_metrics = {}
        try:
            y_train_pred = model.predict(X)
            
            if "regression" in problem_type:
                train_metrics["rmse"] = float(np.sqrt(mean_squared_error(y, y_train_pred)))
                train_metrics["r2"] = float(r2_score(y, y_train_pred))
            else:  # Classification
                train_metrics["accuracy"] = float(accuracy_score(y, y_train_pred))
                avg = 'weighted' if problem_type == "multiclass_classification" else None
                train_metrics["f1"] = float(f1_score(y, y_train_pred, average=avg))
        except Exception as e:
            logger.error(f"Error calculating training metrics: {str(e)}")
            train_metrics["error"] = str(e)
        
        self.metrics["train"] = train_metrics
        
        # Log test-train metric differences to check for overfitting
        diff_metrics = {}
        for metric in ["rmse", "r2", "accuracy", "f1"]:
            if (metric in train_metrics and metric in test_metrics and
                isinstance(train_metrics[metric], (int, float)) and 
                isinstance(test_metrics[metric], (int, float))):
                diff_metrics[metric] = train_metrics[metric] - test_metrics[metric]
        
        self.metrics["train_test_diff"] = diff_metrics
        self.evaluation_results["metrics"] = self.metrics
        
        # Log all metrics
        for metric, value in test_metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"Test {metric}: {value:.4f}")
    
    def _evaluate_model_complexity(self, model: Any) -> None:
        """
        Evaluate model complexity metrics.
        
        Args:
            model: Trained model object
        """
        complexity = {}
        
        # Extract model size
        try:
            model_size_bytes = sys.getsizeof(model)
            complexity["model_size_bytes"] = model_size_bytes
            complexity["model_size_mb"] = model_size_bytes / (1024 * 1024)
        except Exception as e:
            logger.error(f"Error calculating model size: {str(e)}")
        
        # Extract number of parameters for different model types
        try:
            # For sklearn tree-based models
            if hasattr(model, 'n_estimators') and hasattr(model, 'estimators_'):
                n_estimators = model.n_estimators
                complexity["n_estimators"] = n_estimators
                
                # Try to get tree depths
                if hasattr(model, 'estimators_') and len(model.estimators_) > 0:
                    tree_depths = [est.get_depth() for est in model.estimators_ if hasattr(est, 'get_depth')]
                    if tree_depths:
                        complexity["mean_tree_depth"] = float(np.mean(tree_depths))
                        complexity["max_tree_depth"] = int(np.max(tree_depths))
            
            # For single decision tree
            elif hasattr(model, 'tree_'):
                complexity["tree_depth"] = int(model.get_depth() if hasattr(model, 'get_depth') else -1)
                complexity["n_nodes"] = int(model.tree_.node_count if hasattr(model.tree_, 'node_count') else -1)
            
            # For neural networks with keras (TF/PyTorch models)
            elif hasattr(model, 'count_params'):
                complexity["n_parameters"] = int(model.count_params())
            elif hasattr(model, 'summary'):
                # Just note that it's a neural network, summary can't be stored as a metric
                complexity["model_type"] = "neural_network"
            
            # For linear models
            elif hasattr(model, 'coef_'):
                complexity["n_nonzero_coefs"] = int(np.count_nonzero(model.coef_))
                complexity["n_features"] = int(model.coef_.size)
            
            # For SVM
            elif hasattr(model, 'support_vectors_'):
                complexity["n_support_vectors"] = int(model.support_vectors_.shape[0])
        
        except Exception as e:
            logger.error(f"Error calculating model complexity metrics: {str(e)}")
        
        self.evaluation_results["complexity"] = complexity
    
    def _generate_visualizations(
        self, 
        model: Any, 
        X: np.ndarray, 
        y: np.ndarray, 
        X_test: Optional[np.ndarray],
        y_test: Optional[np.ndarray],
        problem_type: str
    ) -> None:
        """
        Generate and save evaluation visualizations.
        
        Args:
            model: Trained model object
            X: Training features
            y: Training targets
            X_test: Test features
            y_test: Test targets
            problem_type: Type of machine learning problem
        """
        viz_config = self.config["evaluation"]["visualization"]
        plots = viz_config["plots"]
        vis_dir = os.path.join(
            self.config["reporting"]["output_dir"],
            self.config["reporting"]["visualizations_dir"]
        )
        
        self.visualizations = {}
        
        # Create predictions if we have a test set
        if X_test is not None and y_test is not None:
            y_pred = model.predict(X_test)
            y_pred_proba = None
            if problem_type in ["binary_classification", "multiclass_classification"]:
                if hasattr(model, "predict_proba"):
                    y_pred_proba = model.predict_proba(X_test)
        
        # Generate selected plots
        for plot_type in plots:
            try:
                fig = plt.figure(figsize=(10, 8))
                
                # Classification-specific plots
                if problem_type in ["binary_classification", "multiclass_classification"]:
                    if plot_type == "confusion_matrix" and X_test is not None:
                        self._plot_confusion_matrix(y_test, y_pred, vis_dir)
                    
                    elif plot_type == "roc_curve" and X_test is not None and y_pred_proba is not None:
                        self._plot_roc_curve(y_test, y_pred_proba, problem_type, vis_dir)
                    
                    elif plot_type == "pr_curve" and X_test is not None and y_pred_proba is not None:
                        self._plot_pr_curve(y_test, y_pred_proba, problem_type, vis_dir)
                
                # Regression-specific plots
                elif "regression" in problem_type:
                    if plot_type == "residuals" and X_test is not None:
                        self._plot_residuals(y_test, y_pred, vis_dir)
                
                # Feature importance plot
                if plot_type == "feature_importance":
                    self._plot_feature_importance(model, vis_dir)
                
                # Learning curve plot
                if plot_type == "learning_curve":
                    self._plot_learning_curve(model, X, y, vis_dir)
                
                plt.close(fig)
            except Exception as e:
                logger.error(f"Error generating {plot_type} plot: {str(e)}")
        
        self.evaluation_results["visualizations"] = self.visualizations
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, vis_dir: str) -> None:
        """
        Plot and save confusion matrix.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            vis_dir: Directory to save visualizations
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(vis_dir, "confusion_matrix.png"))
        self.visualizations["confusion_matrix"] = "confusion_matrix.png"
    
    def _plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, problem_type: str, vis_dir: str) -> None:
        """
        Plot and save ROC curve.
        
        Args:
            y_true: True target values
            y_pred_proba: Predicted probabilities
            problem_type: Type of machine learning problem
            vis_dir: Directory to save visualizations
        """
        plt.figure(figsize=(10, 8))
        
        if problem_type == "binary_classification":
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
        else:  # Multiclass
            y_bin = label_binarize(y_true, classes=np.unique(y_true))
            for i in range(y_bin.shape[1]):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f"Class {i} (area = {roc_auc:.2f})")
        
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(vis_dir, "roc_curve.png"))
        self.visualizations["roc_curve"] = "roc_curve.png"
    
    def _plot_pr_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, problem_type: str, vis_dir: str) -> None:
        """
        Plot and save Precision-Recall curve.
        
        Args:
            y_true: True target values
            y_pred_proba: Predicted probabilities
            problem_type: Type of machine learning problem
            vis_dir: Directory to save visualizations
        """
        plt.figure(figsize=(10, 8))
        
        if problem_type == "binary_classification":
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, color="blue", lw=2, label=f"PR curve (area = {pr_auc:.2f})")
        else:  # Multiclass
            y_bin = label_binarize(y_true, classes=np.unique(y_true))
            for i in range(y_bin.shape[1]):
                precision, recall, _ = precision_recall_curve(y_bin[:, i], y_pred_proba[:, i])
                pr_auc = auc(recall, precision)
                plt.plot(recall, precision, lw=2, label=f"Class {i} (area = {pr_auc:.2f})")
        
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall (PR) Curve")
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(vis_dir, "pr_curve.png"))
        self.visualizations["pr_curve"] = "pr_curve.png"
    
    def _plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, vis_dir: str) -> None:
        """
        Plot and save residuals plot.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            vis_dir: Directory to save visualizations
        """
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 8))
        sns.histplot(residuals, kde=True, color="blue", bins=30)
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.title("Residuals Distribution")
        plt.savefig(os.path.join(vis_dir, "residuals.png"))
        self.visualizations["residuals"] = "residuals.png"
    
    def _plot_feature_importance(self, model: Any, vis_dir: str) -> None:
        """
        Plot and save feature importance plot.
        
        Args:
            model: Trained model object
            vis_dir: Directory to save visualizations
        """
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(10, 8))
            plt.title("Feature Importances")
            plt.bar(range(len(importances)), importances[indices], align="center")
            plt.xticks(range(len(importances)), [self.feature_names[i] for i in indices], rotation=90)
            plt.xlabel("Feature")
            plt.ylabel("Importance")
            plt.savefig(os.path.join(vis_dir, "feature_importance.png"))
            self.visualizations["feature_importance"] = "feature_importance.png"
    
    def _plot_learning_curve(self, model: Any, X: np.ndarray, y: np.ndarray, vis_dir: str) -> None:
        """
        Plot and save learning curve.
        
        Args:
            model: Trained model object
            X: Training features
            y: Training targets
            vis_dir: Directory to save visualizations
        """
        train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, n_jobs=-1)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=(10, 8))
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.title("Learning Curve")
        plt.legend(loc="best")
        plt.savefig(os.path.join(vis_dir, "learning_curve.png"))
        self.visualizations["learning_curve"] = "learning_curve.png"
    
    def _evaluate_interpretability(self, model: Any, X: np.ndarray, problem_type: str) -> None:
        """
        Evaluate model interpretability using SHAP and permutation importance.
        
        Args:
            model: Trained model object
            X: Training features
            problem_type: Type of machine learning problem
        """
        interpretability_config = self.config["evaluation"]["interpretability"]
        methods = interpretability_config["methods"]
        max_display = interpretability_config["max_display"]
        
        interpretability_results = {}
        
        if "shap" in methods:
            try:
                explainer = shap.Explainer(model, X)
                shap_values = explainer(X)
                shap.summary_plot(shap_values, X, feature_names=self.feature_names, max_display=max_display, show=False)
                plt.savefig(os.path.join(self.config["reporting"]["output_dir"], "shap_summary.png"))
                interpretability_results["shap_summary"] = "shap_summary.png"
            except Exception as e:
                logger.error(f"Error calculating SHAP values: {str(e)}")
        
        if "permutation_importance" in methods:
            try:
                result = permutation_importance(model, X, n_repeats=10, random_state=42, n_jobs=-1)
                perm_importance = result.importances_mean
                indices = np.argsort(perm_importance)[::-1]
                plt.figure(figsize=(10, 8))
                plt.title("Permutation Importances")
                plt.bar(range(len(perm_importance)), perm_importance[indices], align="center")
                plt.xticks(range(len(perm_importance)), [self.feature_names[i] for i in indices], rotation=90)
                plt.xlabel("Feature")
                plt.ylabel("Importance")
                plt.savefig(os.path.join(self.config["reporting"]["output_dir"], "permutation_importance.png"))
                interpretability_results["permutation_importance"] = "permutation_importance.png"
            except Exception as e:
                logger.error(f"Error calculating permutation importance: {str(e)}")
        
        self.evaluation_results["interpretability"] = interpretability_results
    
    def _perform_statistical_tests(self, model: Any, X: np.ndarray, y: np.ndarray, problem_type: str) -> None:
        """
        Perform statistical tests on model predictions.
        
        Args:
            model: Trained model object
            X: Training features
            y: Training targets
            problem_type: Type of machine learning problem
        """
        statistical_tests_config = self.config["evaluation"]["statistical_tests"]
        significance_level = statistical_tests_config["significance_level"]
        bootstrap_iterations = statistical_tests_config["bootstrap_iterations"]
        
        statistical_tests_results = {}
        
        if "regression" in problem_type:
            try:
                y_pred = model.predict(X)
                residuals = y - y_pred
                
                # Perform Shapiro-Wilk test for normality
                shapiro_test = stats.shapiro(residuals)
                statistical_tests_results["shapiro_wilk"] = {
                    "statistic": float(shapiro_test.statistic),
                    "p_value": float(shapiro_test.pvalue)
                }
                
                # Perform bootstrap confidence intervals for mean and variance
                bootstrap_means = []
                bootstrap_vars = []
                for _ in range(bootstrap_iterations):
                    sample = resample(residuals)
                    bootstrap_means.append(np.mean(sample))
                    bootstrap_vars.append(np.var(sample))
                
                mean_ci = np.percentile(bootstrap_means, [2.5, 97.5])
                var_ci = np.percentile(bootstrap_vars, [2.5, 97.5])
                
                statistical_tests_results["bootstrap_mean_ci"] = mean_ci.tolist()
                statistical_tests_results["bootstrap_var_ci"] = var_ci.tolist()
            except Exception as e:
                logger.error(f"Error performing statistical tests: {str(e)}")
        
        self.evaluation_results["statistical_tests"] = statistical_tests_results
    
    def _evaluate_calibration(
        self, 
        model: Any, 
        X: np.ndarray, 
        y: np.ndarray, 
        X_test: Optional[np.ndarray], 
        y_test: Optional[np.ndarray]
    ) -> None:
        """
        Evaluate model calibration for classification problems.
        
        Args:
            model: Trained model object
            X: Training features
            y: Training targets
            X_test: Test features
            y_test: Test targets
        """
        calibration_config = self.config["evaluation"]["calibration"]
        method = calibration_config["method"]
        bins = calibration_config["bins"]
        
        calibration_results = {}
        
        try:
            if X_test is not None and y_test is not None:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=bins, strategy=method)
                
                plt.figure(figsize=(10, 8))
                plt.plot(prob_pred, prob_true, marker="o", label="Calibration curve")
                plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
                plt.xlabel("Predicted probability")
                plt.ylabel("True probability")
                plt.title("Calibration Curve")
                plt.legend(loc="best")
                plt.savefig(os.path.join(self.config["reporting"]["output_dir"], "calibration_curve.png"))
                calibration_results["calibration_curve"] = "calibration_curve.png"
        except Exception as e:
            logger.error(f"Error evaluating calibration: {str(e)}")
        
        self.evaluation_results["calibration"] = calibration_results
    
    def _log_to_mlflow(self) -> None:
        """
        Log evaluation results to MLflow.
        """
        try:
            mlflow.log_params(self.config)
            mlflow.log_metrics(self.metrics)
            
            for key, value in self.visualizations.items():
                mlflow.log_artifact(os.path.join(self.config["reporting"]["output_dir"], value))
            
            for key, value in self.evaluation_results.items():
                if isinstance(value, (int, float, str)):
                    mlflow.log_metric(key, value)
                elif isinstance(value, dict):
                    mlflow.log_dict(value, f"{key}.json")
        except Exception as e:
            logger.error(f"Error logging to MLflow: {str(e)}")
    
    def _generate_reports(self) -> None:
        """
        Generate evaluation reports in specified formats.
        """
        output_dir = self.config["reporting"]["output_dir"]
        formats = self.config["reporting"]["format"]
        
        try:
            if "json" in formats:
                with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
                    json.dump(self.evaluation_results, f, indent=4)
            
            if "csv" in formats:
                pd.DataFrame(self.evaluation_results).to_csv(os.path.join(output_dir, "evaluation_results.csv"), index=False)
            
            if "html" in formats:
                pd.DataFrame(self.evaluation_results).to_html(os.path.join(output_dir, "evaluation_results.html"), index=False)
        except Exception as e:
            logger.error(f"Error generating reports: {str(e)}")
    
    def _evaluate_model(
        self,
        model: Any, 
        X: np.ndarray, 
        y_true: np.ndarray, 
        y_pred: Optional[np.ndarray] = None,
        y_prob: Optional[np.ndarray] = None,
        problem_type: str = None,
        is_test_set: bool = True,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, Union[float, Dict]]:
        """
        Comprehensive model evaluation for any model type.
        
        Args:
            model: Trained model object
            X: Feature data
            y_true: True target values
            y_pred: Predicted values (if already computed)
            y_prob: Predicted probabilities for classification (if already computed)
            problem_type: Type of problem ('regression', 'binary_classification', 'multiclass_classification')
                         If None, will be auto-detected
            is_test_set: Whether this is test data (True) or training data (False)
            sample_weight: Optional sample weights for weighted metrics
            
        Returns:
            Dictionary containing all relevant metrics
        """
        metrics = {}
        
        try:
            # Auto-detect problem type if not provided
            if problem_type is None:
                problem_type = auto_detect_problem_type(y_true)
                logger.info(f"Auto-detected problem type: {problem_type}")
            
            # Make predictions if not provided
            if y_pred is None:
                try:
                    y_pred = model.predict(X)
                    logger.debug(f"Generated predictions with shape {y_pred.shape}")
                except Exception as e:
                    logger.error(f"Error making predictions: {str(e)}")
                    metrics["error"] = f"Prediction error: {str(e)}"
                    return metrics
            
            # Get prediction probabilities for classification if not provided
            if y_prob is None and problem_type in ["binary_classification", "multiclass_classification"]:
                try:
                    if hasattr(model, "predict_proba"):
                        y_prob = model.predict_proba(X)
                        logger.debug(f"Generated probability predictions with shape {y_prob.shape}")
                    elif hasattr(model, "decision_function"):
                        y_prob = model.decision_function(X)
                        logger.debug("Used decision_function for probabilities")
                except Exception as e:
                    logger.warning(f"Could not generate probability predictions: {str(e)}")
            
            # Calculate appropriate metrics based on problem type
            if "regression" in problem_type:
                metrics = self._evaluate_regression_metrics(y_true, y_pred, sample_weight)
            elif problem_type == "binary_classification":
                metrics = self._evaluate_binary_classification_metrics(y_true, y_pred, y_prob, sample_weight)
            elif problem_type == "multiclass_classification":
                metrics = self._evaluate_multiclass_classification_metrics(y_true, y_pred, y_prob, sample_weight)
            else:
                logger.warning(f"Unknown problem type: {problem_type}")
                metrics["error"] = f"Unknown problem type: {problem_type}"
                
            # Add prediction time metrics if it's a test set (more relevant for inference speed)
            if is_test_set:
                metrics["prediction_metrics"] = self._measure_prediction_performance(model, X)
                
            # Detect edge cases and add warnings
            edge_case_metrics = self._detect_edge_cases(y_true, y_pred, problem_type)
            if edge_case_metrics:
                metrics["edge_cases"] = edge_case_metrics
                
            # Add statistical significance of results for test sets
            if is_test_set and self.config["evaluation"]["statistical_tests"]["enabled"]:
                significance = self._calculate_statistical_significance(y_true, y_pred, problem_type)
                metrics["statistical_significance"] = significance
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}", exc_info=True)
            metrics["error"] = str(e)
            return metrics

    def _evaluate_regression_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate regression-specific metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            sample_weight: Optional sample weights
            
        Returns:
            Dictionary of regression metrics
        """
        metrics = {}
        metrics_list = self.config["evaluation"]["metrics"]["regression"]
        
        try:
            # Basic regression metrics
            if "mse" in metrics_list:
                metrics["mse"] = float(mean_squared_error(y_true, y_pred, sample_weight=sample_weight))
            
            if "rmse" in metrics_list or True:  # Always calculate RMSE
                metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight)))
            
            if "mae" in metrics_list:
                metrics["mae"] = float(mean_absolute_error(y_true, y_pred, sample_weight=sample_weight))
            
            if "r2" in metrics_list:
                metrics["r2"] = float(r2_score(y_true, y_pred, sample_weight=sample_weight))
            
            if "explained_variance" in metrics_list:
                metrics["explained_variance"] = float(explained_variance_score(y_true, y_pred, sample_weight=sample_weight))
            
            # Advanced metrics
            if "median_absolute_error" in metrics_list:
                from sklearn.metrics import median_absolute_error
                metrics["median_absolute_error"] = float(median_absolute_error(y_true, y_pred))
            
            if "max_error" in metrics_list:
                from sklearn.metrics import max_error
                metrics["max_error"] = float(max_error(y_true, y_pred))
            
            # Add residuals statistics
            residuals = y_true - y_pred
            metrics["residuals_mean"] = float(np.mean(residuals))
            metrics["residuals_std"] = float(np.std(residuals))
            metrics["residuals_skew"] = float(stats.skew(residuals))
            metrics["residuals_kurtosis"] = float(stats.kurtosis(residuals))
            
            # Residuals normality test (Shapiro-Wilk)
            try:
                if len(residuals) <= 5000:  # Shapiro-Wilk has a sample size limitation
                    shapiro_test = stats.shapiro(residuals)
                    metrics["residuals_normality_pvalue"] = float(shapiro_test.pvalue)
                    metrics["residuals_normal_dist"] = shapiro_test.pvalue > 0.05
                else:
                    # For large samples, use Anderson-Darling test
                    ad_test = stats.anderson(residuals, 'norm')
                    metrics["residuals_anderson_stat"] = float(ad_test.statistic)
            except Exception as e:
                logger.warning(f"Error in normality test: {str(e)}")
            
            # Heteroscedasticity test (Breusch-Pagan)
            try:
                from statsmodels.stats.diagnostic import het_breuschpagan
                bp_test = het_breuschpagan(residuals, X=np.ones((len(residuals), 1)))
                metrics["heteroscedasticity_pvalue"] = float(bp_test[1])
                metrics["is_heteroscedastic"] = bp_test[1] < 0.05
            except Exception as e:
                logger.warning(f"Error in heteroscedasticity test: {str(e)}")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error calculating regression metrics: {str(e)}")
            return {"error": str(e)}

    def _evaluate_binary_classification_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate binary classification metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            y_prob: Predicted probabilities
            sample_weight: Optional sample weights
            
        Returns:
            Dictionary of binary classification metrics
        """
        metrics = {}
        metrics_list = self.config["evaluation"]["metrics"]["classification"]
        
        try:
            # Basic classification metrics
            if "accuracy" in metrics_list:
                metrics["accuracy"] = float(accuracy_score(y_true, y_pred, sample_weight=sample_weight))
            
            if "precision" in metrics_list:
                metrics["precision"] = float(precision_score(y_true, y_pred, average='binary', sample_weight=sample_weight))
            
            if "recall" in metrics_list:
                metrics["recall"] = float(recall_score(y_true, y_pred, average='binary', sample_weight=sample_weight))
                metrics["sensitivity"] = metrics["recall"]  # Alias for medical applications
            
            if "f1" in metrics_list:
                metrics["f1"] = float(f1_score(y_true, y_pred, average='binary', sample_weight=sample_weight))
            
            # Specificity (true negative rate)
            if "specificity" in metrics_list:
                from sklearn.metrics import confusion_matrix
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            
            # ROC AUC if probabilities are available
            if "roc_auc" in metrics_list and y_prob is not None:
                if y_prob.ndim > 1 and y_prob.shape[1] >= 2:
                    # Use the positive class probability
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1], sample_weight=sample_weight))
                else:
                    # Use the provided scores directly
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob, sample_weight=sample_weight))
            
            # PR AUC if probabilities are available
            if "pr_auc" in metrics_list and y_prob is not None:
                if y_prob.ndim > 1 and y_prob.shape[1] >= 2:
                    # Use the positive class probability
                    metrics["pr_auc"] = float(average_precision_score(y_true, y_prob[:, 1], sample_weight=sample_weight))
                else:
                    metrics["pr_auc"] = float(average_precision_score(y_true, y_prob, sample_weight=sample_weight))
            
            # Log loss if probabilities are available
            if "log_loss" in metrics_list and y_prob is not None:
                try:
                    metrics["log_loss"] = float(log_loss(y_true, y_prob, sample_weight=sample_weight))
                except Exception as e:
                    logger.warning(f"Error calculating log_loss: {str(e)}")
            
            # Brier score for probability calibration
            if "brier_score" in metrics_list and y_prob is not None:
                from sklearn.metrics import brier_score_loss
                try:
                    if y_prob.ndim > 1 and y_prob.shape[1] >= 2:
                        metrics["brier_score"] = float(brier_score_loss(y_true, y_prob[:, 1], sample_weight=sample_weight))
                    else:
                        metrics["brier_score"] = float(brier_score_loss(y_true, y_prob, sample_weight=sample_weight))
                except Exception as e:
                    logger.warning(f"Error calculating brier_score: {str(e)}")
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred).tolist()
            metrics["confusion_matrix"] = cm
            
            # Calculate class distribution and imbalance metrics
            class_distribution = np.unique(y_true, return_counts=True)[1]
            metrics["class_distribution"] = class_distribution.tolist()
            metrics["class_imbalance_ratio"] = float(np.max(class_distribution) / np.min(class_distribution))
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error calculating binary classification metrics: {str(e)}")
            return {"error": str(e)}

    def _evaluate_multiclass_classification_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calculate multiclass classification metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            y_prob: Predicted probabilities
            sample_weight: Optional sample weights
            
        Returns:
            Dictionary of multiclass classification metrics
        """
        metrics = {}
        metrics_list = self.config["evaluation"]["metrics"]["classification"]
        
        try:
            # Basic multiclass metrics
            if "accuracy" in metrics_list:
                metrics["accuracy"] = float(accuracy_score(y_true, y_pred, sample_weight=sample_weight))
            
            if "precision" in metrics_list:
                metrics["precision_micro"] = float(precision_score(y_true, y_pred, average='micro', sample_weight=sample_weight))
                metrics["precision_macro"] = float(precision_score(y_true, y_pred, average='macro', sample_weight=sample_weight))
                metrics["precision_weighted"] = float(precision_score(y_true, y_pred, average='weighted', sample_weight=sample_weight))
            
            if "recall" in metrics_list:
                metrics["recall_micro"] = float(recall_score(y_true, y_pred, average='micro', sample_weight=sample_weight))
                metrics["recall_macro"] = float(recall_score(y_true, y_pred, average='macro', sample_weight=sample_weight))
                metrics["recall_weighted"] = float(recall_score(y_true, y_pred, average='weighted', sample_weight=sample_weight))
            
            if "f1" in metrics_list:
                metrics["f1_micro"] = float(f1_score(y_true, y_pred, average='micro', sample_weight=sample_weight))
                metrics["f1_macro"] = float(f1_score(y_true, y_pred, average='macro', sample_weight=sample_weight))
                metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average='weighted', sample_weight=sample_weight))
            
            # Add per-class metrics
            unique_classes = np.unique(np.concatenate([y_true, y_pred]))
            per_class_metrics = {}
            
            for cls in unique_classes:
                cls_metrics = {}
                binary_y_true = (y_true == cls).astype(int)
                binary_y_pred = (y_pred == cls).astype(int)
                
                cls_metrics["precision"] = float(precision_score(binary_y_true, binary_y_pred, sample_weight=sample_weight))
                cls_metrics["recall"] = float(recall_score(binary_y_true, binary_y_pred, sample_weight=sample_weight))
                cls_metrics["f1"] = float(f1_score(binary_y_true, binary_y_pred, sample_weight=sample_weight))
                
                # Add ROC AUC if probabilities are available
                if y_prob is not None and y_prob.shape[1] > cls:
                    try:
                        cls_metrics["roc_auc"] = float(roc_auc_score(binary_y_true, y_prob[:, cls], sample_weight=sample_weight))
                        cls_metrics["pr_auc"] = float(average_precision_score(binary_y_true, y_prob[:, cls], sample_weight=sample_weight))
                    except Exception as e:
                        logger.warning(f"Error calculating AUC for class {cls}: {str(e)}")
                
                per_class_metrics[f"class_{cls}"] = cls_metrics
            
            metrics["per_class"] = per_class_metrics
            
            # Multi-class ROC AUC if probabilities are available
            if "roc_auc" in metrics_list and y_prob is not None:
                try:
                    # One-vs-Rest ROC AUC
                    metrics["roc_auc_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class='ovr', sample_weight=sample_weight))
                    
                    # One-vs-One ROC AUC
                    try:
                        metrics["roc_auc_ovo"] = float(roc_auc_score(y_true, y_prob, multi_class='ovo', sample_weight=sample_weight))
                    except Exception:
                        pass  # OVO can fail with certain conditions
                except Exception as e:
                    logger.warning(f"Error calculating multiclass ROC AUC: {str(e)}")
            
            # Log loss if probabilities are available
            if "log_loss" in metrics_list and y_prob is not None:
                try:
                    metrics["log_loss"] = float(log_loss(y_true, y_prob, sample_weight=sample_weight))
                except Exception as e:
                    logger.warning(f"Error calculating log_loss: {str(e)}")
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred).tolist()
            metrics["confusion_matrix"] = cm
            
            # Calculate class distribution and imbalance metrics
            class_distribution = np.unique(y_true, return_counts=True)[1]
            metrics["class_distribution"] = class_distribution.tolist()
            metrics["class_imbalance_ratio"] = float(np.max(class_distribution) / np.min(class_distribution))
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error calculating multiclass classification metrics: {str(e)}")
            return {"error": str(e)}

    def _measure_prediction_performance(self, model: Any, X: np.ndarray) -> Dict[str, float]:
        """
        Measure model prediction performance (speed, memory usage).
        
        Args:
            model: Trained model
            X: Input data
            
        Returns:
            Dictionary of performance metrics
        """
        metrics = {}
        
        try:
            # Measure prediction time
            start_time = time.time()
            model.predict(X)
            prediction_time = time.time() - start_time
            
            metrics["prediction_time_seconds"] = prediction_time
            metrics["predictions_per_second"] = len(X) / prediction_time if prediction_time > 0 else 0
            
            # Try to get memory usage
            try:
                import psutil
                process = psutil.Process(os.getpid())
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                model.predict(X)
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                metrics["memory_usage_mb"] = memory_after - memory_before
            except ImportError:
                logger.warning("psutil not available for memory measurements")
            except Exception as e:
                logger.warning(f"Error measuring memory usage: {str(e)}")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error measuring prediction performance: {str(e)}")
            return {"error": str(e)}

    def _detect_edge_cases(self, y_true: np.ndarray, y_pred: np.ndarray, problem_type: str) -> Dict[str, Any]:
        """
        Detect potential edge cases or issues in model predictions.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            problem_type: Type of machine learning problem
            
        Returns:
            Dictionary of edge case metrics
        """
        edge_cases = {}
        
        try:
            if "regression" in problem_type:
                # Check for extreme residuals
                residuals = y_true - y_pred
                std_residual = np.std(residuals)
                extreme_residuals = np.abs(residuals) > 3 * std_residual
                edge_cases["extreme_residuals_count"] = int(np.sum(extreme_residuals))
                edge_cases["extreme_residuals_percentage"] = float(np.mean(extreme_residuals) * 100)
                
                # Check for predictions outside the range of training data
                min_y_true, max_y_true = np.min(y_true), np.max(y_true)
                out_of_range = (y_pred < min_y_true) | (y_pred > max_y_true)
                edge_cases["out_of_range_predictions_count"] = int(np.sum(out_of_range))
                edge_cases["out_of_range_predictions_percentage"] = float(np.mean(out_of_range) * 100)
                
            else:  # Classification
                # Check for high confidence errors
                if hasattr(self, 'y_prob') and self.y_prob is not None:
                    y_prob_max = np.max(self.y_prob, axis=1)
                    high_conf_errors = (y_true != y_pred) & (y_prob_max > 0.9)
                    edge_cases["high_confidence_errors_count"] = int(np.sum(high_conf_errors))
                    edge_cases["high_confidence_errors_percentage"] = float(np.mean(high_conf_errors) * 100 if len(high_conf_errors) > 0 else 0)
                
                # Check for classes never predicted
                unique_true = set(np.unique(y_true))
                unique_pred = set(np.unique(y_pred))
                never_predicted = unique_true - unique_pred
                if never_predicted:
                    edge_cases["never_predicted_classes"] = [int(c) for c in never_predicted]
                
                # Check for classes that don't exist in true data
                hallucinated = unique_pred - unique_true
                if hallucinated:
                    edge_cases["hallucinated_classes"] = [int(c) for c in hallucinated]
            
            # Check for NaNs or infs in predictions
            invalid_preds = ~np.isfinite(y_pred)
            edge_cases["invalid_predictions_count"] = int(np.sum(invalid_preds))
            
            return edge_cases
        
        except Exception as e:
            logger.error(f"Error detecting edge cases: {str(e)}")
            return {"error": str(e)}

    def _calculate_statistical_significance(self, y_true: np.ndarray, y_pred: np.ndarray, problem_type: str) -> Dict[str, Any]:
        """
        Calculate statistical significance of the model's predictions.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            problem_type: Type of machine learning problem
            
        Returns:
            Dictionary of statistical significance metrics
        """
        significance = {}
        
        try:
            if "regression" in problem_type:
                # Permutation test for R²
                r2_original = r2_score(y_true, y_pred)
                r2_permutation = []
                n_permutations = min(1000, 10 * len(y_true))  # Cap at 1000 permutations
                
                for _ in range(n_permutations):
                    y_permuted = np.random.permutation(y_true)
                    r2_permutation.append(r2_score(y_permuted, y_pred))
                
                p_value = np.mean(np.array(r2_permutation) >= r2_original)
                significance["r2_p_value"] = float(p_value)
                significance["r2_is_significant"] = p_value < 0.05
                
            else:  # Classification
                # Chi-squared test for independence between true and predicted labels
                from scipy.stats import chi2_contingency
                contingency_table = confusion_matrix(y_true, y_pred)
                chi2, p_value, _, _ = chi2_contingency(contingency_table)
                
                significance["chi2"] = float(chi2)
                significance["chi2_p_value"] = float(p_value)
                significance["predictions_dependent_on_true"] = p_value < 0.05
            
            return significance
        except Exception as e:
            logger.error(f"Error calculating statistical significance: {str(e)}")
            return {"error": str(e)}


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args)
    
    # Load model
    model = joblib.load(args.model_path)
    
    # Load data
    data = pd.read_csv(args.data_path)
    X = data.drop(columns=[args.target_col]).values
    y = data[args.target_col].values
    
    # Initialize evaluator
    evaluator = ModelEvaluator(config)
    
    # Evaluate model
    results = evaluator.evaluate(model, X, y, problem_type=args.problem_type)
    
    # Print results
    print(json.dumps(results, indent=4))