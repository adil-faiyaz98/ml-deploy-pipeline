"""
Model Training Pipeline

This module implements a comprehensive model training pipeline with:
- Configuration management
- Experiment tracking with MLflow
- Distributed training support
- Multi-model ensemble capabilities
- Advanced regularization techniques
- Cross-validation strategies
- Incremental training support
- Model serialization and export
- Detailed performance metrics
- Resource optimization
"""

import argparse
import gc
import json
import logging
import os
import pickle
import sys
import time
import warnings
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import ray
import tensorflow as tf
import torch
import yaml
from optuna.integration import OptunaSearchCV
from packaging import version
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
    StackingClassifier, StackingRegressor,
    VotingClassifier, VotingRegressor
)
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score
)
from sklearn.model_selection import (
    GroupKFold, KFold, LeaveOneGroupOut, RepeatedStratifiedKFold,
    TimeSeriesSplit, cross_val_score, train_test_split
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer,
    RobustScaler, StandardScaler
)
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/model_training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("model_training")

# Suppress warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TensorFlow verbosity

# Constants
PROBLEM_TYPES = ["binary_classification", "multiclass_classification", "regression"]
ACCEPTED_MODEL_TYPES = [
    "random_forest", "gradient_boosting", "xgboost", "linear", "svm", 
    "neural_network", "ensemble", "custom", "lightgbm", "catboost",
    "deep_learning", "transformer", "automl"
]


class ModelTrainer:
    """
    Comprehensive model training and experimentation framework.
    
    This class handles the entire model development process, including:
    1. Data preprocessing and feature engineering
    2. Hyperparameter optimization
    3. Model training with cross-validation
    4. Performance evaluation
    5. Model serialization and export
    6. Experiment tracking
    """
    
    def __init__(self, config_path: str = None, config: Dict = None):
        """
        Initialize the model trainer with configuration.
        
        Args:
            config_path: Path to configuration YAML file
            config: Configuration dictionary (overrides config_path if provided)
        """
        # Initialize default configuration
        self.config = self._get_default_config()
        
        # Load configuration from file if provided
        if config_path:
            self._load_config_from_file(config_path)
            
        # Override configuration from dictionary if provided
        if config:
            self._update_config(config)
            
        # Initialize MLflow
        self._setup_mlflow()
        
        # Initialize random state for reproducibility
        self._set_random_seed(self.config["training"]["random_state"])
        
        # Initialize empty attributes
        self.models = {}
        self.best_model = None
        self.best_model_params = None
        self.best_pipeline = None
        self.best_score = -np.inf
        self.feature_importances = None
        self.training_history = {}
        self.preprocessor = None
        self.feature_pipeline = None
        self.calibration_model = None
        self.model_size_bytes = 0
        self.dataset_statistics = {}
        self.run_id = None
        
        # Create output directories
        for directory in [
            self.config["output"]["model_dir"],
            self.config["output"]["metrics_dir"],
            "logs",
            "artifacts"
        ]:
            os.makedirs(directory, exist_ok=True)
        
        # Configure Ray for distributed training if enabled
        if self.config["distributed_training"]["enabled"]:
            self._setup_distributed_training()
            
        # Set up GPU if available and configured
        if self.config["distributed_training"]["use_gpu"]:
            self._setup_gpu()
        
        logger.info(f"ModelTrainer initialized with problem type: {self.config['data']['problem_type']}")
        
    def _get_default_config(self) -> Dict:
        """
        Get default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "data": {
                "train_data_path": "data/train.csv",
                "validation_data_path": "data/validation.csv",
                "test_data_path": "data/test.csv",
                "target_column": "target",
                "feature_columns": None,  # Use all except target if None
                "categorical_columns": [],
                "numerical_columns": [],
                "datetime_columns": [],
                "text_columns": [],
                "id_columns": [],  # These will be excluded from model training
                "problem_type": "binary_classification",  # binary_classification, multiclass_classification, regression
                "stratify": True,
                "train_size": 0.8,
                "validation_size": 0.1,
                "test_size": 0.1,
                "random_state": 42,
                "drop_na": True,
                "impute_strategy": "median",  # None, "mean", "median", "most_frequent", "constant"
                "sampling_strategy": None,  # None, "over", "under", "smote"
                "weight_column": None,  # Column name for instance weights
                "time_column": None,  # Column for time series forecasting
                "group_column": None,  # Column for grouped cross-validation
                "max_rows": None,  # Limit number of rows for debugging
                "data_format": "csv",  # csv, parquet, sqlite, bigquery, etc.
                "s3_bucket": None,  # Optional S3 bucket for data
                "data_version": "latest"  # Data version (for versioned datasets)
            },
            "preprocessing": {
                "scaling": "standard",  # None, "standard", "minmax", "robust", "power"
                "encoding": "onehot",  # None, "onehot", "ordinal", "target", "binary"
                "feature_selection": {
                    "enabled": False,
                    "method": "rfe",  # "rfe", "selectkbest", "from_model"
                    "n_features": 0.8  # Can be int or float (proportion)
                },
                "dimensionality_reduction": {
                    "enabled": False,
                    "method": "pca",  # "pca", "tsne", "umap"
                    "n_components": 0.95
                },
                "feature_engineering": {
                    "polynomial_features": False,
                    "interaction_terms": False,
                    "clustering_features": False,
                    "custom_transformers": []
                },
                "outlier_removal": {
                    "enabled": False,
                    "method": "iqr",  # "iqr", "z_score", "isolation_forest"
                    "threshold": 3.0
                },
                "custom_preprocessing": None  # Path to custom preprocessing script
            },
            "model": {
                "type": "random_forest",  # See ACCEPTED_MODEL_TYPES
                "params": {},
                "ensemble": {
                    "enabled": False,
                    "models": ["random_forest", "gradient_boosting", "xgboost"],
                    "voting": "soft"  # "soft", "hard"
                },
                "stacking": {
                    "enabled": False,
                    "base_models": ["random_forest", "gradient_boosting", "xgboost"],
                    "meta_model": "logistic_regression"
                },
                "custom_model_path": None,
                "pretrained_model_path": None,
                "model_checkpoint": None,
                "model_registry_uri": None
            },
            "training": {
                "cv_strategy": "kfold",  # None, "kfold", "stratified_kfold", "group_kfold", "time_series_split"
                "cv_folds": 5,
                "scoring": "accuracy",  # Primary metric for optimization
                "early_stopping": True,
                "early_stopping_rounds": 10,
                "early_stopping_metric": "validation_loss",
                "class_weight": "balanced",  # None, "balanced", "balanced_subsample", {0: w0, 1: w1, ...}
                "sample_weight_column": None,
                "random_state": 42,
                "epochs": 100,
                "batch_size": 32,
                "incremental_training": False,
                "warm_start": False,
                "save_checkpoints": True,
                "checkpoint_frequency": 5,  # Save every N epochs
                "use_mixed_precision": False,
                "gradient_accumulation_steps": 1,
                "max_train_time_seconds": 3600  # Maximum training time in seconds
            },
            "hyperparameter_optimization": {
                "enabled": True,
                "method": "optuna",  # "optuna", "grid", "random", "bayesian", "hyperopt", "ray"
                "n_trials": 100,
                "timeout": 3600,  # Seconds
                "cv": 3,
                "scoring": "accuracy",  # Can be different from training scoring
                "direction": "maximize",  # "maximize", "minimize"
                "param_grid": None,  # Optionally specified param grid
                "search_space": None,  # Custom search space definition
                "pruner": "hyperband",  # "hyperband", "median", "none"
                "n_jobs": -1,
                "refit": True,
                "study_name": None  # Name for Optuna study persistence
            },
            "distributed_training": {
                "enabled": False,
                "num_workers": 4,
                "use_gpu": False,
                "gpu_per_worker": 0.5,
                "framework": "ray",  # "ray", "dask", "spark", "horovod"
                "strategy": "data_parallel",  # "data_parallel", "model_parallel"
                "batch_size_per_worker": 32,
                "communication_backend": "nccl",  # "nccl", "gloo"
                "synchronization_frequency": 1,  # How often workers sync in epochs
                "parameter_server": False,  # Use parameter server architecture
                "cluster_address": None  # Address if using existing cluster
            },
            "mlflow": {
                "enabled": True,
                "tracking_uri": None,
                "experiment_name": "model_training",
                "register_model": True,
                "registry_model_name": None,
                "log_artifacts": True,
                "log_data_profile": True,
                "log_explanations": True,
                "tags": {},  # Additional tags for MLflow runs
                "auto_log": True,  # Automatic logging of parameters/metrics
                "log_system_metrics": True,  # Log system metrics (CPU, memory)
                "log_code": True,  # Save source code snapshot
                "allow_infer_pip_requirements": True
            },
            "evaluation": {
                "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
                "calibration": True,
                "feature_importance": True,
                "cross_validation": True,
                "test_split_evaluation": True,
                "prediction_drift_detection": True,
                "explanations": True,  # Generate model explanations
                "explanation_method": "shap",  # "shap", "lime", "permutation"
                "fairness_metrics": False,  # Evaluate model fairness
                "fairness_sensitive_features": [],  # Features for fairness evaluation
                "bias_mitigation": False,  # Apply bias mitigation techniques
                "confidence_intervals": True,  # Compute confidence intervals for metrics
                "bootstrap_samples": 1000,  # Number of bootstrap samples
                "confusion_matrix": True,
                "pr_curve": True,
                "roc_curve": True
            },
            "output": {
                "model_dir": "models",
                "model_name": None,  # Auto-generated if None
                "metrics_dir": "metrics",
                "save_format": "joblib",  # "joblib", "pickle", "onnx", "savedmodel", "torchscript"
                "save_preprocessing": True,
                "multiple_formats": False,  # Save model in multiple formats
                "compress": True,  # Compress saved model
                "version_models": True,  # Add version suffix to model files
                "export_for_serving": False,  # Export model for TensorFlow Serving, etc.
                "export_platform": "tensorflow-serving"  # TensorFlow Serving, TorchServe, etc.
            },
            "advanced": {
                "memory_optimization": False,
                "mixed_precision": False,
                "verbosity": 2,
                "reproduce_results": True,
                "export_as_service": False,
                "production_ready": True,
                "metadata_tracking": True,
                "model_interpretability": True,
                "feature_store_integration": None,  # Feature store connection details
                "feature_store_enabled": False,
                "experiment_tracking_system": "mlflow",  # "mlflow", "wandb", "neptune", etc.
                "alert_on_failure": False,
                "alert_email": None,
                "parallel_preprocessing": False,
                "cache_preprocessed_data": False,
                "cache_dir": ".cache",
                "profile_code": False  # Enable code profiling
            },
            "monitoring": {
                "enabled": False,
                "metrics_to_monitor": ["accuracy", "drift"],
                "monitoring_period": "daily",  # "hourly", "daily", "weekly"
                "alert_threshold": 0.05,  # Alert if metric changes by this amount
                "baseline_drift_threshold": 0.1,  # Maximum acceptable data drift
                "performance_degradation_threshold": 0.1,  # Maximum acceptable performance drop
                "store_predictions": True,  # Store predictions for monitoring
                "monitoring_service_url": None  # URL of monitoring service
            }
        }
    
    def _load_config_from_file(self, config_path: str) -> None:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                
            logger.info(f"Loaded configuration from {config_path}")
            self._update_config(loaded_config)
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {str(e)}")
            raise
    
    def _update_config(self, new_config: Dict) -> None:
        """
        Update configuration with new values.
        
        Args:
            new_config: New configuration dictionary
        """
        def _deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    _deep_update(d[k], v)
                else:
                    d[k] = v
        
        _deep_update(self.config, new_config)
        
        # Validate configuration after update
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration settings."""
        # Validate problem type
        problem_type = self.config["data"]["problem_type"]
        if problem_type not in PROBLEM_TYPES:
            raise ValueError(f"Invalid problem_type: {problem_type}. Must be one of {PROBLEM_TYPES}")
            
        # Validate model type
        model_type = self.config["model"]["type"]
        if model_type not in ACCEPTED_MODEL_TYPES:
            raise ValueError(f"Invalid model_type: {model_type}. Must be one of {ACCEPTED_MODEL_TYPES}")
            
        # Validate custom model if specified
        if model_type == "custom" and not self.config["model"]["custom_model_path"]:
            raise ValueError("custom_model_path must be specified when model type is 'custom'")
            
        # Validate data paths
        train_path = self.config["data"]["train_data_path"]
        if not os.path.exists(train_path):
            logger.warning(f"Training data file not found: {train_path}. Will attempt to download or generate.")
            
        # Validate scoring metrics
        for scoring in [self.config["training"]["scoring"], self.config["hyperparameter_optimization"]["scoring"]]:
            if scoring not in ["accuracy", "precision", "recall", "f1", "roc_auc", 
                              "mean_squared_error", "mean_absolute_error", "r2", "neg_log_loss",
                              "neg_root_mean_squared_error"]:
                logger.warning(f"Unusual scoring metric: {scoring}. Make sure this is supported.")
                
        # Validate distributed training config
        if self.config["distributed_training"]["enabled"] and self.config["distributed_training"]["use_gpu"]:
            if self.config["distributed_training"]["framework"] not in ["ray", "horovod"]:
                logger.warning(f"GPU support might be limited with {self.config['distributed_training']['framework']} framework")
    
    def _set_random_seed(self, seed: int) -> None:
        """
        Set random seed for all libraries to ensure reproducibility.
        
        Args:
            seed: Random seed to use
        """
        if not self.config["advanced"]["reproduce_results"]:
            logger.info("Reproducibility not enforced - skipping random seed setting")
            return
            
        # Set seeds for libraries
        np.random.seed(seed)
        import random
        random.seed(seed)
        
        # Set PyTorch seed
        try:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
            
        # Set TensorFlow seed
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
            os.environ['PYTHONHASHSEED'] = str(seed)
        except ImportError:
            pass
            
        logger.info(f"Random seed set to {seed} across all frameworks")
    
    def _setup_mlflow(self) -> None:
        """Set up MLflow tracking."""
        if not self.config["mlflow"]["enabled"]:
            return
            
        # Set tracking URI if provided
        if self.config["mlflow"]["tracking_uri"]:
            mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
            
        # Create or set experiment
        experiment_name = self.config["mlflow"]["experiment_name"]
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow experiment set to: {experiment_name}")
            
            # Enable autologging if configured
            if self.config["mlflow"]["auto_log"]:
                mlflow.sklearn.autolog(log_models=True)
                
                if "xgboost" in sys.modules:
                    mlflow.xgboost.autolog()
                    
                if "pytorch" in sys.modules or "torch" in sys.modules:
                    mlflow.pytorch.autolog()
                    
                if "tensorflow" in sys.modules:
                    mlflow.tensorflow.autolog()
                
                logger.info("MLflow autologging enabled")
                
        except Exception as e:
            logger.warning(f"Failed to set up MLflow experiment: {str(e)}")
    
    def _setup_distributed_training(self) -> None:
        """Set up distributed training framework."""
        if self.config["distributed_training"]["framework"] == "ray":
            try:
                # Initialize Ray
                ray_resources = {}
                
                if self.config["distributed_training"]["use_gpu"]:
                    ray_resources["gpu"] = self.config["distributed_training"]["gpu_per_worker"]
                
                ray.shutdown()  # Ensure Ray is not already initialized
                
                # Use existing cluster if specified
                if self.config["distributed_training"]["cluster_address"]:
                    ray.init(
                        address=self.config["distributed_training"]["cluster_address"],
                        ignore_reinit_error=True
                    )
                    logger.info(f"Connected to Ray cluster at {self.config['distributed_training']['cluster_address']}")
                else:
                    ray.init(
                        num_cpus=self.config["distributed_training"]["num_workers"],
                        resources=ray_resources,
                        ignore_reinit_error=True
                    )
                    logger.info(f"Ray initialized with {self.config['distributed_training']['num_workers']} workers")
                    
            except Exception as e:
                logger.error(f"Failed to initialize Ray: {str(e)}")
                self.config["distributed_training"]["enabled"] = False
                
        elif self.config["distributed_training"]["framework"] == "horovod":
            try:
                import horovod.keras as hvd
                hvd.init()
                
                # Set up TensorFlow for Horovod
                import tensorflow as tf
                gpus = tf.config.experimental.list_physical_devices('GPU')
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                if gpus:
                    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
                    
                logger.info(f"Horovod initialized with {hvd.size()} workers")
                
            except ImportError:
                logger.error("Horovod not installed. Please install with: pip install horovod")
                self.config["distributed_training"]["enabled"] = False
            except Exception as e:
                logger.error(f"Failed to initialize Horovod: {str(e)}")
                self.config["distributed_training"]["enabled"] = False
    
    def _setup_gpu(self) -> None:
        """Set up GPU for training."""
        # Check if GPU is available with PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
                logger.info(f"PyTorch detected {device_count} CUDA devices: {device_names}")
                
                # Set default device to GPU
                self.torch_device = torch.device("cuda:0")
            else:
                logger.warning("PyTorch: No CUDA devices available, falling back to CPU")
                self.torch_device = torch.device("cpu")
                self.config["distributed_training"]["use_gpu"] = False
        except ImportError:
            logger.debug("PyTorch not available")
            
        # Check if GPU is available with TensorFlow
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Set memory growth to avoid grabbing all GPU memory
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"TensorFlow detected {len(gpus)} GPUs")
                except RuntimeError as e:
                    logger.error(f"Error setting up TensorFlow GPU: {e}")
            else:
                logger.warning("TensorFlow: No GPU devices available, falling back to CPU")
                self.config["distributed_training"]["use_gpu"] = False
        except ImportError:
            logger.debug("TensorFlow not available")
            
        # Check if mixed precision is available
        if self.config["training"]["use_mixed_precision"]:
            try:
                from tensorflow.keras import mixed_precision
                policy = mixed_precision.Policy('mixed_float16')
                mixed_precision.set_global_policy(policy)
                logger.info("Mixed precision training enabled with float16")
            except ImportError:
                logger.warning("Mixed precision requested but not available with installed TensorFlow")
            except Exception as e:
                logger.error(f"Error setting up mixed precision: {e}")
    
    def _get_data_loader(self, file_path: str) -> pd.DataFrame:
        """
        Load data from various file formats.
        
        Args:
            file_path: Path to data file
            
        Returns:
            Loaded DataFrame
        """
        # Extract file extension
        file_format = self.config["data"]["data_format"].lower()
        
        # Check if file exists
        if not os.path.exists(file_path):
            # Try S3 if configured
            if self.config["data"]["s3_bucket"]:
                try:
                    import boto3
                    s3_path = file_path.replace('s3://', '')
                    bucket_name = self.config["data"]["s3_bucket"]
                    key = '/'.join(s3_path.split('/')[1:])
                    
                    s3 = boto3.client('s3')
                    local_file = f"/tmp/{os.path.basename(file_path)}"
                    s3.download_file(bucket_name, key, local_file)
                    
                    file_path = local_file
                    logger.info(f"Downloaded {bucket_name}/{key} to {local_file}")
                except ImportError:
                    raise ImportError("boto3 is required for S3 access. Install with pip install boto3")
                except Exception as e:
                    raise ValueError(f"Error downloading from S3: {str(e)}")
            else:
                raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Load based on format
        if file_format == "csv":
            return pd.read_csv(file_path)
        elif file_format == "parquet":
            return pd.read_parquet(file_path)
        elif file_format == "json":
            return pd.read_json(file_path)
        elif file_format == "excel":
            return pd.read_excel(file_path)
        elif file_format == "sql" or file_format == "sqlite":
            import sqlite3
            conn = sqlite3.connect(file_path)
            query = "SELECT * FROM data"  # Default table name, should be configurable
            return pd.read_sql(query, conn)
        elif file_format == "bigquery":
            try:
                from google.cloud import bigquery
                client = bigquery.Client()
                query = f"SELECT * FROM `{file_path}`"
                return client.query(query).to_dataframe()
            except ImportError:
                raise ImportError("google-cloud-bigquery is required for BigQuery access")
        else:
            raise ValueError(f"Unsupported data format: {file_format}")

    def _log_dataset_statistics(self, X_train: pd.DataFrame, y_train: pd.Series,
                              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> None:
        """
        Log statistics about the dataset.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
        """
        stats = {
            "training_samples": len(X_train),
            "feature_count": X_train.shape[1],
            "validation_samples": len(X_val) if X_val is not None else 0,
            "memory_usage_mb": X_train.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Add problem type specific statistics
        if self.config["data"]["problem_type"] in ["binary_classification", "multiclass_classification"]:
            # Class distribution
            train_class_counts = y_train.value_counts().to_dict()
            stats["class_distribution"] = train_class_counts
            
            # Class balance metrics
            if len(train_class_counts) > 1:
                class_values = list(train_class_counts.values())
                class_imbalance = max(class_values) / min(class_values)
                stats["class_imbalance_ratio"] = class_imbalance
                
                # Log warning if highly imbalanced
                if class_imbalance > 10:
                    logger.warning(
                        f"High class imbalance detected: {class_imbalance:.2f}. "
                        "Consider using class weights or resampling."
                    )
                    
        elif self.config["data"]["problem_type"] == "regression":
            # Regression target statistics
            stats["target_mean"] = float(y_train.mean())
            stats["target_std"] = float(y_train.std())
            
    def load_and_preprocess_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Load and preprocess the dataset, splitting it into train, validation, and test sets.
        
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        start_time = time.time()
        logger.info("Loading and preprocessing data...")
        
        try:
            # Load training data
            train_data = self._get_data_loader(self.config["data"]["train_data_path"])
            
            # Apply row limit if specified (useful for debugging)
            if self.config["data"]["max_rows"]:
                train_data = train_data.head(self.config["data"]["max_rows"])
                logger.info(f"Limited data to {self.config['data']['max_rows']} rows")
            
            # Extract target and features
            target_col = self.config["data"]["target_column"]
            id_cols = self.config["data"]["id_columns"]
            
            # Validate target column exists
            if target_col not in train_data.columns:
                raise ValueError(f"Target column '{target_col}' not found in dataset")
            
            # Determine feature columns
            if self.config["data"]["feature_columns"]:
                feature_cols = self.config["data"]["feature_columns"]
            else:
                # Use all columns except target and ID columns
                feature_cols = [col for col in train_data.columns 
                               if col != target_col and col not in id_cols]
                
            logger.info(f"Selected {len(feature_cols)} features for training")
            
            # Extract features and target
            X = train_data[feature_cols]
            y = train_data[target_col]
            
            # Handle missing values
            if self.config["data"]["drop_na"]:
                initial_rows = len(X)
                X = X.dropna()
                y = y.iloc[X.index]
                dropped_rows = initial_rows - len(X)
                if dropped_rows > 0:
                    logger.warning(f"Dropped {dropped_rows} rows with missing values ({dropped_rows/initial_rows:.2%} of data)")
            
            # Handle separate validation and test sets if provided
            X_val, y_val, X_test, y_test = None, None, None, None
            
            if os.path.exists(self.config["data"]["validation_data_path"]):
                logger.info("Loading separate validation dataset")
                val_data = self._get_data_loader(self.config["data"]["validation_data_path"])
                X_val = val_data[feature_cols]
                y_val = val_data[target_col]
                
            if os.path.exists(self.config["data"]["test_data_path"]):
                logger.info("Loading separate test dataset")
                test_data = self._get_data_loader(self.config["data"]["test_data_path"])
                X_test = test_data[feature_cols]
                y_test = test_data[target_col]
            
            # If external datasets weren't loaded, split the training data
            if X_val is None or X_test is None:
                logger.info("Splitting data into train/validation/test")
                
                # First split: train+val vs test
                stratify_col = y if self.config["data"]["stratify"] else None
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y, 
                    test_size=self.config["data"]["test_size"],
                    random_state=self.config["data"]["random_state"],
                    stratify=stratify_col
                )
                
                # Second split: train vs validation
                if X_val is None:
                    # Adjust validation size to be relative to the train+val set
                    rel_val_size = self.config["data"]["validation_size"] / (1 - self.config["data"]["test_size"])
                    stratify_val = y_temp if self.config["data"]["stratify"] else None
                    
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_temp, y_temp,
                        test_size=rel_val_size,
                        random_state=self.config["data"]["random_state"],
                        stratify=stratify_val
                    )
                else:
                    X_train, y_train = X_temp, y_temp
            else:
                X_train, y_train = X, y
            
            # Create preprocessor
            self.preprocessor = self._build_preprocessor(X_train)
            
            # Apply preprocessing to all datasets
            if self.preprocessor:
                X_train = self._apply_preprocessor(X_train, fit=True)
                
                if X_val is not None:
                    X_val = self._apply_preprocessor(X_val)
                    
                if X_test is not None:
                    X_test = self._apply_preprocessor(X_test)
            
            # Apply handling for imbalanced classes
            if self.config["data"]["problem_type"] in ["binary_classification", "multiclass_classification"] \
               and self.config["data"]["sampling_strategy"]:
                X_train, y_train = self._handle_imbalanced_data(X_train, y_train)
            
            # Log dataset statistics
            self._log_dataset_statistics(X_train, y_train, X_val, y_val)
            
            # Store dataset info
            self.dataset_statistics.update({
                "train_samples": len(X_train),
                "val_samples": len(X_val) if X_val is not None else 0,
                "test_samples": len(X_test) if X_test is not None else 0,
                "features": list(X_train.columns) if hasattr(X_train, 'columns') else None,
                "preprocessing_time": time.time() - start_time
            })
            
            logger.info(f"Data preparation completed in {time.time() - start_time:.2f} seconds")
            logger.info(f"Train: {X_train.shape}, Validation: {X_val.shape if X_val is not None else 'None'}, "
                       f"Test: {X_test.shape if X_test is not None else 'None'}")
            
            return X_train, y_train, X_val, y_val, X_test, y_test
            
        except Exception as e:
            logger.error(f"Error in data loading and preprocessing: {str(e)}", exc_info=True)
            raise
            
    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Build preprocessing pipeline based on configuration.
        
        Args:
            X: Feature DataFrame to build preprocessor for
            
        Returns:
            Sklearn ColumnTransformer for preprocessing
        """
        try:
            # Get column information
            categorical_cols = self.config["data"]["categorical_columns"]
            numerical_cols = self.config["data"]["numerical_columns"]
            
            # Auto-detect column types if not explicitly specified
            if not categorical_cols and not numerical_cols:
                categorical_cols = [col for col in X.columns if X[col].dtype == 'object' or X[col].nunique() < 10]
                numerical_cols = [col for col in X.columns if col not in categorical_cols]
                
                logger.info(f"Auto-detected {len(categorical_cols)} categorical and {len(numerical_cols)} numerical features")
            
            # Create preprocessing steps
            preprocessor_steps = []
            
            # Numerical features preprocessing
            if numerical_cols:
                num_transformer_steps = []
                
                # Add scaling if specified
                scaling = self.config["preprocessing"]["scaling"]
                if scaling == "standard":
                    num_transformer_steps.append(("scaler", StandardScaler()))
                elif scaling == "minmax":
                    num_transformer_steps.append(("scaler", MinMaxScaler()))
                elif scaling == "robust":
                    num_transformer_steps.append(("scaler", RobustScaler()))
                elif scaling == "power":
                    num_transformer_steps.append(("scaler", PowerTransformer()))
                    
                # Create numerical pipeline
                if num_transformer_steps:
                    num_transformer = Pipeline(num_transformer_steps)
                    preprocessor_steps.append(("numerical", num_transformer, numerical_cols))
                
            # Categorical features preprocessing
            if categorical_cols:
                cat_transformer_steps = []
                
                # Add encoding if specified
                encoding = self.config["preprocessing"]["encoding"]
                if encoding == "onehot":
                    cat_transformer_steps.append(
                        ("encoder", OneHotEncoder(sparse=False, handle_unknown="ignore"))
                    )
                elif encoding == "ordinal":
                    cat_transformer_steps.append(
                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
                    )
                    
                # Create categorical pipeline
                if cat_transformer_steps:
                    cat_transformer = Pipeline(cat_transformer_steps)
                    preprocessor_steps.append(("categorical", cat_transformer, categorical_cols))
                    
            # Return complete preprocessor
            if preprocessor_steps:
                return ColumnTransformer(preprocessor_steps, remainder="passthrough")
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error building preprocessor: {str(e)}", exc_info=True)
            return None
            
    def _apply_preprocessor(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Apply preprocessing to features.
        
        Args:
            X: Features to preprocess
            fit: Whether to fit the preprocessor
            
        Returns:
            Preprocessed features
        """
        try:
            if self.preprocessor is None:
                return X
                
            if fit:
                return self.preprocessor.fit_transform(X)
            else:
                return self.preprocessor.transform(X)
                
        except Exception as e:
            logger.error(f"Error applying preprocessor: {str(e)}", exc_info=True)
            # Fall back to original data if preprocessing fails
            logger.warning("Falling back to original unpreprocessed data")
            return X
    
    def _handle_imbalanced_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle imbalanced datasets with resampling techniques.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Resampled features and target
        """
        sampling_strategy = self.config["data"]["sampling_strategy"]
        if not sampling_strategy:
            return X, y
            
        try:
            if sampling_strategy == "over":
                from imblearn.over_sampling import RandomOverSampler
                sampler = RandomOverSampler(random_state=self.config["data"]["random_state"])
                X_resampled, y_resampled = sampler.fit_resample(X, y)
                logger.info(f"Applied oversampling: {len(y)} -> {len(y_resampled)} samples")
                return X_resampled, y_resampled
                
            elif sampling_strategy == "under":
                from imblearn.under_sampling import RandomUnderSampler
                sampler = RandomUnderSampler(random_state=self.config["data"]["random_state"])
                X_resampled, y_resampled = sampler.fit_resample(X, y)
                logger.info(f"Applied undersampling: {len(y)} -> {len(y_resampled)} samples")
                return X_resampled, y_resampled
                
            elif sampling_strategy == "smote":
                from imblearn.over_sampling import SMOTE
                sampler = SMOTE(random_state=self.config["data"]["random_state"])
                X_resampled, y_resampled = sampler.fit_resample(X, y)
                logger.info(f"Applied SMOTE: {len(y)} -> {len(y_resampled)} samples")
                return X_resampled, y_resampled
                
            else:
                logger.warning(f"Unknown sampling strategy: {sampling_strategy}")
                return X, y
                
        except ImportError:
            logger.warning("imbalanced-learn package not found. Install with 'pip install imbalanced-learn'")
            return X, y
        except Exception as e:
            logger.error(f"Error applying resampling: {str(e)}", exc_info=True)
            return X, y
            
    def create_model(self, model_type: str = None, params: Dict = None) -> Any:
        """
        Create a model instance based on configuration.
        
        Args:
            model_type: Type of model to create (overrides config)
            params: Model parameters (overrides config)
            
        Returns:
            Model instance
        """
        if model_type is None:
            model_type = self.config["model"]["type"]
            
        if params is None:
            params = self.config["model"]["params"]
            
        # Set problem-specific defaults
        is_classification = self.config["data"]["problem_type"] in [
            "binary_classification", "multiclass_classification"
        ]
        
        # Common params for sklearn models
        common_params = {
            "random_state": self.config["training"]["random_state"]
        }
        
        if is_classification and self.config["training"]["class_weight"]:
            common_params["class_weight"] = self.config["training"]["class_weight"]
            
        # Add any params from config
        for param, value in params.items():
            common_params[param] = value
            
        try:
            # Create appropriate model based on type
            if model_type == "random_forest":
                if is_classification:
                    return RandomForestClassifier(**common_params)
                else:
                    return RandomForestRegressor(**common_params)
                    
            elif model_type == "gradient_boosting":
                if is_classification:
                    return GradientBoostingClassifier(**common_params)
                else:
                    return GradientBoostingRegressor(**common_params)
                    
            elif model_type == "xgboost":
                if is_classification:
                    return XGBClassifier(**common_params)
                else:
                    return XGBRegressor(**common_params)
                    
            elif model_type == "lightgbm":
                try:
                    from lightgbm import LGBMClassifier, LGBMRegressor
                    if is_classification:
                        return LGBMClassifier(**common_params)
                    else:
                        return LGBMRegressor(**common_params)
                except ImportError:
                    logger.error("LightGBM not installed. Install with 'pip install lightgbm'")
                    raise
                    
            elif model_type == "catboost":
                try:
                    from catboost import CatBoostClassifier, CatBoostRegressor
                    if is_classification:
                        return CatBoostClassifier(**common_params)
                    else:
                        return CatBoostRegressor(**common_params)
                except ImportError:
                    logger.error("CatBoost not installed. Install with 'pip install catboost'")
                    raise
                    
            elif model_type == "linear":
                if is_classification:
                    return LogisticRegression(**common_params)
                else:
                    return Ridge(**common_params)
                    
            elif model_type == "neural_network":
                if is_classification:
                    return MLPClassifier(**common_params)
                else:
                    return MLPRegressor(**common_params)
                    
            elif model_type == "custom":
                # Load custom model from path
                custom_path = self.config["model"]["custom_model_path"]
                if not custom_path:
                    raise ValueError("custom_model_path must be specified for custom models")
                    
                # Import and create custom model
                import importlib.util
                spec = importlib.util.spec_from_file_location("custom_model", custom_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Call create_model function from custom module
                return module.create_model(**common_params)
                
            elif model_type == "deep_learning":
                # Create a TensorFlow/Keras or PyTorch model based on available libraries
                if "tensorflow" in sys.modules:
                    return self._create_tf_model(common_params)
                elif "torch" in sys.modules:
                    return self._create_torch_model(common_params)
                else:
                    raise ImportError("Neither TensorFlow nor PyTorch are available for deep learning")
            
            elif model_type == "transformer":
                try:
                    from transformers import AutoModelForSequenceClassification
                    model_name = params.get("model_name", "distilbert-base-uncased")
                    num_labels = params.get("num_labels", 2)
                    return AutoModelForSequenceClassification.from_pretrained(
                        model_name, num_labels=num_labels
                    )
                except ImportError:
                    logger.error("Transformers library not installed. Install with 'pip install transformers'")
                    raise
                    
            elif model_type == "ensemble":
                return self._create_ensemble_model(common_params)
                
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            logger.error(f"Error creating model of type {model_type}: {str(e)}", exc_info=True)
            raise
    
    def _create_ensemble_model(self, common_params: Dict) -> Any:
        """
        Create an ensemble model based on configuration.
        
        Args:
            common_params: Common parameters for all models
            
        Returns:
            Ensemble model
        """
        is_classification = self.config["data"]["problem_type"] in [
            "binary_classification", "multiclass_classification"
        ]
        
        ensemble_config = self.config["model"]["ensemble"]
        if not ensemble_config["enabled"]:
            raise ValueError("Ensemble is not enabled in configuration")
            
        # Create base models
        estimators = []
        for i, model_type in enumerate(ensemble_config["models"]):
            model_name = f"{model_type}_{i}"
            model = self.create_model(model_type, common_params)
            estimators.append((model_name, model))
            
        # Create ensemble model
        if ensemble_config.get("stacking", False):
            # Stacking ensemble
            meta_model = self.create_model(ensemble_config.get("meta_model", "linear"), common_params)
            
            if is_classification:
                return StackingClassifier(
                    estimators=estimators,
                    final_estimator=meta_model,
                    cv=self.config["training"]["cv_folds"],
                    n_jobs=-1
                )
            else:
                return StackingRegressor(
                    estimators=estimators,
                    final_estimator=meta_model,
                    cv=self.config["training"]["cv_folds"],
                    n_jobs=-1
                )
        else:
            # Voting ensemble
            voting = ensemble_config.get("voting", "soft" if is_classification else "hard")
            
            if is_classification:
                return VotingClassifier(
                    estimators=estimators,
                    voting=voting,
                    n_jobs=-1
                )
            else:
                return VotingRegressor(
                    estimators=estimators,
                    n_jobs=-1
                )
    
    def optimize_hyperparameters(self, X_train, y_train, X_val=None, y_val=None):
        """
        Perform hyperparameter optimization.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Best model with optimized hyperparameters
        """
        if not self.config["hyperparameter_optimization"]["enabled"]:
            logger.info("Hyperparameter optimization disabled, skipping")
            return self.create_model()
            
        start_time = time.time()
        logger.info("Starting hyperparameter optimization...")
        
        # Get optimization parameters
        method = self.config["hyperparameter_optimization"]["method"]
        n_trials = self.config["hyperparameter_optimization"]["n_trials"]
        timeout = self.config["hyperparameter_optimization"]["timeout"]
        direction = self.config["hyperparameter_optimization"]["direction"]
        scoring = self.config["hyperparameter_optimization"]["scoring"]
        n_jobs = self.config["hyperparameter_optimization"]["n_jobs"]
        
        try:
            if method == "optuna":
                # Define search space based on model type
                model_type = self.config["model"]["type"]
                search_space = self._get_optuna_search_space(model_type)
                
                # Create a study
                study_name = self.config["hyperparameter_optimization"]["study_name"] or f"study_{int(time.time())}"
                
                import optuna
                pruner = None
                if self.config["hyperparameter_optimization"]["pruner"] == "hyperband":
                    pruner = optuna.pruners.HyperbandPruner()
                elif self.config["hyperparameter_optimization"]["pruner"] == "median":
                    pruner = optuna.pruners.MedianPruner()
                
                study = optuna.create_study(
                    direction=direction,
                    study_name=study_name,
                    pruner=pruner
                )
                
                # Define objective function
                def objective(trial):
                    # Get hyperparameters for this trial
                    params = {}
                    for param_name, param_config in search_space.items():
                        param_type = param_config["type"]
                        if param_type == "categorical":
                            params[param_name] = trial.suggest_categorical(param_name, param_config["values"])
                        elif param_type == "int":
                            params[param_name] = trial.suggest_int(
                                param_name, param_config["low"], param_config["high"], 
                                step=param_config.get("step", 1)
                            )
                        elif param_type == "float":
                            params[param_name] = trial.suggest_float(
                                param_name, param_config["low"], param_config["high"],
                                log=param_config.get("log", False)
                            )
                    
                    # Create and train model with these params
                    model = self.create_model(params=params)
                    
                    # Use cross-validation if configured
                    cv = self.config["hyperparameter_optimization"]["cv"]
                    if cv > 1:
                        scores = cross_val_score(
                            model, X_train, y_train, 
                            cv=cv, scoring=scoring, n_jobs=1
                        )
                        score = scores.mean()
                    else:
                        # Train on training set, evaluate on validation set
                        model.fit(X_train, y_train)
                        if X_val is not None and y_val is not None:
                            y_pred = model.predict(X_val)
                            score = self._calculate_metric(y_val, y_pred, scoring)
                        else:
                            y_pred = model.predict(X_train)
                            score = self._calculate_metric(y_train, y_pred, scoring)
                    
                    # For minimization metrics, negate the score
                    if "neg_" in scoring or scoring in ["mean_squared_error", "mean_absolute_error"]:
                        score = -score
                        
                    return score
                
                # Run optimization
                study.optimize(
                    objective, 
                    n_trials=n_trials,
                    timeout=timeout,
                    n_jobs=n_jobs,
                    show_progress_bar=True
                )
                
                # Get best parameters and create final model
                best_params = study.best_params
                logger.info(f"Best parameters found: {best_params}")
                logger.info(f"Best score achieved: {study.best_value}")
                
                # Save optimization results
                optimization_results = {
                    "best_params": best_params,
                    "best_score": study.best_value,
                    "n_trials": len(study.trials),
                    "duration_seconds": time.time() - start_time
                }
                
                self.best_model_params = best_params
                
                # Create and return best model
                best_model = self.create_model(params=best_params)
                
                # Log to MLflow if enabled
                if self.config["mlflow"]["enabled"] and mlflow.active_run():
                    mlflow.log_params({"hp_" + k: v for k, v in best_params.items()})
                    mlflow.log_metrics({
                        "hp_optimization_time": time.time() - start_time,
                        "hp_best_score": study.best_value
                    })
                
                return best_model
                
            elif method == "grid":
                # Get param grid from config
                param_grid = self.config["hyperparameter_optimization"]["param_grid"]
                if not param_grid:
                    logger.warning("No parameter grid provided for grid search")
                    return self.create_model()
                
                # Create base model
                model = self.create_model()
                
                # Run grid search
                from sklearn.model_selection import GridSearchCV
                search = GridSearchCV(
                    model, param_grid,
                    scoring=scoring,
                    cv=self.config["hyperparameter_optimization"]["cv"],
                    n_jobs=n_jobs,
                    verbose=1,
                    refit=True
                )
                
                search.fit(X_train, y_train)
                
                logger.info(f"Best parameters found: {search.best_params_}")
                logger.info(f"Best score achieved: {search.best_score_}")
                
                self.best_model_params = search.best_params_
                return search.best_estimator_
                
            elif method == "ray":
                try:
                    from ray.tune.sklearn import TuneSearchCV
                    from ray.tune.schedulers import ASHAScheduler
                    from sklearn.model_selection import KFold
                    
                    # Create search space for Ray
                    model_type = self.config["model"]["type"]
                    search_space = self._get_ray_search_space(model_type)
                    
                    # Create base model
                    model = self.create_model()
                    
                    # Create scheduler for early stopping
                    scheduler = ASHAScheduler(
                        max_t=self.config["hyperparameter_optimization"]["n_trials"],
                        grace_period=1,
                        reduction_factor=2
                    )
                    
                    # Create search
                    search = TuneSearchCV(
                        model,
                        search_space,
                        scheduler=scheduler,
                        n_trials=n_trials,
                        scoring=scoring,
                        cv=KFold(self.config["hyperparameter_optimization"]["cv"]),
                        n_jobs=n_jobs,
                        refit=True,
                        verbose=1
                    )
                    
                    search.fit(X_train, y_train)
                    
                    logger.info(f"Best parameters found: {search.best_params_}")
                    logger.info(f"Best score achieved: {search.best_score_}")
                    
                    self.best_model_params = search.best_params_
                    return search.best_estimator_
                    
                except ImportError:
                    logger.error("Ray is not installed. Please install ray[tune]")
                    return self.create_model()
                    
            else:
                logger.warning(f"Unsupported optimization method: {method}")
                return self.create_model()
                
        except Exception as e:
            logger.error(f"Error during hyperparameter optimization: {str(e)}", exc_info=True)
            logger.warning("Falling back to default model")
            return self.create_model()
        finally:
            logger.info(f"Hyperparameter optimization completed in {time.time() - start_time:.2f} seconds")
    
    def train(self):
        """
        Execute the full training pipeline.
        
        Returns:
            Trained model
        """
        start_time = time.time()
        logger.info(f"Starting model training pipeline for {self.config['data']['problem_type']}")
        
        try:
            # Start MLflow run if enabled
            if self.config["mlflow"]["enabled"]:
                tags = self.config["mlflow"]["tags"] or {}
                tags.update({
                    "problem_type": self.config["data"]["problem_type"],
                    "model_type": self.config["model"]["type"],
                    "data_path": self.config["data"]["train_data_path"]
                })
                
                with mlflow.start_run(run_name=f"train_{int(time.time())}", tags=tags) as run:
                    self.run_id = run.info.run_id
                    logger.info(f"MLflow run started with ID: {self.run_id}")
                    
                    # Load and prepare data
                    X_train, y_train, X_val, y_val, X_test, y_test = self.load_and_preprocess_data()
                    
                    # Log parameters
                    self._log_parameters()
                    
                    # Optimize hyperparameters
                    if self.config["hyperparameter_optimization"]["enabled"]:
                        model = self.optimize_hyperparameters(X_train, y_train, X_val, y_val)
                    else:
                        model = self.create_model()
                    
                    # Train model
                    self._train_model(model, X_train, y_train, X_val, y_val)
                    
                    # Evaluate model
                    self.evaluate(X_test, y_test, is_test_set=True)
                    
                    # Save model
                    self.save_model()
                    
                    # Log
                    
    def _get_optuna_search_space(self, model_type: str) -> Dict:
        """
        Define hyperparameter search space for Optuna optimization.
        
        Args:
            model_type: Type of model to optimize
            
        Returns:
            Search space definition
        """
        # Custom search space from config if provided
        if self.config["hyperparameter_optimization"]["search_space"]:
            return self.config["hyperparameter_optimization"]["search_space"]
            
        # Default search spaces by model type
        search_space = {}
        
        if model_type == "random_forest":
            search_space = {
                "n_estimators": {"type": "int", "low": 50, "high": 500},
                "max_depth": {"type": "int", "low": 3, "high": 30},
                "min_samples_split": {"type": "int", "low": 2, "high": 20},
                "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
                "max_features": {"type": "categorical", "values": ["sqrt", "log2", None]}
            }
        elif model_type == "gradient_boosting":
            search_space = {
                "n_estimators": {"type": "int", "low": 50, "high": 500},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
                "max_depth": {"type": "int", "low": 3, "high": 10},
                "subsample": {"type": "float", "low": 0.5, "high": 1.0},
                "min_samples_split": {"type": "int", "low": 2, "high": 20}
            }
        elif model_type == "xgboost":
            search_space = {
                "n_estimators": {"type": "int", "low": 50, "high": 500},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
                "max_depth": {"type": "int", "low": 3, "high": 10},
                "subsample": {"type": "float", "low": 0.5, "high": 1.0},
                "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
                "gamma": {"type": "float", "low": 0, "high": 5}
            }
        elif model_type == "lightgbm":
            search_space = {
                "n_estimators": {"type": "int", "low": 50, "high": 500},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
                "max_depth": {"type": "int", "low": 3, "high": 10},
                "num_leaves": {"type": "int", "low": 20, "high": 100},
                "feature_fraction": {"type": "float", "low": 0.5, "high": 1.0},
                "bagging_fraction": {"type": "float", "low": 0.5, "high": 1.0},
                "min_child_samples": {"type": "int", "low": 5, "high": 100}
            }
        elif model_type == "linear":
            is_classification = self.config["data"]["problem_type"] in ["binary_classification", "multiclass_classification"]
            if is_classification:
                search_space = {
                    "C": {"type": "float", "low": 0.001, "high": 10.0, "log": True},
                    "penalty": {"type": "categorical", "values": ["l1", "l2", "elasticnet"]},
                    "solver": {"type": "categorical", "values": ["liblinear", "saga"]}
                }
            else:
                search_space = {
                    "alpha": {"type": "float", "low": 0.001, "high": 10.0, "log": True},
                    "fit_intercept": {"type": "categorical", "values": [True, False]}
                }
        elif model_type == "neural_network":
            search_space = {
                "hidden_layer_sizes": {"type": "categorical", 
                    "values": [(50,), (100,), (50, 50), (100, 50), (100, 100)]},
                "activation": {"type": "categorical", "values": ["tanh", "relu"]},
                "alpha": {"type": "float", "low": 0.0001, "high": 0.01, "log": True},
                "learning_rate_init": {"type": "float", "low": 0.001, "high": 0.1, "log": True}
            }
        elif model_type == "ensemble":
            # For ensemble, search over the meta-learner parameters
            search_space = {
                "final_estimator__C": {"type": "float", "low": 0.001, "high": 10.0, "log": True}
            }
        
        return search_space
    
    def _get_ray_search_space(self, model_type: str) -> Dict:
        """
        Define hyperparameter search space for Ray Tune.
        
        Args:
            model_type: Type of model to optimize
            
        Returns:
            Ray Tune search space
        """
        from ray.tune.search.sample import randint, uniform, loguniform, choice
        
        # Custom search space from config if provided
        if self.config["hyperparameter_optimization"]["search_space"]:
            return self.config["hyperparameter_optimization"]["search_space"]
            
        # Convert Optuna space to Ray Tune space
        optuna_space = self._get_optuna_search_space(model_type)
        ray_space = {}
        
        for param_name, param_config in optuna_space.items():
            if param_config["type"] == "int":
                ray_space[param_name] = randint(param_config["low"], param_config["high"])
            elif param_config["type"] == "float":
                if param_config.get("log", False):
                    ray_space[param_name] = loguniform(param_config["low"], param_config["high"])
                else:
                    ray_space[param_name] = uniform(param_config["low"], param_config["high"])
            elif param_config["type"] == "categorical":
                ray_space[param_name] = choice(param_config["values"])
        
        return ray_space
    
    def _calculate_metric(self, y_true, y_pred, metric_name: str) -> float:
        """
        Calculate evaluation metric.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            metric_name: Name of metric to calculate
            
        Returns:
            Calculated metric value
        """
        problem_type = self.config["data"]["problem_type"]
        
        # For classification metrics
        if metric_name == "accuracy":
            return accuracy_score(y_true, y_pred)
        elif metric_name == "precision":
            if problem_type == "multiclass_classification":
                return precision_score(y_true, y_pred, average='weighted')
            else:
                return precision_score(y_true, y_pred)
        elif metric_name == "recall":
            if problem_type == "multiclass_classification":
                return recall_score(y_true, y_pred, average='weighted')
            else:
                return recall_score(y_true, y_pred)
        elif metric_name == "f1":
            if problem_type == "multiclass_classification":
                return f1_score(y_true, y_pred, average='weighted')
            else:
                return f1_score(y_true, y_pred)
        elif metric_name == "roc_auc":
            if hasattr(self.best_model, 'predict_proba'):
                y_prob = self.best_model.predict_proba(X_test)
                if problem_type == "multiclass_classification":
                    return roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
                else:
                    return roc_auc_score(y_true, y_prob[:, 1])
            return 0.0
            
        # For regression metrics
        elif metric_name == "mean_squared_error" or metric_name == "mse":
            return mean_squared_error(y_true, y_pred)
        elif metric_name == "root_mean_squared_error" or metric_name == "rmse":
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric_name == "mean_absolute_error" or metric_name == "mae":
            return mean_absolute_error(y_true, y_pred)
        elif metric_name == "r2":
            return r2_score(y_true, y_pred)
        elif metric_name == "neg_mean_squared_error" or metric_name == "neg_mse":
            return -mean_squared_error(y_true, y_pred)
        elif metric_name == "neg_root_mean_squared_error" or metric_name == "neg_rmse":
            return -np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric_name == "neg_mean_absolute_error" or metric_name == "neg_mae":
            return -mean_absolute_error(y_true, y_pred)
        else:
            logger.warning(f"Unknown metric: {metric_name}")
            return 0.0
            
    def _create_tf_model(self, params):
        """
        Create a TensorFlow/Keras deep learning model.
        
        Args:
            params: Model parameters
            
        Returns:
            TensorFlow model
        """
        problem_type = self.config["data"]["problem_type"]
        
        # Get number of features from input shape
        n_features = None
        try:
            # This will be populated after data loading
            n_features = self.dataset_statistics.get("n_features", 10)
        except:
            n_features = 10  # Default if unknown
            
        # Determine output layer configuration
        if problem_type == "binary_classification":
            output_units = 1
            output_activation = "sigmoid"
            loss = "binary_crossentropy"
            metrics = ["accuracy", tf.keras.metrics.AUC()]
        elif problem_type == "multiclass_classification":
            # Determine number of classes
            n_classes = self.dataset_statistics.get("n_classes", 2)
            output_units = n_classes
            output_activation = "softmax"
            loss = "categorical_crossentropy"
            metrics = ["accuracy", tf.keras.metrics.AUC()]
        else:  # Regression
            output_units = 1
            output_activation = "linear"
            loss = "mse"
            metrics = ["mae", "mse"]
            
        # Get model architecture parameters
        hidden_units = params.get("hidden_units", [64, 32])
        activation = params.get("activation", "relu")
        dropout_rate = params.get("dropout_rate", 0.2)
        learning_rate = params.get("learning_rate", 0.001)
            
        # Create model
        model = tf.keras.Sequential()
        
        # Input layer
        model.add(tf.keras.layers.Input(shape=(n_features,)))
        
        # Hidden layers
        for units in hidden_units:
            model.add(tf.keras.layers.Dense(units, activation=activation))
            if dropout_rate > 0:
                model.add(tf.keras.layers.Dropout(dropout_rate))
                
        # Output layer
        model.add(tf.keras.layers.Dense(output_units, activation=output_activation))
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        # Create a scikit-learn compatible wrapper
        from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
        
        if problem_type in ["binary_classification", "multiclass_classification"]:
            return KerasClassifier(
                model=model,
                epochs=self.config["training"]["epochs"],
                batch_size=self.config["training"]["batch_size"],
                verbose=1
            )
        else:
            return KerasRegressor(
                model=model,
                epochs=self.config["training"]["epochs"],
                batch_size=self.config["training"]["batch_size"],
                verbose=1
            )
            
    def _create_torch_model(self, params):
        """
        Create a PyTorch deep learning model.
        
        Args:
            params: Model parameters
            
        Returns:
            PyTorch model wrapped for scikit-learn compatibility
        """
        import torch
        import torch.nn as nn
        from skorch import NeuralNetClassifier, NeuralNetRegressor
        
        problem_type = self.config["data"]["problem_type"]
        
        # Get number of features
        n_features = self.dataset_statistics.get("n_features", 10)
        
        # Define network architecture
        hidden_units = params.get("hidden_units", [64, 32])
        activation = params.get("activation", "relu")
        dropout_rate = params.get("dropout_rate", 0.2)
        
        # Map string activation to PyTorch activation
        activation_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid
        }
        activation_fn = activation_map.get(activation, nn.ReLU)
        
        # Create PyTorch model class
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList()
                
                # Input layer
                self.layers.append(nn.Linear(n_features, hidden_units[0]))
                self.layers.append(activation_fn())
                if dropout_rate > 0:
                    self.layers.append(nn.Dropout(dropout_rate))
                
                # Hidden layers
                for i in range(len(hidden_units) - 1):
                    self.layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
                    self.layers.append(activation_fn())
                    if dropout_rate > 0:
                        self.layers.append(nn.Dropout(dropout_rate))
                
                # Output layer
                if problem_type in ["binary_classification", "multiclass_classification"]:
                    n_classes = self.dataset_statistics.get("n_classes", 2)
                    out_units = n_classes if n_classes > 2 else 1
                    self.layers.append(nn.Linear(hidden_units[-1], out_units))
                    if n_classes > 2:
                        self.layers.append(nn.Softmax(dim=1))
                    else:
                        self.layers.append(nn.Sigmoid())
                else:
                    self.layers.append(nn.Linear(hidden_units[-1], 1))
                    
            def forward(self, X):
                for layer in self.layers:
                    X = layer(X)
                return X
        
        # Create skorch wrapper
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        if problem_type in ["binary_classification", "multiclass_classification"]:
            n_classes = self.dataset_statistics.get("n_classes", 2)
            
            if n_classes > 2:
                criterion = nn.CrossEntropyLoss
            else:
                criterion = nn.BCEWithLogitsLoss
                
            return NeuralNetClassifier(
                Net,
                max_epochs=self.config["training"]["epochs"],
                batch_size=self.config["training"]["batch_size"],
                criterion=criterion,
                optimizer=torch.optim.Adam,
                lr=params.get("learning_rate", 0.001),
                device=device,
                verbose=1
            )
        else:
            return NeuralNetRegressor(
                Net,
                max_epochs=self.config["training"]["epochs"],
                batch_size=self.config["training"]["batch_size"],
                criterion=nn.MSELoss,
                optimizer=torch.optim.Adam,
                lr=params.get("learning_rate", 0.001),
                device=device,
                verbose=1
            )
        
    def _train_model(self, model, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model with appropriate settings and tracking.
        
        Args:
            model: Model to train
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
        """
        start_time = time.time()
        logger.info("Starting model training...")
        
        # Check if it's a PyTorch or TensorFlow model for special handling
        is_tf_model = hasattr(model, 'model') and hasattr(model.model, 'fit')
        is_torch_model = hasattr(model, 'module_') or 'skorch' in str(type(model))
        
        # Prepare monitoring callbacks for deep learning models
        callbacks = []
        
        if is_tf_model:
            # Set up TensorFlow callbacks
            if self.config["training"]["early_stopping"]:
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor=self.config["training"]["early_stopping_metric"],
                    patience=self.config["training"]["early_stopping_rounds"],
                    restore_best_weights=True
                )
                callbacks.append(early_stopping)
                
            if self.config["training"]["save_checkpoints"]:
                model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(self.config["output"]["model_dir"], "checkpoints/model_{epoch}"),
                    save_best_only=True,
                    monitor=self.config["training"]["early_stopping_metric"]
                )
                callbacks.append(model_checkpoint)
                
            # Set up MLflow logging
            if self.config["mlflow"]["enabled"] and mlflow.active_run():
                from mlflow.tensorflow import autolog
                autolog(log_models=True, disable=False)
        
        # Handle sample weights if configured
        sample_weight = None
        if self.config["training"]["sample_weight_column"] and self.config["training"]["sample_weight_column"] in X_train:
            sample_weight = X_train[self.config["training"]["sample_weight_column"]]
            X_train = X_train.drop(columns=[self.config["training"]["sample_weight_column"]])
            
        # Apply class weights if configured for classification
        if (self.config["data"]["problem_type"] in ["binary_classification", "multiclass_classification"] and 
            self.config["training"]["class_weight"]):
            if self.config["training"]["class_weight"] == "balanced":
                class_weights = class_weight.compute_class_weight(
                    'balanced', classes=np.unique(y_train), y=y_train
                )
                sample_weight = compute_sample_weight('balanced', y_train)
            elif self.config["training"]["class_weight"] == "balanced_subsample":
                sample_weight = compute_sample_weight('balanced_subsample', y_train)
            elif isinstance(self.config["training"]["class_weight"], dict):
                # User-specified class weights
                sample_weight = np.array([self.config["training"]["class_weight"].get(y, 1.0) for y in y_train])
        
        # Apply cross-validation if configured
        if self.config["training"]["cv_strategy"] and self.config["training"]["cv_folds"] > 1:
            logger.info(f"Training with {self.config['training']['cv_strategy']} "
                      f"cross-validation using {self.config['training']['cv_folds']} folds")
            
            cv_results = self._train_with_cv(model, X_train, y_train, sample_weight)
            
            # Store CV results
            self.training_history["cv_results"] = cv_results
            
            # Train final model on all data
            logger.info("Training final model on all data")
            self._fit_model(model, X_train, y_train, sample_weight, X_val, y_val)
            
        else:
            # Standard training without cross-validation
            self._fit_model(model, X_train, y_train, sample_weight, X_val, y_val)
        
        # Store the trained model
        self.best_model = model
        
        # Calculate training time
        training_time = time.time() - start_time
        self.training_history["training_time"] = training_time
        
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        
        # Extract feature importance if available
        self._extract_feature_importance()
        
        # Calculate model size
        self._calculate_model_size()
        
    def _fit_model(self, model, X_train, y_train, sample_weight=None, X_val=None, y_val=None):
        """
        Fit model with appropriate method based on model type.
        
        Args:
            model: Model to fit
            X_train: Training features
            y_train: Training targets
            sample_weight: Sample weights (optional)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        try:
            # Basic check if model requires fitting
            if not hasattr(model, 'fit'):
                logger.warning("Model does not have fit method")
                return
                
            # Handle different model types
            if hasattr(model, 'model') and hasattr(model.model, 'fit'):
                # TensorFlow/Keras model
                validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
                
                # Set validation split if no explicit validation data
                validation_split = 0.0 if validation_data else 0.1
                
                callbacks = []
                
                # Add early stopping if configured
                if self.config["training"]["early_stopping"]:
                    early_stopping = tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=self.config["training"]["early_stopping_rounds"],
                        restore_best_weights=True
                    )
                    callbacks.append(early_stopping)
                
                # Fit the model
                history = model.model.fit(
                    X_train, y_train,
                    epochs=self.config["training"]["epochs"],
                    batch_size=self.config["training"]["batch_size"],
                    validation_data=validation_data,
                    validation_split=validation_split,
                    callbacks=callbacks,
                    sample_weight=sample_weight,
                    verbose=1
                )
                
                # Store training history
                if hasattr(history, 'history'):
                    self.training_history["epochs"] = len(history.history.get('loss', []))
                    self.training_history["loss"] = history.history.get('loss', [])
                    self.training_history["val_loss"] = history.history.get('val_loss', [])
                    
            elif hasattr(model, 'fit') and 'xgb' in str(type(model)).lower():
                # XGBoost model
                fit_params = {
                    "verbose": self.config["advanced"]["verbosity"] > 0
                }
                
                # Add early stopping if configured
                if self.config["training"]["early_stopping"] and X_val is not None and y_val is not None:
                    fit_params["eval_set"] = [(X_val, y_val)]
                    fit_params["early_stopping_rounds"] = self.config["training"]["early_stopping_rounds"]
                
                # Add sample weights if provided
                if sample_weight is not None:
                    fit_params["sample_weight"] = sample_weight
                    if X_val is not None and y_val is not None:
                        fit_params["eval_sample_weight"] = [sample_weight[:len(X_val)]]
                
                # Fit the model
                model.fit(X_train, y_train, **fit_params)
                
                # Store training history if available
                if hasattr(model, 'evals_result_'):
                    self.training_history["xgb_results"] = model.evals_result_
                    
            elif hasattr(model, 'fit') and 'lightgbm' in str(type(model)).lower():
                # LightGBM model
                fit_params = {
                    "verbose": self.config["advanced"]["verbosity"] > 0
                }
                
                # Add early stopping if configured
                if self.config["training"]["early_stopping"] and X_val is not None and y_val is not None:
                    fit_params["eval_set"] = [(X_val, y_val)]
                    fit_params["early_stopping_rounds"] = self.config["training"]["early_stopping_rounds"]
                
                # Add sample weights if provided
                if sample_weight is not None:
                    fit_params["sample_weight"] = sample_weight
                
                # Fit the model
                model.fit(X_train, y_train, **fit_params)
                
            else:
                # Standard scikit-learn API
                fit_params = {}
                
                if sample_weight is not None:
                    fit_params["sample_weight"] = sample_weight
                    
                # Simple fit for sklearn models
                model.fit(X_train, y_train, **fit_params)
                
        except Exception as e:
            logger.error(f"Error during model fitting: {str(e)}", exc_info=True)
            raise RuntimeError(f"Model training failed: {str(e)}")
            
    def _train_with_cv(self, model, X_train, y_train, sample_weight=None):
        """
        Train model using cross-validation.
        
        Args:
            model: Model to train
            X_train: Training features
            y_train: Training targets
            sample_weight: Sample weights (optional)
            
        Returns:
            Cross-validation results
        """
        # Choose CV strategy based on configuration
        cv_strategy = self.config["training"]["cv_strategy"]
        n_folds = self.config["training"]["cv_folds"]
        
        if cv_strategy == "kfold":
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=self.config["training"]["random_state"])
        elif cv_strategy == "stratified_kfold":
            cv = RepeatedStratifiedKFold(n_splits=n_folds, random_state=self.config["training"]["random_state"])
        elif cv_strategy == "group_kfold":
            if self.config["data"]["group_column"] is None:
                logger.warning("Group column not specified for GroupKFold, falling back to KFold")
                cv = KFold(n_splits=n_folds, shuffle=True, random_state=self.config["training"]["random_state"])
            else:
                groups = self.config["data"]["group_column"]
                cv = GroupKFold(n_splits=n_folds)
        elif cv_strategy == "time_series_split":
            cv = TimeSeriesSplit(n_splits=n_folds)
        elif cv_strategy == "leave_one_group_out":
            cv = LeaveOneGroupOut()
        else:
            logger.warning(f"Unknown CV strategy: {cv_strategy}, falling back to KFold")
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=self.config["training"]["random_state"])
        
        # Prepare parameters for cross_val_score
        cv_params = {
            "estimator": model,
            "X": X_train,
            "y": y_train,
            "cv": cv,
            "scoring": self.config["training"]["scoring"],
            "n_jobs": -1 if self.config["hyperparameter_optimization"]["n_jobs"] == -1 else 1
        }
        
        if sample_weight is not None:
            cv_params["fit_params"] = {"sample_weight": sample_weight}
        
        # Run cross-validation
        try:
            cv_scores = cross_val_score(**cv_params)
            logger.info(f"Cross-validation scores: {cv_scores}")
            logger.info(f"Mean CV score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # Log to MLflow if enabled
            if self.config["mlflow"]["enabled"] and mlflow.active_run():
                mlflow.log_metric("cv_score_mean", cv_scores.mean())
                mlflow.log_metric("cv_score_std", cv_scores.std())
                for i, score in enumerate(cv_scores):
                    mlflow.log_metric(f"cv_score_fold_{i+1}", score)
            
            return {
                "scores": cv_scores.tolist(),
                "mean": float(cv_scores.mean()),
                "std": float(cv_scores.std()),
                "folds": n_folds
            }
            
        except Exception as e:
            logger.error(f"Error during cross-validation: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def _extract_feature_importance(self):
        """Extract and store feature importance information from the model."""
        if not self.best_model or not self.config["evaluation"]["feature_importance"]:
            return
            
        try:
            feature_names = self.dataset_statistics.get("features", 
                                                     [f"feature_{i}" for i in range(100)])
            
            # Get base model from pipeline if needed
            model = self.best_model
            if hasattr(model, 'steps') and len(model.steps) > 0:
                model = model.steps[-1][1]  # Get the last step (actual model)
                
            # Extract feature importance based on model type
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
                self.feature_importances = dict(zip(feature_names[:len(importances)], importances))
                
                # Log top features
                top_features = sorted(
                    self.feature_importances.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:20]
                
                logger.info(f"Top features by importance: {top_features}")
                
                # Log to MLflow if enabled
                if self.config["mlflow"]["enabled"] and mlflow.active_run():
                    for feat, imp in self.feature_importances.items():
                        mlflow.log_metric(f"feature_importance_{feat}", imp)
                    
                    # Log feature importance plot
                    try:
                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        
                        # Create plot
                        plt.figure(figsize=(12, 8))
                        sorted_features = sorted(self.feature_importances.items(), 
                                               key=lambda x: x[1], reverse=True)
                        features, importances = zip(*sorted_features)
                        
                        sns.barplot(x=list(importances)[:20], y=list(features)[:20])
                        plt.title('Feature Importance')
                        plt.tight_layout()
                        
                        # Save plot
                        plot_path = os.path.join(self.config["output"]["metrics_dir"], 
                                               "feature_importance.png")
                        plt.savefig(plot_path)
                        plt.close()
                        
                        # Log to MLflow
                        mlflow.log_artifact(plot_path, "feature_importance")
                    except Exception as e:
                        logger.warning(f"Failed to create feature importance plot: {str(e)}")
                        
            elif hasattr(model, 'coef_'):
                # Linear models
                coef = model.coef_
                if coef.ndim > 1:
                    # Handle multi-class case - take mean or first class for simplicity
                    coef = np.mean(coef, axis=0) if coef.shape[0] > 1 else coef[0]
                
                self.feature_importances = dict(zip(feature_names[:len(coef)], 
                                               np.abs(coef).tolist()))
                
                # Log to MLflow if enabled
                if self.config["mlflow"]["enabled"] and mlflow.active_run():
                    for feat, imp in self.feature_importances.items():
                        mlflow.log_metric(f"feature_coefficient_{feat}", imp)
            
            # Try SHAP for feature importance if configured
            elif self.config["evaluation"]["explanation_method"] == "shap":
                self._calculate_shap_feature_importance()
                
        except Exception as e:
            logger.error(f"Error extracting feature importance: {str(e)}", exc_info=True)

    def _calculate_shap_feature_importance(self):
        """Calculate feature importance using SHAP values."""
        try:
            import shap
            
            # Create a small sample for SHAP calculations
            X_sample = self.X_train.sample(min(100, len(self.X_train))).values
            
            # Choose explainer based on model type
            if 'xgboost' in str(type(self.best_model)).lower():
                explainer = shap.TreeExplainer(self.best_model)
            elif hasattr(self.best_model, 'predict_proba'):
                # For any model with predict_proba
                explainer = shap.KernelExplainer(
                    self.best_model.predict_proba, shap.kmeans(X_sample, 10)
                )
            else:
                explainer = shap.KernelExplainer(
                    self.best_model.predict, shap.kmeans(X_sample, 10)
                )
                
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # Convert to feature importance
            if isinstance(shap_values, list):
                # For multi-class, take the mean across classes
                shap_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                shap_importance = np.abs(shap_values).mean(axis=0)
                
            # Store feature importance
            feature_names = self.dataset_statistics.get("features", 
                                                     [f"feature_{i}" for i in range(len(shap_importance))])
            self.feature_importances = dict(zip(feature_names[:len(shap_importance)], shap_importance))
            
            # Log plot if MLflow enabled
            if self.config["mlflow"]["enabled"] and mlflow.active_run():
                # Create summary plot
                plt.figure()
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                                show=False, plot_size=(12, 8))
                plot_path = os.path.join(self.config["output"]["metrics_dir"], "shap_summary.png")
                plt.savefig(plot_path)
                plt.close()
                
                # Log to MLflow
                mlflow.log_artifact(plot_path, "shap_explanation")
                
        except ImportError:
            logger.warning("SHAP not installed. Install with 'pip install shap'")
        except Exception as e:
            logger.error(f"Error calculating SHAP feature importance: {str(e)}")

    def _calculate_model_size(self):
        """Calculate and store model size in bytes."""
        try:
            # Create a temporary file to measure model size
            with tempfile.NamedTemporaryFile() as tmp:
                # Save model to file
                if self.config["output"]["save_format"] == "joblib":
                    joblib.dump(self.best_model, tmp.name)
                else:
                    pickle.dump(self.best_model, tmp.name)
                    
                # Get file size
                tmp.flush()
                self.model_size_bytes = os.path.getsize(tmp.name)
                
            # Log model size
            logger.info(f"Model size: {self.model_size_bytes / (1024 * 1024):.2f} MB")
            
            # Log to MLflow if enabled
            if self.config["mlflow"]["enabled"] and mlflow.active_run():
                mlflow.log_metric("model_size_bytes", self.model_size_bytes)
                mlflow.log_metric("model_size_mb", self.model_size_bytes / (1024 * 1024))
                
        except Exception as e:
            logger.error(f"Error calculating model size: {str(e)}")

    def evaluate(self, X_test, y_test, is_test_set=True):
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            is_test_set: Whether this is the final test set or a validation set
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.best_model:
            logger.error("No model available for evaluation")
            return {}
            
        start_time = time.time()
        logger.info(f"Starting model evaluation on {'test' if is_test_set else 'validation'} set")
        
        try:
            # Basic metrics
            metrics = {}
            metrics["evaluation_time"] = time.time() - start_time
            
            # Get predictions
            y_pred = self.best_model.predict(X_test)
            
            # Compute probabilities if available (for classification)
            y_prob = None
            if self.config["data"]["problem_type"] in ["binary_classification", "multiclass_classification"]:
                if hasattr(self.best_model, "predict_proba"):
                    try:
                        y_prob = self.best_model.predict_proba(X_test)
                    except Exception as e:
                        logger.warning(f"Error getting prediction probabilities: {str(e)}")
            
            # Calculate metrics based on problem type
            if self.config["data"]["problem_type"] in ["binary_classification", "multiclass_classification"]:
                metrics = self._evaluate_classification(y_test, y_pred, y_prob)
            else:
                metrics = self._evaluate_regression(y_test, y_pred)
                
            # Add prediction timing
            metrics["prediction_time_ms"] = self._measure_prediction_time(X_test) * 1000
            
            # Generate explanations if configured
            if self.config["evaluation"]["explanations"]:
                explanations = self._generate_explanations(X_test)
                if explanations:
                    metrics["explanations"] = explanations
                    
            # Log metrics
            metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items() 
                                   if isinstance(v, (int, float))])
            logger.info(f"Evaluation metrics: {metrics_str}")
            
            # Log to MLflow
            if self.config["mlflow"]["enabled"] and mlflow.active_run():
                prefix = "test_" if is_test_set else "val_"
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"{prefix}{metric_name}", value)
                
                # Log ROC curve for classification
                if "roc_auc" in metrics and y_prob is not None:
                    self._log_roc_curve(y_test, y_prob, is_test_set)
                    
                # Log PR curve for classification
                if "average_precision" in metrics and y_prob is not None:
                    self._log_pr_curve(y_test, y_prob, is_test_set)
                    
                # Log confusion matrix for classification
                if self.config["data"]["problem_type"] in ["binary_classification", "multiclass_classification"]:
                    self._log_confusion_matrix(y_test, y_pred, is_test_set)
                    
                # Log residuals plot for regression
                if self.config["data"]["problem_type"] == "regression":
                    self._log_residuals_plot(y_test, y_pred, is_test_set)
                    
            # Save metrics to file
            metrics_file = os.path.join(
                self.config["output"]["metrics_dir"],
                f"{'test' if is_test_set else 'validation'}_metrics.json"
            )
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
                
            return metrics
                
        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}", exc_info=True)
            return {"error": str(e)}

    def _evaluate_classification(self, y_true, y_pred, y_prob=None):
        """
        Calculate classification-specific metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, log_loss, 
            confusion_matrix, classification_report
        )
        
        metrics = {}
        
        # Basic metrics
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        
        # Handle multi-class vs binary
        is_multiclass = len(np.unique(y_true)) > 2
        avg_method = 'macro' if is_multiclass else 'binary'
        
        metrics["precision"] = float(precision_score(y_true, y_pred, average=avg_method, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, average=avg_method, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, average=avg_method, zero_division=0))
        
        # Calculate ROC AUC if probabilities available
        if y_prob is not None:
            try:
                if is_multiclass:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'))
                else:
                    # For binary classification
                    if y_prob.shape[1] == 2:  # Two probabilities per sample
                        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
                    else:  # One probability per sample
                        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
                    
                    # PR AUC (average precision) for binary classification
                    if y_prob.shape[1] == 2:
                        metrics["average_precision"] = float(average_precision_score(y_true, y_prob[:, 1]))
                    else:
                        metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
            except Exception as e:
                logger.warning(f"Error calculating ROC AUC: {str(e)}")
                
            # Log loss
            try:
                metrics["log_loss"] = float(log_loss(y_true, y_prob))
            except Exception as e:
                logger.warning(f"Error calculating log loss: {str(e)}")
        
        # Add confusion matrix
        cm = confusion_matrix(y_true, y_pred).tolist()
        metrics["confusion_matrix"] = cm
        
        # Calculate class distribution and imbalance metrics
        class_counts = np.bincount(y_true.astype(int))
        metrics["class_distribution"] = class_counts.tolist()
        
        # Calculate imbalance ratio if more than one class
        if len(class_counts) > 1:
            imbalance_ratio = float(np.max(class_counts) / np.min(class_counts))
            metrics["class_imbalance_ratio"] = imbalance_ratio
            
        # Add classification report summary
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics["classification_report"] = report
        
        return metrics

    def _evaluate_regression(self, y_true, y_pred):
        """
        Calculate regression-specific metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error, r2_score,
            median_absolute_error, explained_variance_score
        )
        
        metrics = {}
        
        # Basic regression metrics
        metrics["mean_squared_error"] = float(mean_squared_error(y_true, y_pred))
        metrics["root_mean_squared_error"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics["mean_absolute_error"] = float(mean_absolute_error(y_true, y_pred))
        metrics["median_absolute_error"] = float(median_absolute_error(y_true, y_pred))
        metrics["r2"] = float(r2_score(y_true, y_pred))
        metrics["explained_variance"] = float(explained_variance_score(y_true, y_pred))
        
        # Calculate mean absolute percentage error (MAPE) avoiding division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-10))
            mape = np.mean(np.where(np.isfinite(mape), mape, 0)) * 100
        metrics["mean_absolute_percentage_error"] = float(mape)
        
        # Calculate residuals statistics
        residuals = y_true - y_pred
        metrics["residuals_mean"] = float(np.mean(residuals))
        metrics["residuals_std"] = float(np.std(residuals))
        
        # Check for residuals normality
        from scipy import stats
        _, p_value = stats.shapiro(residuals[:min(len(residuals), 5000)])
        metrics["residuals_normality_pvalue"] = float(p_value)
        
        # Calculate R² confidence intervals if configured
        if self.config["evaluation"]["confidence_intervals"]:
            r2_ci = self._bootstrap_r2_confidence_interval(y_true, y_pred)
            if r2_ci:
                metrics["r2_ci_lower"] = r2_ci[0]
                metrics["r2_ci_upper"] = r2_ci[1]
                
        return metrics

    def _bootstrap_r2_confidence_interval(self, y_true, y_pred, n_samples=1000, alpha=0.05):
        """
        Calculate bootstrap confidence interval for R².
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            n_samples: Number of bootstrap samples
            alpha: Alpha level for confidence interval
            
        Returns:
            Tuple of (lower, upper) confidence bounds
        """
        try:
            rng = np.random.RandomState(self.config["training"]["random_state"])
            n_samples = min(n_samples, self.config["evaluation"]["bootstrap_samples"])
            
            # Store bootstrap R² values
            bootstrap_r2 = []
            
            for _ in range(n_samples):
                # Generate bootstrap sample indices
                indices = rng.randint(0, len(y_true), len(y_true))
                
                # Calculate R² on bootstrap sample
                r2 = r2_score(y_true[indices], y_pred[indices])
                bootstrap_r2.append(r2)
                
            # Calculate confidence interval
            lower = np.percentile(bootstrap_r2, alpha/2 * 100)
            upper = np.percentile(bootstrap_r2, (1 - alpha/2) * 100)
            
            return float(lower), float(upper)
            
        except Exception as e:
            logger.warning(f"Error calculating R² confidence interval: {str(e)}")
            return None

    def _measure_prediction_time(self, X):
        """
        Measure average prediction time per sample.
        
        Args:
            X: Data to predict
            
        Returns:
            Average prediction time per sample in seconds
        """
        if not self.best_model:
            return 0
            
        try:
            # Warm up
            _ = self.best_model.predict(X[:min(10, len(X))])
            
            # Time predictions
            start_time = time.time()
            _ = self.best_model.predict(X)
            total_time = time.time() - start_time
            
            # Calculate average time per sample
            avg_time = total_time / len(X)
            
            return avg_time
            
        except Exception as e:
            logger.warning(f"Error measuring prediction time: {str(e)}")
            return 0

    def _generate_explanations(self, X):
        """
        Generate model explanations using configured method.
        
        Args:
            X: Features to explain
            
        Returns:
            Dictionary of explanations
        """
        explanation_method = self.config["evaluation"]["explanation_method"]
        if not explanation_method:
            return None
            
        try:
            # Take a small sample for explanations
            sample_size = min(100, len(X))
            X_sample = X[:sample_size]
            
            if explanation_method == "shap":
                return self._generate_shap_explanations(X_sample)
            elif explanation_method == "lime":
                return self._generate_lime_explanations(X_sample)
            elif explanation_method == "permutation":
                return self._generate_permutation_explanations(X)
            else:
                logger.warning(f"Unknown explanation method: {explanation_method}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating explanations: {str(e)}")
            return None

    def _generate_shap_explanations(self, X_sample):
        """
        Generate SHAP explanations.
        
        Args:
            X_sample: Sample features to explain
            
        Returns:
            Dictionary with SHAP explanations
        """
        try:
            import shap
            
            # Choose explainer based on model type
            if 'xgboost' in str(type(self.best_model)).lower():
                explainer = shap.TreeExplainer(self.best_model)
            elif hasattr(self.best_model, 'predict_proba'):
                explainer = shap.KernelExplainer(
                    self.best_model.predict_proba, shap.kmeans(X_sample, 10)
                )
            else:
                explainer = shap.KernelExplainer(
                    self.best_model.predict, shap.kmeans(X_sample, 10)
                )
                
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # Create and save summary plot
            plt.figure()
            if isinstance(shap_values, list):
                # Multi-class case
                shap.summary_plot(shap_values[0], X_sample, show=False)
            else:
                shap.summary_plot(shap_values, X_sample, show=False)
                
            plot_path = os.path.join(self.config["output"]["metrics_dir"], "shap_summary.png")
            plt.savefig(plot_path)
            plt.close()
            
            # Calculate global feature importance from SHAP
            if isinstance(shap_values, list):
                # For multi-class, average across classes
                importance = np.mean([np.mean(np.abs(s), axis=0) for s in shap_values], axis=0)
            else:
                importance = np.mean(np.abs(shap_values), axis=0)
                
            # Match with feature names
            feature_names = self.dataset_statistics.get("features", 
                                                     [f"feature_{i}" for i in range(len(importance))])
            shap_importance = dict(zip(feature_names[:len(importance)], importance.tolist()))
            
            # Keep only top 20 for return value
            shap_importance = dict(sorted(shap_importance.items(), 
                                        key=lambda x: x[1], reverse=True)[:20])
            
            return {
                "method": "shap",
                "feature_importance": shap_importance,
                "plot_path": plot_path
            }
            
        except ImportError:
            logger.warning("SHAP not installed. Install with 'pip install shap'")
            return None
        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {str(e)}")
            return None

    def _generate_lime_explanations(self, X_sample):
        """
        Generate LIME explanations.
        
        Args:
            X_sample: Sample features to explain
            
        Returns:
            Dictionary with LIME explanations
        """
        try:
            from lime import lime_tabular
            
            # Create explainer
            feature_names = self.dataset_statistics.get("features", None)
            explainer = lime_tabular.LimeTabularExplainer(
                X_sample.values if hasattr(X_sample, 'values') else X_sample,
                feature_names=feature_names,
                class_names=self.dataset_statistics.get("class_names", None),
                mode="classification" if self.config["data"]["problem_type"] in [
                    "binary_classification", "multiclass_classification"
                ] else "regression"
            )
            
            # Choose prediction function
            if hasattr(self.best_model, 'predict_proba'):
                predict_fn = self.best_model.predict_proba
            else:
                predict_fn = self.best_model.predict
                
            # Generate explanation for a single sample
            exp = explainer.explain_instance(
                X_sample.iloc[0].values if hasattr(X_sample, 'iloc') else X_sample[0],
                predict_fn,
                num_features=20
            )
            
            # Save explanation as HTML
            html_path = os.path.join(self.config["output"]["metrics_dir"], "lime_explanation.html")
            exp.save_to_file(html_path)
            
            # Get feature importance from explanation
            explanation = exp.as_list()
            lime_importance = dict(explanation)
            
            return {
                "method": "lime",
                "feature_importance": lime_importance,
                "html_path": html_path
            }
            
        except ImportError:
            logger.warning("LIME not installed. Install with 'pip install lime'")
            return None
        except Exception as e:
            logger.error(f"Error generating LIME explanations: {str(e)}")
            return None

    def _generate_permutation_explanations(self, X):
        """
        Generate permutation feature importance.
        
        Args:
            X: Features to explain
            
        Returns:
            Dictionary with permutation importance
        """
        try:
            from sklearn.inspection import permutation_importance
            
            # Generate permutation importance
            r = permutation_importance(
                self.best_model, X, self.y_test,
                n_repeats=10,
                random_state=self.config["training"]["random_state"]
            )
            
            # Match with feature names
            feature_names = self.dataset_statistics.get("features", 
                                                     [f"feature_{i}" for i in range(len(r.importances_mean))])
            perm_importance = dict(zip(feature_names[:len(r.importances_mean)], r.importances_mean.tolist()))
            
            # Sort and take top 20
            perm_importance = dict(sorted(perm_importance.items(), 
                                        key=lambda x: x[1], reverse=True)[:20])
            
            # Create and save plot
            plt.figure(figsize=(12, 8))
            sorted_idx = r.importances_mean.argsort()[::-1]
            plt.boxplot(r.importances[sorted_idx].T, vert=False, 
                       labels=[feature_names[i] for i in sorted_idx][:20])
            plt.title("Permutation Importance")
            plt.tight_layout()
            
            plot_path = os.path.join(self.config["output"]["metrics_dir"], "permutation_importance.png")
            plt.savefig(plot_path)
            plt.close()
            
            return {
                "method": "permutation",
                "feature_importance": perm_importance,
                "plot_path": plot_path
            }
            
        except Exception as e:
            logger.error(f"Error generating permutation explanations: {str(e)}")
            return None

    def _log_roc_curve(self, y_true, y_prob, is_test_set):
        """
        Create and log ROC curve plot.
        
        Args:
            y_true: True target values
            y_prob: Predicted probabilities
            is_test_set: Whether this is test data
        """
        try:
            from sklearn.metrics import roc_curve, auc
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 8))
            
            # Handle multi-class vs binary
            if self.config["data"]["problem_type"] == "multiclass_classification":
                # One-vs-Rest ROC curve for each class
                n_classes = y_prob.shape[1]
                
                for i in range(n_classes):
                    # Compute ROC curve and AUC
                    y_true_binary = (y_true == i).astype(int)
                    fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
                    roc_auc = auc(fpr, tpr)
                    
                    # Plot ROC curve
                    plt.plot(fpr, tpr, lw=2, 
                           label=f'Class {i} (AUC = {roc_auc:.2f})')
            else:
                # Binary classification
                if y_prob.ndim > 1 and y_prob.shape[1] >= 2:
                    # Use second column for positive class probability
                    y_score = y_prob[:, 1]
                else:
                    y_score = y_prob
                    
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.2f})')
                
            # Plot diagonal reference line
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True)
            
            # Save plot
            set_name = "test" if is_test_set else "val"
            plot_path = os.path.join(self.config["output"]["metrics_dir"], f"{set_name}_roc_curve.png")
            plt.savefig(plot_path)
            plt.close()
            
            # Log to MLflow
            if self.config["mlflow"]["enabled"] and mlflow.active_run():
                mlflow.log_artifact(plot_path)
                
        except Exception as e:
            logger.warning(f"Error creating ROC curve: {str(e)}")

    def _log_pr_curve(self, y_true, y_prob, is_test_set):
        """
        Create and log precision-recall curve plot.
        
        Args:
            y_true: True target values
            y_prob: Predicted probabilities
            is_test_set: Whether this is test data
        """
        try:
            from sklearn.metrics import precision_recall_curve, average_precision_score
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 8))
            
            # Handle multi-class vs binary
            if self.config["data"]["problem_type"] == "multiclass_classification":
                # One-vs-Rest PR curve for each class
                n_classes = y_prob.shape[1]
                
                for i in range(n_classes):
                    # Compute PR curve and AP
                    y_true_binary = (y_true == i).astype(int)
                    precision, recall, _ = precision_recall_curve(y_true_binary, y_prob[:, i])
                    ap = average_precision_score(y_true_binary, y_prob[:, i])
                    
                    # Plot PR curve
                    plt.plot(recall, precision, lw=2, 
                           label=f'Class {i} (AP = {ap:.2f})')
            else:
                # Binary classification
                if y_prob.ndim > 1 and y_prob.shape[1] >= 2:
                    # Use second column for positive class probability
                    y_score = y_prob[:, 1]
                else:
                    y_score = y_prob
                    
                precision, recall, _ = precision_recall_curve(y_true, y_score)
                ap = average_precision_score(y_true, y_score)
                
                plt.plot(recall, precision, lw=2, 
                       label=f'PR curve (AP = {ap:.2f})')
                
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="best")
            plt.grid(True)
            
            # Save plot
            set_name = "test" if is_test_set else "val"
            plot_path = os.path.join(self.config["output"]["metrics_dir"], f"{set_name}_pr_curve.png")
            plt.savefig(plot_path)
            plt.close()
            
            # Log to MLflow
            if self.config["mlflow"]["enabled"] and mlflow.active_run():
                mlflow.log_artifact(plot_path)
                
        except Exception as e:
            logger.warning(f"Error creating PR curve: {str(e)}")

    def _log_confusion_matrix(self, y_true, y_pred, is_test_set):
        """
        Create and log confusion matrix plot.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            is_test_set: Whether this is test data
        """
        try:
            from sklearn.metrics import confusion_matrix
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.dataset_statistics.get("class_names"), 
                       yticklabels=self.dataset_statistics.get("class_names"))
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            
            # Save plot
            set_name = "test" if is_test_set else "val"
            plot_path = os.path.join(self.config["output"]["metrics_dir"], f"{set_name}_confusion_matrix.png")
            plt.savefig(plot_path)
            plt.close()
            
            # Log to MLflow
            if self.config["mlflow"]["enabled"] and mlflow.active_run():
                mlflow.log_artifact(plot_path)
                
        except Exception as e:
            logger.warning(f"Error creating confusion matrix plot: {str(e)}")

    def _log_residuals_plot(self, y_true, y_pred, is_test_set):
        """
        Create and log residuals plot for regression.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            is_test_set: Whether this is test data
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Calculate residuals
            residuals = y_true - y_pred
            
            # Create plot
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Residuals vs Predicted
            axes[0].scatter(y_pred, residuals, alpha=0.5)
            axes[0].axhline(y=0, color='r', linestyle='-')
            axes[0].set_xlabel('Predicted Values')
            axes[0].set_ylabel('Residuals')
            axes[0].set_title('Residuals vs Predicted Values')
            axes[0].grid(True)
            
            # Residuals distribution
            sns.histplot(residuals, kde=True, ax=axes[1])
            axes[1].set_xlabel('Residuals')
            axes[1].set_title('Residuals Distribution')
            axes[1].grid(True)
            
            plt.tight_layout()
            
            # Save plot
            set_name = "test" if is_test_set else "val"
            plot_path = os.path.join(self.config["output"]["metrics_dir"], f"{set_name}_residuals.png")
            plt.savefig(plot_path)
            plt.close()
            
            # Log to MLflow
            if self.config["mlflow"]["enabled"] and mlflow.active_run():
                mlflow.log_artifact(plot_path)
                
        except Exception as e:
            logger.warning(f"Error creating residuals plot: {str(e)}")

    def _log_parameters(self):
        """Log training parameters to MLflow."""
        if not self.config["mlflow"]["enabled"] or not mlflow.active_run():
            return
            
        try:
            # Log key configuration parameters
            params = {
                "model_type": self.config["model"]["type"],
                "problem_type": self.config["data"]["problem_type"],
                "random_state": self.config["training"]["random_state"],
                "cv_strategy": self.config["training"]["cv_strategy"],
                "cv_folds": self.config["training"]["cv_folds"],
                "hyperparameter_optimization": str(self.config["hyperparameter_optimization"]["enabled"]),
                "hyperopt_method": self.config["hyperparameter_optimization"]["method"],
                "hyperopt_n_trials": self.config["hyperparameter_optimization"]["n_trials"]
            }
            
            # Log model-specific parameters
            if self.config["model"]["params"]:
                for k, v in self.config["model"]["params"].items():
                    params[f"model_param_{k}"] = str(v)
                    
            # Log parameters to MLflow
            mlflow.log_params(params)
            
            # Log full config as artifact
            config_path = os.path.join(self.config["output"]["metrics_dir"], "config.json")
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
                
            mlflow.log_artifact(config_path)
            
        except Exception as e:
            logger.warning(f"Error logging parameters to MLflow: {str(e)}")

    def train(self):
        """
        Execute the full training pipeline.
        
        Returns:
            Trained model
        """
        start_time = time.time()
        logger.info(f"Starting model training pipeline for {self.config['data']['problem_type']}")
        
        try:
            # Start MLflow run if enabled
            if self.config["mlflow"]["enabled"]:
                tags = self.config["mlflow"]["tags"] or {}
                tags.update({
                    "problem_type": self.config["data"]["problem_type"],
                    "model_type": self.config["model"]["type"],
                    "data_path": self.config["data"]["train_data_path"]
                })
                
                with mlflow.start_run(run_name=f"train_{int(time.time())}", tags=tags) as run:
                    self.run_id = run.info.run_id
                    logger.info(f"MLflow run started with ID: {self.run_id}")
                    
                    # Load and prepare data
                    X_train, y_train, X_val, y_val, X_test, y_test = self.load_and_preprocess_data()
                    
                    # Save references to data
                    self.X_train, self.y_train = X_train, y_train
                    self.X_val, self.y_val = X_val, y_val
                    self.X_test, self.y_test = X_test, y_test
                    
                    # Log parameters
                    self._log_parameters()
                    
                    # Optimize hyperparameters
                    if self.config["hyperparameter_optimization"]["enabled"]:
                        model = self.optimize_hyperparameters(X_train, y_train, X_val, y_val)
                    else:
                        model = self.create_model()
                    
                    # Train model
                    self._train_model(model, X_train, y_train, X_val, y_val)
                    
                    # Evaluate model
                    val_metrics = {}
                    if X_val is not None and y_val is not None:
                        val_metrics = self.evaluate(X_val, y_val, is_test_set=False)
                        logger.info(f"Validation metrics: {val_metrics}")
                    
                    test_metrics = {}
                    if X_test is not None and y_test is not None:
                        test_metrics = self.evaluate(X_test, y_test, is_test_set=True)
                        logger.info(f"Test metrics: {test_metrics}")
                    
                    # Save model
                    model_path = self.save_model()
                    
                    # Log model to MLflow
                    if self.config["mlflow"]["register_model"]:
                        model_info = self.register_model_in_mlflow(model_path)
                        logger.info(f"Model registered in MLflow: {model_info}")
                    
                    # Return results summary
                    return {
                        "model": self.best_model,
                        "model_path": model_path,
                        "training_time": time.time() - start_time,
                        "val_metrics": val_metrics,
                        "test_metrics": test_metrics,
                        "feature_importance": self.feature_importances,
                        "run_id": self.run_id
                    }
            else:
                # Training without MLflow
                # Load and prepare data
                X_train, y_train, X_val, y_val, X_test, y_test = self.load_and_preprocess_data()
                
                # Save references to data
                self.X_train, self.y_train = X_train, y_train
                self.X_val, self.y_val = X_val, y_val
                self.X_test, self.y_test = X_test, y_test
                
                # Optimize hyperparameters
                if self.config["hyperparameter_optimization"]["enabled"]:
                    model = self.optimize_hyperparameters(X_train, y_train, X_val, y_val)
                else:
                    model = self.create_model()
                
                # Train model
                self._train_model(model, X_train, y_train, X_val, y_val)
                
                # Evaluate model
                val_metrics = {}
                if X_val is not None and y_val is not None:
                    val_metrics = self.evaluate(X_val, y_val, is_test_set=False)
                    logger.info(f"Validation metrics: {val_metrics}")
                
                test_metrics = {}
                if X_test is not None and y_test is not None:
                    test_metrics = self.evaluate(X_test, y_test, is_test_set=True)
                    logger.info(f"Test metrics: {test_metrics}")
                
                # Save model
                model_path = self.save_model()
                
                # Return results summary
                return {
                    "model": self.best_model,
                    "model_path": model_path,
                    "training_time": time.time() - start_time,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "feature_importance": self.feature_importances
                }
                
        except Exception as e:
            logger.error(f"Error in training pipeline: {str(e)}", exc_info=True)
            raise
        finally:
            logger.info(f"Training pipeline completed in {time.time() - start_time:.2f} seconds")

    def save_model(self):
        """
        Save trained model to disk.
        
        Returns:
            Path to saved model
        """
        if not self.best_model:
            logger.error("No model available to save")
            return None
            
        try:
            # Create output directory if it doesn't exist
            os.makedirs(self.config["output"]["model_dir"], exist_ok=True)
            
            # Generate model name if not provided
            model_name = self.config["output"]["model_name"]
            if not model_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = f"{self.config['model']['type']}_{timestamp}"
                
            # Add version if configured
            if self.config["output"]["version_models"]:
                model_version = datetime.now().strftime("%Y%m%d%H%M%S")
                model_name = f"{model_name}_v{model_version}"
                
            # Determine file extension
            save_format = self.config["output"]["save_format"]
            if save_format == "joblib":
                ext = ".joblib"
            elif save_format == "pickle":
                ext = ".pkl"
            elif save_format == "onnx":
                ext = ".onnx"
            elif save_format == "savedmodel":
                ext = ""  # TensorFlow SavedModel is a directory
            elif save_format == "torchscript":
                ext = ".pt"
            else:
                ext = ".joblib"  # Default to joblib
                
            # Create full path
            model_path = os.path.join(self.config["output"]["model_dir"], f"{model_name}{ext}")
            
            # Save model based on format
            if save_format == "joblib":
                joblib.dump(self.best_model, model_path, compress=self.config["output"]["compress"])
            elif save_format == "pickle":
                with open(model_path, 'wb') as f:
                    pickle.dump(self.best_model, f)
            elif save_format == "onnx":
                self._save_as_onnx(model_path)
            elif save_format == "savedmodel":
                self._save_as_tensorflow(model_path)
            elif save_format == "torchscript":
                self._save_as_torchscript(model_path)
                
            logger.info(f"Model saved to {model_path}")
            
            # Save preprocessor if configured
            if self.config["output"]["save_preprocessing"] and self.preprocessor:
                preprocessor_path = os.path.join(self.config["output"]["model_dir"], f"{model_name}_preprocessor{ext}")
                if save_format == "joblib":
                    joblib.dump(self.preprocessor, preprocessor_path, compress=self.config["output"]["compress"])
                else:
                    with open(preprocessor_path, 'wb') as f:
                        pickle.dump(self.preprocessor, f)
                        
                logger.info(f"Preprocessor saved to {preprocessor_path}")
                
            # Save model in multiple formats if configured
            if self.config["output"]["multiple_formats"]:
                self._save_in_multiple_formats(model_name)
                
            # Export model for serving if configured
            if self.config["output"]["export_for_serving"]:
                self._export_for_serving(model_name)
                
            return model_path
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}", exc_info=True)
            return None

    def _save_as_onnx(self, model_path):
        """
        Save model in ONNX format.
        
        Args:
            model_path: Path to save model
        """
        try:
            import onnxmltools
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            
            # Get number of features
            n_features = self.X_train.shape[1]
            
            # Define initial types
            initial_type = [('input', FloatTensorType([None, n_features]))]
            
            # Convert model to ONNX
            onx = convert_sklearn(self.best_model, initial_types=initial_type)
            
            # Save model
            with open(model_path, "wb") as f:
                f.write(onx.SerializeToString())
                
            logger.info(f"Model saved in ONNX format to {model_path}")
            
        except ImportError:
            logger.error("ONNX conversion requires onnxmltools and skl2onnx. Install with 'pip install onnxmltools skl2onnx'")
            raise
        except Exception as e:
            logger.error(f"Error saving model as ONNX: {str(e)}", exc_info=True)
            raise

    def _save_as_tensorflow(self, model_path):
        """
        Save model in TensorFlow SavedModel format.
        
        Args:
            model_path: Path to save model
        """
        try:
            import tensorflow as tf
            
            # Check if model is already a TensorFlow model
            if hasattr(self.best_model, 'model') and isinstance(self.best_model.model, tf.keras.Model):
                # Direct save for Keras models
                self.best_model.model.save(model_path)
            else:
                # Create a TensorFlow wrapper for scikit-learn models
                input_shape = self.X_train.shape[1:]
                
                class SklearnModel(tf.keras.Model):
                    def __init__(self, sklearn_model):
                        super().__init__()
                        self.sklearn_model = sklearn_model
                        
                    def call(self, inputs):
                        # Convert to numpy
                        x = inputs.numpy() if hasattr(inputs, 'numpy') else inputs
                        # Reshape if needed
                        if len(x.shape) == 1:
                            x = x.reshape(1, -1)
                        # Make prediction
                        if hasattr(self.sklearn_model, 'predict_proba'):
                            result = self.sklearn_model.predict_proba(x)
                            return tf.convert_to_tensor(result, dtype=tf.float32)
                        else:
                            result = self.sklearn_model.predict(x)
                            return tf.convert_to_tensor(result, dtype=tf.float32)
                
                # Create model
                tf_model = SklearnModel(self.best_model)
                
                # Create input signature
                input_signature = [tf.TensorSpec(shape=(None,) + input_shape, dtype=tf.float32)]
                
                # Create and save model
                tf.saved_model.save(
                    tf_model,
                    model_path,
                    signatures={'serving_default': tf_model.call.get_concrete_function(
                        tf.TensorSpec(shape=(None,) + input_shape, dtype=tf.float32)
                    )}
                )
                
            logger.info(f"Model saved in TensorFlow SavedModel format to {model_path}")
            
        except ImportError:
            logger.error("TensorFlow SavedModel conversion requires tensorflow. Install with 'pip install tensorflow'")
            raise
        except Exception as e:
            logger.error(f"Error saving model as TensorFlow SavedModel: {str(e)}", exc_info=True)
            raise

    def _save_as_torchscript(self, model_path):
        """
        Save model in PyTorch TorchScript format.
        
        Args:
            model_path: Path to save model
        """
        try:
            import torch
            
            # Check if model is a PyTorch model
            if hasattr(self.best_model, 'module_') or 'skorch' in str(type(self.best_model)):
                # Get the PyTorch module
                if hasattr(self.best_model, 'module_'):
                    module = self.best_model.module_
                else:
                    module = self.best_model
                    
                # Create example input
                example_input = torch.rand(1, self.X_train.shape[1])
                
                # Trace the model
                traced_model = torch.jit.trace(module, example_input)
                
                # Save the traced model
                torch.jit.save(traced_model, model_path)
                
                logger.info(f"Model saved in TorchScript format to {model_path}")
            else:
                logger.error("Model is not a PyTorch model, cannot save as TorchScript")
                raise ValueError("Model is not a PyTorch model")
                
        except ImportError:
            logger.error("TorchScript conversion requires PyTorch. Install with 'pip install torch'")
            raise
        except Exception as e:
            logger.error(f"Error saving model as TorchScript: {str(e)}", exc_info=True)
            raise

    def _save_in_multiple_formats(self, model_name):
        """
        Save model in multiple formats.
        
        Args:
            model_name: Base name for model files
        """
        # Define formats to save
        formats = ["joblib", "pickle"]
        
        # Try to save in each format
        for fmt in formats:
            try:
                # Skip if it's the primary format
                if fmt == self.config["output"]["save_format"]:
                    continue
                    
                # Set file extension
                if fmt == "joblib":
                    ext = ".joblib"
                elif fmt == "pickle":
                    ext = ".pkl"
                elif fmt == "onnx":
                    ext = ".onnx"
                else:
                    ext = f".{fmt}"
                    
                # Create path
                path = os.path.join(self.config["output"]["model_dir"], f"{model_name}_{fmt}{ext}")
                
                # Save based on format
                if fmt == "joblib":
                    joblib.dump(self.best_model, path, compress=self.config["output"]["compress"])
                elif fmt == "pickle":
                    with open(path, 'wb') as f:
                        pickle.dump(self.best_model, f)
                elif fmt == "onnx":
                    self._save_as_onnx(path)
                elif fmt == "savedmodel":
                    self._save_as_tensorflow(path)
                elif fmt == "torchscript":
                    self._save_as_torchscript(path)
                    
                logger.info(f"Model also saved in {fmt} format: {path}")
                
            except Exception as e:
                logger.warning(f"Failed to save model in {fmt} format: {str(e)}")

    def _export_for_serving(self, model_name):
        """
        Export model for production serving.
        
        Args:
            model_name: Base name for model files
        """
        platform = self.config["output"]["export_platform"]
        
        try:
            if platform == "tensorflow-serving":
                # Export as TensorFlow SavedModel with versioning
                export_dir = os.path.join(self.config["output"]["model_dir"], "tensorflow-serving", model_name, "1")
                self._save_as_tensorflow(export_dir)
                
                # Create assets and variables directories if they don't exist
                os.makedirs(os.path.join(export_dir, "assets"), exist_ok=True)
                os.makedirs(os.path.join(export_dir, "variables"), exist_ok=True)
                
                logger.info(f"Model exported for TensorFlow Serving: {export_dir}")
                
            elif platform == "torchserve":
                import torch
                from torch.package import PackageExporter
                
                # Create MAR file
                export_dir = os.path.join(self.config["output"]["model_dir"], "torchserve")
                os.makedirs(export_dir, exist_ok=True)
                
                # Save model in torch format
                model_file = os.path.join(export_dir, f"{model_name}.pt")
                self._save_as_torchscript(model_file)
                
                # Create properties file
                with open(os.path.join(export_dir, f"{model_name}.properties"), 'w') as f:
                    f.write("model_name={}\n".format(model_name))
                    f.write("model_version=1.0\n")
                    f.write("handler=model_handler.py\n")
                    
                logger.info(f"Model exported for TorchServe: {export_dir}")
                
            else:
                logger.warning(f"Unsupported serving platform: {platform}")
                
        except Exception as e:
            logger.error(f"Error exporting model for serving: {str(e)}", exc_info=True)

    def register_model_in_mlflow(self, model_path):
        """
        Register model in MLflow Model Registry.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Model version info
        """
        if not self.config["mlflow"]["enabled"] or not mlflow.active_run():
            logger.warning("MLflow not enabled, skipping model registration")
            return None
            
        try:
            # Get model name
            model_name = self.config["mlflow"]["registry_model_name"]
            if not model_name:
                model_name = self.config["output"]["model_name"] or "default_model"
                
            # Log model to MLflow
            mlflow.sklearn.log_model(
                self.best_model,
                "model",
                registered_model_name=model_name
            )
            
            logger.info(f"Model registered in MLflow Model Registry as {model_name}")
            
            # Get model version
            client = mlflow.tracking.MlflowClient()
            model_versions = client.search_model_versions(f"name='{model_name}'")
            latest_version = max([int(mv.version) for mv in model_versions]) if model_versions else 0
            
            return {
                "name": model_name,
                "version": latest_version,
                "run_id": self.run_id
            }
            
        except Exception as e:
            logger.error(f"Error registering model in MLflow: {str(e)}", exc_info=True)
            return None

    def load_model(self, model_path):
        """
        Load a saved model.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Loaded model
        """
        try:
            # Determine load method based on file extension
            if model_path.endswith(".joblib"):
                model = joblib.load(model_path)
            elif model_path.endswith(".pkl"):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            elif model_path.endswith(".onnx"):
                import onnxruntime as ort
                model = ort.InferenceSession(model_path)
            elif os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "saved_model.pb")):
                # TensorFlow SavedModel
                import tensorflow as tf
                model = tf.saved_model.load(model_path)
            elif model_path.endswith(".pt") or model_path.endswith(".pth"):
                import torch
                model = torch.jit.load(model_path)
            else:
                # Try joblib as default
                model = joblib.load(model_path)
                
            logger.info(f"Model loaded successfully from {model_path}")
            self.best_model = model
            return model
            
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}", exc_info=True)
            raise

    def predict(self, X, return_proba=False):
        """
        Make predictions using the trained model.
        
        Args:
            X: Features to predict on
            return_proba: Whether to return probabilities (for classification)
            
        Returns:
            Predictions, optionally with probabilities
        """
        if self.best_model is None:
            raise ValueError("No model available. Train or load a model first.")
            
        try:
            # Preprocess input if preprocessor is available
            if self.preprocessor is not None:
                X = self._apply_preprocessor(X)
                
            # Make predictions
            predictions = self.best_model.predict(X)
            
            # Return probabilities if requested and available (for classification)
            if return_proba and hasattr(self.best_model, 'predict_proba'):
                probabilities = self.best_model.predict_proba(X)
                return predictions, probabilities
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}", exc_info=True)
            raise

    def export_for_deployment(self, output_dir=None, format="all"):
        """
        Export model for deployment with all necessary artifacts.
        
        Args:
            output_dir: Directory to export to (defaults to config output dir)
            format: Export format - 'all', 'onnx', 'tf', 'torch', or 'joblib'
            
        Returns:
            Dict with paths to exported artifacts
        """
        if self.best_model is None:
            raise ValueError("No model available. Train or load a model first.")
            
        try:
            # Use config output dir if not specified
            if output_dir is None:
                output_dir = self.config["output"]["model_dir"]
                
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate base filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"deployment_model_{timestamp}"
            
            # Export model in specified formats
            artifacts = {"metadata": {}, "files": {}}
            
            # Build metadata
            artifacts["metadata"] = {
                "created_at": datetime.now().isoformat(),
                "problem_type": self.config["data"]["problem_type"],
                "model_type": self.config["model"]["type"],
                "feature_count": self.dataset_statistics.get("n_features", None),
                "training_samples": self.dataset_statistics.get("train_samples", None),
                "model_size_bytes": self.model_size_bytes,
                "model_performance": {}
            }
            
            # Add performance metrics if available
            if hasattr(self, "evaluation_results"):
                artifacts["metadata"]["model_performance"] = self.evaluation_results
            
            # Save as joblib (default scikit-learn format)
            if format in ["all", "joblib"]:
                model_path = os.path.join(output_dir, f"{base_name}.joblib")
                joblib.dump(self.best_model, model_path, compress=3)
                artifacts["files"]["joblib"] = model_path
                
                # Save preprocessor if available
                if self.preprocessor is not None:
                    preprocessor_path = os.path.join(output_dir, f"{base_name}_preprocessor.joblib")
                    joblib.dump(self.preprocessor, preprocessor_path, compress=3)
                    artifacts["files"]["preprocessor"] = preprocessor_path
            
            # Export as ONNX
            if format in ["all", "onnx"]:
                try:
                    import skl2onnx
                    from skl2onnx.common.data_types import FloatTensorType
                    
                    # Get number of features
                    n_features = self.X_train.shape[1] if hasattr(self, "X_train") else None
                    if n_features is None:
                        logger.warning("Feature count unknown, using default of 10 for ONNX export")
                        n_features = 10
                        
                    # Create initial type for conversion
                    initial_type = [('float_input', FloatTensorType([None, n_features]))]
                    
                    # Convert to ONNX
                    onnx_model = skl2onnx.convert_sklearn(
                        self.best_model, 
                        initial_types=initial_type,
                        name=base_name
                    )
                    
                    # Save model
                    onnx_path = os.path.join(output_dir, f"{base_name}.onnx")
                    with open(onnx_path, "wb") as f:
                        f.write(onnx_model.SerializeToString())
                        
                    artifacts["files"]["onnx"] = onnx_path
                    logger.info(f"Model exported to ONNX: {onnx_path}")
                except ImportError:
                    logger.warning("skl2onnx not installed, skipping ONNX export")
                except Exception as e:
                    logger.error(f"Error exporting to ONNX: {str(e)}")
            
            # Export as TensorFlow SavedModel
            if format in ["all", "tf"]:
                try:
                    import tensorflow as tf
                    
                    # Create TF wrapper for sklearn model
                    class SklearnModelWrapper(tf.Module):
                        def __init__(self, model):
                            self.model = model
                        
                        @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
                        def predict(self, x):
                            # Convert to numpy, make prediction, convert back to tensor
                            result = self.model.predict(x.numpy())
                            return tf.convert_to_tensor(result, dtype=tf.float32)
                        
                        @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
                        def predict_proba(self, x):
                            if hasattr(self.model, 'predict_proba'):
                                result = self.model.predict_proba(x.numpy())
                                return tf.convert_to_tensor(result, dtype=tf.float32)
                            else:
                                return tf.convert_to_tensor([[0.0]], dtype=tf.float32)
                    
                    # Create and save wrapper
                    tf_wrapper = SklearnModelWrapper(self.best_model)
                    tf_path = os.path.join(output_dir, f"{base_name}_tf")
                    tf.saved_model.save(tf_wrapper, tf_path)
                    
                    artifacts["files"]["tensorflow"] = tf_path
                    logger.info(f"Model exported to TensorFlow: {tf_path}")
                except ImportError:
                    logger.warning("tensorflow not installed, skipping TensorFlow export")
                except Exception as e:
                    logger.error(f"Error exporting to TensorFlow: {str(e)}")
            
            # Export as inference script with dependencies
            inference_script = self._generate_inference_script(base_name)
            script_path = os.path.join(output_dir, "inference.py")
            
            with open(script_path, "w") as f:
                f.write(inference_script)
                
            artifacts["files"]["inference_script"] = script_path
            
            # Save metadata
            metadata_path = os.path.join(output_dir, f"{base_name}_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(artifacts["metadata"], f, indent=2)
                
            artifacts["files"]["metadata"] = metadata_path
            
            logger.info(f"Model exported for deployment to {output_dir}")
            return artifacts
            
        except Exception as e:
            logger.error(f"Error exporting model for deployment: {str(e)}", exc_info=True)
            raise

    def _generate_inference_script(self, model_name):
        """Generate a standalone inference script for the model."""
        script = f"""#!/usr/bin/env python
# Model inference script for {model_name}
# Generated on {datetime.now().isoformat()}

import argparse
import json
import sys
import os
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Union, Any

# Load model and preprocessor
MODEL_PATH = os.path.join(os.path.dirname(__file__), "{model_name}.joblib")
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), "{model_name}_preprocessor.joblib")

def load_model():
    \"\"\"Load the trained model and preprocessor.\"\"\"
    model = joblib.load(MODEL_PATH)
    
    # Load preprocessor if available
    preprocessor = None
    if os.path.exists(PREPROCESSOR_PATH):
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        
    return model, preprocessor

def preprocess(data, preprocessor):
    \"\"\"Preprocess input data using the trained preprocessor.\"\"\"
    if preprocessor is None:
        return data
    
    # Apply preprocessing
    return preprocessor.transform(data)

def predict(data, model, preprocessor=None, return_proba=False):
    \"\"\"
    Make predictions with the model.
    
    Args:
        data: DataFrame or array of features
        model: Trained model
        preprocessor: Optional preprocessor
        return_proba: Whether to return probabilities
        
    Returns:
        Predictions and optionally probabilities
    \"\"\"
    # Convert data to appropriate format
    if isinstance(data, pd.DataFrame):
        X = data.values
    else:
        X = np.array(data)
    
    # Reshape single sample
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    
    # Apply preprocessing if provided
    if preprocessor is not None:
        X = preprocess(X, preprocessor)
    
    # Make prediction
    y_pred = model.predict(X)
    
    # Return probabilities if requested and available
    if return_proba and hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X)
        return y_pred, y_proba
    
    return y_pred

def predict_from_json(json_data, model, preprocessor=None):
    \"\"\"
    Make predictions from JSON data.
    
    Args:
        json_data: JSON string or dict with 'features' key
        model: Trained model
        preprocessor: Optional preprocessor
        
    Returns:
        Dict with prediction results
    \"\"\"
    # Parse JSON if string
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    # Extract features
    if 'features' not in data:
        raise ValueError("JSON data must contain 'features' key")
        
    features = data['features']
    return_proba = data.get('return_probabilities', False)
    
    # Make prediction
    if return_proba and hasattr(model, 'predict_proba'):
        y_pred, y_proba = predict(features, model, preprocessor, return_proba=True)
        result = {
            'predictions': y_pred.tolist(),
            'probabilities': y_proba.tolist()
        }
    else:
        y_pred = predict(features, model, preprocessor)
        result = {
            'predictions': y_pred.tolist()
        }
    
    return result

def main():
    \"\"\"Command-line interface for model inference.\"\"\"
    parser = argparse.ArgumentParser(description='Model Inference Script')
    parser.add_argument('--input', '-i', type=str, required=True,
                      help='Path to input file (CSV) or JSON string')
    parser.add_argument('--output', '-o', type=str, default=None,
                      help='Path to output file (JSON)')
    parser.add_argument('--proba', '-p', action='store_true',
                      help='Include prediction probabilities')
    
    args = parser.parse_args()
    
    # Load model and preprocessor
    model, preprocessor = load_model()
    
    # Process input
    if args.input.endswith('.csv'):
        # CSV input
        data = pd.read_csv(args.input)
        
        if args.proba and hasattr(model, 'predict_proba'):
            y_pred, y_proba = predict(data, model, preprocessor, return_proba=True)
            result = {
                'predictions': y_pred.tolist(),
                'probabilities': y_proba.tolist()
            }
        else:
            y_pred = predict(data, model, preprocessor)
            result = {
                'predictions': y_pred.tolist()
            }
    else:
        # Assume JSON input
        with open(args.input, 'r') as f:
            json_data = f.read()
            
        result = predict_from_json(json_data, model, preprocessor)
    
    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
    else:
        print(json.dumps(result, indent=2))
    
if __name__ == '__main__':
    main()
"""
        return script

    def calibrate_model(self, X_cal, y_cal, method='isotonic'):
        """
        Calibrate model probabilities for better uncertainty estimates.
        
        Args:
            X_cal: Calibration features
            y_cal: Calibration targets
            method: Calibration method ('isotonic' or 'sigmoid')
            
        Returns:
            Calibrated model
        """
        if self.config["data"]["problem_type"] not in ["binary_classification", "multiclass_classification"]:
            logger.warning("Probability calibration only applicable for classification problems")
            return self.best_model
            
        try:
            from sklearn.calibration import CalibratedClassifierCV
            
            # Create calibrated model
            calibrated_model = CalibratedClassifierCV(
                self.best_model,
                method=method,
                cv='prefit'  # Use already fitted model
            )
            
            # Fit calibration model
            calibrated_model.fit(X_cal, y_cal)
            
            # Replace model with calibrated version
            self.best_model = calibrated_model
            
            # Log calibration info to MLflow if enabled
            if self.config["mlflow"]["enabled"] and mlflow.active_run():
                mlflow.log_param("calibration_method", method)
                
                # Evaluate calibration quality
                from sklearn.metrics import brier_score_loss
                y_prob = calibrated_model.predict_proba(X_cal)
                if y_prob.shape[1] == 2:  # Binary case
                    brier = brier_score_loss(y_cal, y_prob[:, 1])
                else:  # Multiclass case - use average
                    brier_scores = []
                    for i in range(y_prob.shape[1]):
                        bin_y = (y_cal == i).astype(int)
                        brier_scores.append(brier_score_loss(bin_y, y_prob[:, i]))
                    brier = np.mean(brier_scores)
                    
                mlflow.log_metric("calibration_brier_score", brier)
                
                # Create calibration plots
                self._log_calibration_plots(X_cal, y_cal)
                
            logger.info(f"Model calibrated using {method} method")
            return self.best_model
            
        except ImportError:
            logger.warning("scikit-learn not installed with calibration support")
            return self.best_model
        except Exception as e:
            logger.error(f"Error calibrating model: {str(e)}", exc_info=True)
            return self.best_model

    def _log_calibration_plots(self, X, y):
        """Create and log calibration plots."""
        try:
            import matplotlib.pyplot as plt
            from sklearn.calibration import calibration_curve
            
            plt.figure(figsize=(10, 8))
            
            # Get predictions from calibrated model
            y_prob = self.best_model.predict_proba(X)
            
            # Handle binary vs multiclass
            if y_prob.shape[1] == 2:  # Binary case
                prob_pos = y_prob[:, 1]
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y, prob_pos, n_bins=10)
                    
                plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                       label="Calibrated Model")
                       
                # Add reference line
                plt.plot([0, 1], [0, 1], "k:", label="Perfect calibration")
                
                plt.xlabel("Mean predicted probability")
                plt.ylabel("Fraction of positives")
                plt.title("Calibration Curve")
                plt.legend(loc="best")
                plt.grid(True)
                
            else:  # Multiclass case - show top 3 classes
                for i in range(min(3, y_prob.shape[1])):
                    bin_y = (y == i).astype(int)
                    prob_pos = y_prob[:, i]
                    
                    fraction_of_positives, mean_predicted_value = calibration_curve(
                        bin_y, prob_pos, n_bins=10)
                        
                    plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                           label=f"Class {i}")
                           
                # Add reference line
                plt.plot([0, 1], [0, 1], "k:", label="Perfect calibration")
                
                plt.xlabel("Mean predicted probability")
                plt.ylabel("Fraction of positives")
                plt.title("Calibration Curves")
                plt.legend(loc="best")
                plt.grid(True)
                
            # Save plot and log to MLflow
            plot_path = os.path.join(self.config["output"]["metrics_dir"], "calibration_curve.png")
            plt.savefig(plot_path)
            plt.close()
            
            if self.config["mlflow"]["enabled"] and mlflow.active_run():
                mlflow.log_artifact(plot_path)
                
        except Exception as e:
            logger.warning(f"Error creating calibration plots: {str(e)}")

    def crossvalidate_extensive(self, X, y, n_splits=5, scoring_metrics=None):
        """
        Perform extensive cross-validation with multiple metrics.
        
        Args:
            X: Features
            y: Targets
            n_splits: Number of CV splits
            scoring_metrics: List of metrics to evaluate
            
        Returns:
            Dictionary of CV results
        """
        try:
            from sklearn.model_selection import cross_validate, KFold, StratifiedKFold
            
            # Use default metrics if not provided
            if scoring_metrics is None:
                if self.config["data"]["problem_type"] in ["binary_classification", "multiclass_classification"]:
                    scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc_ovr']
                else:
                    scoring_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
            
            # Choose CV strategy based on problem type
            if self.config["data"]["problem_type"] in ["binary_classification", "multiclass_classification"]:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.config["training"]["random_state"])
            else:
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.config["training"]["random_state"])
                
            # Run cross-validation
            cv_results = cross_validate(
                self.best_model,
                X, y,
                cv=cv,
                scoring=scoring_metrics,
                return_train_score=True,
                return_estimator=True,
                n_jobs=-1
            )
            
            # Format results
            results = {
                "cv_splits": n_splits,
                "metrics": {},
                "estimators": cv_results["estimator"]
            }
            
            # Process metrics
            for metric in scoring_metrics:
                test_scores = cv_results[f"test_{metric}"]
                train_scores = cv_results[f"train_{metric}"]
                
                results["metrics"][metric] = {
                    "test_mean": float(np.mean(test_scores)),
                    "test_std": float(np.std(test_scores)),
                    "test_scores": test_scores.tolist(),
                    "train_mean": float(np.mean(train_scores)),
                    "train_std": float(np.std(train_scores)),
                    "train_scores": train_scores.tolist()
                }
                
            # Log to MLflow if enabled
            if self.config["mlflow"]["enabled"] and mlflow.active_run():
                for metric, values in results["metrics"].items():
                    mlflow.log_metric(f"cv_{metric}_mean", values["test_mean"])
                    mlflow.log_metric(f"cv_{metric}_std", values["test_std"])
                    
            logger.info(f"Extensive cross-validation completed with {n_splits} splits")
            return results
            
        except Exception as e:
            logger.error(f"Error in extensive cross-validation: {str(e)}", exc_info=True)
            return {"error": str(e)}



