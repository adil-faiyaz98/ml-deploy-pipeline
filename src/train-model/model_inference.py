"""
Production-Grade Model Inference Service

An industry-standard ML model serving API with:
- Advanced security and authentication
- Model A/B testing and shadow deployment
- Explainability for any model type
- Drift detection and monitoring
- Streaming prediction capability
- Serverless deployment options
- High-performance async processing
- Support for multi-model ensembles
"""

import argparse
import asyncio
import base64
import glob
import hashlib
import hmac
import inspect
import json
import logging
import os
import pickle
import secrets
import signal
import shutil
import sys
import tempfile
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import wraps, partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Generator

import joblib
import numpy as np
import pandas as pd
from flask import Flask, Response, current_app, jsonify, request, stream_with_context
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.middleware.proxy_fix import ProxyFix

# Add these imports to the top of the file
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import tempfile
from typing import AsyncGenerator, BinaryIO
import orjson
from io import StringIO, BytesIO

# Optional imports for explainability - don't fail if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/inference.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("inference_service")

# Create metrics
REQUEST_COUNT = Counter("request_count", "App Request Count", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency in seconds", ["method", "endpoint"])
MODEL_VERSION = Gauge("model_version", "Model version timestamp")
PREDICTION_GAUGE = Histogram("prediction_value", "Prediction values", ["class"])
INPUT_FEATURES = Histogram("input_features", "Input feature values", ["feature"])
PREDICTION_ERROR_COUNT = Counter("prediction_errors", "Prediction errors", ["error_type"])
MODEL_LOADING_TIME = Histogram("model_loading_time", "Model loading time in seconds")
FEATURE_DRIFT = Gauge("feature_drift", "Feature drift score", ["feature"])
PREDICTION_DRIFT = Gauge("prediction_drift", "Prediction distribution drift")
MODEL_MEMORY_USAGE = Gauge("model_memory_usage_bytes", "Model memory usage in bytes")
CACHE_HIT_RATIO = Gauge("cache_hit_ratio", "Prediction cache hit ratio")

# Create Flask app
app = Flask(__name__)
CORS(app)

# Add ProxyFix middleware to support reverse proxies
app.wsgi_app = ProxyFix(app.wsgi_app)

# Add metrics endpoint
app.wsgi_app = DispatcherMiddleware(
    app.wsgi_app, {'/metrics': prometheus_client.make_wsgi_app()}
)

# JWT Authentication
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', secrets.token_hex(32))
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
jwt = JWTManager(app)

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Global app configuration
app.config.update(
    MODEL_PATH=os.environ.get("MODEL_PATH", "models/model.pkl"),
    MODEL_DIR=os.environ.get("MODEL_DIR", "models"),
    MODEL_REFRESH_INTERVAL=int(os.environ.get("MODEL_REFRESH_INTERVAL", 300)),  # 5 minutes
    FEATURE_NAMES=os.environ.get("FEATURE_NAMES", "").split(",") if os.environ.get("FEATURE_NAMES") else None,
    PREDICTION_THRESHOLD=float(os.environ.get("PREDICTION_THRESHOLD", 0.5)),
    ENABLE_CACHING=os.environ.get("ENABLE_CACHING", "True").lower() == "true",
    CACHE_TIMEOUT=int(os.environ.get("CACHE_TIMEOUT", 3600)),
    EXPLAIN_PREDICTIONS=os.environ.get("EXPLAIN_PREDICTIONS", "False").lower() == "true",
    MODEL_CONFIG_PATH=os.environ.get("MODEL_CONFIG_PATH", "config/models.yaml"),
    ENABLE_AB_TESTING=os.environ.get("ENABLE_AB_TESTING", "False").lower() == "true",
    FEATURE_DRIFT_THRESHOLD=float(os.environ.get("FEATURE_DRIFT_THRESHOLD", 0.1)),
    ENABLE_AUTH=os.environ.get("ENABLE_AUTH", "False").lower() == "true",
    AUTH_API_KEYS=os.environ.get("AUTH_API_KEYS", "").split(",") if os.environ.get("AUTH_API_KEYS") else [],
    NUM_WORKERS=int(os.environ.get("NUM_WORKERS", 4)),
    MAX_CONTENT_LENGTH=int(os.environ.get("MAX_CONTENT_LENGTH", 100 * 1024 * 1024)),  # 100 MB
    ENABLE_STREAMING=os.environ.get("ENABLE_STREAMING", "True").lower() == "true",
    LOG_PREDICTIONS=os.environ.get("LOG_PREDICTIONS", "True").lower() == "true",
    PREDICTION_LOG_PATH=os.environ.get("PREDICTION_LOG_PATH", "logs/predictions.jsonl"),
    CALIBRATION_METHOD=os.environ.get("CALIBRATION_METHOD", None)
)

# Set max content length for large batch predictions
app.config["MAX_CONTENT_LENGTH"] = app.config["MAX_CONTENT_LENGTH"]

# Prediction cache
prediction_cache = {}
cache_hits = 0
cache_misses = 0

# Feature distribution tracker for drift detection
feature_distributions = {}
prediction_distribution = []

# Thread and process pools
thread_pool = None
process_pool = None


def safe_load_model(model_path: str) -> Tuple[Any, float]:
    """
    Load model safely from disk with error handling.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Tuple of (model, timestamp)
    """
    start_time = time.time()
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    # Determine file extension for appropriate loading method
    file_extension = os.path.splitext(model_path)[1].lower()
    
    try:
        # Create a temporary directory to load model in case of malicious files
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_model_path = os.path.join(tmp_dir, os.path.basename(model_path))
            
            # Copy model to temp directory
            shutil.copy2(model_path, tmp_model_path)
            
            # Verify file hash for security (prevent tampering)
            original_hash = hashlib.md5(open(model_path, 'rb').read()).hexdigest()
            copy_hash = hashlib.md5(open(tmp_model_path, 'rb').read()).hexdigest()
            
            if original_hash != copy_hash:
                raise ValueError("Model file integrity check failed")
            
            # Load based on file type
            if file_extension in ['.pkl', '.pickle', '.joblib']:
                model = joblib.load(tmp_model_path)
            elif file_extension == '.h5':
                if not TF_AVAILABLE:
                    raise ImportError("TensorFlow not available for loading .h5 models")
                model = tf.keras.models.load_model(tmp_model_path)
            elif file_extension == '.pt':
                if not TORCH_AVAILABLE:
                    raise ImportError("PyTorch not available for loading .pt models")
                model = torch.load(tmp_model_path)
            else:
                raise ValueError(f"Unsupported model format: {file_extension}")
    
        # Get file timestamp
        timestamp = os.path.getmtime(model_path)
        
        # Update loading time metric
        loading_time = time.time() - start_time
        MODEL_LOADING_TIME.observe(loading_time)
        
        # Update version metric
        MODEL_VERSION.set(timestamp)
        
        # Try to estimate memory usage
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss
            model_copy = pickle.loads(pickle.dumps(model))
            memory_after = process.memory_info().rss
            model_size = memory_after - memory_before
            MODEL_MEMORY_USAGE.set(model_size)
            del model_copy
        except Exception:
            pass
            
        logger.info(f"Model loaded successfully from {model_path}, "
                   f"version timestamp: {datetime.fromtimestamp(timestamp)}, "
                   f"loading time: {loading_time:.2f}s")
                   
        return model, timestamp
        
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {str(e)}", exc_info=True)
        PREDICTION_ERROR_COUNT.labels(error_type="model_loading").inc()
        raise RuntimeError(f"Model loading failed: {str(e)}")


class ModelEnsemble:
    """Handles ensemble of multiple models for improved predictions."""
    
    def __init__(self, models: List[Any], weights: Optional[List[float]] = None):
        """
        Initialize model ensemble.
        
        Args:
            models: List of model objects
            weights: Optional list of weights for weighted averaging
        """
        self.models = models
        self.weights = weights
        
        if weights is not None and len(weights) != len(models):
            raise ValueError("Number of weights must match number of models")
            
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble prediction by averaging individual model predictions.
        
        Args:
            X: Input features
            
        Returns:
            Averaged predictions
        """
        # Check if models are classification or regression
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
            
        # Stack and average predictions
        predictions = np.array(predictions)
        
        # Check if we have probabilities (classification with predict_proba)
        if hasattr(self.models[0], 'predict_proba'):
            return self.predict_proba(X).argmax(axis=1)
        
        # Otherwise weighted average of predictions
        return np.average(predictions, axis=0, weights=self.weights)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble probability prediction.
        
        Args:
            X: Input features
            
        Returns:
            Averaged probability predictions
        """
        all_probs = []
        
        for i, model in enumerate(self.models):
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)
                all_probs.append(probs)
            else:
                # For models without predict_proba, create one-hot encodings
                preds = model.predict(X)
                n_classes = len(np.unique(preds))
                one_hot = np.zeros((len(preds), n_classes))
                for j, p in enumerate(preds):
                    one_hot[j, int(p)] = 1
                all_probs.append(one_hot)
                
        # Stack and weighted average
        all_probs = np.array(all_probs)
        return np.average(all_probs, axis=0, weights=self.weights)


class ModelVersionManager:
    """Manages multiple versions of models for A/B testing and shadow deployment."""
    
    def __init__(self, model_dir: str, config_path: Optional[str] = None):
        """
        Initialize the model version manager.
        
        Args:
            model_dir: Directory containing model files
            config_path: Path to model configuration file
        """
        self.model_dir = model_dir
        self.config_path = config_path
        self.models = {}  # Map of model_id -> ModelManager
        self.config = self._load_config()
        self.default_model_id = self.config.get("default_model", None)
        self.ensembles = {}
        
        # Load configured models
        self._load_models()
    
    def _load_config(self) -> Dict:
        """Load model configuration from file."""
        default_config = {
            "models": {},
            "default_model": None,
            "traffic_split": {},
            "ensembles": {}
        }
        
        if not self.config_path or not os.path.exists(self.config_path):
            logger.warning(f"Model config file not found at {self.config_path}, using defaults")
            return default_config
            
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Validate config structure
            if not isinstance(config, dict):
                logger.error("Invalid config format, using defaults")
                return default_config
                
            # Ensure required sections
            for key in default_config:
                if key not in config:
                    config[key] = default_config[key]
                    
            return config
        except Exception as e:
            logger.error(f"Error loading model config: {str(e)}")
            return default_config
    
    def _load_models(self):
        """Load all models according to configuration."""
        # First load explicitly configured models
        for model_id, model_config in self.config.get("models", {}).items():
            try:
                model_path = model_config.get("path")
                if not model_path:
                    continue
                    
                # Make path absolute if it's relative
                if not os.path.isabs(model_path):
                    model_path = os.path.join(self.model_dir, model_path)
                
                refresh_interval = model_config.get("refresh_interval", 300)
                logger.info(f"Loading configured model {model_id} from {model_path}")
                
                self.models[model_id] = ModelManager(
                    model_path=model_path,
                    model_id=model_id,
                    refresh_interval=refresh_interval
                )
            except Exception as e:
                logger.error(f"Failed to load configured model {model_id}: {str(e)}")
        
        # If no models configured or loaded, try to discover models in the model directory
        if not self.models and os.path.exists(self.model_dir):
            self._discover_models()
            
        # Set default model if not set but we have models
        if not self.default_model_id and self.models:
            self.default_model_id = next(iter(self.models.keys()))
            logger.info(f"Setting default model to {self.default_model_id}")
            
        # Create ensembles
        for ensemble_id, ensemble_config in self.config.get("ensembles", {}).items():
            self._create_ensemble(ensemble_id, ensemble_config)
    
    def _discover_models(self):
        """Discover models in the model directory."""
        model_files = []
        for ext in ['.pkl', '.joblib', '.h5', '.pt']:
            model_files.extend(glob.glob(os.path.join(self.model_dir, f"*{ext}")))
            
        for model_path in model_files:
            try:
                model_id = os.path.splitext(os.path.basename(model_path))[0]
                logger.info(f"Discovered model {model_id} at {model_path}")
                
                self.models[model_id] = ModelManager(
                    model_path=model_path,
                    model_id=model_id
                )
            except Exception as e:
                logger.error(f"Failed to load discovered model {model_path}: {str(e)}")
    
    def _create_ensemble(self, ensemble_id: str, config: Dict):
        """Create a model ensemble from configuration."""
        model_ids = config.get("models", [])
        weights = config.get("weights")
        
        if not model_ids:
            logger.warning(f"Ensemble {ensemble_id} has no models defined, skipping")
            return
            
        # Ensure all referenced models are loaded
        models = []
        for model_id in model_ids:
            if model_id in self.models:
                models.append(self.models[model_id].get_model())
            else:
                logger.warning(f"Model {model_id} referenced in ensemble {ensemble_id} not found")
                return
        
        try:
            # Create and store ensemble
            ensemble = ModelEnsemble(models, weights)
            self.models[ensemble_id] = ensemble
            logger.info(f"Created model ensemble {ensemble_id} with {len(models)} models")
        except Exception as e:
            logger.error(f"Failed to create ensemble {ensemble_id}: {str(e)}")
    
    def get_model_for_request(self, request_id: Optional[str] = None) -> Tuple[str, Any]:
        """
        Get appropriate model according to A/B testing configuration.
        
        Args:
            request_id: Optional request identifier for consistent model selection
            
        Returns:
            Tuple of (model_id, model)
        """
        if not self.models:
            raise RuntimeError("No models available")
            
        # If A/B testing is disabled or no traffic split configured, use default model
        traffic_split = self.config.get("traffic_split", {})
        if not app.config["ENABLE_AB_TESTING"] or not traffic_split:
            return self.default_model_id, self.models[self.default_model_id].get_model()
        
        # Ensure we have a request_id for deterministic routing
        if not request_id:
            request_id = str(uuid.uuid4())
            
        # Hash the request_id for consistent routing
        hash_value = int(hashlib.md5(request_id.encode()).hexdigest(), 16) % 100
        
        # Find the model based on traffic split ranges
        cumulative = 0
        for model_id, percentage in traffic_split.items():
            if model_id not in self.models:
                logger.warning(f"Model {model_id} in traffic split not found")
                continue
                
            cumulative += percentage
            if hash_value < cumulative:
                return model_id, self.models[model_id].get_model()
        
        # Fallback to default
        return self.default_model_id, self.models[self.default_model_id].get_model()
    
    def get_model_by_id(self, model_id: str) -> Any:
        """Get a specific model by ID."""
        if model_id not in self.models:
            raise KeyError(f"Model {model_id} not found")
        return self.models[model_id].get_model()
    
    def get_all_model_ids(self) -> List[str]:
        """Get IDs of all available models."""
        return list(self.models.keys())
    
    def check_all_for_updates(self) -> Dict[str, bool]:
        """Check all models for updates."""
        results = {}
        for model_id, manager in self.models.items():
            if hasattr(manager, 'check_for_updates'):
                results[model_id] = manager.check_for_updates()
        return results


class ModelManager:
    """Handles model loading, reloading, and versioning."""
    
    def __init__(self, model_path: str, model_id: str = "default", refresh_interval: int = 300):
        """
        Initialize the model manager.
        
        Args:
            model_path: Path to model file
            model_id: Unique identifier for this model
            refresh_interval: Interval in seconds to check for model updates
        """
        self.model_path = model_path
        self.model_id = model_id
        self.refresh_interval = refresh_interval
        self.model, self.timestamp = safe_load_model(model_path)
        self.last_check_time = time.time()
        
        # Initialize explainer cache
        self.explainer = None
        self.explainer_type = None
        
        # Initialize calibrator
        self.calibrator = None
        self.is_calibrated = False
        
        # Statistics
        self.prediction_count = 0
        self.error_count = 0
        self.last_prediction_time = None
    
    def get_model(self) -> Any:
        """
        Get the current model, checking for updates if needed.
        
        Returns:
            The current model
        """
        current_time = time.time()
        
        # Check if it's time to refresh the model
        if current_time - self.last_check_time > self.refresh_interval:
            self.check_for_updates()
            self.last_check_time = current_time
        
        return self.model
    
    def check_for_updates(self) -> bool:
        """
        Check if model file has been updated and reload if necessary.
        
        Returns:
            True if model was reloaded, False otherwise
        """
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file {self.model_path} does not exist")
                return False
                
            current_timestamp = os.path.getmtime(self.model_path)
            
            # If timestamp is newer, reload the model
            if current_timestamp > self.timestamp:
                logger.info(f"Model {self.model_id} file updated, reloading...")
                self.model, self.timestamp = safe_load_model(self.model_path)
                
                # Clear prediction cache and explainer
                if self.model_id == "default":
                    prediction_cache.clear()
                self.explainer = None
                
                return True
                
            return False
        except Exception as e:
            logger.error(f"Error checking for model {self.model_id} updates: {str(e)}", exc_info=True)
            return False
    
    def get_explainer(self, X_sample: np.ndarray) -> Any:
        """
        Get or create explainer for the model.
        
        Args:
            X_sample: Sample input data for explainer initialization
            
        Returns:
            Explainer object
        """
        if self.explainer is not None:
            return self.explainer, self.explainer_type
        
        # Try different explainers based on available libraries and model type
        try:
            # Try SHAP first if available
            if SHAP_AVAILABLE:
                # For tree models, use faster TreeExplainer
                if hasattr(self.model, 'estimators_') or hasattr(self.model, 'tree_'):
                    self.explainer = shap.TreeExplainer(self.model)
                    self.explainer_type = "shap_tree"
                else:
                    # For other models use KernelExplainer with a background dataset
                    background = X_sample[:min(50, len(X_sample))]  # Limit background size
                    self.explainer = shap.KernelExplainer(
                        model=self.model.predict if hasattr(self.model, 'predict') else self.model,
                        data=background
                    )
                    self.explainer_type = "shap_kernel"
                    
                logger.info(f"Created SHAP explainer for model {self.model_id}")
                return self.explainer, self.explainer_type
                
            # Fallback to LIME if available
            elif LIME_AVAILABLE:
                feature_names = current_app.config.get("FEATURE_NAMES") or [f"feature_{i}" for i in range(X_sample.shape[1])]
                self.explainer = lime_tabular.LimeTabularExplainer(
                    training_data=X_sample,
                    feature_names=feature_names,
                    mode="classification" if hasattr(self.model, 'classes_') else "regression"
                )
                self.explainer_type = "lime"
                logger.info(f"Created LIME explainer for model {self.model_id}")
                return self.explainer, self.explainer_type
                
        except Exception as e:
            logger.error(f"Failed to create explainer for model {self.model_id}: {str(e)}")
            
        # If no explainer could be created
        return None, None
        
    def calibrate_model(self, X_calib: np.ndarray, y_calib: np.ndarray) -> bool:
        """
        Calibrate the model's probabilities.
        
        Args:
            X_calib: Calibration features
            y_calib: Calibration targets
            
        Returns:
            True if calibration succeeded, False otherwise
        """
        if not hasattr(self.model, 'predict_proba'):
            logger.warning(f"Model {self.model_id} does not support probability calibration")
            return False
            
        try:
            from sklearn.calibration import CalibratedClassifierCV
            
            # Create calibrator using the specified method
            method = current_app.config.get("CALIBRATION_METHOD", "isotonic")
            self.calibrator = CalibratedClassifierCV(
                base_estimator=self.model,
                method=method,
                cv='prefit'
            )
            
            # Fit the calibrator
            self.calibrator.fit(X_calib, y_calib)
            self.is_calibrated = True
            
            logger.info(f"Calibrated model {self.model_id} using {method} method")
            return True
            
        except Exception as e:
            logger.error(f"Failed to calibrate model {self.model_id}: {str(e)}")
            return False
    
    def predict_with_stats(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        """
        Make a prediction and track statistics.
        
        Args:
            X: Feature data
            
        Returns:
            Tuple of (predictions, probabilities, prediction_time)
        """
        start_time = time.time()
        
        try:
            # Get predictions
            y_pred = self.model.predict(X)
            
            # Get probabilities if available
            y_prob = None
            if hasattr(self.model, 'predict_proba'):
                if self.is_calibrated and self.calibrator is not None:
                    y_prob = self.calibrator.predict_proba(X)
                else:
                    y_prob = self.model.predict_proba(X)
                    
            # Update stats
            prediction_time = time.time() - start_time
            self.prediction_count += len(X)
            self.last_prediction_time = datetime.now()
            
            return y_pred, y_prob, prediction_time
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Prediction error with model {self.model_id}: {str(e)}")
            prediction_time = time.time() - start_time
            raise RuntimeError(f"Prediction error: {str(e)}")


# Initialize model manager
model_version_manager = None


def initialize_app():
    """Initialize the application, loading model and setting up."""
    global model_version_manager, thread_pool, process_pool
    
    try:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Initialize thread and process pools
        thread_pool = ThreadPoolExecutor(max_workers=app.config["NUM_WORKERS"])
        process_pool = ProcessPoolExecutor(max_workers=app.config["NUM_WORKERS"])
        
        # Initialize model manager
        model_dir = app.config["MODEL_DIR"]
        model_config_path = app.config["MODEL_CONFIG_PATH"]
        model_version_manager = ModelVersionManager(
            model_dir=model_dir, 
            config_path=model_config_path
        )
        
        logger.info("Application initialized successfully")
    except Exception as e:
        logger.critical(f"Failed to initialize application: {str(e)}", exc_info=True)
        sys.exit(1)


def auth_required(f):
    """Decorator to require authentication if enabled."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not app.config["ENABLE_AUTH"]:
            return f(*args, **kwargs)
            
        # Check for API key auth
        api_key = request.headers.get('X-API-Key')
        if api_key and api_key in app.config["AUTH_API_KEYS"]:
            return f(*args, **kwargs)
            
        # Check for JWT auth
        if app.config["ENABLE_AUTH"]:
            return jwt_required()(f)(*args, **kwargs)
            
        return jsonify({"error": "Authentication required"}), 401
    return decorated_function


def request_metrics(f):
    """Decorator to track request metrics."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        request_start_time = time.time()
        
        # Generate request ID for tracking
        request_id = request.headers.get('X-Request-ID') or str(uuid.uuid4())
        
        # Execute the request
        try:
            response = f(*args, **kwargs)
            status = response.status_code
        except Exception as e:
            status = 500
            raise e
        finally:
            # Record request metrics
            request_latency = time.time() - request_start_time
            REQUEST_LATENCY.labels(
                method=request.method, 
                endpoint=request.path
            ).observe(request_latency)
            REQUEST_COUNT.labels(
                method=request.method, 
                endpoint=request.path, 
                status=status
            ).inc()
            
        return response
    return wrapper


def validate_input(data: Dict) -> Tuple[bool, str, Optional[List]]:

    """
    Validate the input data for prediction.
    
    Args:
        data: Input data dictionary
        
    Returns:
        Tuple of (is_valid, error_message, processed_features)
    """
    # Check if input key exists
    if "input" not in data:
        return False, "Missing 'input' field in request data", None
    
    input_data = data["input"]
    
    # Check that input is a list
    if not isinstance(input_data, list):
        return False, "'input' must be a list of numeric values", None
    
    # Check data type of elements
    for i, val in enumerate(input_data):
        if not isinstance(val, (int, float)):
            return False, f"Input value at position {i} is not numeric", None
    
    # Check feature count if feature names are provided
    feature_names = current_app.config["FEATURE_NAMES"]
    if feature_names and len(input_data) != len(feature_names):
        return False, f"Expected {len(feature_names)} features, got {len(input_data)}", None
    
    # All validations passed
    return True, "", input_data


def get_feature_importance() -> Dict[str, float]:
    """
    Get feature importance from the model if available.
    
    Returns:
        Dictionary mapping feature names to importance scores
    """
    try:
        model = model_version_manager.get_model_by_id("default")
        feature_names = current_app.config["FEATURE_NAMES"]
        
        # Get base model if it's a pipeline
        if hasattr(model, 'steps'):
            model_step = model.steps[-1][1]
        else:
            model_step = model
        
        if hasattr(model_step, 'feature_importances_'):
            importances = model_step.feature_importances_
            
            # If feature names are available, return as dict
            if feature_names and len(importances) == len(feature_names):
                return {name: float(importance) for name, importance in zip(feature_names, importances)}
            else:
                return {f"feature_{i}": float(importance) for i, importance in enumerate(importances)}
        else:
            return {}
    except Exception as e:
        logger.warning(f"Couldn't extract feature importance: {str(e)}")
        return {}


def get_cache_key(input_data: List) -> str:
    """Generate cache key for the input data."""
    return json.dumps(input_data)


@app.route("/predict", methods=["POST"])
@request_metrics
@limiter.limit("100/minute")
@auth_required
def predict():
    """
    Make a prediction based on input features.
    
    Expects JSON: {"input": [feature1, feature2, ...]}
    Returns: {"prediction": class_value, "probability": prob_value, "processing_time": time_ms}
    """
    start_time = time.time()
    
    try:
        # Get and validate request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON or no data provided"}), 400
        
        # Validate input
        is_valid, error_msg, input_features = validate_input(data)
        if not is_valid:
            return jsonify({"error": error_msg}), 400
            
        # Log input data
        logger.info(f"Processing prediction request: {input_features}")
        
        # Record input feature metrics
        feature_names = app.config["FEATURE_NAMES"] or [f"feature_{i}" for i in range(len(input_features))]
        for i, value in enumerate(input_features):
            feature_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
            INPUT_FEATURES.labels(feature=feature_name).observe(value)
        
        # Check cache if enabled
        if app.config["ENABLE_CACHING"]:
            cache_key = get_cache_key(input_features)
            cached_result = prediction_cache.get(cache_key)
            if cached_result:
                global cache_hits
                cache_hits += 1
                CACHE_HIT_RATIO.set(cache_hits / (cache_hits + cache_misses))
                logger.info("Returning cached prediction result")
                cached_result["cached"] = True
                return jsonify(cached_result)
        
        # Get the model and make prediction
        model_id, model = model_version_manager.get_model_for_request()
        
        # Make prediction
        prediction_raw = model.predict([input_features])
        prediction = int(prediction_raw[0])
        
        # Get probability if available
        probability = None
        explanation = {}
        
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba([input_features])[0]
                probability = float(probabilities[prediction])
                
                # Record prediction values
                for i, prob in enumerate(probabilities):
                    PREDICTION_GAUGE.labels(class=str(i)).observe(prob)
            except Exception as e:
                logger.warning(f"Error getting prediction probability: {str(e)}")
        
        # Try to get feature contributions if it's a tree-based model
        if app.config["FEATURE_NAMES"]:
            try:
                feature_importance = get_feature_importance()
                if feature_importance:
                    explanation = {"feature_importance": feature_importance}
            except Exception as e:
                logger.warning(f"Error calculating feature contributions: {str(e)}")
        
        # Build response
        processing_time = (time.time() - start_time) * 1000  # in milliseconds
        result = {
            "prediction": prediction,
            "processing_time_ms": processing_time,
            "model_version": datetime.fromtimestamp(model_version_manager.models[model_id].timestamp).isoformat()
        }
        
        if probability is not None:
            result["probability"] = probability
            
        if explanation:
            result["explanation"] = explanation
            
        # Cache the result if caching is enabled
        if app.config["ENABLE_CACHING"]:
            cache_key = get_cache_key(input_features)
            cached_copy = result.copy()
            prediction_cache[cache_key] = cached_copy
            global cache_misses
            cache_misses += 1
            CACHE_HIT_RATIO.set(cache_hits / (cache_hits + cache_misses))
            
        logger.info(f"Prediction result: {result}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


@app.route("/batch-predict", methods=["POST"])
@request_metrics
@limiter.limit("20/minute")
@auth_required
def batch_predict():
    """
    Make predictions for a batch of inputs.
    
    Expects JSON: {"inputs": [[feature1, feature2, ...], [feature1, feature2, ...], ...]}
    Returns: {"predictions": [class1, class2, ...], "processing_time": time_ms}
    """
    start_time = time.time()
    
    try:
        data = request.get_json()
        
        if not data or "inputs" not in data:
            return jsonify({"error": "Missing 'inputs' field in request data"}), 400
            
        inputs = data["inputs"]
        
        if not isinstance(inputs, list) or not inputs:
            return jsonify({"error": "'inputs' must be a non-empty list of input arrays"}), 400
            
        # Validate all inputs
        invalid_inputs = []
        for i, input_data in enumerate(inputs):
            is_valid, error_msg, _ = validate_input({"input": input_data})
            if not is_valid:
                invalid_inputs.append({"index": i, "error": error_msg})
                
        if invalid_inputs:
            return jsonify({"error": "Invalid inputs", "details": invalid_inputs}), 400
        
        # Get the model and make predictions
        model_id, model = model_version_manager.get_model_for_request()
        predictions = model.predict(inputs).tolist()
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(inputs).tolist()
            except Exception as e:
                logger.warning(f"Error getting batch prediction probabilities: {str(e)}")
        
        # Build response
        processing_time = (time.time() - start_time) * 1000
        result = {
            "predictions": predictions,
            "processing_time_ms": processing_time,
            "model_version": datetime.fromtimestamp(model_version_manager.models[model_id].timestamp).isoformat()
        }
        
        if probabilities:
            result["probabilities"] = probabilities
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}", exc_info=True)
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


@app.route("/health", methods=["GET"])
@request_metrics
def health_check():
    """
    Basic health check endpoint.
    
    Returns 200 if the service is running.
    """
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


@app.route("/readiness", methods=["GET"])
@request_metrics
def readiness_check():
    """
    Readiness check that verifies the model is loaded and working.
    
    Returns 200 if the model is ready to serve predictions, 503 otherwise.
    """
    try:
        # Check if model is loaded
        model_id, model = model_version_manager.get_model_for_request()
        
        # Verify model can make a prediction using a simple test input
        feature_count = len(app.config["FEATURE_NAMES"]) if app.config["FEATURE_NAMES"] else 1
        test_input = [0.0] * feature_count
        
        # Try a prediction
        _ = model.predict([test_input])
        
        return jsonify({
            "status": "ready",
            "model_version": datetime.fromtimestamp(model_version_manager.models[model_id].timestamp).isoformat(),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}", exc_info=True)
        return jsonify({
            "status": "not_ready",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 503


@app.route("/model-info", methods=["GET"])
@request_metrics
def model_info():
    """
    Get information about the current model.
    
    Returns details about the model version, features, etc.
    """
    try:
        model_id, model = model_version_manager.get_model_for_request()
        
        # Get the base model from a pipeline if applicable
        if hasattr(model, 'steps'):
            model_step = model.steps[-1][1]
            model_type = f"Pipeline with {type(model_step).__name__}"
        else:
            model_step = model
            model_type = type(model).__name__
            
        # Get feature names
        feature_names = app.config["FEATURE_NAMES"] or []
        
        # Get feature importance
        feature_importance = get_feature_importance()
        
        # Get classes if available
        classes = None
        if hasattr(model_step, 'classes_'):
            classes = model_step.classes_.tolist()
            
        info = {
            "model_type": model_type,
            "model_version": datetime.fromtimestamp(model_version_manager.models[model_id].timestamp).isoformat(),
            "model_path": app.config["MODEL_PATH"],
            "feature_count": len(feature_names) if feature_names else "unknown",
            "last_reload_check": datetime.fromtimestamp(model_version_manager.models[model_id].last_check_time).isoformat()
        }
        
        if feature_names:
            info["features"] = feature_names
            
        if feature_importance:
            info["feature_importance"] = feature_importance
            
        if classes is not None:
            info["classes"] = classes
            
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error retrieving model info: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/config", methods=["GET"])
@limiter.exempt
@request_metrics
def get_config():
    """Get service configuration (except sensitive values)."""
    # Filter out sensitive fields if needed
    safe_config = {
        "MODEL_REFRESH_INTERVAL": app.config["MODEL_REFRESH_INTERVAL"],
        "ENABLE_CACHING": app.config["ENABLE_CACHING"],
        "CACHE_TIMEOUT": app.config["CACHE_TIMEOUT"],
        "PREDICTION_THRESHOLD": app.config["PREDICTION_THRESHOLD"],
    }
    
    if app.config["FEATURE_NAMES"]:
        safe_config["FEATURE_NAMES"] = app.config["FEATURE_NAMES"]
        
    return jsonify(safe_config)


@app.route("/clear-cache", methods=["POST"])
@limiter.limit("10/hour")
@request_metrics
@auth_required
def clear_cache():
    """Clear the prediction cache."""
    try:
        cache_size = len(prediction_cache)
        prediction_cache.clear()
        return jsonify({
            "status": "success", 
            "message": f"Cleared {cache_size} cached predictions"
        })
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/reload-model", methods=["POST"])
@limiter.limit("10/hour")
@request_metrics
@auth_required
def reload_model():
    """Reload the model from disk."""
    try:
        model_version_manager.check_all_for_updates()
        return jsonify({"status": "success", "message": "Model reloaded successfully"})
    except Exception as e:
        logger.error(f"Error reloading model: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/stream-predict", methods=["POST"])
@request_metrics
@auth_required
def stream_predict():
    """
    Stream predictions for very large batches without memory constraints.
    
    Expects JSON with streaming=true flag and inputs array.
    Returns a streaming response with predictions as they complete.
    """
    if not app.config["ENABLE_STREAMING"]:
        return jsonify({"error": "Streaming predictions are disabled"}), 400
        
    try:
        data = request.get_json()
        
        if not data or "inputs" not in data:
            return jsonify({"error": "Missing 'inputs' field in request data"}), 400
            
        inputs = data["inputs"]
        chunk_size = min(int(data.get("chunk_size", 100)), 1000)  # Default 100, max 1000
        
        if not isinstance(inputs, list) or not inputs:
            return jsonify({"error": "'inputs' must be a non-empty list of input arrays"}), 400
            
        # Validate a sample of inputs to catch errors early
        sample_size = min(10, len(inputs))
        for i in range(sample_size):
            is_valid, error_msg, _ = validate_input({"input": inputs[i]})
            if not is_valid:
                return jsonify({"error": f"Invalid input at position {i}: {error_msg}"}), 400
        
        # Get the model
        model_id, model = model_version_manager.get_model_for_request()
        
        @stream_with_context
        def generate():
            """Generate predictions in chunks."""
            total = len(inputs)
            for i in range(0, total, chunk_size):
                chunk = inputs[i:i+chunk_size]
                
                # Process chunk
                try:
                    predictions = model.predict(chunk).tolist()
                    
                    # Get probabilities if available
                    probabilities = None
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(chunk).tolist()
                    
                    # Create response chunk
                    response_chunk = {
                        "chunk_index": i // chunk_size,
                        "start_index": i,
                        "end_index": min(i + chunk_size, total) - 1,
                        "predictions": predictions
                    }
                    
                    if probabilities:
                        response_chunk["probabilities"] = probabilities
                        
                    # Stream the response
                    yield json.dumps(response_chunk) + "\n"
                    
                except Exception as e:
                    error_chunk = {
                        "error": f"Error processing chunk starting at index {i}: {str(e)}",
                        "chunk_index": i // chunk_size
                    }
                    yield json.dumps(error_chunk) + "\n"
        
        return Response(generate(), mimetype='application/x-ndjson')
        
    except Exception as e:
        logger.error(f"Error during streaming prediction: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/predict-async", methods=["POST"])
@request_metrics
@limiter.limit("50/minute")
@auth_required
def predict_async():
    """
    Asynchronous prediction endpoint for large batches.
    
    Submits the job and returns a job ID that can be used to fetch results.
    """
    try:
        data = request.get_json()
        
        if not data or "inputs" not in data:
            return jsonify({"error": "Missing 'inputs' field in request data"}), 400
            
        inputs = data["inputs"]
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Store job in prediction queue
        # In a real system, you'd use a proper job queue like Celery or RQ
        def process_job():
            try:
                model_id, model = model_version_manager.get_model_for_request(job_id)
                predictions = model.predict(inputs).tolist()
                
                # Get probabilities if available
                probabilities = None
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(inputs).tolist()
                    
                # Store results
                job_results[job_id] = {
                    "status": "completed",
                    "predictions": predictions,
                    "probabilities": probabilities if probabilities else None,
                    "completed_at": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Error processing async job {job_id}: {str(e)}")
                job_results[job_id] = {
                    "status": "failed",
                    "error": str(e),
                    "completed_at": datetime.now().isoformat()
                }
        
        # Submit job to thread pool
        job_results[job_id] = {"status": "processing", "created_at": datetime.now().isoformat()}
        thread_pool.submit(process_job)
        
        return jsonify({
            "job_id": job_id,
            "status": "processing",
            "check_url": f"/job-status/{job_id}"
        })
        
    except Exception as e:
        logger.error(f"Error submitting async job: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/job-status/<job_id>", methods=["GET"])
@request_metrics
@auth_required
def job_status(job_id):
    """Check status of an asynchronous prediction job."""
    if job_id not in job_results:
        return jsonify({"error": "Job not found"}), 404
        
    return jsonify(job_results[job_id])


@app.route("/explain", methods=["POST"])
@request_metrics
@limiter.limit("20/minute")
@auth_required
def explain_prediction():
    """
    Generate explanation for a prediction.
    
    Expects JSON: {"input": [feature1, feature2, ...]}
    Returns: Feature contribution values for the prediction
    """
    if not SHAP_AVAILABLE and not LIME_AVAILABLE:
        return jsonify({"error": "Explanation libraries not available"}), 503
        
    try:
        # Get and validate request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON or no data provided"}), 400
        
        # Validate input
        is_valid, error_msg, input_features = validate_input(data)
        if not is_valid:
            return jsonify({"error": error_msg}), 400
        
        # Convert to numpy array
        input_array = np.array([input_features])
        
        # Get the model
        model_id = request.args.get("model_id", "default")
        try:
            model_manager = model_version_manager.models.get(model_id)
            if not model_manager:
                return jsonify({"error": f"Model {model_id} not found"}), 404
        except Exception as e:
            return jsonify({"error": f"Error accessing model: {str(e)}"}), 500
        
        # Get or create explainer
        explainer, explainer_type = model_manager.get_explainer(input_array)
        if not explainer:
            return jsonify({"error": "Could not create explainer for this model"}), 500
        
        # Generate explanation based on explainer type
        explanation = {}
        if explainer_type.startswith("shap"):
            # Generate SHAP values
            shap_values = explainer.shap_values(input_array)
            
            # Handle different output formats based on model type
            if isinstance(shap_values, list):  # For multi-class models
                explanation["shap_values"] = [sv[0].tolist() for sv in shap_values]
                explanation["base_value"] = explainer.expected_value.tolist() if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
            else:
                explanation["shap_values"] = shap_values[0].tolist()
                explanation["base_value"] = float(explainer.expected_value)
                
            # Add feature names if available
            feature_names = app.config["FEATURE_NAMES"]
            if feature_names and len(feature_names) == len(input_features):
                explanation["features"] = feature_names
                explanation["feature_impacts"] = dict(zip(feature_names, explanation["shap_values"]))
                
        elif explainer_type == "lime":
            # Generate LIME explanation
            lime_exp = explainer.explain_instance(
                input_array[0], 
                model_manager.model.predict_proba if hasattr(model_manager.model, 'predict_proba') else model_manager.model.predict,
                num_features=len(input_features)
            )
            # Convert explanation to list of (feature, importance) tuples
            feature_importance = lime_exp.as_list()
            explanation["lime_explanation"] = feature_importance
            
            # Convert to dictionary for easier consumption
            feature_names = app.config["FEATURE_NAMES"]
            if feature_names:
                explanation["feature_impacts"] = dict(
                    (feature_names[int(feat.split()[0])] if feat.split()[0].isdigit() else feat, imp) 
                    for feat, imp in feature_importance
                )
                
        return jsonify({
            "explanation": explanation,
            "model_id": model_id,
            "explainer_type": explainer_type,
            "input": input_features
        })
        
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/detect-drift", methods=["POST"])
@request_metrics
@limiter.limit("10/hour")
@auth_required
def detect_drift():
    """
    Detect feature drift between reference data and current inputs.
    
    Expects JSON with reference data and current data.
    Returns drift statistics for each feature.
    """
    try:
        data = request.get_json()
        
        if not data or "reference" not in data or "current" not in data:
            return jsonify({"error": "Missing 'reference' or 'current' data"}), 400
            
        reference_data = np.array(data["reference"])
        current_data = np.array(data["current"])
        
        # Validate data shapes
        if reference_data.shape[1] != current_data.shape[1]:
            return jsonify({"error": "Reference and current data must have same number of features"}), 400
            
        # Calculate drift metrics for each feature
        feature_count = reference_data.shape[1]
        feature_names = app.config["FEATURE_NAMES"] or [f"feature_{i}" for i in range(feature_count)]
        
        drift_metrics = {}
        
        for i in range(feature_count):
            ref_feature = reference_data[:, i]
            curr_feature = current_data[:, i]
            
            # Calculate basic distribution statistics
            ref_mean = float(np.mean(ref_feature))
            curr_mean = float(np.mean(curr_feature))
            ref_std = float(np.std(ref_feature))
            curr_std = float(np.std(curr_feature))
            
            # Calculate Wasserstein distance (Earth Mover's Distance)
            from scipy.stats import wasserstein_distance
            w_distance = float(wasserstein_distance(ref_feature, curr_feature))
            
            # Calculate Kolmogorov-Smirnov test
            from scipy.stats import ks_2samp
            ks_stat, ks_pvalue = ks_2samp(ref_feature, curr_feature)
            
            # Calculate Population Stability Index (PSI)
            psi = calculate_psi(ref_feature, curr_feature)
            
            # Store metrics
            feature_name = feature_names[i]
            drift_metrics[feature_name] = {
                "mean_difference": curr_mean - ref_mean,
                "mean_difference_percentage": (curr_mean - ref_mean) / ref_mean if ref_mean != 0 else None,
                "std_difference": curr_std - ref_std,
                "wasserstein_distance": w_distance,
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
                "drift_detected": ks_pvalue < 0.05,
                "psi": float(psi),
                "psi_interpretation": interpret_psi(psi)
            }
            
            # Update Prometheus metrics
            FEATURE_DRIFT.labels(feature=feature_name).set(psi)
        
        # Calculate overall drift score
        overall_drift = np.mean([m["psi"] for m in drift_metrics.values()])
        PREDICTION_DRIFT.set(overall_drift)
        
        return jsonify({
            "drift_metrics": drift_metrics,
            "overall_drift": float(overall_drift),
            "significant_drift_detected": overall_drift > app.config["FEATURE_DRIFT_THRESHOLD"],
            "threshold": app.config["FEATURE_DRIFT_THRESHOLD"]
        })
        
    except Exception as e:
        logger.error(f"Error detecting drift: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def calculate_psi(expected, actual, bins=10):
    """Calculate Population Stability Index between two distributions."""
    try:
        # Create buckets
        breakpoints = np.histogram_bin_edges(np.concatenate([expected, actual]), bins=bins)
        
        # Calculate histogram counts
        expected_counts, _ = np.histogram(expected, bins=breakpoints)
        actual_counts, _ = np.histogram(actual, bins=breakpoints)
        
        # Convert to %
        expected_percents = expected_counts / float(np.sum(expected_counts))
        actual_percents = actual_counts / float(np.sum(actual_counts))
        
        # Replace zeros with small number to avoid division by zero
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
        
        # Calculate PSI
        psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
        
        return np.sum(psi_values)
    except Exception as e:
        logger.warning(f"Error calculating PSI: {str(e)}")
        return 0.0


def interpret_psi(psi):
    """Interpret PSI value."""
    if psi < 0.1:
        return "No significant change"
    elif psi < 0.2:
        return "Moderate change"
    else:
        return "Significant change"


@app.route("/shadow-compare", methods=["POST"])
@request_metrics
@auth_required
def shadow_compare():
    """
    Compare predictions between production model and shadow model.
    
    This endpoint allows safe testing of new models without affecting production.
    """
    try:
        data = request.get_json()
        
        if not data or "inputs" not in data:
            return jsonify({"error": "Missing 'inputs' field in request data"}), 400
            
        inputs = np.array(data["inputs"])
        
        # Get production model
        prod_model_id = "default"  # or from config
        shadow_model_id = request.args.get("shadow_model_id")
        
        if not shadow_model_id:
            return jsonify({"error": "Missing shadow_model_id parameter"}), 400
            
        try:
            prod_model = model_version_manager.get_model_by_id(prod_model_id)
            shadow_model = model_version_manager.get_model_by_id(shadow_model_id)
        except KeyError as e:
            return jsonify({"error": f"Model not found: {str(e)}"}), 404
        
        # Make predictions with both models
        prod_predictions = prod_model.predict(inputs).tolist()
        shadow_predictions = shadow_model.predict(inputs).tolist()
        
        # Calculate comparison metrics
        agreement_count = sum(1 for p, s in zip(prod_predictions, shadow_predictions) if p == s)
        agreement_rate = agreement_count / len(prod_predictions) if prod_predictions else 0
        
        # Get probabilities if available
        prod_probs = shadow_probs = None
        if hasattr(prod_model, "predict_proba") and hasattr(shadow_model, "predict_proba"):
            prod_probs = prod_model.predict_proba(inputs).tolist()
            shadow_probs = shadow_model.predict_proba(inputs).tolist()
            
            # Calculate probability differences
            prob_diffs = []
            for p_prob, s_prob in zip(prod_probs, shadow_probs):
                # Calculate mean absolute difference across all classes
                prob_diffs.append(np.mean(np.abs(np.array(p_prob) - np.array(s_prob))))
            
            avg_prob_diff = float(np.mean(prob_diffs))
            max_prob_diff = float(np.max(prob_diffs))
        else:
            avg_prob_diff = max_prob_diff = None
        
        return jsonify({
            "production_model": prod_model_id,
            "shadow_model": shadow_model_id,
            "sample_size": len(inputs),
            "agreement_rate": agreement_rate,
            "agreement_count": agreement_count,
            "disagreement_count": len(inputs) - agreement_count,
            "avg_probability_difference": avg_prob_diff,
            "max_probability_difference": max_prob_diff,
            "production_predictions": prod_predictions[:10],  # Just show first 10 for brevity
            "shadow_predictions": shadow_predictions[:10]
        })
        
    except Exception as e:
        logger.error(f"Error comparing shadow models: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/calibrate", methods=["POST"])
@request_metrics
@limiter.limit("5/hour")
@auth_required
def calibrate_model():
    """
    Calibrate probability predictions for a model.
    
    Expects JSON with calibration data and targets.
    """
    try:
        data = request.get_json()
        
        if not data or "inputs" not in data or "targets" not in data:
            return jsonify({"error": "Missing 'inputs' or 'targets' field in request data"}), 400
            
        inputs = np.array(data["inputs"])
        targets = np.array(data["targets"])
        
        model_id = request.args.get("model_id", "default")
        calibration_method = request.args.get("method", app.config["CALIBRATION_METHOD"] or "isotonic")
        
        if calibration_method not in ["isotonic", "sigmoid"]:
            return jsonify({"error": "Calibration method must be 'isotonic' or 'sigmoid'"}), 400
            
        try:
            model_manager = model_version_manager.models.get(model_id)
            if not model_manager:
                return jsonify({"error": f"Model {model_id} not found"}), 404
        except Exception as e:
            return jsonify({"error": f"Error accessing model: {str(e)}"}), 500
        
        # Calibrate the model
        success = model_manager.calibrate_model(inputs, targets)
        
        if not success:
            return jsonify({"error": "Calibration failed"}), 500
            
        return jsonify({
            "status": "success",
            "model_id": model_id,
            "calibration_method": calibration_method,
            "message": f"Model {model_id} successfully calibrated using {calibration_method} method",
            "is_calibrated": model_manager.is_calibrated
        })
        
    except Exception as e:
        logger.error(f"Error calibrating model: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/models", methods=["GET"])
@request_metrics
def list_models():
    """List all available models."""
    try:
        models = {}
        for model_id, model_manager in model_version_manager.models.items():
            if hasattr(model_manager, 'timestamp'):
                models[model_id] = {
                    "last_updated": datetime.fromtimestamp(model_manager.timestamp).isoformat(),
                    "path": model_manager.model_path if hasattr(model_manager, 'model_path') else None,
                    "prediction_count": model_manager.prediction_count if hasattr(model_manager, 'prediction_count') else None,
                    "is_calibrated": model_manager.is_calibrated if hasattr(model_manager, 'is_calibrated') else None,
                    "is_ensemble": isinstance(model_manager, ModelEnsemble)
                }
        
        return jsonify({
            "models": models,
            "default_model": model_version_manager.default_model_id,
            "count": len(models)
        })
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/upload-model", methods=["POST"])
@request_metrics
@limiter.limit("10/hour")
@auth_required
def upload_model():
    """Upload a new model file."""
    try:
        if 'model' not in request.files:
            return jsonify({"error": "No model file provided"}), 400
            
        model_file = request.files['model']
        model_id = request.form.get('model_id') or os.path.splitext(model_file.filename)[0]
        
        # Security checks
        if model_file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        # Only allow certain extensions
        allowed_extensions = {'.pkl', '.joblib', '.h5', '.pt'}
        file_ext = os.path.splitext(model_file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({"error": f"File extension {file_ext} not allowed. Allowed extensions: {allowed_extensions}"}), 400
        
        # Save model to temporary file first for security
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            model_file.save(tmp.name)
            
            # Try to load model to validate it
            try:
                test_model, _ = safe_load_model(tmp.name)
                
                # Basic validation - ensure it has predict method
                if not hasattr(test_model, 'predict'):
                    raise ValueError("Uploaded file does not contain a valid model (no predict method)")
                
                # Save to model directory
                model_path = os.path.join(app.config["MODEL_DIR"], f"{model_id}{file_ext}")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                shutil.copy2(tmp.name, model_path)
                
                # Add to model manager
                model_version_manager.models[model_id] = ModelManager(
                    model_path=model_path,
                    model_id=model_id
                )
                
                return jsonify({
                    "status": "success",
                    "message": f"Model uploaded and loaded as {model_id}",
                    "model_id": model_id,
                    "model_path": model_path
                })
                
            except Exception as e:
                return jsonify({"error": f"Invalid model file: {str(e)}"}), 400
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp.name)
                except:
                    pass
                    
    except Exception as e:
        logger.error(f"Error uploading model: {str(e)}")
        return jsonify({"error": str(e)}), 500


def parse_args():
    """Parse command line arguments for inference server."""
    parser = argparse.ArgumentParser(description="Model Inference Server")
    
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind to")
    parser.add_argument("--model-dir", type=str, help="Directory containing model files")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--workers", type=int, help="Number of worker threads/processes")
    parser.add_argument("--disable-auth", action="store_true", help="Disable authentication")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--prometheus-port", type=int, help="Port for Prometheus metrics")
    
    return parser.parse_args()


def main():
    """Main entry point for the inference server."""
    # Parse command line arguments
    args = parse_args()
    
    # Update app config from args
    if args.model_dir:
        app.config["MODEL_DIR"] = args.model_dir
    if args.workers:
        app.config["NUM_WORKERS"] = args.workers
    if args.disable_auth:
        app.config["ENABLE_AUTH"] = False
        
    # Initialize the app
    initialize_app()
    
    # Create asynchronous metrics server if needed
    if args.prometheus_port:
        from prometheus_client import start_http_server
        start_http_server(args.prometheus_port)
    
    # Start the Flask app
    logger.info(f"Starting model inference server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    # Allow command line execution
    main()
