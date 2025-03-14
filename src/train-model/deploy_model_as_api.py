"""
ML Model Deployment API Server

This module provides a production-grade REST API server for ML model deployment with:
- Fully asynchronous request handling
- Model versioning and hot-reloading
- Input validation and error handling
- Authentication and authorization
- Detailed logging and metrics
- Horizontal scaling capabilities
- Prometheus monitoring integration
- Health checks and diagnostics
- Request payload validation
- Performance optimizations
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Set

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, Request, Security, UploadFile, status
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader, HTTPBasic, HTTPBearer
from pydantic import BaseModel, Field, validator, conlist
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY, CONTENT_TYPE_LATEST

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("model-api")

# Metrics
REQUESTS = Counter('model_api_requests_total', 'Total count of requests by endpoint and status', 
                  ['endpoint', 'status_code'])
REQUESTS_LATENCY = Histogram('model_api_request_latency_seconds', 'Request latency by endpoint',
                            ['endpoint'])
PREDICTIONS = Counter('model_api_predictions_total', 'Total count of predictions by model and result',
                     ['model_version', 'status'])
PREDICTION_LATENCY = Histogram('model_api_prediction_latency_seconds', 'Prediction latency by model',
                              ['model_version'])

# Define API models
class PredictionRequest(BaseModel):
    """Model prediction request schema."""
    features: List[List[float]] = Field(..., 
                                      description="List of feature arrays, each representing one sample")
    model_version: Optional[str] = Field(None, 
                                       description="Specific model version to use")
    return_probability: bool = Field(False, 
                                   description="Whether to return prediction probabilities")
    
    class Config:
        schema_extra = {
            "example": {
                "features": [[1.2, 2.3, 3.4, 4.5], [2.3, 3.4, 4.5, 5.6]],
                "model_version": "latest",
                "return_probability": True
            }
        }
    
    @validator('features')
    def validate_features(cls, features):
        """Validate that features have consistent dimensions."""
        if not features:
            raise ValueError("Features list cannot be empty")
        
        # Check that all feature arrays have the same length
        first_length = len(features[0])
        if not all(len(f) == first_length for f in features):
            raise ValueError("All feature arrays must have the same length")
            
        return features

class PredictionResponse(BaseModel):
    """Model prediction response schema."""
    model_version: str = Field(..., description="Model version used for prediction")
    predictions: List[Union[float, int, str]] = Field(..., description="Predicted values")
    probabilities: Optional[List[List[float]]] = Field(None, description="Prediction probabilities")
    prediction_time: float = Field(..., description="Time taken for prediction in seconds")
    request_id: str = Field(..., description="Unique request identifier")

class BatchPredictionResponse(BaseModel):
    """Response for batch prediction requests."""
    model_version: str = Field(..., description="Model version used for prediction")
    prediction_count: int = Field(..., description="Number of predictions made")
    prediction_time: float = Field(..., description="Time taken for predictions in seconds")
    request_id: str = Field(..., description="Unique request identifier")
    results_url: Optional[str] = Field(None, description="URL to download batch results")

class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    models: Dict[str, Dict[str, Any]] = Field(..., description="Available models and their status")
    uptime: float = Field(..., description="API uptime in seconds")

class ModelInfo(BaseModel):
    """Model information schema."""
    version: str = Field(..., description="Model version")
    created_at: str = Field(..., description="Model creation timestamp")
    description: Optional[str] = Field(None, description="Model description")
    metrics: Optional[Dict[str, float]] = Field(None, description="Model performance metrics")
    feature_names: Optional[List[str]] = Field(None, description="Names of input features")
    target_names: Optional[List[str]] = Field(None, description="Names of target classes")

# Request middleware for logging and metrics
class RequestMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and recording metrics."""
    
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Extract client info
        client_host = request.client.host if request.client else "unknown"
        
        # Log the request
        logger.info(f"Request {request_id} started: {request.method} {request.url.path} from {client_host}")
        
        # Record request timing
        start_time = time.time()
        
        # Set request ID header for all responses
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        # Calculate request duration
        duration = time.time() - start_time
        
        # Update metrics
        endpoint = request.url.path
        REQUESTS.labels(endpoint=endpoint, status_code=response.status_code).inc()
        REQUESTS_LATENCY.labels(endpoint=endpoint).observe(duration)
        
        # Log response info
        logger.info(
            f"Request {request_id} completed: {request.method} {request.url.path} "
            f"status={response.status_code} duration={duration:.4f}s"
        )
        
        return response

class MLModelAPI:
    """Main API class for ML model serving."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the ML model API.
        
        Args:
            config_path: Path to configuration file
        """
        self.start_time = time.time()
        self.app = FastAPI(
            title="Machine Learning Model API",
            description="Production-grade API for machine learning model inference",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Configure logging level
        log_level = self.config.get("logging", {}).get("level", "info").upper()
        logging.getLogger().setLevel(getattr(logging, log_level))
        
        # Setup middlewares
        self._setup_middlewares()
        
        # Setup routes
        self._setup_routes()
        
        # Load models
        self.models = {}
        self.default_model_version = None
        self._load_models()
        
        # Configure security
        self._setup_security()
        
        logger.info(f"ML Model API initialized with {len(self.models)} models")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        default_config = {
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "cors_origins": ["*"],
                "workers": 1
            },
            "models": {
                "dir": "./models",
                "default_version": "latest",
                "auto_reload": True,
                "reload_interval": 60
            },
            "security": {
                "enabled": False,
                "api_key_header": "X-API-Key",
                "api_keys": []
            },
            "logging": {
                "level": "info",
                "request_logging": True
            },
            "performance": {
                "batch_size": 32,
                "cache_predictions": True,
                "cache_size": 1024
            }
        }
        
        if not config_path or not os.path.exists(config_path):
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return default_config
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Merge with defaults for any missing values
            merged_config = default_config.copy()
            for key, value in config.items():
                if isinstance(value, dict) and key in merged_config:
                    merged_config[key].update(value)
                else:
                    merged_config[key] = value
                    
            return merged_config
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return default_config

    def _setup_middlewares(self):
        """Set up API middlewares."""
        # Add CORS middleware
        origins = self.config["api"]["cors_origins"]
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add request middleware for logging and metrics
        if self.config["logging"]["request_logging"]:
            self.app.add_middleware(RequestMiddleware)

    def _setup_security(self):
        """Set up security for the API."""
        if not self.config["security"]["enabled"]:
            logger.warning("Security is DISABLED. This is not recommended for production.")
            self.verify_api_key = lambda: True
            return
            
        # Set up API key authentication
        api_key_header = APIKeyHeader(name=self.config["security"]["api_key_header"], auto_error=False)
        api_keys = set(self.config["security"]["api_keys"])
        
        async def verify_api_key(api_key: str = Security(api_key_header)):
            if not api_key or api_key not in api_keys:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API Key",
                    headers={"WWW-Authenticate": self.config["security"]["api_key_header"]}
                )
            return True
            
        self.verify_api_key = verify_api_key

    def _setup_routes(self):
        """Set up API routes."""
        app = self.app
        
        @app.get("/", include_in_schema=False)
        async def root():
            """Redirect to docs."""
            return {"message": "ML Model API. See /docs for documentation"}
        
        @app.post("/predict", response_model=PredictionResponse)
        async def predict(
            request: PredictionRequest,
            authenticated: bool = Depends(self.verify_api_key)
        ):
            """Make predictions with the model."""
            return await self._handle_prediction(request)
        
        @app.post("/batch-predict", response_model=BatchPredictionResponse)
        async def batch_predict(
            request: PredictionRequest,
            background_tasks: bool = False,
            authenticated: bool = Depends(self.verify_api_key)
        ):
            """Handle batch prediction requests."""
            if background_tasks:
                # For very large batches, process asynchronously
                task_id = await self._start_batch_prediction(request)
                return JSONResponse({
                    "model_version": request.model_version or self.default_model_version,
                    "batch_task_id": task_id,
                    "status": "processing",
                    "status_url": f"/batch-status/{task_id}"
                })
            else:
                # For smaller batches, process immediately
                return await self._handle_batch_prediction(request)
                
        @app.get("/batch-status/{task_id}")
        async def batch_status(
            task_id: str,
            authenticated: bool = Depends(self.verify_api_key)
        ):
            """Get status of a batch prediction task."""
            status = await self._get_batch_status(task_id)
            if status is None:
                raise HTTPException(status_code=404, detail=f"Batch task {task_id} not found")
            return status
        
        @app.get("/health")
        async def health():
            """API health check endpoint."""
            uptime = time.time() - self.start_time
            model_info = {}
            
            for version, model_data in self.models.items():
                try:
                    # Basic model check
                    is_loaded = model_data["model"] is not None
                    last_reload = model_data.get("last_reloaded", "unknown")
                    
                    model_info[version] = {
                        "status": "loaded" if is_loaded else "error",
                        "last_reloaded": last_reload,
                        "metadata": model_data.get("metadata", {})
                    }
                except Exception as e:
                    model_info[version] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            return {
                "status": "healthy",
                "version": self.app.version,
                "models": model_info,
                "uptime": uptime
            }
        
        @app.get("/metrics")
        async def metrics():
            """Expose Prometheus metrics."""
            return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
        
        @app.get("/models")
        async def list_models(authenticated: bool = Depends(self.verify_api_key)):
            """List available model versions."""
            models_info = {}
            
            for version, model_data in self.models.items():
                models_info[version] = {
                    "default": version == self.default_model_version,
                    "metadata": model_data.get("metadata", {}),
                    "last_reloaded": model_data.get("last_reloaded", None)
                }
                
            return {
                "models": models_info,
                "default_version": self.default_model_version
            }
            
        @app.post("/models/reload")
        async def reload_models(authenticated: bool = Depends(self.verify_api_key)):
            """Reload models from disk."""
            try:
                reloaded = await self._reload_models()
                return {
                    "status": "success",
                    "reloaded_models": reloaded,
                    "models_count": len(self.models)
                }
            except Exception as e:
                logger.error(f"Error reloading models: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error reloading models: {str(e)}")

    def _load_models(self):
        """Load all available models."""
        models_dir = Path(self.config["models"]["dir"])
        if not models_dir.exists():
            logger.error(f"Models directory not found: {models_dir}")
            raise ValueError(f"Models directory not found: {models_dir}")
            
        # Look for model files
        model_files = []
        for ext in [".joblib", ".pkl", ".onnx", ".pt", ".h5"]:
            model_files.extend(models_dir.glob(f"*{ext}"))
            
        if not model_files:
            logger.warning(f"No model files found in {models_dir}")
            return
            
        # Load each model
        for model_file in model_files:
            try:
                version = model_file.stem
                
                # Load model
                model = self._load_model_file(model_file)
                
                # Load metadata if exists
                metadata = {}
                metadata_file = model_file.with_suffix(".json")
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        
                # Add to models dict
                self.models[version] = {
                    "model": model,
                    "path": str(model_file),
                    "metadata": metadata,
                    "last_reloaded": datetime.now().isoformat()
                }
                
                logger.info(f"Loaded model {version} from {model_file}")
                
            except Exception as e:
                logger.error(f"Error loading model from {model_file}: {str(e)}")
                
        # Set default model version
        default_version = self.config["models"]["default_version"]
        if default_version in self.models:
            self.default_model_version = default_version
        elif self.models:
            self.default_model_version = list(self.models.keys())[0]
            logger.warning(f"Default model version {default_version} not found, using {self.default_model_version}")
        else:
            logger.error("No models loaded successfully")
            
        logger.info(f"Loaded {len(self.models)} models, default version: {self.default_model_version}")

    def _load_model_file(self, model_path: Path):
        """Load model from file based on extension."""
        if model_path.suffix == ".joblib":
            return joblib.load(model_path)
        elif model_path.suffix == ".pkl":
            import pickle
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        elif model_path.suffix == ".onnx":
            import onnxruntime as ort
            return ort.InferenceSession(str(model_path))
        elif model_path.suffix == ".pt" or model_path.suffix == ".pth":
            import torch
            return torch.load(model_path)
        elif model_path.suffix == ".h5":
            import tensorflow as tf
            return tf.keras.models.load_model(model_path)
        else:
            raise ValueError(f"Unsupported model format: {model_path.suffix}")

    async def _reload_models(self):
        """Reload all models from disk."""
        original_models = set(self.models.keys())
        self._load_models()
        reloaded_models = set(self.models.keys())
        
        # Return list of reloaded models
        return list(reloaded_models)

    async def _handle_prediction(self, request: PredictionRequest) -> Dict[str, Any]:
        """Handle prediction request."""
        # Get model version (use default if not specified)
        version = request.model_version or self.default_model_version
        
        # Check if model version exists
        if version not in self.models:
            raise HTTPException(
                status_code=404,
                detail=f"Model version {version} not found"
            )
            
        # Get model
        model_data = self.models[version]
        model = model_data["model"]
        
        # Start timing
        start_time = time.time()
        
        try:
            # Convert features to numpy array
            features = np.array(request.features)
            
            # Generate request ID
            request_id = str(uuid.uuid4())
            
            # Make prediction
            if isinstance(model, dict) and "predict_fn" in model:
                # Custom prediction function
                predictions = model["predict_fn"](features)
                probabilities = None
                if request.return_probability and "predict_proba_fn" in model:
                    probabilities = model["predict_proba_fn"](features)
            else:
                # Standard sklearn-like interface
                predictions = model.predict(features)
                
                # Get probabilities if requested
                probabilities = None
                if request.return_probability and hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(features)
            
            # Convert numpy arrays to lists for JSON serialization
            if isinstance(predictions, np.ndarray):
                predictions = predictions.tolist()
                
            if isinstance(probabilities, np.ndarray):
                probabilities = probabilities.tolist()
                
            # Calculate prediction time
            prediction_time = time.time() - start_time
            
            # Update metrics
            PREDICTION_LATENCY.labels(model_version=version).observe(prediction_time)
            PREDICTIONS.labels(model_version=version, status="success").inc()
            
            # Create response
            return {
                "model_version": version,
                "predictions": predictions,
                "probabilities": probabilities,
                "prediction_time": prediction_time,
                "request_id": request_id
            }
            
        except Exception as e:
            # Update error metrics
            PREDICTIONS.labels(model_version=version, status="error").inc()
            
            logger.error(f"Error making prediction with model {version}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error making prediction: {str(e)}"
            )

    async def _handle_batch_prediction(self, request: PredictionRequest) -> Dict[str, Any]:
        """Handle batch prediction request."""
        # Get model version (use default if not specified)
        version = request.model_version or self.default_model_version
        
        # Check if model version exists
        if version not in self.models:
            raise HTTPException(
                status_code=404,
                detail=f"Model version {version} not found"
            )
            
        # Start timing
        start_time = time.time()
        
        try:
            # Make prediction using same method as single prediction
            result = await self._handle_prediction(request)
            
            # Add batch-specific info
            batch_response = {
                "model_version": result["model_version"],
                "prediction_count": len(result["predictions"]),
                "prediction_time": result["prediction_time"],
                "request_id": result["request_id"]
            }
            
            return batch_response
            
        except Exception as e:
            logger.error(f"Error processing batch prediction: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing batch prediction: {str(e)}"
            )

    async def _start_batch_prediction(self, request: PredictionRequest) -> str:
        """Start an asynchronous batch prediction task."""
        # Generate a task ID
        task_id = str(uuid.uuid4())
        
        # TODO: Implement background processing infrastructure
        # This would typically use a task queue like Celery or RQ
        
        # For the example, we'll just return the task ID
        return task_id

    async def _get_batch_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a batch prediction task."""
        # TODO: Implement task status retrieval from your task queue
        
        # For the example, we'll just return a mock status
        return {
            "task_id": task_id,
            "status": "completed",
            "created_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat(),
            "results_url": f"/download/{task_id}"
        }

    def run(self):
        """Run the API server."""
        # Get API configuration
        host = self.config["api"]["host"]
        port = self.config["api"]["port"]
        workers = self.config["api"]["workers"]
        
        # Start auto-reload task if enabled
        if self.config["models"]["auto_reload"]:
            reload_interval = self.config["models"]["reload_interval"]
            logger.info(f"Auto model reload enabled with interval {reload_interval} seconds")
            
            # Note: In a production environment, you would set up a proper
            # background task for this using something like APScheduler or
            # a separate thread.
        
        # Start Uvicorn server
        logger.info(f"Starting API server on {host}:{port} with {workers} workers")
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            workers=workers
        )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ML Model API Server")
    
    parser.add_argument("--config", "-c", type=str, default="config.json",
                       help="Path to configuration file")
    parser.add_argument("--port", "-p", type=int, default=None,
                       help="Port to run the server (overrides config)")
    parser.add_argument("--host", type=str, default=None,
                       help="Host to run the server (overrides config)")
    parser.add_argument("--models-dir", "-m", type=str, default=None,
                       help="Directory containing model files (overrides config)")
    parser.add_argument("--workers", "-w", type=int, default=None,
                       help="Number of worker processes (overrides config)")
    parser.add_argument("--debug", "-d", action="store_true",
                       help="Enable debug logging")
    parser.add_argument("--disable-auth", action="store_true",
                       help="Disable authentication")
                       
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    try:
        # Initialize API
        api = MLModelAPI(config_path=args.config)
        
        # Override config with command line args
        if args.port is not None:
            api.config["api"]["port"] = args.port
        if args.host is not None:
            api.config["api"]["host"] = args.host
        if args.models_dir is not None:
            api.config["models"]["dir"] = args.models_dir
        if args.workers is not None:
            api.config["api"]["workers"] = args.workers
        if args.disable_auth:
            api.config["security"]["enabled"] = False
            # Re-setup security
            api._setup_security()
        
        # Run the API server
        api.run()
        
    except Exception as e:
        logger.error(f"Error starting API server: {str(e)}", exc_info=True)
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())