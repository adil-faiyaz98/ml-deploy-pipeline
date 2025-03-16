import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, Union, List, Any, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.pytorch
import joblib
import json
from PIL import Image
import io

from base_model import MLModel


class CNNModel(MLModel):
    """Convolutional Neural Network model implementation with PyTorch."""
    
    def __init__(self,
                 model_name: str = "cnn",
                 input_shape: Tuple[int, int, int] = (3, 32, 32),  # Channels, Height, Width
                 num_classes: int = 10,
                 architecture: str = "simple",  # 'simple', 'vgg_like', 'resnet_like'
                 learning_rate: float = 0.001,
                 batch_size: int = 64,
                 device: str = None,
                 tracking_uri: Optional[str] = None,
                 registry_uri: Optional[str] = None,
                 experiment_name: Optional[str] = None,
                 **kwargs):
        """
        Initialize CNN model.
        
        Args:
            model_name: Name of the model
            input_shape: Shape of input data (channels, height, width)
            num_classes: Number of classes for classification
            architecture: Type of CNN architecture
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            device: Device to use ('cpu', 'cuda', or None to auto-detect)
            tracking_uri: MLflow tracking server URI
            registry_uri: MLflow model registry URI
            experiment_name: MLflow experiment name
            **kwargs: Additional parameters for model architecture
        """
        if experiment_name is None:
            experiment_name = "cnn_models"
            
        super().__init__(model_name, "classification", tracking_uri, registry_uri, experiment_name)
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.logger.info(f"Using device: {self.device}")
        
        # Store parameters for MLflow
        self.params = {
            "input_shape": input_shape,
            "num_classes": num_classes,
            "architecture": architecture,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "device": str(self.device),
            **kwargs
        }
        
        # Initialize model architecture
        self._build_model(**kwargs)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialize criterion (loss function)
        self.criterion = nn.CrossEntropyLoss()
        
    def _build_model(self, **kwargs):
        """Build the CNN model based on the specified architecture."""
        if self.architecture == "simple":
            self.model = SimpleCNN(self.input_shape, self.num_classes, **kwargs)
        elif self.architecture == "vgg_like":
            self.model = VGGLikeCNN(self.input_shape, self.num_classes, **kwargs)
        elif self.architecture == "resnet_like":
            self.model = ResNetLikeCNN(self.input_shape, self.num_classes, **kwargs)
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
            
        self.model.to(self.device)
        
    def fit(self, 
            X: Union[np.ndarray, torch.Tensor], 
            y: Union[np.ndarray, torch.Tensor],
            validation_data: Optional[Tuple] = None,
            epochs: int = 10,
            log_to_mlflow: bool = True,
            callbacks: List[Callable] = None,
            data_augmentation: bool = False,
            **kwargs) -> Dict[str, float]:
        """
        Fit CNN model to the data.
        
        Args:
            X: Training features
            y: Training labels
            validation_data: Optional tuple of (X_val, y_val) for validation
            epochs: Number of training epochs
            log_to_mlflow: Whether to log metrics and model to MLflow
            callbacks: List of callback functions to execute during training
            data_augmentation: Whether to use data augmentation
            **kwargs: Additional parameters for training
            
        Returns:
            Dictionary with training and validation metrics
        """
        self.logger.info(f"Training CNN model with {len(X)} samples for {epochs} epochs")
        
        # Convert data to tensors if needed
        X_tensor, y_tensor = self._prepare_data(X, y)
        
        # Prepare validation data
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_tensor, y_val_tensor = self._prepare_data(X_val, y_val)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            val_loader = None
            
        # Create data loaders
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Data augmentation transforms if enabled
        if data_augmentation:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            ])
        else:
            transform = None
        
        # Initialize metrics tracking
        history = {
            'train_loss': [],
            'train_accuracy': []
        }
        
        if val_loader is not None:
            history['val_loss'] = []
            history['val_accuracy'] = []
        
        start_time = time.time()
        
        # Training loop
        for epoch in range(epochs):
            epoch_start = time.time()
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # Apply data augmentation if enabled
                if transform is not None:
                    inputs = self._apply_batch_transform(inputs, transform)
                
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Track metrics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                if batch_idx % 10 == 0:
                    self.logger.debug(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx+1}/{len(train_loader)}, '
                                     f'Loss: {running_loss/(batch_idx+1):.4f}, '
                                     f'Accuracy: {100.*correct/total:.2f}%')
            
            # Compute epoch metrics
            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = correct / total
            
            history['train_loss'].append(epoch_loss)
            history['train_accuracy'].append(epoch_accuracy)
            
            # Validation
            if val_loader is not None:
                val_loss, val_accuracy = self._evaluate_model(val_loader)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                
                self.logger.info(f'Epoch: {epoch+1}/{epochs}, '
                               f'Time: {time.time()-epoch_start:.2f}s, '
                               f'Train Loss: {epoch_loss:.4f}, '
                               f'Train Accuracy: {100.*epoch_accuracy:.2f}%, '
                               f'Val Loss: {val_loss:.4f}, '
                               f'Val Accuracy: {100.*val_accuracy:.2f}%')
            else:
                self.logger.info(f'Epoch: {epoch+1}/{epochs}, '
                               f'Time: {time.time()-epoch_start:.2f}s, '
                               f'Train Loss: {epoch_loss:.4f}, '
                               f'Train Accuracy: {100.*epoch_accuracy:.2f}%')
            
            # Execute callbacks if any
            if callbacks is not None:
                for callback in callbacks:
                    callback(epoch, history)
        
        training_time = time.time() - start_time
        self.logger.info(f"Model training completed in {training_time:.2f} seconds")
        
        # Final metrics
        final_metrics = {
            "training_time": training_time,
            "final_train_loss": history['train_loss'][-1],
            "final_train_accuracy": history['train_accuracy'][-1]
        }
        
        if val_loader is not None:
            final_metrics["final_val_loss"] = history['val_loss'][-1]
            final_metrics["final_val_accuracy"] = history['val_accuracy'][-1]
        
        # Log to MLflow
        if log_to_mlflow:
            with mlflow.start_run(run_name=f"{self.model_name}_training") as run:
                # Log parameters
                mlflow.log_params(self.params)
                
                # Log metrics
                for metric_name, metric_value in final_metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log model
                mlflow.pytorch.log_model(self.model, "model")
                
                # Log training history
                self._log_training_history(history, mlflow)
                
                # Log model summary and visualization
                self._log_model_visualization(mlflow)
                
                self.logger.info(f"Training metrics and model logged to MLflow run: {run.info.run_id}")
        
        return final_metrics
    
    def _prepare_data(self, X, y):
        """Convert data to PyTorch tensors."""
        # Convert inputs to tensor if needed
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X).float()
        elif isinstance(X, torch.Tensor):
            X_tensor = X.float()
        else:
            raise ValueError("X must be a numpy array or PyTorch tensor")
            
        # Convert targets to tensor if needed
        if isinstance(y, np.ndarray):
            y_tensor = torch.from_numpy(y).long()
        elif isinstance(y, torch.Tensor):
            y_tensor = y.long()
        else:
            raise ValueError("y must be a numpy array or PyTorch tensor")
            
        # Check shapes
        if len(X_tensor.shape) == 3:  # Assume single image, add batch dimension
            X_tensor = X_tensor.unsqueeze(0)
        elif len(X_tensor.shape) == 2:  # Assume flat image, need to reshape
            batch_size = X_tensor.shape[0]
            X_tensor = X_tensor.view(batch_size, *self.input_shape)
            
        if len(y_tensor.shape) > 1 and y_tensor.shape[1] > 1:  # One-hot encoded
            y_tensor = y_tensor.argmax(dim=1)
            
        return X_tensor, y_tensor
    
    def _apply_batch_transform(self, batch, transform):
        """Apply data augmentation transforms to a batch of images."""
        # Convert tensor batch to PIL images, apply transform, convert back to tensor
        batch_size = batch.size(0)
        channels, height, width = batch.size()[1:]
        
        transformed_batch = torch.zeros_like(batch)
        for i in range(batch_size):
            img_tensor = batch[i]
            # Convert to PIL Image
            img_pil = transforms.ToPILImage()(img_tensor)
            # Apply transform
            img_transformed = transform(img_pil)
            # Convert back to tensor
            img_tensor = transforms.ToTensor()(img_transformed)
            transformed_batch[i] = img_tensor
            
        return transformed_batch
        
    def _evaluate_model(self, loader):
        """Evaluate the model on a data loader."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return running_loss / len(loader), correct / total
    
    def _log_training_history(self, history, mlflow_client):
        """Log training history plots to MLflow."""
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax[0].plot(history['train_loss'], label='Training')
        if 'val_loss' in history:
            ax[0].plot(history['val_loss'], label='Validation')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].set_title('Training and Validation Loss')
        ax[0].legend()
        
        # Plot accuracy
        ax[1].plot(history['train_accuracy'], label='Training')
        if 'val_accuracy' in history:
            ax[1].plot(history['val_accuracy'], label='Validation')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].set_title('Training and Validation Accuracy')
        ax[1].legend()
        
        plt.tight_layout()
        
        # Save figure to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        plt.close(fig)
        
        buf.seek(0)
        mlflow_client.log_artifact(
            "history_plots.png",
            io.BytesIO(buf.getvalue())
        )
        
    def _log_model_visualization(self, mlflow_client):
        """Log model visualization to MLflow."""
        # Log model architecture summary
        summary = []
        summary.append("Model Architecture:")
        summary.append("-" * 80)
        summary.append(str(self.model))
        summary.append("-" * 80)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        summary.append(f"Total parameters: {total_params:,}")
        summary.append(f"Trainable parameters: {trainable_params:,}")
        summary.append(f"Non-trainable parameters: {total_params - trainable_params:,}")
        
        with open("model_summary.txt", "w") as f:
            f.write("\n".join(summary))
            
        mlflow_client.log_artifact("model_summary.txt")
        
        # Visualize first layer filters if convolutional
        if hasattr(self.model, 'visualize_filters'):
            try:
                filter_img = self.model.visualize_filters()
                plt.figure(figsize=(10, 10))
                plt.imshow(filter_img)
                plt.axis('off')
                plt.title('First Layer Filters')
                plt.tight_layout()
                
                # Save figure to a buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150)
                plt.close()
                
                buf.seek(0)
                mlflow_client.log_artifact(
                    "first_layer_filters.png",
                    io.BytesIO(buf.getvalue())
                )
            except Exception as e:
                self.logger.warning(f"Failed to visualize filters: {str(e)}")
    
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Make predictions with the trained CNN model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class indices
        """
        self.model.eval()
        
        # Convert input to tensor if needed
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X).float()
        elif isinstance(X, torch.Tensor):
            X_tensor = X.float()
        else:
            raise ValueError("X must be a numpy array or PyTorch tensor")
            
        # Check shapes
        if len(X_tensor.shape) == 3:  # Single image
            X_tensor = X_tensor.unsqueeze(0)
        elif len(X_tensor.shape) == 2:  # Flat image
            batch_size = X_tensor.shape[0]
            X_tensor = X_tensor.view(batch_size, *self.input_shape)
            
        # Move to device
        X_tensor = X_tensor.to(self.device)
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = outputs.max(1)
            
        return predicted.cpu().numpy()
    
    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Class probabilities
        """
        self.model.eval()
        
        # Convert input to tensor if needed
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X).float()
        elif isinstance(X, torch.Tensor):
            X_tensor = X.float()
        else:
            raise ValueError("X must be a numpy array or PyTorch tensor")
            
        # Check shapes
        if len(X_tensor.shape) == 3:  # Single image
            X_tensor = X_tensor.unsqueeze(0)
        elif len(X_tensor.shape) == 2:  # Flat image
            batch_size = X_tensor.shape[0]
            X_tensor = X_tensor.view(batch_size, *self.input_shape)
            
        # Move to device
        X_tensor = X_tensor.to(self.device)
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
        return probabilities.cpu().numpy()
    
    def evaluate(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
        """
        Evaluate the model performance.
        
        Args:
            X: Test features
            y: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Prepare data
        X_tensor, y_tensor = self._prepare_data(X, y)
        test_dataset = TensorDataset(X_tensor, y_tensor)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Evaluate
        test_loss, test_accuracy = self._evaluate_model(test_loader)
        
        # Get predictions for additional metrics
        y_pred = self.predict(X)
        y_true = y_tensor.cpu().numpy()
        
        # Calculate metrics
        metrics = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        return metrics
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
        """
        if not path.endswith('.pt'):
            path = f"{path}.pt"
            
        # Save model state dict and metadata
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'params': self.params,
            'architecture': self.architecture,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes
        }
        
        torch.save(model_data, path)
        self.logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "CNNModel":
        """
        Load model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded model instance
        """
        if not path.endswith('.pt'):
            path = f"{path}.pt"
            
        # Load model data
        model_data = torch.load(path, map_location=torch.device('cpu'))
        
        # Create new instance
        instance = cls(
            model_name=model_data.get('params', {}).get('model_name', 'loaded_cnn'),
            input_shape=model_data.get('input_shape'),
            num_classes=model_data.get('num_classes'),
            architecture=model_data.get('architecture', 'simple')
        )
        
        # Load state dict
        instance.model.load_state_dict(model_data['model_state_dict'])
        instance.optimizer.load_state_dict(model_data['optimizer_state_dict'])
        
        return instance
    
    def feature_maps(self, X: Union[np.ndarray, torch.Tensor], layer_idx: int = 0) -> np.ndarray:
        """
        Get feature maps from a specific layer of the CNN.
        
        Args:
            X: Input features
            layer_idx: Index of the layer to extract maps from
            
        Returns:
            Feature maps as numpy array
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Convert input to tensor if needed
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X).float()
        elif isinstance(X, torch.Tensor):
            X_tensor = X.float()
        else:
            raise ValueError("X must be a numpy array or PyTorch tensor")
            
        # Check shapes
        if len(X_tensor.shape) == 3:  # Single image
            X_tensor = X_tensor.unsqueeze(0)
        elif len(X_tensor.shape) == 2:  # Flat image
            batch_size = X_tensor.shape[0]
            X_tensor = X_tensor.view(batch_size, *self.input_shape)
            
        # Move to device
        X_tensor = X_tensor.to(self.device)
        
        # Get feature maps
        feature_maps = self.model.get_feature_maps(X_tensor, layer_idx)
        
        return feature_maps.cpu().numpy()


class SimpleCNN(nn.Module):
    """Simple CNN architecture."""
    
    def __init__(self, input_shape, num_classes, **kwargs):
        super(SimpleCNN, self).__init__()
        channels, height, width = input_shape
        
        # First convolutional block
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # Calculate size after convolutions and pooling
        self.feature_size = height // 8 * width // 8 * 128
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Store feature maps for visualization
        self.feature_maps = []
        
    def forward(self, x):
        self.feature_maps = []
        
        # First block
        x = self.conv1(x)
        self.feature_maps.append(x.detach())
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        
        # Second block
        x = self.conv2(x)
        self.feature_maps.append(x.detach())
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        
        # Third block
        x = self.conv3(x)
        self.feature_maps.append(x.detach())
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_feature_maps(self, x, layer_idx=0):
        """Get feature maps for a specific layer."""
        # Reset feature maps
        self.feature_maps = []
        
        # Forward pass to populate feature maps
        _ = self.forward(x)
        
        if layer_idx < len(self.feature_maps):
            return self.feature_maps[layer_idx]
        else:
            raise ValueError(f"Layer index {layer_idx} out of range")
    
    def visualize_filters(self):
        """Visualize filters from the first convolutional layer."""
        # Get weights from first conv layer
        weights = self.conv1.weight.data.cpu().numpy()
        
        # Number of filters
        n_filters = weights.shape[0]
        
        # Create a grid to display filters
        grid_size = int(np.ceil(np.sqrt(n_filters)))
        
        # Create figure
        fig = plt.figure(figsize=(10, 10))
        
        # Plot each filter
        for i in range(n_filters):
            plt.subplot(grid_size, grid_size, i+1)
            # For RGB inputs
            if weights.shape[1] == 3:
                # Normalize filter for display
                filt = weights[i].transpose(1, 2, 0)
                filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-10)
                plt.imshow(filt)
            # For grayscale inputs
            else:
                plt.imshow(weights[i, 0], cmap='gray')
            plt.axis('off')
        
        plt.tight_layout()
        
        # Convert figure to image
        fig.canvas.draw()
        filter_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        filter_img = filter_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return filter_img


class VGGLikeCNN(nn.Module):
    """VGG-like CNN architecture."""
    
    def __init__(self, input_shape, num_classes, **kwargs):
        super(VGGLikeCNN, self).__init__()
        channels, height, width = input_shape
        
        # VGG-like architecture
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Calculate size after convolutions and pooling
        self.feature_size = height // 8 * width // 8 * 256
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )
        
        # Store feature maps for visualization
        self.feature_maps = []
        
    def forward(self, x):
        self.feature_maps = []
        
        # Extract features through convolutional layers
        for i, layer in enumerate(self.features):
            x = layer(x)
            # Store feature maps after ReLU activations
            if isinstance(layer, nn.ReLU) and i % 3 == 2:  # Store after each block
                self.feature_maps.append(x.detach())
                
        # Flatten the features
        x = x.view(x.size(0), -1)
        
        # Pass through classifier
        x = self.classifier(x)
        
        return x
    
    def get_feature_maps(self, x, layer_idx=0):
        """Get feature maps for a specific layer."""
        # Reset feature maps
        self.feature_maps = []
        
        # Forward pass to populate feature maps
        _ = self.forward(x)
        
        if layer_idx < len(self.feature_maps):
            return self.feature_maps[layer_idx]
        else:
            raise ValueError(f"Layer index {layer_idx} out of range")
    
    def visualize_filters(self):
        """Visualize filters from the first convolutional layer."""
        # Get weights from first conv layer (first layer in features)
        first_conv = None
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                first_conv = layer
                break
                
        weights = first_conv.weight.data.cpu().numpy()
        
        # Number of filters
        n_filters = weights.shape[0]
        
        # Create a grid to display filters
        grid_size = int(np.ceil(np.sqrt(n_filters)))
        
        # Create figure
        fig = plt.figure(figsize=(10, 10))
        
        # Plot each filter
        for i in range(n_filters):
            plt.subplot(grid_size, grid_size, i+1)
            # For RGB inputs
            if weights.shape[1] == 3:
                # Normalize filter for display
                filt = weights[i].transpose(1, 2, 0)
                filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-10)
                plt.imshow(filt)
            # For grayscale inputs
            else:
                plt.imshow(weights[i, 0], cmap='gray')
            plt.axis('off')
        
        plt.tight_layout()
        
        # Convert figure to image
        fig.canvas.draw()
        filter_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        filter_img = filter_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return filter_img


class ResidualBlock(nn.Module):
    """Basic residual block for ResNet."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out


class ResNetLikeCNN(nn.Module):
    """ResNet-like CNN architecture."""
    
    def __init__(self, input_shape, num_classes, **kwargs):
        super(ResNetLikeCNN, self).__init__()
        
        channels, height, width = input_shape
        
        # Initial convolutional layer
        self.initial = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Calculate size after convolutions and pooling
        # Initial conv and maxpool reduce dimensions by factor of 4
        # layer2 and layer3 each reduce dimensions by factor of 2
        final_h = height // 16
        final_w = width // 16
        self.feature_size = final_h * final_w * 256
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
        # Store feature maps for visualization
        self.feature_maps = []
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        # First block might have stride > 1
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # Remaining blocks have stride = 1
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        self.feature_maps = []
        
        # Initial conv
        x = self.initial(x)
        self.feature_maps.append(x.detach())
        
        # Residual layers
        x = self.layer1(x)
        self.feature_maps.append(x.detach())
        
        x = self.layer2(x)
        self.feature_maps.append(x.detach())
        
        x = self.layer3(x)
        self.feature_maps.append(x.detach())
        
        # Global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # FC layer
        x = self.fc(x)
        
        return x
    
    def get_feature_maps(self, x, layer_idx=0):
        """Get feature maps for a specific layer."""
        # Reset feature maps
        self.feature_maps = []
        
        # Forward pass to populate feature maps
        _ = self.forward(x)
        
        if layer_idx < len(self.feature_maps):
            return self.feature_maps[layer_idx]
        else:
            raise ValueError(f"Layer index {layer_idx} out of range")
    
    def visualize_filters(self):
        """Visualize filters from the first convolutional layer."""
        # Get weights from first conv layer
        weights = self.initial[0].weight.data.cpu().numpy()
        
        # Number of filters
        n_filters = weights.shape[0]
        
        # Create a grid to display filters
        grid_size = int(np.ceil(np.sqrt(n_filters)))
        
        # Create figure
        fig = plt.figure(figsize=(12, 12))
        
        # Plot each filter
        for i in range(min(64, n_filters)):  # Show up to 64 filters
            plt.subplot(grid_size, grid_size, i+1)
            # For RGB inputs
            if weights.shape[1] == 3:
                # Normalize filter for display
                filt = weights[i].transpose(1, 2, 0)
                filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-10)
                plt.imshow(filt)
            # For grayscale inputs
            else:
                plt.imshow(weights[i, 0], cmap='gray')
            plt.axis('off')
        
        plt.tight_layout()
        
        # Convert figure to image
        fig.canvas.draw()
        filter_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        filter_img = filter_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return filter_img