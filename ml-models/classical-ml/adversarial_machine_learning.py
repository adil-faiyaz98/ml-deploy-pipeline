import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, Union, List, Any, Tuple, Callable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import foolbox as fb
import art.attacks.evasion as art_attacks
from art.estimators.classification import PyTorchClassifier
import mlflow
import mlflow.pytorch
import joblib
import json

from base_model import MLModel

class AdversarialMachineLearningModel(MLModel):
    """Implementation of adversarial machine learning defense mechanisms."""
    
    def __init__(
        self,
        model_name: str = "adversarial_ml",
        base_model: Optional[nn.Module] = None,
        model_type: str = "classification",
        defense_method: str = "adversarial_training",
        attack_methods: List[str] = ["fgsm", "pgd"],
        epsilon: float = 0.03,
        defense_strength: float = 0.5,
        input_shape: Optional[Tuple[int, ...]] = None,
        num_classes: int = 10,
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize Adversarial Machine Learning model.
        
        Args:
            model_name: Name of the model
            base_model: PyTorch model to defend (if None, a simple CNN will be created)
            model_type: Type of task ('classification' or 'regression')
            defense_method: Defense method ('adversarial_training', 'feature_squeezing', 'input_defense')
            attack_methods: List of attack methods to defend against
            epsilon: Perturbation size for adversarial examples
            defense_strength: Parameter controlling trade-off between robustness and accuracy
            input_shape: Shape of input data (e.g., (3, 32, 32) for CIFAR)
            num_classes: Number of output classes
            tracking_uri: MLflow tracking server URI
            registry_uri: MLflow model registry URI
            experiment_name: MLflow experiment name
            device: Device to use for computations ('cpu' or 'cuda')
        """
        if experiment_name is None:
            experiment_name = "adversarial_ml_models"
        super().__init__(model_name, model_type, tracking_uri, registry_uri, experiment_name)
        
        self.defense_method = defense_method
        self.attack_methods = attack_methods
        self.epsilon = epsilon
        self.defense_strength = defense_strength
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.device = device
        
        # Create model if not provided
        self.base_model = base_model
        if self.base_model is None and input_shape is not None:
            self._create_base_model()
        elif self.base_model is not None:
            self.base_model = self.base_model.to(self.device)
        
        # Track parameters for MLflow
        self.params = {
            "defense_method": defense_method,
            "attack_methods": str(attack_methods),
            "epsilon": epsilon,
            "defense_strength": defense_strength,
            "num_classes": num_classes,
            "device": device
        }
        
        # Initialize attack and defense methods
        self._initialize_attack_methods()
        self._initialize_defense_methods()
        
        self.model = self.base_model  # For consistency with other models
        self.is_fitted = False

    def _create_base_model(self):
        """Create a simple CNN model if none is provided."""
        if len(self.input_shape) == 3:  # Image data (channels, height, width)
            # Simple CNN for image classification
            channels, height, width = self.input_shape
            self.base_model = nn.Sequential(
                nn.Conv2d(channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64 * (height//4) * (width//4), 128),
                nn.ReLU(),
                nn.Linear(128, self.num_classes)
            ).to(self.device)
        else:
            # Simple MLP for tabular data
            input_dim = self.input_shape[0]
            self.base_model = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, self.num_classes)
            ).to(self.device)
    
    def _initialize_attack_methods(self):
        """Initialize attack methods."""
        self.attack_functions = {}
        
        # Set up attacks using Foolbox
        if hasattr(self, 'base_model') and self.base_model is not None:
            # Create preprocessing functions for foolbox
            preprocessing = dict(mean=[0.0] * self.input_shape[0], 
                                std=[1.0] * self.input_shape[0])
            
            try:
                # Create foolbox model
                self.fb_model = fb.PyTorchModel(self.base_model, bounds=(0, 1), 
                                              preprocessing=preprocessing, 
                                              device=self.device)
                
                # Initialize attacks
                for attack in self.attack_methods:
                    if attack.lower() == "fgsm":
                        self.attack_functions[attack] = fb.attacks.FGSM()
                    elif attack.lower() == "pgd":
                        self.attack_functions[attack] = fb.attacks.PGD(steps=10, rel_stepsize=0.25)
                    elif attack.lower() == "deepfool":
                        self.attack_functions[attack] = fb.attacks.L2DeepFoolAttack(steps=50)
                    elif attack.lower() == "cw":
                        self.attack_functions[attack] = fb.attacks.L2CarliniWagnerAttack(steps=1000)
            except Exception as e:
                self.logger.warning(f"Could not initialize Foolbox attacks: {e}")
                
            # Set up attacks using ART
            try:
                # Create ART classifier
                loss_fn = nn.CrossEntropyLoss()
                optimizer = optim.Adam(self.base_model.parameters())
                
                self.art_classifier = PyTorchClassifier(
                    model=self.base_model,
                    loss=loss_fn,
                    optimizer=optimizer,
                    input_shape=self.input_shape,
                    nb_classes=self.num_classes,
                    clip_values=(0, 1),
                    device_type=self.device
                )
                
                # Initialize ART attacks
                for attack in self.attack_methods:
                    if attack.lower() == "fgsm" and "fgsm_art" not in self.attack_functions:
                        self.attack_functions["fgsm_art"] = art_attacks.FastGradientMethod(self.art_classifier, eps=self.epsilon)
                    elif attack.lower() == "pgd" and "pgd_art" not in self.attack_functions:
                        self.attack_functions["pgd_art"] = art_attacks.ProjectedGradientDescent(
                            self.art_classifier, eps=self.epsilon, eps_step=self.epsilon/10, max_iter=10
                        )
                    elif attack.lower() == "boundary" and "boundary" not in self.attack_functions:
                        self.attack_functions["boundary"] = art_attacks.BoundaryAttack(self.art_classifier)
                        
            except Exception as e:
                self.logger.warning(f"Could not initialize ART attacks: {e}")
    
    def _initialize_defense_methods(self):
        """Initialize defense methods."""
        self.defense_functions = {}
        
        if self.defense_method == "adversarial_training":
            # Will be implemented in the fit method
            pass
        elif self.defense_method == "feature_squeezing":
            self.defense_functions["feature_squeezing"] = self._feature_squeezing
        elif self.defense_method == "input_defense":
            self.defense_functions["input_defense"] = self._input_defense
        
    def _feature_squeezing(self, X: torch.Tensor, bit_depth: int = 5) -> torch.Tensor:
        """Implement feature squeezing defense."""
        max_val = torch.max(X)
        X_normalized = X / max_val if max_val > 0 else X
        
        # Reduce bit depth
        max_level = 2**bit_depth - 1
        X_reduced = torch.round(X_normalized * max_level) / max_level
        
        # Scale back
        return X_reduced * max_val
    
    def _input_defense(self, X: torch.Tensor, filter_size: int = 2) -> torch.Tensor:
        """Implement input defense (spatial smoothing)."""
        if len(X.shape) == 4:  # Batch of images
            # Simple average pooling as defense
            return nn.functional.avg_pool2d(X, kernel_size=filter_size, stride=1, padding=filter_size//2)
        return X  # Non-image data is returned as is
    
    def _generate_adversarial_examples(self, X: torch.Tensor, y: torch.Tensor, 
                                      attack_method: str) -> torch.Tensor:
        """Generate adversarial examples using specified attack method."""
        if attack_method not in self.attack_functions:
            self.logger.warning(f"Attack method {attack_method} not found, using original data.")
            return X
        
        try:
            if "art" in attack_method:
                # Using ART for attack
                X_np = X.cpu().numpy()
                adv_examples = self.attack_functions[attack_method].generate(X_np)
                return torch.tensor(adv_examples, device=self.device)
            else:
                # Using Foolbox for attack
                criterion = fb.criteria.Misclassification(y)
                _, adv_examples, success = self.attack_functions[attack_method](self.fb_model, X, criterion, epsilons=self.epsilon)
                return adv_examples
        except Exception as e:
            self.logger.error(f"Error generating adversarial examples: {e}")
            return X

    def fit(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], 
           validation_data: Optional[tuple] = None, log_to_mlflow: bool = True,
           epochs: int = 10, batch_size: int = 32, lr: float = 0.001,
           **kwargs) -> Dict[str, float]:
        """
        Fit model with adversarial training defense.
        
        Args:
            X: Training features
            y: Training labels
            validation_data: Optional tuple of (X_val, y_val) for validation
            log_to_mlflow: Whether to log metrics and model to MLflow
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            **kwargs: Additional parameters
            
        Returns:
            Dict with training metrics
        """
        start_time = time.time()
        self.logger.info(f"Starting adversarial training with {X.shape[0]} samples")
        
        # Convert numpy to torch if needed
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.long, device=self.device)
        
        # Move data to device
        X = X.to(self.device)
        y = y.to(self.device)
        
        # Create data loader
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.base_model.parameters(), lr=lr)
        
        # Training loop
        train_metrics = {}
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            self.base_model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                # Generate adversarial examples if using adversarial training
                if self.defense_method == "adversarial_training":
                    # Randomly select an attack method
                    if self.attack_methods:
                        attack_idx = np.random.randint(0, len(self.attack_methods))
                        attack_method = self.attack_methods[attack_idx]
                        
                        # Mix clean and adversarial examples
                        if np.random.random() < self.defense_strength:
                            adv_batch_X = self._generate_adversarial_examples(batch_X, batch_y, attack_method)
                            
                            # Only use successful adversarial examples
                            with torch.no_grad():
                                outputs = self.base_model(batch_X)
                                adv_outputs = self.base_model(adv_batch_X)
                                
                                # Where predictions changed
                                pred = outputs.argmax(dim=1)
                                adv_pred = adv_outputs.argmax(dim=1)
                                successful = (pred != adv_pred)
                                
                                # Mix original and adversarial examples
                                mixed_batch_X = batch_X.clone()
                                mixed_batch_X[successful] = adv_batch_X[successful]
                                batch_X = mixed_batch_X
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.base_model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward and optimize
                loss.backward()
                optimizer.step()
                
                # Track metrics
                train_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Epoch metrics
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            self.logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            
            # Validation
            if validation_data is not None:
                X_val, y_val = validation_data
                
                # Convert numpy to torch if needed
                if isinstance(X_val, np.ndarray):
                    X_val = torch.tensor(X_val, dtype=torch.float32, device=self.device)
                if isinstance(y_val, np.ndarray):
                    y_val = torch.tensor(y_val, dtype=torch.long, device=self.device)
                
                # Move data to device
                X_val = X_val.to(self.device)
                y_val = y_val.to(self.device)
                
                # Evaluate on clean validation data
                val_metrics = self._calculate_metrics(X_val, y_val, prefix="val_clean_")
                
                # Evaluate on adversarial validation data
                for attack_method in self.attack_methods:
                    adv_X_val = self._generate_adversarial_examples(X_val, y_val, attack_method)
                    adv_metrics = self._calculate_metrics(adv_X_val, y_val, prefix=f"val_adv_{attack_method}_")
                    val_metrics.update(adv_metrics)
                
                # Track best model
                if val_metrics["val_clean_accuracy"] > best_val_acc:
                    best_val_acc = val_metrics["val_clean_accuracy"]
                    # Save best model state
                    self.best_model_state = {k: v.cpu().detach() for k, v in self.base_model.state_dict().items()}
                
                self.logger.info(f"Validation Acc: {val_metrics['val_clean_accuracy']:.4f}")
                
                # Update train metrics
                train_metrics.update(val_metrics)
            
            # Update training metrics
            train_metrics[f"epoch_{epoch+1}_loss"] = train_loss
            train_metrics[f"epoch_{epoch+1}_accuracy"] = train_acc
        
        # Restore best model if validation was performed
        if hasattr(self, 'best_model_state'):
            self.base_model.load_state_dict({k: v.to(self.device) for k, v in self.best_model_state.items()})
        
        # Final metrics
        train_metrics["training_time_seconds"] = time.time() - start_time
        train_metrics["final_train_loss"] = train_loss
        train_metrics["final_train_accuracy"] = train_acc
        
        self.is_fitted = True
        
        # Log to MLflow
        if log_to_mlflow:
            with mlflow.start_run(run_name=f"{self.model_name}_training") as run:
                # Log parameters
                mlflow.log_params(self.params)
                
                # Log metrics
                for metric_name, metric_value in train_metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log model
                mlflow.pytorch.log_model(self.base_model, "model")
                
                # Log robustness analysis
                if validation_data is not None:
                    self._log_robustness_analysis(X_val, y_val, mlflow)
                
                self.logger.info(f"Model and metrics logged to MLflow run: {run.info.run_id}")
        
        return train_metrics
    
    def _calculate_metrics(self, X: torch.Tensor, y: torch.Tensor, prefix: str = "") -> Dict[str, float]:
        """Calculate metrics for model evaluation."""
        self.base_model.eval()
        metrics = {}
        
        with torch.no_grad():
            outputs = self.base_model(X)
            _, predicted = torch.max(outputs, 1)
            
            # Accuracy
            correct = (predicted == y).sum().item()
            total = y.size(0)
            metrics[f"{prefix}accuracy"] = correct / total
            
            # Loss
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, y).item()
            metrics[f"{prefix}loss"] = loss
            
            # Other metrics if classification
            if self.model_type == "classification":
                # Convert to numpy for sklearn metrics
                y_np = y.cpu().numpy()
                predicted_np = predicted.cpu().numpy()
                
                # Calculate metrics for multi-class
                metrics[f"{prefix}top5_accuracy"] = self._calculate_top_k_accuracy(outputs, y, k=5)
        
        return metrics
    
    def _calculate_top_k_accuracy(self, outputs: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
        """Calculate top-k accuracy."""
        # Get top-k predictions
        _, pred = outputs.topk(k, 1, True, True)
        pred = pred.t()  # Transpose to shape (k, batch_size)
        
        # Check if true label is in top-k predictions
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        
        # Count samples with at least one correct prediction in top-k
        correct_k = correct.reshape(-1).float().sum(0, keepdim=True)
        
        return correct_k.item() / targets.size(0)
    
    def _log_robustness_analysis(self, X_val: torch.Tensor, y_val: torch.Tensor, mlflow_client):
        """Log robustness analysis visualizations to MLflow."""
        # Create robustness curve
        epsilons = [0.0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3]
        accuracies = []
        
        # Use FGSM for robustness analysis
        for eps in epsilons:
            # Create temporary attack with current epsilon
            if hasattr(self, 'art_classifier'):
                attack = art_attacks.FastGradientMethod(self.art_classifier, eps=eps)
                adv_X = torch.tensor(attack.generate(X_val.cpu().numpy()), device=self.device)
            else:
                # Fallback to original data if attacks aren't available
                adv_X = X_val
                
            # Calculate accuracy
            metrics = self._calculate_metrics(adv_X, y_val, prefix="")
            accuracies.append(metrics["accuracy"])
        
        # Plot robustness curve
        plt.figure(figsize=(10, 6))
        plt.plot(epsilons, accuracies, marker='o', linestyle='-', linewidth=2)
        plt.xlabel('Perturbation Size (ε)')
        plt.ylabel('Accuracy')
        plt.title('Robustness Analysis - Accuracy vs. Attack Strength')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("robustness_curve.png", dpi=300)
        mlflow_client.log_artifact("robustness_curve.png")
        plt.close()
        
        # Log numerical values
        robustness_data = pd.DataFrame({
            'epsilon': epsilons,
            'accuracy': accuracies
        })
        robustness_data.to_csv("robustness_data.csv", index=False)
        mlflow_client.log_artifact("robustness_data.csv")
        
        # Log example adversarial images if X is image data
        if len(X_val.shape) == 4 and X_val.shape[1] in [1, 3]:
            self._log_adversarial_examples(X_val[:5], y_val[:5], mlflow_client)
    
    def _log_adversarial_examples(self, X: torch.Tensor, y: torch.Tensor, mlflow_client):
        """Log examples of adversarial attacks."""
        if "fgsm" in self.attack_functions:
            attack_method = "fgsm"
        elif "pgd" in self.attack_functions:
            attack_method = "pgd"
        else:
            attack_method = list(self.attack_functions.keys())[0] if self.attack_functions else None
        
        if attack_method is None:
            return
            
        adv_X = self._generate_adversarial_examples(X, y, attack_method)
        
        # Plot original and adversarial examples
        fig, axes = plt.subplots(2, X.shape[0], figsize=(3*X.shape[0], 6))
        
        for i in range(X.shape[0]):
            # Original image
            img = X[i].cpu().permute(1, 2, 0).numpy()
            if img.shape[2] == 1:  # Grayscale
                img = img.squeeze(2)
            axes[0, i].imshow(img, cmap='gray' if img.ndim == 2 else None)
            axes[0, i].set_title(f"Original: {y[i].item()}")
            axes[0, i].axis('off')
            
            # Adversarial image
            adv_img = adv_X[i].cpu().permute(1, 2, 0).numpy()
            if adv_img.shape[2] == 1:  # Grayscale
                adv_img = adv_img.squeeze(2)
            axes[1, i].imshow(adv_img, cmap='gray' if adv_img.ndim == 2 else None)
            
            # Get predicted label for adversarial image
            with torch.no_grad():
                adv_pred = self.base_model(adv_X[i].unsqueeze(0))
                adv_label = adv_pred.argmax(dim=1).item()
            
            axes[1, i].set_title(f"Adversarial: {adv_label}")
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig("adversarial_examples.png", dpi=300)
        mlflow_client.log_artifact("adversarial_examples.png")
        plt.close()
        
        # Calculate and log perturbation statistics
        perturbation = (adv_X - X).cpu().numpy()
        perturbation_stats = {
            "mean_perturbation": float(np.mean(np.abs(perturbation))),
            "max_perturbation": float(np.max(np.abs(perturbation))),
            "l2_norm": float(np.mean(np.sqrt(np.sum(np.square(perturbation), axis=(1, 2, 3)))))
        }
        
        with open("perturbation_stats.json", "w") as f:
            json.dump(perturbation_stats, f)
        mlflow_client.log_artifact("perturbation_stats.json")
    
    def predict(self, X: Union[np.ndarray, torch.Tensor], apply_defense: bool = True) -> np.ndarray:
        """
        Make predictions with the model, optionally applying defenses.
        
        Args:
            X: Input data
            apply_defense: Whether to apply defense to input
            
        Returns:
            Numpy array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Convert numpy to torch if needed
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        # Apply defense if requested
        if apply_defense and self.defense_method in ["feature_squeezing", "input_defense"]:
            X = self.defense_functions[self.defense_method](X)
        
        # Make predictions
        self.base_model.eval()
        with torch.no_grad():
            outputs = self.base_model(X)
            _, predictions = torch.max(outputs, 1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X: Union[np.ndarray, torch.Tensor], apply_defense: bool = True) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input data
            apply_defense: Whether to apply defense to input
            
        Returns:
            Numpy array of class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Convert numpy to torch if needed
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        # Apply defense if requested
        if apply_defense and self.defense_method in ["feature_squeezing", "input_defense"]:
            X = self.defense_functions[self.defense_method](X)
        
        # Make predictions
        self.base_model.eval()
        with torch.no_grad():
            outputs = self.base_model(X)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
        """
        if not path.endswith('.pt'):
            path = f"{path}.pt"
        
        save_dict = {
            'model_state_dict': self.base_model.state_dict(),
            'params': self.params,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes
        }
        
        torch.save(save_dict, path)
        self.logger.info(f"Model saved to {path}")
        
        # Also save model metadata
        metadata_path = f"{path.rsplit('.', 1)[0]}_metadata.json"
        metadata = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "defense_method": self.defense_method,
            "attack_methods": self.attack_methods,
            "epsilon": self.epsilon,
            "defense_strength": self.defense_strength,
            "input_shape": self.input_shape,
            "num_classes": self.num_classes
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        self.logger.info(f"Model metadata saved to {metadata_path}")
    
    @classmethod
    def load(cls, path: str) -> "AdversarialMachineLearningModel":
        """
        Load model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded AdversarialMachineLearningModel instance
        """
        if not path.endswith('.pt'):
            path = f"{path}.pt"
        
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        
        # Load metadata
        metadata_path = f"{path.rsplit('.', 1)[0]}_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create instance
        instance = cls(
            model_name=metadata["model_name"],
            model_type=metadata["model_type"],
            defense_method=metadata["defense_method"],
            attack_methods=metadata["attack_methods"],
            epsilon=metadata["epsilon"],
            defense_strength=metadata["defense_strength"],
            input_shape=metadata["input_shape"],
            num_classes=metadata["num_classes"]
        )
        
        # Load model weights
        instance.base_model.load_state_dict(checkpoint['model_state_dict'])
        instance.params = checkpoint['params']
        instance.is_fitted = True
        
        return instance