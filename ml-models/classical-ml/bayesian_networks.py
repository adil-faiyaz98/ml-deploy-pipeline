import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Optional, Union, List, Any, Tuple
from pgmpy.models import BayesianNetwork, BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination
import joblib
import json
import mlflow
import mlflow.pyfunc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from base_model import MLModel

class CustomBayesianNetworkWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper for Bayesian Network to use with MLflow."""

    def __init__(self, model, inference_engine=None):
        self.model = model
        if inference_engine is None and model is not None:
            self.inference_engine = VariableElimination(model)
        else:
            self.inference_engine = inference_engine

    def predict(self, context, model_input):
        """Predict method for MLflow."""
        if isinstance(model_input, pd.DataFrame):
            return self._predict_dataframe(model_input)
        else:
            raise ValueError("Input must be a pandas DataFrame")

    def _predict_dataframe(self, df):
        """Make predictions for a dataframe."""
        results = []
        for _, row in df.iterrows():
            evidence = row.to_dict()
            # Remove any NaN values from evidence
            evidence = {k: v for k, v in evidence.items() if pd.notna(v)}
            predictions = {}
            for node in self.model.nodes():
                if node not in evidence:
                    query_result = self.inference_engine.query(variables=[node], evidence=evidence)
                    predictions[node] = query_result.values
            results.append(predictions)
        return results


class BayesianNetworkModel(MLModel):
    """Bayesian Network model implementation with MLflow integration."""

    def __init__(
        self,
        model_name: str = "bayesian_network",
        edges: Optional[List[Tuple[str, str]]] = None,
        estimator_type: str = "maximum_likelihood",
        prior_type: str = "BDeu",
        equivalent_sample_size: float = 10.0,
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ):
        """
        Initialize Bayesian Network model.

        Args:
            model_name: Name of the model
            edges: List of edge tuples (parent, child) defining the network structure
            estimator_type: Type of estimator ('maximum_likelihood' or 'bayesian')
            prior_type: Type of prior for Bayesian estimator ('BDeu', 'K2', 'dirichlet')
            equivalent_sample_size: Equivalent sample size for Bayesian estimator
            tracking_uri: MLflow tracking server URI
            registry_uri: MLflow model registry URI
            experiment_name: MLflow experiment name
        """
        model_type = "classification"  # Can be used flexibly for classification/regression
        if experiment_name is None:
            experiment_name = "bayesian_network_models"
        super().__init__(model_name, model_type, tracking_uri, registry_uri, experiment_name)

        self.edges = edges
        self.estimator_type = estimator_type
        self.prior_type = prior_type
        self.equivalent_sample_size = equivalent_sample_size

        # Store parameters for MLflow
        self.params = {
            "edges": str(edges) if edges is not None else None,
            "estimator_type": estimator_type,
            "prior_type": prior_type,
            "equivalent_sample_size": equivalent_sample_size,
        }

        # Initialize model
        if edges is not None:
            self.model = BayesianModel(edges)
        else:
            self.model = None

        # Initialize inference engine
        self.inference_engine = None

        # Track whether structure was learned from data
        self.structure_learned = edges is None

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        validation_data: Optional[Tuple[pd.DataFrame, Optional[pd.Series]]] = None,
        log_to_mlflow: bool = True,
        **kwargs
    ) -> Dict[str, float]:
        """
        Fit the Bayesian Network model.

        Args:
            X: Training data (pandas DataFrame with variable names as columns)
            y: Unused for BayesianNetwork, everything is in X
            validation_data: Optional tuple of (X_val, y_val) for validation
            log_to_mlflow: Whether to log metrics and model to MLflow
            **kwargs: Additional parameters

        Returns:
            A dictionary containing training metrics
        """
        self.logger.info(f"Fitting Bayesian Network with {X.shape[0]} samples")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame for Bayesian Networks")

        start_time = time.time()

        # If no structure was provided, note that learning is not implemented
        if self.model is None:
            self.logger.info("No structure provided, learning structure from data")
            raise ValueError("Automatic structure learning not implemented. Please provide edges.")

        # Fit model parameters
        if self.estimator_type == "maximum_likelihood":
            _ = MaximumLikelihoodEstimator(self.model, data=X)
            self.model.fit(data=X, estimator=MaximumLikelihoodEstimator)
        elif self.estimator_type == "bayesian":
            _ = BayesianEstimator(self.model, data=X)
            self.model.fit(
                data=X,
                estimator=BayesianEstimator,
                prior_type=self.prior_type,
                equivalent_sample_size=self.equivalent_sample_size,
            )
        else:
            raise ValueError(f"Unknown estimator type: {self.estimator_type}")

        # Initialize inference engine
        self.inference_engine = VariableElimination(self.model)
        train_time = time.time() - start_time
        self.logger.info(f"Model training completed in {train_time:.2f} seconds")

        train_metrics = {
            "training_time_seconds": train_time,
            "num_nodes": len(self.model.nodes()),
            "num_edges": len(self.model.edges()),
            "avg_markov_blanket_size": np.mean(
                [len(self.model.get_markov_blanket(node)) for node in self.model.nodes()]
            ),
        }

        # Validation
        if validation_data is not None:
            X_val = validation_data[0]
            val_metrics = self._calculate_metrics(X_val, prefix="val_")
            train_metrics.update(val_metrics)

        # Log to MLflow
        if log_to_mlflow:
            with mlflow.start_run(run_name=f"{self.model_name}_training"):
                mlflow.log_params(self.params)
                for metric_name, metric_value in train_metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

                model_wrapper = CustomBayesianNetworkWrapper(self.model, self.inference_engine)
                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=model_wrapper
                )

                self._log_network_visualization(mlflow)
                self._log_cpt_tables(mlflow)
                self.logger.info("Training metrics and model logged to MLflow.")

        return train_metrics

    def _calculate_metrics(self, X: pd.DataFrame, prefix: str = "") -> Dict[str, float]:
        """
        Calculate validation metrics for Bayesian Networks.

        Args:
            X: Validation data
            prefix: Prefix for metrics

        Returns:
            Dictionary with log-likelihood metrics
        """
        metrics = {}
        log_likelihood = 0.0

        for _, row in X.iterrows():
            for node in self.model.nodes():
                evidence = {var: val for var, val in row.items() if var != node and pd.notna(val)}
                if evidence:
                    query_result = self.inference_engine.query(variables=[node], evidence=evidence)
                    node_value = row[node]
                    if pd.notna(node_value):
                        prob_values = query_result.values
                        node_states = query_result.state_names[node]
                        if node_value in node_states:
                            idx = node_states.index(node_value)
                            probability = prob_values[idx]
                            log_likelihood += np.log(probability) if probability > 0 else -np.inf

        metrics[f"{prefix}log_likelihood"] = log_likelihood
        metrics[f"{prefix}avg_log_likelihood"] = log_likelihood / X.shape[0]
        return metrics

    def predict(self, X: pd.DataFrame, variables: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Predict missing variables in X.

        Args:
            X: Features (evidence variables) as a DataFrame
            variables: Variable names to predict (predict all missing if None)

        Returns:
            DataFrame with predicted states
        """
        if self.inference_engine is None:
            raise ValueError("Model has not been fitted yet")

        results = []
        for _, row in X.iterrows():
            evidence = {var: val for var, val in row.items() if pd.notna(val)}

            if variables is None:
                vars_to_predict = [node for node in self.model.nodes() if node not in evidence]
            else:
                vars_to_predict = [var for var in variables if var not in evidence]

            predicted_values = {}
            for var in vars_to_predict:
                query_result = self.inference_engine.query(variables=[var], evidence=evidence)
                max_idx = np.argmax(query_result.values)
                predicted_values[var] = query_result.state_names[var][max_idx]
                for idx, state in enumerate(query_result.state_names[var]):
                    predicted_values[f"{var}_{state}_prob"] = query_result.values[idx]
            results.append(predicted_values)

        return pd.DataFrame(results)

    def predict_probability(self, X: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
        """
        Return probability distributions for specified variables.

        Args:
            X: Evidence DataFrame
            variables: Variables to compute probabilities for

        Returns:
            DataFrame containing probability distributions
        """
        if self.inference_engine is None:
            raise ValueError("Model has not been fitted yet")

        results = []
        for _, row in X.iterrows():
            evidence = {var: val for var, val in row.items() if pd.notna(val)}
            predicted_values = {}
            for var in variables:
                if var in evidence:
                    # Known value
                    possible_states = self.model.get_cpds(var).state_names[var]
                    for state in possible_states:
                        predicted_values[f"{var}_{state}_prob"] = (
                            1.0 if state == evidence[var] else 0.0
                        )
                else:
                    query_result = self.inference_engine.query(variables=[var], evidence=evidence)
                    for idx, state in enumerate(query_result.state_names[var]):
                        predicted_values[f"{var}_{state}_prob"] = query_result.values[idx]
            results.append(predicted_values)
        return pd.DataFrame(results)

    def _log_network_visualization(self, mlflow_client):
        """
        Create and log network visualization to MLflow.
        """
        plt.figure(figsize=(12, 8))
        G = nx.DiGraph()
        G.add_nodes_from(self.model.nodes())
        G.add_edges_from(self.model.edges())
        pos = nx.spring_layout(G, seed=42)
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color="lightblue",
            node_size=1500,
            arrowsize=20,
            font_size=12,
            font_weight="bold",
            arrows=True,
        )
        plt.title("Bayesian Network Structure")
        plt.tight_layout()
        plt.savefig("network_structure.png", dpi=300, bbox_inches="tight")
        mlflow_client.log_artifact("network_structure.png")
        plt.close()

    def _log_cpt_tables(self, mlflow_client):
        """
        Log conditional probability tables to MLflow.
        """
        with open("cpt_tables.md", "w") as f:
            f.write("# Conditional Probability Tables\n\n")
            for node in self.model.nodes():
                f.write(f"## {node}\n\n")
                cpd = self.model.get_cpds(node)
                f.write("```\n")
                f.write(str(cpd))
                f.write("\n```\n\n")
        mlflow_client.log_artifact("cpt_tables.md")

    def save(self, path: str) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save the model
        """
        if not path.endswith(".pkl"):
            path = f"{path}.pkl"

        model_data = {
            "structure": list(self.model.edges()),
            "cpds": [cpd.to_dict() for cpd in self.model.get_cpds()],
        }

        with open(path, "wb") as f:
            joblib.dump(model_data, f)
        self.logger.info(f"Model saved to {path}")

        metadata = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "params": self.params,
            "nodes": list(self.model.nodes()),
        }
        metadata_path = f"{path.rsplit('.', 1)[0]}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        self.logger.info(f"Model metadata saved to {metadata_path}")

    @classmethod
    def load(cls, path: str) -> "BayesianNetworkModel":
        """
        Load model from disk.
        """
        if not path.endswith(".pkl"):
            path = f"{path}.pkl"

        with open(path, "rb") as f:
            model_data = joblib.load(f)

        edges = model_data["structure"]
        instance = cls(edges=edges)
        instance.model = BayesianModel(edges)

        # Reconstruct CPDs
        for cpd_dict in model_data["cpds"]:
            instance.model.add_cpds(BayesianNetworkModel._dict_to_cpd(cpd_dict))

        instance.inference_engine = VariableElimination(instance.model)

        metadata_path = f"{path.rsplit('.', 1)[0]}_metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            instance.model_name = metadata["model_name"]
            instance.model_type = metadata["model_type"]
            instance.params = metadata["params"]

        return instance

    @staticmethod
    def _dict_to_cpd(cpd_dict):
        """
        Convert dictionary to CPD object (inverse of cpd.to_dict()).
        This might be simplified for brevity, but conceptually would reconstruct a CPD.
        """
        from pgmpy.factors.discrete import TabularCPD

        variable = cpd_dict["variable"]
        variable_card = cpd_dict["cardinality"][variable]
        evidence = cpd_dict["cardinality"].keys() - {variable}
        evidence_list = list(evidence)
        evidence_card = [cpd_dict["cardinality"][e] for e in evidence_list]
        values = np.array(cpd_dict["values"])
        cpd = TabularCPD(variable=variable, variable_card=variable_card,
                         values=values, evidence=evidence_list,
                         evidence_card=evidence_card)
        return cpd

