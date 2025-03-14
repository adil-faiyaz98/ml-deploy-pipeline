# fairness_audit.py
import pandas as pd
import numpy as np
from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset

# Load original and transformed datasets
df_original = pd.read_csv("data/training_data.csv")
df_transformed = pd.read_csv("data/training_data_balanced.csv")

# Create BinaryLabelDataset
dataset_orig = BinaryLabelDataset(df=df_original, label_names=["outcome"], protected_attribute_names=["gender", "race"])
dataset_transformed = BinaryLabelDataset(df=df_transformed, label_names=["outcome"], protected_attribute_names=["gender", "race"])

# Compute fairness metrics
metric = ClassificationMetric(dataset_orig, dataset_transformed)
print("Disparate Impact Ratio:", metric.disparate_impact())
print("Statistical Parity Difference:", metric.statistical_parity_difference())
print("Fairness audit completed.")