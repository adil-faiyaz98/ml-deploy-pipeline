# bias_detection.py
import pandas as pd
import aif360
from aif360.datasets import StandardDataset
from aif360.algorithms.bias_mitigation import Reweighing

# Load dataset
df = pd.read_csv("data/training_data.csv")
dataset = StandardDataset(df, label_name="outcome", protected_attribute_names=["gender", "race"], privileged_classes=[[1], [1]])

# Apply bias mitigation
reweigh = Reweighing()
dataset_transformed = reweigh.fit_transform(dataset)
df_transformed = dataset_transformed.convert_to_dataframe()[0]
df_transformed.to_csv("data/training_data_balanced.csv", index=False)
print("Bias detection and mitigation completed.")