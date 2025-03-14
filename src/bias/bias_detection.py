# bias_detection.py
import pandas as pd
import numpy as np
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/training_data.csv")

# Convert to AIF360 StandardDataset
dataset = StandardDataset(
    df, 
    label_name="outcome", 
    favorable_classes=[1], 
    protected_attribute_names=["gender", "race"], 
    privileged_classes=[[1], [1]]  # Modify based on dataset
)

# Compute Bias Metrics Before Mitigation
metric = BinaryLabelDatasetMetric(dataset, privileged_groups=[{"gender": 1, "race": 1}], unprivileged_groups=[{"gender": 0, "race": 0}])

print(f"💡 **Bias Before Mitigation:**")
print(f" - Statistical Parity Difference: {metric.statistical_parity_difference():.4f}")
print(f" - Disparate Impact: {metric.disparate_impact():.4f}")
print(f" - Equal Opportunity Difference: {metric.equal_opportunity_difference():.4f}")

# Apply Bias Mitigation (Reweighing)
reweigh = Reweighing(privileged_groups=[{"gender": 1, "race": 1}], unprivileged_groups=[{"gender": 0, "race": 0}])
dataset_transformed = reweigh.fit_transform(dataset)

# Convert back to Pandas DataFrame
df_transformed = dataset_transformed.convert_to_dataframe()[0]
df_transformed.to_csv("data/training_data_balanced.csv", index=False)
print("✅ Bias Mitigation Completed. Balanced dataset saved.")

# Compute Bias Metrics After Mitigation
metric_after = BinaryLabelDatasetMetric(dataset_transformed, privileged_groups=[{"gender": 1, "race": 1}], unprivileged_groups=[{"gender": 0, "race": 0}])

print(f"💡 **Bias After Mitigation:**")
print(f" - Statistical Parity Difference: {metric_after.statistical_parity_difference():.4f}")
print(f" - Disparate Impact: {metric_after.disparate_impact():.4f}")
print(f" - Equal Opportunity Difference: {metric_after.equal_opportunity_difference():.4f}")

# Train a Simple Model (Optional)
X = df.drop(columns=["outcome"])
y = df["outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Check Accuracy Before & After Mitigation
accuracy = accuracy_score(y_test, y_pred)
print(f"🚀 Model Accuracy Before Mitigation: {accuracy:.4f}")

X_balanced = df_transformed.drop(columns=["outcome"])
y_balanced = df_transformed["outcome"]
X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

clf.fit(X_train_bal, y_train_bal)
y_pred_bal = clf.predict(X_test_bal)
accuracy_balanced = accuracy_score(y_test_bal, y_pred_bal)
print(f"🚀 Model Accuracy After Mitigation: {accuracy_balanced:.4f}")
