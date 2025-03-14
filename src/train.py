# Model Training Script (train.py)

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow

# Load Data
data = pd.read_csv("dataset.csv")
X = data.drop("target", axis=1)
y = data["target"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Save Model
joblib.dump(model, "models/model.pkl")

# Log Metrics to MLflow
mlflow.log_metric("accuracy", accuracy)
mlflow.sklearn.log_model(model, "random_forest")

print("Model Training Completed!")



