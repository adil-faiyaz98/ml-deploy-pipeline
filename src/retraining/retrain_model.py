# Model Retraining Script (retrain_model.py)

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow

# Load New Data
data = pd.read_csv("new_dataset.csv")
X = data.drop("target", axis=1)
y = data["target"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train New Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate New Model
y_pred = model.predict(X_test)
new_accuracy = accuracy_score(y_test, y_pred)
print(f"New Model Accuracy: {new_accuracy}")

# Load Old Model Accuracy
with open("models/accuracy.txt", "r") as f:
    old_accuracy = float(f.read().strip())

# Compare Accuracies
if new_accuracy > old_accuracy:
    print("New model is better. Updating...")
    joblib.dump(model, "models/model.pkl")
    with open("models/accuracy.txt", "w") as f:
        f.write(str(new_accuracy))
    mlflow.log_metric("accuracy", new_accuracy)
else:
    print("New model is worse. Retaining old model.")

print("Retraining Completed!")