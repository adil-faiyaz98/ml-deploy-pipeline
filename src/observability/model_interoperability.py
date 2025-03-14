# model_interpretability.py
import shap
import xgboost
import numpy as np
import matplotlib.pyplot as plt

# Load model & data
model = xgboost.Booster()
model.load_model("models/model.xgb")
X = np.load("data/sample_data.npy")

# Generate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Plot summary
shap.summary_plot(shap_values, X)
plt.savefig("logs/shap_summary.png")
print("Model interpretability analysis completed.")