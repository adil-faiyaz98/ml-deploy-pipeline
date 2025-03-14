# Model Inference Script (inference.py)

import joblib
from flask import Flask, request, jsonify

# Load Model
model = joblib.load("models/model.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    prediction = model.predict([data["input"]])
    return jsonify({"prediction": int(prediction[0])})

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "Model is healthy"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
