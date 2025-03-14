# optimization/model_quantization.py
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import logging

logging.basicConfig(filename='logs/model_quantization.log', level=logging.INFO)

# Load model
model = tf.keras.models.load_model("models/ml_model.h5")

# Convert to TensorFlow Lite format with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()

# Save the quantized model
with open("models/quantized_ml_model.tflite", "wb") as f:
    f.write(quantized_model)

logging.info("Model quantization completed.")
print("Model quantization completed.")