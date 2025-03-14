# optimization/model_compression.py
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import logging

logging.basicConfig(filename='logs/model_compression.log', level=logging.INFO)

# Load model
model = tf.keras.models.load_model("models/ml_model.h5")

# Apply weight pruning
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
pruned_model = prune_low_magnitude(model)
pruned_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
pruned_model.save("models/compressed_ml_model.h5")
logging.info("Model compression and pruning completed.")
print("Model compression completed.")