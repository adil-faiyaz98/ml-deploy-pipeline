
# differential_privacy_train.py
import tensorflow as tf
import tensorflow_privacy as tfp

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

optimizer = tfp.DPAdamGaussianOptimizer(l2_norm_clip=1.0, noise_multiplier=1.1, num_microbatches=10)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

print("Training with differential privacy completed.")
