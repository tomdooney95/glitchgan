"""Convert generator_210.keras (Keras 2) to Keras 3 format.

Approach:
  1. Load the old model via tf_keras (Keras 2 compatibility layer)
  2. Extract all weights as numpy arrays
  3. Rebuild the identical architecture in Keras 3
  4. Transfer the weights
  5. Save as generator_210_keras3.keras

Run from the project root:
    python scripts/convert_tf_generator.py
"""

from pathlib import Path
import numpy as np

SRC  = Path("weights/tensorflow/generator_210.keras")
DEST = Path("weights/tensorflow/generator_210_keras3.keras")

# ── Step 1: load with tf_keras (Keras 2 compat) ──────────────────────────────
print("Loading with tf_keras...")
import tf_keras
old_model = tf_keras.models.load_model(str(SRC), compile=False)
weights = old_model.get_weights()
print(f"  Loaded. {len(weights)} weight arrays.")

# ── Step 2: rebuild architecture in Keras 3 ───────────────────────────────────
# Architecture is identical to tensorflow_reference/model_components.py
# get_generator_model(noise_dim=100, num_classes=7)
print("Rebuilding in Keras 3...")
import keras
from keras import layers

noise       = keras.Input(shape=(100,), name="noise")
class_input = keras.Input(shape=(7,),   name="class")

x = layers.Dense(32, use_bias=False, name="class_emb")(class_input)
x = layers.Concatenate(name="concat")([noise, x])
x = layers.Dense(4096, use_bias=False, name="dense_proj")(x)
x = layers.ReLU(name="relu_0")(x)
x = layers.Reshape((256, 16), name="reshape")(x)

for i, filters in enumerate([512, 256, 128, 64]):
    x = layers.UpSampling1D(2,  name=f"up_{i}")(x)
    x = layers.Conv1D(filters, 18, padding="same", use_bias=False, name=f"conv_{i}")(x)
    x = layers.BatchNormalization(momentum=0.99, epsilon=0.001, name=f"bn_{i}")(x)
    x = layers.ReLU(name=f"relu_{i+1}")(x)

x = layers.UpSampling1D(2, name="up_4")(x)
x = layers.Conv1D(1, 18, padding="same", use_bias=False, name="conv_4")(x)
x = layers.Activation("linear", name="linear")(x)
x = layers.Flatten(name="flatten")(x)

new_model = keras.Model([noise, class_input], x, name="generator")

# ── Step 3: transfer weights ──────────────────────────────────────────────────
# Both models have the same number of weight arrays in the same order.
new_model.set_weights(weights)
print(f"  Weights transferred ({len(weights)} arrays).")

# ── Step 4: quick sanity check ────────────────────────────────────────────────
test_noise = np.random.randn(1, 100).astype("float32")
test_class = np.eye(7)[:1].astype("float32")
out_old = old_model.predict([test_noise, test_class], verbose=0)
out_new = new_model.predict([test_noise, test_class], verbose=0)
max_diff = np.abs(out_old - out_new).max()
print(f"  Max output difference old vs new: {max_diff:.2e}  (should be ~0)")
assert max_diff < 1e-4, f"Weight transfer failed — max diff {max_diff}"

# ── Step 5: save in Keras 3 format ───────────────────────────────────────────
new_model.save(str(DEST))
print(f"Saved to {DEST}")
print("Done. Update TF_GENERATOR_PATH in the notebook to point to the new file.")
