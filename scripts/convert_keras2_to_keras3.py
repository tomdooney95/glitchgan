"""
Convert generator_210.keras (Keras 2 format) to Keras 3 format.

Strategy:
  1. Build the generator architecture fresh in Keras 3.
  2. Extract numpy arrays directly from the Keras 2 .keras zip/h5 file,
     bypassing the incompatible Keras 2 deserialisation.
  3. Map weights by layer index and set them on the Keras 3 model.
  4. Save to weights/tensorflow/generator_210_keras3.keras.
"""

import zipfile
import io
import numpy as np
import h5py
import keras
from keras import layers

# ---------------------------------------------------------------------------
# 1. Build generator in Keras 3
# ---------------------------------------------------------------------------
# Architecture mirrors tensorflow_reference/model_components.py::get_generator_model

def upsample_block(x, filters, activation, kernel_size=18, strides=1,
                   up_size=2, padding="same", use_bn=False, use_bias=False):
    x = layers.UpSampling1D(up_size)(x)
    x = layers.Conv1D(filters, kernel_size, strides=strides,
                      padding=padding, use_bias=use_bias)(x)
    if use_bn:
        x = layers.BatchNormalization(momentum=0.99, epsilon=0.001)(x)
    if activation is not None:
        x = activation(x)
    return x


def build_generator(noise_dim=100, num_classes=7):
    noise = layers.Input(shape=(noise_dim,))
    class_input = layers.Input(shape=(num_classes,))

    class_embedding = layers.Dense(32, use_bias=False)(class_input)
    combined = layers.Concatenate()([noise, class_embedding])
    x = layers.Dense(4096, use_bias=False)(combined)
    x = layers.ReLU()(x)
    x = layers.Reshape((256, 16))(x)

    x = upsample_block(x, 512,  layers.ReLU(), use_bn=True)
    x = upsample_block(x, 256,  layers.ReLU(), use_bn=True)
    x = upsample_block(x, 128,  layers.ReLU(), use_bn=True)
    x = upsample_block(x, 64,   layers.ReLU(), use_bn=True)
    x = upsample_block(x, 1,    layers.Activation("linear"), use_bn=False)

    x = layers.Flatten()(x)
    return keras.Model([noise, class_input], x, name="generator")


model = build_generator()
model.summary()

# ---------------------------------------------------------------------------
# 2. Extract weights from the Keras 2 .keras file
# ---------------------------------------------------------------------------
KERAS2_PATH = "weights/tensorflow/generator_210.keras"

with zipfile.ZipFile(KERAS2_PATH) as zf:
    with zf.open("model.weights.h5") as f:
        raw = f.read()

lcd = "_layer_checkpoint_dependencies"

def get_vars(h5, layer_name):
    """Return sorted list of numpy arrays for a layer."""
    group = h5[lcd][layer_name]["vars"]
    return [group[k][()] for k in sorted(group.keys(), key=int)]


with h5py.File(io.BytesIO(raw), "r") as h5:
    # Dense layers
    dense_kernel     = get_vars(h5, "dense")[0]        # (7, 32)
    dense1_kernel    = get_vars(h5, "dense_1")[0]      # (132, 4096)

    # Conv1D kernels (no bias)
    conv_k = [get_vars(h5, f"conv1d{'_' + str(i) if i else ''}")[0]
              for i in range(5)]
    # shapes: (18,16,512), (18,512,256), (18,256,128), (18,128,64), (18,64,1)

    # BatchNorm: vars order is [gamma, beta, moving_mean, moving_variance]
    bn_vars = [get_vars(h5, f"batch_normalization{'_' + str(i) if i else ''}")
               for i in range(4)]

# ---------------------------------------------------------------------------
# 3. Map weights onto Keras 3 layers by name
# ---------------------------------------------------------------------------
# Print layer names so we can see the Keras 3 indexing
for i, layer in enumerate(model.layers):
    print(f"{i:3d}  {layer.name:<35}  {type(layer).__name__}")

# Build name→weight mapping
weight_map = {}

# Identify layers by type and order (Keras 3 uses 0-based naming per type)
dense_layers   = [l for l in model.layers if isinstance(l, layers.Dense)]
conv_layers    = [l for l in model.layers if isinstance(l, layers.Conv1D)]
bn_layers      = [l for l in model.layers if isinstance(l, layers.BatchNormalization)]

assert len(dense_layers) == 2, f"Expected 2 Dense layers, got {len(dense_layers)}"
assert len(conv_layers)  == 5, f"Expected 5 Conv1D layers, got {len(conv_layers)}"
assert len(bn_layers)    == 4, f"Expected 4 BatchNorm layers, got {len(bn_layers)}"

# Dense: first is class_embedding (7→32), second is projection (132→4096)
dense_layers[0].set_weights([dense_kernel])
dense_layers[1].set_weights([dense1_kernel])

# Conv1D: in order of construction
for i, (layer, kernel) in enumerate(zip(conv_layers, conv_k)):
    layer.set_weights([kernel])
    print(f"Set Conv1D layer '{layer.name}' kernel {kernel.shape}")

# BatchNorm: [gamma, beta, moving_mean, moving_variance]
for i, (layer, vars_list) in enumerate(zip(bn_layers, bn_vars)):
    gamma, beta, moving_mean, moving_var = vars_list
    layer.set_weights([gamma, beta, moving_mean, moving_var])
    print(f"Set BN layer '{layer.name}' vars {[v.shape for v in vars_list]}")

# ---------------------------------------------------------------------------
# 4. Quick sanity check — run a forward pass
# ---------------------------------------------------------------------------
noise_test  = np.random.randn(4, 100).astype("float32")
label_test  = np.zeros((4, 7), dtype="float32")
label_test[:, 0] = 1.0
out = model([noise_test, label_test], training=False)
print(f"\nSanity check output shape: {out.shape}")   # expect (4, 8192)
print(f"Output mean: {float(np.mean(out)):.6f}  std: {float(np.std(out)):.6f}")

# ---------------------------------------------------------------------------
# 5. Save in Keras 3 format
# ---------------------------------------------------------------------------
OUT_PATH = "weights/tensorflow/generator_210_keras3.keras"
model.save(OUT_PATH)
print(f"\nSaved Keras 3 model to {OUT_PATH}")

# Verify round-trip
model2 = keras.models.load_model(OUT_PATH)
out2 = model2([noise_test, label_test], training=False)
max_diff = float(np.max(np.abs(np.array(out) - np.array(out2))))
print(f"Round-trip max weight diff: {max_diff:.2e}  (should be ~0)")
