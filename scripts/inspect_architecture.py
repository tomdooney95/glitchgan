"""
Dump the actual GlitchGAN (cDVGAN) TensorFlow architecture for signal_length=8192,
to rebuild the cDVGAN_Architecture LaTeX table in the paper -- the table currently
there was copied from a different, stale config (its Reshape/output-shape columns
don't match what the code actually produces for an 8192-sample input; e.g. the
discriminator's first conv there has stride 2, but the real first conv block is
stride 1 -- see get_discriminator_model()'s first _conv_block() call).

Builds the discriminator, first-derivative (DV) discriminator, and generator
fresh via the same factory functions GlitchGAN's cDVGAN base class uses
(glitchgan.tf.model_components), walks each model's layers, and groups the
Conv1D/UpSampling1D+Conv1D/BatchNorm/Activation/Dropout runs emitted by
_conv_block()/_upsample_block() back into single "Convolutional"/"Transposed
conv." rows -- matching the paper table's row granularity rather than Keras's
layer-by-layer granularity. cDVGAN2's second-derivative discriminator is not
included: GlitchGAN subclasses cDVGAN specifically (see gan_models.py), and
history.json's keys (d_loss/d2d_loss/g_loss/g_loss2d/g_loss_combined, no
d2d2_loss) confirm the epoch-210 checkpoint was trained as cDVGAN, not cDVGAN2.

Run in the cdvgan conda env (has glitchgan + keras 3 installed):
    conda run -n cdvgan python scripts/inspect_architecture.py
"""

import keras

from glitchgan.tf.gan_models import GlitchGAN
from glitchgan.tf.model_components import (
    get_derivative_discriminator_model,
    get_discriminator_model,
    get_generator_model,
)

SIGNAL_LENGTH = GlitchGAN.SIGNAL_LENGTH  # 8192
NUM_CLASSES = GlitchGAN.NUM_CLASSES      # 7
NOISE_DIM = 100

GROUPABLE = {"BatchNormalization", "LeakyReLU", "ReLU", "Activation", "Dropout"}

COLUMNS = ["Operation", "Output shape", "Kernel", "Stride", "Dropout", "BN", "Activation", "Params"]
WIDTHS = [24, 18, 8, 8, 9, 4, 12, 12]


def shape_str(layer):
    dims = tuple(layer.output.shape[1:])
    return "(" + ",".join(str(d) for d in dims) + ")"


def activation_name(layer):
    cls = layer.__class__.__name__
    if cls == "LeakyReLU":
        return "Leaky ReLU"
    if cls == "ReLU":
        return "ReLU"
    if cls == "Activation":
        return layer.activation.__name__.capitalize()
    return "--"


def describe_model(model):
    """Returns list of row dicts, in the model's topological layer order."""
    rows = []
    layers = model.layers
    was_upsample = False
    i = 0
    while i < len(layers):
        layer = layers[i]
        cls = layer.__class__.__name__

        if cls == "UpSampling1D":
            was_upsample = True
            i += 1
            continue

        if cls == "Conv1D":
            j = i + 1
            bn, activation, dropout = False, "--", 0
            while j < len(layers) and layers[j].__class__.__name__ in GROUPABLE:
                nxt = layers[j]
                nxt_cls = nxt.__class__.__name__
                if nxt_cls == "BatchNormalization":
                    bn = True
                elif nxt_cls == "Dropout":
                    dropout = nxt.rate
                else:
                    activation = activation_name(nxt)
                j += 1
            rows.append(dict(
                op="Transposed conv." if was_upsample else "Convolutional",
                shape=shape_str(layer), kernel=layer.kernel_size[0], stride=layer.strides[0],
                dropout=dropout, bn=bn, activation=activation,
                params=sum(l.count_params() for l in layers[i:j]),
            ))
            was_upsample = False
            i = j
            continue

        was_upsample = False
        if cls == "InputLayer":
            label = "Class input" if layer.output.shape[-1] == NUM_CLASSES else "Input"
            rows.append(dict(op=label, shape=shape_str(layer), kernel="--", stride="--",
                             dropout=0, bn=False, activation="--", params=0))
        elif cls == "Dense":
            act = layer.activation.__name__ if layer.activation else "linear"
            rows.append(dict(op="Dense", shape=shape_str(layer), kernel="--", stride="--",
                             dropout=0, bn=False, activation=act.capitalize(),
                             params=layer.count_params()))
        elif cls in ("Reshape", "Flatten", "Concatenate", "GlobalAveragePooling1D", "Add"):
            label = {"GlobalAveragePooling1D": "Global Avg. Pooling"}.get(cls, cls)
            rows.append(dict(op=label, shape=shape_str(layer), kernel="--", stride="--",
                             dropout=0, bn=False, activation="--", params=0))
        elif cls == "Embedding":
            rows.append(dict(op="Class embedding", shape=shape_str(layer), kernel="--", stride="--",
                             dropout=0, bn=False, activation="--", params=layer.count_params()))
        elif cls == "ArgmaxLayer":
            rows.append(dict(op="Argmax (class index)", shape=shape_str(layer), kernel="--",
                             stride="--", dropout=0, bn=False, activation="--", params=0))
        elif cls == "ReduceSumDotLayer":
            rows.append(dict(op="Scalar product", shape=shape_str(layer), kernel="--", stride="--",
                             dropout=0, bn=False, activation="--", params=0))
        else:
            rows.append(dict(op=cls, shape=shape_str(layer), kernel="--", stride="--",
                             dropout=0, bn=False, activation="--", params=layer.count_params()))
        i += 1

    return rows


def print_table(title, model):
    total = model.count_params()
    rows = describe_model(model)

    print(f"\n{'=' * sum(WIDTHS)}\n{title}  --  {total:,} parameters ({total / 1e6:.3f}M)\n{'=' * sum(WIDTHS)}")
    header = "".join(c.ljust(w) for c, w in zip(COLUMNS, WIDTHS))
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = [r["op"], r["shape"], str(r["kernel"]), str(r["stride"]),
                str(r["dropout"]), "Y" if r["bn"] else "", r["activation"], f"{r['params']:,}"]
        print("".join(c.ljust(w) for c, w in zip(cells, WIDTHS)))
    return total


def main():
    print(f"GlitchGAN / cDVGAN architecture check -- signal_length={SIGNAL_LENGTH}, "
         f"num_classes={NUM_CLASSES}, noise_dim={NOISE_DIM}\n"
         f"keras {keras.__version__}")

    discriminator = get_discriminator_model(SIGNAL_LENGTH, NUM_CLASSES)
    deriv_discriminator = get_derivative_discriminator_model(SIGNAL_LENGTH - 1, NUM_CLASSES)
    generator = get_generator_model(NOISE_DIM, NUM_CLASSES)

    p_disc = print_table("Discriminator", discriminator)
    p_deriv = print_table("DV Discriminator (first derivative)", deriv_discriminator)
    p_gen = print_table("Generator", generator)

    total = p_disc + p_deriv + p_gen
    print(f"\n{'=' * sum(WIDTHS)}")
    print(f"Discriminator:     {p_disc:>12,}  ({p_disc / 1e6:.3f}M)")
    print(f"DV Discriminator:  {p_deriv:>12,}  ({p_deriv / 1e6:.3f}M)")
    print(f"Generator:         {p_gen:>12,}  ({p_gen / 1e6:.3f}M)")
    print(f"Total:             {total:>12,}  ({total / 1e6:.3f}M)")
    print(f"{'=' * sum(WIDTHS)}")

    print(
        "\nTraining hyperparameters (cDVGAN class / train.py CLI defaults -- NOT "
        "recoverable from the .keras checkpoint file itself, no training config/log "
        "was found alongside it, so confirm against whatever command actually "
        "produced generator_210_keras3.keras before trusting these for the paper):"
    )
    print("  Optimizer:              RMSprop (lr=1e-4, rho=0.9, epsilon=1e-7)")
    print("  Batch size:             64 (train.py --batch-size default -- "
         "paper table currently says 512, these do not match)")
    print("  Epochs:                 500 (matches paper table)")
    print("  Loss:                   Wasserstein + gradient penalty (gp_weight=10.0)")
    print("  Critic steps (d_steps): 5 discriminator updates per generator update "
         "(not present in the old table at all)")


if __name__ == "__main__":
    main()
