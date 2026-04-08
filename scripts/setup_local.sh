#!/usr/bin/env bash
# setup_local.sh
#
# Copy large files from CIT that are excluded from git:
#   - Training data (.npy)
#   - Derivative data (.npy)
#   - PyTorch generator checkpoints (.pt)
#   - Gravity Spy CNN model (.h5)
#
# Usage:
#   bash scripts/setup_local.sh [CIT_USER] [CIT_PROJECT_DIR]
#
# The LIGO cluster is behind a jump host (ssh.ligo.org → ldas-pcdev12).
# Add this to ~/.ssh/config to avoid specifying it every time:
#
#   Host cit
#       HostName ldas-pcdev12.ligo.caltech.edu
#       User tom.dooney
#       ProxyJump tom.dooney@ssh.ligo.org
#
# With that config in place this script works with the default arguments.
# Without it, set CIT_USER and the SSH_JUMP variable below.

CIT_USER="${1:-tom.dooney}"
CIT_PROJECT="${2:-/home/tom.dooney/BayesWave_distillation/bonz_project_deepextractor/cDVGAN_for_DeepExtractor}"
CIT_GSPY_MODEL="/home/meesde.boer/gw_learn/GravitySpy/models/sidd-cqg-paper-O3-model.h5"

# If ~/.ssh/config has a 'cit' Host entry, leave SSH_ARGS empty.
# Otherwise use the ProxyJump form:
#   SSH_ARGS="-e 'ssh -J ${CIT_USER}@ssh.ligo.org'"
CIT_HOST="cit"   # matches the Host alias in ~/.ssh/config
SSH_ARGS=""       # e.g. "-e 'ssh -J ${CIT_USER}@ssh.ligo.org'" if no config alias

set -euo pipefail
cd "$(dirname "$0")/.."   # run from project root regardless of where script is called

# ── Training data ─────────────────────────────────────────────────────────────
echo "==> Syncing training data..."
mkdir -p data
rsync -avz --progress $SSH_ARGS \
  "${CIT_HOST}:${CIT_PROJECT}/data/glitch_GAN_samples_scaled_balanced.npy" \
  "${CIT_HOST}:${CIT_PROJECT}/data/glitch_GAN_labels_balanced.npy" \
  "${CIT_HOST}:${CIT_PROJECT}/data/glitch_GAN_label_order.npy" \
  data/

# ── Derivative data (download in progress on CIT) ────────────────────────────
echo "==> Syncing derivative data..."
rsync -avz --progress $SSH_ARGS \
  "${CIT_HOST}:${CIT_PROJECT}/data/glitch_GAN_deriv_samples_balanced.npy" \
  data/ || echo "[WARN] Derivative data not yet available on CIT — skipping."

# ── PyTorch generator checkpoints ─────────────────────────────────────────────
echo "==> Syncing PyTorch checkpoints..."
mkdir -p weights/pytorch
rsync -avz --progress $SSH_ARGS \
  "${CIT_HOST}:${CIT_PROJECT}/GAN_outputs/cDVGAN/generator_*.pt" \
  weights/pytorch/
rsync -avz --progress $SSH_ARGS \
  "${CIT_HOST}:${CIT_PROJECT}/GAN_outputs/cDVGAN/config.json" \
  "${CIT_HOST}:${CIT_PROJECT}/GAN_outputs/cDVGAN/history.json" \
  weights/pytorch/ 2>/dev/null || true

# ── Gravity Spy CNN model ──────────────────────────────────────────────────────
echo "==> Syncing Gravity Spy model..."
mkdir -p models
rsync -avz --progress $SSH_ARGS \
  "${CIT_HOST}:${CIT_GSPY_MODEL}" \
  models/

echo ""
echo "Done. Directory layout:"
echo "  data/           — training signals and labels"
echo "  weights/tensorflow/ — TF generator_210.keras"
echo "  weights/pytorch/    — PyTorch generator_*.pt checkpoints"
echo "  models/             — Gravity Spy sidd-cqg-paper-O3-model.h5"
