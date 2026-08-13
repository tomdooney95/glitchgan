"""
Run the multi-class few-shot experiment across several random seeds, to
check whether a result found at a single seed (e.g. the augmentation
"rescue" effect) is robust or a lucky draw from which specific real samples
happened to get selected. Orchestrates build_multiclass_dataset_noisy.py +
classify_multiclass.py as subprocesses for each seed (so each seed is a
fully independent replicate of the whole pipeline -- data selection,
balancing, generation, and classifier training all reseeded together),
writing results to seed-specific subdirectories, then aggregates
(mean/std/min/max across seeds) into one summary table.

Runs sequentially (one seed/config training run at a time) rather than in
parallel, so each subprocess starts with a clean TF session and GPU memory
is freed between runs.

Usage:
    python scripts/run_multiclass_seed_sweep.py \\
        --holdout-npz data_holdout90/holdout_real.npz \\
        --generator-checkpoint GAN_outputs_holdout90/cDVGAN/monitor/generator_210.keras \\
        --train-samples-per-class 8 \\
        --val-frac 0.2 \\
        --target-per-class 5000 \\
        --seeds 1,2,3,4,5 \\
        --out-dir multiclass_seed_sweep_fewshot8
"""

import argparse
import os
import subprocess
import sys

import numpy as np

CONFIGS = ["real_natural", "real_balanced", "real_fake_augmented", "fake_only"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the multi-class few-shot experiment across several seeds and "
                     "aggregate results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--holdout-npz", type=str, required=True)
    parser.add_argument("--generator-checkpoint", type=str, required=True)
    parser.add_argument("--train-samples-per-class", type=int, default=None,
                        help="Passed through to build_multiclass_dataset_noisy.py. If "
                             "omitted, --train-frac is used instead.")
    parser.add_argument("--train-frac", type=float, default=None,
                        help="Alternative to --train-samples-per-class (proportional "
                             "split instead of undersample-to-minority).")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--target-per-class", type=int, default=None)
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5",
                        help="Comma-separated list of seeds, one full pipeline replicate "
                             "each (used for both dataset-building and classifier training).")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--gpu", type=str, default=None,
                        help="If given, sets CUDA_VISIBLE_DEVICES for the classifier "
                             "training subprocesses (dataset building doesn't need a GPU).")
    return parser.parse_args()


def run(cmd, env=None):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(cmd)}")


def main():
    args = parse_args()
    if args.train_samples_per_class is None and args.train_frac is None:
        raise ValueError("Specify either --train-samples-per-class or --train-frac.")

    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    gpu_env = os.environ.copy()
    if args.gpu is not None:
        gpu_env["CUDA_VISIBLE_DEVICES"] = args.gpu

    for seed in seeds:
        seed_dir = os.path.join(args.out_dir, f"seed{seed}")
        dataset_npz = os.path.join(seed_dir, "dataset.npz")

        print(f"\n{'=' * 70}\nSEED {seed}: building dataset\n{'=' * 70}")
        build_cmd = [
            sys.executable, os.path.join(SCRIPT_DIR, "build_multiclass_dataset_noisy.py"),
            "--holdout-npz", args.holdout_npz,
            "--generator-checkpoint", args.generator_checkpoint,
            "--val-frac", str(args.val_frac),
            "--seed", str(seed),
            "--out-npz", dataset_npz,
        ]
        if args.train_samples_per_class is not None:
            build_cmd += ["--train-samples-per-class", str(args.train_samples_per_class)]
        else:
            build_cmd += ["--train-frac", str(args.train_frac)]
        if args.target_per_class is not None:
            build_cmd += ["--target-per-class", str(args.target_per_class)]
        run(build_cmd)

        for config in CONFIGS:
            print(f"\n{'=' * 70}\nSEED {seed}: training [{config}]\n{'=' * 70}")
            classify_cmd = [
                sys.executable, os.path.join(SCRIPT_DIR, "classify_multiclass.py"),
                "--dataset-npz", dataset_npz,
                "--config", config,
                "--seed", str(seed),
                "--out-dir", os.path.join(seed_dir, config),
            ]
            run(classify_cmd, env=gpu_env)

    print(f"\n{'=' * 70}\nAGGREGATING ACROSS {len(seeds)} SEEDS\n{'=' * 70}")
    per_config_acc = {c: [] for c in CONFIGS}
    for seed in seeds:
        for config in CONFIGS:
            path = os.path.join(args.out_dir, f"seed{seed}", config, "multiclass_results.npz")
            if not os.path.exists(path):
                print(f"  [missing] {path}")
                continue
            r = np.load(path, allow_pickle=True)
            per_config_acc[config].append(float(r["accuracy"]))

    print(f"\n{'Config':<22}{'seeds':>7}{'mean':>10}{'std':>10}{'min':>10}{'max':>10}   per-seed")
    for config in CONFIGS:
        accs = per_config_acc[config]
        if not accs:
            print(f"{config:<22}  no results")
            continue
        accs_arr = np.array(accs)
        per_seed_str = ", ".join(f"{a:.3f}" for a in accs)
        print(f"{config:<22}{len(accs):>7}{accs_arr.mean():>10.3f}{accs_arr.std():>10.3f}"
             f"{accs_arr.min():>10.3f}{accs_arr.max():>10.3f}   [{per_seed_str}]")

    np.savez(
        os.path.join(args.out_dir, "seed_sweep_summary.npz"),
        seeds=np.array(seeds),
        **{f"{config}_accs": np.array(per_config_acc[config]) for config in CONFIGS},
    )
    print(f"\nSaved: {os.path.join(args.out_dir, 'seed_sweep_summary.npz')}")


if __name__ == "__main__":
    main()
