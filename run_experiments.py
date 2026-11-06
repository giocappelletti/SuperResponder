"""
run_experiments.py
==================
Runs, through subprocess.run, the experiments of the paper for the IML pipeline
and/or for Orthrus. The two pipelines are independent: they share only the
dataset and the definition of the outer splits, so they can also be run on
different machines.

    python run_experiments.py --pipeline iml                 # LR, Linear SVM, Extra Trees (hold-out and LOCO)
    python run_experiments.py --pipeline orthrus             # Orthrus (hold-out and LOCO) + interpretability
    python run_experiments.py --pipeline iml orthrus --cv holdout

Useful options:
    --cv holdout loco   validation schemes to run (default: both)
    --kernel-shap       (Orthrus) also compare the composed importances with KernelSHAP (slow)
    --dry-run           print the commands without running them
    --no-skip           re-run the steps already completed (by default they are skipped)
    --keep-going        do not stop at the first error

The output of every step is also saved in results/logs/<step>.log.
"""

import os
import sys
import time
import argparse
import subprocess

REPO = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
RES = os.path.join(REPO, "results")
N_SPLITS = 5
LATENT_DIMS = [10, 20, 30, 50, 100, 150, 200, 250, 300, 350, 400]
XAI_DIM = 150
KERNEL_DIMS = [150, 30]
IML_MODELS = ["LR", "SVM", "ET"]


def dim_dir(cv, d):
    return os.path.join(RES, "orthrus", cv, f"dim_{d}__wd_1e-05__lr_0.001")


def expected_folds(data, cv):
    """Names of all the folds that a complete run must contain (read from the dataset)."""
    sys.path.insert(0, os.path.join(REPO, "common"))
    import splits
    from preprocessing import load_dataset
    try:
        df, _ = load_dataset(data)
    except FileNotFoundError:
        return ["<dataset not found>"]
    return [s["name"] for s in splits.get_splits(df, mode=cv, n_splits=N_SPLITS)]


def iml_steps(a):
    steps = []
    for cv in a.cv:
        folds = expected_folds(a.data, cv)
        for m in a.iml_models:
            cmd = [PY, "IML/run_iml.py", "--cv", cv, "--model", m, "--data", a.data,
                   "--n-splits", str(N_SPLITS)]
            if a.no_skip:
                cmd.append("--overwrite")
            # complete only if every fold has its metrics (a partial run is resumed, not skipped)
            done = [os.path.join(RES, "iml", cv, m, f, "metrics.csv") for f in folds]
            steps.append((f"iml_{cv}_{m}", cmd, done))
    return steps


def orthrus_steps(a):
    steps = []
    for cv in a.cv:
        folds = expected_folds(a.data, cv)
        cmd = [PY, "Orthrus/train_orthrus.py", "--cv", cv, "--data", a.data,
               "--n-splits", str(N_SPLITS), "--latent-dims", *map(str, a.latent_dims)]
        done = [os.path.join(dim_dir(cv, d), "results.csv") for d in a.latent_dims]
        steps.append((f"orthrus_train_{cv}", cmd, done))

        cmd = [PY, "Orthrus/explain_orthrus.py", "--cv", cv, "--latent-dims", str(XAI_DIM), "--no-kernel-shap"]
        done = [os.path.join(dim_dir(cv, XAI_DIM), f, "xai", "composed_only",
                             "feature_importance_AE_Classifier_Head.csv") for f in folds]
        steps.append((f"orthrus_xai_{cv}", cmd, done))

        if a.kernel_shap and cv == "holdout":
            for d in a.kernel_dims:
                cmd = [PY, "Orthrus/explain_orthrus.py", "--cv", cv, "--latent-dims", str(d)]
                done = [os.path.join(dim_dir(cv, d), f, "xai", "n_samples_7000",
                                     "feature_importance_AE_Classifier_Head.csv") for f in folds]
                steps.append((f"orthrus_kernelshap_d{d}_{cv}", cmd, done))
    return steps


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pipeline", nargs="+", choices=["iml", "orthrus"], default=["iml", "orthrus"])
    p.add_argument("--cv", nargs="+", choices=["holdout", "loco"], default=["holdout", "loco"])
    p.add_argument("--data", default=os.path.join(REPO, "datasets", "raw_dataset.csv"))
    p.add_argument("--iml-models", nargs="+", default=IML_MODELS, choices=IML_MODELS)
    p.add_argument("--latent-dims", type=int, nargs="+", default=LATENT_DIMS)
    p.add_argument("--kernel-shap", action="store_true")
    p.add_argument("--kernel-dims", type=int, nargs="+", default=KERNEL_DIMS)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-skip", action="store_true")
    p.add_argument("--keep-going", action="store_true")
    a = p.parse_args()
    a.data = os.path.abspath(a.data)

    if not os.path.exists(a.data) and not a.dry_run:
        sys.exit(f"Dataset not found: {a.data}\nCopy raw_dataset.csv into datasets/ or use --data.")

    steps = []
    if "iml" in a.pipeline:
        steps += iml_steps(a)
    if "orthrus" in a.pipeline:
        steps += orthrus_steps(a)

    log_dir = os.path.join(RES, "logs")
    os.makedirs(log_dir, exist_ok=True)

    failed = []
    for i, (name, cmd, done) in enumerate(steps, 1):
        header = f"[{i}/{len(steps)}] {name}"
        if done and not a.no_skip and all(os.path.exists(f) for f in done):
            print(f"{header}: already completed, skipping (use --no-skip to re-run)")
            continue
        print(f"{header}\n    {' '.join(cmd)}")
        if a.dry_run:
            continue
        log_path = os.path.join(log_dir, f"{name}.log")
        t0 = time.time()
        with open(log_path, "w") as log:
            res = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        mins = (time.time() - t0) / 60
        if res.returncode == 0:
            print(f"    OK in {mins:.1f} min  (log: {os.path.relpath(log_path, REPO)})")
        else:
            print(f"    ERROR (exit code {res.returncode}) after {mins:.1f} min, see {os.path.relpath(log_path, REPO)}")
            failed.append(name)
            if not a.keep_going:
                sys.exit(1)

    if failed:
        sys.exit(f"Failed steps: {failed}")
    print("Done.")


if __name__ == "__main__":
    main()
