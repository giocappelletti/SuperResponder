"""
run_iml.py
==========
Interpretable Machine Learning (IML) pipeline: Logistic Regression, Linear SVM
or Extra Trees on the CLR-transformed taxa, with SHAP-driven feature selection,
evaluated with nested cross-validation (no information from the outer test set
is used for preprocessing, feature ranking, feature selection or tuning).

For every outer fold:
1. Hyperparameter tuning on the outer training set only:
       GridSearchCV(Pipeline[CLR + StandardScaler -> classifier], grid,
                    cv=5 (StratifiedKFold, no shuffle), scoring='accuracy')
   CLR and scaling are inside the pipeline, so the scaler is refit in every inner fold.
2. Out-of-fold SHAP ranking on the outer training set only: for each of the 5
   inner folds (the same ones used by the grid search) the tuned pipeline is
   refit on the inner training part and SHAP values are computed on the inner
   validation part. The 5 validation parts cover the whole outer training set;
   taxa are ranked by mean |SHAP| over these rows.
3. Top-n ablation: for every n, the classifier is retrained on the n top-ranked
   taxa and, as a baseline, on n random taxa; in both cases the hyperparameters
   are re-tuned with the same grid and the same inner folds
   (CLR + scaling on all 4,630 taxa -> column selection -> classifier).
   For n = 4630 the pipeline tuned in step 1 is used directly.
4. Evaluation, once, on the outer test set.

Outer splits (common/splits.py), identical to Orthrus:
   --cv holdout : fold i = train_test_split(random_state=42+i, stratify=response), 80/20
   --cv loco    : Leave-One-Cohort-Out over BioProject (Responder vs Non-Responder only)

Standard deviations across folds use ddof=1.

Usage (from the repository root):
    python IML/run_iml.py --cv holdout --model LR
    python IML/run_iml.py --cv loco    --model SVM
"""

import os
import sys
import logging
import argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
for _p in (os.path.join(REPO_ROOT, "common"), os.path.dirname(os.path.abspath(__file__))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from joblib import Parallel, delayed

import splits as shared_splits
from preprocessing import load_dataset, CLRScaler
from models import get_ml_models_and_grids, inner_cv_splitter, SCORING, N_JOBS, SHORT_NAMES
from shap_explainer import shap_values_positive_class

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger("IML")


TOP_N_FEATS_LIST = [10, 20, 30, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 4630]
ORR_CATEGORIES   = ['CR', 'PR', 'PD', 'Dead']
SEED = 42


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cv", choices=["holdout", "loco"], default="holdout")
    p.add_argument("--n-splits", type=int, default=5,
                   help="Number of outer splits for --cv holdout (ignored for loco)")
    p.add_argument("--model", choices=list(SHORT_NAMES), required=True,
                   help="SVM = Linear SVM, LR = Logistic Regression, ET = Extra Trees")
    p.add_argument("--data", default=os.path.join(REPO_ROOT, "datasets", "raw_dataset.csv"))
    p.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "results", "iml"))
    p.add_argument("--top-n", type=int, nargs="+", default=TOP_N_FEATS_LIST,
                   help="Feature-subset sizes to evaluate (default: all the values of the paper)")
    p.add_argument("--folds", nargs="+", default=None,
                   help="Run only these folds (e.g. fold_1 fold_2), to split the work across "
                        "processes; per-fold results are saved anyway")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run folds that already have results (by default they are skipped, "
                        "so an interrupted run can be resumed)")
    p.add_argument("--max-folds", type=int, default=None, help="Quick tests only")
    return p.parse_args()


def metrics(y_true, y_pred):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        # macro averages, as in Orthrus (macro recall = balanced accuracy)
        "f1":        f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def build_pipeline(model):
    """CLR + StandardScaler (fitted inside every fit) -> classifier."""
    return Pipeline([
        ('clrscaler', CLRScaler()),
        ('classifier',  clone(model)),
    ])


def tune_full_pipeline(model, grid, X_train, y_train):
    """Grid search on all taxa, same strategy as Orthrus (GridSearchCV, cv=5, accuracy)."""
    gs = GridSearchCV(build_pipeline(model),
                      {f"classifier__{k}": v for k, v in grid.items()},
                      cv=inner_cv_splitter(), scoring=SCORING, n_jobs=N_JOBS)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, {k.replace("classifier__", ""): v for k, v in gs.best_params_.items()}


def oof_shap_ranking(best_model_pipeline, X_train, y_train):
    """
    Out-of-fold SHAP ranking on the outer training set only, on the same inner
    folds used by the grid search. In each inner fold scaler and classifier are
    refitted from scratch on the inner training part.
    """
    oof_shap = np.zeros(X_train.shape, dtype=float)
    inner_cv = inner_cv_splitter()

    for k, (tr, va) in enumerate(inner_cv.split(X_train, y_train)):
        pipe = clone(best_model_pipeline)          # same hyperparameters, not fitted
        pipe.fit(X_train.iloc[tr], y_train[tr])

        scaler = pipe.named_steps['clrscaler']
        clf    = pipe.named_steps['classifier']

        X_tr_scaled = scaler.transform(X_train.iloc[tr])
        X_va_scaled = scaler.transform(X_train.iloc[va])

        oof_shap[va] = shap_values_positive_class(clf, X_tr_scaled, X_va_scaled)
        logger.info(f"  OOF SHAP: inner fold {k + 1}/{inner_cv.get_n_splits()} done")

    mean_abs = pd.Series(np.abs(oof_shap).mean(axis=0), index=X_train.columns)
    return mean_abs.sort_values(ascending=False)


def precompute_inner_folds(clrscaler, X_train, y_train):
    """
    CLR + scaling do not depend on the classifier hyperparameters: they are
    computed once per inner fold (scaler fitted on the inner training part only).
    This is equivalent to having CLRScaler inside the GridSearchCV pipeline,
    without recomputing it for every hyperparameter combination.
    """
    folds = []
    for tr, va in inner_cv_splitter().split(X_train, y_train):
        sc = clone(clrscaler).fit(X_train.iloc[tr], y_train[tr])
        folds.append((sc.transform(X_train.iloc[tr]), y_train[tr],
                      sc.transform(X_train.iloc[va]), y_train[va]))
    return folds


def _cv_score(base_clf, params, cols, inner_folds):
    accs = []
    for Ztr, ytr, Zva, yva in inner_folds:
        clf = clone(base_clf).set_params(**params).fit(Ztr[cols], ytr)
        accs.append(accuracy_score(yva, clf.predict(Zva[cols])))
    return np.mean(accs)


def tune_on_subset(base_clf, grid, cols, inner_folds, Z_train, y_train):
    """
    Grid search on the columns `cols` only: same grid, same 5 inner folds,
    accuracy scoring, refit on the outer training set (as GridSearchCV).
    Ties are resolved in favor of the first combination of the grid (as GridSearchCV).
    """
    combos = list(ParameterGrid(grid))
    scores = Parallel(n_jobs=N_JOBS)(delayed(_cv_score)(base_clf, p, cols, inner_folds) for p in combos)
    best = combos[int(np.argmax(scores))]
    model = clone(base_clf).set_params(**best).fit(Z_train[cols], y_train)
    return model, best


def main():
    args = parse_args()
    model_name = SHORT_NAMES[args.model]          # e.g. "LR" -> "Logistic Regression"

    dataset, taxa_cols = load_dataset(args.data)

    labelenc = LabelEncoder()
    labels = labelenc.fit_transform(dataset['response'])   # 0 - NR, 1 - R
    dataset_taxa = dataset[taxa_cols]

    splits = shared_splits.get_splits(dataset, mode=args.cv, n_splits=args.n_splits)
    all_fold_names = [s["name"] for s in splits]
    if args.max_folds:
        splits = splits[:args.max_folds]
    if args.folds:
        splits = [s for s in splits if s["name"] in args.folds]

    out_root = os.path.join(args.out_dir, args.cv, args.model)
    os.makedirs(out_root, exist_ok=True)

    for split in splits:
        fold = split["name"]
        fold_idx = all_fold_names.index(fold)
        logger.info(f"{model_name} | {fold} ({fold_idx + 1}/{len(all_fold_names)})")
        fold_dir = os.path.join(out_root, fold)
        if os.path.isfile(os.path.join(fold_dir, "metrics.csv")) and not args.overwrite:
            logger.info(f"{model_name} | {fold}: results already present, skipping (use --overwrite to re-run)")
            continue
        shared_splits.save_test_ids(fold_dir, split)

        # Same per-fold seed as Orthrus (random_state = 42 + fold index);
        # random feature subsets are reproducible even when running one fold at a time
        cfg = get_ml_models_and_grids(random_state=SEED + fold_idx)[model_name]
        rng = np.random.default_rng(SEED + fold_idx)

        tr_pos = shared_splits.positions(dataset, split["train_ids"])
        te_pos = shared_splits.positions(dataset, split["test_ids"])

        outer_train = dataset_taxa.iloc[tr_pos]
        outer_test  = dataset.iloc[te_pos]
        outer_train_labels = labels[tr_pos]
        outer_test_labels  = labels[te_pos]

        # --- 1. Grid search on the outer training set only, all features ---
        best_model_pipeline, best_params_full = tune_full_pipeline(
            cfg["model"], cfg["params"], outer_train, outer_train_labels)
        logger.info(f"  Best hyperparameters (all features): {best_params_full}")

        # --- 2. Out-of-fold SHAP ranking on the outer training set only ---
        ranking = oof_shap_ranking(best_model_pipeline, outer_train, outer_train_labels)
        pd.DataFrame({"feature": ranking.index,
                      "mean_abs_shap_oof": ranking.values,
                      "rank": np.arange(1, len(ranking) + 1)}
                     ).to_csv(os.path.join(fold_dir, "shap_ranking.csv"), index=False)
        all_top_feat = ranking.index.tolist()

        # --- 3. Top-n vs random-n ablation, hyperparameters re-tuned for every n ---
        clrscaler = best_model_pipeline.named_steps['clrscaler']   # fitted on the outer training set
        base_clf    = clone(cfg["model"])
        Z_train = clrscaler.transform(outer_train)
        Z_test  = clrscaler.transform(outer_test[taxa_cols])
        inner_folds = precompute_inner_folds(clrscaler, outer_train, outer_train_labels)

        metric_rows, orr_rows, pred_rows, param_rows = [], [], [], []
        for n in args.top_n:
            random_cols = [taxa_cols[j] for j in rng.choice(len(taxa_cols), size=n, replace=False)]

            for selection, cols in (("shap", all_top_feat[:n]), ("random", random_cols)):
                if n >= len(taxa_cols):
                    best_params = best_params_full
                    y_pred = best_model_pipeline.predict(outer_test[taxa_cols])
                else:
                    model, best_params = tune_on_subset(base_clf, cfg["params"], list(cols),
                                                        inner_folds, Z_train, outer_train_labels)
                    y_pred = model.predict(Z_test[list(cols)])

                metric_rows.append({"fold": fold, "n_features": n, "selection": selection,
                                    "n_test": len(y_pred), **metrics(outer_test_labels, y_pred)})
                param_rows.append({"fold": fold, "n_features": n, "selection": selection,
                                   "best_params": str(best_params)})
                pred_rows.append(pd.DataFrame({
                    "fold": fold, "n_features": n, "selection": selection,
                    "samples": outer_test['samples'].to_numpy(),
                    "BioProject": outer_test['BioProject'].to_numpy(),
                    "y_true": outer_test_labels, "y_pred": y_pred}))

                if args.cv == "holdout":     # LOCO: Responder vs Non-Responder only
                    for cat in ORR_CATEGORIES:
                        mask = (outer_test['ORR'] == cat).to_numpy()
                        if mask.sum() > 0:
                            orr_rows.append({"fold": fold, "n_features": n, "selection": selection,
                                             "category": cat, "n_samples": int(mask.sum()),
                                             "accuracy": accuracy_score(outer_test_labels[mask], y_pred[mask])})
            logger.info(f"  n={n} done")

        # per-fold saving (folds can be run in different processes / days)
        pd.DataFrame(metric_rows).to_csv(os.path.join(fold_dir, "metrics.csv"), index=False)
        pd.DataFrame(param_rows).to_csv(os.path.join(fold_dir, "best_params.csv"), index=False)
        pd.concat(pred_rows).to_csv(os.path.join(fold_dir, "predictions.csv"), index=False)
        if orr_rows:
            pd.DataFrame(orr_rows).to_csv(os.path.join(fold_dir, "orr.csv"), index=False)

    aggregate(out_root)


def aggregate(out_root):
    """Summary over all the folds present in out_root (also if run in separate processes)."""
    fold_dirs = sorted(d for d in os.listdir(out_root)
                       if os.path.isfile(os.path.join(out_root, d, "metrics.csv")))
    df_m = pd.concat(pd.read_csv(os.path.join(out_root, d, "metrics.csv")) for d in fold_dirs)
    df_m.to_csv(os.path.join(out_root, "metrics_per_fold.csv"), index=False)

    # mean and std (ddof=1) over folds
    summary = (df_m.groupby(["selection", "n_features"])
                   [["accuracy", "f1", "precision", "recall"]]
                   .agg(["mean", lambda s: s.std(ddof=1)]))
    summary.columns = [f"{m}_{'mean' if a == 'mean' else 'std'}" for m, a in summary.columns]
    summary["n_folds"] = df_m.groupby(["selection", "n_features"]).size()
    summary.to_csv(os.path.join(out_root, "summary_metrics.csv"))

    orr_files = [os.path.join(out_root, d, "orr.csv") for d in fold_dirs
                 if os.path.isfile(os.path.join(out_root, d, "orr.csv"))]
    if orr_files:
        df_o = pd.concat(pd.read_csv(f) for f in orr_files)
        df_o.to_csv(os.path.join(out_root, "orr_per_fold.csv"), index=False)
        (df_o.groupby(["selection", "n_features", "category"])["accuracy"]
             .agg(mean="mean", std=lambda s: s.std(ddof=1), n_folds="count")
             .to_csv(os.path.join(out_root, "summary_orr.csv")))

    pd.concat(pd.read_csv(os.path.join(out_root, d, "predictions.csv")) for d in fold_dirs
              ).to_csv(os.path.join(out_root, "predictions.csv"), index=False)

    logger.info(f"Results saved in {out_root} ({len(fold_dirs)} folds)")
    print(summary[["accuracy_mean", "accuracy_std", "recall_mean", "recall_std", "n_folds"]].round(4))


if __name__ == "__main__":
    main()
