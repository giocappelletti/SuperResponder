"""
explain_orthrus.py
==================
Interpretability of Orthrus. Since the encoder and the downstream linear
classifiers are linear, the importance of every input taxon is obtained in
closed form by the composition I = W_enc^T w (Eq. 3 of the paper), for Linear
SVM, Logistic Regression and the Orthrus classification head. For Extra Trees,
the non-negative Gini importances of the latent dimensions are given a sign
through their correlation with the response on the training set (Eq. 4).

Optionally (without --no-kernel-shap), the composed importances are compared
with end-to-end KernelSHAP for several numbers of samples (slow).

Reads the artifacts saved by Orthrus/train_orthrus.py.

Usage (from the repository root):
    python Orthrus/explain_orthrus.py --cv holdout --latent-dims 150 --no-kernel-shap
    python Orthrus/explain_orthrus.py --cv holdout --latent-dims 150     # with KernelSHAP
"""

import os
import time
import argparse
import joblib
import pandas as pd
import numpy as np
import torch
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap

from collections import Counter
from scipy.special import expit


# ==========================================
# Settings
# ==========================================
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import LinearAE

parser = argparse.ArgumentParser()
parser.add_argument("--cv", choices=["holdout", "loco"], default="holdout")
parser.add_argument("--latent-dims", type=int, nargs="+", default=[150])
parser.add_argument("--no-kernel-shap", action="store_true",
                    help="Compute only the analytical composition W_enc^T w (fast), "
                         "without the comparison with KernelSHAP.")
parser.add_argument("--nsamples", type=int, nargs="+", default=[1000, 3000, 7000])
parser.add_argument("--out-root", default=os.path.join(REPO_ROOT, "results", "orthrus"))
args = parser.parse_args()

base_out_dir = os.path.join(args.out_root, args.cv)     # written by train_orthrus.py

# Top-k used for plots and overlaps
top_k_list = [20, 50]

# KernelSHAP
run_kernel_shap        = not args.no_kernel_shap
n_kernel_background    = 200
n_kernel_explain       = 114   # all test samples (in LOCO: min(114, cohort size))
n_kernel_nsamples_list = args.nsamples if run_kernel_shap else [None]
n_kernel_kmeans        = 50

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"--> Running on device: {device}")


# ==========================================
# sklearn-like adapter for the classification head of the AE
# ==========================================
class _AEHeadSKLearn:
    def __init__(self, model):
        w = model.classifier.weight.detach().cpu().numpy()  # [1, latent_dim]
        b = model.classifier.bias.detach().cpu().numpy()    # [1]
        self.coef_      = w
        self.intercept_ = b

    def decision_function(self, X):
        return (X @ self.coef_.T + self.intercept_).flatten()

    def predict_proba(self, X):
        logits = self.decision_function(X)
        p_pos  = expit(logits).reshape(-1, 1)
        return np.hstack([1 - p_pos, p_pos])


# ==========================================
# Artifact loading
# ==========================================
def load_training_artifacts(exp_dir, device):
    ckpt  = torch.load(os.path.join(exp_dir, 'model_state.pt'),
                       map_location=device, weights_only=True)
    model = LinearAE(ckpt['input_dim'], ckpt['latent_dim']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    scaler          = joblib.load(os.path.join(exp_dir, 'scaler.joblib'))
    le              = joblib.load(os.path.join(exp_dir, 'label_encoder.joblib'))
    best_estimators = joblib.load(os.path.join(exp_dir, 'best_estimators.joblib'))

    arrs         = np.load(os.path.join(exp_dir, 'arrays.npz'))
    feature_cols = np.load(os.path.join(exp_dir, 'feature_cols.npy'), allow_pickle=True)

    print(f"  [LOAD] Artifacts loaded from: {exp_dir}")
    return {
        'model':           model,
        'scaler':          scaler,
        'le':              le,
        'best_estimators': best_estimators,
        'X_train_scaled':  arrs['X_train_scaled'],
        'X_test_scaled':   arrs['X_test_scaled'],
        'X_train_latent':  arrs['X_train_latent'],
        'X_test_latent':   arrs['X_test_latent'],
        'y_train':         arrs['y_train'],
        'y_test':          arrs['y_test'],
        'feature_cols':    feature_cols,
        'input_dim':       int(ckpt['input_dim']),
        'latent_dim':      int(ckpt['latent_dim']),
    }


# ==========================================
# Utilities
# ==========================================
def shorten_taxon(name, max_levels=3):
    parts = [p.strip() for p in name.split(';')]
    return ' | '.join(parts[-max_levels:]) if len(parts) >= max_levels else ' | '.join(parts)


def shorten_feature_names(feature_names, max_levels=3):
    shortened  = [shorten_taxon(n, max_levels) for n in feature_names]
    counts     = Counter(shortened)
    duplicates = {k for k, v in counts.items() if v > 1}
    if duplicates:
        result = []
        for orig, short in zip(feature_names, shortened):
            if short in duplicates:
                parts = [p.strip() for p in orig.split(';')]
                short = ' | '.join(parts[-min(max_levels + 1, len(parts)):])
            result.append(short)
        return result
    return shortened


# ==========================================
# Method 1 — Analytical composition encoder x classifier
# ==========================================
def get_composed_importance(model, ml_model, ml_name, X_train_latent, y_train):
    """
    Composition W_enc.T @ w_classifier -> signed vector [input_dim].

    Returns:
        importance [input_dim] : absolute magnitude (for ranking)
        signed     [input_dim] : signed vector (for sign and coloring)
    """
    W_enc = model.latent.weight.detach().cpu().numpy()   # [latent_dim, input_dim]

    if ml_name in ("Linear SVM", "Logistic Regression", "AE Classifier Head"):
        w_ml     = ml_model.coef_.flatten()
        composed = W_enc.T @ w_ml
        return np.abs(composed), composed

    elif ml_name == "ExtraTrees":
        # Sign approximated via latent-target correlation on the training set
        w_imp       = ml_model.feature_importances_
        corr_matrix = np.corrcoef(X_train_latent.T, y_train)
        corr_signs  = np.sign(corr_matrix[-1, :-1])
        w_signed    = w_imp * corr_signs
        composed    = W_enc.T @ w_signed
        return np.abs(composed), composed

    else:
        raise ValueError(f"Unsupported model: {ml_name}")


# ==========================================
# Method 2 — End-to-end KernelSHAP (original feature space)
# ==========================================
def get_shap_kernel(model, ml_model, ml_name,
                    X_train_scaled, X_test_scaled, device,
                    n_background=200, n_explain=60, nsamples=4096, n_kmeans=10):
    """
    Black-box KernelSHAP on the whole pipeline x -> encode -> classify -> P(class=1).

    Returns:
        shap_kernel [n_explain, input_dim]
        X_explained [n_explain, input_dim]
        elapsed_sec : float
    """
    def predict_fn(X_np):
        X_t = torch.tensor(X_np.astype(np.float32)).to(device)
        with torch.no_grad():
            z = model.encode(X_t).cpu().numpy()

        if ml_name == "Linear SVM":
            scores = ml_model.decision_function(z)
            p_pos  = expit(scores).reshape(-1, 1)
        else:
            p_pos = ml_model.predict_proba(z)[:, 1].reshape(-1, 1)

        return np.hstack([1 - p_pos, p_pos])

    rng        = np.random.default_rng(42)
    bg_idx     = rng.choice(len(X_train_scaled), size=min(n_background, len(X_train_scaled)), replace=False)
    background = shap.kmeans(X_train_scaled[bg_idx], n_kmeans)

    n_explain_eff = min(n_explain, len(X_test_scaled))
    X_explained   = X_test_scaled[:n_explain_eff]

    explainer   = shap.KernelExplainer(predict_fn, background)
    t0          = time.time()
    shap_values = explainer.shap_values(X_explained, nsamples=nsamples, silent=False)
    elapsed_sec = time.time() - t0

    if isinstance(shap_values, list):
        shap_kernel = np.array(shap_values[1])
    else:
        arr = np.array(shap_values)
        shap_kernel = arr[:, :, 1] if (arr.ndim == 3 and arr.shape[-1] == 2) else arr

    return shap_kernel, X_explained, elapsed_sec


# ==========================================
# Analysis: top-k overlap and sign agreement
# ==========================================
def compute_topk_overlap(composed_imp, shap_kernel_matrix, feature_cols, top_k):
    """Overlap between the top-k taxa according to the analytical composition and KernelSHAP."""
    top_comp_idx = np.argsort(composed_imp)[::-1][:top_k]
    kern_abs     = np.mean(np.abs(shap_kernel_matrix), axis=0)
    top_kern_idx = np.argsort(kern_abs)[::-1][:top_k]

    overlap_set = set(feature_cols[top_comp_idx]) & set(feature_cols[top_kern_idx])
    n_overlap   = len(overlap_set)
    pct_overlap = n_overlap / top_k
    return overlap_set, n_overlap, pct_overlap


def compute_sign_agreement(signed_comp, shap_kernel_matrix):
    """
    Global sign agreement over ALL features.

    For each taxon compares sign(composed[i]) with sign(mean_kernel_shap[i]).
    Features whose sign is exactly 0 in at least one of the two methods are excluded.

    Returns:
        pct_agree  : fraction of agreement (0-1)
        n_agree    : number of agreeing features
        n_valid    : number of features with non-zero sign in both methods
        agree_mask : bool [input_dim], True where the signs agree
                     (False for disagreement or excluded features)
    """
    sign_comp = np.sign(signed_comp)
    sign_kern = np.sign(np.mean(shap_kernel_matrix, axis=0))

    # Both signs must be non-zero
    valid      = (sign_comp != 0) & (sign_kern != 0)
    agree_mask = np.zeros(len(signed_comp), dtype=bool)

    if valid.sum() == 0:
        return 0.0, 0, 0, agree_mask

    agree_mask[valid] = sign_comp[valid] == sign_kern[valid]
    n_agree   = agree_mask.sum()
    n_valid   = int(valid.sum())
    pct_agree = n_agree / n_valid
    return pct_agree, int(n_agree), n_valid, agree_mask


def compute_sign_agreement_subset(signed_comp, shap_kernel_matrix, indices):
    """
    Sign agreement restricted to a subset of features (e.g. the top-k overlap).

    Parameters:
        signed_comp        : array [input_dim] of signed values from the composition
        shap_kernel_matrix : array [n_explain, input_dim]
        indices            : integer array of the selected features

    Returns:
        pct_agree, n_agree, n_valid, agree_mask_sub
        (same semantics as compute_sign_agreement, on the selected features only)
    """
    agree_mask_sub = np.zeros(len(indices), dtype=bool)
    if len(indices) == 0:
        return 0.0, 0, 0, agree_mask_sub

    sign_comp = np.sign(signed_comp[indices])
    sign_kern = np.sign(np.mean(shap_kernel_matrix[:, indices], axis=0))

    valid = (sign_comp != 0) & (sign_kern != 0)
    if valid.sum() == 0:
        return 0.0, 0, 0, agree_mask_sub

    agree_mask_sub[valid] = sign_comp[valid] == sign_kern[valid]
    n_agree   = int(agree_mask_sub.sum())
    n_valid   = int(valid.sum())
    pct_agree = n_agree / n_valid
    return pct_agree, n_agree, n_valid, agree_mask_sub


def compute_shap_sign_stability(shap_kernel_matrix):
    """
    Stability of the SHAP sign of each feature, in [0, 1]:
        1.0 -> all samples have the same sign (very stable)
        0.5 -> signs equally split (unreliable)
    A sign agreement on features with low stability is less informative.
    """
    sign_matrix = np.sign(shap_kernel_matrix)             # +1 / 0 / -1 per sample
    stability   = np.abs(sign_matrix.mean(axis=0))        # close to 1 -> stable
    return stability


def log_sign_agreement_details(feature_cols, signed_comp, shap_kernel_matrix,
                                agree_mask, top_k_list, log_print):
    """Logs the disagreeing features that appear in at least one of the top-k lists,
    together with the stability of their SHAP sign."""
    sign_comp  = np.sign(signed_comp)
    sign_kern  = np.sign(np.mean(shap_kernel_matrix, axis=0))
    kern_abs   = np.mean(np.abs(shap_kernel_matrix), axis=0)
    stability  = compute_shap_sign_stability(shap_kernel_matrix)

    for top_k in top_k_list:
        top_comp_idx = np.argsort(np.abs(signed_comp))[::-1][:top_k]
        top_kern_idx = np.argsort(kern_abs)[::-1][:top_k]
        in_any_topk  = np.zeros(len(feature_cols), dtype=bool)
        in_any_topk[top_comp_idx] = True
        in_any_topk[top_kern_idx] = True

        discordant_topk = np.where(~agree_mask & in_any_topk)[0]
        if len(discordant_topk) == 0:
            log_print(f"       No disagreeing feature in the top-{top_k}.")
        else:
            log_print(f"       Disagreeing features in the top-{top_k} ({len(discordant_topk)}):")
            for idx in discordant_topk:
                sc  = '↑' if sign_comp[idx] > 0 else '↓'
                sk  = '↑' if sign_kern[idx] > 0 else '↓'
                stab = stability[idx]
                log_print(
                    f"         Comp={sc} Kern={sk}  "
                    f"[stab_SHAP={stab:.2f}]  "
                    f"{feature_cols[idx]}"
                )


# ==========================================
# Plot: composition vs KernelSHAP (bars)
# ==========================================
def plot_comp_vs_kernel(composed_imp, signed, shap_kernel_matrix,
                        feature_cols, ml_name, top_k, n_samples_dir,
                        fold_name, n_kernel_explain=None):
    """
    Two panels: (1) composition encoder x classifier, (2) end-to-end KernelSHAP.
    Returns the set of overlapping features, or None if KernelSHAP is not available.
    """
    if shap_kernel_matrix is None:
        return None

    feature_cols = np.array(feature_cols)

    top_comp_idx  = np.argsort(composed_imp)[::-1][:top_k]
    shap_kern_abs = np.mean(np.abs(shap_kernel_matrix), axis=0)
    top_kern_idx  = np.argsort(shap_kern_abs)[::-1][:top_k]
    overlap       = set(feature_cols[top_comp_idx]) & set(feature_cols[top_kern_idx])

    def _bar_panel(ax, vals_idx, vals_scores, feat_names_all,
                   sign_arr, highlight, title, xlabel):
        names       = feat_names_all[vals_idx]
        scores      = vals_scores[vals_idx]
        order       = np.argsort(scores)
        names_ord   = names[order]
        scores_ord  = scores[order]
        names_short = np.array(shorten_feature_names(names_ord, max_levels=3))
        signs       = sign_arr[vals_idx][order]
        colors      = ['#d62728' if s > 0 else '#1f77b4' for s in signs]

        ax.barh(range(top_k), scores_ord, color=colors)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(names_short, fontsize=8 if top_k > 20 else 9)
        for tick, orig_name in zip(ax.get_yticklabels(), names_ord):
            if orig_name in highlight:
                tick.set_fontweight('bold')
                tick.set_color('#2ca02c')
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(title, fontsize=10, pad=8)
        ax.grid(axis='x', linestyle='--', alpha=0.4)

    # Sign for the KernelSHAP panel: signed mean on the top_kern_idx
    kern_sign_full = np.zeros(len(feature_cols))
    kern_sign_full[top_kern_idx] = np.mean(shap_kernel_matrix[:, top_kern_idx], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(22, max(10, top_k * 0.5)))
    plt.subplots_adjust(left=0.28, wspace=0.45)

    _bar_panel(axes[0], top_comp_idx, composed_imp, feature_cols, signed,
               overlap, f"(1) Composition\nEncoder × {ml_name}", "|W_enc.T × w_clf|")
    _bar_panel(axes[1], top_kern_idx, shap_kern_abs, feature_cols, kern_sign_full,
               overlap,
               "(2) End-to-end KernelSHAP"
               + (f"\n(n_explain={n_kernel_explain})" if n_kernel_explain else ""),
               "mean |SHAP|")

    fig.legend(handles=[
        mpatches.Patch(color='#d62728', label='↑ Increases P(response=1)'),
        mpatches.Patch(color='#1f77b4', label='↓ Decreases P(response=1)'),
        mpatches.Patch(color='#2ca02c', label='Shared by the two methods (bold)'),
    ], loc='lower center', ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        f"Top-{top_k} features  |  {ml_name}  |  {fold_name}\n"
        f"Overlap composition vs KernelSHAP: {len(overlap)}/{top_k} ({len(overlap)/top_k:.0%})",
        fontsize=12, y=1.01
    )
    plt.tight_layout()

    plt.savefig(os.path.join(n_samples_dir,
        f"plot_comp_vs_kernel_top{top_k}_{ml_name.replace(' ', '_')}.png"),
        dpi=300, bbox_inches='tight')
    plt.close()
    return overlap


# ==========================================
# Feature importance CSV
# ==========================================
def save_importance_csv(composed_imp, signed, feature_cols,
                        ml_name, n_samples_dir,
                        shap_kernel_matrix=None, agree_mask=None):
    """CSV with importances, sign agreement and SHAP sign stability for every feature."""
    feature_cols = np.array(feature_cols)
    data = {
        'feature':             feature_cols,
        'composed_importance': composed_imp,
        'composed_signed':     signed,
    }
    if shap_kernel_matrix is not None:
        mean_abs    = np.mean(np.abs(shap_kernel_matrix), axis=0)
        mean_signed = np.mean(shap_kernel_matrix, axis=0)
        stability   = compute_shap_sign_stability(shap_kernel_matrix)

        data['shap_kernel_mean_abs']    = mean_abs
        data['shap_kernel_mean_signed'] = mean_signed
        data['shap_sign_stability']     = stability

    df_out = pd.DataFrame(data)
    df_out['rank_composed'] = df_out['composed_importance'].rank(ascending=False).astype(int)

    if shap_kernel_matrix is not None:
        df_out['rank_shap_kernel'] = df_out['shap_kernel_mean_abs'].rank(ascending=False).astype(int)
        if agree_mask is not None:
            # NaN where the sign is zero in at least one of the two methods
            valid    = (np.sign(signed) != 0) & (np.sign(mean_signed) != 0)
            # object array: works with both numpy 1.x and 2.x (strings mixed with NaN)
            sign_col = np.where(agree_mask, 'agree', 'disagree').astype(object)
            sign_col[~valid] = np.nan
            df_out['sign_agreement'] = sign_col

    df_out = df_out.sort_values('rank_composed')
    fname  = os.path.join(n_samples_dir,
        f"feature_importance_{ml_name.replace(' ', '_')}.csv")
    df_out.to_csv(fname, index=False)
    return df_out


# ==========================================
# Global statistics CSV
# ==========================================
def save_stats_row(n_samples_dir, ml_name, fold_name,
                   n_kernel_nsamples,
                   overlap_pct_top20, overlap_pct_top50,
                   sign_agr_pct, sign_agr_overlap20, sign_agr_overlap50, kernel_time_sec):
    """Appends one row to the statistics CSV (stats_kernelshap_<nsamples>.csv)."""
    row = {
        'n_kernel_nsamples':  n_kernel_nsamples,
        'ml_name':            ml_name,
        'fold':               fold_name,
        'overlap_pct_top20':  round(overlap_pct_top20 * 100, 2),
        'overlap_pct_top50':  round(overlap_pct_top50 * 100, 2),
        'sign_agreement_pct': round(sign_agr_pct      * 100, 2),
        'sign_agr_overlap20': round(sign_agr_overlap20 * 100, 2),
        'sign_agr_overlap50': round(sign_agr_overlap50 * 100, 2),
        'kernel_time_sec':    round(kernel_time_sec, 2),
    }
    stats_path = os.path.join(n_samples_dir, f"stats_kernelshap_{n_kernel_nsamples}.csv")
    df_row     = pd.DataFrame([row])

    if os.path.exists(stats_path):
        df_existing = pd.read_csv(stats_path)
        df_out      = pd.concat([df_existing, df_row], ignore_index=True)
    else:
        df_out = df_row

    df_out.to_csv(stats_path, index=False)
    return df_row


# ==========================================
# Main interpretability loop
# ==========================================
for n_kernel_nsamples in n_kernel_nsamples_list:
    print(f"\n{'='*60}\n  n_kernel_nsamples = {n_kernel_nsamples}\n{'='*60}")

    for which_latent_dim in args.latent_dims:
        lr        = 1e-3
        l1_lambda = 1e-5
        dim_dir = os.path.join(base_out_dir, f"dim_{which_latent_dim}__wd_{l1_lambda}__lr_{lr}")
        if not os.path.isdir(dim_dir):
            print(f"[SKIP] Folder not found: {dim_dir}")
            continue

        # Folds actually present: fold_1..fold_N (hold-out) or loco_<BioProject>
        fold_names = sorted(d for d in os.listdir(dim_dir)
                            if os.path.isdir(os.path.join(dim_dir, d))
                            and (d.startswith("fold_") or d.startswith("loco_")))

        for i, fold_name in enumerate(fold_names):
            np.random.seed(42 + i)

            config_name = f"dim_{which_latent_dim}__wd_{l1_lambda}__lr_{lr}/{fold_name}"
            exp_dir     = os.path.join(base_out_dir, config_name)

            xai_dir = os.path.join(exp_dir, 'xai')
            os.makedirs(xai_dir, exist_ok=True)

            sub = f'n_samples_{n_kernel_nsamples}' if run_kernel_shap else 'composed_only'
            n_samples_dir = os.path.join(xai_dir, sub)
            os.makedirs(n_samples_dir, exist_ok=True)

            arts = load_training_artifacts(exp_dir, device)

            model           = arts['model']
            best_estimators = arts['best_estimators']
            feature_cols    = arts['feature_cols']
            X_train_scaled  = arts['X_train_scaled']
            X_test_scaled   = arts['X_test_scaled']
            X_train_latent  = arts['X_train_latent']
            X_test_latent   = arts['X_test_latent']
            y_train         = arts['y_train']
            y_test          = arts['y_test']

            ae_head_sklearn = _AEHeadSKLearn(model)
            all_classifiers = {**best_estimators, "AE Classifier Head": ae_head_sklearn}

            log_file_path = os.path.join(n_samples_dir, "interpretability_log.txt")

            with open(log_file_path, "w", encoding="utf-8") as log_file:
                def log_print(message):
                    print(message)
                    log_file.write(message + "\n")

                log_print(f"\n{'='*60}")
                log_print(f"  [INTERPRETABILITY] {config_name}")
                log_print(f"  Top-k analysed: {top_k_list}")
                log_print(f"  Classifiers: {list(all_classifiers.keys())}")
                log_print(f"{'='*60}")

                overlap_summary     = {}   # ml_name -> {top_k: (n, pct)}
                sign_agr_summary    = {}   # ml_name -> (pct, n_agree, n_valid)
                kernel_time_summary = {}   # ml_name -> elapsed_sec

                for ml_name, clf in all_classifiers.items():
                    log_print(f"\n  >> {ml_name}")

                    # ---- Method 1: analytical composition ----
                    log_print("     [1] Analytical composition encoder x classifier...")
                    if ml_name == "ExtraTrees":
                        log_print("         [!] Sign approximated via latent-target correlation")

                    composed_imp, signed = get_composed_importance(
                        model, clf, ml_name, X_train_latent, y_train
                    )

                    for top_k in top_k_list:
                        top_comp_idx = np.argsort(composed_imp)[::-1][:top_k]
                        log_print(f"     Top-{top_k} (composition):")
                        for fname, fabs, fsign in zip(feature_cols[top_comp_idx],
                                                      composed_imp[top_comp_idx],
                                                      signed[top_comp_idx]):
                            log_print(
                                f"       [{'↑' if fsign > 0 else '↓'}] "
                                f"{fname:<50} {fabs:.6f}"
                            )

                    # ---- Method 2: KernelSHAP ----
                    shap_kernel = None
                    X_explained = None
                    n_explained = None
                    elapsed_sec = None

                    if run_kernel_shap:
                        log_print(
                            f"     [2] End-to-end KernelSHAP "
                            f"(background={n_kernel_background}, "
                            f"explain={n_kernel_explain}, "
                            f"nsamples={n_kernel_nsamples})..."
                        )
                        if ml_name == "Linear SVM":
                            log_print("         [!] P(class=1) approximated with expit(decision_function)")

                        try:
                            shap_kernel, X_explained, elapsed_sec = get_shap_kernel(
                                model          = model,
                                ml_model       = clf,
                                ml_name        = ml_name,
                                X_train_scaled = X_train_scaled,
                                X_test_scaled  = X_test_scaled,
                                device         = device,
                                n_background   = n_kernel_background,
                                n_explain      = n_kernel_explain,
                                nsamples       = n_kernel_nsamples,
                                n_kmeans       = n_kernel_kmeans,
                            )
                            n_explained = len(X_explained)
                            kernel_time_summary[ml_name] = elapsed_sec
                            log_print(
                                f"     KernelSHAP completed in {elapsed_sec:.1f}s "
                                f"({elapsed_sec/60:.1f} min)"
                            )

                            shap_kern_abs = np.mean(np.abs(shap_kernel), axis=0)
                            for top_k in top_k_list:
                                top_kern_idx = np.argsort(shap_kern_abs)[::-1][:top_k]
                                log_print(f"     Top-{top_k} (KernelSHAP, on {n_explained} samples):")
                                for fname, fval in zip(feature_cols[top_kern_idx],
                                                       shap_kern_abs[top_kern_idx]):
                                    log_print(f"       {fname:<50} {fval:.6f}")

                        except Exception as e:
                            log_print(f"     [WARNING] KernelSHAP failed: {e}")
                            shap_kernel = None

                    # ---- Overlap for each top_k ----
                    overlap_summary[ml_name] = {}

                    if shap_kernel is not None:
                        log_print("     Overlap composition vs KernelSHAP:")

                        overlap_sign_agr = {}   # top_k -> sign agreement on the overlap

                        for top_k in top_k_list:
                            ov_set, n_ov, pct_ov = compute_topk_overlap(
                                composed_imp, shap_kernel, feature_cols, top_k
                            )
                            overlap_summary[ml_name][top_k] = (n_ov, pct_ov)

                            ov_indices = np.where(np.isin(feature_cols, list(ov_set)))[0]

                            pct_agr_ov, n_agr_ov, n_val_ov, _ = compute_sign_agreement_subset(
                                signed, shap_kernel, ov_indices
                            )
                            overlap_sign_agr[top_k] = pct_agr_ov

                            log_print(
                                f"       top-{top_k:2d}: {n_ov:2d}/{top_k} ({pct_ov:.0%})"
                                + (f"  → {', '.join(sorted(shorten_taxon(n, 1) for n in ov_set))}"
                                if n_ov > 0 and top_k == min(top_k_list) else "")
                            )
                            log_print(
                                f"               Sign agreement in the top-{top_k} overlap: "
                                f"{n_agr_ov}/{n_val_ov} ({pct_agr_ov:.1%})"
                            )

                        # ---- Global sign agreement (features with non-zero sign in both methods) ----
                        log_print("     Sign agreement (all features, non-zero sign in both methods):")
                        pct_agr, n_agr, n_val, agree_mask = compute_sign_agreement(
                            signed, shap_kernel
                        )
                        sign_agr_summary[ml_name] = (pct_agr, n_agr, n_val)
                        log_print(f"       {n_agr}/{n_val} agreeing features ({pct_agr:.1%})")

                        stability = compute_shap_sign_stability(shap_kernel)
                        valid_mask = (np.sign(signed) != 0) & (np.sign(np.mean(shap_kernel, axis=0)) != 0)
                        log_print(
                            f"       Mean SHAP sign stability (valid features): "
                            f"{stability[valid_mask].mean():.3f}  "
                            f"[0.5=random, 1.0=perfect]"
                        )

                        log_sign_agreement_details(
                            feature_cols, signed, shap_kernel,
                            agree_mask, top_k_list, log_print
                        )
                    else:
                        agree_mask = None
                        for top_k in top_k_list:
                            overlap_summary[ml_name][top_k] = (0, 0.0)

                    # ---- Bar plots composition vs kernel ----
                    if shap_kernel is not None:
                        for top_k in top_k_list:
                            log_print(f"     Plot top-{top_k}: composition vs kernel...")
                            plot_comp_vs_kernel(
                                composed_imp       = composed_imp,
                                signed             = signed,
                                shap_kernel_matrix = shap_kernel,
                                feature_cols       = feature_cols,
                                ml_name            = ml_name,
                                top_k              = top_k,
                                n_samples_dir      = n_samples_dir,
                                fold_name          = fold_name,
                                n_kernel_explain   = n_explained
                            )

                    # ---- Importance CSV ----
                    save_importance_csv(
                        composed_imp       = composed_imp,
                        signed             = signed,
                        feature_cols       = feature_cols,
                        ml_name            = ml_name,
                        n_samples_dir      = n_samples_dir,
                        shap_kernel_matrix = shap_kernel,
                        agree_mask         = agree_mask,
                    )
                    log_print("     Importance CSV saved.")

                    # ---- Global statistics row ----
                    if shap_kernel is not None:
                        _, pct_ov20 = overlap_summary[ml_name].get(20, (0, 0.0))
                        _, pct_ov50 = overlap_summary[ml_name].get(50, (0, 0.0))
                        pct_agr, _, _ = sign_agr_summary.get(ml_name, (0.0, 0, 0))
                        t_sec         = kernel_time_summary.get(ml_name, float('nan'))
                        save_stats_row(
                            n_samples_dir      = n_samples_dir,
                            ml_name            = ml_name,
                            fold_name          = fold_name,
                            n_kernel_nsamples  = n_kernel_nsamples,
                            overlap_pct_top20  = pct_ov20,
                            overlap_pct_top50  = pct_ov50,
                            sign_agr_pct       = pct_agr,
                            sign_agr_overlap20 = overlap_sign_agr.get(20, 0.0),
                            sign_agr_overlap50 = overlap_sign_agr.get(50, 0.0),
                            kernel_time_sec    = t_sec,
                        )
                        log_print("     Statistics row appended to stats_kernelshap.csv.")

                # ---- Final summary ----
                if run_kernel_shap:
                    header_topk = "  ".join(f"Overlap@{k}" for k in top_k_list)
                    log_print(
                        f"\n  {'Model':<30}  {header_topk}  "
                        f"{'Sign Agr.':>12}  {'KernelSHAP time':>18}"
                    )
                    log_print(f"  {'-'*90}")
                    for ml_name in all_classifiers:
                        ov_parts = []
                        for top_k in top_k_list:
                            n_ov, pct_ov = overlap_summary.get(ml_name, {}).get(top_k, (0, 0.0))
                            ov_parts.append(f"{n_ov:>2}/{top_k} ({pct_ov:>4.0%})")
                        ov_str = "  ".join(ov_parts)

                        if ml_name in sign_agr_summary:
                            pct_agr, n_agr, n_val = sign_agr_summary[ml_name]
                            agr_str = f"{n_agr}/{n_val} ({pct_agr:.1%})"
                        else:
                            agr_str = "N/A"

                        if ml_name in kernel_time_summary:
                            t     = kernel_time_summary[ml_name]
                            t_str = f"{t:.1f}s ({t/60:.1f} min)"
                        else:
                            t_str = "N/A"

                        log_print(f"  {ml_name:<30}  {ov_str}  {agr_str:>12}  {t_str:>18}")

            print(f"\n[DONE] Interpretability completed for: {config_name}")
