"""
train_orthrus.py
================
Trains Orthrus, a multi-task linear autoencoder (shared linear encoder, linear
decoder and linear classification head), and the downstream classifiers
(Linear SVM, Logistic Regression, Extra Trees and a majority-vote Ensemble)
on its latent space, for every latent dimension and every outer fold.

Loss: L = 0.5 * MSE(reconstruction) + 0.5 * BCE(classification) + lambda * ||W||_1

Within each outer fold, the scaler, Orthrus and the downstream classifiers are
fitted on the outer training set only; the outer test set is used once, for the
final evaluation.

Usage (from the repository root):
    python Orthrus/train_orthrus.py --cv holdout
    python Orthrus/train_orthrus.py --cv holdout --latent-dims 30 150
    python Orthrus/train_orthrus.py --cv loco    --latent-dims 150
"""

import os
import sys
import argparse

import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report, accuracy_score,
    f1_score, precision_score, recall_score,
)
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")                                   # non-interactive backend, safe on servers
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D                 # needed for 3D plots
import seaborn as sns
sns.set_theme(style="whitegrid")      # same style for all the plots (Figure 2 of the paper included)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, os.path.join(REPO_ROOT, "common"))
import splits as shared_splits                          # same outer splits as the IML pipeline
from models import get_ml_models_and_grids, INNER_CV, SCORING, N_JOBS   # same models/grids as the IML pipeline
from preprocessing import load_dataset, clr_pseudocount
from model import LinearAE
from ensemble import majority_vote                      # majority vote, ties -> Orthrus head

# ==========================================
# Settings and output folders
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--cv", choices=["holdout", "loco"], default="holdout",
                    help="holdout: 5 stratified 80/20 splits, fold i = train_test_split(random_state=42+i) "
                         "as in the paper; loco: Leave-One-Cohort-Out over BioProject")
parser.add_argument("--n-splits", type=int, default=5)
parser.add_argument("--data", default=os.path.join(REPO_ROOT, "datasets", "raw_dataset.csv"))
parser.add_argument("--out-root", default=os.path.join(REPO_ROOT, "results", "orthrus"))
parser.add_argument("--latent-dims", type=int, nargs="+",
                    default=[10, 20, 30, 50, 100, 150, 200, 250, 300, 350, 400])
parser.add_argument("--epochs", type=int, default=1000)
args = parser.parse_args()

base_out_dir = os.path.join(args.out_root, args.cv)     # results/orthrus/<holdout|loco>
os.makedirs(base_out_dir, exist_ok=True)

# ==========================================
# 1. Data loading and preparation (once, outside the loops)
# ==========================================
print("1. Loading and preparing data...")
df_clean, taxa_cols = load_dataset(args.data)
le           = LabelEncoder()
y            = le.fit_transform(df_clean['response'])
orr_values   = df_clean['ORR'].values

X_raw        = df_clean[taxa_cols].values.astype('float32')
feature_cols = pd.Index(taxa_cols)

# CLR: sample-wise transformation with a fixed pseudocount -> no parameter is
# estimated from other samples, so it can be applied before the split.
X_clr        = clr_pseudocount(X_raw, pseudo_count=1e-6)

splits = shared_splits.get_splits(df_clean, mode=args.cv, n_splits=args.n_splits)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"--> Running on device: {device}")


# ==========================================
# Helper: 3D PCA of the latent space
# ==========================================
def plot_pca_3d(model, X_train_tensor, X_test_tensor,
                y_train, y_test, le,
                save_path, device):
    """
    Extracts the current latent space of the model, applies a 3-component PCA
    and saves a 3D scatter plot with train (circles) and test (triangles).
    Style of Figure 2 of the paper: no title, large fonts and legend.
    """
    model.eval()
    with torch.no_grad():
        Z_train = model.encode(X_train_tensor.to(device)).cpu().numpy()
        Z_test  = model.encode(X_test_tensor.to(device)).cpu().numpy()

    # PCA on the whole latent space, fitted on the training set
    n_components = min(3, Z_train.shape[1])
    pca = PCA(n_components=n_components)
    Z_train_pca = pca.fit_transform(Z_train)
    Z_test_pca  = pca.transform(Z_test)

    var_exp = pca.explained_variance_ratio_ * 100  # explained variance (%)

    classes      = le.classes_
    palette      = plt.cm.tab10.colors             # one color per class
    class_colors = {c: palette[i % 10] for i, c in enumerate(classes)}

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')

    for cls_idx, cls_name in enumerate(classes):
        color = class_colors[cls_name]

        # Train - circles
        mask_tr = (y_train == cls_idx)
        if mask_tr.any():
            ax.scatter(Z_train_pca[mask_tr, 0],
                       Z_train_pca[mask_tr, 1],
                       Z_train_pca[mask_tr, 2] if n_components == 3 else np.zeros(mask_tr.sum()),
                       c=[color], marker='o', alpha=0.6, s=30,
                       label=f"{cls_name} – train")

        # Test - triangles
        mask_te = (y_test == cls_idx)
        if mask_te.any():
            ax.scatter(Z_test_pca[mask_te, 0],
                       Z_test_pca[mask_te, 1],
                       Z_test_pca[mask_te, 2] if n_components == 3 else np.zeros(mask_te.sum()),
                       c=[color], marker='^', alpha=0.9, s=50, edgecolors='k', linewidths=0.4,
                       label=f"{cls_name} – test")

    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)", fontsize=16, labelpad=16)
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)", fontsize=16, labelpad=16)
    ax.set_zlabel(f"PC3 ({var_exp[2]:.1f}%)" if n_components == 3 else "PC3 (n/a)",
                  fontsize=16, labelpad=16)
    ax.tick_params(axis='both', labelsize=13)
    ax.set_box_aspect(None, zoom=0.9)        # keeps the PC3 label inside the image
    ax.legend(loc='upper left', fontsize=14, ncol=2, markerscale=1.5)
    # bbox_inches='tight' underestimates the 3D axis labels: add some padding
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.4)
    plt.close()


# ==========================================
# Artifact saving
# ==========================================
def save_training_artifacts(exp_dir, model, scaler, le,
                             best_estimators, feature_cols,
                             X_train_scaled, X_test_scaled,
                             X_train_latent, X_test_latent,
                             y_train, y_test,
                             input_dim, latent_dim):
    torch.save(
        {'state_dict': model.state_dict(),
         'input_dim':  input_dim,
         'latent_dim': latent_dim},
        os.path.join(exp_dir, 'model_state.pt')
    )
    joblib.dump(scaler,           os.path.join(exp_dir, 'scaler.joblib'))
    joblib.dump(le,               os.path.join(exp_dir, 'label_encoder.joblib'))
    joblib.dump(best_estimators,  os.path.join(exp_dir, 'best_estimators.joblib'))
    np.savez_compressed(
        os.path.join(exp_dir, 'arrays.npz'),
        X_train_scaled  = X_train_scaled,
        X_test_scaled   = X_test_scaled,
        X_train_latent  = X_train_latent,
        X_test_latent   = X_test_latent,
        y_train         = y_train,
        y_test          = y_test,
    )
    np.save(
        os.path.join(exp_dir, 'feature_cols.npy'),
        np.array(feature_cols, dtype=object)
    )
    print(f"  [SAVE] Artifacts saved in: {exp_dir}")


# ==========================================
# Settings
# ==========================================
HEAD_NAME = "Neural Network"     # Orthrus classification head (name used in the output files)
W_REC, W_CLS = 0.5, 0.5          # weights of the reconstruction and classification losses

# names in results.csv (as in the paper: "SVM" for Linear SVM)
RESULTS_NAME = {"Linear SVM": "SVM"}
ORR_CATEGORIES = ['CR', 'Dead', 'PD', 'PR']
avg_mode = 'macro'


def compute_metrics(y_true, y_pred):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred, average=avg_mode, zero_division=0),
        "precision": precision_score(y_true, y_pred, average=avg_mode, zero_division=0),
        "recall":    recall_score(y_true, y_pred, average=avg_mode, zero_division=0),   # = balanced accuracy
    }


# ==========================================
# Main loop
# ==========================================
for which_latent_dim in args.latent_dims:

    scores              = {}   # model -> metric -> list over folds
    subgroup_accuracies = {}   # model -> ORR category -> list over folds (hold-out only)

    seed = 42

    for i, split in enumerate(splits):
        np.random.seed(seed+i)
        torch.manual_seed(seed+i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed+i)

        # Outer split shared with the IML pipeline (common/splits.py)
        tr_pos = shared_splits.positions(df_clean, split["train_ids"])
        te_pos = shared_splits.positions(df_clean, split["test_ids"])
        X_train_clr, X_test_clr = X_clr[tr_pos], X_clr[te_pos]
        y_train,     y_test     = y[tr_pos],     y[te_pos]
        orr_train,   orr_test   = orr_values[tr_pos], orr_values[te_pos]

        # Scaling (fitted on the training set only)
        scaler         = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_clr)
        X_test_scaled  = scaler.transform(X_test_clr)

        # Tensors and DataLoaders
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train,        dtype=torch.float32).view(-1, 1)
        X_test_tensor  = torch.tensor(X_test_scaled,  dtype=torch.float32)
        y_test_tensor  = torch.tensor(y_test,         dtype=torch.float32).view(-1, 1)

        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                                  batch_size=32, shuffle=True)
        test_loader  = DataLoader(TensorDataset(X_test_tensor,  y_test_tensor),
                                  batch_size=32, shuffle=False)

        input_dim = X_train_scaled.shape[1]

        # ML models and hyperparameter grids (common/models.py, shared with the IML pipeline)
        ml_models_and_grids = get_ml_models_and_grids(random_state=seed+i)

        # ==========================================
        # 2. Training
        # ==========================================
        epochs    = args.epochs
        lr        = 1e-3
        l1_lambda = 1e-5

        criterion_reconstruction = nn.MSELoss()
        criterion_classification = nn.BCEWithLogitsLoss()
        global_results = []

        config_name = f"dim_{which_latent_dim}__wd_{l1_lambda}__lr_{lr}/{split['name']}"
        exp_dir     = os.path.join(base_out_dir, config_name)
        os.makedirs(exp_dir, exist_ok=True)
        shared_splits.save_test_ids(exp_dir, split)

        plot_dir = os.path.join(exp_dir, 'Plot')
        os.makedirs(plot_dir, exist_ok=True)

        log_file_path = os.path.join(exp_dir, "experiment_log.txt")

        # Epochs at which the 3D PCA of the latent space is saved
        pca_checkpoints = {
            0:            "epoch_001",
            epochs // 2:  f"epoch_{epochs // 2 + 1:03d}",
            epochs - 1:   f"epoch_{epochs:03d}",
        }

        with open(log_file_path, "w", encoding="utf-8") as log_file:
            def log_print(message):
                print(message)
                log_file.write(message + "\n")

            log_print(f"\n{'='*60}")
            log_print(f"Run: {config_name}")
            log_print(f"{'='*60}")

            test_predictions = {}
            w_rec, w_cls = W_REC, W_CLS

            def multitask_loss(rec, x, pred, yb):
                # L = 0.5 * MSE + 0.5 * BCE + lambda * L1 norm of all the weight matrices
                loss_recon = criterion_reconstruction(rec, x)
                loss_class = criterion_classification(pred, yb)
                l1_penalty = sum(p.abs().sum() for name, p in model.named_parameters() if 'weight' in name)
                return (w_rec * loss_recon) + (w_cls * loss_class) + (l1_lambda * l1_penalty)

            model     = LinearAE(input_dim, which_latent_dim).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # PCA of the latent space before training
            plot_pca_3d(
                model          = model,
                X_train_tensor = X_train_tensor,
                X_test_tensor  = X_test_tensor,
                y_train        = y_train,
                y_test         = y_test,
                le             = le,
                save_path      = os.path.join(plot_dir, "pca_untrained.png"),
                device         = device,
            )

            train_losses, test_losses = [], []

            for epoch in range(epochs):
                model.train()
                running_train_loss = 0.0
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    reconstructed_x, pred_y = model(batch_x)
                    total_loss = multitask_loss(reconstructed_x, batch_x, pred_y, batch_y)
                    total_loss.backward()
                    optimizer.step()
                    running_train_loss += total_loss.item()

                avg_train_loss = running_train_loss / len(train_loader)
                train_losses.append(avg_train_loss)

                # test loss: monitoring only, never used for any decision
                model.eval()
                running_test_loss = 0.0
                with torch.no_grad():
                    for batch_x_t, batch_y_t in test_loader:
                        batch_x_t, batch_y_t = batch_x_t.to(device), batch_y_t.to(device)
                        rec_t, pred_t = model(batch_x_t)
                        running_test_loss += multitask_loss(rec_t, batch_x_t, pred_t, batch_y_t).item()

                avg_test_loss = running_test_loss / len(test_loader)
                test_losses.append(avg_test_loss)

                if (epoch + 1) % 50 == 0 or epoch == 0:
                    log_print(f"  Epoch [{epoch+1:3d}/{epochs}] | "
                               f"Train Loss: {avg_train_loss:.4f} | "
                               f"Test Loss: {avg_test_loss:.4f}")

                if epoch in pca_checkpoints:
                    tag       = pca_checkpoints[epoch]
                    pca_path  = os.path.join(plot_dir, f"pca_{tag}.png")
                    plot_pca_3d(
                        model          = model,
                        X_train_tensor = X_train_tensor,
                        X_test_tensor  = X_test_tensor,
                        y_train        = y_train,
                        y_test         = y_test,
                        le             = le,
                        save_path      = pca_path,
                        device         = device,
                    )
                    log_print(f"  [PCA] 3D PCA saved: {pca_path}")

            # Loss plot
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='blue')
            plt.plot(range(1, epochs + 1), test_losses,  label='Test Loss',  color='red')
            plt.title(f'Multi-Task Loss: {config_name}')
            plt.xlabel('Epochs')
            plt.ylabel(f'Loss ({w_rec}*Recon + {w_cls}*Class + L1)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(plot_dir, 'loss.png'), bbox_inches='tight')
            plt.close()

            # Orthrus classification head and latent representations
            model.eval()
            with torch.no_grad():
                _, pred_logits = model(X_test_tensor.to(device))
                X_train_latent = model.encode(X_train_tensor.to(device)).cpu().numpy()
                X_test_latent  = model.encode(X_test_tensor.to(device)).cpu().numpy()
            test_predictions[HEAD_NAME] = (torch.sigmoid(pred_logits).cpu().numpy() >= 0.5
                                           ).astype(int).flatten()

            # ==========================================
            # 3. Classical ML models on the reduced space
            # ==========================================
            log_print(f"\n  [ML] Training classifiers...")
            best_estimators = {}
            all_test_preds  = []

            for ml_name, ml_config in ml_models_and_grids.items():
                grid = GridSearchCV(ml_config["model"], ml_config["params"],
                                    cv=INNER_CV, scoring=SCORING, n_jobs=N_JOBS)
                grid.fit(X_train_latent, y_train)
                best_model = grid.best_estimator_
                best_estimators[ml_name] = best_model

                y_pred = best_model.predict(X_test_latent)
                all_test_preds.append(y_pred)
                test_predictions[ml_name] = y_pred

            # ==========================================
            # 4. Ensemble: majority vote of the 3 classifiers and of the Orthrus head.
            #    A 2-2 tie is resolved by the Orthrus head.
            # ==========================================
            head_preds = test_predictions[HEAD_NAME]
            voters = all_test_preds + [head_preds]
            test_predictions["Ensemble"] = majority_vote(voters, tiebreak=head_preds)

            # ---- Metrics per model ----
            local_results = []
            for m_name, preds in test_predictions.items():
                met = compute_metrics(y_test, preds)
                for k_m, v_m in met.items():
                    scores.setdefault(m_name, {}).setdefault(k_m, []).append(v_m)
                local_results.append({'Model': m_name, 'Accuracy': met["accuracy"]})
                global_results.append({'ML_Model': m_name, 'Accuracy_Test': met["accuracy"],
                                       'F1': met["f1"], 'Precision': met["precision"], 'Recall': met["recall"]})
                log_print(f"\n    --- {m_name} ---")
                log_print(f"    Accuracy: {met['accuracy']:.4f} | F1: {met['f1']:.4f} | "
                          f"Precision: {met['precision']:.4f} | Recall: {met['recall']:.4f}")
                log_print(classification_report(y_test, preds, target_names=le.classes_, zero_division=0))

            # ---- Accuracy per ORR subgroup (hold-out only; LOCO is Responder vs Non-Responder only) ----
            if args.cv == "holdout":
                log_print("\n  [ORR EVALUATION] Accuracy per ORR subgroup...")
                orr_fold_rows = []
                for cat in ORR_CATEGORIES:
                    idx_cat = np.where(orr_test == cat)[0]
                    if len(idx_cat) > 0:
                        log_print(f"    - Category {cat}: {len(idx_cat)} samples.")
                        for model_name, preds in test_predictions.items():
                            acc_sub = accuracy_score(y_test[idx_cat], preds[idx_cat])
                            subgroup_accuracies.setdefault(model_name, {}).setdefault(cat, []).append(acc_sub)
                            orr_fold_rows.append({"fold": split["name"], "model": model_name, "category": cat,
                                                  "n_samples": len(idx_cat), "accuracy": round(acc_sub, 4)})
                    else:
                        log_print(f"    - Category {cat}: no samples in this test split.")
                orr_fold_path = os.path.join(exp_dir, "orr_fold.csv")
                pd.DataFrame(orr_fold_rows).to_csv(orr_fold_path, index=False)
                log_print(f"\n  [ORR] Per-fold accuracy saved in: {orr_fold_path}")

            # Test predictions of the fold
            pd.DataFrame({
                "samples":    df_clean['samples'].values[te_pos],
                "BioProject": df_clean['BioProject'].values[te_pos],
                "y_true":     y_test,
                **{f"pred_{m.replace(' ', '_')}": p for m, p in test_predictions.items()},
            }).to_csv(os.path.join(exp_dir, "predictions.csv"), index=False)

            # Accuracy plot
            custom_colors = {
                'Neural Network': 'orange', 'Linear SVM': 'red',
                'Logistic Regression': 'green', 'ExtraTrees': 'blue',
                'Ensemble': 'brown'
            }
            df_local = pd.DataFrame(local_results).sort_values('Accuracy', ascending=False)
            plt.figure(figsize=(8, 6))
            bp = sns.barplot(x='Accuracy', y='Model', data=df_local,
                             palette=custom_colors, hue='Model', legend=False)
            for p in bp.patches:
                w = p.get_width()
                plt.text(w + 0.01, p.get_y() + p.get_height() / 2, f'{w:.4f}', va='center')
            plt.title(f'Test accuracy\n({config_name})')
            plt.xlim(0, 1.1)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'models_accuracy.png'), dpi=300)
            plt.close()

            # ==========================================
            # 5. Artifact saving (read by explain_orthrus.py)
            # ==========================================
            save_training_artifacts(
                exp_dir         = exp_dir,
                model           = model,
                scaler          = scaler,
                le              = le,
                best_estimators = best_estimators,
                feature_cols    = feature_cols,
                X_train_scaled  = X_train_scaled,
                X_test_scaled   = X_test_scaled,
                X_train_latent  = X_train_latent,
                X_test_latent   = X_test_latent,
                y_train         = y_train,
                y_test          = y_test,
                input_dim       = input_dim,
                latent_dim      = which_latent_dim,
            )

        # ==========================================
        # Fold report
        # ==========================================
        csv_path  = os.path.join(exp_dir, "metrics_test.csv")
        df_global = pd.DataFrame(global_results).sort_values('Accuracy_Test', ascending=False)
        df_global.to_csv(csv_path, index=False)
        print(f"Fold results saved in '{csv_path}'.")

    # ==========================================
    # results.csv: mean and std over folds (std with ddof=1)
    # ==========================================
    results_dir = os.path.join(base_out_dir, f"dim_{which_latent_dim}__wd_{l1_lambda}__lr_{lr}")
    os.makedirs(results_dir, exist_ok=True)

    def _std(v):
        return float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")

    rows = []
    for m_name, met in scores.items():
        rows.append({
            "model":         RESULTS_NAME.get(m_name, m_name),
            "accuracy":      round(np.mean(met["accuracy"]),  4),
            "acc_std":       round(_std(met["accuracy"]),     4),
            "f1":            round(np.mean(met["f1"]),        4),
            "f1_std":        round(_std(met["f1"]),           4),
            "precision":     round(np.mean(met["precision"]), 4),
            "precision_std": round(_std(met["precision"]),    4),
            "recall":        round(np.mean(met["recall"]),    4),
            "recall_std":    round(_std(met["recall"]),       4),
            "n_folds":       len(met["accuracy"]),
        })
    csv_results_path = os.path.join(results_dir, "results.csv")
    pd.DataFrame(rows).to_csv(csv_results_path, index=False)
    print(f"  [RESULTS] Metrics saved in: {csv_results_path}")

    if subgroup_accuracies:
        rows = []
        for model_name, cats in subgroup_accuracies.items():
            row = {"model": model_name}
            for cat in ORR_CATEGORIES:
                acc_list = cats.get(cat, [])
                row[f"{cat}_accuracy"] = round(np.mean(acc_list), 4) if acc_list else np.nan
                row[f"{cat}_std"]      = round(_std(acc_list), 4)    if acc_list else np.nan
            rows.append(row)
        pd.DataFrame(rows).set_index("model").to_csv(os.path.join(results_dir, "metrics_orr.csv"), index=True)
