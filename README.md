# Machine and Deep Learning Prediction of Immunotherapy Response from Gut Microbiome Profiles

Code of the paper:

> **Toward Precision Oncology: Machine and Deep Learning Prediction of Immunotherapy Response from Gut Microbiome Profiles**
> *Manuscript under review.* <!-- add journal, year and DOI after publication -->

The repository implements the two complementary approaches proposed in the paper to predict the response
to immune checkpoint blockade (Responder vs Non-Responder) from 4,630 species-level gut microbiome features:

- **IML** (Interpretable Machine Learning): Logistic Regression, Linear SVM and Extra Trees on the
  CLR-transformed taxa, with SHAP-driven feature selection.
- **Orthrus**: a multi-task linear autoencoder whose latent space is optimized jointly for reconstruction
  and classification, used as input for the same classifiers, with an exact closed-form attribution of the
  predictions to the input taxa.

Both pipelines are evaluated with nested cross-validation on the same outer splits (hold-out) and with
leave-one-cohort-out (LOCO) validation over the nine cohorts.

## Authors

- Matteo Lombardi (matteo.lombardi@unimore.it)
- Giovanni Cappelletti (giovanni.cappelletti@unimore.it)

## Repository structure

```
.
├── run_experiments.py         # runs the IML and/or Orthrus experiments of the paper
├── common/                    # shared by the two pipelines
│   ├── preprocessing.py       # data loading, CLR transformation and scaling
│   ├── splits.py              # outer splits: hold-out and leave-one-cohort-out
│   └── models.py              # classifiers (Linear SVM, LR, Extra Trees) and hyperparameter grids
├── IML/
│   ├── run_iml.py             # nested CV, out-of-fold SHAP ranking, SHAP vs random feature subsets
│   └── shap_explainer.py      # SHAP values (LinearExplainer / TreeExplainer)
├── Orthrus/
│   ├── model.py               # Orthrus: linear encoder + decoder + classification head
│   ├── train_orthrus.py       # training of Orthrus and of the downstream classifiers
│   ├── ensemble.py            # majority vote (2-2 ties resolved by the Orthrus head)
│   └── explain_orthrus.py     # taxon importance by composition W_enc^T w (+ optional KernelSHAP comparison)
├── datasets/                  # put raw_dataset.csv here (not versioned)
└── results/                   # created by the scripts (not versioned)
```

## Installation

**Python 3.12** is required.

```bash
git clone https://github.com/<user>/<repository>.git
cd <repository>
conda create -n orthrus python=3.12 -y
conda activate orthrus
pip install -r requirements.txt
```

For a GPU build of PyTorch, install `torch` following https://pytorch.org before `requirements.txt`.

## Data

The harmonized dataset (569 patients from nine public cohorts, 4,630 taxa) is available on request:
https://drive.google.com/file/d/1pTqmrwlL63eZ5uwnWHAa9V0f3pa5MduM/view?usp=drive_link
(please send a request with your name, affiliation and research interest).

Copy the file to `datasets/raw_dataset.csv`. Each row is a patient; the scripts use the columns `samples`,
`BioProject` (cohort), `response` (R / NR), `ORR` (CR, PR, PD, Dead) and the taxa columns, whose names
start with `k__`. Taxa abundances are already thresholded (values below 10 set to zero) and normalized
to a total sum of 100 (TSS).

## Methods in brief

**Preprocessing** (`common/preprocessing.py`). CLR transformation with a fixed pseudocount (1e-6), followed by
standard scaling. The CLR acts on each sample independently with fixed constants, so it cannot propagate
information between training and test data; the scaler, the only data-dependent step, is always fitted on
training data only. In the IML pipeline the CLR + scaling step is part of the tuned pipeline (refit in every
inner fold); in Orthrus the scaler is fitted on the outer training set.

**Evaluation** (`common/splits.py`).
- *Hold-out*: 5 repetitions; fold *i* = `train_test_split(test_size=0.2, random_state=42+i, stratify=response)`.
- *LOCO*: in each of nine folds one cohort (BioProject) is the test set and the models are trained on
  the other eight; balanced accuracy (macro recall) is reported for each held-out cohort.

Within each outer training set, hyperparameters are selected with `GridSearchCV(cv=5, scoring='accuracy')`
(stratified 5-fold inner CV) and the selected model is refit on the whole outer training set. The outer test
set is used only once, for the final evaluation.

| Model | Estimator | Grid |
|---|---|---|
| Linear SVM | `LinearSVC(dual=False, max_iter=10000)` | C ∈ {0.001, 0.01, 0.1, 1, 10, 100}, class_weight ∈ {None, balanced} |
| Logistic Regression | `LogisticRegression(solver='liblinear', max_iter=10000)` | C ∈ {0.001, …, 100}, penalty ∈ {l1, l2}, class_weight ∈ {None, balanced} |
| Extra Trees | `ExtraTreesClassifier()` | n_estimators ∈ {50, 100, 200, 500}, max_depth ∈ {5, 10, 20, None}, min_samples_split ∈ {2, 5, 10}, class_weight ∈ {None, balanced} |

**IML** (`IML/run_iml.py`). For each outer fold: (1) grid search on all taxa; (2) out-of-fold SHAP ranking:
on the same five inner folds, the tuned model is refit on four folds and SHAP values are computed on the
remaining one, and taxa are ranked by mean |SHAP|; (3) the model is retrained on the top-*n* taxa
(*n* = 10, …, 500) and, as a baseline, on *n* random taxa, re-tuning the hyperparameters for every subset;
(4) evaluation on the outer test set.

**Orthrus** (`Orthrus/train_orthrus.py`). Linear encoder `z = W_enc x`, linear decoder and linear
classification head, trained with `L = 0.5·MSE + 0.5·BCE + λ‖W‖₁` (Adam, learning rate 1e-3, λ = 1e-5,
batch size 32, 1,000 epochs). The encoder then projects training and test data into the latent space,
where Linear SVM, Logistic Regression and Extra Trees are tuned and trained; their predictions and that of
the Orthrus head are combined by majority vote (Ensemble). Latent dimensions 10–400 are evaluated.

**Interpretability of Orthrus** (`Orthrus/explain_orthrus.py`). For the linear classifiers and the Orthrus
head the importance of each taxon is `I = W_encᵀ w`; for Extra Trees the Gini importances of the latent
dimensions are signed by their correlation with the response on the training set. Optionally, the
attributions are compared with end-to-end KernelSHAP.

## Running the experiments

All commands are run from the repository root.

### Everything

```bash
python run_experiments.py                          # IML and Orthrus, hold-out and LOCO
python run_experiments.py --pipeline iml           # IML only
python run_experiments.py --pipeline orthrus       # Orthrus only
python run_experiments.py --pipeline orthrus --kernel-shap   # + KernelSHAP comparison (slow, hours)
python run_experiments.py --dry-run                # print the commands without running them
```

Completed steps are skipped, so an interrupted run can be resumed (`--no-skip` re-runs everything).
The output of each step is also saved in `results/logs/`.

### IML pipeline

```bash
# Logistic Regression, hold-out (5 outer folds)
python IML/run_iml.py --cv holdout --model LR

# Linear SVM and Extra Trees, leave-one-cohort-out
python IML/run_iml.py --cv loco --model SVM
python IML/run_iml.py --cv loco --model ET

# only some subset sizes and folds (e.g. for a quick test)
python IML/run_iml.py --cv holdout --model LR --top-n 10 50 4630 --folds fold_1
```

### Orthrus

```bash
# training, hold-out, all latent dimensions (10-400)
python Orthrus/train_orthrus.py --cv holdout

# training, only some latent dimensions; leave-one-cohort-out
python Orthrus/train_orthrus.py --cv holdout --latent-dims 30 150
python Orthrus/train_orthrus.py --cv loco --latent-dims 150

# importance of the taxa (composition W_enc^T w), d = 150
python Orthrus/explain_orthrus.py --cv holdout --latent-dims 150 --no-kernel-shap

# same, with the comparison with KernelSHAP (slow)
python Orthrus/explain_orthrus.py --cv holdout --latent-dims 150
```

## Outputs

| Path | Content |
|---|---|
| `results/iml/<cv>/<model>/summary_metrics.csv` | mean ± std over folds, for each selection (SHAP / random) and subset size |
| `results/iml/holdout/<model>/summary_orr.csv` | accuracy per response category (CR, PR, PD, Dead) |
| `results/iml/<cv>/<model>/<fold>/shap_ranking.csv` | out-of-fold SHAP ranking of the taxa |
| `results/iml/<cv>/<model>/<fold>/best_params.csv` | selected hyperparameters for every subset |
| `results/iml/<cv>/<model>/<fold>/predictions.csv` | test predictions |
| `results/orthrus/<cv>/dim_<d>__wd_1e-05__lr_0.001/results.csv` | Orthrus head, classifiers and Ensemble: mean ± std over folds |
| `results/orthrus/holdout/dim_<d>__wd_1e-05__lr_0.001/metrics_orr.csv` | accuracy per response category |
| `results/orthrus/<cv>/dim_<d>__.../<fold>/` | test metrics and predictions, trained model, loss and accuracy plots |
| `results/orthrus/<cv>/dim_<d>__.../<fold>/Plot/pca_*.png` | 3D PCA of the latent space before training and during training (Figure 2: d = 30, `fold_2`, `pca_untrained.png` and `pca_epoch_1000.png`) |
| `results/orthrus/<cv>/dim_<d>__.../<fold>/xai/composed_only/feature_importance_<model>.csv` | importance of every taxon |

`<cv>` is `holdout` or `loco`; hold-out folds are named `fold_1`…`fold_5`, LOCO folds `loco_<BioProject>`.
Each fold folder also contains `test_samples.csv`, the IDs of its test samples, which are identical in the
two pipelines.

## Citation

If you use this code, please cite the paper (reference to be completed after publication).

## License

CC BY-NC-SA 4.0. See `LICENSE`.
