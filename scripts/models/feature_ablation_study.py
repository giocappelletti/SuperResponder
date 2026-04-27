import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import statistics
import pandas as pd
import numpy as np

from sklearn.calibration import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import clone

from fileio import dataloader, serializer
from visualization import plotter
from analysis import SHAPExplainer
from models import Models
from logger import logger


def explainability():
    
    # Build path
    dataset_path = os.path.join(project_root, 'datasets', 'raw_dataset.csv')

    # Get dataframe loaded in memory, avoid dropping the "response" column
    dataset, taxa_cols, _ = dataloader.load_dataset(dataset_path, drop_response = False)
    
    # Instantiate repo classes and inject dependencies
    models = Models(serializer, plotter)

    # Needed to encode categorical labels (Responder/Non responder) as 0/1
    labelenc = LabelEncoder()
    
    dataset_taxa = dataset[taxa_cols] # Use entire dataset since we employ nested cross validation
    
    dataset_labels = labelenc.fit_transform(dataset['response']) # O - Non responder, 1 - Responder

    # Select the top N features retuned by SHAP
    top_n_feats_list = [10, 20, 30, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 4630]
    
    # Results data structures
    accuracies_per_n = {n: [] for n in top_n_feats_list}
    fold_results_per_n = {n: [] for n in top_n_feats_list}
    random_fold_res_per_n = {n: [] for n in top_n_feats_list}
    rand_acc_per_n = {n: [] for n in top_n_feats_list}
    features_matrix_per_n = {n: [] for n in top_n_feats_list}

    np.random.seed(42)
    
    n_splits = 20
    
    outer_cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
    
    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(dataset_taxa, dataset_labels)):
        logger.info(f"FOLD {fold_idx + 1}/{n_splits}")
        outer_train, outer_test = dataset_taxa.iloc[outer_train_idx], dataset.iloc[outer_test_idx]
        outer_train_labels, outer_test_labels = dataset_labels[outer_train_idx], dataset_labels[outer_test_idx]

        # Fit on outer train set to find best hyperparameters via Grid Search
        best_model_pipeline, _ = models.evaluate_classifier(
            outer_train, 
            outer_train_labels,
            skip_cv = True
        )
        
        # Get best model found by grid search and preprocessing object
        best_model = best_model_pipeline.named_steps['classifier']
        scaler = best_model_pipeline.named_steps.get('smartscaler')
        
        # Scale data. This step maybe is redundant since sklearn should handle the Complete Pipeline 
        if scaler is not None:
            scaled_outer_train = pd.DataFrame(scaler.transform(outer_train), columns=outer_train.columns, index=outer_train.index)
            scaled_outer_test = pd.DataFrame(scaler.transform(outer_test[taxa_cols]), columns=taxa_cols, index=outer_test.index)
        else:
            scaled_outer_train = outer_train.copy()
            scaled_outer_test = outer_test[taxa_cols].copy()
        
        explainer = SHAPExplainer(plotter, best_model, scaled_outer_train) 
        shap_vals = explainer.compute_shap_values(scaled_outer_test, visualize=False, extract_index=1)
        
        # Get all top features from SHAP in the current fold only once
        all_top_feat = explainer.get_top_features(shap_vals, scaled_outer_test.columns, head=len(taxa_cols))

        for n in top_n_feats_list:
            # Get top N features and random N features
            top_feat = all_top_feat[:n]

            random_feats = np.random.choice(range(4630), size=n, replace=False).tolist()
            random_cols = scaled_outer_train.columns[random_feats]

            features_matrix_per_n[n].append(top_feat)
            
            # Clone the classifier to avoid data leakages
            retrained_model = clone(best_model)
            random_model = clone(best_model)
            
            retrained_model.fit(scaled_outer_train[top_feat], outer_train_labels)
            y_pred = retrained_model.predict(scaled_outer_test[top_feat])
            
            random_model.fit(scaled_outer_train[random_cols], outer_train_labels)
            y_pred_random = random_model.predict(scaled_outer_test[random_cols])

            fold_results_per_n[n].append(accuracy_score(y_true=outer_test_labels, y_pred=y_pred))
            random_fold_res_per_n[n].append(accuracy_score(y_true=outer_test_labels, y_pred=y_pred_random))

            # Analyze ORR subgroups
            subsets = {
                'CR': outer_test[outer_test['ORR'] == 'CR'],
                'PR': outer_test[outer_test['ORR'] == 'PR'],
                'PD': outer_test[outer_test['ORR'] == 'PD'],
                'Dead': outer_test[outer_test['ORR'] == 'Dead']
            }
            
            accuracies_dict = {}
            rand_acc_dict = {}
            for name, subset_df in subsets.items():
                if subset_df.shape[0] > 0:
                    subset_features = scaled_outer_test.loc[subset_df.index, top_feat]
                    subset_rand_feat = scaled_outer_test.loc[subset_df.index, random_cols]
                    subset_labels_encoded = labelenc.transform(subset_df['response'])
                    y_pred_subset = retrained_model.predict(subset_features)
                    y_pred_rand_sub = random_model.predict(subset_rand_feat)
                    accuracies_dict[name] = accuracy_score(subset_labels_encoded, y_pred_subset)
                    rand_acc_dict[name] = accuracy_score(subset_labels_encoded, y_pred_rand_sub)
                else:
                    accuracies_dict[name] = None
                    rand_acc_dict[name] = None
            
            accuracies_per_n[n].append(accuracies_dict)
            rand_acc_per_n[n].append(rand_acc_dict)
            

    # Aggregate all results in a dataframe
    final_means = {}
    final_stds = {}
    random_means = {}
    random_stds = {}
    final_accuracies = {}
    random_fin_accs = {}

    for n in top_n_feats_list:
        final_means[n] = pd.Series(fold_results_per_n[n]).mean()
        final_stds[n] = pd.Series(fold_results_per_n[n]).std()
        random_means[n] = pd.Series(random_fold_res_per_n[n]).mean()
        random_stds[n] = pd.Series(random_fold_res_per_n[n]).std()
        
        cr_accs = [acc['CR'] for acc in accuracies_per_n[n] if acc['CR'] is not None]
        pr_accs = [acc['PR'] for acc in accuracies_per_n[n] if acc['PR'] is not None]
        pd_accs = [acc['PD'] for acc in accuracies_per_n[n] if acc['PD'] is not None]
        dead_accs = [acc['Dead'] for acc in accuracies_per_n[n] if acc['Dead'] is not None]

        rand_cr_accs = [acc['CR'] for acc in rand_acc_per_n[n] if acc['CR'] is not None]
        rand_pr_accs = [acc['PR'] for acc in rand_acc_per_n[n] if acc['PR'] is not None]
        rand_pd_accs = [acc['PD'] for acc in rand_acc_per_n[n] if acc['PD'] is not None]
        rand_dead_accs = [acc['Dead'] for acc in rand_acc_per_n[n] if acc['Dead'] is not None]
        
        final_accuracies[n] = {
            'CR': [statistics.fmean(cr_accs), statistics.stdev(cr_accs)] if cr_accs else "NO CR",
            'PR': [statistics.fmean(pr_accs), statistics.stdev(pr_accs)] if pr_accs else "NO PR",
            'PD': [statistics.fmean(pd_accs), statistics.stdev(pd_accs)] if pd_accs else "NO PD",
            'Dead': [statistics.fmean(dead_accs), statistics.stdev(dead_accs)] if dead_accs else "NO DEADS"
        }

        random_fin_accs[n] = {
            'CR': [statistics.fmean(rand_cr_accs), statistics.stdev(rand_cr_accs)] if rand_cr_accs else "NO CR",
            'PR': [statistics.fmean(rand_pr_accs), statistics.stdev(rand_pr_accs)] if rand_pr_accs else "NO PR",
            'PD': [statistics.fmean(rand_pd_accs), statistics.stdev(rand_pd_accs)] if rand_pd_accs else "NO PD",
            'Dead': [statistics.fmean(rand_dead_accs), statistics.stdev(rand_dead_accs)] if rand_dead_accs else "NO DEADS"
        }

        logger.info(f"Aggregated metrics for N={n}: {final_accuracies[n]}")

    mean_df = pd.DataFrame([final_means], index=['Mean Accuracy'])
    std_df = pd.DataFrame([final_stds], index=['Std Accuracy'])
    accuracies_df = pd.DataFrame(final_accuracies)
    random_df = pd.DataFrame([random_means], index=['Random Mean Accuracy'])
    random_std_df = pd.DataFrame([random_stds], index=['Random Std Accuracy'])
    random_fin_df = pd.DataFrame(random_fin_accs)
    
    metrics_df = pd.concat([mean_df, std_df, accuracies_df, random_df, random_std_df, random_fin_df], axis=0)

    serializer.save_file(data = metrics_df,
                         subfolder="features_ablation_study",
                         exp_type="feat_study", 
                         exp_group="feature_ablation_study",
                         save_format="xlsx")

if __name__ == "__main__":
    explainability()
