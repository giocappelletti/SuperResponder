import sys
import os

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import statistics

from sklearn.calibration import LabelEncoder

from fileio import dataloader, serializer
from visualization import plotter
from analysis import SHAPExplainer
from models import Models


def explainability():
    
    dataset_path = os.path.join(project_root, 'datasets', 'table_species_reads.tsv')

    dataset, taxa_cols, meta_cols = dataloader.load_dataset(dataset_path, False)
    
    models = Models(serializer, plotter)
    labelenc = LabelEncoder()
    
    dataset_taxa = dataset[taxa_cols] # Use entire dataset since we employ nested cross validation
    
    dataset_labels = labelenc.fit_transform(dataset['response']) # O - Non responder, 1 - Responder

    top_n_feats_list = [10, 20, 30, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 4630]
    accuracies = {}
    means = {}
    stds = {}
    features_matrix = {}


    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for n in top_n_feats_list:
        # Nested cross validation
        fold_results = []
        top_feat_list = []
        all_accuracies = []
        print(f"LabelEncoder mapping: {list(labelenc.classes_)} -> {list(range(len(labelenc.classes_)))}")

        print(f"TOP {n} FEATURES")

        for _, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(dataset_taxa, dataset_labels)):
            outer_train, outer_test = dataset_taxa.iloc[outer_train_idx], dataset.iloc[outer_test_idx]
            outer_train_labels, outer_test_labels = dataset_labels[outer_train_idx], dataset_labels[outer_test_idx]

            # Fit on outer train set 
            best_model_pipeline, _, = models.evaluate_classifier(outer_train, # Don't use metadata
                                                                outer_train_labels,
                                                                skip_cv=True)
            
            best_model = best_model_pipeline.named_steps['classifier'] # Get fitted model instance
            
            explainer = SHAPExplainer(plotter, best_model)
            
            shap_vals = explainer.compute_shap_values(outer_test[taxa_cols], visualize = False, extract_index = 1)

            top_feat = explainer.get_top_features(shap_vals, outer_test[taxa_cols].columns, head=n) # Get the n top features list

            top_feat_list.append(top_feat)
            
            top_feat_train = outer_train[top_feat] # Select only n columns

            best_model.fit(top_feat_train, outer_train_labels)

            complete_resp = outer_test[outer_test['ORR'] == 'CR']
            partial_resp = outer_test[outer_test['ORR'] == 'PR']
            partial_dis = outer_test[outer_test['ORR'] == 'PD']
            dead = outer_test[outer_test['ORR'] == 'Dead']
            meta_complete = complete_resp[meta_cols]
            meta_partial_resp = partial_resp[meta_cols]
            meta_partial_dis = partial_dis[meta_cols]
            meta_dead = dead[meta_cols]

            subsets = {
                'CR': pd.concat([meta_complete, complete_resp[top_feat]], axis=1),
                'PR': pd.concat([meta_partial_resp, partial_resp[top_feat]], axis=1),
                'PD': pd.concat([meta_partial_dis, partial_dis[top_feat]], axis=1),
                'Dead': pd.concat([meta_dead, dead[top_feat]], axis=1)
            }

            accuracies_dict = {}

            for name, subset_df in subsets.items():
                if not subset_df.empty:
                    if subset_df.shape[0] > 1:
                        subset_features = subset_df[top_feat]
                        subset_labels_raw = subset_df['response']

                        # Use the same LabelEncoder to transform test set labels
                        subset_labels_encoded = labelenc.transform(subset_labels_raw)
                        y_pred_subset = best_model.predict(subset_features)

                        accuracies_dict[name] = accuracy_score(subset_labels_encoded, y_pred_subset)
                        if name == "Dead":
                            print("DEAD PREDICTIONS")
                            print(y_pred_subset)

                else:
                    accuracies_dict[name] = None

            all_accuracies.append(accuracies_dict)

       
            y_pred = best_model.predict(outer_test[top_feat])
            fold_results.append(accuracy_score(y_true=outer_test_labels, y_pred=y_pred))
        
        
        CR_acc = [acc_dict.get('CR', None) for acc_dict in all_accuracies]
        PR_acc = [acc_dict.get('PR', None) for acc_dict in all_accuracies]
        PD_acc = [acc_dict.get('PD', None) for acc_dict in all_accuracies]
        Dead_acc = [acc_dict.get('Dead', None) for acc_dict in all_accuracies]

        CR_acc = [x for x in CR_acc if x is not None]
        PR_acc = [x for x in PR_acc if x is not None]
        PD_acc = [x for x in PD_acc if x is not None]
        Dead_acc = [x for x in Dead_acc if x is not None]

        if CR_acc:
            mean_cr_acc = statistics.fmean(CR_acc)
        if PR_acc:
            mean_pr_acc = statistics.fmean(PR_acc)
        if PD_acc:
            mean_pd_acc = statistics.fmean(PD_acc)
        if Dead_acc:
            mean_dead_acc = statistics.fmean(Dead_acc)
        else:
            mean_dead_acc = "NO DEADS"

        accuracies_dict= {
            'CR': mean_cr_acc,
            'PR': mean_pr_acc,
            'PD': mean_pd_acc,
            'Dead': mean_dead_acc
        }
        print(accuracies_dict)
        accuracies[n] = accuracies_dict
        
        means[n] = pd.DataFrame(fold_results).mean().to_dict()
        stds[n] = pd.DataFrame(fold_results).std().to_dict()
        features_matrix[n] = top_feat_list

    
    mean_df = pd.DataFrame(means)
    std_df = pd.DataFrame(stds)
    accuracies_df = pd.DataFrame(accuracies)
    top_feat_df = pd.DataFrame(features_matrix)
    
    metrics_df = pd.concat([mean_df, std_df, accuracies_df], axis=0)

    metrics_df.to_excel("features_ablation_study_raw.xlsx", float_format="%.4f", sheet_name="Feature Ablation Study")
    top_feat_df.to_csv("top_features.csv")


if __name__ == "__main__":
    explainability()
