import sys
import os
import statistics

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

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
    accuracies_per_n = {n: [] for n in top_n_feats_list}
    fold_results_per_n = {n: [] for n in top_n_feats_list}
    features_matrix_per_n = {n: [] for n in top_n_feats_list}


    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(dataset_taxa, dataset_labels)):
        print(f"\n--- FOLD {fold_idx + 1}/5 ---")
        outer_train, outer_test = dataset_taxa.iloc[outer_train_idx], dataset.iloc[outer_test_idx]
        outer_train_labels, outer_test_labels = dataset_labels[outer_train_idx], dataset_labels[outer_test_idx]

        # Fit on outer train set to find best hyperparameters via Grid Search
        best_model_pipeline, _ = models.evaluate_classifier(
            outer_train, 
            outer_train_labels,
            skip_cv=True
        )

        best_model = best_model_pipeline.named_steps['classifier']
        scaler = best_model_pipeline.named_steps.get('smartscaler')
        
        # Trasformazione dei dati (Cruciale per il microbiota, es. TSS o CLR)
        if scaler is not None:
            scaled_outer_train = pd.DataFrame(scaler.transform(outer_train), columns=outer_train.columns, index=outer_train.index)
            scaled_outer_test = pd.DataFrame(scaler.transform(outer_test[taxa_cols]), columns=taxa_cols, index=outer_test.index)
        else:
            scaled_outer_train = outer_train.copy()
            scaled_outer_test = outer_test[taxa_cols].copy()
        
        # Computiamo i valori SHAP ESCLUSIVAMENTE SUL TRAIN SET (Evita il Data Leakage)
        explainer = SHAPExplainer(plotter, best_model)
        shap_vals = explainer.compute_shap_values(scaled_outer_test, visualize=False, extract_index=1)
        
        # Estraiamo l'ordine delle feature dal più importante al meno importante
        # Utilizziamo len(taxa_cols) per ottenere tutte le feature ordinate
        all_top_feat = explainer.get_top_features(shap_vals, scaled_outer_test.columns, head=len(taxa_cols))

        for n in top_n_feats_list:
            # Seleziona le top N feature
            top_feat = all_top_feat[:n]
            features_matrix_per_n[n].append(top_feat)
            
            # Clona il classificatore per avere un modello intatto con i best parameters
            retrained_model = clone(best_model)
            
            # Train e Test sui dati trasformati e ridotti
            retrained_model.fit(scaled_outer_train[top_feat], outer_train_labels)
            y_pred = retrained_model.predict(scaled_outer_test[top_feat])
            
            fold_results_per_n[n].append(accuracy_score(y_true=outer_test_labels, y_pred=y_pred))
            
            # Analisi Sottogruppi ORR
            subsets = {
                'CR': outer_test[outer_test['ORR'] == 'CR'],
                'PR': outer_test[outer_test['ORR'] == 'PR'],
                'PD': outer_test[outer_test['ORR'] == 'PD'],
                'Dead': outer_test[outer_test['ORR'] == 'Dead']
            }
            
            accuracies_dict = {}
            for name, subset_df in subsets.items():
                if subset_df.shape[0] > 0:
                    subset_features = scaled_outer_test.loc[subset_df.index, top_feat]
                    subset_labels_encoded = labelenc.transform(subset_df['response'])
                    y_pred_subset = retrained_model.predict(subset_features)
                    accuracies_dict[name] = accuracy_score(subset_labels_encoded, y_pred_subset)
                else:
                    accuracies_dict[name] = None
            
            accuracies_per_n[n].append(accuracies_dict)
            print(accuracies_dict)

    # Aggregazione finale delle metriche su tutti i fold per ciascun N
    final_means = {}
    final_stds = {}
    final_accuracies = {}
    
    for n in top_n_feats_list:
        final_means[n] = pd.Series(fold_results_per_n[n]).mean()
        final_stds[n] = pd.Series(fold_results_per_n[n]).std()
        
        cr_accs = [acc['CR'] for acc in accuracies_per_n[n] if acc['CR'] is not None]
        pr_accs = [acc['PR'] for acc in accuracies_per_n[n] if acc['PR'] is not None]
        pd_accs = [acc['PD'] for acc in accuracies_per_n[n] if acc['PD'] is not None]
        dead_accs = [acc['Dead'] for acc in accuracies_per_n[n] if acc['Dead'] is not None]
        
        final_accuracies[n] = {
            'CR': statistics.fmean(cr_accs) if cr_accs else "NO CR",
            'PR': statistics.fmean(pr_accs) if pr_accs else "NO PR",
            'PD': statistics.fmean(pd_accs) if pd_accs else "NO PD",
            'Dead': statistics.fmean(dead_accs) if dead_accs else "NO DEADS"
        }
        print(f"Aggregated metrics for N={n}: {final_accuracies[n]}")

    mean_df = pd.DataFrame([final_means], index=['Mean Accuracy'])
    std_df = pd.DataFrame([final_stds], index=['Std Accuracy'])
    accuracies_df = pd.DataFrame(final_accuracies)
    top_feat_df = pd.DataFrame(features_matrix_per_n)

    metrics_df = pd.concat([mean_df, std_df, accuracies_df], axis=0)

    #metrics_df.to_excel("features_ablation_study_raw_random_cv.xlsx", float_format="%.4f", sheet_name="Feature Ablation Study")
    top_feat_df.to_csv("top_features.csv")


if __name__ == "__main__":
    explainability()
