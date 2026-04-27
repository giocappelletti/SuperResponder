import sys
import os

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd

from sklearn.calibration import LabelEncoder

from fileio import dataloader, serializer
from visualization import plotter
from analysis import SHAPExplainer
from models import Models



def get_top_features():
    
    # Build data path
    dataset_path = os.path.join(project_root, 'datasets', 'raw_dataset.csv')

    # Get dataframe loaded in memory
    dataset, taxa_cols = dataloader.load_dataset(dataset_path, False)
    
    # Instantiate repo classes and inject dependencies
    models = Models(serializer, plotter)

    # Needed to encode labels
    labelenc = LabelEncoder()
    
    dataset_taxa = dataset[taxa_cols] # Use entire dataset since we employ nested cross validation
    
    dataset_labels = labelenc.fit_transform(dataset['response']) # O - Non responder, 1 - Responder

    
    features_matrix_per_n = []


    np.random.seed(42)
    n_splits = 5
    outer_cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
    
    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(dataset_taxa, dataset_labels)):
        print(f"\n--- FOLD {fold_idx + 1}/{n_splits} ---")
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
        
        if scaler is not None:
            scaled_outer_train = pd.DataFrame(scaler.transform(outer_train), columns=outer_train.columns, index=outer_train.index)
            scaled_outer_test = pd.DataFrame(scaler.transform(outer_test[taxa_cols]), columns=taxa_cols, index=outer_test.index)
        else:
            scaled_outer_train = outer_train.copy()
            scaled_outer_test = outer_test[taxa_cols].copy()
        
        explainer = SHAPExplainer(plotter, best_model, scaled_outer_train) 
        shap_vals = explainer.compute_shap_values(scaled_outer_test, visualize=False, extract_index=1)
        
        all_top_feat = explainer.get_top_features(shap_vals, scaled_outer_test.columns, head=50)

        features_matrix_per_n.append(all_top_feat)
            
    top_feat_df = pd.DataFrame(features_matrix_per_n).T

    top_feat_df.to_csv("top_features_LR.csv")


if __name__ == "__main__":
    get_top_features()
