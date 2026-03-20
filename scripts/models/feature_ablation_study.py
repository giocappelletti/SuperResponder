import sys
import os

from sklearn.model_selection import StratifiedKFold, train_test_split

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import random
import pandas as pd

from sklearn.calibration import LabelEncoder

from fileio import dataloader, serializer
from visualization import plotter
from utils import Splitter
from analysis import SHAPExplainer
from models import Models


def explainability():
    
    dataset_path = os.path.join(project_root, 'datasets', 'raw_dataset.csv')

    dataset, taxa_cols, _ = dataloader.load_dataset(dataset_path, False)
    
    models = Models(serializer, plotter)
    labelenc = LabelEncoder()
    
    #train_set, _, train_labels_raw, _ = Splitter(dataloader, serializer).split_train_test(dataset)

    train_taxa = dataset[taxa_cols] # Use entire dataset since we employ nested cross validation
    
    train_labels = labelenc.fit_transform(dataset['response']) # O - Non responder, 1 - Responder

    scorings = ['accuracy', 'precision', 'recall', 'f1']

    top_n_feats_list = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 4630]
    results = {}
    stds = {}

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for n in top_n_feats_list:
        # Nested cross validation
        fold_results = []

        for _, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(train_taxa, train_labels)):
            outer_train, outer_test = train_taxa.iloc[outer_train_idx], train_taxa.iloc[outer_test_idx]
            outer_train_labels = train_labels[outer_train_idx]
           
            # Fit on outer train set 
            model_pipeline, _, = models.evaluate_classifier(outer_train, # Don't use metadata
                                                            outer_train_labels,
                                                            skip_cv=True)
            model = model_pipeline.named_steps['classifier'] # Get fitted model instance
            
            explainer = SHAPExplainer(plotter, model)
            
            shap_vals = explainer.compute_shap_values(outer_test, visualize = False, extract_index = 1)

            top_feat = explainer.get_top_features(shap_vals, outer_test.columns, head=n) # Get the n top features list

            top_feat_train = outer_train[top_feat] # Select only n columns

            rf_topn = model.__class__(**model.get_params()) # Instantiate a new RF with same params 
            
            rf_topn.fit(top_feat_train, outer_train_labels) # Fit on dataset subset (N selected columns)

            # Cross validate (inner train set/test set)
            res_top = models.cross_validation_analysis(
                rf_topn,
                top_feat_train,
                outer_train_labels,
                n_folds = 5,
                scorings = scorings,
                njobs = -1,
                random_state = 42,
                shuffle = True)
            
            
            cv_metrics_agg = res_top.cv_metrics_agg
            for key, value in cv_metrics_agg.items():
                mean = value['mean']
                fold_results.append({key: mean})
        
        results[n] = pd.DataFrame(fold_results).mean().to_dict()
        stds[n] = pd.DataFrame(fold_results).std().to_dict()

        print(results[n])

    
    mean_df = pd.DataFrame(results)
    std_df = pd.DataFrame(stds)
    metrics_df = pd.concat([mean_df, std_df], axis=0)
    metrics_df.to_excel("features_ablation_study.xlsx", float_format="%.4f", sheet_name="Feature Ablation Study")
    

if __name__ == "__main__":
    explainability()
