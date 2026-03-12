import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import random

from sklearn.calibration import LabelEncoder

from fileio import dataloader, serializer
from visualization import plotter
from utils import Splitter
from analysis import SHAPExplainer
from models import Models


def explainability():
    

    dataset_path = os.path.join(project_root, 'datasets', 'table_species_reads.tsv')
    scorings = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    dataset, taxa_cols, _ = dataloader.load_dataset(dataset_path, False)
    
    models = Models(serializer, plotter)
    labelenc = LabelEncoder()
    
    train_set, test_set, train_labels_raw, _ = Splitter(dataloader, serializer).split_train_test(dataset)

    train_taxa = train_set[taxa_cols]
    test_taxa = test_set[taxa_cols]
    
    train_labels = labelenc.fit_transform(train_labels_raw) # Fit on train_labels_raw to get all possible classes
    
    model_pipeline, _, _, _ = models.evaluate_classifier(train_taxa, # Don't use metadata
                                                         train_labels)

    model = model_pipeline.named_steps['classifier'] # Get fitted model instance
    
    # Apply transformation and scaling to test data
    scaled_test = model_pipeline.named_steps['smartscaler'].fit_transform(test_taxa) 

    # Compute SHAP values to get most important feature used by the model 
    explainer = SHAPExplainer(plotter, model)

    # Compute shap values on label encoded as 1
    shap_vals = explainer.compute_shap_values(scaled_test, visualize = False, extract_index = 1)
    top_feat = explainer.get_top_features(shap_vals, scaled_test.columns) # Get the 50 top features list

    top_feat_train = train_taxa[top_feat] # Select only 50 columns

    rf_top50 = model.__class__(**model.get_params()) # Instantiate a new RF with same params 
    
    rf_top50.fit(top_feat_train, train_labels) # Fit on dataset subset (50 selected columns)

    # Cross validate 
    res_top = models.cross_validation_analysis(
        rf_top50,
        top_feat_train,
        train_labels,
        n_folds = 5,
        scorings = scorings,
        njobs = -1,
        random_state = 42,
        shuffle = True)
    
    print(res_top['cv_metrics_agg'])
    
    # Pick 50 random features
    random_feats = [random.randint(0, 4630) for _ in range(50)]
    
    random_cols = train_taxa.columns[random_feats]
    feat_random = train_taxa[random_cols]

    rf_random = model.__class__(**model.get_params()) # Instantiate a new RF with same params 

    rf_random.fit(feat_random, train_labels) # Fit on random subset of data

    # Cross validate and get mean + std of scorings
    res_rnd = models.cross_validation_analysis(
        rf_random,
        feat_random,
        train_labels,
        n_folds = 5,
        scorings = scorings,
        njobs = -1,
        random_state = 42,
        shuffle = True
    )

    print(res_rnd['cv_metrics_agg'])
    

if __name__ == "__main__":
    explainability()
