import sys
import os
import random

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from sklearn.calibration import LabelEncoder

from fileio import dataloader, serializer
from visualization import plotter
from models import Models


def random_forest_perf():
    
    models = Models(serializer, plotter)
    labelenc = LabelEncoder()

    perc_dataset_path = os.path.join(project_root, 'datasets', 'raw_dataset.csv')
    raw_dataset_path = os.path.join(project_root, 'datasets', 'table_species_reads.tsv')
    collap_dataset_path = os.path.join(project_root, 'datasets', 'genus_r.tsv')

    perc_dataset, taxa_cols, _ = dataloader.load_dataset(perc_dataset_path, False)
    raw_dataset, _, _ = dataloader.load_dataset(raw_dataset_path, False)
    collap_dataset, collap_taxa_cols, _ = dataloader.load_dataset(collap_dataset_path, False)
        
    perc_train_taxa = perc_dataset[taxa_cols] 
    raw_train_taxa = raw_dataset[taxa_cols]
    collap_train_taxa = collap_dataset[collap_taxa_cols]

    random_feats = [random.randint(0, 4630) for _ in range(50)]
    random_cols = perc_train_taxa.columns[random_feats]
    feat_random = perc_train_taxa[random_cols]
    
    perc_labels = labelenc.fit_transform(perc_dataset['response']) # O - Non responder, 1 - Responder
    raw_labels = labelenc.transform(raw_dataset['response'])
    collap_labels = labelenc.transform(collap_dataset['response'])

    _, perc_results = models.evaluate_classifier(perc_train_taxa, perc_labels)
    _, raw_results = models.evaluate_classifier(raw_train_taxa, raw_labels)
    _, collap_results = models.evaluate_classifier(collap_train_taxa, collap_labels)
    _, random_results = models.evaluate_classifier(feat_random, perc_labels)


    fold_perc_res = perc_results["5-fold CV"]
    fold_raw_res = raw_results["5-fold CV"]
    fold_collap_res = collap_results["5-fold CV"]
    fold_random_res = random_results["5-fold CV"]

    perc_df = pd.DataFrame(fold_perc_res)
    raw_df = pd.DataFrame(fold_raw_res)
    collap_df = pd.DataFrame(fold_collap_res)
    random_df = pd.DataFrame(fold_random_res)

    df = pd.concat([perc_df, raw_df, collap_df, random_df], axis = 1)

    df.to_excel("random_forest.xlsx", float_format="%.4f", sheet_name="Random Forest")
    

if __name__ == "__main__":
    random_forest_perf()
