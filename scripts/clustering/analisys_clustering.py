import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from sklearn.preprocessing import StandardScaler

from dataTransformers.data_transformers import CLRTransformer
from dataTransformers.scaler import SmartScaler
from preprocessing.preprocess import Preprocessor
from fileio.df_loader import dataloader
from clustering.clustering import Clustering


def analysis_clustering():

    # Define file paths
    dataset_path = os.path.join(project_root, 'datasets', 'raw_dataset.csv')
    modenesi_dataset_path = os.path.join(project_root, 'datasets', 'modenesi.csv')

    # Load reference dataset (Full)
    ref_dataset, ref_taxa_cols, ref_meta_cols = dataloader.load_dataset(dataset_path)

    # Load target dataset (Modenesi)
    modenesi = dataloader.load_dataset(modenesi_dataset_path, sanitize=False,
                                                drop_response=False, index_col=0)
    
    # Istance preprocessor with default values
    # config/features.yaml, CLR transformer and Standard Scaler
    preprocessor = Preprocessor()

    # Run preprocessor
    X_ref_taxa, _ = preprocessor.initialize(
        complete_df=ref_dataset,
        use_metadata=False,
        taxa_cols=ref_taxa_cols
    )

    # Set to numeric values, if errors set NaN
    modenesi[ref_taxa_cols] = modenesi[ref_taxa_cols].apply(pd.to_numeric, errors='coerce')

    # Scale both using the same scaler instance to ensure consistency
    # CLR transformer and Standard Scaler
    scaler = SmartScaler()
    scaled_ref_taxa = scaler.fit_transform(X_ref_taxa)
    scaled_modenesi_taxa = scaler.transform(modenesi[ref_taxa_cols].copy())
    
    # Instance clustering class with default conf file
    clustering = Clustering()
    
    # Run metadata analysis
    summary_df = clustering.metadata_analysis(scaled_ref_taxa, ref_dataset, ref_meta_cols)
    
    # Run response analysis
    contingency, chi2, p_value = clustering.response_analysis(data_matrix=scaled_ref_taxa, 
                                                              dataset=dataset_path, 
                                                              orig_dataset=dataset_path)
    
    # Run clustering analysis on modenesi data
    cluster_df = clustering.cluster_analysis(scaled_modenesi_taxa, scaled_ref_taxa) 


if __name__ == "__main__":
    analysis_clustering()