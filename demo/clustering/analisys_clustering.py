import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from sklearn.preprocessing import StandardScaler

from dataTransformers.data_transformers import CLRTransformer
from dataTransformers.scaler import Scaler
from preprocessing.preprocess import Preprocessor
from fileio.df_loader import DataLoader
from clustering.clustering import Clustering


if __name__ == "__main__":

    # Define paths

    dataset_path = os.path.join(project_root, 'datasets', 'raw_dataset.csv')
    modenesi_dataset_path = os.path.join(project_root, 'datasets', 'splitted', 'Full', 'modenesi.csv')
    dataset_config_path = os.path.join(project_root, 'config', 'features.yaml')
    clustering_config_path = os.path.join(project_root, 'config', 'clustering.yaml') 

    # Instance dataloader
    dataloader = DataLoader()

    # Load reference dataset (Full)
    ref_dataset, ref_taxa_cols, ref_meta_cols = dataloader.load_dataset(dataset_path)

    # Load target dataset (Modenesi)
    modenesi = dataloader.load_dataset(modenesi_dataset_path, sanitize=False,
                                                drop_response=False, index_col=0)
    
    # Istance preprocessor
    preprocessor = Preprocessor(
        config_path=dataset_config_path,
        transformation=CLRTransformer,
        scaler=StandardScaler
    )

    # Run preprocessor
    X_ref_taxa, _ = preprocessor.initialize(
        complete_df=ref_dataset,
        use_metadata=False,
        taxa_cols=ref_taxa_cols
    )

    # Set to numeric values, if errors set NaN
    modenesi[ref_taxa_cols] = modenesi[ref_taxa_cols].apply(pd.to_numeric, errors='coerce')

    # Scale both using the same scaler instance to ensure consistency
    scaler = Scaler()
    scaled_ref_taxa = scaler.fit_transform(X_ref_taxa)
    scaled_modenesi_taxa = scaler.transform(modenesi[ref_taxa_cols].copy())
    
    # Instance clustering class
    clustering = Clustering(clustering_config_path)
    
    # Run metadata analysis
    _ = clustering.metadata_analysis(scaled_ref_taxa, ref_dataset, ref_meta_cols)
    
    # Run response analysis
    _, _, _ = clustering.response_analysis(data_matrix=scaled_ref_taxa, dataset=dataset_path, orig_dataset=dataset_path)
    
    # Run modenesi analysis
    _ = clustering.modenesi_analysis(scaled_modenesi_taxa, scaled_ref_taxa) 