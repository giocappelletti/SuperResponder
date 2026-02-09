import yaml
import pandas as pd
from sklearn.preprocessing import StandardScaler

from dataTransformers.data_transformers import CLRTransformer
from dataTransformers.scaler import Scaler
from preprocessing.preprocess import Preprocessor
from utils.df_loader import DataLoader
from clustering.clustering import Clustering


if __name__ == "__main__":

    dataset_path = 'datasets/raw_dataset.csv'
    modenesi_dataset_path = 'datasets/splitted/Full/modenesi.csv'
    unifrac_dataset_path = 'datasets/unifrac/dm_588_bacteria_w_unifrac.tsv'
    dataset_config_path = './config/dataset.yaml'

    dataloader = DataLoader()

    # Load reference dataset (Full)
    ref_dataset, ref_taxa_cols, _ = dataloader.load_dataset(dataset_path)
    # Load target dataset (Modenesi)
    modenesi = dataloader.load_dataset(modenesi_dataset_path, sanitize=False,
                                                drop_response=False, index_col=0)
    
    preprocessor = Preprocessor(
        config_path=dataset_config_path,
        transformation=CLRTransformer,
        scaler=StandardScaler
    )

    X_ref_taxa, _ = preprocessor.initialize(
        complete_df=ref_dataset,
        use_metadata=False,
        taxa_cols=ref_taxa_cols
    )


    modenesi[ref_taxa_cols] = modenesi[ref_taxa_cols].apply(pd.to_numeric, errors='coerce')

    
    # Scale both using the same scaler instance to ensure consistency
    scaler = Scaler()
    scaled_ref_taxa = scaler.fit_transform(X_ref_taxa)
    scaled_modenesi_taxa = scaler.transform(modenesi[ref_taxa_cols].copy())

    """
    dataset, dataset_aligned, distance_matrix_aligned, orig_dataset = dataloader.load_dataset(
        unifrac_dataset_path, unifrac=True, orig_dataset_path=dataset_path
    )
    """
    clustering = Clustering()
    
    #metadata_analysis_df = clustering.metadata_analysis(scaled_taxa, dataset, meta_cols)

    #_, _, _ = clustering.response_analysis(data_matrix=distance_matrix_aligned, dataset=dataset_aligned, orig_dataset=orig_dataset)
    #_, _, _ = clustering.response_analysis(data_matrix=scaled_taxa, dataset=dataset, orig_dataset=dataset_path)
    
    #clustering.modenesi_analysis(scaled_modenesi_taxa, scaled_ref_taxa) 