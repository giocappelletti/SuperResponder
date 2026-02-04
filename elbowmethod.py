import yaml
import pandas as pd
from sklearn.preprocessing import StandardScaler

from dataTransformers.data_transformers import CLRTransformer
from dataTransformers.scaler import Scaler
from preprocessing.preprocess import Preprocessor
from utils.df_loader import DataLoader
from clustering.clustering import Clustering
from visualization.plotting import Plotter


if __name__ == "__main__":

    dataset_path = './datasets/raw_dataset.csv'
    dataset_config_path = './config/dataset.yaml'

    dataloader = DataLoader()

    dataset, taxa_cols, meta_cols = dataloader.load_dataset(dataset_path, drop_response=False)


    preprocessor = Preprocessor(
        config_path=dataset_config_path,
        transformation=CLRTransformer,
        scaler=StandardScaler
    )

    X_taxa, engine = preprocessor.initialize(
        complete_df=dataset,
        use_metadata=False,
        taxa_cols=taxa_cols
    )
    
    
    scaler = Scaler() 
    scaled_taxa = scaler.fit_transform(X_taxa)

    clustering = Clustering()
    
    #inertia, silhouette = clustering.compute_elbow_silhouette(X_scaled_taxa)
    
    #cluster_df = clustering.kmedoids(scaled_taxa)

    metadata_analysis_df = clustering.metadata_analysis(scaled_taxa, dataset, meta_cols)
    