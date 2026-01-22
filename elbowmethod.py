import yaml
import pandas as pd
from sklearn.preprocessing import StandardScaler

from DataTransformers.data_transformers import CLRTransformer
from DataTransformers.scaler import Scaler
from Preprocessing.preprocess import Preprocessor
from Utils.df_loader import DataLoader
from Clustering.clustering import Clustering
from Visualization.plotting import Plotter


if __name__ == "__main__":

    dataset_path = './old/DR_Modelli/Datasets/raw_dataset.csv'
    dataset_config_path = './Config/dataset.yaml'

    DataLoader = DataLoader()

    print("Loading dataset...")

    dataset, taxa_cols, meta_cols = DataLoader.load_dataset(dataset_path)

    print("Initializing preprocessor...")

    preprocessor = Preprocessor(
        config_path=dataset_config_path,
        transformation=CLRTransformer,
        scaler=StandardScaler
    )

    print("Preprocessing dataset...")

    X_taxa, engine = preprocessor.initialize(
        complete_df=dataset,
        use_metadata=False,
        taxa_cols=taxa_cols
    )
    
    scaler = Scaler()
    print("Scaling taxa...")
    X_scaled_taxa = scaler.fit_transform(X_taxa)

    clustering = Clustering()
    
    print("Computing Elbow and Silhouette scores for K-Medoids...")

    ks, inertias, silhouettes, tick_values, metric = clustering.compute_elbow_silhouette(X_scaled_taxa)

    print("Plotting Elbow and Silhouette scores...")

    plotter = Plotter(output_dir="clustering_plots")
    plotter.plot_elbow_and_silhouette(
        ks, inertias, silhouettes, tick_values, metric, visualize=True)