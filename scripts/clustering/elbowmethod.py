import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sklearn.preprocessing import StandardScaler

from dataTransformers.data_transformers import CLRTransformer
from dataTransformers.scaler import SmartScaler
from preprocessing.preprocess import Preprocessor
from fileio.df_loader import dataloader
from clustering.clustering import Clustering


if __name__ == "__main__":

    # Define file paths
    dataset_path = os.path.join(project_root, 'datasets', 'raw_dataset.csv')
    dataset_config_path = os.path.join(project_root, 'config', 'features.yaml')
    clustering_config_path = os.path.join(project_root, 'config', 'clustering.yaml') 
    
    # Load dataset
    dataset, taxa_cols, meta_cols = dataloader.load_dataset(dataset_path)

    # Instance preprocessor
    preprocessor = Preprocessor(
        config_path=dataset_config_path,
        transformer=CLRTransformer,
        scaler=StandardScaler
    )
    
    # Run preprocessor
    X_taxa, _ = preprocessor.initialize(
        complete_df=dataset,
        use_metadata=False,
        taxa_cols=taxa_cols
    )
    
    # Scale and transform data
    scaled_taxa = SmartScaler().fit_transform(X_taxa)

    # Compute elbow and silhouette scores
    inertia, silhouette = Clustering(clustering_config_path).compute_elbow_silhouette(scaled_taxa)
    
    