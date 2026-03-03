import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sklearn.preprocessing import StandardScaler

from preprocessing import CLRTransformer, SmartScaler, Preprocessor
from analysis import Clustering
from analysis.clustering import Clustering

from fileio import dataloader, serializer
from visualization import plotter


def elbow_silhouette():

    # Define file paths
    dataset_path = os.path.join(project_root, 'datasets', 'raw_dataset.csv')
    
    # Instantiate dependencies
    clustering_instance = Clustering(dataloader, serializer, plotter)

    # Load dataset
    dataset, taxa_cols, meta_cols = dataloader.load_dataset(dataset_path)

    preprocessor = Preprocessor(
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
    inertia, silhouette = clustering_instance.compute_elbow_silhouette(scaled_taxa)
    

if __name__ == "__main__":
    elbow_silhouette()