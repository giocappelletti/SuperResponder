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


def pca_clustering():

    # Define file paths    
    dataset_path = os.path.join(project_root, 'datasets', 'raw_dataset.csv')
    unifrac_dataset_path = os.path.join(project_root, 'datasets', 'unifrac', 'dm_588_bacteria_w_unifrac.tsv')

    # Instantiate dependencies
    clustering = Clustering(dataloader, serializer, plotter)

    # Load reference dataset (Full)
    ref_dataset, ref_taxa_cols, _ = dataloader.load_dataset(dataset_path)
    
    preprocessor = Preprocessor(
        transformer = CLRTransformer,
        scaler = StandardScaler
    )

    # Run preprocessor
    X_ref_taxa, _ = preprocessor.initialize(
        complete_df = ref_dataset,
        use_metadata = False,
        taxa_cols = ref_taxa_cols
    )

    # Scale and transform data
    scaler = SmartScaler()
    scaled_ref_taxa = scaler.fit_transform(X_ref_taxa)
    
    # Load Unifrac distance file
    dataset, dataset_aligned, distance_matrix_aligned, orig_dataset = dataloader.load_dataset(
        unifrac_dataset_path, unifrac=True, orig_dataset_path=dataset_path
    )
    
    # Run PCA
    fitted_pca, medoids, = clustering.pca_mds(distance_matrix_aligned, pca_type="pca")


if __name__ == "__main__":
    pca_clustering()