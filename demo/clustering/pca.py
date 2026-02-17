import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
    unifrac_dataset_path = os.path.join(project_root, 'datasets', 'unifrac', 'dm_588_bacteria_w_unifrac.tsv')
    dataset_config_path = os.path.join(project_root, 'config', 'features.yaml')
    clustering_config_path = os.path.join(project_root, 'config', 'clustering.yaml')

    # Instance dataloader
    dataloader = DataLoader()

    # Load reference dataset (Full)
    ref_dataset, ref_taxa_cols, _ = dataloader.load_dataset(dataset_path)
    
    # Instance preprocessor
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

    # Scale data
    scaler = Scaler()
    scaled_ref_taxa = scaler.fit_transform(X_ref_taxa)
    
    # Load Unifrac distance file
    dataset, dataset_aligned, distance_matrix_aligned, orig_dataset = dataloader.load_dataset(
        unifrac_dataset_path, unifrac=True, orig_dataset_path=dataset_path
    )
    
    # Run PCA
    _, _, = Clustering(clustering_config_path).pca_mds(distance_matrix_aligned, pca_type="pca")
