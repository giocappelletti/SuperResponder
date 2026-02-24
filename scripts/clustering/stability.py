import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sklearn.preprocessing import StandardScaler

from preprocessing.data_transformers import CLRTransformer
from preprocessing.scaler import SmartScaler
from preprocessing.preprocess import Preprocessor
from fileio.df_loader import dataloader
from clustering.clustering import Clustering


def stability_analysis():

    # Define file paths
    dataset_path = os.path.join(project_root, 'datasets', 'raw_dataset.csv')
    dataset_config_path = os.path.join(project_root, 'config', 'features.yaml')
    clustering_config_path = os.path.join(project_root, 'config', 'clustering.yaml') 
    
    # Load reference dataset (Full)
    ref_dataset, ref_taxa_cols, _ = dataloader.load_dataset(dataset_path)
    
    # Instance preprocessor
    preprocessor = Preprocessor(
        config_path=dataset_config_path,
        transformer=CLRTransformer,
        scaler=StandardScaler
    )

    # Run preprocessor
    X_ref_taxa, _ = preprocessor.initialize(
        complete_df=ref_dataset,
        use_metadata=False,
        taxa_cols=ref_taxa_cols
    )
    
    # Scale and transform data
    scaler = SmartScaler()
    scaled_ref_taxa = scaler.fit_transform(X_ref_taxa)

    clustering = Clustering(clustering_config_path)

    # Run stability analysis
    ari_results, fm_results, ari_matrix, fm_matrixc = clustering.cluster_stability_ari_fm(scaled_ref_taxa)


if __name__ == "__main__":
    stability_analysis()