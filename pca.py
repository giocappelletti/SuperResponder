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

    dataset_path = 'datasets/raw_dataset.csv'
    modenesi_dataset_path = 'datasets/splitted/Full/modenesi.csv'
    unifrac_dataset_path = 'datasets/unifrac/dm_588_bacteria_w_unifrac.tsv'
    dataset_config_path = './config/dataset.yaml'

    dataloader = DataLoader()

    # Load reference dataset (Full)
    ref_dataset, ref_taxa_cols, _ = dataloader.load_dataset(dataset_path)
    
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

    scaler = Scaler()
    scaled_ref_taxa = scaler.fit_transform(X_ref_taxa)

    modena_samples = [f"modena{i}" for i in range(1, 20)]  # modena1,...,modena19
    is_modena = ref_dataset.index.str.lower().isin(modena_samples)
    print(is_modena)
    print(type(is_modena))
    clustering = Clustering()

    """

    dataset, dataset_aligned, distance_matrix_aligned, orig_dataset = dataloader.load_dataset(
        unifrac_dataset_path, unifrac=True, orig_dataset_path=dataset_path
    )
    
    """
    #_, _, = clustering.pca_mds(scaled_ref_taxa, pca_type="kpca")
    