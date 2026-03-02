import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sklearn.calibration import LabelEncoder

from fileio.df_loader import dataloader
from preprocessing.preprocess import Preprocessor
from models.models import Models
from utils.splitter import Splitter


def pls_feature_importance():

    # Define file paths
    dataset_path = os.path.join(project_root, 'datasets', 'raw_dataset.csv')

    # Load dataset
    dataset, taxa_cols, _ = dataloader.load_dataset(dataset_path, drop_response=False)
    
    train_df, _ = Splitter().split_train_test(dataset)

    
    # Instance preprocessor
    preprocessor = Preprocessor()
    
    # Run preprocessor
    train_taxa, _ = preprocessor.initialize(
        complete_df=train_df,
        use_metadata=False,
        taxa_cols=taxa_cols
    )

    Models().pls_feature_importance(
        dataset=dataset,
        X_train=train_taxa[taxa_cols],
        taxa_cols=taxa_cols
    )


if __name__ == "__main__":
    pls_feature_importance()