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


def cross_validation():

    # Define file paths
    dataset_path = os.path.join(project_root, 'datasets', 'raw_dataset.csv')

    # Load dataset
    dataset, taxa_cols, _ = dataloader.load_dataset(dataset_path, drop_response=False)
    
    train_df, _ = Splitter().split_train_test(dataset)

    # Encode Responder/Non-Responder as 0/1
    le = LabelEncoder()

    train_labels = le.fit_transform(train_df['response'])

    # Instance preprocessor
    preprocessor = Preprocessor()
    
    # Run preprocessor
    train_taxa, _ = preprocessor.initialize(
        complete_df=train_df,
        use_metadata=False,
        taxa_cols=taxa_cols
    )

    Models().evaluate_classifier_with_optuna(
        train_data=train_taxa[taxa_cols],
        train_labels=train_labels
    )


if __name__ == "__main__":
    cross_validation()