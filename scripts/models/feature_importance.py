import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils import Splitter
from models import Models

from fileio import dataloader, serializer
from visualization import plotter


def run_feature_importance():
    
    # Define file paths
    dataset_path = os.path.join(project_root, 'datasets', 'raw_dataset.csv')
    model_path = "results/model_evaluation/20260303_174615/classifier_results/ExtraTrees.pkl"
    
    # Instantiate dependencies
    models = Models(serializer, plotter)

    # Load dataset
    dataset, taxa_cols, _ = dataloader.load_dataset(dataset_path, drop_response=False)
    
    # Split to obtain a training set
    train_df, _ = Splitter(dataloader, serializer).split_train_test(dataset)
    
    # Remove index column, it's not needed
    train_df.reset_index(drop=True)

    # Run feature importance
    result = models.features_importance(train_df[taxa_cols], model_path)

    print(result)


if __name__ == "__main__":
    run_feature_importance()
