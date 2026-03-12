import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import StandardScaler

from preprocessing import CLRTransformer, Preprocessor
from models import Models
from utils import Splitter

from fileio import dataloader, serializer
from visualization import plotter


def cross_validation():

    # Define file paths
    dataset_path = os.path.join(project_root, 'datasets', 'raw_dataset.csv')

    # Load dataset
    dataset, taxa_cols, _ = dataloader.load_dataset(dataset_path, drop_response=False)
    
    train_df, _, _, _ = Splitter(dataloader, serializer).split_train_test(dataset)

    # Encode Responder/Non-Responder as 0/1
    le = LabelEncoder()

    train_labels = le.fit_transform(train_df['response'])

    train_taxa = train_df[taxa_cols] # Exclude metadata

    # Instance models class injecting dependencies
    models = Models(serializer, plotter)

    _, results, _ = models.evaluate_classifier(train_taxa[taxa_cols],
                                               train_labels)

if __name__ == "__main__":
    cross_validation()