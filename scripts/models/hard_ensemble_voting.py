import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sklearn.calibration import LabelEncoder

from utils import Splitter
from models import Models

from fileio import dataloader, serializer
from visualization import plotter


def hard_ensemble_voting():

    # Define models paths
    models_paths = [
        "results/model_evaluation/20260303_174707/classifier_results/LogReg.pkl",
        "results/model_evaluation/20260303_174721/classifier_results/Ridge.pkl"
    ] 

    # Define dataset 
    dataset_path = os.path.join(project_root, 'datasets', 'raw_dataset.csv')

    # Instantiate dependencies
    models_instance = Models(serializer, plotter)

    # Load dataset
    dataset, taxa_cols, _ = dataloader.load_dataset(dataset_path, drop_response = False)
    
    _, test_df, _, _ = Splitter(dataloader=dataloader, serializer=serializer).split_train_test(dataset)
    
    le = LabelEncoder()
    y_test = le.fit_transform(test_df['response'])

    label_map = {'NR': 'Non Responder', 'R': 'Responder'}
    labels = [label_map[c] for c in le.classes_]
    
    x_test = test_df.drop(columns = ['response'])

    models_instance.hard_voting_ensemble(models_paths,
                                        x_test,
                                        y_test,
                                        labels,
                                        compute_roc = True,
                                        taxa_cols = taxa_cols,
                                        visualize = True,
                                        save = False)


if __name__ == "__main__":
    hard_ensemble_voting()