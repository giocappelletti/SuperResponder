import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sklearn.calibration import LabelEncoder

from preprocessing.data_transformers import CLRTransformer
from fileio.df_loader import dataloader
from preprocessing.preprocess import Preprocessor
from models.models import Models
from utils.splitter import Splitter


def hard_ensemble_voting():

    # Define models paths
    models_paths = [
        "results/model_evaluation/20260227_164640/classifier_results/LogisticRegression.pkl",
        "results/model_evaluation/20260227_171142/classifier_results/RidgeClassifier.pkl"
    ] 

    # Define dataset 
    dataset_path = os.path.join(project_root, 'datasets', 'raw_dataset.csv')

    # Load dataset
    dataset, taxa_cols, _ = dataloader.load_dataset(dataset_path, drop_response=False)
    
    _, test_df = Splitter().split_train_test(dataset)

    le = LabelEncoder()
    y_test = le.fit_transform(test_df['response'])

    label_map = {'NR': 'Non Responder', 'R': 'Responder'}
    labels = [label_map[c] for c in le.classes_]
    
    x_test = test_df.drop(columns=['response'])

    Models().hard_voting_ensemble(models_paths,
                                  x_test,
                                  y_test,
                                  labels,
                                  compute_roc=True,
                                  taxa_cols=taxa_cols,
                                  visualize=True,
                                  save=True
                                  )


if __name__ == "__main__":
    hard_ensemble_voting()