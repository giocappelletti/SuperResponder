import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from fileio import dataloader, serializer
from visualization import plotter
from preprocessing import Preprocessor
from utils import Splitter
from analysis import SHAPExplainer


def explainability():
    
    dataset_path = os.path.join(project_root, 'datasets', 'raw_dataset.csv')

    dataset, taxa_cols, _ = dataloader.load_dataset(dataset_path, False)
    
    prep = Preprocessor()

    train_set, test_set, train_labels, _ = Splitter(dataloader, serializer).split_train_test(dataset, False)
    
    scaled_train = prep.initialize(train_set, False, taxa_cols, True)    
    scaled_test = prep.initialize(test_set, False, taxa_cols, True)

    model = RandomForestClassifier(n_estimators=500,
                                          random_state=42,
                                          n_jobs=-1,
                                          class_weight="balanced")
    

    model.fit(scaled_train, train_labels)

    shap_values = SHAPExplainer(plotter, model).compute_shap_values(scaled_test)


if __name__ == "__main__":
    explainability()
