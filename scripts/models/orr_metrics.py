import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

from models import Models
from utils import Splitter
from fileio import dataloader, serializer
from visualization import plotter
from logger import logger


def _calculate_metrics_for_classes(y_true, y_pred, is_resp):
    metrics_dict = {}

    # Overall Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    metrics_dict['Accuracy'] = round(accuracy, 4)

    # Compute metrics for class 1 "responder" if ORR is CR or PR, 0 "non responder" for PD and Dead
    precision_1 = precision_score(y_true, y_pred, pos_label=is_resp, zero_division=0)
    recall_1 = recall_score(y_true, y_pred, pos_label=is_resp, zero_division=0)
    f1_1 = f1_score(y_true, y_pred, pos_label=is_resp, zero_division=0)
    
    metrics_dict[f'Class {is_resp}'] = {
        'Precision': round(precision_1, 4),
        'Recall': round(recall_1, 4),
        'F1-Score': round(f1_1, 4)
    }
    
    return metrics_dict


def orr_metrics():

    # Define file paths
    dataset_path = os.path.join(project_root, 'datasets', 'raw_dataset.csv')

    # Load dataset
    dataset, taxa_cols, _ = dataloader.load_dataset(dataset_path, drop_response=False)

    train_df, test_df, train_labels, _ = Splitter(dataloader, serializer).split_train_test(dataset)

    # Encode Responder/Non-Responder as 0/1
    le = LabelEncoder()

    train_labels = le.fit_transform(train_df['response'])
    
    logger.info(f"LabelEncoder mapping: {list(le.classes_)} -> {list(range(len(le.classes_)))}")
    
    # Exclude metadata
    train_taxa = train_df[taxa_cols] 

    # Instance models class injecting dependencies
    models = Models(serializer, plotter)

    fitted_model, _ = models.evaluate_classifier(train_taxa, train_labels)

    # Define the four subsets based on 'ORR'
    complete_resp = test_df[test_df['ORR'] == 'CR']
    partial_resp = test_df[test_df['ORR'] == 'PR']
    partial_dis = test_df[test_df['ORR'] == 'PD']
    dead = test_df[test_df['ORR'] == 'Dead']
    
    subsets = {
        'Complete Response (CR)': complete_resp,
        'Partial Response (PR)': partial_resp,
        'Partial Disease (PD)': partial_dis,
        'Dead': dead
    }

    all_subset_metrics = {}

    logger.info("Computing metrics for ORR subsets")

    for name, subset_df in subsets.items():
        if not subset_df.empty:
            subset_features = subset_df[taxa_cols]
            subset_labels_raw = subset_df['response']

            # Use the same LabelEncoder to transform test set labels
            subset_labels_encoded = le.transform(subset_labels_raw)
            
            y_pred_subset = fitted_model.predict(subset_features)

            is_resp = 1 if name == 'Complete Response (CR)' or name == 'Partial Response (PR)' else 0

            all_subset_metrics[name] = _calculate_metrics_for_classes(
                subset_labels_encoded, y_pred_subset, is_resp
            )
        else:
            all_subset_metrics[name] = "No data in this subset"

    for name, metrics in all_subset_metrics.items():
        print(f"\nSubset: {name}")
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items(): # Iterate through 'Accuracy', 'Class n'
                if isinstance(value, dict): 
                    print(f"  {metric_name}:")
                    for sub_metric_name, sub_value in value.items(): # Iterate through 'Precision', 'Recall', 'F1-Score'
                        print(f"    {sub_metric_name}: {sub_value}")
                else: # For 'Accuracy'
                    print(f"  {metric_name}: {value}")
        else:
            print(f"  {metrics}")


if __name__ == "__main__":
    orr_metrics()
