import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils import Splitter
from preprocessing import SmartScaler
from analysis import DimensionalityReduction

from fileio import dataloader, serializer
from visualization import plotter


def dimensionality_reduction():
    
    # Define file paths
    dataset_path = os.path.join(project_root, 'datasets', 'raw_dataset.csv')
    
    # Instance Splitter with default conf file
    splitter = Splitter(dataloader, serializer)

    # Scalers must be fitted on different dataset, thus use two objects to avoid mixing the classes' internal states
    full_scaler = SmartScaler()
    split_scaler = SmartScaler()
    
    # Instance dimensionality reduction class using default conf file
    dim_red = DimensionalityReduction()
    
    # Convert labels from categorical to numerical values, e.g. Responder/Non-Responder becomes 0/1
    le = LabelEncoder()

    # Used to gather all results -> build a single window with multiple plots
    data = {}
    cumulative_vars = {}

    # Load Dataset, drop response and sanitize
    raw_dataset, taxa_cols, meta_cols = dataloader.load_dataset(dataset_path, drop_response=False, sanitize=True)

    # Split
    train_set, test_set = splitter.split_train_test(raw_dataset)

    # Encode labels
    y = le.fit_transform(raw_dataset['response'])
    y_train = le.transform(train_set['response'])
    y_test = le.transform(test_set['response'])
    
    # Drop response column and select only numeric columns. We don't need metadata here
    dataset = raw_dataset.drop(columns=['response'])[taxa_cols]
    
    # Transform and Scale 
    dataset_transformed = full_scaler.transform_then_fit_transform(dataset)
    train_transformed = split_scaler.transform_then_fit_transform(train_set[taxa_cols])
    
    # Test set must not be fitted
    test_transformed = split_scaler.transform(test_set[taxa_cols])

    # Compute Principal Components Analysis
    components, var_ratio, cum_var_ratio, _ = dim_red.PCA(dataset_transformed)

    data['PCA'] = components
    cumulative_vars['PCA'] = cum_var_ratio

    # Compute Kernel PCA
    components, per_comp, cum_per_comp = dim_red.KPCA(dataset_transformed)
    
    data['KPCA'] = components
    cumulative_vars['KPCA'] = cum_per_comp

    # Compute T-Stochastic Neighbour Embedding
    results = dim_red.TSNE(dataset_transformed)

    data['TSNE'] = results

    # Compute Principal Coordinate Analysis with bray-curtis distance. Avoid scaling data
    components, values, prop_expl, cum_prop_expl = dim_red.PCOA(dataset, distance='braycurtis')

    data['PCOA_bc'] = values
    cumulative_vars['PCOA_bc'] = cum_prop_expl

    # Compute PCOA with jensen-shannon distance. Avoid scaling data
    components, values, prop_expl, cum_prop_expl = dim_red.PCOA(dataset, distance='jensenshannon')

    data['PCOA_js'] = values
    cumulative_vars['PCOA_js'] = cum_prop_expl

    # Compute Partial Least Squares Discriminant Analysis 
    results = dim_red.PLS_DA(train_transformed, y_train, test_transformed)

    data['PLS_DA'] = results

    # Create a dictionary of labels for each DR method
    # For PCA, KPCA, PCOA, TSNE, the labels are 'y' (full dataset)
    # For PLS_DA, the labels are 'y_test' as 'results' corresponds to test_transformed
    labels_for_dr = {
        'PCA': y,
        'KPCA': y,
        'TSNE': y,
        'PCOA_bc': y,
        'PCOA_js': y,
        'PLS_DA': y_test
    }

    # Plot Dimensionality Reduction results
    plotter.plot_DR(data = data, 
                    method = "dimensionality_reduction", 
                    n_components = 2, 
                    per_comp = per_comp, 
                    labels_dict=labels_for_dr, 
                    color_map=["red", "green"],)
    
    # Plot cumulative variance
    plotter.plot_cumulative_vars(data = cumulative_vars, save = True)

    # Compute Linear Discriminant Analysis
    results = dim_red.LDA(train_transformed, y_train, test_transformed)

    dataframe = pd.DataFrame({'Comp 1': results, 'Response': y_test})
    plotter.plot_LDA(dataframe, transform_method = "CLR")


if __name__ == "__main__":
    dimensionality_reduction()
