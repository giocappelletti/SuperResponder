import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd 

from fileio.df_loader import dataloader
from fileio.serialization import serializer
from correlation.correlation import Correlator
from visualization.plotting import plotter
from utils.utils import get_first_column_index


def correlation():
    
    # Define file paths
    dataset_path = os.path.join(project_root, 'datasets', 'raw_dataset.csv')
    
    # Load Dataset, drop response and sanitize
    dataset, taxa_cols, meta_cols = dataloader.load_dataset(dataset_path, drop_response=False, sanitize=True)
    
    # Instance correlator
    correlator = Correlator()

    # Remove unwanted columns
    toberemoved = ['samples', 'medoids', 'submedoids']

    filetered_meta_cols = list(set(meta_cols) - set(toberemoved))

    # Compute chi-squared and p-values between each pair of metadata columns
    log_p_value_matrix, chi2_matrix = correlator.correlate_metadata(dataset, filetered_meta_cols)

    # Plot results
    plotter.plot_metadata_correlation(log_p_value_matrix, chi2_matrix)

    
    # Compute Cramer's V correlation with 'response' and 'ORR' target columns
    corr_dfs = correlator.chi2_cramers_v(dataset, meta_cols, ['response', 'ORR'])

    # Plot Cramer's V for each target
    for target in corr_dfs['Target'].unique():
        single_target_df = corr_dfs[corr_dfs['Target'] == target].copy()
        plotter.plot_cramers_v(single_target_df, target_name=target, visualize=True, save=True)
    
    # Save results to disk
    serializer.save_file(corr_dfs,
                         "correlation",
                         exp_type="chi2_cramers_v_all_targets",
                         exp_group="correlation",
                         save_format="csv")
    
    
    corr_matrix, resp_count = correlator.correlate_with_std(dataset,
                                                            target=['response', 'ORR'], 
                                                            start_index=get_first_column_index(dataset, 'k__'))
    
    serializer.save_file(corr_matrix['response'],
                         subfolder = "correlation",
                         exp_type = "correlation_response",
                         exp_group = "correlation",
                         save_format = "csv")

    serializer.save_file(corr_matrix['ORR'],
                         subfolder = "correlation",
                         exp_type = "correlation_ORR",
                         exp_group = "correlation",
                         save_format = "csv")
    
    
if __name__ == "__main__":
    correlation()