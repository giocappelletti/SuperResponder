import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fileio.df_loader import dataloader
from fileio.serialization import serializer
from correlation.correlation import Correlator
from visualization.plotting import plotter
import pandas as pd # Import pandas for DataFrame operations


if __name__ == "__main__":
    
    dataset_path = "datasets/raw_dataset.csv"
    
    dataset, taxa_cols, meta_cols = dataloader.load_dataset(dataset_path, drop_response=False, sanitize=True)
    
    correlator = Correlator()

    meta_cols.remove('samples')
    meta_cols.remove('medoids')
    meta_cols.remove('submedoids')

    #log_p_value_matrix, chi2_matrix = correlator.correlate_metadata(dataset, meta_cols)

    #plotter.plot_metadata_correlation(log_p_value_matrix, chi2_matrix)

    corr_dfs = correlator.chi2_cramers_v(dataset, meta_cols, ['response', 'ORR'])

    # Itera attraverso i target unici nel DataFrame concatenato
    for target in corr_dfs['Target'].unique():
        # Filtra il DataFrame per il target corrente
        single_target_df = corr_dfs[corr_dfs['Target'] == target].copy()
        
        # Plotta i risultati di Cramer's V per il target corrente
        plotter.plot_cramers_v(single_target_df, target_name=target, visualize=True, save=True)

    
    serializer.save_file(corr_dfs,
                         "correlation",
                         exp_type="chi2_cramers_v_all_targets", # Cambiato exp_type per riflettere tutti i target
                         exp_group="correlation",
                         save_format="csv")
    
