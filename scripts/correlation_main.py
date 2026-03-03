import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from analysis import Correlator
from utils import get_first_column_index

from fileio import dataloader, serializer
from visualization import plotter


def correlation():
    
    # Define file paths
    dataset_path = os.path.join(project_root, 'datasets', 'raw_dataset.csv')
    
    # Load Dataset, drop response and sanitize
    dataset, taxa_cols, meta_cols = dataloader.load_dataset(dataset_path, drop_response=False, sanitize=True)
    
    # Instance correlator and inject dependencies
    correlator = Correlator(dataloader, serializer, plotter)

    # Remove unwanted columns
    toberemoved = ['samples', 'medoids', 'submedoids']

    filtered_meta_cols = list(set(meta_cols) - set(toberemoved))

    # Compute chi-squared and p-values between each pair of metadata columns
    log_p_value_matrix, chi2_matrix = correlator.correlate_metadata(dataset, 
                                                                    filtered_meta_cols, 
                                                                    visualize = True,
                                                                    save = False)

    # Compute Cramer's V correlation with 'response' and 'ORR' target columns
    corr_dfs = correlator.chi2_cramers_v(dataset, 
                                         meta_cols, 
                                         target = ['response', 'ORR'],
                                         visualize = True,
                                         save = False)
    
    
    corr_matrix, resp_count = correlator.correlate_with_std(dataset,
                                                            target = ['response', 'ORR'], 
                                                            start_index = get_first_column_index(dataset, 'k__'),
                                                            save = False)
    
    
if __name__ == "__main__":
    correlation()