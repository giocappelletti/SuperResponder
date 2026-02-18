import pandas as pd
import numpy as np

from fileio.df_loader import dataloader
from utils.utils import compute_contingency
from logger.logger import logger


class Correlator:
    """
    Handles correlation analysis between variables.
    """

    def __init__(self):
        self.logger = logger
    

    def correlate_metadata(self, dataset: pd.DataFrame | str, meta_cols: list):
        """
        Computes the correlation between metadata variables using the chi-squared test.

        Parameters
        ----------
            dataset: pd.DataFrame or str
                DataFrame with metadata or path to it.
            meta_cols: list
                List of metadata columns to correlate.

        Returns
        -------
            log_p_value_matrix: pd.DataFrame
                Matrix of -log10(p-values).
            chi2_matrix: pd.DataFrame
                Matrix of chi-squared statistics.
                
        """

        if isinstance(dataset, str):
            dataset = dataloader.load_dataset(dataset, drop_response=False, sanitize=False)        

        # Create a matrix to memorize p_values
        p_matrix = pd.DataFrame(np.zeros((len(meta_cols), len(meta_cols))), 
                                index=meta_cols, columns=meta_cols)

        # Create a matrix to memorize chi-squared values
        chi2_matrix = pd.DataFrame(np.zeros((len(meta_cols), len(meta_cols))), 
                                        index=meta_cols, columns=meta_cols)

        self.logger.info("Computing correlation between metadata columns")
        # Compute chi-sqared for every couple of columns
        for i, var1 in enumerate(meta_cols):
            for j, var2 in enumerate(meta_cols):
                _, chi2, p, _, _ = compute_contingency(dataset, var1, var2)

                p_matrix.iloc[i, j] = p
                chi2_matrix.iloc[i, j] = chi2

        # Apply log transform
        np.seterr(divide='ignore')
        log_p_value_matrix = -np.log10(p_matrix)

        return log_p_value_matrix, chi2_matrix
           
    
    def _cramers_v(self, dataset: pd.DataFrame, meta_cols: list, target: str):   
        """
        Compute contingencies and cramers V, then returns formatted results.
        """

        dataset = dataset[meta_cols]

        results = {}
        for col in dataset.columns:
            if col == target:
                continue
            
            _, chi2, p_value, n, shape = compute_contingency(dataset, col, target)

            # Compute Cramer's V
            r = shape[0]
            k = shape[1]
            
            phi2 = chi2 / n
            phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
            
            rcorr = r - ((r - 1)**2) / (n - 1)
            kcorr = k - ((k - 1)**2) / (n - 1)
            
            min_dim_corr_minus_1 = min((kcorr - 1), (rcorr - 1))
            if min_dim_corr_minus_1 <= 0:
                cramer_v = 0.0 # Cramer's V is 0 if one dimension is 1 (no variance)
            else:
                cramer_v = np.sqrt(phi2corr / min_dim_corr_minus_1)
            
            results[col] = {'chi2_statistic': chi2, 'p_value': p_value, 'cramers_v': cramer_v}
        
        return results


    def _create_results_df(self, results, target_name):
        """
        Helper method to format Cramer's V and Chi-Squared results into a DataFrame.
        """
        df_results = pd.DataFrame(results).T.reset_index()
        df_results.columns = ['Column', 'Chi-Squared', 'P-value', "Cramers_V"]
        df_results['Target'] = target_name
        
        return df_results
    

    def chi2_cramers_v(self, dataset: pd.DataFrame | str, meta_cols: list, targets: list):
        """
        Computes Chi-Squared and Cramer's V statistics for multiple target variables.

        Parameters
        ----------
            dataset: pd.DataFrame or str
                DataFrame with metadata or path to it.
            meta_cols: list
                List of metadata columns to analyze.
            targets: list
                List of target columns to compute correlations against.

        Returns
        -------
            results_df: pd.DataFrame
                Concatenated DataFrame containing Chi-Squared, P-values, and Cramer's V for each target.
                
        """
        
        if isinstance(dataset, str):
            dataset = dataloader.load_dataset(dataset, drop_response=False, sanitize=False)

        all_results_dfs = []

        for target in targets:
            if target not in meta_cols:
                self.logger.warning(f"Target {target} not found in metadata columns, skipping analysis")
                continue

            self.logger.info(f"Computing Chi-Squared and Cramer's V statistics for target {target}")
            results = self._cramers_v(dataset, meta_cols, target)
            all_results_dfs.append(self._create_results_df(results, target))

        return pd.concat(all_results_dfs, ignore_index=True)

        
        