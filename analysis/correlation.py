from typing import Literal

import pandas as pd
import numpy as np

from scipy.stats import kendalltau, pointbiserialr
from sklearn.preprocessing import LabelEncoder

from utils import compute_contingency
from logger import logger


class Correlator:
    """
    Handles correlation analysis between variables.
    Parameters
    ----------
        dataloader: DataLoader
            Object to handle data loading.
        serializer: Serializer
            Object to handle data serialization.
        plotter: Plotter
            Object to handle data visualization.
            
    """

    def __init__(self, dataloader, serializer, plotter):
        self.logger = logger
        self.dataloader = dataloader
        self.serializer = serializer
        self.plotter = plotter


    @staticmethod
    def _cramers_v(dataset: pd.DataFrame, meta_cols: list, target: str):   
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


    @staticmethod
    def _create_results_df(results, target_name):
        """
        Helper method to format Cramer's V and Chi-Squared results into a DataFrame.
        """

        df_results = pd.DataFrame(results).T.reset_index()
        df_results.columns = ['Column', 'Chi-Squared', 'P-value', "Cramers_V"]
        df_results['Target'] = target_name
        
        return df_results
    

    def _correlate_single_target_with_std(self, dataset: pd.DataFrame, target_name: str, start_index: int):
        """
        Helper function to compute correlation between a single target variable and numeric columns.
        Supported targets are 'response' and 'ORR'.
        Uses Point-Biserial correlation for binary 'response' or Kendall's Tau for 'ORR'.
        """
        
        # Make a copy to avoid modifying the original dataset passed to the helper
        local_dataset = dataset.copy()

        selected_columns = [target_name] + local_dataset.columns[start_index:].tolist()
        local_dataset = local_dataset[selected_columns]

        if target_name not in ["response", "ORR"]:
            self.logger.error(f"target must be either 'response' or 'ORR', got {target_name}")
            raise ValueError

        self.logger.info(f"Computing correlation between {target_name} column and numeric columns")
        le = LabelEncoder()
        if target_name == "response":
            # Ensure 'response' is numeric (0 or 1)
            if not pd.api.types.is_numeric_dtype(local_dataset['response']):
                self.logger.warning(f"'{target_name}' column has non-numeric values. Encoding to numeric.")
                
                local_dataset['response'] = le.fit_transform(local_dataset['response'])
            
            # After encoding (or if already numeric), verify it's binary
            if set(local_dataset['response'].dropna().unique()) != {0, 1}:
                self.logger.error(f"Column 'response' must contain only 0 and 1 values for point-biserial correlation"
                                  f"{local_dataset['response'].dropna().unique()} found after encoding.")
                raise ValueError()
            
        elif target_name == "ORR": # target_name == "ORR"
            # Ensure 'ORR' is numeric. It's categorical (PD, PR, CR, Dead) and needs encoding.
            if not pd.api.types.is_numeric_dtype(local_dataset['ORR']):
                self.logger.warning(f"'{target_name}' column has non-numeric values. Encoding to numeric.")
                local_dataset['ORR'] = le.fit_transform(local_dataset['ORR'])

            if not pd.api.types.is_numeric_dtype(local_dataset['ORR']):
                self.logger.error("Column 'ORR' must be numeric")
                raise ValueError
            
            # If it was already numeric, no encoding needed.
            # No further check on unique values needed for Kendall's Tau as it handles ordinal data.
        else: # Invalid target_name
            self.logger.error(f"target must be either 'response' or 'ORR', got {target_name}")
            raise ValueError("Invalid target name.")

        # Select only numeric columns (excluding target column)
        # Ensure target_name is numeric after potential encoding
        if not pd.api.types.is_numeric_dtype(local_dataset[target_name]):
            self.logger.error(f"Target column '{target_name}' is not numeric after processing. Cannot proceed with correlation.")
            raise TypeError()

        numeric_columns = [col for col in local_dataset.columns if pd.api.types.is_numeric_dtype(local_dataset[col]) and col != target_name]

        column_name = f'{target_name}_correlation'

        correlation_matrix = pd.DataFrame(index = numeric_columns, columns = [column_name])

        for col in numeric_columns:
            # Use local_dataset for valid_data to ensure encoded values are used
            valid_data = local_dataset[[target_name, col]].dropna()

            if len(valid_data) < 3:
                self.logger.warning(f"Column '{col}' has insufficient data ({len(valid_data)} valid rows), Setting NAN")                
                correlation_matrix.loc[col, column_name] = np.nan
                continue

            if valid_data[col].std() == 0:
                self.logger.warning(f"Column '{col}' has zero standard deviation, Setting NAN")
                correlation_matrix.loc[col, column_name] = np.nan
                continue

            # Compute based on specific column
            if target_name == "response":
                correlation, _ = pointbiserialr(valid_data[target_name], valid_data[col])
            elif target_name == "ORR":  
                correlation, _ = kendalltau(valid_data[target_name], valid_data[col])

            correlation_matrix.loc[col, column_name] = correlation

        # Calculate STD_dev using local_dataset
        correlation_matrix['STD_dev'] = local_dataset[numeric_columns].std() 

        # Prepare DataFrame for stacking
        df_result = correlation_matrix.reset_index().rename(columns = {'index': 'Column'})
        df_result['Target'] = target_name

        # Counts significative correlations (absolute value > 0.1)
        count_significant_correlations = (df_result[column_name].abs() > 0.1).sum()

        return df_result, count_significant_correlations


    def chi2_cramers_v(self, 
                       dataset: pd.DataFrame | str, 
                       meta_cols: list, 
                       target: list,
                       visualize: bool = True, 
                       save: bool = False,
                       save_format: Literal['csv', 'tsv', 'xlsx'] = "csv") -> pd.DataFrame:
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
            visualize: bool, default True
                Whether to display the plots.
            save: bool, default False
                Whether to save the plots to disk.
            save_format: Literal['csv', 'tsv', 'xlsx'], default "csv"
                Format to use when saving results if save is True.

        Returns
        -------
            results_df: pd.DataFrame
                Concatenated DataFrame containing Chi-Squared, P-values, and Cramer's V for each target.
                
        """
        
        dataset = self.dataloader._get_dataframe(dataset)

        all_results_dfs = []

        for target in target:
            if target not in meta_cols:
                self.logger.warning(f"Target {target} not found in metadata columns, skipping analysis")
                continue

            self.logger.info(f"Computing Chi-Squared and Cramer's V statistics for target {target}")
            results = self._cramers_v(dataset, meta_cols, target)
            all_results_dfs.append(self._create_results_df(results, target))

        corr_dfs = pd.concat(all_results_dfs, ignore_index = True)


        # Plot Cramer's V for each target
        for target in corr_dfs['Target'].unique():
            single_target_df = corr_dfs[corr_dfs['Target'] == target].copy()
            self.plotter._plot_cramers_v(single_target_df, target, visualize, save)
        
        if save:
            self.serializer.save_file(corr_dfs,
                                      "correlation",
                                      exp_type = "chi2_cramers_v_all_targets",
                                      exp_group = "correlation",
                                      save_format = save_format)
        
        return corr_dfs
                                      

    def correlate_metadata(self, 
                           dataset: pd.DataFrame | str, 
                           meta_cols: list,
                           visualize: bool = True, 
                           save: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
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

        dataset = self.dataloader._get_dataframe(dataset)

        # Create a matrix to memorize p_values
        p_matrix = pd.DataFrame(np.zeros((len(meta_cols), len(meta_cols))), 
                                index = meta_cols, 
                                columns = meta_cols)

        # Create a matrix to memorize chi-squared values
        chi2_matrix = pd.DataFrame(np.zeros((len(meta_cols), len(meta_cols))), 
                                   index = meta_cols, 
                                   columns = meta_cols)

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

        self.plotter._plot_metadata_correlation(log_p_value_matrix, 
                                                chi2_matrix,
                                                visualize,
                                                save)

        return log_p_value_matrix, chi2_matrix


    def correlate_with_std(self, 
                           dataset: pd.DataFrame | str, 
                           target: str | list[str], 
                           start_index: int,
                           save: bool = False) -> tuple[pd.DataFrame, int] | tuple[dict[str, pd.DataFrame], dict]:
        """
        Computes correlation between a target variable (or variables) and numeric columns starting from a specific index.
        Supported targets are 'response' and 'ORR'.
        Uses Point-Biserial correlation for binary 'response' or Kendall's Tau for 'ORR'.

        Parameters
        ----------
            dataset: pd.DataFrame or str
                The dataset containing the variables.
            target: str or list[str]
                The target column name or a list of up to two target column names.
                Must be either 'response' or 'ORR' for single string, or a list containing them.
            start_index: int
                The starting integer index for the numeric columns to analyze.

        Returns
        -------
            If target is a single string:
                correlation_matrix: pd.DataFrame
                    DataFrame containing correlation coefficients and standard deviations.
                count_significant_correlations: int
                    Number of columns with an absolute correlation coefficient greater than 0.1.
            If target is a list of strings:
                all_correlation_dfs: dict[str, pd.DataFrame]
                    Dictionary of DataFrames containing correlation coefficients, standard deviations, and target name.
                all_counts: dict
                    Dictionary mapping each target name to its count of columns with absolute correlation > 0.1.
        """

        dataset = self.dataloader._get_dataframe(dataset)

        if isinstance(target, str):
            return self._correlate_single_target_with_std(dataset, target, start_index)
        
        elif isinstance(target, list):
            if not (1 <= len(target) <= 2):
                self.logger.error(f"If 'target' is a list, it must contain 1 or 2 elements, got {len(target)}")
                raise ValueError("Target list must contain 1 or 2 elements.")
            
            correlation_dfs = {}
            all_counts = {}

            for t in target:
                if t not in ["response", "ORR"]:
                    self.logger.error(f"Invalid target '{t}' in list. Supported targets are 'response' and 'ORR'.")
                    raise ValueError(f"Invalid target '{t}' in list.")

                corr_matrix_df, count = self._correlate_single_target_with_std(dataset, t, start_index)
                correlation_dfs[t] = corr_matrix_df
                all_counts[t] = count
                self.logger.info(f"Found {count} columns with correlation > 0.1 for {t} column")
            
            if save:
                self.serializer.save_file(correlation_dfs['response'],
                            subfolder = "correlation",
                            exp_type = "correlation_response",
                            exp_group = "correlation",
                            save_format = "csv")

                self.serializer.save_file(correlation_dfs['ORR'],
                                    subfolder = "correlation",
                                    exp_type = "correlation_ORR",
                                    exp_group = "correlation",
                                    save_format = "csv")

            return correlation_dfs, all_counts
        
        else:
            self.logger.error(f"Invalid type for 'target'. Must be str or list[str], got {type(target)}")
            raise TypeError("Target must be a string or a list of strings.")
