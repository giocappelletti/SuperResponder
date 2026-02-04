import pandas as pd
import numpy as np
import os
import csv
import yaml

from logger.logger import logger

class DataLoader:
    """
    Utility class for loading datasets with caching support.
    
    Parameters
    ----------
        cache_dir (str): Path for storing cached datasets.
    """

    def __init__(self, cache_dir=".cached_datasets"):
        self.cache_dir = cache_dir
        self.logger = logger


    def _set_index_to_samples(self, dataset: pd.DataFrame):
        """
        Set "samples" column as index if it exists.
        """
        if 'samples' in dataset.columns:
            dataset = dataset.set_index('samples')
        
        return dataset



    def _sanitize_raw_data(self, df: pd.DataFrame, drop_response=True):
        """
        Performs basic data sanitization (type conversion and indexing) based on the dataset type.
        """
        dataset = df.copy()

        
        self._set_index_to_samples(dataset)
        

        if 'response' in dataset.columns and drop_response:
            dataset = dataset.drop(columns=['response'])

        taxa_cols = [col for col in dataset.columns if col.startswith('k__')]
        meta_cols = [col for col in dataset.columns if col not in taxa_cols]

        dataset[taxa_cols] = dataset[taxa_cols].apply(pd.to_numeric, errors='coerce')

        return dataset, taxa_cols, meta_cols
            

    def _load_unifrac(self, df: pd.DataFrame, orig_dataset_path: str):
        """
        Clean unifrac datasets and intersect with original dataset.
        """

        df.index = df.index.str.strip()
        df.columns = df.columns.str.strip()

        orig_dataset = self.load_dataset(orig_dataset_path, drop_response=False, sanitize=False)

        orig_dataset = self._set_index_to_samples(orig_dataset)

        common_samples = orig_dataset.index.intersection(df.index)
        
        dataset_aligned = orig_dataset.loc[common_samples].copy()
        distance_matrix_aligned = df.loc[common_samples, common_samples].values
        
        return df, dataset_aligned, distance_matrix_aligned, orig_dataset


    def _cache_dataset(self, dataset: pd.DataFrame, cache_path: str):
        """
        Save optimized feather file for next usage.
        """
        self.logger.info(f"Caching dataset to {cache_path} for future use")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        try:
            dataset = dataset.copy()
            dataset.to_feather(cache_path)
            
            self.logger.info(f"Dataset cached successfully")

        except Exception as e:
            self.logger.warning(f"Failed to cache dataset to {cache_path}: {e}")


    def _load_cached_dataset(self, cache_path: str):
        """
        Load dataset from cache if available.
        """
        if os.path.exists(cache_path):            
            self.logger.info(f"Loading cached dataset from {cache_path}")
            
            try:
                cached_dataset = pd.read_feather(cache_path)
            except Exception as e:
                self.logger.error(f"Failed to load cached dataset from {cache_path}: {e}")
                raise
            
            return cached_dataset
    

    def _infer_csv_separator(self, path: str) -> str:
        """
        Infers the separator used in a CSV file by reading a sample of the file.
        Defaults to comma if unable to infer.
        """
        try:
            with open(path, 'r', newline='', encoding='utf-8') as f:
                sample = f.read(4096)  # Read first 4KB
                dialect = csv.Sniffer().sniff(sample, delimiters=',;')
                inferred_separator = dialect.delimiter
                self.logger.info(f"Inferred CSV separator for {os.path.basename(path)}: '{inferred_separator}'")
                return inferred_separator
        except csv.Error:
            self.logger.warning(f"Could not sniff CSV delimiter for {os.path.basename(path)}. Defaulting to comma.")
            return ','
   
    
    def load_dataset(self, 
                     path: str,
                     drop_response=True,
                     sanitize=True, 
                     unifrac=False, 
                     orig_dataset_path : str = None,
                     index_col=None) -> tuple[pd.DataFrame, list, list] | tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """
        Loads dataset using a Feather cached version if available. 
        Otherwise loads file, sanitizes it, and saves a cache for next time.
        
        Parameters
        ----------
            path (str): Path to the dataset file
            drop_response (bool, default=True): Whether to drop the 'response' column if present
            unifrac (bool, default=False): Whether the dataset is a Unifrac distance matrix
            sanitize (bool, default=True): Whether to sanitize the raw data after loading.

        Returns
        -------
            ** pd.Dataframe or tuple (pd.DataFrame, list, list) or tuple (pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame)**
            
            If `unifrac` is False:
                if `sanitize` is True:
                    - dataset (pd.DataFrame): Loaded and sanitized dataset
                    - taxa_cols (list): List of taxa column names
                    - meta_cols (list): List of metadata column names
                else:
                    - dataset (pd.DataFrame): Loaded dataset

            If `unifrac` is True:
                - dataset (pd.DataFrame): unifrac dataset
                - dataset_aligned (pd.DataFrame): Aligned distance matrix with samples
                - distance_matrix_aligned (np.ndarray): Aligned distance matrix as numpy array
                - orig_dataset (pd.DataFrame): Original raw dataset with metadata

        Examples
        --------
        >>> dataset, taxa_cols, meta_cols = dataloader.load_dataset('datasets/raw_dataset.csv')
        """

        # Create a unique cache name based on the original filename
        extension = path.split('.')[-1]
        base_name = os.path.basename(path).replace(extension, 'feather')
        cache_path = os.path.join(self.cache_dir, base_name)   

        if os.path.exists(cache_path):            
            dataset = self._load_cached_dataset(cache_path)

            if unifrac:
                assert orig_dataset_path is not None, "orig_dataset_path must be provided if unifrac=True"
                return self._load_unifrac(dataset, orig_dataset_path)

            if sanitize:
                return self._sanitize_raw_data(dataset, drop_response) 
            else:
                return dataset
        
        if not unifrac:
            self.logger.info(f"Cache not found. Loading {extension} file from {path} (this may take a while)")
        else:
            self.logger.info(f"Loading unifrac file from {path}")

        inferred_separator = self._infer_csv_separator(path) if extension == 'csv' else '\t'

        try:
            if extension in ['tsv', 'csv']:
                dataset = pd.read_csv(path, sep=inferred_separator, low_memory=False, decimal=',', engine='c', index_col=index_col)
            
            else:
                raise ValueError(f"Unsupported file extension: {extension}")
        
        except FileNotFoundError:
            self.logger.error(f"File not found: {path}")
            raise

        except Exception as e:
            self.logger.error(f"Failed to load dataset from {path}: {e}")
            raise
            
        if extension == "csv":
            self._cache_dataset(dataset, cache_path)
        
        if unifrac:
            assert orig_dataset_path is not None, "orig_dataset_path must be provided if unifrac=True"
            return self._load_unifrac(dataset, orig_dataset_path)

        if sanitize:
            return self._sanitize_raw_data(dataset, drop_response)
        else:
            return dataset
        
       