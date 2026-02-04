import pandas as pd
import numpy as np
import os
import csv

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
    

    def _sanitize_raw_data(self, df: pd.DataFrame, drop_response=True, unifrac=False):
        """
        Performs basic data sanitization (type conversion and indexing) based on the dataset type.
        """
        dataset = df.copy()

        if not unifrac:
            if 'samples' in dataset.columns:
                dataset = dataset.set_index('samples')

            if 'response' in dataset.columns and drop_response:
                dataset = dataset.drop(columns=['response'])

            taxa_cols = [col for col in dataset.columns if col.startswith('k__')]
            meta_cols = [col for col in dataset.columns if col not in taxa_cols]

            dataset[taxa_cols] = dataset[taxa_cols].apply(pd.to_numeric, errors='coerce')

            return dataset, taxa_cols, meta_cols
        
        else:
            dataset.index = dataset.index.str.strip()
            dataset.columns = dataset.columns.str.strip()

            common_samples = dataset.index.intersection(dataset.index)
            
            dataset_aligned = dataset.loc[common_samples].copy()
            distance_matrix_aligned = dataset.loc[common_samples, common_samples].values

            return dataset, dataset_aligned, distance_matrix_aligned


    def _cache_path(self, path: str, extension: str, drop_response=True):
        """
        Generate cache path based on original dataset path and drop_response flag.
        """
        base_name = os.path.basename(path).replace(extension, 'feather')
        cache_path = os.path.join(self.cache_dir, base_name)

        if not drop_response:
            cache_path = cache_path.replace('.feather', '_with_response.feather')
        
        return cache_path


    def _cache_dataset(self, dataset: pd.DataFrame, cache_path: str):
        """
        Save optimized feather file for next usage.
        """
        self.logger.info(f"Caching dataset to {cache_path} for future use")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        try:
            dataset = dataset.copy()
            dataset.reset_index().to_feather(cache_path)
            self.logger.info(f"Dataset {cache_path} cached successfully")

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
            
            # If an 'index' column exists, set it as the DataFrame index
            if 'index' in cached_dataset.columns:
                cached_dataset = cached_dataset.set_index('index')
            
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
   
    
    def load_dataset(self, path: str, drop_response=True, unifrac=False, sanitize=True) -> tuple[pd.DataFrame, list, list] | tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
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
            **tuple (pd.DataFrame, list, list) or tuple (pd.DataFrame, pd.DataFrame, np.ndarray)**
            
            If `unifrac` is False:
                - dataset (pd.DataFrame): Loaded and sanitized dataset
                - taxa_cols (list): List of taxa column names
                - meta_cols (list): List of metadata column names
            If `unifrac` is True:
                - dataset (pd.DataFrame): Original distance matrix
                - dataset_aligned (pd.DataFrame): Aligned distance matrix with samples
                - distance_matrix_aligned (np.ndarray): Aligned distance matrix as numpy array

        Examples
        --------
        >>> dataset, taxa_cols, meta_cols = dataloader.load_dataset('datasets/raw_dataset.csv')
        """
        # Create a unique cache name based on the original filename
        extension = path.split('.')[-1]
        cache_path = self._cache_path(path, extension, drop_response)        

        if os.path.exists(cache_path):            
            dataset = self._load_cached_dataset(cache_path)

            if sanitize:
                return self._sanitize_raw_data(dataset, drop_response, unifrac) 
            else:
                return dataset, [], []
        
        self.logger.info(f"Cache not found. Loading {extension} file from {path} (this may take a while)")

        inferred_separator = self._infer_csv_separator(path) if extension == 'csv' else '\t'

        try:
            if extension in ['tsv', 'csv']:
                dataset = pd.read_csv(path, sep=inferred_separator, low_memory=False, decimal=',', engine='c')
            
            else:
                raise ValueError(f"Unsupported file extension: {extension}")
        
        except FileNotFoundError:
            self.logger.error(f"File not found: {path}")
            raise

        except Exception as e:
            self.logger.error(f"Failed to load dataset from {path}: {e}")
            raise
            
        self._cache_dataset(dataset, cache_path)
        
        if sanitize:
            return self._sanitize_raw_data(dataset, drop_response, unifrac)
        else:
            return dataset, [], []  
        
       