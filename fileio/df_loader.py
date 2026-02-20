import pandas as pd
import numpy as np
import os
import csv
import yaml

from logger.logger import logger
from fileio.serialization import serializer

class DataLoader:
    """
    Utility class for loading datasets with caching to feather files support.
    
    Parameters
    ----------
        cache_dir: str
            Path for storing cached datasets.
    """

    def __init__(self, cache_dir=".cached_datasets"):
        self.cache_dir = cache_dir
        self.logger = logger


    def _set_index_to_samples(self, dataset: pd.DataFrame, index: str):
        """
        Set index column as 'index' if it exists.
        """

        if index in dataset.columns:
            dataset = dataset.set_index(index)
        else:
            self.logger.warning(f"Index column '{index}' not found in the dataset, index not set")
        
        return dataset
            

    def _load_unifrac(self, df: pd.DataFrame, orig_dataset_path: str, index: str = None):
        """
        Clean unifrac datasets and intersect with original dataset.
        """

        df.index = df.index.str.strip()
        df.columns = df.columns.str.strip()

        orig_dataset = self.load_dataset(orig_dataset_path, drop_response=False, sanitize=False)

        if index is not None:
            orig_dataset = self._set_index_to_samples(orig_dataset, index)

        common_samples = orig_dataset.index.intersection(df.index)
        
        dataset_aligned = orig_dataset.loc[common_samples].copy()
        distance_matrix_aligned = df.loc[common_samples, common_samples].values
        
        return df, dataset_aligned, distance_matrix_aligned, orig_dataset


    def _load_cached_dataset(self, cache_path: str):
        """
        Load dataset from cache if available.
        """

        if os.path.exists(cache_path):            
            self.logger.info(f"Loading feather dataset from {cache_path}")
            
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
   
    def _sanitize_raw_data(self, df: pd.DataFrame, drop_response = True, index: str = None):
        """
        Performs basic data sanitization (type conversion and indexing) based on the dataset type.
        """

        dataset = df.copy()

        if index is not None:
            dataset = self._set_index_to_samples(dataset, index)        

        if 'response' in dataset.columns and drop_response:
            dataset = dataset.drop(columns=['response'])

        taxa_cols = [col for col in dataset.columns if col.startswith('k__')]
        meta_cols = [col for col in dataset.columns if col not in taxa_cols]

        dataset[taxa_cols] = dataset[taxa_cols].apply(pd.to_numeric, errors='coerce')

        return dataset, taxa_cols, meta_cols
    

    def load_dataset(self, 
                     path: str,
                     drop_response = True,
                     sanitize = True, 
                     unifrac = False, 
                     orig_dataset_path : str = None,
                     index_col = None,
                     index : str = None,
                     cache_dataset = True,
                     load_from_cache = True) -> tuple[pd.DataFrame, list, list] | tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """
        Loads dataset using a Feather cached version if available. 
        Otherwise loads file, sanitizes it, and saves a cache for next time.
        
        Parameters
        ----------
            path: str 
                Path to the dataset file
            drop_response: bool , default=True
                Whether to drop the 'response' column if present
            unifrac: bool, default=False 
                Whether the dataset is a Unifrac distance matrix
            sanitize: bool, default=True 
                Whether to sanitize the raw data after loading.
            index_col: int
                Wether to set index column.
            set_index: str, default None
                Wether to set dataframe index to specified string. If None, index won't be set.
            cache_dataset: bool, default True
                Wether to save the dataframe in feather format for subsequent use.
            load_from_cache: bool, default True
                Wether to attempt loading the dataframe feather file if it exists.

        Returns
        -------
            ** pd.Dataframe or tuple (pd.DataFrame, list, list) or tuple (pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame)**
            
            If `unifrac` is False:
                if `sanitize` is True:
                    - dataset (pd.DataFrame): Loaded and sanitized dataset
                    - taxa_cols (list): List of taxa columns (numeric values for bacteria)
                    - meta_cols (list): List of metadata column (categorical info about patients)
                else:
                    - dataset (pd.DataFrame): Dataset object

            If `unifrac` is True:
                - dataset (pd.DataFrame): Unifrac dataset
                - dataset_aligned (pd.DataFrame): Aligned distance matrix with samples
                - distance_matrix_aligned (np.ndarray): Aligned distance matrix as numpy array
                - orig_dataset (pd.DataFrame): Original raw dataset with metadata

        Examples
        --------
        >>> dataset, taxa_cols, meta_cols = dataloader.load_dataset('datasets/raw_dataset.csv')
        """

        # Create a unique cache name based on the original filename
        extension = path.split('.')[-1]
        base_name = os.path.basename(path).replace(f".{extension}", "")
        cache_path = path.replace("datasets", self.cache_dir).replace(f".{extension}", f".feather")  

        if load_from_cache and os.path.exists(cache_path):            
            dataset = self._load_cached_dataset(cache_path)
            
            if sanitize:
                return self._sanitize_raw_data(dataset, drop_response, index) 
            else:
                return dataset
        
        if not unifrac:
            self.logger.info(f"Loading {extension} file from {path} (this may take a while)")
        else:
            self.logger.info(f"Loading Unifrac file from {path}")
            index_col = 0

        inferred_separator = self._infer_csv_separator(path) if extension == 'csv' else '\t'

        try:
            if extension in ['tsv', 'csv']:
                dataset = pd.read_csv(path, 
                                      sep=inferred_separator, 
                                      low_memory=False, 
                                      decimal=',', 
                                      engine='c', 
                                      index_col=index_col)
            
            else:
                raise ValueError(f"Unsupported file extension: {extension}")
        
        except FileNotFoundError:
            self.logger.error(f"File not found: {path}")
            raise

        except Exception as e:
            self.logger.error(f"Failed to load dataset from {path}: {e}")
            raise
        
        if cache_dataset:
            serializer.cache_dataset(dataset, os.path.basename(base_name))
        
        if unifrac:
            if orig_dataset_path is None:
                self.logger.error("orig_dataset_path must be provided if unifrac=True")
                raise ValueError
            
            return self._load_unifrac(dataset, orig_dataset_path, index)

        if sanitize:
            return self._sanitize_raw_data(dataset, drop_response, index)
        else:
            return dataset


# Global dataloader instance
dataloader = DataLoader()