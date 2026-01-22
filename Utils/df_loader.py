import pandas as pd
import os

class DataLoader:
    """
    Optimized Utility class for loading microbiome datasets with caching support.
    """
    def __init__(self, cache_dir=".cached_datasets"):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def _sanitize_raw_data(self, df):
        """
        Performs basic data sanitization (type conversion and indexing).
        Handles commas and ensures numeric types for taxonomic features.
        """
        dataset = df.copy()

        if 'samples' in dataset.columns:
            dataset = dataset.set_index('samples')

        if 'response' in dataset.columns:
            dataset = dataset.drop(columns=['response'])

        taxa_cols = [col for col in dataset.columns if col.startswith('k__')]
        meta_cols = [col for col in dataset.columns if col not in taxa_cols]

        dataset[taxa_cols] = (dataset[taxa_cols].replace(',', '.', regex=True).apply(pd.to_numeric, errors='coerce'))

        return dataset, taxa_cols, meta_cols

    def load_dataset(self, path):
        """
        Loads dataset using a Feather cache if available. 
        Otherwise, loads CSV, sanitizes it, and saves a cache for next time.
        """
        # Create a unique cache name based on the original filename
        base_name = os.path.basename(path).replace('.csv', '.feather')
        cache_path = os.path.join(self.cache_dir, base_name)

        if os.path.exists(cache_path):
            print(f"Loading cached dataset from {cache_path}...")
            raw_df = pd.read_feather(cache_path)
            
            # Identify columns for the return values
            dataset, taxa_cols, meta_cols = self._sanitize_raw_data(raw_df)
            return dataset, taxa_cols, meta_cols

        print(f"Cache not found. Loading CSV from {path} (this may take a while)...")
        
        try:
            # Use decimal=',' to speed up initial C-engine parsing
            raw_df = pd.read_csv(path, low_memory=False, decimal=',', engine='c')
            dataset, taxa_cols, meta_cols = self._sanitize_raw_data(raw_df)
            
            # Save to cache for next time (reset_index is needed for Feather)
            dataset.reset_index().to_feather(cache_path)
            
            return dataset, taxa_cols, meta_cols
        
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            raise