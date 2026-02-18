import yaml
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from dataTransformers.data_transformers import CLRTransformer
from logger.logger import logger
from utils.utils import format_pipeline
from utils.validators import validate_config

class Preprocessor:
    """
    Handles preprocessing for both metadata (categorical, ordinal, numeric) 
    and compositional (microbiome) data.

    Parameters
    ----------
        config_path: str
            Path to the YAML configuration file.
        transformation: class
            Class for compositional transformation (e.g., CLR).
        scaler: class
            Scaler class (e.g., StandardScaler).
    """

    def __init__(self, config_path="config/features.yaml", transformer=CLRTransformer, scaler=StandardScaler):

        self.logger = logger

        self.config_path = config_path

        self.logger.info(f"Loading preprocessing configuration from {self.config_path}")

        self.transformation = transformer

        self.logger.info(f"Preprocessor: Using transformation {self.transformation.__name__}")

        self.scaler = scaler

        self.logger.info(f"Preprocessor: Using scaler {self.scaler.__name__}")

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        params = validate_config(self.config, "features")
        self.useless_metadata = params['useless_metadata']
        self.categorical_features = params['categorical_features']
        self.ordinal_features = params['ordinal_features']
        self.numeric_features = params['numeric_features']
        self.age_order = params['age_order']

        self.logger.info(
            f"Loaded features from config file {self.config_path}: \n"
            f"  - Useless Metadata: {', '.join(self.useless_metadata) if self.useless_metadata else 'None'} \n"
            f"  - Categorical Features: {', '.join(self.categorical_features) if self.categorical_features else 'None'} \n"
            f"  - Ordinal Features: {', '.join(self.ordinal_features) if self.ordinal_features else 'None'} \n"
            f"  - Numeric Features: {', '.join(self.numeric_features) if self.numeric_features else 'None'} \n"
            f"  - Age Order: {', '.join(self.age_order) if self.age_order else 'None'} \n"
        )


    def initialize(self, complete_df: pd.DataFrame, use_metadata=True, taxa_cols: list = None) -> tuple:
        """
        Orchestrates the pipeline setup based on the 'test' logic (metadata vs no metadata).
        
        Parameters
        ----------
            complete_df: pd.DataFrame
                The complete DataFrame from DataLoader.
            use_metadata: bool, default=True
                Boolean flag to determine if metadata should be included.
            taxa_cols: list, default=None
                List of taxonomic columns.
        Returns
        -------
            tuple (DataFrame, ColumnTransformer)
                Transformed DataFrame and ColumnTransformer
        """

        if not use_metadata:
            # Only taxa, no preprocessing pipeline for metadata
            return complete_df[taxa_cols], None

        # Filter features present in the dataframe
        features_to_keep = [col for col in complete_df.columns if col not in self.useless_metadata]
        filtered_dataset = complete_df[features_to_keep].copy()

        # Use the existing setup_pipeline logic
        return self._setup_pipeline(
            filtered_dataset, 
            self.useless_metadata, 
            self.categorical_features, 
            self.ordinal_features, 
            self.numeric_features, 
            compositional_features=taxa_cols
        )


    def _fillna_metadata(self, dataset, useless_metadata):
        """
        Fills missing values in key metadata columns to prevent pipeline crashes.
        Operates on a copy to avoid unintended side effects on the original DataFrame.
        """
        # Work on a copy to ensure immutability of the input
        dataframe = dataset.copy()
        
        # Lowercase and fill missing for 'sex'
        if 'sex' not in useless_metadata and 'sex' in dataframe.columns:
            dataframe['sex'] = dataframe['sex'].str.lower().fillna('missing')

        # Convert to string and fill missing for 'age' (ordinal grouping)
        if 'age' not in useless_metadata and 'age' in dataframe.columns:
            dataframe['age'] = dataframe['age'].fillna('missing').astype(str)

        # Fill missing for 'atb' (antibiotics)
        if 'atb' not in useless_metadata and 'atb' in dataframe.columns:
            dataframe['atb'] = dataframe['atb'].fillna('missing')
            
        return dataframe


    def _build_preprocessor_engine(self, categorical_features, ordinal_features, 
                                   numeric_features, compositional_features):
        """
        Internal method to construct the Scikit-learn ColumnTransformer engine.
        """

        # Pipeline for standard categorical features (One-Hot Encoding)
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))
        ])

        # Pipeline for ordinal features like Age groups
        ord_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ordinal', OrdinalEncoder(
                categories=[self.age_order], 
                handle_unknown='use_encoded_value', 
                unknown_value=-1
            )),
        ])

        # Pipeline for numeric features (e.g., sequencing depth metrics)
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', self.scaler() if self.scaler else 'passthrough')
        ])

        # Pipeline for microbiome taxa (Compositional transformation + Scaling)
        comp_pipe = Pipeline([
            ('transformation', self.transformation()),
            ('scaler', self.scaler() if self.scaler else 'passthrough')
        ])

        self.logger.info(
            f"Built preprocessing pipelines:\n"
            f"{format_pipeline(cat_pipe, 'Categorical features')}"
            f"{format_pipeline(ord_pipe, 'Ordinal features')}"
            f"{format_pipeline(num_pipe, 'Numeric features')}"
            f"{format_pipeline(comp_pipe, 'Compositional features')}"
        )

        # Combine all pipelines into a single ColumnTransformer
        return ColumnTransformer([
            ('cat', cat_pipe, categorical_features),
            ('ord', ord_pipe, ordinal_features),
            ('num', num_pipe, numeric_features),
            ('comp', comp_pipe, compositional_features),
        ])

    
    def _setup_pipeline(self, X_dataset, useless_metadata, 
                       categorical_features=None, ordinal_features=None, 
                       numeric_features=None, compositional_features=None):
        """
        Main entry point to prepare the dataset and the preprocessing engine.
        """
        
        # Ensure default empty lists if none provided
        cat_f = categorical_features or []
        ord_f = ordinal_features or []
        num_f = numeric_features or []
        comp_f = compositional_features or []

        # 1. Handle missing values explicitly (returns a new DF)
        X_prepared = self._fillna_metadata(X_dataset, useless_metadata)
        
        # 2. Build the transformer engine
        engine = self._build_preprocessor_engine(cat_f, ord_f, num_f, comp_f)

        return X_prepared, engine


    
