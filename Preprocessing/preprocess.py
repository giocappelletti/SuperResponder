import yaml
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Assuming CLRTransformer is imported correctly from your local module
from DataTransformers.data_transformers import CLRTransformer

class Preprocessor:
    """
    Handles preprocessing for both metadata (categorical, ordinal, numeric) 
    and compositional (microbiome) data.
    """

    def __init__(self, config_path="Config/dataset.yaml", transformation=CLRTransformer, scaler=StandardScaler):
        """
        Initializes the preprocessor with specific transformations and scaling.
        
        Args:
            transformation: Class for compositional transformation (e.g., CLR).
            scaler: Scikit-learn scaler class (e.g., StandardScaler).
        """ 
        
        self.config_path = config_path
        self.transformation = transformation
        self.scaler = scaler

        # Load configuration once during initialization
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Load features from YAML
        self.age_order = self.config.get('age_order', [])
        self.useless_metadata = self.config.get('useless_metadata', [])
        self.categorical_features = self.config.get('categorical_features', [])
        self.ordinal_features = self.config.get('ordinal_features', [])
        self.numeric_features = self.config.get('numeric_features', [])

    def initialize(self, complete_df, use_metadata=True, taxa_cols=None):
        """
        Orchestrates the pipeline setup based on the 'test' logic (metadata vs no metadata).
        
        Args:
            complete_df: The complete DataFrame from DataLoader.
            use_metadata: Boolean flag to determine if metadata should be included.
            taxa_cols: List of taxonomic columns.
        """
        if not use_metadata:
            # Only taxa, no preprocessing pipeline for metadata
            return complete_df[taxa_cols], None

        # Filter features present in the dataframe
        features_to_keep = [col for col in complete_df.columns if col not in self.useless_metadata]
        X_set_mod = complete_df[features_to_keep].copy()

        # Use the existing setup_pipeline logic
        return self._setup_pipeline(
            X_set_mod, 
            self.useless_metadata, 
            self.categorical_features, 
            self.ordinal_features, 
            self.numeric_features, 
            compositional_features=taxa_cols
        )

    def _fillna_metadata(self, X_dataset, useless_metadata):
        """
        Fills missing values in key metadata columns to prevent pipeline crashes.
        Operates on a copy to avoid unintended side effects on the original DataFrame.
        """
        # Work on a copy to ensure immutability of the input
        df = X_dataset.copy()
        
        # Lowercase and fill missing for 'sex'
        if 'sex' not in useless_metadata and 'sex' in df.columns:
            df['sex'] = df['sex'].str.lower().fillna('missing')

        # Convert to string and fill missing for 'age' (ordinal grouping)
        if 'age' not in useless_metadata and 'age' in df.columns:
            df['age'] = df['age'].fillna('missing').astype(str)

        # Fill missing for 'atb' (antibiotics)
        if 'atb' not in useless_metadata and 'atb' in df.columns:
            df['atb'] = df['atb'].fillna('missing')
            
        return df

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
        
        Returns:
            X_prepared: The DataFrame with filled missing values.
            preprocessor: The fitted-ready ColumnTransformer.
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


    

