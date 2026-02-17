import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sklearn.preprocessing import LabelEncoder

from utils.splitter import Splitter
from fileio.df_loader import DataLoader
from dataTransformers.scaler import Scaler
from dimensionality_reduction.dimensionality_reduction import DimensionalityReduction
from visualization.plotting import plotter


if __name__ == "__main__":
    
    dataset_path = "datasets/raw_dataset.csv"
    split_config_file = "config/split.yaml"
    dim_config_file = "config/dimensionality_reduction.yaml"

    dataloader = DataLoader()
    splitter = Splitter(config=split_config_file)
    full_scaler = Scaler()
    split_scaler = Scaler()
    dim_red = DimensionalityReduction(config_path=dim_config_file)
    le = LabelEncoder()

    data = {}

    # Load Dataset, drop response and sanitize
    raw_dataset, taxa_cols, meta_cols = dataloader.load_dataset(dataset_path, drop_response=False, sanitize=True, set_index=False)

    # Split
    train_set, test_set = splitter.split_train_test(raw_dataset)

    # Encode labels
    y = le.fit_transform(raw_dataset['response'])
    y_train = le.transform(train_set['response'])
    y_test = le.transform(test_set['response'])
    
    # Drop response column and select only numeric columns
    dataset = raw_dataset.drop(columns=['response'])[taxa_cols]
    
    # Transform and Scale 
    dataset_transformed = full_scaler.transform_then_fit_transform(dataset)

    train_transformed = split_scaler.transform_then_fit_transform(train_set[taxa_cols])
    
    # Test set must not be fitted
    test_transformed = split_scaler.transform(test_set[taxa_cols])

    # PCA
    components, var_ratio, cum_var_ratio, _= dim_red.PCA(dataset_transformed)

    data['PCA'] = components

    components, per_comp, cum_per_comp = dim_red.KPCA(dataset_transformed)
    
    data['KPCA'] = components

    # Avoid scaling data for PCOA
    components, values, prop_expl, cum_prop_expl = dim_red.PCOA(dataset)

    data['PCOA'] = components

    results = dim_red.TSNE(dataset_transformed)

    data['TSNE'] = results

    results = dim_red.PLS_DA(train_transformed, y_train, test_transformed)

    data['PLS_DA'] = results

    plotter.plot_DR(data, "dimensionality_reduction", 2, per_comp=per_comp)


