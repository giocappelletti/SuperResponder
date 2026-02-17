import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

from logger.logger import logger
from fileio.df_loader import DataLoader
from fileio.serialization import serializer
from utils.validators import validate_config


class Splitter:
    """
    Helper class that handles dataset splitting in training and test sets, 
    along with data collapsing and selection.

    Parameters
    ----------
        config (str): Path to the YAML configuration file.
    """
    def __init__(self, config = "config/split.yaml"):
        
        with open (config, "r") as f:
            self.config = yaml.safe_load(f)

        self.logger = logger
        self.dataloader = DataLoader()


    def split_train_test(self, dataset: pd.DataFrame | str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits a dataset into training and test sets, resetting indexes.
        
        Parameters
        -------
            dataset (pd.DataFrame or str): The dataframe to split or a path to the dataset file.

        Returns
        -------
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and test sets dataframes.
        """

        params = validate_config(self.config, "split")

        test_size = params['test_size']
        random_state = params['random_state']
        shuffle = params['shuffle']
        stratify = params['stratify']
        save = params['save']
        save_format = params['save_format']
        path = params['path']

        if isinstance(dataset, str):
            dataset = self.dataloader.load_dataset(dataset, sanitize=False, drop_response=False)

        self.logger.info(f"Splitting dataset into training {(1 - test_size)*100}% and test {test_size*100}% datasets")

        train_set, test_set = train_test_split(dataset, test_size=test_size, stratify=dataset[stratify], 
                                            random_state=random_state, shuffle=shuffle)

        train_set = train_set.reset_index(drop=True)
        test_set = test_set.reset_index(drop=True)

        if save:
            train_path = os.path.join(path, "train_set")
            test_path = os.path.join(path, "test_set")
            os.makedirs(path, exist_ok=True)
            serializer.write_to_disk(train_set, train_path, save_format, index=False)
            serializer.write_to_disk(test_set, test_path, save_format, index=False)

        return train_set, test_set


    def split_by_type(self, dataset: pd.DataFrame | str, dname: str = None, column_types: dict = None) -> dict[str, pd.DataFrame]:
        """
        Splits a dataframe by different column types specified in config file.
        
        Parameters
        -------
            dataset (pd.DataFrame or str): The dataframe to split by type or a path to the dataset file
            dname (str): Name of the dataframe to be saved on disk
        
        Returns
        -------
            dict[str, pd.DataFrame]: A dictionary containing the datasets splitted by types
        """

        params = validate_config(self.config, "type_split")

        save = params['save']
        save_format = params['save_format']
        path = params['path']
        
        if column_types is None:
            column_types = params['column_types']   # Dict


        if isinstance(dataset, str):
            dataset = self.dataloader.load_dataset(dataset, sanitize=False, drop_response=False)

        datasets = {}

        for column, values in column_types.items():
            if column in dataset.columns:
                self.logger.info(f"Splitting dataset by {column} type")
                for col_type in values:
                    splitted = dataset[dataset[column] == col_type]
                    datasets[col_type] = splitted.copy()
                    if save:
                        column_path = os.path.join(path, column)
                        os.makedirs(path, exist_ok=True)
                        os.makedirs(column_path, exist_ok=True)
                        splitted_path = os.path.join(column_path, f"{col_type}_{dname}")
                        serializer.write_to_disk(splitted, splitted_path, save_format, index=False)
            else:
                self.logger.warning(f"'{column}' not found in dataset. Skipping split by type for this column")

        return datasets


    def collapse(self, df_collapsed: pd.DataFrame | str, ref_dataset: pd.DataFrame | str, dname: str = None):
        """
        Collapses dataset by filtering out unwanted column types and sorting the results, while simultaneously
        cross-checking the presence of the values in a reference dataframe.

        Parameters
        -------
            df_collapsed (pd.DataFrame or str): The dataframe to collapse or a path to the dataset file.
            ref_dataset (pd.DataFrame or str): The reference dataframe for cross-checking or a path to the dataset file.
            dname (str): Name of the dataframe to be saved on disk.

        Returns
        -------
            pd.DataFrame: The collapsed and filtered dataframe.
            
        """
        
        params = validate_config(self.config, "collapse")

        save = params['save']
        save_format = params['save_format']
        path = params['path']
        sort_by = params['sort_by']
        column_types = params['column_types']   # Dict

        if isinstance(df_collapsed, str):
            df_collapsed = self.dataloader.load_dataset(df_collapsed, sanitize=False, drop_response=False)

        if isinstance(ref_dataset, str):
            ref_dataset = self.dataloader.load_dataset(ref_dataset, sanitize=False, drop_response=False)

        filter_condition = pd.Series(True, index=df_collapsed.index)

        # Dynamically apply filters to dataframe
        for column, value in column_types.items():
            if column in df_collapsed.columns:
                filter_condition = filter_condition & (~df_collapsed[column].isin(value))
            else:
                self.logger.warning(f"Column '{column}' specified in collapse config not found in the dataset. Skipping this filter condition.")

        self.logger.info("Collapsing dataset")
        df = df_collapsed[filter_condition].sort_values(by=sort_by)
        ref_df = ref_dataset.sort_values(by=sort_by)

        collapsed_dataset = df[df[sort_by].isin(ref_df[sort_by])]
        
        if save:
            os.makedirs(path, exist_ok=True)
            os.makedirs(os.path.join(path, "collapsed"), exist_ok=True)
            collapsed_path = os.path.join(path, "collapsed", dname)
            serializer.write_to_disk(collapsed_dataset, collapsed_path, save_format, index=False)

        return collapsed_dataset
    
