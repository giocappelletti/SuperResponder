import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.splitter import Splitter


def split_dataset():
    
    # Define file paths 
    raw_dataset = os.path.join(project_root, 'datasets', 'raw_dataset.csv')
    genus = os.path.join(project_root, 'datasets', 'genus_r.tsv')
        
    # Instance splitter object using deafult config file
    splitter = Splitter()

    # Split raw dataset in train and test sets
    train_set, test_set = splitter.split_train_test(raw_dataset)

    # Filter by cancer type (see config file)
    train_datasets = splitter.split_by_type(train_set, "train")
    test_datasets = splitter.split_by_type(test_set, "test")
    
    # Extract italians
    modenesi = splitter.split_by_type(genus, "modenesi", {'country': ['Italy']})
    
    # Split genus dataset in train and test sets
    genus_train, genus_test = splitter.split_train_test(genus)
    
    # Collapse datasets
    train = splitter.collapse(genus_train, raw_dataset, "train_collapse")
    test = splitter.collapse(genus_test, raw_dataset, "test_collapse")


if __name__ == "__main__":
    split_dataset()