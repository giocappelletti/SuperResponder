import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)


from utils.splitter import Splitter


if __name__ == "__main__":

    config_file = "config/split.yaml"
    splitter = Splitter(config=config_file)

    raw = "datasets/raw_dataset.csv"
    genus = "datasets/genus_r.tsv"
    
    # Split raw dataset in train and test sets
    train_set, test_set = splitter.split_train_test(raw)

    # Filter by cancer type
    train_datasets = splitter.split_by_type(train_set, "train")
    test_datasets = splitter.split_by_type(test_set, "test")
    
    # Extract italians
    modenesi = splitter.split_by_type(genus, "modenesi", {'country': ['Italy']})
    
    genus_train, genus_test = splitter.split_train_test(genus)
    
    # Collapse datasets
    train = splitter.collapse(genus_train, raw, "train_collapse")
    test = splitter.collapse(genus_test, raw, "test_collapse")
    

    