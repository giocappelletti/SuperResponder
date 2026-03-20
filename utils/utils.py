import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.pipeline import Pipeline


def sanitize_filename(name):
    """
    Sanitize the filename by replacing spaces with underscores and converting to lowercase.
    """

    return name.lower().replace(" ", "_")


def format_pipeline(pipeline: Pipeline, name):
    """
    Helper method to format a Scikit-learn Pipeline for readable logging.

    Parameters:
        pipeline (Pipeline): L'oggetto pipeline di scikit-learn.
    """

    pipeline_str = f"  - {name} pipeline:\n"
    for step_name, step_estimator in pipeline.steps:
        pipeline_str += f"    - {step_name}: {repr(step_estimator)}\n"
    return pipeline_str


def format_dict(data: dict, indent: int = 0):
    """
    Helper method to format a dictionary (potentially nested) for readable logging.

    Parameters:
        data (dict): Il dizionario da formattare.
    """
    
    formatted_str = ""
    indent_str = "  " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            formatted_str += f"{indent_str}- {key}:\n{format_dict(value, indent + 1)}"
        else:
            formatted_str += f"{indent_str}- {key}: {value}\n"
    return formatted_str


def compute_contingency(dataset: pd.DataFrame, var1: str, var2: str):
        """
        Computes a single contingency table for two given column.
        """
        
        contingency = pd.crosstab(dataset[var1], dataset[var2])

        chi2, p, _, _ = chi2_contingency(contingency)

        summed = contingency.sum().sum()

        contingency_prop  = contingency / summed

        return contingency_prop, chi2, p, summed, contingency.shape 


def get_first_column_index(dataset: pd.DataFrame, start_column: str) -> int | None:
    """
    Returns the integer index of the first column that starts with start_column.
    Returns None if no such column is found.
    """

    for i, col in enumerate(dataset.columns):
        if col.startswith(start_column):
            return i
    return None
