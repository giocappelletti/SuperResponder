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
    """

    pipeline_str = f"  - {name} pipeline:\n"
    for step_name, step_estimator in pipeline.steps:
        pipeline_str += f"    - {step_name}: {repr(step_estimator)}\n"
    return pipeline_str


def compute_contingency(dataset: pd.DataFrame, var1: str, var2: str):
        """
        Computes a single contingency table for two given column.
        """
        
        contingency = pd.crosstab(dataset[var1], dataset[var2])

        chi2, p, _, _ = chi2_contingency(contingency)

        summed = contingency.sum().sum()

        contingency_prop  = contingency / summed

        return contingency_prop, chi2, p, summed, contingency.shape 
