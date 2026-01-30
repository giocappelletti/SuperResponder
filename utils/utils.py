from sklearn.pipeline import Pipeline
from logger.logger import logger

def validate_config_param_type(param_name: str, param_value, expected_type: type):
    """
    Validates the type of yaml configuration file parameters.
    """

    if not isinstance(param_value, expected_type):
        logger.error(f"{param_name} must be of type {expected_type.__name__}, got {type(param_value).__name__}")
        raise ValueError(f"Invalid type for {param_name}")


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

def validate_lists_of_strings(lists_of_strings: list[list]):
    """
    Validates a list of lists of strings.
    """
    
    for l in lists_of_strings:
        validate_config_param_type("list_of_strings", l, list)
        for s in l:
            validate_config_param_type("string", s, str)

def validate_list_of_strings(list_of_strings: list):
    """
    Validates a list of strings.
    """
    
    validate_config_param_type("list_of_strings", list_of_strings, list)
    for s in list_of_strings:
        validate_config_param_type("string", s, str)