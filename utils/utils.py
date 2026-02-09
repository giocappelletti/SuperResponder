from sklearn.pipeline import Pipeline
from logger.logger import logger

def validate_config_param_type(param_name: str, param_value, expected_type: type) -> bool:
    """
    Validates the type of yaml configuration file parameters.
    """

    if param_value is None:
        logger.warning(f"{param_name} is None, skipping validation")
        return False
    
    if not isinstance(param_value, expected_type):
        logger.error(f"{param_name} must be of type {expected_type.__name__}, got {type(param_value).__name__}")
        raise ValueError(f"Invalid type for {param_name}")
        return True



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


def validate_lists_of_strings(lists_of_strings: dict):
    """
    Validates a list of lists of strings.
    """
    for k, v in lists_of_strings.items():
        if validate_config_param_type(k, v, list):
            for s in v:
                validate_config_param_type("string", s, str)


def validate_list_of_strings(list_of_strings: list):
    """
    Validates a list of strings.
    """
    for k, v in list_of_strings.items():
        if validate_config_param_type(k, v, list):
            for s in list_of_strings:
                validate_config_param_type("string", s, str)


def validate_cluster_config(config, section):
        """
        Helper to validate kmedoids params in yaml config file
        """
        config = config.get(section, {})

        random_state = config.get('random_state', 42)        
        validate_config_param_type("random_state", random_state, int)

        metric = config.get('metric', 'euclidean')
        validate_config_param_type("metric", metric, str)
        assert metric in ['euclidean', 'braycurtis', 'unifrac'], \
        f"Unsupported distance metric: {metric}. Supported metrics are 'euclidean', 'unifrac', \
         'braycurtis'"

        if section in ["elbow_silhouette", "analysis"]:
            n_clusters = config.get('n_clusters', 20)
            validate_config_param_type("n_clusters", n_clusters, int)
        
            if section == "elbow_silhouette":
                assert n_clusters > 2, "n_clusters must be greater than 2"
            
            if section == "analysis":
                assert n_clusters > 1, "n_clusters must be greater than 1"

        if section in ["elbow_silhouette", "modenesi", "pca"]:
            njobs = config.get('n_jobs', -1) 
            validate_config_param_type("n_jobs", njobs, int)
            assert njobs > -1, f"Njobs must be either -1 or a positive integer, got {njobs}"
        
        if section in ["kmedoids", "modenesi", "analysis"]:
            save = config.get('save', True)
            validate_config_param_type("save", save, bool)

            save_format = config.get('save_format', 'csv')
            validate_config_param_type("save_format", save_format, str)
            assert save_format in ['csv', 'tsv', 'xlsx'], \
                f"Unsupported save format: {save_format}. Supported formats are 'csv', 'tsv', and 'xlsx'"
        
        if section == "analysis":
            useless_metadata = config.get('useless_metadata', [])
            validate_list_of_strings({"useless_metadata": useless_metadata})
        
        if section in ["elbow_silhouette", "pca", "analysis"]:
            visualize = config.get('visualize', True)
            validate_config_param_type("visualize", visualize, bool)

        if section in ["kmedoids", "modenesi", "pca"]:
            k_values = config.get('k_values', [2])
            validate_config_param_type("k_values", k_values, list)

            for k in k_values:
                validate_config_param_type("k_s", k, int)
                assert k > 1, "k must be a positive integer, got {k}"
        
        return k_values, metric, random_state, save, save_format


def validate_cluster_config(config: dict, section: str) -> dict:
    """
    """
    s_config = config.get(section, {})
    
    section_requirements = {k: list(v.keys()) for k, v in config.items() if isinstance(v, dict)}

    if section not in section_requirements:
        logger.error(f"Unsupported configuration section: {section}")

    params = {}
    needed = section_requirements[section]

    # Logica di validazione centralizzata
    if "random_state" in needed:
        params["random_state"] = s_config.get('random_state', 42)
        validate_config_param_type("random_state", params["random_state"], int)

    if "metric" in needed:
        params["metric"] = s_config.get('metric', 'euclidean')
        validate_config_param_type("metric", params["metric"], str)
        assert params["metric"] in ['euclidean', 'braycurtis', 'unifrac'], f"Unsupported metric: {params['metric']}"

    if "n_clusters" in needed:
        params["n_clusters"] = s_config.get('n_clusters', 20)
        validate_config_param_type("n_clusters", params["n_clusters"], int)
        min_val = 3 if section == "elbow_silhouette" else 2
        assert params["n_clusters"] >= min_val, f"n_clusters must be >= {min_val}"

    if "k_values" in needed:
        params["k_values"] = s_config.get('k_values', [2])
        validate_config_param_type("k_values", params["k_values"], list)
        for k in params["k_values"]:
            validate_config_param_type("k", k, int)
            assert k > 1, f"k must be > 1"

    if "njobs" in needed:
        params["njobs"] = s_config.get('njobs', -1)
        validate_config_param_type("njobs", params["njobs"], int)
        assert params["njobs"] >= -1, "njobs must be >= -1"

    if "save" in needed:
        params["save"] = s_config.get('save', True)
        validate_config_param_type("save", params["save"], bool)

    if "save_format" in needed:
        params["save_format"] = s_config.get('save_format', 'csv')
        validate_config_param_type("save_format", params["save_format"], str)
        assert params["save_format"] in ['csv', 'tsv', 'xlsx'], f"Invalid format: {params['save_format']}"

    if "visualize" in needed:
        params["visualize"] = s_config.get('visualize', True)
        validate_config_param_type("visualize", params["visualize"], bool)

    if "n_components" in needed:
        params["n_components"] = s_config.get('n_components', 2)
        validate_config_param_type("n_components", params["n_components"], int)
        assert params["n_components"] > 0, "n_components must be > 0"

    if "useless_metadata" in needed:
        params["useless_metadata"] = s_config.get('useless_metadata', [])
        validate_list_of_strings("useless_metadata", params["useless_metadata"])

    if "kernel" in needed:
        params["kernel"] = s_config.get('kernel', 'rbf')
        validate_config_param_type("kernel", params["kernel"], str)
    
    if "gamma" in needed:
        params["gamma"] = s_config.get('gamma', 'scale')
        validate_config_param_type("gamma", params["gamma"], str)

    return params