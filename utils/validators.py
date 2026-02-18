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


def validate_dict_of_strings(dict_of_strings: dict):
    """
    Validates a dict of lists of strings.
    """
    for k, v in dict_of_strings.items():
        if validate_config_param_type(k, v, list):
            for s in v:
                validate_config_param_type("string", s, str)


def validate_config(config: dict, section: str) -> dict:
    """
    Validates a specific section of the configuration dictionary and extracts required parameters.

    Parameters
    ----------
        config: dict
            The full configuration dictionary loaded from a YAML file.
        section: str
            The specific section name to validate (e.g., 'split', 'kmedoids').

    Returns
    -------
        dict: A dictionary containing the validated parameters for the requested section.
    """

    s_config = config.get(section, {})

    params = {}
    
    # Get needed params for each section -> avoid checking the entire config file
    section_requirements = {k: list(v.keys()) for k, v in config.items() if isinstance(v, dict)}

    if section not in section_requirements:
        logger.error(f"Unsupported configuration section: {section}")

    needed = section_requirements[section]
    
    if "random_state" in needed:
        params["random_state"] = s_config.get('random_state', 42)
        validate_config_param_type("random_state", params["random_state"], int)

    if "metric" in needed:
        params["metric"] = s_config.get('metric', 'euclidean')
        validate_config_param_type("metric", params["metric"], str)
        assert params["metric"] in ['euclidean', 'braycurtis', 'unifrac'], \
            f"Unsupported metric: {params['metric']}"

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
        assert params["save_format"] in ['csv', 'tsv', 'xlsx'], \
            f"Invalid save format: {params['save_format']}"

    if "visualize" in needed:
        params["visualize"] = s_config.get('visualize', True)
        validate_config_param_type("visualize", params["visualize"], bool)

    if "n_components" in needed:
        params["n_components"] = s_config.get('n_components', 2)
        validate_config_param_type("n_components", params["n_components"], int)
        assert params["n_components"] >= -1, \
            "n_components must be a positive integer or -1 if all components are to be used"
        if params["n_components"] == -1:
            params["n_components"] = None
    

    if "kernel" in needed:
        params["kernel"] = s_config.get('kernel', 'rbf')
        validate_config_param_type("kernel", params["kernel"], str)
        assert params["kernel"] in ['linear', 'poly', 'rbf', 'sigmoid', 'cosine'], \
            f"Unsupported kernel: {params['kernel']}"
    
    if "gamma" in needed:
        params["gamma"] = s_config.get('gamma', 'scale')
        validate_config_param_type("gamma", params["gamma"], float)
    
    if "n_splits" in needed:
        params["n_splits"] = s_config.get('n_splits', 10)
        validate_config_param_type("n_splits", params["n_splits"], int)
        assert params["n_splits"] > 1, "n_splits must be > 1"
    
    if "test_size" in needed:
        params["test_size"] = s_config.get('test_size', 0.2)
        validate_config_param_type("test_size", params["test_size"], float)
        assert 0 <  params["test_size"] < 1, \
            "test_size must be between 0 and 1 (excluded)"
    
    if "shuffle" in needed:
        params["shuffle"] = s_config.get('shuffle', True)
        validate_config_param_type("shuffle", params["shuffle"], bool)
    
    if "stratify" in needed:
        params["stratify"] = s_config.get('stratify', 'response')
        validate_config_param_type("stratify", params["stratify"], str)

    if "path" in needed:
        params["path"] = s_config.get('path', 'datasets/split/')
        validate_config_param_type("path", params["path"], str)

    if "column_types" in needed:
        params["column_types"] = s_config.get('column_types', {})
        validate_dict_of_strings(params["column_types"])

    if "sort_by" in needed:
        params["sort_by"] = s_config.get('sort_by', 'response')
        validate_config_param_type("sort_by", params["sort_by"], str)
    
    if "useless_metadata" in needed:
        params["useless_metadata"] = s_config.get('useless_metadata', [])
        validate_dict_of_strings({"useless_metadata": params["useless_metadata"]})
    
    if "categorical_features" in needed:
        params["categorical_features"] = s_config.get('categorical_features', [])
        validate_dict_of_strings({"categorical_features": params["categorical_features"]})
    
    if "ordinal_features" in needed:
        params["ordinal_features"] = s_config.get('ordinal_features', [])
        validate_dict_of_strings({"ordinal_features": params["ordinal_features"]})
    
    if "numeric_features" in needed:
        params["numeric_features"] = s_config.get('numeric_features', [])
        validate_dict_of_strings({"numeric_features": params["numeric_features"]})
    
    if "age_order" in needed:
        params["age_order"] = s_config.get('age_order', [])
        validate_dict_of_strings({"age_order": params["age_order"]})

    if "max_iter" in needed:
        params["max_iter"] = s_config.get('max_iter', 1000)
        validate_config_param_type("max_iter", params["max_iter"], int)
        assert params["max_iter"] > 0, "max_iter must be > 0"

    if "perplexity" in needed:
        params["perplexity"] = s_config.get('perplexity', 30)
        validate_config_param_type("perplexity", params["perplexity"], int)
        assert params["perplexity"] > 0, "perplexity must be > 0"

    if "learning_rate" in needed:
        params["learning_rate"] = s_config.get('learning_rate', 'auto')
        validate_config_param_type("learning_rate", params["learning_rate"], int)
        assert params["learning_rate"] > 0, "learning_rate must be > 0"

    return params