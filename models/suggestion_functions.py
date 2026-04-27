# Suggestion functions to be used with Optuna package

def suggest_logistic_regression_params(trial):
    """
    Suggests hyperparameters for LogisticRegression.
    """
    
    C = trial.suggest_float('C', 1e-4, 1e2, log=True)
    solver = trial.suggest_categorical('solver', ['liblinear', 'lbfgs', 'saga'])
    
    params = {'C': C, 'solver': solver}
    
    if solver == 'liblinear':
        params['penalty'] = trial.suggest_categorical('penalty', ['l1', 'l2'])
    elif solver == 'saga':
        params['penalty'] = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
        if params['penalty'] == 'elasticnet':
            params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
    else: 
        params['penalty'] = 'l2' 
    
    return params


def suggest_ridge_classifier_params(trial):
    """
    Suggests hyperparameters for RidgeClassifier.
    """
    
    alpha = trial.suggest_float('alpha', 1e-4, 1e2, log=True)
    
    return {'alpha': alpha}


def suggest_svm_params(trial):
    """
    Suggests hyperparameters for SVC.
    """

    C = trial.suggest_float('C', 1e-4, 1e2, log=True)
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid'])
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])

    params = {'C': C, 'kernel': kernel, 'gamma': gamma}

    if kernel == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 5)
        params['coef0'] = trial.suggest_float('coef0', 0.0, 1.0)
    elif kernel == 'sigmoid':
        params['coef0'] = trial.suggest_float('coef0', 0.0, 1.0)

    return params


def suggest_mlp_classifier_params(trial):
    """
    Suggests hyperparameters for MLPClassifier.
    """
    
    hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (50, 50), (100, 50)])
    activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
    solver = trial.suggest_categorical('solver', ['adam', 'sgd'])
    alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
    learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)

    return {
    'hidden_layer_sizes': hidden_layer_sizes,
    'activation': activation,
    'solver': solver,
    'alpha': alpha,
    'learning_rate_init': learning_rate_init
    }


def suggest_xgb_params(trial):
    """
    Suggests hyperparameters for XGBoost.
    """
    
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    gamma = trial.suggest_float('gamma', 0.0, 1.0)
    reg_alpha = trial.suggest_float('reg_alpha', 0.0, 1.0)
    reg_lambda = trial.suggest_float('reg_lambda', 0.0, 1.0)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
    tree_method = 'hist'
    eval_metric = 'logloss'

    return {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'gamma': gamma,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'min_child_weight': min_child_weight,
        'tree_method': tree_method,
        'eval_metric': eval_metric,
        'random_state': 42
    }


def suggest_rf_params(trial):
    """
    Suggests hyperparameters for Random Forest.
    """

    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    min_impurity_decrease = trial.suggest_float('min_impurity_decrease', 0.0, 0.5)
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
    oob_score = trial.suggest_categorical('oob_score', [True, False])
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    random_state = 42

    return {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'min_impurity_decrease': min_impurity_decrease,
        'bootstrap': bootstrap,
        'oob_score': oob_score if bootstrap else False,
        'criterion': criterion,
        'max_features': max_features,
        'random_state': random_state
    }


def suggest_extra_trees_params(trial):
    """
    Suggests hyperparameters for ExtraTreesClassifier.
    """
    
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)   
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    min_impurity_decrease = trial.suggest_float('min_impurity_decrease', 0.0, 0.5)
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    random_state = 42

    return {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'min_impurity_decrease': min_impurity_decrease,
        'bootstrap': bootstrap,
        'criterion': criterion,
        'max_features': max_features,
        'random_state': random_state
    }
