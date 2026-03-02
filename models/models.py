
import joblib
import optuna
from sklearn.calibration import cross_val_predict
import yaml
import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from scipy.stats import mode
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, f1_score, precision_score, recall_score, \
                            roc_auc_score, roc_curve, precision_recall_curve, auc, confusion_matrix

from .suggestion_functions import suggest_xgb_params, suggest_rf_params, suggest_extra_trees_params
from logger.logger import logger
from utils.utils import format_param_grid
from utils.validators import validate_config
from fileio.serialization import serializer
from visualization.plotting import plotter
from preprocessing.data_transformers import CLRTransformer, TSSTransformer
from preprocessing.preprocess import preprocessor


class Models:
    """
    Class to handle model training, evaluation, and ensemble methods.
    It provides tools for cross-validation, grid search, and visualization of results.
    """

    def __init__(self, config_path = "config/models.yaml"):
        self.logger = logger

        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        

    def _build_pipeline(self, transf_name, scaler_name, classifier_name):
        """
        Dynamic pipeline building based on YAML file.
        """

        transformation = (
            CLRTransformer if transf_name == "CLR" else
            TSSTransformer if transf_name == "TSS" else
            None
        )

        scaler = (
            StandardScaler if scaler_name == "Standard" else
            RobustScaler if scaler_name == "Robust" else
            MinMaxScaler if scaler_name == "MinMax" else
            None
        )

        classifier = (
            LogisticRegression if classifier_name == "LogReg" else
            RidgeClassifier if classifier_name == "Ridge" else
            SVC if classifier_name == "SVM" else
            MLPClassifier if classifier_name == "MLP" else
            XGBClassifier if classifier_name == "XGB" else
            RandomForestClassifier if classifier_name == "RF" else
            ExtraTreesClassifier if classifier_name == "ExtraTrees" else
            None
        )
        
        pipeline = Pipeline([
            ('transformation', transformation()),
            ('scaler', scaler()),
            ('classifier', classifier())
        ]) 

        return pipeline


    def _build_param_grid(self, param_grid):
        """
        Dynamic parameter grid building based on YAML file.
        """

        self.logger.info(format_param_grid(param_grid))

        processed_param_grid = {}
        
        for key, value in param_grid.items():
            if isinstance(value, dict) and value.get('_type_') == 'logspace':
                # Build logspace from yaml file
                processed_param_grid[key] = np.logspace(value['start'], value['stop'], value['num'])
            else:
                processed_param_grid[key] = value

        return processed_param_grid


    

    def _get_model_suggestion_func(self, classifier_name: str):
        """
        Dispatches to the correct parameter suggestion function based on classifier name.
        """
        
        suggestion_map = {
            'XGB': suggest_xgb_params,
            'RF': suggest_rf_params,
            'ExtraTrees': suggest_extra_trees_params,
        }

        func = suggestion_map.get(classifier_name)
        
        if func is None:
            self.logger.error(f"No Optuna suggestion function defined for classifier: {classifier_name}")
            raise ValueError(f"Unsupported classifier for Optuna: {classifier_name}")
        
        return func


    def _fit_model_with_grid_search(self, pipeline, train_data, train_labels, param_grid_config, n_folds, njobs, random_state, verbose):
        """
        Fits the model, optionally performing a GridSearchCV.
        """
        best_params = None
        if param_grid_config is not None:
            param_grid = self._build_param_grid(param_grid_config)

            grid_search = GridSearchCV(
                estimator = pipeline,
                param_grid = param_grid,
                scoring = 'accuracy',
                cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state),
                n_jobs = njobs,
                verbose = verbose
            )

            self.logger.info("Fitting model with grid search")
            grid_search.fit(train_data, train_labels)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            self.logger.info("No param grid found, fitting model without grid search")
            best_model = pipeline
            best_model.fit(train_data, train_labels)

        return best_model, best_params


    def _compute_predictions_and_probas(self, model, data):
        """
        Computes predictions and probabilities/decision function for a given model and data.
        """

        y_pred = model.predict(data)
        y_proba = None

        try:
            y_proba = model.predict_proba(data)[:, 1]

        except AttributeError:
            try:
                y_proba = model.decision_function(data)
            
            except AttributeError:
                self.logger.warning("Unable to compute probabilities or decision function.")
        
        return y_pred, y_proba


    def _perform_cross_validation_analysis(self, best_model, train_data, train_labels, n_folds, scorings, njobs, random_state):
        """
        Performs cross-validation and collects data for ROC and PR curves.
        """

        if scorings is None:
            self.logger.info("Using default scores: ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']")
            scorings = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

        self.logger.info("Computing cross-validation metrics")
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        cv_results_raw = cross_validate(
            best_model,
            train_data,
            train_labels,
            cv = cv,
            scoring = scorings,
            n_jobs = njobs
        )

        interp_truepos_list = []
        roc_aucs = []
        prec_rec = []
        y_cv_pred = []
        y_cv_true = []
        mean_falseposrate = np.linspace(0, 1, 100)
        mean_recall = np.linspace(0, 1, 100)

        X_train_np = train_data.values if hasattr(train_data, 'values') else train_data
        y_train_np = train_labels.values if hasattr(train_labels, 'values') else train_labels

        for _, (train_idx, test_idx) in enumerate(cv.split(X_train_np, y_train_np)):
            X_train_fold = train_data.iloc[train_idx] if hasattr(train_data, 'iloc') else train_data[train_idx]
            X_test_fold  = train_data.iloc[test_idx]  if hasattr(train_data, 'iloc') else train_data[test_idx]
            y_train_fold = y_train_np[train_idx]
            y_test_fold  = y_train_np[test_idx]

            best_model.fit(X_train_fold, y_train_fold)
            
            y_pred_fold, y_scores_fold = self._compute_predictions_and_probas(best_model, X_test_fold)
            y_cv_pred.extend(y_pred_fold)
            y_cv_true.extend(y_test_fold)

            if y_scores_fold is not None:
                precision, recall, _ = precision_recall_curve(y_test_fold, y_scores_fold)
                prec_rec.append(np.interp(mean_recall, recall[::-1], precision[::-1]))
                fpr, tpr, _ = roc_curve(y_test_fold, y_scores_fold)
                roc_auc = auc(fpr, tpr)
                tpr_interp = np.interp(mean_falseposrate, fpr, tpr)
                tpr_interp[0] = 0.0
                interp_truepos_list.append(tpr_interp)
                roc_aucs.append(roc_auc)

        # Aggregate CV metrics
        cv_metrics_agg = {score: {'mean': round(np.mean(cv_results_raw[f'test_{score}']), 2), 
                                  'std': round(np.std(cv_results_raw[f'test_{score}']), 2)} 
                          for score in scorings if f'test_{score}' in cv_results_raw}

        return cv_metrics_agg, interp_truepos_list, roc_aucs, prec_rec, y_cv_pred, y_cv_true, mean_falseposrate, mean_recall


    def _aggregate_evaluation_metrics(self, train_labels, y_train_pred, y_train_proba, val_labels, y_val_pred, y_val_proba, cv_metrics_agg, splits_label, classifier_name):
        """
        Aggregates all calculated metrics into a results dictionary and a DataFrame for plotting.
        """

        results = {
            'transformation': None,
            'scaler': None,
            'reduction': None,
            'metrics': {
                'train': {
                    'Accuracy': round(accuracy_score(train_labels, y_train_pred), 2),
                    'Precision': round(precision_score(train_labels, y_train_pred), 2),
                    'Recall': round(recall_score(train_labels, y_train_pred), 2),
                    'F1': round(f1_score(train_labels, y_train_pred), 2),
                }
            }
        }

        if y_train_proba is not None and len(np.unique(train_labels)) > 1:
            try:
                results['metrics']['train']['roc_auc'] = round(roc_auc_score(train_labels, y_train_proba), 2)
            
            except Exception as e:
                self.logger.warning(f"Could not compute ROC AUC for train set: {e}")

        chosen_split_metrics_for_df = {} # Metrics to be used for the DataFrame (mean values)

        if val_labels is not None and y_val_pred is not None:
            val_metrics = {
                'Accuracy': round(accuracy_score(val_labels, y_val_pred), 2),
                'Precision': round(precision_score(val_labels, y_val_pred), 2),
                'Recall': round(recall_score(val_labels, y_val_pred), 2),
                'F1': round(f1_score(val_labels, y_val_pred), 2),
            }
            
            if y_val_proba is not None and len(np.unique(val_labels)) > 1:
                try:
                    val_metrics['roc_auc'] = round(roc_auc_score(val_labels, y_val_proba), 2)
                
                except Exception as e:
                    self.logger.warning(f"Could not compute ROC AUC for validation set: {e}")
            
            results['metrics'][splits_label] = val_metrics # Store validation metrics directly
            chosen_split_metrics_for_df = val_metrics 
        
        else:
            # For CV, cv_metrics_agg contains {'score': {'mean': X, 'std': Y}}
            # We need to extract only the mean for the DataFrame and data_values
            chosen_split_metrics_for_df = {score: data['mean'] for score, data in cv_metrics_agg.items()}
            results['metrics'][splits_label] = cv_metrics_agg # Store full CV metrics (mean and std) in results

        metrics_df = pd.DataFrame([results['metrics']['train'],
                                   chosen_split_metrics_for_df], # Use the extracted means for DataFrame
                                   index=['Train', splits_label]).T

        # Prepare data_values for plotting (e.g., for a table)
        # This list is used in the original evaluate_classifier return, so it needs to be populated.
        data_values = [
            classifier_name,
            f"{float(chosen_split_metrics_for_df.get('Accuracy', 0)):.2f}",
            f"{float(chosen_split_metrics_for_df.get('Precision', 0)):.2f}",
            f"{float(chosen_split_metrics_for_df.get('Recall', 0)):.2f}",
            f"{float(chosen_split_metrics_for_df.get('F1', 0)):.2f}",
            f"{float(chosen_split_metrics_for_df.get('roc_auc', 0)):.2f}"
        ]

        return results, metrics_df, data_values


    def _plot_evaluation_results(self, classifier_name, metrics_df, train_labels, y_train_pred,
                                 y_true_val_plot, y_pred_val_plot, fpr_train, tpr_train, auc_train,
                                 cv_roc_results, precision_train, recall_train, ap_train, baseline,
                                 cv_pr_results, visualize, save, splits_label):
        """
        Generates and saves/displays all evaluation plots.
        """
        # 1. Plot Metrics Comparison
        plotter._plot_metrics_comparison(metrics_df, classifier_name, visualize=visualize, save=save)

        # 2. Plot Confusion Matrices
        plotter._plot_confusion_matrices(
            train_labels, y_train_pred,
            y_true_val_plot, y_pred_val_plot,
            classifier_name,
            val_set_name=splits_label,
            visualize=visualize,
            save=save
        )

        # 3. Plot ROC Curves
        if fpr_train is not None and cv_roc_results is not None:
            plotter._plot_performance_curves(
                curve_type='roc',
                x_data=fpr_train,
                y_data=tpr_train,
                metric_train_value=auc_train,
                cv_results=cv_roc_results,
                model_name=classifier_name,
                visualize=visualize,
                save=save
            )
        else:
            self.logger.warning("No ROC curves could be plotted")

        # 4. Plot PR Curves
        if precision_train is not None and cv_pr_results is not None:
            plotter._plot_performance_curves(
                curve_type='pr',
                x_data=recall_train,
                y_data=precision_train,
                metric_train_value=ap_train,
                baseline_value=baseline,
                cv_results=cv_pr_results,
                model_name=classifier_name,
                visualize=visualize,
                save=save
            )
        else:
            self.logger.warning("No Precision-Recall curves could be plotted")


    def _save_evaluation_results(self, results, best_model, metrics_df, classifier_name, save_format, save):
        """
        Saves the evaluation results (JSON, PKL, metrics DataFrame) to disk.
        """
        if save:
            # JSON Results
            serializer.save_file(results,
                                 "classifier_results",
                                 classifier_name,
                                 "model_evaluation",
                                 "json")

            # Model in pickle format
            serializer.save_file(best_model,
                                 "classifier_results",
                                 classifier_name,
                                 "model_evaluation",
                                 "pkl")

            # Metrics DataFrame
            serializer.save_file(metrics_df,
                                 "classifier_results",
                                 classifier_name,
                                 "model_evaluation",
                                 save_format)

    def _objective(self, trial, x, y, base_pipeline, classifier_name, n_folds, shuffle, random_state, scoring, njobs):
        """
        Objective function for Optuna optimization.
        """

        suggestion_func = self._get_model_suggestion_func(classifier_name)
        classifier_params = suggestion_func(trial)

        # Create a clone of the base_pipeline and set the classifier parameters
        optimized_pipeline = base_pipeline.set_params(**{f'classifier__{k}': v for k, v in classifier_params.items()})

        # Cross-validation
        cv = StratifiedKFold(n_splits = n_folds, shuffle = shuffle, random_state = random_state) 
        score = cross_val_score(optimized_pipeline, x, y, cv = cv, scoring = scoring, n_jobs = njobs).mean() 
        
        return score


    def _predict(self, best_pipeline, x, y, model_name):
        """
        Computes predictions and probabilities on the training set, then performs cross-validation analysis.
        """

        self.logger.info("Predicting on cross-validation folds set")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        y_cv_pred = cross_val_predict(best_pipeline, x, y, cv = cv)
        y_cv_proba_all = cross_val_predict(best_pipeline, x, y, cv = cv, method='predict_proba')

        y_cv_proba_class_1 = y_cv_proba_all[:, 1]

        cv_accuracy = accuracy_score(y, y_cv_pred)
        cv_roc_auc = roc_auc_score(y, y_cv_proba_class_1)

        cv_precision_class_0 = precision_score(y, y_cv_pred, pos_label=0)
        cv_recall_class_0 = recall_score(y, y_cv_pred, pos_label=0)
        cv_f1_class_0 = f1_score(y, y_cv_pred, pos_label=0)

        cv_precision_class_1 = precision_score(y, y_cv_pred, pos_label=1)
        cv_recall_class_1 = recall_score(y, y_cv_pred, pos_label=1)
        cv_f1_class_1 = f1_score(y, y_cv_pred, pos_label=1)

        cm = confusion_matrix(y, y_cv_pred)

        confusion_matrix_table = [
            {
                "actual": 0,
                "predicted_0": int(cm[0, 0]),
                "predicted_1": int(cm[0, 1])
            },
            {
                "actual": 1,
                "predicted_0": int(cm[1, 0]),
                "predicted_1": int(cm[1, 1])
            }
        ]
      
        class_report = classification_report(y, y_cv_pred)
        self.logger.info(f"Classification report:\n{class_report}")

        proba_df = pd.DataFrame(
            y_cv_proba_all,
            columns=["prob_class_0", "prob_class_1"]
        )

        best_params = best_pipeline.named_steps['classifier'].get_params()

        metrics = {
            'cv_accuracy': cv_accuracy,
            'cv_roc_auc': cv_roc_auc,
            'metrics_class_0': {
                'precision': cv_precision_class_0,
                'recall': cv_recall_class_0,
                'f1_score': cv_f1_class_0
            },
            'metrics_class_1': {
                'precision': cv_precision_class_1,
                'recall': cv_recall_class_1,
                'f1_score': cv_f1_class_1
            },
            'best_params': best_params,  
            'confusion_matrix': confusion_matrix_table
        }

        return metrics, proba_df


    def evaluate_classifier(self,
                            train_data: pd.DataFrame,
                            train_labels: pd.Series,
                            val_data: pd.DataFrame = None, 
                            val_labels: pd.Series = None, 
                            verbose: int = 1,
                            pipeline: Pipeline = None,
                            param_grid: dict = None):
        """
        Evaluates a model using cross-validation.
        If param_grid is passed, computes GridSearchCV on train_data/train_labels.
        Otherwise, fits directly the model on train_data.

        If val_data and val_labels are passed, predictions are calculated on the validation set
        and the saved metrics will be those of the validation set (instead of aggregated CV metrics).

        Parameters
        ----------
        train_data: pd.DataFrame
            Training features.
        train_labels: pd.Series
            Training target labels.
        val_data: pd.DataFrame, optional
            Validation features. Defaults to None.
        val_labels: pd.Series, optional
            Validation target labels. Defaults to None.
        verbose: int, optional
            Verbosity level for GridSearchCV. Defaults to 1.
        pipeline: Pipeline, optional
            Pre-built scikit-learn Pipeline. Defaults to None. Overrides YAML configuration
        param_grid: dict, optional
            Parameter grid for GridSearchCV. Defaults to None. Overrides YAML configuration
            
        Returns
        -------
        best_model: sklearn.base.BaseEstimator
            The best fitted model (or the original pipeline if no grid search).
        results:
            JSON object with results, including full CV metrics (mean and std).
        metrics_df: pd.DataFrame
            DataFrame containing the comparison of metrics between train and validation/CV.
        data_values: List
            List of formatted metric values for the chosen split (mean values).
        """
        
        params = validate_config(self.config, "cross_val")

        classifier_name = params.get('classifier', None)
        transformation_name = params.get('transformation', None)
        scaler_name = params.get('scaler', None)
        textual_param_grid = param_grid if param_grid is not None else params.get('param_grid', None)
        n_folds = params.get('n_folds', 5)
        njobs = params.get('njobs', -1) # Corrected from n_jobs to njobs based on validator
        save = params.get('save', False)
        visualize = params.get('visualize', True)
        scorings = params.get('scorings', None)
        random_state = params.get('random_state', 42)
        save_format = params.get('save_format', 'csv')

        splits_label = 'val' if (val_data is not None and val_labels is not None) else f'{n_folds}-fold CV'

        # 2. Build pipeline if not provided
        if pipeline is None:
            pipeline = self._build_pipeline(transformation_name, scaler_name, classifier_name)

        # 3. Fit model (with or without grid search)
        best_model, best_params = self._fit_model_with_grid_search(
            pipeline, train_data, train_labels, textual_param_grid, n_folds, njobs, random_state, verbose
        )

        # 4. Compute predictions and probabilities for training set
        self.logger.info("Predicting on train set")
        y_train_pred, y_train_proba = self._compute_predictions_and_probas(best_model, train_data)

        # 5. Perform cross-validation analysis
        cv_metrics_agg, interp_truepos_list, roc_aucs, prec_rec, y_cv_pred, y_cv_true, mean_falseposrate, mean_recall = \
            self._perform_cross_validation_analysis(best_model, train_data, train_labels, n_folds, scorings, njobs, random_state)

        # Total confusion matrix
        if len(y_cv_true) > 0:
            cm = confusion_matrix(y_cv_true, y_cv_pred)
            plotter._plot_confusion_matrix(cm, visualize, save)

        # 6. Compute predictions and probabilities for validation set (if provided)
        y_val_pred, y_val_proba = None, None
        if val_data is not None and val_labels is not None:
            self.logger.info("Computing metrics on validation set")
            y_val_pred, y_val_proba = self._compute_predictions_and_probas(best_model, val_data)

        # 7. Aggregate all metrics
        results, metrics_df, data_values = self._aggregate_evaluation_metrics(
            train_labels, y_train_pred, y_train_proba,
            val_labels, y_val_pred, y_val_proba,
            cv_metrics_agg, splits_label, classifier_name
        )
        results['param_grid'] = textual_param_grid
        results['best_params'] = best_params

        try:
            if 'preprocessor' in pipeline.named_steps:
                if pipeline.named_steps['preprocessor'].transformers:
                    comp_pipeline = pipeline.named_steps['preprocessor'].transformers[3][1] # Compositional Features
                else:
                    comp_pipeline = pipeline # If preprocessor is the only step or directly the pipeline
            else:
                comp_pipeline = pipeline

            results['transformation'] = str(comp_pipeline.named_steps.get('transformation', comp_pipeline).__class__.__name__)
            results['scaler'] = str(comp_pipeline.named_steps.get('scaler', comp_pipeline).__class__.__name__)
        
        except Exception as e:
            self.logger.warning(f"Could not find preprocessor pipeline: {e}")

        # Prepare data for plotting
        y_true_val_plot = val_labels if val_labels is not None else np.array(y_cv_true)
        y_pred_val_plot = y_val_pred if y_val_pred is not None else np.array(y_cv_pred)

        fpr_train = None
        tpr_train = None
        auc_train = None

        if y_train_proba is not None and len(np.unique(train_labels)) > 1:
            fpr_train, tpr_train, _ = roc_curve(train_labels, y_train_proba)
            auc_train = roc_auc_score(train_labels, y_train_proba)

        mean_tpr = np.mean(interp_truepos_list, axis=0) if interp_truepos_list else None
        if mean_tpr is not None:
            mean_tpr[-1] = 1.0

        cv_roc_results = None
        if interp_truepos_list:
            cv_roc_results = {
                'mean_tpr': mean_tpr,
                'mean_fpr': mean_falseposrate, # Added mean_fpr for plotting
                'mean_auc': np.mean(roc_aucs) if roc_aucs else 0.0, # Default to 0.0 if empty
                'std_auc': np.std(roc_aucs) if roc_aucs else None
            }

        precision_train, recall_train, ap_train, baseline = None, None, None, None
        if y_train_proba is not None and len(np.unique(train_labels)) > 1:
            precision_train, recall_train, _ = precision_recall_curve(train_labels, y_train_proba)
            ap_train = average_precision_score(train_labels, y_train_proba)
            baseline = np.sum(train_labels == 1) / len(train_labels)

        cv_pr_results = None
        if prec_rec:
            cv_pr_results = {
                'mean_recall': mean_recall,
                'mean_precision': np.mean(prec_rec, axis=0),
                'mean_ap': np.mean(roc_aucs) if roc_aucs else 0.0 # Using roc_aucs as a proxy for AP for now
            }

        # 8. Plot evaluation results
        self._plot_evaluation_results(
            classifier_name, metrics_df, train_labels, y_train_pred,
            y_true_val_plot, y_pred_val_plot, fpr_train, tpr_train, auc_train,
            cv_roc_results, precision_train, recall_train, ap_train, baseline,
            cv_pr_results, visualize, save, splits_label
        )

        # 9. Save evaluation results
        self._save_evaluation_results(results, best_model, metrics_df, classifier_name, save_format, save)

        return best_model, results, metrics_df, data_values
    

    def evaluate_classifier_with_optuna(self, 
                                        train_data: pd.DataFrame, 
                                        train_labels: pd.Series, 
                                        pipeline: Pipeline = None):
        """
        Optimizes classifier hyperparameters using Optuna.

        Parameters
        ----------
        train_data: pd.DataFrame
            Training features.
        train_labels: pd.Series
            Training target labels.
        pipeline: Pipeline, optional
            Pre-built scikit-learn Pipeline. If None, it will be built from YAML config.
            This pipeline should include the classifier step.
        n_trials: int, optional
            Number of Optuna trials. Defaults to 100.

        Returns
        -------
        best_pipeline: Pipeline
            The pipeline with the best hyperparameters found by Optuna.
        best_score: float
            The best score achieved during Optuna optimization.
        best_params: dict
            The best hyperparameters found by Optuna.
        """

        params = validate_config(self.config, "optuna")
        
        classifier_name = params.get('classifier', None)
        transformation_name = params.get('transformation', None)
        scaler_name = params.get('scaler', None)
        random_state = params.get('random_state', 42)
        njobs = params.get('njobs', -1)
        n_trials = params.get('n_trials', 100)
        direction = params.get('direction', 'maximize')
        n_folds = params.get('n_folds', 5)
        save = params.get('save', False)
        save_format = params.get('save_format', 'csv')
        shuffle = params.get('shuffle', False)
        scoring = params.get('scoring', 'accuracy')


        if pipeline is None:
            pipeline = self._build_pipeline(transformation_name, scaler_name, classifier_name)

        self.logger.info("Optimizing hyperparameters with Optuna")

        study = optuna.create_study(sampler = optuna.samplers.TPESampler(seed = random_state), direction = direction)
        
        study.optimize(lambda trial: self._objective(trial,
                                                     train_data, 
                                                     train_labels, 
                                                     pipeline, 
                                                     classifier_name, 
                                                     n_folds = n_folds, 
                                                     shuffle = shuffle, 
                                                     random_state = random_state, 
                                                     scoring = scoring,
                                                     njobs = njobs), 
                                                     n_trials = n_trials, 
                                                     n_jobs = njobs)

        best_params = study.best_params
        
        # Set the best parameters on the classifier step of the pipeline
        pipeline.set_params(**{f'classifier__{k}': v for k, v in best_params.items()})
        
        self.logger.info(f"Fitting model with best hyperparameters: {best_params}")
        
        # Fit the best pipeline on the full training data
        pipeline.fit(train_data, train_labels)

        metrics, proba_df = self._predict(pipeline, train_data, train_labels, classifier_name)

        if save:
            # Save the best pipeline
            serializer.save_file(pipeline, "optuna_results", classifier_name, "best_pipeline", "pkl")

            # Save the best parameters
            serializer.save_file(best_params, "optuna_results", classifier_name, "best_params", "json")

            serializer.save_file(proba_df, "optuna_results", classifier_name, "proba_df", save_format)

            serializer.save_file(metrics, "optuna_results", classifier_name, "metrics", "json")
        

        return pipeline, study.best_value, best_params


    def hard_voting_ensemble(self, 
                             models: list, 
                             X_set: pd.DataFrame, 
                             Y_set: pd.Series,
                             labels: list, 
                             compute_roc: bool = True, 
                             taxa_cols: list = None,
                             visualize: bool = True,
                             save: bool = False):
        """
        Performs hard voting ensemble on the given models and dataset.
        
        Parameters
        ----------
        models: list
            List of paths to fitted models saved as pkl files.
        X_set: pd.DataFrame
            Features for prediction
        Y_set: pd.Series
            True labels
        labels:
            List of mapped labels from LabelEncoder output
        compute_roc: bool, default=True
            Whether to compute and plot ROC/PR curves
        taxa_cols: list, optional
            List of taxa columns to filter X_set if necessary
        visualize: bool, default True
            Wether to visualize plots
        save: bool, default False
            Wether to save results and plots to disk

        Returns
        -------
        metrics: dict
            Dictionary containing accuracy, precision, recall, and f1.
        """

        pred_list = []
        proba_list = []

        for path in models:
            try:
                model = joblib.load(path)
                self.logger.info(f"Loaded model from {path}")

            except Exception as e:
                self.logger.error(f"Unable to load model from {path}: {e}")
                raise e

            X_train_full, _ = preprocessor.initialize(
                X_set,
                taxa_cols
            )

            X_train = X_train_full[taxa_cols]
            
            self.logger.info(f"Running prediction for model {path.split('/')[-1]}")
            pred_list.append(model.predict(X_train))
            
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_train)
                proba_list.append(proba)
            else:
                decision = lambda X: 1 / (1 + np.exp(-model.decision_function(X)))
                probs = np.vstack([1 - decision(X_train), decision(X_train)]).T
                proba_list.append(probs)

        pred_array = np.vstack(pred_list)  # shape: (n_models, n_examples)
        final_pred, _ = mode(pred_array, axis=0)  # scipy.stats.mode
        final_pred = final_pred.flatten()

        avg_proba = np.mean(np.stack(proba_list, axis=0), axis=0)  # (n_examples, n_classes)

        self.logger.info(f"Computing models metrics")
        acc = accuracy_score(Y_set, final_pred)
        prec = precision_score(Y_set, final_pred, average='binary')
        rec = recall_score(Y_set, final_pred, average='binary')
        f1 = f1_score(Y_set, final_pred, average='binary')
        roc_auc = roc_auc_score(Y_set, avg_proba[:, 1]) if compute_roc else "---"

        cm = confusion_matrix(Y_set, final_pred)

        metrics = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-score': f1}
        if compute_roc:
            metrics['roc_auc'] = roc_auc

        df_barplot = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])

        confidences = np.max(avg_proba, axis=1)

        correct = (final_pred == Y_set)
        df_conf = pd.DataFrame({'confidence': confidences, 'correct': correct})

        plotter._plot_models_metrics(cm,
                                     df_barplot,
                                     confidences,
                                     df_conf,
                                     labels,
                                     "ensemble",
                                     visualize,
                                     save
                                     )

        if compute_roc:
            fpr, tpr, _ = roc_curve(Y_set, avg_proba[:, 1])
            precision, recall, _ = precision_recall_curve(Y_set, avg_proba[:, 1])
            pr_auc = auc(recall, precision)
        
            plotter._plot_performance_curves(curve_type='roc',
                                  x_data=fpr, 
                                  y_data=tpr, 
                                  metric_train_value=roc_auc, 
                                  visualize=visualize, 
                                  save=save)
            plotter._plot_performance_curves(curve_type='pr',
                                 x_data=recall, 
                                 y_data=precision, 
                                 metric_train_value=pr_auc,
                                 baseline_value=(sum(Y_set) / len(Y_set)),
                                 visualize=visualize, 
                                 save=save)


    def pls_feature_importance(self, 
                           dataset: pd.DataFrame,
                           X_train: pd.DataFrame,
                           taxa_cols: list,
                           n_components: int = 260,
                           head: int = 25,
                           visualize: bool = True,
                           save: bool = False):
        """
        Computes feature importance using PLS regression and VIP scores.        
        
        Parameters
        ----------
        dataset: pd.DataFrame
            The original dataset containing the target 'response'.
        X_train: pd.DataFrame
            The processed training features (used for column names).
        taxa_cols: list
            List of taxonomic feature names.
        n_components: int, default 260
            Number of components for PLS regression.
        head: int, default 25
            Number of top features to display in the plot.
        visualize: bool, default True
            Whether to visualize the feature importance plot.
        save: bool, default False
            Whether to save the VIP scores and plot to disk.
            

        Returns
        -------
        vip_df: pd.DataFrame
            DataFrame containing all features and their corresponding VIP scores.
        vip_top: pd.DataFrame
            DataFrame containing the top 'head' features by VIP score.
        """

        self.logger.info("Computing PLS feature importance")

        # Encode the response variable to numerical format
        le = LabelEncoder()
        y_encoded = le.fit_transform(dataset['response'])
        dataset_TSS = dataset[taxa_cols].div(dataset[taxa_cols].sum(axis=1), axis=0)

        self.logger.info("Fitting PLS regression")
        pls = PLSRegression(n_components=n_components) #260 because 252 cover 90% of variance
        _, _ = pls.fit_transform(dataset_TSS, y_encoded) 

        W = pls.x_weights_   # Shape: (n_features, n_components)
        Q = pls.y_loadings_  # Shape: (n_targets, n_components)
        T = pls.x_scores_    # Projection, Shape: (n_samples, n_components)

        # Compute SSY for every component
        SSY_h = np.sum((T**2) * np.sum(Q**2, axis=0), axis=0)
        total_SSY = np.sum(SSY_h)

        K = W.shape[0]  # Number of features (K)
        A = W.shape[1]  # Number of components (A)
        
        self.logger.info("Computing VIP scores")
        vip_scores = np.sqrt((K / A) * np.sum((W**2) * SSY_h, axis=1) / total_SSY)
        
        vip_df = pd.DataFrame({
            'feature': X_train.columns,
            'coeffs': vip_scores
        }).sort_values('coeffs', ascending=False)

        plotter._plot_feature_importance(vip_df.head(head), visualize, save)
