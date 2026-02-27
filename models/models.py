
import time
import joblib
from sklearn.cross_decomposition import PLSRegression
import yaml
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve, auc, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from scipy.stats import mode


from logger.logger import logger
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
        

    def evaluate_classifier(self,
                            train_data: pd.DataFrame,
                            train_labels: pd.Series,
                            val_data: pd.DataFrame = None, 
                            val_labels: pd.Series = None, 
                            verbose: int = 1,
                            pipeline: Pipeline = None):
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

        Returns
        -------
        best_model: sklearn.base.BaseEstimator
            The best fitted model (or the original pipeline if no grid search).
        results:
            JSON object with results.
        """

        params = validate_config(self.config, "cross_val")
        
        classifier = params.get('classifier', None)
        transformation = params.get('transformation', None)
        scaler = params.get('scaler', None)
        param_grid = params.get('param_grid', None)
        n_splits = params.get('n_splits', 5)
        njobs = params.get('n_jobs', -1)
        save = params.get('save', False)
        visualize = params.get('visualize', True)
        scorings = params.get('scorings', None)
        n_splits = params.get('n_splits', 5)
        njobs = params.get('n_jobs', -1)
        
        best_params = None
        results = {}
        execution_time = 0
        textual_param_grid = param_grid

        splits_label = 'val' if (val_data is not None and val_labels is not None) else f'{n_splits}-fold CV'

        if pipeline is None:
            pipeline = Pipeline([
                ('transformation', eval(transformation)()),
                ('scaler', eval(scaler)()),
                ('classifier', eval(classifier)())
            ]) 

        # Grid search / fit
        if param_grid is not None:
            self.logger.info("Param Grid:")
            self.logger.info(param_grid)

            param_grid = {key: eval(value) for key, value in param_grid.items()}

            grid_search = GridSearchCV(
                estimator = pipeline,
                param_grid = param_grid,
                scoring = 'accuracy',
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
                n_jobs = njobs,
                verbose = verbose
            )

            start_time = time.perf_counter()
            grid_search.fit(train_data, train_labels)
            end_time = time.perf_counter()
            execution_time = end_time - start_time

            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

        else:
            self.logger.info("No param grid found, fitting model")
            best_model = pipeline
            best_model.fit(train_data, train_labels)

        self.logger.info("Predicting on train set")

        # predictions on train 
        y_train_pred = best_model.predict(train_data)

        # Probability / decision_function for the train
        try:
            self.logger.info("Predicting probabilities on train set")
            y_train_proba = best_model.predict_proba(train_data)[:, 1]
            
        except Exception:
            try:
                self.logger.info("Predicting decision function on train set")
                y_train_proba = best_model.decision_function(train_data)
            
            except Exception:
                self.logger.warning("Unable to compute probabilities or decision function on train set")
                y_train_proba = None

        # Compute CV Metrics
        if scorings is None:
            self.logger.info("Using default scores:")
            scorings = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            self.logger.info(scorings)
        else:
            scoring_list = scorings

        self.logger.info("Computing cross-validation metrics")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=20)

        cv_results = cross_validate(
            best_model,
            train_data,
            train_labels,
            cv = cv,
            scoring = scoring_list,
            n_jobs = njobs
        )

        interp_truepos_list = []
        roc_aucs = []
        prec_rec = []
        y_cv_pred = []
        y_cv_true = [] 
        mean_falseposrate = np.linspace(0, 1, 100) 
        mean_recall = np.linspace(0, 1, 100)        
        fold = 1

        X_train_np = train_data.values if hasattr(train_data, 'values') else train_data
        y_train_np = train_labels.values if hasattr(train_labels, 'values') else train_labels

        for train_idx, test_idx in cv.split(X_train_np, y_train_np):
            X_train_fold = train_data.iloc[train_idx] if hasattr(train_data, 'iloc') else train_data[train_idx]
            X_test_fold  = train_data.iloc[test_idx]  if hasattr(train_data, 'iloc') else train_data[test_idx]
            y_train_fold = y_train_np[train_idx]
            y_test_fold  = y_train_np[test_idx]

            best_model.fit(X_train_fold, y_train_fold)

            try:
                y_scores = best_model.decision_function(X_test_fold)
                y_pred_fold = y_scores > 0
            
            except Exception as e:
                print(e)
                y_scores = best_model.predict_proba(X_test_fold)[:, 1]
                y_pred_fold = y_scores > 0.5
                        
            y_cv_pred.extend(y_pred_fold)
            y_cv_true.extend(y_test_fold)

            if y_scores is not None:
                self.logger.info(f"Computing metrics for fold {fold}")
                precision, recall, _ = precision_recall_curve(y_test_fold, y_scores)
                prec_rec.append(np.interp(mean_recall, recall[::-1], precision[::-1]))
                fpr, tpr, _ = roc_curve(y_test_fold, y_scores)
                roc_auc = auc(fpr, tpr)
                tpr_interp = np.interp(mean_falseposrate, fpr, tpr)
                tpr_interp[0] = 0.0
                interp_truepos_list.append(tpr_interp)
                roc_aucs.append(roc_auc)
            
            fold += 1
        
        mean_tpr = np.mean(interp_truepos_list, axis=0) if interp_truepos_list else None
        mean_tpr[-1] = 1.0 if mean_tpr is not None else None
        mean_fpr = np.linspace(0, 1, 100)

        # Aggregate CV metrics
        cv_metrics_agg = {}
        for score in scoring_list:
            metric_key = f'test_{score}'
            if metric_key in cv_results:
                cv_metrics_agg[score] = round(np.mean(cv_results[metric_key]), 2)

        # Total confusion matrix
        if len(y_cv_true) > 0:
            cm = confusion_matrix(y_cv_true, y_cv_pred)
            plotter._plot_confusion_matrix(cm, visualize, save)

        self.logger.info("Generating results")

        # results for JSON
        results = {
            'transformation': None,
            'scaler': None,
            'reduction': None,
            'param_grid': textual_param_grid,
            'best_params': best_params,
            'fine_tuning_time': round(execution_time, 0),
            'metrics': {
                'train': {
                    'Accuracy': round(accuracy_score(train_labels, y_train_pred), 2),
                    'Precision': round(precision_score(train_labels, y_train_pred), 2),
                    'Recall': round(recall_score(train_labels, y_train_pred), 2),
                    'F1': round(f1_score(train_labels, y_train_pred), 2),
                }
            }
        }

        if y_train_proba is not None:
            try:
                if len(np.unique(train_labels)) > 1: # roc_auc_score requires at least two classes
                    results['metrics']['train']['roc_auc'] = round(roc_auc_score(train_labels, y_train_proba), 2)

            except Exception as e:
                print(e)

        y_val_pred = None
        if val_data is not None and val_labels is not None:
            self.logger.info("Computing metrics on validation set")
            
            y_val_pred = best_model.predict(val_data)
            
            try:
                y_val_proba = best_model.predict_proba(val_data)[:, 1]
            
            except Exception as e:
                print(e)
                try:
                    y_val_proba = best_model.decision_function(val_data)
                except Exception as e:
                    y_val_proba = None
                    print(e)

            val_metrics = {
                'Accuracy': round(accuracy_score(val_labels, y_val_pred), 2),
                'Precision': round(precision_score(val_labels, y_val_pred), 2),
                'Recall': round(recall_score(val_labels, y_val_pred), 2),
                'F1': round(f1_score(val_labels, y_val_pred), 2),
            }

            if y_val_proba is not None:
                try:
                    if len(np.unique(val_labels)) > 1:
                        val_metrics['roc_auc'] = round(roc_auc_score(val_labels, y_val_proba), 2)
                except Exception as e:
                    print(e)
            results['metrics'][splits_label] = val_metrics
            chosen_split_metrics = val_metrics

        else:
            results['metrics'][splits_label] = cv_metrics_agg
            chosen_split_metrics = cv_metrics_agg

        try:
            if 'preprocessor' in pipeline.named_steps:
                if pipeline.named_steps['preprocessor'].transformers:
                    comp_pipeline = pipeline.named_steps['preprocessor'].transformers[3][1]
                else:
                    comp_pipeline = pipeline
            else:
                comp_pipeline = pipeline

            results['transformation'] = str(comp_pipeline.named_steps.get('transformation', comp_pipeline).__class__.__name__)
            results['scaler'] = str(comp_pipeline.named_steps.get('scaler', comp_pipeline).__class__.__name__)
        
        except Exception as e:
            print(e)

        # Prepare metrics for printing
        data_values = [
            classifier,
            f"{float(chosen_split_metrics.get('Accuracy', 0)):.2f}",
            f"{float(chosen_split_metrics.get('Precision', 0)):.2f}",
            f"{float(chosen_split_metrics.get('Recall', 0)):.2f}",
            f"{float(chosen_split_metrics.get('F1', 0)):.2f}",
            f"{float(chosen_split_metrics.get('roc_auc', 0)):.2f}" # Use 'roc_auc' key
        ]

        metrics_df = pd.DataFrame([results['metrics']['train'], 
                                   results['metrics'][splits_label]], 
                                   index=['Train', splits_label]).T


        # 1. Plot Metrics Comparison
        plotter._plot_metrics_comparison(metrics_df, classifier, visualize=visualize, save=save)

        # 2. Plot Confusion Matrices
        y_true_val_plot = val_labels if val_labels is not None else np.array(y_cv_true)
        y_pred_val_plot = y_val_pred if y_val_pred is not None else np.array(y_cv_pred)
        
        plotter._plot_confusion_matrices(
            train_labels, y_train_pred,
            y_true_val_plot, y_pred_val_plot,
            classifier,
            val_set_name=splits_label,
            visualize=visualize,
            save=save
        )

        # 3. Plot ROC Curves
        fpr_train, tpr_train, auc_train = None, None, None
        if y_train_proba is not None and len(np.unique(train_labels)) > 1:
            fpr_train, tpr_train, _ = roc_curve(train_labels, y_train_proba)
            auc_train = roc_auc_score(train_labels, y_train_proba)

        # Cross validation roc curves
        cv_roc_results = None
        if interp_truepos_list:
            # Ensure roc_aucs is not empty before calculating mean and std
            cv_roc_results = {
                'mean_fpr': mean_fpr,
                'mean_tpr': mean_tpr,
                'mean_auc': np.mean(roc_aucs) if roc_aucs else 0.0, # Default to 0.0 if empty
                'std_auc': np.std(roc_aucs) if roc_aucs else None
            }
        
        if fpr_train is not None or cv_roc_results is not None:
            plotter._plot_performance_curves(
                curve_type='roc',
                x_data=fpr_train,
                y_data=tpr_train,
                metric_train_value=auc_train,
                cv_results=cv_roc_results,
                model_name=classifier,
                visualize=visualize,
                save=save
            )

        # 4. Plot PR Curves
        precision_train, recall_train, ap_train, baseline = None, None, None, None
        if y_train_proba is not None and len(np.unique(train_labels)) > 1:
            precision_train, recall_train, _ = precision_recall_curve(train_labels, y_train_proba)
            ap_train = average_precision_score(train_labels, y_train_proba)
            baseline = np.sum(train_labels == 1) / len(train_labels)

        # Cross validation precision recall and ap
        cv_pr_results = None
        if prec_rec:
            cv_pr_results = {
                'mean_recall': mean_recall,
                'mean_precision': np.mean(prec_rec, axis=0),
                'mean_ap': np.mean(roc_aucs) if roc_aucs else 0.0 # Using roc_aucs as a proxy for AP for now
            }

        if precision_train is not None or cv_pr_results is not None:
            plotter._plot_performance_curves(
                curve_type='pr',
                x_data=recall_train,
                y_data=precision_train,
                metric_train_value=ap_train,
                baseline_value=baseline,
                cv_results=cv_pr_results,
                model_name=classifier,
                visualize=visualize,
                save=save
            )

        if save:
            serializer.save_file(results,
                                 "classifier_results",
                                 classifier,
                                 "model_evaluation",
                                 "json")
            serializer.save_file(best_model,
                                 "classifier_results",
                                 classifier,
                                 "model_evaluation",
                                 "pkl")
            serializer.save_file(metrics_df,
                                 "classifier_results",
                                 classifier,
                                 "model_evaluation",
                                 "csv")

        return best_model, results, metrics_df, data_values
    

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
            roc_auc = roc_auc_score(Y_set, avg_proba[:, 1])
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

        Returns
        -------
        vip_df: pd.DataFrame
            DataFrame containing all features and their corresponding VIP scores.
        vip_top25: pd.DataFrame
            DataFrame containing the top 25 features by VIP score.
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
            'VIP': vip_scores
        }).sort_values('VIP', ascending=False)

        vip_top25 = vip_df.head(head)

        plotter._plot_feature_importance(vip_top25, visualize, save)
