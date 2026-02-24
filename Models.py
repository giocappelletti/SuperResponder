#!/usr/bin/env python
# coding: utf-8

# ## Inizializzazioni

# In[1]:


import joblib, json,os, time
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# Statistical and Utility
from sklearn.base import clone
from scipy.stats import mode
from sklearn.model_selection import train_test_split

# Data Preprocessing
from skbio.stats.composition import clr, multi_replace #Libreria presente solo su linux
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,OrdinalEncoder
from sklearn.cross_decomposition import PLSRegression
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler

# Modeling and Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline # Usiamo imblearn se vogliamo usare SMOTE
from sklearn.model_selection import GridSearchCV,StratifiedKFold,cross_validate

# Dimensionality Reduction
from sklearn.decomposition import PCA, KernelPCA


# Classifiers
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

# Evaluation Metrics
from sklearn.metrics import (accuracy_score, roc_auc_score, recall_score,precision_score, f1_score, roc_curve, classification_report,
                            make_scorer, precision_recall_curve, average_precision_score,ConfusionMatrixDisplay, auc, confusion_matrix)


# In[ ]:


#Full
path_train = './Datasets/Splitted/Full/train_set.csv'
path_test = './Datasets/Splitted/Full/test_set.csv'
path_dataset = './Datasets/raw_dataset.csv'

#Train e test isolati con solo il Cancro 1 ovvero "MM"
#path_train = './Datasets/Splitted/Cancer/C1/C1_train_set.csv'
#path_test = './Datasets/Splitted/Cancer/C1/C1_test_set.csv'

#Train e test isolati con solo il Cancro 1 ovvero "NSCLC"
#path_train = './Datasets/Splitted/Cancer/C2/C2_train_set.csv'
#path_test = './Datasets/Splitted/Cancer/C2/C2_test_set.csv'


dataset = pd.read_csv(path_dataset)
train_df = pd.read_csv(path_train)
test_df = pd.read_csv(path_test)

#Modenesi
modenesi_df = pd.read_csv("Datasets/Splitted/Full/modenesi.csv")
collapsed_modenesi_df = pd.read_csv("Datasets/Splitted/Collapsed/modenesi_collapsed.csv")

#Dataset con feature collassate
collapsed_train_df = pd.read_csv('Datasets/Splitted/Collapsed/genus_r_train_set.csv')
collapsed_test_df = pd.read_csv('Datasets/Splitted/Collapsed/genus_r_test_set.csv')

# Preparazione target
le = LabelEncoder()
#0 -> NR
#1 -> R
y_train = le.fit_transform(train_df['response'])
y_test = le.transform(test_df['response'])
y_modenesi = le.transform(modenesi_df['response'])
y_collapsed_modenesi = le.transform(collapsed_modenesi_df['response'])
label_map = {'NR': 'Non Responder', 'R': 'Responder'}
labels = [label_map[c] for c in le.classes_]

#y_collapsed_train e y_collapsed_test sono tecnicamente identici a y_train e y_test però in realtà cambia l'ordine dei samples
#quindi vanno usati separatamente
y_collapsed_train = le.transform(collapsed_train_df['response'])
y_collapsed_test = le.transform(collapsed_test_df['response'])


# Full: sia metadati che taxa
X_train_full = train_df.drop(columns=['response'])
X_test_full = test_df.drop(columns=['response'])

X_modenesi_full = modenesi_df.drop(columns=['response'])
X_modenesi_full['atb'] = X_modenesi_full['atb'].fillna('missing') 

X_collapsed_modenesi_full = collapsed_modenesi_df.drop(columns=['response'])
X_collapsed_modenesi_full['atb'] = X_collapsed_modenesi_full['atb'].fillna('missing') 


X_collapsed_train_full = collapsed_train_df.drop(columns=['response'])
X_collapsed_test_full = collapsed_test_df.drop(columns=['response'])

# Estraiamo i taxa
taxa_cols = [col for col in X_train_full.columns if col.startswith('k__')]
collapsed_taxa_cols = [col for col in X_collapsed_train_full.columns if col.startswith('k__')]
meta_cols = [col for col in train_df.columns if col not in taxa_cols]
features_names = dataset.columns.tolist()


# Convertiamole a float in entrambi i dataset
for df in [X_train_full, X_test_full, X_modenesi_full]:
    df[taxa_cols] = df[taxa_cols].replace(',', '.', regex=True).apply(pd.to_numeric)

for df in [X_collapsed_train_full,X_collapsed_test_full, X_collapsed_modenesi_full]:
    df[collapsed_taxa_cols] = df[collapsed_taxa_cols].replace(',', '.', regex=True).apply(pd.to_numeric)

X_train = X_train_full[taxa_cols]
X_test = X_test_full[taxa_cols]

X_modenesi = X_modenesi_full[taxa_cols]
X_collapsed_modenesi = X_collapsed_modenesi_full[collapsed_taxa_cols]

X_collapsed_train = X_collapsed_train_full[collapsed_taxa_cols]
X_X_collapsed_test = X_collapsed_test_full[collapsed_taxa_cols]


# ## Classi e Funzioni utili

# In[3]:


class CLRTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, pseudo_count=None):
        self.pseudo_count = pseudo_count

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.pseudo_count is not None:
            X = X + self.pseudo_count
        else:
            X = multi_replace(X)
        return clr(X)

    def __setstate__(self, state):
        # in caso il pickle non avesse mai salvato pseudo_count
        if 'pseudo_count' not in state:
            state['pseudo_count'] = None
        self.__dict__.update(state)



class TSSTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.div(X.sum(axis=1), axis=0)


class RankingFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=100, ranking_path='Results/v3_14-04-2025/RFE_Ranking/ranking_FullTrain.csv'):
        self.threshold = threshold
        self.ranking_path = ranking_path
        self.selected_features = None

        self.feature_ranking = pd.read_csv(ranking_path, index_col=0)
        self.feature_ranking = self.feature_ranking.squeeze()  # Converti in Series

    def fit(self, X, y=None):

        # Selezioniamo le features con ranking <= threshold
        self.selected_features = self.feature_ranking[self.feature_ranking <= self.threshold].index.tolist()
        #print(f"NUmero feature: {len(self.selected_features)}")
        return self

    def transform(self, X):
        return X[self.selected_features]



class PLSVarianceSelector(BaseEstimator, TransformerMixin):
    def __init__(self, variance_threshold=0.9, n_components = None):
        self.variance_threshold = variance_threshold
        self.n_components = n_components
        self.pls_ = None

    def fit(self, X, y):
        components = X.shape[1] if self.n_components == None else self.n_components

        self.pls_ = PLSRegression(n_components=components)
        self.pls_.fit(X, y)

        # Calcoliamo la varianza spiegata
        X_transformed = self.pls_.transform(X)
        #print(X_transformed.var())
        if self.n_components == None:
            explained_variance = np.var(X_transformed, axis=0)
            total_var = explained_variance.sum()
            explained_ratio = explained_variance / total_var
            cumulative_variance = np.cumsum(explained_ratio) 

            if np.any(cumulative_variance >= self.variance_threshold):
                self.n_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1
                #print(self.n_components)
            else:
                self.n_components = components

        return self

    def transform(self, X):
        X_transformed = self.pls_.transform(X)
        return X_transformed[:, :self.n_components]

""" 
    Classe per utilizzare metodi di dimensionality reduction solo sulle feature tassonomiche anche quando si utilizzano i metadati
    Semplicemente si applica la riduzione solo alle ultime n_comp_features colonne 
    n_comp_features deve valere:
    - 4630 se si sta usando il dataset originale
    - 1128 se si sta usando il dataset collassato
"""
class CompReduction(BaseEstimator, TransformerMixin):
    def __init__(self, n_comp_features, reduction):
        self.n_comp_features = n_comp_features
        self.reduction = reduction

    def fit(self, X, y=None):
        X_comp = X[:, -self.n_comp_features:]
        self.reduction.fit(X_comp)
        return self

    def transform(self, X):
        X_meta = X[:, :-self.n_comp_features]
        X_comp = X[:, -self.n_comp_features:]
        X_comp_reduced = self.reduction.transform(X_comp)
        return np.hstack((X_meta, X_comp_reduced))


# In[4]:


"""
    In caso di utilizzo dei metadati prepara il dataset e crea un preprocessore con i vari encoder
"""
def create_metadata_df(X_train,useless_metadata,  transformation=CLRTransformer, scaler=StandardScaler, categorical_features=[], ordinal_features=[],numeric_features=[], compositional_features=taxa_cols):

    if 'sex' not in useless_metadata:
        X_train['sex'] = X_train['sex'].str.lower().fillna('missing')
        #X_train['sex'] = X_train['sex'].fillna('missing') # nel dataset dei cinesi del 28/05/2025 la colonna sex è solo NA quindi .str da errore

    if 'age' not in useless_metadata:
        X_train['age'] = X_train['age'].fillna('missing').astype(str)

    if 'atb' not in useless_metadata:
        X_train['atb'] = X_train['atb'].fillna('missing')              

    age_order = [
        '10-20', '20-30', '30-40', '40-50', 
        '50-60', '60-70', '70-80', '80-90', '90-100', 'missing'
    ]

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(
            handle_unknown='ignore', 
            drop='if_binary'
        ))
    ])

    ordinal_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ordinal', OrdinalEncoder(
            categories=[age_order], 
            handle_unknown='use_encoded_value', 
            unknown_value=-1
        )),
    ])

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', scaler()) if scaler is not None else ('scaler', None)
    ])

    compositional_trasformer = Pipeline(steps=[
        ('transformation', transformation()),
        ('scaler', scaler()) if scaler is not None else ('scaler', None)
    ])


    preprocessor = ColumnTransformer([
        ('cat', categorical_transformer, categorical_features),
        ('ord', ordinal_transformer, ordinal_features),
        ('num', numeric_transformer, numeric_features),
        ('comp', compositional_trasformer, compositional_features),
    ])


    return X_train, preprocessor


# In[5]:


def evaluate_classifier(X_train, y_train, pipeline, X_validation=None, y_validation=None, param_grid=None, 
                        directory='results', model_name='model', save=True, verbose=1, n_splits=5):
    """
    Se param_grid è passato -> GridSearchCV su X_train/y_train.
    Altrimenti -> fit diretto della pipeline su X_train/y_train.

    Se viene passato X_validation,y_validation -> viene calcolata la previsione su validation
    e le metriche salvate saranno quelle su validation (invece delle metriche aggregate CV).
    """
    y_train_proba, y_train_pred, best_params, best_model = None, None, None, None
    results = {}
    execution_time = 0
    textual_param_grid = param_grid

    # label usata nei json: 'Validation' se passo un validation set, altrimenti 'CV'
    splits_label = 'validation' if (X_validation is not None and y_validation is not None) else f'{n_splits}-fold CV'

    # Grid search / fit
    if param_grid is not None:
        textual_param_grid = param_grid
        param_grid = {key: eval(value) for key, value in param_grid.items()}


        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring='accuracy',
            cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
            n_jobs=-1,
            verbose=verbose
        )

        start_time = time.time()
        grid_search.fit(X_train, y_train)
        end_time = time.time()
        execution_time = end_time - start_time

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

    else:
        best_model = pipeline
        best_model.fit(X_train, y_train)

    if save:
        os.makedirs(directory, exist_ok=True)
        joblib.dump(best_model, os.path.join(directory, f'{model_name}.pkl'))

    # predictions su train 
    y_train_pred = best_model.predict(X_train)

    # Probabilità / decision_function per il train
    try:
        y_train_proba = best_model.predict_proba(X_train)[:, 1]
    except Exception:
        try:
            y_train_proba = best_model.decision_function(X_train)
        except Exception:
            y_train_proba = None

    # Calcolo metriche CV
    scoring_list = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_results = cross_validate(
        best_model,
        X_train,
        y_train,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
        scoring=scoring_list,
        n_jobs=-1
    )
    df_cv = pd.DataFrame(cv_results)
    print("\nRisultati Cross-Validation per fold:")
    print(df_cv[[c for c in df_cv.columns if c.startswith('test_')]].round(4))

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=20)
    tprs = []; roc_aucs = []; mean_fpr = np.linspace(0, 1, 100) # Liste curva RCO
    tprs_pr = []; mean_recall = np.linspace(0, 1, 100) # Liste curva PR
    y_cv_pred = []; y_cv_true = [] # Liste per la  confusion matrix
    fold = 1

    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    y_train_np = y_train.values if hasattr(y_train, 'values') else y_train

    for train_idx, test_idx in cv.split(X_train_np, y_train_np):
        X_train_fold = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
        X_test_fold  = X_train.iloc[test_idx]  if hasattr(X_train, 'iloc') else X_train[test_idx]
        y_train_fold = y_train_np[train_idx]
        y_test_fold  = y_train_np[test_idx]

        best_model.fit(X_train_fold, y_train_fold)

        try:
            y_scores = best_model.decision_function(X_test_fold)
            y_pred_fold = y_scores > 0
        except Exception:
            y_scores = best_model.predict_proba(X_test_fold)[:, 1]
            y_pred_fold = y_scores > 0.5


        cm = confusion_matrix(y_test_fold, y_pred_fold)
        print(f"Confusion matrix - Fold {fold}:\n{cm}\n")
        fold += 1
        y_cv_pred.extend(y_pred_fold)
        y_cv_true.extend(y_test_fold)

        if y_scores is not None:
            precision, recall, _ = precision_recall_curve(y_test_fold, y_scores)
            tprs_pr.append(np.interp(mean_recall, recall[::-1], precision[::-1]))
            fpr, tpr, _ = roc_curve(y_test_fold, y_scores)
            roc_auc = auc(fpr, tpr)
            tpr_interp = np.interp(mean_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)
            roc_aucs.append(roc_auc)

    # metriche CV aggregate
    cv_metrics_agg = {}
    for s in scoring_list:
        metric_key = f'test_{s}'
        if metric_key in cv_results:
            name = s.replace('_', ' ').title().replace('Roc Auc', 'ROC AUC')
            cv_metrics_agg[name] = round(np.mean(cv_results[metric_key]), 2)

    # Confusion matrix complessiva CV (dalle predizioni accumulate sui fold)
    if len(y_cv_true) > 0:
        cm = confusion_matrix(y_cv_true, y_cv_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    else:
        disp = None

    # results per il JSON
    # 'train' (sempre) e poi o 'CV' o 'Validation' a seconda dei parametri passati
    results = {
        'transformation': None,
        'scaler': None,
        'reduction': None,
        'param_grid': textual_param_grid,
        'best_params': best_params,
        'fine_tuning_time': round(execution_time, 0),
        'metrics': {
            'train': {
                'Accuracy': round(accuracy_score(y_train, y_train_pred), 2),
                'Precision': round(precision_score(y_train, y_train_pred), 2),
                'Recall': round(recall_score(y_train, y_train_pred), 2),
                'F1': round(f1_score(y_train, y_train_pred), 2),
            }
        }
    }

    if y_train_proba is not None:
        try:
            results['metrics']['train']['ROC AUC'] = round(roc_auc_score(y_train, y_train_proba), 2)
        except Exception:
            pass

    # Se è passato un validation set -> calcolo metriche su validation e le salvo
    if X_validation is not None and y_validation is not None:
        # predizioni sul validation set
        y_val_pred = best_model.predict(X_validation)
        try:
            y_val_proba = best_model.predict_proba(X_validation)[:, 1]
        except Exception:
            try:
                y_val_proba = best_model.decision_function(X_validation)
            except Exception:
                y_val_proba = None

        val_metrics = {
            'Accuracy': round(accuracy_score(y_validation, y_val_pred), 2),
            'Precision': round(precision_score(y_validation, y_val_pred), 2),
            'Recall': round(recall_score(y_validation, y_val_pred), 2),
            'F1': round(f1_score(y_validation, y_val_pred), 2),
        }
        if y_val_proba is not None:
            try:
                val_metrics['ROC AUC'] = round(roc_auc_score(y_validation, y_val_proba), 2)
            except Exception:
                pass

        results['metrics'][splits_label] = val_metrics
        chosen_split_metrics = val_metrics

    else:
        # nessun validation set: salvo le metriche aggregate CV
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
    except Exception:
        pass

    try:
        headers = ["Modello", "Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
        accuracy  = f"{float(chosen_split_metrics.get('Accuracy', 0)):.2f}"
        precision = f"{float(chosen_split_metrics.get('Precision', 0)):.2f}"
        recall    = f"{float(chosen_split_metrics.get('Recall', 0)):.2f}"
        f1        = f"{float(chosen_split_metrics.get('F1', 0)):.2f}"
        roc_auc   = f"{float(chosen_split_metrics.get('ROC AUC', 0)):.2f}"

        data_values = [model_name, accuracy, precision, recall, f1, roc_auc]
        column_widths = [max(len(h), len(d)) for h, d in zip(headers, data_values)]
        header_cells = [h.ljust(w) for h, w in zip(headers, column_widths)]
        header_row = "| " + " | ".join(header_cells) + " |"
        divider_cells = ["-" * w for w in column_widths]
        divider_row = "| " + " | ".join(divider_cells) + " |"
        data_cells = [d.ljust(w) for d, w in zip(data_values, column_widths)]
        data_row = "| " + " | ".join(data_cells) + " |"
        print(header_row)
        print(divider_row)
        print(data_row)
    except Exception:
        pass

    # se save=True salviamo il JSON con results
    if save:
        with open(os.path.join(directory, f'{model_name}_results.json'), 'w') as f:
            json.dump(results, f, indent=4)

    metrics_df = pd.DataFrame([results['metrics']['train'], results['metrics'][splits_label]], index=['Train', splits_label]).T

    if not save:
        print("\nMetriche complete:")
        print(metrics_df)

    plt.figure(figsize=(10, 6))
    metrics_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Confronto Metriche Train-' + splits_label)
    plt.xlabel('Metrica')
    plt.ylabel('Valore')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(directory, f'{model_name}_metrics_comparison.png'))
    else:
        plt.show()
    plt.close()

    # ROC/PR plotting 
    if len(tprs) > 0:
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(roc_aucs) if len(roc_aucs) > 0 else None
        std_auc = np.std(roc_aucs) if len(roc_aucs) > 0 else None

        if y_train_proba is not None:
            plt.figure()
            try:
                fpr_train, tpr_train, _ = roc_curve(y_train_np, y_train_proba)
                auc_train = roc_auc_score(y_train_np, y_train_proba)
                plt.plot(fpr_train, tpr_train, label=f'Train (AUC = {auc_train:.2f})', lw=2)
            except Exception:
                pass

            if mean_auc is not None:
                plt.plot(mean_fpr, mean_tpr, lw=2, label=f'Cross-Val (AUC = {mean_auc:.2f} ± {std_auc:.2f})')

            plt.plot([0, 1], [0, 1], 'k--', lw=1)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Curva ROC')
            plt.legend(loc='lower right')
            if save:
                plt.savefig(os.path.join(directory, f'{model_name}_roc_curve.png'))
            else:
                plt.show()
            plt.close()

            # PR curve
            try:
                precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_proba)
                ap_train = average_precision_score(y_train, y_train_proba)
                plt.figure()
                plt.plot(recall_train, precision_train, label=f'Train (AP = {ap_train:.2f})')
                if len(tprs_pr) > 0:
                    mean_precision = np.mean(tprs_pr, axis=0)
                    plt.plot(mean_recall, mean_precision, label=f'Cross-Val (AP ~ {np.mean(roc_aucs):.2f})', lw=2)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.legend()
                plt.grid(linestyle='--', alpha=0.5)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.tight_layout()
                if save:
                    plt.savefig(os.path.join(directory, f'{model_name}_pr_curve.png'))
                else:
                    plt.show()
                plt.close()
            except Exception:
                pass

    # Confusion matrix: train + CV/Validation
    try:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, ax=ax[0])
        ax[0].set_title('Train Confusion Matrix')
        if disp is not None:
            disp.plot(cmap='Blues', ax=ax[1])
            ax[1].set_title(f'{splits_label} Confusion Matrix')
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(directory, f'{model_name}_confusion_matrix.png'))
        else:
            plt.show()
        plt.close()
    except Exception:
        pass

    return best_model


# In[6]:


""" 
    Restituisce il dataset modificato a seconda del test scelto:
    le feature tassonomiche sono presenti in tutti i test
    Test 0: senza metadati
    Test 1: tutti i metadati
    Test 2: solo metadati selezionati (country, therapy, cancer, sex, atb e age)
    Test 3: metadati selezionati e categorie minoritarie aggregate
    Test 4: metadati selezionati + Platform
    Test 5: metadati selezionati + LibraryLayout
"""
def initialiaze_metadata(X_full, test, medoid=None, taxa_cols=taxa_cols):
    match test:
        case 0: #Senza metadati
            return X_full[taxa_cols], None
        case 1:
            useless_metadata = ['samples', 'SRA', 'BioProject', 'ORR']
            features = [col for col in  X_full.columns if col not in useless_metadata]
            X_set_mod = X_full[features].copy()
            #Se il dataset è diviso per medoid non utilizziamo la colonna 'medoids' poichè è costante
            #Se il dataset è diviso per submedoid non utilizziamo la colonna 'medoids' e 'submedoids' poichè sono costanti
            #(Le colonne vanno anche tolte dal dataset)
            base_categorical = ['country', 'continent', 'therapy', 'cancer','LibraryLayout', 'Platform', 'sex', 'atb']
            if medoid == "M1" or medoid == "M2":
                categorical_features = [*base_categorical, 'submedoids']
            elif medoid == "M1_1" or medoid == "M1_2" or medoid == "M2_1" or medoid == "M2_2" :   
                categorical_features = [*base_categorical]
            else:# medoid = None, usiamo sia medoid che submedoid
                categorical_features = [*base_categorical, 'medoids','submedoids']

            ordinal_features = ['age']
            numeric_features = ['avgSpotLen', 'Bases']

        case 2: # Metadati selezionati

            useless_metadata = ['samples', 'SRA', 'BioProject', 'ORR', 'Platform', 'Bases','avgSpotLen', 'LibraryLayout', 'medoids', 'submedoids', 'continent']
            selected_metadata = [col for col in X_full.columns if col not in useless_metadata and col not in taxa_cols]
            #print(f"Metadati selezionati: {selected_metadata}")# ['country', 'sex', 'age', 'atb', 'cancer', 'therapy']
            features = [col for col in  X_full.columns if col not in useless_metadata]
            X_set_mod = X_full[features].copy()
            categorical_features = ['country', 'therapy', 'cancer','sex', 'atb']
            ordinal_features = ['age']
            numeric_features = []
        case 3: # Metadati selezionati e categorie minoritarie aggregate
            useless_metadata = ['samples', 'SRA', 'BioProject', 'ORR', 'Platform', 'Bases','avgSpotLen', 'LibraryLayout', 'medoids', 'submedoids', 'continent']
            features = [col for col in  X_full.columns if col not in useless_metadata]
            X_set_mod = X_full[features].copy()

            X_set_mod['country'] = X_set_mod['country'].where(
                X_set_mod['country'].isin(['France', 'North America', 'United Kingdom']), 
                'Other'
            )

            X_set_mod['therapy'] = X_set_mod['therapy'].where(
                X_set_mod['therapy'].isin(['PDL-1', 'CTLA-4+PD-1', 'PD-1']), 
                'Other'
            )
            categorical_features = ['country', 'therapy', 'cancer','sex', 'atb']
            ordinal_features = ['age']
            numeric_features = []
        case 4: # Metadati selezionati + Platform
            useless_metadata = ['samples', 'SRA', 'BioProject', 'ORR', 'Bases','avgSpotLen', 'LibraryLayout', 'medoids', 'submedoids', 'continent']
            features = [col for col in  X_full.columns if col not in useless_metadata]
            X_set_mod = X_full[features].copy()
            categorical_features = ['country', 'therapy', 'cancer','sex', 'atb', 'Platform']
            ordinal_features = ['age']
            numeric_features = []
        case 5: # Con LibraryLayout
            useless_metadata = ['samples', 'SRA', 'BioProject', 'ORR', 'Bases','avgSpotLen', 'medoids', 'submedoids', 'continent', 'Platform']
            features = [col for col in  X_full.columns if col not in useless_metadata]
            X_set_mod = X_full[features].copy()
            categorical_features = ['country', 'therapy', 'cancer','sex', 'atb', 'LibraryLayout']
            ordinal_features = ['age']
            numeric_features = []



    X_set_mod, preprocessor = create_metadata_df(X_set_mod,useless_metadata = useless_metadata,categorical_features = categorical_features,
                                                    ordinal_features = ordinal_features,numeric_features = numeric_features,
                                                    compositional_features=taxa_cols)

    return X_set_mod, preprocessor


# In[ ]:


""" 
    Stampa gli score in formato markdown
    format = 1 -> score in markdown
    format = 2 -> score in latex
"""
def printMetrics(set_name, model_name, accuracy,precision,recall,f1,roc_auc, printHeader=True, ensemble=False, format=1):
    if isinstance(roc_auc, (float, int)):
        roc_auc = round(roc_auc, 2)
    data_values = [f"{float(accuracy):.2f}", f"{float(precision):.2f}", f"{float(recall):.2f}", f"{float(f1):.2f}", f"{roc_auc}"]
    headers = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
    if ensemble:
        headers = ["Test            ", *headers]
        data_values = [set_name, *data_values]
    else:
        headers = ["Modello     ", *headers]
        data_values = [model_name, *data_values]

    if format == 1:
        # Larghezza massima per ogni colonna
        column_widths = [max(len(h), len(d)) for h, d in zip(headers, data_values)]

        header_cells = [h.ljust(w) for h, w in zip(headers, column_widths)]
        header_row = "| " + " | ".join(header_cells) + " |"

        divider_cells = ["-" * w for w in column_widths]
        divider_row = "| " + " | ".join(divider_cells) + " |"

        data_cells = [d.ljust(w) for d, w in zip(data_values, column_widths)]
        data_row = "| " + " | ".join(data_cells) + " |"
        if printHeader:
            print(header_row)
            print(divider_row)
        print(data_row)

    if format == 2:
        data_row = " & ".join(data_values) + " \\\\"
        print(data_row)
        print("\hline")


# In[8]:


"""
    Calcola le metriche (accuracy, precision, recall, f1, roc_auc) e genera i grafici
"""
def computeGraph(y_set, y_pred, y_proba, set_name, output_dir, model_name,compute_roc=True, printHeader=True, ensemble=False, format=1):
    y_proba_pos = y_proba[:,1]

    # Calcoliamo le  metriche
    acc = accuracy_score(y_set, y_pred)
    prec = precision_score(y_set, y_pred, average='binary')
    rec = recall_score(y_set, y_pred, average='binary')
    f1 = f1_score(y_set, y_pred, average='binary')
    roc_auc = roc_auc_score(y_set, y_proba_pos) if compute_roc else "---"

    # Stampiamo le metriche
    printMetrics(set_name, model_name, acc, prec, rec, f1, roc_auc, printHeader=printHeader, ensemble=ensemble, format=format)


    # --- Matrice di confusione ---
    cm = confusion_matrix(y_set, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predetto'); plt.ylabel('Reale')
    plt.title(f'Matrice di Confusione ({set_name})')
    plt.savefig(os.path.join(output_dir, f'confusion_matrix.png'))
    plt.close()

    # --- Barplot delle metriche ---
    metrics = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-score': f1}
    if compute_roc:
        metrics['ROC AUC'] = roc_auc
    df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    plt.figure(figsize=(6,4))
    sns.barplot(x='Metric', y='Value', data=df, palette='viridis')
    plt.ylim(0,1)
    plt.title(f'Metriche ({set_name})')
    plt.savefig(os.path.join(output_dir, f'barplot_metrics.png'))
    plt.close()

    # --- Istogramma delle confidenze ---
    confidences = np.max(y_proba, axis=1)
    plt.figure(figsize=(6,4))
    sns.histplot(confidences, bins=10, kde=True, color='skyblue')
    plt.xlabel('Massima Probabilità (Confidenza)')
    plt.ylabel('Conteggio')
    plt.title(f'Istogramma delle Confidenze ({set_name})')
    plt.savefig(os.path.join(output_dir, f'hist_confidences.png'))
    plt.close()



    # --- Boxplot: corrette vs errate ---
    correct = (y_pred == y_set)
    df_conf = pd.DataFrame({'confidence': confidences, 'correct': correct})
    plt.figure(figsize=(6,4))
    sns.boxplot(x='correct', y='confidence', data=df_conf)
    plt.xlabel('Correttezza')
    plt.xticks([0,1], ['Errate', 'Corrette'])
    plt.ylabel('Confidenza')
    plt.title(f'Confidenze: corrette vs errate ({set_name})')
    plt.savefig(os.path.join(output_dir, f'box_confidences.png'))
    plt.close()

    # --- Curve ROC e Precision-Recall ---
    if compute_roc:
        fpr, tpr, _ = roc_curve(y_set, y_proba_pos)
        precision, recall, _ = precision_recall_curve(y_set, y_proba_pos)
        roc_auc = roc_auc_score(y_set, y_proba_pos)
        pr_auc = auc(recall, precision)
        # ROC Curve
        plt.figure()
        plt.plot(fpr, tpr, label=f'Modello (AUC={roc_auc:.3f})')
        plt.plot([0,1],[0,1], linestyle='--', color='gray', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Curva ROC ({set_name})')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'roc_curve.png'))
        plt.close()
        # Precision-Recall Curve
        baseline = sum(y_set==1)/len(y_set)
        plt.figure()
        plt.plot(recall, precision, label=f'Modello (AUC={pr_auc:.3f})')
        plt.plot([0,1],[baseline,baseline], linestyle='--', color='gray', label='Baseline')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Curva Precision-Recall ({set_name})')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'pr_curve.png'))
        plt.close()


# In[9]:


"""
    Esegue un ensemble dei modelli: predizione per maggioranza (moda: Hard Voting Ensemble)
"""
def ensemble_evaluate(models_path, X_set, y_set, set_name, base_dir, first,compute_roc=True,taxa_cols=taxa_cols, format=1):
    # Creazione della directory di output
    output_dir = os.path.join(base_dir, "Ensemble", set_name)
    os.makedirs(output_dir, exist_ok=True)

    # Predizioni di ogni modello e probabilità predette
    pred_list = []
    proba_list = []
    for name, model_dict in models_path.items():
        path = model_dict["path"]
        test = model_dict["test"]
        model = joblib.load(path)
        X_train_mod, _ = initialiaze_metadata(X_set, test, taxa_cols=taxa_cols)

        # Previsione delle etichette con il modello
        y_pred = model.predict(X_train_mod)
        pred_list.append(y_pred)
        # Previsione delle probabilità (se disponibile)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_train_mod)
            proba_list.append(proba)
        else:
            decision = lambda X: 1 / (1 + np.exp(-model.decision_function(X)))
            probs = np.vstack([1 - decision(X_train_mod), decision(X_train_mod)]).T
            proba_list.append(probs)

    pred_array = np.vstack(pred_list)  # shape: (n_modelli, n_campioni)
    final_pred, _ = mode(pred_array, axis=0)  # scipy.stats.mode
    final_pred = final_pred.flatten()
    #print(final_pred)

    avg_proba = np.mean(np.stack(proba_list, axis=0), axis=0)  # (n_campioni, n_classi)


    computeGraph(y_set, final_pred, avg_proba, set_name, output_dir,"Ensemble", compute_roc=compute_roc, ensemble=True, printHeader=first, format=format)


# In[10]:


"""
    Carica i modelli in modesl_path e effettua le predizioni su X_set
"""
def model_evaluate(models_path, X_set, y_set, set_name,base_dir, compute_roc=True, taxa_cols=taxa_cols, format=1):

    all_predictions = {}
    all_confidences = {}
    for model_name, model_dict in models_path.items():
        path = model_dict["path"]
        test = model_dict["test"]
        model = joblib.load(path)
        X_set_mod, _ = initialiaze_metadata(X_set, test, taxa_cols=taxa_cols)
        output_dir = os.path.join(base_dir,"Predictions", set_name,model_name)
        os.makedirs(output_dir, exist_ok=True)

        y_pred = model.predict(X_set_mod)
        #print(y_pred)
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_set_mod)
        else:
            decision = lambda X: 1 / (1 + np.exp(-model.decision_function(X)))
            y_proba = np.vstack([1 - decision(X_set_mod), decision(X_set_mod)]).T

        confidence = np.max(y_proba, axis=1)
        all_predictions[model_name] = y_pred
        all_confidences[model_name] = confidence
        printHeader = True if model_name == "LR" else False
        computeGraph(y_set, y_pred, y_proba, set_name,output_dir,model_name,compute_roc=compute_roc, printHeader=printHeader, format=format)
    confidence_df = pd.DataFrame()
    for model_name, _ in models_path.items():
        temp_df = pd.DataFrame({
            'Modello': model_name,
            'Correttezza': all_predictions[model_name] == y_set,
            'Confidenza': all_confidences[model_name]
        })
        confidence_df = pd.concat([confidence_df, temp_df])



    plt.figure(figsize=(12, 6))
    sns.boxplot(x="Modello", y="Confidenza", hue="Correttezza", data=confidence_df)
    plt.title(f"Distribuzione confidenza ({set_name})")
    print(base_dir)

    plt.savefig(
        os.path.join(base_dir, "Predictions", set_name, "all_confidences.png"),
        dpi=300,       
        bbox_inches="tight"  
    )
    plt.close()



# In[11]:


def create_medoids_set(medoids=[1, 2, "1_1", "1_2", "2_1", "2_2"], predict=False):
    sets = {}
    for medoid in medoids:
        if not predict:
            medoid_train_df = train_df[train_df['medoids'] == medoid]
            if medoid_train_df.empty:
                medoid_train_df = train_df[train_df['submedoids'] == medoid]
                medoid_train_df.drop(columns=["medoids","submedoids"], inplace=True)
            else:
                medoid_train_df.drop(columns=["medoids"], inplace=True)

            y_medoid_train = le.transform(medoid_train_df['response'])


            X_medoid_train_full = medoid_train_df.drop(columns=['response'])
            X_medoid_train_full[taxa_cols] = X_medoid_train_full[taxa_cols].replace(',', '.', regex=True).apply(pd.to_numeric)

            sets[f"M{medoid}"] = {"X_train": X_medoid_train_full, "y_train": y_medoid_train}
        else:
            medoid_test_df = test_df[test_df['medoids'] == medoid] #Test isolato
            medoid_modenesi_df = modenesi_df[modenesi_df['medoids'] == medoid] #Modenesi isolati
            if medoid_test_df.empty:
                medoid_test_df = test_df[test_df['submedoids'] == medoid]
                medoid_test_df.drop(columns=["medoids","submedoids"], inplace=True)

                medoid_modenesi_df = modenesi_df[modenesi_df['submedoids'] == medoid]
                medoid_modenesi_df.drop(columns=["medoids","submedoids"], inplace=True)
            else:
                medoid_test_df.drop(columns=["medoids"], inplace=True)
                medoid_modenesi_df.drop(columns=["medoids"], inplace=True)

            y_medoid_test = le.transform(medoid_test_df['response'])

            X_medoid_test_full = medoid_test_df.drop(columns=['response'])
            X_medoid_test_full[taxa_cols] = X_medoid_test_full[taxa_cols].replace(',', '.', regex=True).apply(pd.to_numeric)

            y_medoid_modenesi = le.transform(medoid_modenesi_df['response'])
            X_medoid_modenesi_full = medoid_modenesi_df.drop(columns=['response'])
            X_medoid_modenesi_full[taxa_cols] = X_medoid_modenesi_full[taxa_cols].replace(',', '.', regex=True).apply(pd.to_numeric)

            sets[f"M{medoid}"] ={"Test Isolato": {"X_test": X_medoid_test_full, "y_test": y_medoid_test},
                                "Test Intero": {"X_test": X_test_full, "y_test": y_test},
                                "Modenesi": {"X_test": X_modenesi_full, "y_test": y_modenesi},
                                "Modenesi Isolati": {"X_test": X_medoid_modenesi_full, "y_test": y_medoid_modenesi},}
    return sets


# # Primi Run dei modelli con CLR, TSS e senza trasformazione

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

''' 
Codiche che addestra e valuta il modello scelto con la trasformazione scelta.
Per scegliere il modello e la trasformazione, modificare le variabili model_name e transformation_name
'''
model_name = "MLP"  # Può valere: LR, RLS, SVM e MLP
transformation_name = "TSS"  # Può valere: CLR, TSS e None
save = True

directory = f'Results/v2_01-04-2025/Models/Full/{model_name}/{transformation_name}'
directory = f'Results/v2_01-04-2025/Models/Full/TSS_new'
transformation = (
    CLRTransformer if transformation_name == "CLR" else
    TSSTransformer if transformation_name == "TSS" else
    None
)

classifier = (
    LogisticRegression if model_name == "LR" else
    RidgeClassifier if model_name == "RLS" else
    SVC if model_name == "SVM" else
    MLPClassifier if model_name == "MLP" else
    None
)

pipeline = Pipeline([
    ('transformation', transformation()),
    ('scaler', StandardScaler()),
    ('classifier', classifier())
])

param_grid = (
    {'classifier__penalty': "['l2']",
      "classifier__C": "np.logspace(-3,3,150)",
      'classifier__random_state': "[42]"} if model_name == "LR" else
    {"classifier__alpha": "np.logspace(-5,5,300)","classifier__random_state": "[42]"} if model_name == "RLS" else
    {
        "classifier__kernel": "['linear', 'rbf']",
        "classifier__C": "np.logspace(-4,-4,10)",
        "classifier__gamma": "['scale', 'auto', 0.001, 0.01]",
        "classifier__random_state": "[42]"
    } if model_name == "SVM" else
    {
        "classifier__hidden_layer_sizes": "[(64,), (128,), (256,), (64,32), (128,64), (256,128)]",
        "classifier__activation": "['relu', 'tanh']",
        "classifier__alpha": "[0.0001, 0.001, 0.01, 0.1, 1.0]",
        "classifier__solver": "['adam', 'sgd']",
        "classifier__learning_rate_init": "[0.001, 0.01]",
        "classifier__batch_size": "[16, 32, 64]",
        "classifier__max_iter": "[300]",
        "classifier__early_stopping": "[True]",
        "classifier__random_state": "[42]"
    } if model_name == "MLP" else
    {
        "classifier__alpha": "np.logspace(-5,5,300)",
        "classifier__random_state": "[42]"
    } if model_name == "RLS" else
    {}
)

evaluate_classifier(
    X_train,
    y_train,
    save=save,
    pipeline=pipeline,
    param_grid=param_grid,
    directory=f'{directory}/{model_name}',
    model_name=f"{model_name}"
)


# # Partial Least Squares (PLS)

# ## Variable Importance in Projection (PLS feature importance)
# 
# $$
# \text{VIP}_j = \sqrt{ \frac{K}{A} \cdot \frac{\sum\limits_{h=1}^A (w_{jh}^2 \cdot \text{SSY}_h)}{\sum\limits_{h=1}^A \text{SSY}_h} }
# $$
# 
# Dove
# - K: Numero totale di feature (covariate)
# - A: Numero di componenti PLS
# - $w_{jh}$: Peso della feature $j$ nella componente $h$ 
# - $SSY_h$: Somma dei quadrati della varianza di $Y$ spiegata dalla componente h

# In[7]:


y = np.array(dataset['response'])
dataset_TSS = dataset[taxa_cols].div(dataset[taxa_cols].sum(axis=1), axis=0)

colors = ['green' if label == 0 else 'red' for label in y]
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Non Responder',markerfacecolor='red', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Responder',markerfacecolor='green', markersize=10)
]


# In[ ]:


pls = PLSRegression(n_components=260)#260 perchè 252 coprono il 90% della varianza
X_pls, _ = pls.fit_transform(dataset_TSS, y) 

# Estraiamo pesi (W), loadings Y (Q), e punteggi X (T)
W = pls.x_weights_   # Shape: (n_features, n_components)
Q = pls.y_loadings_  # Shape: (n_targets, n_components)
T = pls.x_scores_    # Proiezione, Shape: (n_samples, n_components)

# Calcoliamo SSY per ogni componente
SSY_h = np.sum((T**2) * np.sum(Q**2, axis=0), axis=0)
total_SSY = np.sum(SSY_h)

K = W.shape[0]  # Numero di feature (K)
A = W.shape[1]  # Numero di componenti (A)
vip_scores = np.sqrt((K / A) * np.sum((W**2) * SSY_h, axis=1) / total_SSY)
vip_df = pd.DataFrame({
    'Feature': X_train.columns,
    'VIP': vip_scores
}).sort_values('VIP', ascending=False)
vip_top25 = vip_df.head(25)
plt.figure(figsize=(10, 6))
plt.barh(vip_top25["Feature"], vip_top25["VIP"], color="skyblue")
plt.xlabel("VIP Score")
plt.ylabel("Feature")
plt.title("Top 25 Feature Importance PLS tramite VIP Scores")
plt.gca().invert_yaxis()  
plt.show()


# ## Pipeline con PLS

# In[ ]:


#Numero componenti che coprono il 90% della varianza sul train: 195
#gamma scale = 1/(n_features*X.var()), con 200 component = 1/ 200 * 9.5 = 0.0005
#kernel lineare e C tra 00002 e 00003 scende un po' lo score del train ma non generalizza bene
directory = 'Results/v2_01-04-2025/Models/Full/PLS'
name= 'PLS_SVM'
pipeline = [
    ('transformation', CLRTransformer()),
    ('scaler', StandardScaler()),
    ('reduction', PLSVarianceSelector(n_components=200))
    ('classifier', SVC())
]
param_grid = {
        "classifier__C": "np.logspace(-4,4,10)",
        "classifier__kernel": "['linear', 'rbf']",
        "classifier__gamma": "['scale', 'auto', 0.001, 0.01]",
        "classifier__random_state": "[42]"
    }

model = evaluate_classifier(X_train, y_train, pipeline=Pipeline(pipeline),
                        save=False, param_grid=None, directory=f'{directory}/TSS', model_name=f'{name}_TSS')


# In[ ]:


full_names = ["PLS_LR_CLR", "PLS_RLS_CLR", "SVM_CLR", "MLP_None"]
models = [
    ("LR", "Results/v2_01-04-2025/Models/Full/LR/CLR/LR_best_model.pkl"),
    ("RLS", "Results/v3_14-04-2025/Models/Smote/RLS/RLS_CLR_SMOTE_best_model.pkl"),
    ("SVM", "Results/v3_14-04-2025/Models/Smote/SVM/SVM_CLR_SMOTE_best_model.pkl"),
    ("MLP", "Results/v3_14-04-2025/Models/Smote/MLP/MLP_None_SMOTE_best_model.pkl")
]
'''  
models = [
    ("LR", "Results/v2_01-04-2025/Models/Full/LR/CLR/LR_best_model.pkl"),
    ("RLS", "Results/v3_14-04-2025/Models/Metadata/Selected/Aggregated/RLS/RLS_best_model.pkl"),
    ("SVM", "Results/v3_14-04-2025/Models/Metadata/Selected/SVM/SVM_best_model.pkl"),
    ("MLP", "Results/v3_14-04-2025/Models/Metadata/Selected/Aggregated/MLP/MLP_best_model.pkl")
]'''



# Directory di salvataggio dei grafici
directory = 'Results/v3_14-04-2025/BestModelsGraphs'
os.makedirs(directory, exist_ok=True)

# Liste dove accumulare metriche e dati per plot aggregati
metrics = []
roc_data = {}
pr_data = {}
cv_predictions = {}
params = {}

# Funzione per estrarre i parametri principali dal modello (in base al tipo)
def get_important_params(name, model):
    # Estraiamo il classificatore dalla pipeline
    estimator = model.steps[-1][1]
    param_dict = {}
    if name == 'LR':  # Logistic Regression
        param_dict['Penalty'] = estimator.penalty
        param_dict['Solver'] = estimator.solver
    elif name == 'RLS':  # Regularized Least Squares 
        param_dict['Alpha'] = estimator.alpha
    elif name == 'SVM':
        param_dict['C'] = estimator.C
        param_dict['Kernel'] = estimator.kernel
        param_dict['Gamma'] = estimator.gamma
    elif name == 'MLP':
        param_dict['Hidden Layers'] = estimator.hidden_layer_sizes
        param_dict['Activation'] = estimator.activation
        param_dict['Alpha'] = estimator.alpha
        param_dict['Solver'] = estimator.solver
        param_dict['Learning Rate'] = estimator.learning_rate_init
        param_dict['Batch Size'] = estimator.batch_size
        param_dict['Max Iter'] = estimator.max_iter
    return param_dict

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20)


for name, path in models:
    model = joblib.load(path)
    params[name] = get_important_params(name, model)

    # Inizializziamo le liste per accumulare le predizioni e le metriche per ogni fold
    y_cv_true = []
    y_cv_pred = []
    y_cv_prob = []  
    fold_tprs = []
    fold_roc_aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fold_pr_aucs = []

    for train_idx, test_idx in cv.split(X_test, y_test):

        if hasattr(X_test, 'iloc'):
            X_train_fold, X_test_fold = X_test.iloc[train_idx], X_test.iloc[test_idx]
        else:
            X_train_fold, X_test_fold = X_test[train_idx], X_test[test_idx]
        if hasattr(y_test, 'iloc'):
            y_train_fold, y_test_fold = y_test.iloc[train_idx], y_test.iloc[test_idx]
        else:
            y_train_fold, y_test_fold = y_test[train_idx], y_test[test_idx]

        cloned_model = clone(model)
        cloned_model.fit(X_train_fold, y_train_fold)
        try:
            y_scores = cloned_model.decision_function(X_test_fold)
            y_pred_fold = y_scores > 0
        except AttributeError:
            y_scores = cloned_model.predict_proba(X_test_fold)[:, 1]
            y_pred_fold = y_scores > 0.5

        y_cv_true.extend(y_test_fold)
        y_cv_pred.extend(y_pred_fold)
        y_cv_prob.extend(y_scores)

        fpr, tpr, _ = roc_curve(y_test_fold, y_scores)
        fold_auc = auc(fpr, tpr)
        fold_roc_aucs.append(fold_auc)
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        fold_tprs.append(tpr_interp)

        precision_fold, recall_fold, _ = precision_recall_curve(y_test_fold, y_scores)
        pr_auc = auc(recall_fold, precision_fold)
        fold_pr_aucs.append(pr_auc)

    mean_tpr = np.mean(fold_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_roc_auc = np.mean(fold_roc_aucs)
    std_roc_auc = np.std(fold_roc_aucs)
    mean_pr_auc = np.mean(fold_pr_aucs)

    acc = accuracy_score(y_cv_true, y_cv_pred)
    prec = precision_score(y_cv_true, y_cv_pred)
    rec = recall_score(y_cv_true, y_cv_pred)
    f1_val = f1_score(y_cv_true, y_cv_pred)
    roc_auc_cv = roc_auc_score(y_cv_true, y_cv_prob)

    metrics.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1_val,
        'ROC AUC': roc_auc_cv
    })

    roc_data[name] = (mean_fpr, mean_tpr, mean_roc_auc)
    precision_all, recall_all, _ = precision_recall_curve(y_cv_true, y_cv_prob)
    pr_auc_all = auc(recall_all, precision_all)
    pr_data[name] = (precision_all, recall_all, pr_auc_all)
    cv_predictions[name] = y_cv_pred

# DataFrame con metriche (cross validation)
df_metrics = pd.DataFrame(metrics).set_index('Model')

# Grafico comparativo delle metriche
fig, ax = plt.subplots(figsize=(12, 8))
df_metrics.plot(kind='bar', rot=0, width=0.8, ax=ax)
plt.title("Confronto Metriche di Performance (CV)", pad=20, fontsize=14)
plt.ylabel("Score", fontsize=12)
plt.xlabel("Modelli", fontsize=12)
plt.ylim(0, 1)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.,
            frameon=True, title="Metriche", fontsize=10)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig(os.path.join(directory, 'metrics.jpg'))
plt.show()

# Plot aggregati delle ROC curve per ogni modello
plt.figure(figsize=(10, 8))
for name, (fpr, tpr, auc_val) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.2f})")
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curve ROC (CV)')
plt.legend()
plt.savefig(os.path.join(directory, 'roc_curve.jpg'))
plt.show()

# Plot aggregati delle Precision-Recall curve per ogni modello
plt.figure(figsize=(10, 8))
for name, (precision, recall, auc_val) in pr_data.items():
    plt.plot(recall, precision, label=f"{name} (AUC = {auc_val:.2f})")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (CV)')
plt.legend()
plt.savefig(os.path.join(directory, 'pr_curve.jpg'))
plt.show()

# Matrici di confusione in griglia (usando le predizioni CV aggregate)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()
for idx, (name, y_pred) in enumerate(cv_predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], cmap='Blues')
    axes[idx].set_title(name)
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(directory, 'conf_matrix_grid.jpg'))
plt.show()

# Stampa dei parametri principali dei modelli
print("\nParametri principali dei modelli:")
for name, model_params in params.items():
    print(f"\n=== {name} ===")
    for k, v in model_params.items():
        print(f"{k}: {v}")

# Salvataggio dei parametri principali su file txt
output_file = "descrizione_modelli.txt"
with open(os.path.join(directory, output_file), "w") as f:
    f.write("Nomi completi:\n\t" + "\n\t".join(full_names) + "\n\n")
    f.write("Parametri principali dei modelli:")
    for name, model_params in params.items():
        f.write(f"\n=== {name} ===\n")
        for k, v in model_params.items():
            f.write(f"{k}: {v}\n")
print(f"Parametri salvati in {output_file}")


# # Errors Analysis
# 
# Codice per caricare i modelli e calcolare le confidenze e la matrice di sovrapposizione degli errori (ovvero quanti errori, in percentuale, fanno uguali identici) sulla cross-validation.
# 
# Per cambiare modelli basta cambiare models_path

# In[ ]:


''' 
base_path = 'Results/v2_01-04-2025'
models_path = {"LR": {"path": f"{base_path}/Models/Full/CLR/LR/LR.pkl", "test": 0},
                "RLS": {"path": f"{base_path}/Models/Full/CLR/RLS/RLS.pkl", "test": 0},
                "SVM": {"path": f"{base_path}/Models/Full/CLR/SVM/SVM.pkl", "test": 0},
                "MLP": {"path": f"{base_path}/Models/Full/CLR/MLP/MLP.pkl", "test": 0}
}
'''

base_path = 'Results/v4_28-04-2025/BestModelsGraphs'
directory = "Results/v4_28-04-2025/BestModelsGraphs"
models_path = {"LR": {"path": f"{base_path}/LR.pkl", "test": 0},
                "RLS": {"path": f"{base_path}/RLS.pkl", "test": 3},
                "SVM": {"path": f"{base_path}/SVM.pkl", "test": 2},
                "MLP": {"path": f"{base_path}/MLP.pkl", "test": 4},
                "RandomForest": {"path": f"{base_path}/RandomForest.pkl", "test": 0},
                "ExtraTrees": {"path": f"{base_path}/ExtraTrees.pkl", "test": 1},
                "XGBoost": {"path": f"{base_path}/XGBoost.pkl", "test": 4},
}

'''
base_path = 'Results/v3_14-04-2025'
directory = "Results/v3_14-04-2025/Errors"
models_path = {"LR": {"path": f"{base_path}/BestModelsGraphs/LR.pkl", "test": 0},
                "RLS": {"path": f"{base_path}/BestModelsGraphs/RLS.pkl", "test": 0},
                "SVM": {"path": f"{base_path}/BestModelsGraphs/SVM.pkl", "test": 0},
                "MLP": {"path": f"{base_path}/BestModelsGraphs/MLP.pkl", "test": 0},
                "RandomForest": {"path": f"{base_path}/BestModelsGraphs/RF.pkl", "test": 0},
                "ExtraTrees": {"path": f"{base_path}/BestModelsGraphs/XT.pkl", "test": 0},
                "XGBoost": {"path": f"{base_path}/BestModelsGraphs/XGB.pkl", "test": 0},
}
'''
directory = f"{base_path}/Errors"

n_splits = 5 
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
models = {}



# Strutture per accumulare risultati
all_predictions = {name: [] for name, _ in models_path.items()}
all_errors = {name: [] for name, _ in models_path.items()}
all_confidences = {name: [] for name, _ in models_path.items()}
all_y_true = []

X_train_np = X_train_full.values if hasattr(X_train_full, 'values') else X_train_full
y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
for train_idx, test_idx in skf.split(X_train_np, y_train_np):
    X_train_fold, X_test_fold = X_train_full.iloc[train_idx], X_train_full.iloc[test_idx]
    y_train_fold, y_test_fold = y_train_np[train_idx], y_train_np[test_idx]

    all_y_true.extend(y_test_fold)

    for name, val in models_path.items():
        path = val['path']
        test = val['test']
        model = joblib.load(path)
        X_train_mod, _ = initialiaze_metadata(X_train_fold.copy(), test)
        X_test_mod, _ = initialiaze_metadata(X_test_fold.copy(), test)

        cloned_model = clone(model)

        cloned_model.fit(X_train_mod, y_train_fold)

        y_pred = cloned_model.predict(X_test_mod)
        all_predictions[name].extend(y_pred)
        all_errors[name].extend(y_pred != y_test_fold)

        if hasattr(cloned_model, "predict_proba"):
            probas = cloned_model.predict_proba(X_test_mod)
            confidence = np.max(probas, axis=1)
        elif hasattr(cloned_model, "decision_function"):
            decision = cloned_model.decision_function(X_test_mod)
            probas = 1 / (1 + np.exp(-decision))  
            confidence = np.maximum(probas, 1 - probas)
        all_confidences[name].extend(confidence)

plt.figure(figsize=(15, 10))

for idx, (name, confidence) in enumerate(all_confidences.items(), 1):
    plt.subplot(2, 4, idx)
    sns.histplot(confidence, bins=30, kde=True)
    plt.title(f"Distribuzione confidenza - {name}")
    plt.xlabel("Valore confidenza")

plt.tight_layout()
plt.savefig(f"{directory}/confidence_distribution.jpg")
plt.show()


all_y_true = np.array(all_y_true)
for name, _ in models_path.items():
    all_predictions[name] = np.array(all_predictions[name])
    all_errors[name] = np.array(all_errors[name])
    all_confidences[name] = np.array(all_confidences[name])



# Matrice di sovrapposizione errori
error_matrix = pd.DataFrame(index=[name for name, _ in models_path.items()], columns=[name for name, _ in models_path.items()])
for i in error_matrix.index:
    for j in error_matrix.columns:
        error_matrix.loc[i, j] = np.sum(np.logical_and(all_errors[i], all_errors[j]))
diagonal = np.diag(error_matrix)  # Estrarre la diagonale (errori totali di ciascun modello)
error_matrix_percent = error_matrix.div(diagonal, axis=0) * 100  
error_matrix_percent = error_matrix_percent.apply(pd.to_numeric, errors='coerce')
# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(error_matrix_percent, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Matrice di sovrapposizione degli errori (Cross-Validation)")
plt.savefig(f"{directory}/error_overlap_cv.jpg")
plt.show()



# Analisi confidenza
confidence_df = pd.DataFrame()

for name, _ in models_path.items():
    temp_df = pd.DataFrame({
        'Modello': name,
        'Correttezza': all_predictions[name] == all_y_true,
        'Confidenza': all_confidences[name]
    })
    confidence_df = pd.concat([confidence_df, temp_df])

# Boxplot confidenza
plt.figure(figsize=(12, 6))
sns.boxplot(x="Modello", y="Confidenza", hue="Correttezza", data=confidence_df)
plt.title("Distribuzione confidenza (Cross-Validation)")
plt.savefig(f"{directory}/confidence_cv.jpg")
plt.show()


# # Feature Importance SVM
# 
# Codice che effettua la feature importance tramite un modello **SVM** già allenato il cui percorso è definito in model_path.
# 
# È adattato per un modello che è stato allenato con il test 2 (metadati selezionati) ma funziona anche per il test 3, in caso di un test diverso bisogna adattare il codice e cambiare le liste delle feature: **categorical_features**, **ordinal_features**, **useless_metadata** e eventualmente aggiungere **numeric_features** se presenti, in modo da ricostruire correttamente i nomi delle feature

# In[ ]:


# ================== CONFIG ==================
n_importance = 20   # Numero di feature da mostrare nei grafici
save = False         # Se True salva i plot e i CSV, altrimenti no
# ============================================

def run_feature_importance(model_path,output_dir, dataset_name, taxa_cols=taxa_cols):
    pipeline = joblib.load(model_path)
    svm_model = pipeline.named_steps[pipeline.steps[-1][0]]
    coefficients = svm_model.coef_.flatten()

    # OneHotEncoder per i nomi delle feature categoriche
    onehot = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']

    categorical_features = ['country', 'therapy', 'cancer', 'sex', 'atb']
    ordinal_features = ['age']  
    useless_metadata = ['samples', 'SRA', 'BioProject', 'ORR']

    feature_names_cat = onehot.get_feature_names_out(categorical_features)


    all_feature_names = np.concatenate([feature_names_cat, ordinal_features, taxa_cols])

    df = pd.DataFrame({'Feature': all_feature_names, 'Coefficient': coefficients})

    if save:
        os.makedirs(output_dir, exist_ok=True)

    def save_results(data, suffix, n, plt_title):
        plt.figure(figsize=(12, min(6 + 0.2*n, 40)))
        plt.barh(data['Feature'].head(n), data['Coefficient'].head(n))
        plt.title(plt_title)
        plt.xlabel('Coefficient Value')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        if save:
            plt.savefig(f"{output_dir}/top_{n}_{suffix}.png", bbox_inches='tight')
        plt.show()

        if save:
            data.to_csv(f"{output_dir}/top_{n}_{suffix}.csv", index=False)

    # Top 100 positivo
    save_results(
        df.sort_values('Coefficient', ascending=False),
        'positive',
        100,
        f'Top 100 Feature - Impatto Positivo (Dataset {dataset_name})'
    )

    # Top 100 negativo
    save_results(
        df.sort_values('Coefficient', ascending=True),
        'negative',
        100,
        f'Top 100 Feature - Impatto Negativo (Dataset {dataset_name})'
    )


    df['Absolute'] = df['Coefficient'].abs()
    top_abs = df.sort_values('Absolute', ascending=False).head(n_importance)
    all_df = df.sort_values('Absolute', ascending=False)
    top_abs['Direction'] = np.where(top_abs['Coefficient'] >= 0, 'Positive', 'Negative')
    all_df['Direction'] = np.where(all_df['Coefficient'] >= 0, 'Positive', 'Negative')

    plt.figure(figsize=(12, 4))
    colors = ['green' if d == 'Positive' else 'red' for d in top_abs['Direction']]
    plt.barh(top_abs['Feature'], top_abs['Coefficient'], color=colors)
    plt.title(f'Top {n_importance} Feature Dataset {dataset_name} - Impatto Assoluto')
    plt.xlabel('Valore coefficiente')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    if save:
        plt.savefig(f"{output_dir}/top_{n_importance}_absolute_{dataset_name}.png", bbox_inches='tight', dpi=300)
    plt.show()

    if save:
        top_abs.drop('Absolute', axis=1).to_csv(f"{output_dir}/top_{n_importance}_absolute_{dataset_name}.csv", index=False)
    return all_df



feature_importance_collapsed_df = run_feature_importance(model_path="Results/v5_12-05-2025/Models/Collappsed/SVM/Test2/SVM.pkl", 
                       output_dir="Results/v5_12-05-2025/BestModels/Collapsed/FeatureImportance/NewThesis", dataset_name="Collassato", taxa_cols=collapsed_taxa_cols, ) # Collassato

feature_importance_original_df =  run_feature_importance(model_path="Results/v3_14-04-2025/Models/Metadata/Selected/SVM/SVM.pkl", 
                       output_dir="Results/v3_14-04-2025/FeatureImportance/NewThesis", dataset_name="Originale", taxa_cols=taxa_cols) # Originale



# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Istogramma dataset originale (4630 features)
ax1.hist(feature_importance_original_df['Absolute'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
ax1.set_xlabel('Valore assoluto del coefficiente', fontsize=14)
ax1.set_ylabel('Frequenza', fontsize=14)
ax1.set_title('Dataset originale', fontsize=16)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.grid(True, alpha=0.3)

ax1.text(0, 1.05, '(a)', transform=ax1.transAxes, fontsize=18, fontweight='bold')

# Statistiche primo plot
mean_abs_full = feature_importance_original_df['Absolute'].mean()
median_abs_full = feature_importance_original_df['Absolute'].median()
ax1.axvline(mean_abs_full, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_abs_full:.4f}')
ax1.axvline(median_abs_full, color='orange', linestyle='--', linewidth=2, label=f'Mediana: {median_abs_full:.4f}')
ax1.legend(fontsize=12)

# Istogramma dataset collassato (1128 features)
ax2.hist(feature_importance_collapsed_df['Absolute'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
ax2.set_xlabel('Valore assoluto del coefficiente', fontsize=14)
ax2.set_ylabel('Frequenza', fontsize=14)
ax2.set_title('Dataset collassato', fontsize=16)
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.grid(True, alpha=0.3)

ax2.text(0, 1.05, '(b)', transform=ax2.transAxes, fontsize=18, fontweight='bold')

# Statistiche secondo plot
mean_abs_collapsed = feature_importance_collapsed_df['Absolute'].mean()
median_abs_collapsed = feature_importance_collapsed_df['Absolute'].median()
ax2.axvline(mean_abs_collapsed, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_abs_collapsed:.4f}')
ax2.axvline(median_abs_collapsed, color='orange', linestyle='--', linewidth=2, label=f'Mediana: {median_abs_collapsed:.4f}')
ax2.legend(fontsize=12)

plt.suptitle('Distribuzione dei coefficienti assoluti SVM', fontsize=18, y=1.02)
plt.tight_layout()
plt.savefig(f"Results/v5_12-05-2025/distribution_comparison.png", bbox_inches='tight', dpi=300)
plt.show()

def print_statistics(df, dataset_name):
    print(f"=== {dataset_name} ===")
    print(f"Numero di feature: {len(df)}")

    print("\nCoefficienti (con segno):")
    print(f"Media: {df['Coefficient'].mean():.6f}")
    print(f"Mediana: {df['Coefficient'].median():.6f}")
    print(f"Deviazione standard: {df['Coefficient'].std():.6f}")
    print(f"Min: {df['Coefficient'].min():.6f}")
    print(f"Max: {df['Coefficient'].max():.6f}")

    print("\nCoefficienti assoluti:")
    print(f"Media: {df['Absolute'].mean():.6f}")
    print(f"Mediana: {df['Absolute'].median():.6f}")
    print(f"Deviazione standard: {df['Absolute'].std():.6f}")
    print(f"Max: {df['Absolute'].max():.6f}")

    threshold_90 = np.percentile(df['Absolute'], 90)
    threshold_95 = np.percentile(df['Absolute'], 95)
    threshold_99 = np.percentile(df['Absolute'], 99)

    print(f"\n=== ANALISI DISTRIBUZIONE ===")
    print(f"90% delle feature hanno coefficiente assoluto ≤ {threshold_90:.6f}")
    print(f"95% delle feature hanno coefficiente assoluto ≤ {threshold_95:.6f}")
    print(f"99% delle feature hanno coefficiente assoluto ≤ {threshold_99:.6f}")

    print("-" * 50)

print_statistics(feature_importance_original_df, "DATASET ORIGINALE (4630 features)")
print()
print_statistics(feature_importance_collapsed_df, "DATASET COLLASSATO (1128 features)")


# # Smote
# 
# Codice usato per runnare i modelli con SMOTE

# In[ ]:


models = {'LR': {'transformation': 'CLR_SMOTE',
                  'param_grid': {
                    'smote__sampling_strategy': "['auto', {0: 280, 1: 280}]",
                    "smote__random_state": "[42]",
                    'classifier__penalty': "['l2']",
                    "classifier__C": "np.logspace(-3,3,150)",
                    'classifier__random_state': "[42]"
                    },
                  'pipeline': Pipeline([
                                        ('clr', CLRTransformer()),
                                        ('scaler', StandardScaler()),
                                        ('smote', SMOTE()),
                                        ('classifier', LogisticRegression())
                                    ])},

            'RLS':{'transformation': 'CLR_SMOTE',
                  'param_grid':  {
                                'smote__sampling_strategy': "[{0: 245, 1: 245},{0: 500, 1: 500},{0: 1000, 1: 1000}, {0: 2000, 1: 2000}]",
                                'classifier__alpha': "np.logspace(-5,5,300)",
                                "smote__random_state": "[42]",
                                "classifier__random_state": "[42]"
                                },
                  'pipeline': Pipeline([
                                        ('clr', CLRTransformer()),
                                        ('scaler', StandardScaler()),
                                        ('smote', SMOTE()),
                                        ('classifier', RidgeClassifier())
                                    ])},

            'SVM':{'transformation': 'CLR_SMOTE',
                  'param_grid': {
                            'smote__sampling_strategy': "[{0: 245, 1: 245},{0: 500, 1: 500},{0: 1000, 1: 1000}]",
                            "classifier__kernel": "['linear', 'rbf']", 
                            "classifier__C": "np.logspace(-4,4,15)",
                            "classifier__gamma": "['scale', 'auto', 0.001, 0.01]",
                            "smote__random_state": "[42]",
                            "classifier__random_state": "[42]",
                  },
                  'pipeline': Pipeline([
                                        ('clr', CLRTransformer()),
                                        ('scaler', StandardScaler()),
                                        ('smote', SMOTE()),
                                        ('classifier', SVC())
                                    ])},

            'MLP':{'transformation': 'None_SMOTE',
                                        'param_grid': {
                            'smote__sampling_strategy': "[{0: 245, 1: 245},{0: 500, 1: 500},{0: 1000, 1: 1000}]",
                            "classifier__hidden_layer_sizes": "[[128],[256],[64, 32]]",
                            "classifier__activation": "['relu']",
                            "classifier__alpha": "[0.0001, 0.001, 0.1, 1, 10]",
                            "classifier__solver": "['adam']",
                            "classifier__learning_rate_init": "[0.001]",
                            "classifier__batch_size": "[32, 64]",
                            "classifier__max_iter": "[200]",
                            "smote__random_state": "[42]",
                            "classifier__random_state": "[42]",
                        },
                  'pipeline': Pipeline([
                                        ('scaler', StandardScaler()),
                                        ('smote', SMOTE()),
                                        ('classifier', MLPClassifier())
                                    ])},               
        }

models = {  'MLP':{'transformation': 'CLR_SMOTE',
                                        'param_grid': {
        "smote__sampling_strategy": "['auto', {0: 300, 1: 300}]",
        "smote__random_state": "[42]",
        "classifier__hidden_layer_sizes": "[(64,), (256,), (64,32), (128,64)]",
        "classifier__activation": "['relu']",
        "classifier__alpha": "[0.0001, 0.001, 0.01, 0.1, 1.0]",
        "classifier__solver": "['adam', 'sgd']",
        "classifier__learning_rate_init": "[0.001, 0.01]",
        "classifier__batch_size": "[16, 32, 64]",
        "classifier__max_iter": "[300]",
        "classifier__early_stopping": "[True]",
        "classifier__random_state": "[42]"
    },
                  'pipeline': Pipeline([
                                        ('transformation', CLRTransformer()),
                                        ('scaler', StandardScaler()),
                                        ('smote', SMOTE()),
                                        ('classifier', MLPClassifier())
                                    ])},              
        }

for name, value in models.items():
    print(name)
    directory = f'./Results/v3_14-04-2025/Models/Smote/{name}_New'
    transformation = value['transformation']
    evaluate_classifier(X_train, y_train, pipeline=value['pipeline'], param_grid=value['param_grid'], 
    directory=f'{directory}', 
    model_name=f'{name}_{transformation}',
    save=True
    )
    print(f"{name} Fatto!")


# # Predittore con solo Americani, Francesi, ...
# 
# Codice per allenare i modelli solo su:
# - **Francesi**
# - **Americani**
# - **Other** (tutte le altre country)

# In[ ]:


# Configurazioni dei paesi da processare
country_configs = [
    {
        'name': 'France',
        'condition': X_train_full['country'] == 'France'
    },
    {
        'name': 'America',
        'condition': X_train_full['country'] == 'North America'
    },
    {
        'name': 'Other',
        'condition': X_train_full['country'].isin(['United Kingdom', 'Netherlands', 'Spain'])
    }
]

# Configurazioni dei modelli da valutare
model_configs = [
    {
        'name': 'LR',
        'classifier': LogisticRegression,
        'param_grid': {
        "smote__sampling_strategy": "['auto', {0: 280, 1: 280}]",
        "smote__random_state": "[42]",
        "classifier__penalty": "['l2']",
        "classifier__C": "np.logspace(-3,3,150)",
        "classifier__random_state": "[42]"
    }
    },
    {
        'name': 'RLS',
        'classifier': RidgeClassifier,
        'param_grid': {
            "classifier__alpha": "np.logspace(-3, 3, 90)",
            "smote__random_state": "[42]",
            "classifier__random_state": "[42]"
        }
    },
    {
        'name': 'SVM',
        'classifier': SVC,
        'param_grid': {
            "smote__sampling_strategy": "[{0: 139, 1: 77}, {0: 139, 1: 139}, {0: 245, 1: 245}, {0: 500, 1: 500}]",
            "classifier__kernel": "['linear', 'rbf']",
            "classifier__C": "np.logspace(-4, 4, 10)",
            "classifier__gamma": "['scale', 'auto', 0.001, 0.01]",
            "smote__random_state": "[42]",
            "classifier__random_state": "[42]"
        }
    },
    {
        'name': 'MLP',
        'classifier': MLPClassifier,
        'param_grid': {
            "smote__sampling_strategy": "[{0: 56, 1: 83}, {0: 83, 1: 83}, {0: 300, 1: 300}]",
            "classifier__hidden_layer_sizes": "[[128], [256], [64, 32]]",
            "classifier__activation": "['relu']",
            "classifier__alpha": "[0.001, 0.1, 1]",
            "classifier__solver": "['adam']",
            "classifier__learning_rate_init": "[0.001]",
            "classifier__batch_size": "[32]",
            "classifier__max_iter": "[200]",
            "smote__random_state": "[42]",
            "classifier__random_state": "[42]"
        }
    }
]

for country in country_configs:
    idxs = X_train_full[country['condition']].index
    X_train_mod = X_train.loc[idxs].copy()
    y_train_mod = y_train[idxs].copy()
    print(f"\nProcessing country: {country['name']} ({len(idxs)} samples)")

    for model in model_configs:
        pipeline = Pipeline([
            ('transformation', CLRTransformer()),
            ('scaler', StandardScaler()),
            ('smote', SMOTE()),
            ('classifier', model['classifier']())
        ])

        directory = f'./Results/v3_14-04-2025/Models/Country/{country["name"]}/{model["name"]}'

        print(f"Evaluating {model['name']} for {country['name']}...")
        evaluate_classifier(
            X_train_mod, 
            y_train_mod,
            pipeline=pipeline,
            param_grid=model['param_grid'],
            directory=directory,
            model_name=model['name']
        )


# # Modelli con metadati
# Codice che in automatico allena tutti i modelli per tutti e 5 i test:
# - **Test 1**: Tutti i metadati
# - **Test 2**: Metadati selezionati
# - **Test 3**: Metadati selezionati e categorie minoritarie aggregate
# - **Test 4**: Metadati selezionati + Platform
# - **Test 5**: Metadati selezionati + LibraryLayout

# In[ ]:


tests = [0,1,2,3,4,5]
model_config = [
    {   # Logistic Regression
        "name": "LR",
        "classifier": LogisticRegression,
        "param_grid": {
        "smote__sampling_strategy": "['auto', {0: 280, 1: 280}]",
        "smote__random_state": "[42]",
        "classifier__penalty": "['l2']",
        "classifier__C": "np.logspace(-3,3,150)",
        "classifier__random_state": "[42]"
    },
    },
    {   # Ridge Classifier
        "name": "RLS",
        "classifier": RidgeClassifier,
        "param_grid": {
            "smote__sampling_strategy": "[{0: 245, 1: 245},{0: 500, 1: 500},{0: 1000, 1: 1000}]",
            "classifier__alpha": "np.logspace(-5,5,100)",
            "smote__random_state": "[42]",
            "classifier__random_state": "[42]"
        },
    },
    {   # SVM
        "name": "SVM",
        "classifier": SVC,
        "param_grid": {
            "classifier__kernel": "['linear', 'rbf']",
            "classifier__C": "np.logspace(-4,4,30)",
            "classifier__gamma": "['scale', 'auto', 0.001, 0.01]",
            "smote__random_state": "[42]",
            "classifier__random_state": "[42]"
        },
    },
    {   # MLP
        "name": "MLP",
        "classifier": MLPClassifier,
        "param_grid": {
            "classifier__hidden_layer_sizes": "[[128],[256],[64, 32]]",
            "classifier__activation": "['relu']",
            "classifier__alpha": "[ 0.001, 0.1, 1, 10]",
            "classifier__solver": "['adam']",
            "classifier__learning_rate_init": "[0.001]",
            "classifier__batch_size": "[32]",
            "classifier__max_iter": "[200]",
            "classifier__random_state": "[42]",
            "smote__random_state": "[42]"
        },
    }
]


for test in tests:
    X_train_mod, preprocessor = initialiaze_metadata(X_collapsed_train_full, test=test, taxa_cols=collapsed_taxa_cols)
    for config in model_config:
        if test == 0:
            pipline =   pipeline = Pipeline([
            ('transformation', CLRTransformer()),
            ('scaler', StandardScaler()),
            ('smote', SMOTE()),
            ('classifier', config["classifier"]())
        ])
        else:
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('smote', SMOTE()),
                ('classifier', config["classifier"]())
            ])

        directory = f'./Results/v3_14-04-2025/Models/Metadata/Test{test}/{config["name"]}'

        evaluate_classifier(
            X_train_mod, 
            y_collapsed_train, 
            pipeline=pipeline,
            param_grid=config["param_grid"],
            save=True,
            directory=directory,
            model_name=config["name"]
        )


# # Recursive Feature Elimination
# 
# Allenamento dei modelli solo sulle feature scelte tramite **Recursive Feature Elimination** (RFE), inizialmente è stata effettuata su tutto il train portando a del **data leakage** successivamente è sil train è stato splittato in train_split e val_split per poi applicare RFE solo sul train_split in modo da valutare gli score sul validation val_split (in modo da non usare il test).

# In[ ]:


model_config = [
    {   # Logistic Regression
        "name": "LR",
        "classifier": LogisticRegression,
        "param_grid": {
        "smote__sampling_strategy": "['auto', {0: 280, 1: 280}]",
        "smote__random_state": "[42]",
        "classifier__penalty": "['l2']",
        "classifier__C": "np.logspace(-3,3,150)",
        "classifier__random_state": "[42]"
    },
    },
    {   # Ridge Classifier
        "name": "RLS",
        "classifier": RidgeClassifier,
        "param_grid": {
            "feature_selector__threshold": "np.arange(2,100,30).tolist()",
            "smote__sampling_strategy": "['auto',{0: 245, 1: 245},{0: 500, 1: 500}]",
            "classifier__alpha": "np.logspace(-5,5,100)",
            "smote__random_state": "[42]",
            "classifier__random_state": "[42]"
        },
    },
    {   # SVM
        "name": "SVM",
        "classifier": SVC,
        "param_grid": {
            "feature_selector__threshold": "np.arange(2,100,30).tolist()",
            "classifier__kernel": "['linear', 'rbf']",
            "classifier__C": "np.logspace(-4,4,30)",
            "classifier__gamma": "['scale', 'auto', 0.001, 0.01]",
            "smote__random_state": "[42]",
            "classifier__random_state": "[42]"
        },
    },
    {   # MLP
        "name": "MLP",
        "classifier": MLPClassifier,
        "param_grid": {
            "feature_selector__threshold": "np.arange(2,100,30).tolist()",
            "classifier__hidden_layer_sizes": "[[128],[256],[64, 32]]",
            "classifier__activation": "['relu']",
            "classifier__alpha": "[ 0.001, 0.1, 1, 10]",
            "classifier__solver": "['adam']",
            "classifier__learning_rate_init": "[0.001]",
            "classifier__batch_size": "[32]",
            "classifier__max_iter": "[200]",
            "classifier__random_state": "[42]",
            "smote__random_state": "[42]"
        },
    }
]


# False per usare i ranking ottenuti su tutto il train (data leakage se si osserva score cross validation)
# True per usare i ranking ottenuti sul train_split e fare una predict sul validation
validation = True 

if not validation:
    X_train_evaluate = X_train
    y_train_evaluate = y_train
    ranking_path = 'Results/v3_14-04-2025/RFE_Ranking/ranking_FullTrain.csv'
    name = "RFE"
else:
    X_train_evaluate, X_val_split, y_train_evaluate, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    ranking_path = 'Results/v3_14-04-2025/RFE_Ranking/ranking_SplittedTrain.csv'
    name = "RFE_VAL"


for config in model_config:
    pipeline = Pipeline([
        ('feature_selector', RankingFeatureSelector(ranking_path=ranking_path)),
        ('transformation', CLRTransformer()),
        ('scaler', StandardScaler()),
        ('smote', SMOTE()),
        ('classifier', config["classifier"]())
    ])

    directory = f'./Results/v3_14-04-2025/Models/{name}/{config["name"]}'
    model = evaluate_classifier(
        X_train_evaluate, 
        y_train_evaluate, 
        X_validation=X_val_split if validation else None,
        y_validation=y_val_split if validation else None,
        pipeline=pipeline,
        param_grid=config["param_grid"],
        save=True,
        directory=directory,
        model_name=config["name"]
    )
    if validation:
        y_pred = model.predict(X_val_split)
        print(classification_report(y_val_split, y_pred))


# # Studio con solo 1 BioProject
# 
# Codice per allenare i modelli solo sui dati dei due BioProject con più samples, ovvero: 
# - BioProject **PRJNA751792**:    183 samples
# - BioProject **PRJEB43119**:     135 samples
# 

# In[69]:


### B1
#148 nel train
# 35 nel test
# tot 183
# distribuzione train/test: 81/19

train_B1_df = train_df[train_df['BioProject'] == 'PRJNA751792'].reset_index()

le_B1 = LabelEncoder()
X_train_B1 = train_B1_df[taxa_cols].replace(',', '.', regex=True).apply(pd.to_numeric)
y_train_B1 = le_B1.fit_transform(train_B1_df['response'])

############ B2
# 100 nel train
# 35 nel test
# tot 135
# distribuzione train/test: 74/26

train_B2_df = train_df[train_df['BioProject'] == 'PRJEB43119'].reset_index()

le_B2 = LabelEncoder()
X_train_B2 = train_B2_df[taxa_cols].replace(',', '.', regex=True).apply(pd.to_numeric)
y_train_B2 = le_B2.fit_transform(train_B2_df['response'])


sets = {'B1': {"X_train": X_train_B1, "y_train": y_train_B1},
        'B2': {"X_train": X_train_B2, "y_train": y_train_B2}}


# In[ ]:


models = {
    'LR': {
        'pipeline': Pipeline([
            ('transformation', CLRTransformer()),
            ('scaler', StandardScaler()),
            ('smote', SMOTE()),
            ('classifier', LogisticRegression())
        ]),
        'param_grid': {
        "smote__sampling_strategy": "['auto', {0: 280, 1: 280}]",
        "smote__random_state": "[42]",
        "classifier__penalty": "['l2']",
        "classifier__C": "np.logspace(-3,3,150)",
        "classifier__random_state": "[42]"
    }
    },
    'RLS': {
        'pipeline': Pipeline([
            ('transformation', CLRTransformer()),
            ('scaler', StandardScaler()),
            ('smote', SMOTE()),
            ('classifier', RidgeClassifier())
        ]),
        'param_grid': {
            "smote__sampling_strategy": "['auto', {0: 100, 1: 100}, {0: 150, 1: 150}, {0: 250, 1: 250}, {0: 350, 1: 350}]",
            "classifier__alpha": "np.logspace(-5, 5, 200)",
            "smote__random_state": "[42]",
            "classifier__random_state": "[42]"
        }
    },
    'SVC': {
        'pipeline': Pipeline([
            ('transformation', CLRTransformer()),
            ('scaler', StandardScaler()),
            ('smote', SMOTE()),
            ('classifier', SVC())
        ]),
        'param_grid': {
            "smote__sampling_strategy": "['auto', {0: 100, 1: 100}, {0: 150, 1: 150}, {0: 250, 1: 250}, {0: 400, 1: 400}]",
            "classifier__kernel": "['linear', 'rbf']",
            "classifier__C": "np.logspace(-4, 4, 10)",
            "classifier__gamma": "['scale', 'auto', 0.001, 0.01]",
            "smote__random_state": "[42]",
            "classifier__random_state": "[42]",
            "classifier__probability": "[True]",
        }
    },
    'MLP': {
        'pipeline': Pipeline([
            ('transformation', CLRTransformer()),
            ('scaler', StandardScaler()),
            ('smote', SMOTE()),
            ('classifier', MLPClassifier())
        ]),
        'param_grid': {
            "smote__sampling_strategy": "['auto', {0: 100, 1: 100}, {0: 200, 1: 200}]",
            "classifier__hidden_layer_sizes": "[[16], [32], [32, 16]]",
            "classifier__activation": "['relu', 'tanh']",
            "classifier__alpha": "[0.0001, 0.001, 0.01]",
            "classifier__solver": "['adam']",
            "classifier__learning_rate_init": "[0.001]",
            "classifier__batch_size": "[8]",
            "classifier__max_iter": "[300]",
            "classifier__random_state": "[42]",
            "smote__random_state": "[42]"
        }
    }
}


base_dir = "Results/v4_28-04-2025/Models/BioProjects"

for bio_project, df in sets.items():
    print("="*50)
    print(bio_project)
    print("="*50)
    for model_name, model_info in models.items():
        print(model_name)
        directory = f"{base_dir}/{bio_project}/{model_name}"
        evaluate_classifier(df["X_train"], df["y_train"], model_info['pipeline'],model_info['param_grid'], save=True, directory=directory, model_name=model_name)


# # Medoid e Dataset Collassato
# 
# Codice per allenare i modelli sui medoid oppure sul dataset collassato a seconda della variabile "sets"

# In[ ]:


sets = {}
#sets['collapsed'] = {"X_train": X_collapsed_train_full, "y_train": y_collapsed_train}

# Funzione per generare sets in caso di allenamento sui medoid e submedoid
sets = create_medoids_set()

model_config = [
    {   # Logistic Regression
        "name": "LR",
        "classifier": LogisticRegression,
        "param_grid": {
        "smote__sampling_strategy": "['auto', {0: 280, 1: 280}]",
        "smote__random_state": "[42]",
        "classifier__penalty": "['l2']",
        "classifier__C": "np.logspace(-3,3,150)",
        "classifier__random_state": "[42]"
    },
    },
    {   # Ridge Classifier
        "name": "RLS",
        "classifier": RidgeClassifier,
        "param_grid": {
        "smote__sampling_strategy": "['auto', {0: 500, 1: 500}]",
        "classifier__alpha": "np.logspace(-4,4,40)",
        "smote__random_state": "[42]",
        "classifier__random_state": "[42]"
    },
    },
    {   # SVM
        "name": "SVM",
        "classifier": SVC,
        "param_grid": {
        "classifier__kernel": "['linear', 'rbf']",
        "classifier__C": "np.logspace(-4,-1,10)",
        "classifier__gamma": "['scale', 'auto', 0.001, 0.01]",
        "smote__random_state": "[42]",
        "classifier__random_state": "[42]"
    },
    },
    {   # MLP
        "name": "MLP",
        "classifier": MLPClassifier,
        "param_grid": {
        "classifier__hidden_layer_sizes": "[[16],[32],[64, 32]]",
        "classifier__activation": "['relu']",
        "classifier__alpha": "[ 0.001, 0.1,1]",
        "classifier__solver": "['adam']",
        "classifier__learning_rate_init": "[0.001]",
        "classifier__batch_size": "[32]",
        "classifier__max_iter": "[1000]",
        "smote__random_state": "[42]",
        "classifier__random_state": "[42]"
    },
    }
]



base_dir = "Results/v5_12-05-2025/Models/"
for name, df in sets.items():
    comp_cols = collapsed_taxa_cols if name == "collapsed" else taxa_cols
    print("="*150)
    print("="*150)
    print(f"\t\t\t\t Dataset {name}")
    print("="*150)
    print("="*150)
    for test in tests:
        print("*"*150)
        print(f"\t\t\t\t test {test}")
        print("*"*150)
        for model in model_config:
            pipeline = [('transformation', CLRTransformer()),
            ('scaler', StandardScaler()),
            ('smote', SMOTE()),
            ('classifier', model['classifier']())
            ]
            X_train_mod, preprocessor = initialiaze_metadata(df["X_train"], test, medoid=name, taxa_cols=comp_cols)
            if preprocessor:
                print("no")
                pipeline = [('preprocessor', preprocessor),
                ('smote', SMOTE()),
                ('classifier', model['classifier']())
                ]
            directory = f"{base_dir}/Medoids/{name}/{model_name}_L2/Test{test}/"
            evaluate_classifier(X_train_mod, df["y_train"], pipeline=Pipeline(pipeline), param_grid=model['param_grid'],save=True, directory=directory, model_name=model_name)



# ## Allenamento dei migliori modelli con PCA e KernelPCA

# In[ ]:


sets = {}
sets['collapsed'] = {"X_train": X_collapsed_train_full, "y_train": y_collapsed_train}
#sets['notCollapsed'] = {"X_train": X_train_full, "y_train": y_train}}

# Funzione per generare sets in caso di allenamento sui medoid e submedoid
#sets = create_medoids_set()


normal_pipeline = [('transformation', CLRTransformer()),
            ('scaler', StandardScaler()),
            ('smote', SMOTE()),
            ]

metadata_pipeline = [('smote', SMOTE()),]


LR_ParamGrid = {
        "smote__sampling_strategy": "['auto', {0: 280, 1: 280}]",
        "smote__random_state": "[42]",
        "classifier__penalty": "['l2']",
        "classifier__C": "np.logspace(-3,3,150)",
        "classifier__random_state": "[42]"
    }


RLS_ParamGrid = {
        "smote__sampling_strategy": "['auto', {0: 500, 1: 500}]",
        "classifier__alpha": "np.logspace(-4,4,40)",
        "smote__random_state": "[42]",
        "classifier__random_state": "[42]"
    }



SVM_ParamGrid = {
        "classifier__kernel": "['linear', 'rbf']",
        "classifier__C": "np.logspace(-4,-1,10)",
        "classifier__gamma": "['scale', 'auto', 0.001, 0.01]",
        "smote__random_state": "[42]",
        "classifier__random_state": "[42]"
    }

MLP_ParamGrid = {
        "classifier__hidden_layer_sizes": "[[16],[32],[64, 32]]",
        "classifier__activation": "['relu']",
        "classifier__alpha": "[ 0.001, 0.1,1]",
        "classifier__solver": "['adam']",
        "classifier__learning_rate_init": "[0.001]",
        "classifier__batch_size": "[32]",
        "classifier__max_iter": "[1000]",
        "smote__random_state": "[42]",
        "classifier__random_state": "[42]"
    }


KPCA_param_grid = { "reduction__n_components": "[50, 150, 250, 500, 1000]",
                    "reduction__kernel": "['rbf']"}

KPCA_metadata_param_grid = { "comp_reduction__reduction__n_components": "[50, 150, 250, 500, 1000]",
                    "comp_reduction__reduction__kernel": "['rbf']"}

PCA_param_grid = { "reduction__n_components": "[0.3, 0.5, 0.7, 0.9, 0.95]"}

PCA_metadata_param_grid = { "comp_reduction__reduction__n_components": "[0.3, 0.5, 0.7, 0.9, 0.95]"}



"""
Per i medoid i test dei migliori sono:
LR: {"M1": 0, "M2": 0, "M1_1": 0, "M1_2":0, "M2_1": 0, "M2_2": 1}
RLS: {"M1": 0, "M2": 2, "M1_1": 2, "M1_2":3, "M2_1": 0, "M2_2": 2}
SVM: {"M1": 1, "M2": 0, "M1_1": 0, "M1_2":1, "M2_1": 2, "M2_2": 2}
MLP: {"M1": 5, "M2": 4, "M1_1": 3, "M1_2":4, "M2_1": 5, "M2_2": 2}
"""

models = {
    "LR": {'classifier':[('classifier', LogisticRegression())],
            'reduction': {'PCA': {**LR_ParamGrid, **PCA_param_grid}, 'KernelPCA': {**LR_ParamGrid, **KPCA_param_grid}}, "test": 0},
    "RLS": {'classifier':[('classifier', RidgeClassifier())],
            'reduction': {'PCA': {**RLS_ParamGrid, **PCA_metadata_param_grid}, 'KernelPCA': {**RLS_ParamGrid, **KPCA_metadata_param_grid}}, "test": 2},
    "SVM": { 'classifier':[('classifier', SVC())],
            'reduction': {'PCA': {**SVM_ParamGrid, **PCA_metadata_param_grid}, 'KernelPCA': {**SVM_ParamGrid, **KPCA_metadata_param_grid}}, "test": 2},
    "MLP": {'classifier':[('classifier', MLPClassifier())],
            'reduction': {'PCA': {**MLP_ParamGrid, **PCA_param_grid}, 'KernelPCA': {**MLP_ParamGrid, **KPCA_param_grid}}, "test": 0},

}

evaluate_medoid = True
base_dir = "Results/v5_12-04-2025/Models/"
for name, df in sets.items():
    medoid = name if evaluate_medoid else None
    comp_cols = collapsed_taxa_cols if name == "collapsed" else taxa_cols
    print("="*150)
    print("="*150)
    print(f"\t\t\t\t Dataset {name}")
    print("="*150)
    print("="*150)
    for model_name, model_val in models.items():
        test = model_val['test']
        for reduction, param_grid in model_val['reduction'].items():
            print(f"\t\t\t\t Modello: {model_name} \t\t Riduzione: {reduction}")
            X_train_mod, preprocessor = initialiaze_metadata(df["X_train"], test,name, medoid = medoid, taxa_cols=comp_cols)
            if test == 0:
                if reduction == "PCA":
                    pipeline = Pipeline([*normal_pipeline, ('reduction', PCA()),*model_val['classifier']])
                else:
                    pipeline = Pipeline([*normal_pipeline, ('reduction', KernelPCA()),*model_val['classifier']])

            else:
                if reduction == "PCA":
                    pipeline = Pipeline([('preprocessor', preprocessor), *metadata_pipeline,
                                        ('comp_reduction', CompReduction(n_comp_features=1128, reduction=PCA())),
                                        *model_val['classifier']])
                else:
                    pipeline = Pipeline([('preprocessor', preprocessor), *metadata_pipeline,
                                            ('comp_reduction', CompReduction(n_comp_features=1128, reduction=KernelPCA())),
                                            *model_val['classifier']])
            directory = f"{base_dir}/{name}/{model_name}/Test{test}/{reduction}"
            print(model_val["reduction"])
            evaluate_classifier(X_train_mod, df["y_train"], pipeline=pipeline, param_grid=model_val["reduction"][reduction],save=False, directory=directory, model_name=model_name)




# # Valutazioni finali
# 
# Codice per caricare i migliori modelli (definendo su che test sono stati allenati) per poi fare una predict su tutti i set presenti nella variabile "**sets**"

# In[ ]:


### B1
#148 nel test
# 35 nel test
# tot 183
# distribuzione test/test: 81/19

test_B1_df = test_df[test_df['BioProject'] == 'PRJNA751792'].reset_index()

X_test_B1 = test_B1_df[taxa_cols].replace(',', '.', regex=True).apply(pd.to_numeric)
y_test_B1 = le.transform(test_B1_df['response'])




############ B2
# 100 nel test
# 35 nel test
# tot 135
# distribuzione test/test: 74/26

test_B2_df = test_df[test_df['BioProject'] == 'PRJEB43119'].reset_index()


X_test_B2 = test_B2_df[taxa_cols].replace(',', '.', regex=True).apply(pd.to_numeric)
y_test_B2 = le.transform(test_B2_df['response'])


sets = {'B1': {"Test Isolato": {"X_test": X_test_B1, "y_test": y_test_B1},
"Test": {"X_test": X_test, "y_test": y_test},
"Modenesi": {"X_test": X_modenesi, "y_test": y_modenesi}},

'B2': {"Test Isolato": {"X_test": X_test_B2, "y_test": y_test_B2},
"Test": {"X_test": X_test, "y_test": y_test},
"Modenesi": {"X_test": X_modenesi, "y_test": y_modenesi}},

'Collapsed': {"Test": {"X_test": X_collapsed_test_full, "y_test": y_collapsed_test},
        "Modenesi": {"X_test": X_collapsed_modenesi_full, "y_test": y_collapsed_modenesi},
        },
'Normal': {"Test": {"X_test": X_test_full, "y_test": y_test},
        "Modenesi Originali": {"X_test": X_modenesi_full, "y_test": y_modenesi
        }}
        }

medoid_set = create_medoids_set(predict=True)

sets = {**sets, **medoid_set}


# In[ ]:


# Configurazione: basta cambiare questi due
dataset_name = "Medoids"      # Deve valere: Medoids, Collapsed, BioProject o Normal
type = "M1_1"              # Usato per dataset_name = Medoids o BioProject 
                                # Per dataset_name = Medoids deve valere M1, M2, M1_1, M1_2, M2_1 o M2_2
                                # Per dataset_name = BioProject deve valere B1 o B2
                        # Per gli altri deve essere uguale a dataset_name,
                        # Esempio: se dataset_name = Collapsed, allora type deve valere Collapsed)
format = 1 # Formato di stampa: 1 = markdown, 2 = latex

if dataset_name == "Collapsed":
    type = dataset_name
    base_dir = f"Results/v5_12-05-2025/BestModels/{dataset_name}"
elif dataset_name == "Medoids":
    base_dir = f"Results/v5_12-05-2025/BestModels/{dataset_name}/{type}"
elif dataset_name == "BioProject":
    base_dir = f"Results/v4_28-04-2025/Models/BioProjects/{type}"
elif dataset_name == "Normal":
    base_dir="Results/v4_28-04-2025/BestModelsGraphs"

col = collapsed_taxa_cols if dataset_name == "Collapsed" else taxa_cols

# Definizione base_paths dinamica
base_paths = {
    "Medoids": f"Results/v5_12-05-2025/BestModels/Medoids/{type}",
    "Collapsed": f"Results/v5_12-05-2025/BestModels/Collapsed/pkls",
    "BioProject": f"Results/v4_28-04-2025/Models/BioProjects/{type}",
    "Normal": f"Results/v4_28-04-2025/BestModelsGraphs"
}


# Dizionario con test associati a ciascun modello e tipo
models_tests = {
    "M1":   {"LR": 0, "RLS": 3, "SVM": 2, "MLP": 5, "RandomForest": 0, "ExtraTrees": 0, "XGBoost": 0},
    "M2":   {"LR": 0, "RLS": 2, "SVM": 2, "MLP": 4, "RandomForest": 0, "ExtraTrees": 2, "XGBoost": 0},
    "M1_1": {"LR": 0, "RLS": 2, "SVM": 0, "MLP": 3, "RandomForest": 0, "ExtraTrees": 0, "XGBoost": 4},
    "M1_2": {"LR": 0, "RLS": 3, "SVM": 2, "MLP": 4, "RandomForest": 1, "ExtraTrees": 4, "XGBoost": 3},
    "M2_1": {"LR": 0, "RLS": 0, "SVM": 2, "MLP": 5, "RandomForest": 3, "ExtraTrees": 1, "XGBoost": 0},
    "M2_2": {"LR": 1, "RLS": 2, "SVM": 2, "MLP": 2, "RandomForest": 1, "ExtraTrees": 2, "XGBoost": 2},
    "BioProject": {"LR": 0, "RLS": 0, "SVM": 0, "MLP": 0, "RandomForest": 0, "ExtraTrees": 0, "XGBoost": 0},
    "Collapsed": {"LR": 2, "RLS": 3, "SVM": 2, "MLP": 0, "RandomForest": 0, "ExtraTrees": 3, "XGBoost": 3},
    "Normal": {"LR": 0, "RLS": 3, "SVM": 2, "MLP": 4, "RandomForest": 0, "ExtraTrees": 1, "XGBoost": 4},
}
models_tests = {
    "M1":   {"LR": 0, "RLS": 3, "SVM": 2, "MLP": 5, "RandomForest": 0, "ExtraTrees": 0, "XGBoost": 0},
    "M2":   {"LR": 0, "RLS": 2, "SVM": 2, "MLP": 4, "RandomForest": 0, "ExtraTrees": 2, "XGBoost": 0},
    "M1_1": {"LR": 0, "RLS": 2, "SVM": 0, "MLP": 3},
    "M1_2": {"LR": 0, "RLS": 3, "SVM": 2, "MLP": 4, "RandomForest": 1, "ExtraTrees": 4, "XGBoost": 3},
    "M2_1": {"LR": 0, "RLS": 0, "SVM": 2, "MLP": 5, "RandomForest": 3, "ExtraTrees": 1, "XGBoost": 0},
    "M2_2": {"LR": 1, "RLS": 2, "SVM": 2, "MLP": 2, "RandomForest": 1, "ExtraTrees": 2, "XGBoost": 2},
    "BioProject": {"LR": 0, "RLS": 0, "SVM": 0, "MLP": 0, "RandomForest": 0, "ExtraTrees": 0, "XGBoost": 0},
    "Collapsed": {"LR": 2, "RLS": 3, "SVM": 2, "MLP": 0, "RandomForest": 0, "ExtraTrees": 3, "XGBoost": 3},
    "Normal": {"LR": 0, "RLS": 3, "SVM": 2, "MLP": 4, "RandomForest": 0, "ExtraTrees": 1, "XGBoost": 4},
}

# Determina dinamicamente il tipo usato per accedere a models_tests
test_key = type if type in models_tests else dataset_name
models = list(models_tests[test_key].keys())

# Costruisce il path dinamicamente
models_path = {
    model: {
        "path": f"{base_paths[dataset_name]}/{model}.pkl",
        "test": models_tests[test_key][model]
    }
    for model in models
}

print(models_path["LR"]["path"])
#evaluate_set = sets[test_key]
evaluate_set = sets[type]

dataset_type = ["M1", "M2", "M1_1", "M1_2", "M2_1", "M2_2", "B1", "B2"]
print("="*130)
print("="*130)
print(f"\t\tPredizioni per dataset: {dataset_name} ")
if type in dataset_type:
    print(f"\t\tCon tipologia: {type}")
print("="*130)
print("="*130)

print("*"*100)
print(f"\t\t  Modelli singoli")
print("*"*100)

for set_name, value in evaluate_set.items():

    #print("="*70)
    #print(f"\t\t  {set_name}")
    #print("="*70)
    print()
    if  set_name not in ["Modenesi", "Modenesi Isolati"]:
        print(f"## Risultati sul {set_name} ") if format == 1 else print(f"\\subsubsection{{Risultati sul {set_name}}}")
    else:
        print(f"## Risultati sui {set_name} ") if format == 1 else print(f"\\subsubsection{{Risultati sui {set_name}}}")

    print()

    compute_roc = False if set_name in ["Modenesi", "Modenesi Isolati"] else True


    model_evaluate(
        models_path,
        value["X_test"],
        value["y_test"],
        set_name,
        base_dir,
        taxa_cols=col,
        compute_roc=compute_roc,
        format=format
    )
    if format == 2:
        print(r"""\end{tabular}
        \end{table}""")


print("*"*100)
print(f"\t\t  Modelli fusi")
print("*"*100)

print(f"## Risultati con fusione dei modelli ") if format == 1 else print(f"\\subsubsection{{Fusione modelli}}")

for idx, (set_name, value) in enumerate(evaluate_set.items()):

    compute_roc = False if set_name in ["Modenesi", "Modenesi Isolati"] else True
    first = idx == 0

    ensemble_evaluate(
        models_path,
        value["X_test"],
        value["y_test"],
        set_name,
        base_dir,
        taxa_cols=col,
        compute_roc=compute_roc,
        first=first,
        format=format
    )


