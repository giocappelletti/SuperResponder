import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler, LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from skbio.stats.composition import multi_replace, clr
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
import shap
import os

###################################
#### PREDIZIONE SENZA METADATI ####
###################################


COLLASSATO = False

OLD_SPLIT = True
SEED = 42

MULTI_REPLACE = False # Serve per trasformazione CLR

if OLD_SPLIT: ### Carica train e test set già splittati dai ragazzi - il test dovrebbe essere bilanciato rispetto alle due classi ###
    train_path = os.path.join('datasets', 'splitted', 'Full', 'train_set.csv')
    df_train = pd.read_csv(train_path, decimal=',')
    taxa_cols = [col for col in df_train.columns if col.startswith('k__')]

    X_train = df_train[taxa_cols]
    y_train = df_train['response']

    test_path = os.path.join('datasets', 'splitted', 'Full', 'test_set.csv')
    df_test = pd.read_csv(test_path, decimal=',')
    X_test = df_test[taxa_cols]
    y_test = df_test['response']

    ### Mappa NR (Non Responder) in 0 e R (Responder) in 1 ###
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

else: ### Carica dataset e splitta casualmente in train e test set ###
    datataset_path = os.path.join('datasets', 'raw_dataset.csv')
    df = pd.read_csv(datataset_path, decimal=',')
    taxa_cols = [col for col in df.columns if col.startswith('k__')]
    X = df[taxa_cols]
    y = df['response']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=SEED, stratify=y_encoded)


### Collassa le feature tassonomiche al livello di Genere ###
if COLLASSATO:
    mapping = {}
    for col in taxa_cols:
        if ';' in col:
            # Trova l'ultimo punto e virgola e prendi tutto ciò che c'è prima
            new_name = col[:col.rfind(';')]
        else:
            # Se non c'è punto e virgola, mantieni il nome originale (non dovrebbe succedere)
            new_name = col
        mapping[col] = new_name

    X_train = X_train.T.groupby(mapping).sum().T
    X_test = X_test.T.groupby(mapping).sum().T


def clr_transform(data, epsilon=1e-6):

    if MULTI_REPLACE:
        data_eps = multi_replace(data)
        data_eps = pd.DataFrame(data_eps, columns=data.columns, index=data.index)
    else:
        data_eps = data + epsilon 

    return clr(data_eps)


X_train_clr = clr_transform(X_train)
X_test_clr = clr_transform(X_test)


if COLLASSATO: # robustscaler per collassato, standard per normale. Gemini dice che robustscaler funziona meglio su dati grezzi, piuttosto che composizionali.
    scaler = RobustScaler() 
else:    
    scaler = StandardScaler() 

X_train_scaled = scaler.fit_transform(X_train_clr)
X_test_scaled = scaler.transform(X_test_clr)


### Grid Search per SVM con 5-Fold Cross-Validation ###
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel': ['linear', 'rbf']
}

grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)


best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test_scaled)
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
acc = accuracy_score(y_test, y_pred)


print("\n--- REPORT ---")
print("Best Parameters:", best_params)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))





exit(-1)

### FEATURE IMPORTANCE - EXPLAINBLE AI ###
print("\n--- FEATURE IMPORTANCE (Top 10) ---")
if best_params['kernel'] == 'linear':    
    if best_model.coef_.shape[0] == 1: # Binario
        coefs = best_model.coef_[0]
    else: # Multiclasse 
        coefs = np.mean(np.abs(best_model.coef_), axis=0)

    importances = np.abs(coefs)

elif best_params['kernel'] == 'rbf':
    result = permutation_importance(
        best_model, 
        X_test_scaled, 
        y_test, 
        n_repeats=10, 
        random_state=SEED, 
        n_jobs=-1
    )
    
    importances = result.importances_mean

feature_names = X_train.columns

feature_importance = pd.Series(
    importances,
    index=feature_names
).sort_values(ascending=False)

print(feature_importance.head(10))




print("\n--- MUTUAL INFORMATION (Top 10) ---")
mi_scores = mutual_info_classif(X_train_scaled, y_train, random_state=SEED)
mi_importance = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)
print(mi_importance.head(10))



if best_params['kernel'] == 'rbf':

    # KernelExplainer è lento, quindi riassumiamo il train set usanso k-means per creare 10-50 profili rappresentativi del training set. Velocizza il calcolo senza perdere troppa precisione.
    X_train_summary = shap.kmeans(X_train_scaled, 10)
    explainer = shap.KernelExplainer(best_model.predict_proba, X_train_summary)
    shap_values = explainer.shap_values(X_test_scaled)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values[:,:,1], 
        X_test_scaled, 
        feature_names=feature_names, 
        show=False 
    )

    plt.tight_layout()
    filename = "shap_summary_plot.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')




######################
### NON COLLASSATO ###
######################

# --- REPORT ---
# Best Parameters: {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'}
# Accuracy: 0.631578947368421
#               precision    recall  f1-score   support

#           NR       0.67      0.63      0.65        62
#            R       0.59      0.63      0.61        52

#     accuracy                           0.63       114
#    macro avg       0.63      0.63      0.63       114
# weighted avg       0.63      0.63      0.63       114


# --- FEATURE IMPORTANCE (Top 10) ---
# k__Bacteria; p__Firmicutes_A; c__Clostridia; o__Tissierellales; f__Tepidimicrobiaceae; g__Mt11; s__sp001282665                      0.027622
# k__Bacteria; p__Actinobacteriota; c__Actinomycetia; o__Actinomycetales; f__Bifidobacteriaceae; g__Bifidobacterium; s__vaginale_G    0.026527
# k__Bacteria; p__Firmicutes_A; c__Clostridia_A; o__Christensenellales; f__CAG-917; g__CAG-917; s__MGYG000002000                      0.026513
# k__Bacteria; p__Bacteroidota; c__Bacteroidia; o__Bacteroidales; f__Rikenellaceae; g__Alistipes_A; s__ihumii                         0.026369
# k__Bacteria; p__Spirochaetota; c__Spirochaetia; o__Sphaerochaetales; f__Sphaerochaetaceae; g__UBA5920; s__MGYG000000677             0.025293
# k__Bacteria; p__Firmicutes; c__Bacilli; o__RF39; f__UBA660; g__HGM10766; s__sp900757295                                             0.024352
# k__Bacteria; p__Firmicutes; c__Bacilli; o__Erysipelotrichales; f__Erysipelatoclostridiaceae; g__Coprobacillus; s__cateniformis      0.024063
# k__Bacteria; p__Firmicutes_A; c__Clostridia; o__Oscillospirales; f__QAKW01; g__QAKW01; s__MGYG000004124                             0.023509
# k__Bacteria; p__Firmicutes_A; c__Clostridia; o__Monoglobales; f__Monoglobaceae; g__UMGS1494; s__sp900552305                         0.022746
# k__Bacteria; p__Actinobacteriota; c__Coriobacteriia; o__Coriobacteriales; f__QAMH01; g__CACZQA01; s__sp900758075                    0.022478
# dtype: float64

# --- MUTUAL INFORMATION (Top 10) ---
# k__Bacteria; p__Firmicutes_A; c__Clostridia; o__Lachnospirales; f__Lachnospiraceae; g__Blautia_A; s__faecis                           0.083224
# k__Bacteria; p__Bacteroidota; c__Bacteroidia; o__Bacteroidales; f__Bacteroidaceae; g__Paraprevotella; s__sp900760455                  0.073526
# k__Bacteria; p__Proteobacteria; c__Gammaproteobacteria; o__Burkholderiales; f__Burkholderiaceae; g__Parasutterella; s__sp000980495    0.072899
# k__Bacteria; p__Bacteroidota; c__Bacteroidia; o__Bacteroidales; f__Rikenellaceae; g__Alistipes_A; s__MGYG000004197                    0.071748
# k__Bacteria; p__Firmicutes_A; c__Clostridia; o__Oscillospirales; f__Oscillospiraceae; g__F23-B02; s__sp900772725                      0.070766
# k__Bacteria; p__Firmicutes_A; c__Clostridia; o__Oscillospirales; f__Acutalibacteraceae; g__UMGS403; s__sp900541975                    0.068966
# k__Bacteria; p__Firmicutes_A; c__Clostridia; o__Oscillospirales; f__Acutalibacteraceae; g__UBA1691; s__sp900544715                    0.067427
# k__Bacteria; p__Bacteroidota; c__Bacteroidia; o__Bacteroidales; f__Bacteroidaceae; g__Phocaeicola; s__sp002493165                     0.065991
# k__Bacteria; p__Firmicutes_A; c__Clostridia; o__Lachnospirales; f__Lachnospiraceae; g__Eubacterium_I; s__ramulus                      0.063403                  0.022478





##################
### COLLASSATO ###
##################

# --- REPORT ---
# Best Parameters: {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}
# Accuracy: 0.6578947368421053
#               precision    recall  f1-score   support

#           NR       0.68      0.69      0.69        62
#            R       0.63      0.62      0.62        52

#     accuracy                           0.66       114
#    macro avg       0.65      0.65      0.65       114
# weighted avg       0.66      0.66      0.66       114


# --- FEATURE IMPORTANCE (Top 10) ---
# k__Bacteria; p__Bacteroidota; c__Bacteroidia; o__Bacteroidales; f__Muribaculaceae; g__CAG-1031              0.024561
# k__Bacteria; p__Firmicutes; c__Bacilli; o__Rubeoparvulales; f__Rubeoparvulaceae; g__Rubeoparvulum           0.020175
# k__Bacteria; p__Eremiobacterota; c__Xenobia; o__Xenobiales; f__Xenobiaceae; g__Bruticola                    0.020175
# k__Bacteria; p__Proteobacteria; c__Gammaproteobacteria; o__Burkholderiales; f__Rhodocyclaceae; g__SFHR01    0.020175
# k__Bacteria; p__Firmicutes_A; c__Clostridia; o__Oscillospirales; f__Acutalibacteraceae; g__UBA1691          0.020175
# k__Bacteria; p__Firmicutes_A; c__Clostridia; o__Oscillospirales; f__Acutalibacteraceae; g__CAG-177          0.020175
# k__Bacteria; p__Firmicutes_A; c__Clostridia_A; o__Christensenellales; f__UBA1242; g__UBA11517               0.019298
# k__Bacteria; p__Firmicutes_A; c__Clostridia; o__Monoglobales; f__Monoglobaceae; g__UMGS1494                 0.019298
# k__Bacteria; p__Firmicutes_A; c__Clostridia_A; o__Christensenellales; f__CAG-917; g__UMGS1688               0.018421
# k__Bacteria; p__Firmicutes_A; c__Clostridia; o__UMGS1883; f__UMGS1883; g__UMGS1883                          0.016667
# dtype: float64

# --- MUTUAL INFORMATION (Top 10) ---
# k__Bacteria; p__Bacteroidota; c__Bacteroidia; o__Bacteroidales; f__Bacteroidaceae; g__OM05-12                         0.066801
# k__Bacteria; p__Firmicutes; c__Bacilli; o__Lactobacillales; f__Lactobacillaceae; g__Weissella                         0.063971
# k__Bacteria; p__Firmicutes_A; c__Clostridia; o__Lachnospirales; f__Lachnospiraceae; g__Sellimonas                     0.057279
# k__Bacteria; p__Spirochaetota; c__Spirochaetia; o__Sphaerochaetales; f__Sphaerochaetaceae; g__Spiro-01                0.057089
# k__Bacteria; p__Firmicutes; c__Bacilli; o__Lactobacillales; f__Lactobacillaceae; g__Lactobacillus                     0.056265
# k__Bacteria; p__Firmicutes_A; c__Clostridia; o__Lachnospirales; f__Lachnospiraceae; g__VSOB01                         0.055141
# k__Bacteria; p__Firmicutes; c__Bacilli; o__Erysipelotrichales; f__Erysipelatoclostridiaceae; g__Massiliomicrobiota    0.052132
# k__Bacteria; p__Firmicutes_A; c__Clostridia; o__Oscillospirales; f__Ruminococcaceae; g__Ruthenibacterium              0.051617
# k__Bacteria; p__Cyanobacteria; c__Vampirovibrionia; o__Gastranaerophilales; f__Gastranaerophilaceae; g__UMGS1477      0.051116
# k__Bacteria; p__Firmicutes_A; c__Clostridia; o__Oscillospirales; f__Acutalibacteraceae; g__RUG420                     0.051085