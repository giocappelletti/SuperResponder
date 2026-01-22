# Lavoro svolto durante un tirocinio presso UNIMORE

Studio sull'applicazione di tecniche di **Machine Learning** per la classificazione binaria della risposta all'immunoterapia in pazienti oncologici, basato sull'analisi del microbiota intestinale.

## Descrizione cartella DR_Modelli/
Contiene del lavoro sulla **Dimensionality Reduction** e allenamento dei seguenti modelli di **Machine Learning**:
  - Logistic Regression (LR), Support Vector Machine (SVM), Ridge (RLS con penalità L2), Multi Layer Perceptron (MLP)

### Struttura
```
DR_Modelli/
├──  DR.ipynb                 # Codice utilizzato per gli esperimenti di Dimensionality Reduction
├──  Models.ipynb             # Codice utilizzato per tutti gli esperimenti con i vari modelli di Machine Learning
├──  split.ipynb              # Codice utilizzato per effettuare gli split train/test dei vari dataset
├──  Datasets                 
│   ├──  raw_dataset.csv              # Dataset completo (esclusi stabili)
│   ├──  Splitted                     
│   │   ├──  Full/                     # 4630 taxa (split originale)
│   │   ├──  Collapsed/                # 1128 taxa collassati
│   │   └──  Cancer/                   # Dataset per tipologia di cancro
│   └──  Online/                       # Dataset esterni
├──  Papers/                  # Contiene un paper datoci dai colleghi di Scienze dalla Vita (`Microbiota.pdf`) e altri trovati online
└──  Results/                 # Contiene tutti i risultati degli esperimenti effettuati divisi in 5 cartelle (i grafici si trovano su drive) ognuna delle quali presenta un file
|                             # `notes.md` con la lista degli esperimenti effettuati, eventuali spiegazioni e una serie di risultati testuali  
└──  requirements.txt         # Librerie e versioni installate; scikit-bio, attualmente, è presente solo su linux
```

## Descrizione cartella EDA_Modelli/
Contiene del lavoro sull'**Exploratory Data Analysis** e allenamento dei seguenti modelli di **Machine Learning**:
  - Random Forest, Extra Trees e XGBoost


## Descrizione del dataset

Il dataset è costituito da dati sul **microbiota intestinale** di pazienti oncologici sottoposti a immunoterapia la cui risposta è presente nel campo `response` costituito da **R** (Responder) e **NR** (Non Responder). È inoltre presente una descrizione più dettagliata della risposta nel campo `ORR` che quindi non va utilizzato in fase di training in modo da evitare **data leakage**.

-  Ogni riga rappresenta un paziente e presenta:
  - **18 metadati** (es. age, sex, ...)
  - **4630 feature tassonomiche** che descrivono il microbiota (dati composizionali: somma = 100% per campione), a volte chiamate **taxa** nelle note
- I campioni in totale sono **784**
- **196** sono però stati esclusi poichè "stabili" (nessun effetto rilevabile della terapia)
- **19 campioni** di Modena, detti **Super Responder** poichè tutti **Responder** in modo significativo, non sono stati inclusi in fase di training ma  sono stati utilizzati come uno dei test set finali


## Esperimenti principali effettuati
1. **Exploratory Data Analysis**  
2. **Dimensionality Reduction**  
3. **Training modelli**  
   - Sul dataset completo (4630 features per le classificazioni tassonomiche)
   - Sul dataset collassato (1128 features per le classificazioni tassonomiche)
4. **Divisioni del dataset**  
   - Per tipologia di cancro  
   - Per nazionalità  
   - Per BioProject  
   - Per medoid/submedoid  
5. **Data Augmentation**  
6. **Utilizzo dei metadati**  
7. **Feature Selection**

Gli elenchi completi degli esperimenti effettuati si possono trovare nei file `notes.md` presenti nelle cinque cartelle in `DR_Modelli/Results`


##  Descrizione dei metadati e Value Counts
- **samples**: ID del paziente
- **avgSpotLen**: ?
- **Bases**: indica il numero totale di basi (nucleotidi) sequenziate per ciascun campione

I valori fanno riferimento al dataset esclusi gli stabili e i modenesi.
```python
# country         provenienza geografica dei campioni
France             265
North America      169
United Kingdom      81
Netherlands         44
Spain               10

# continent       distribuzione geografica a livello continentale
Europe             400
USA                169

# sex             genere dei pazienti
male               352
female             178
Male                14
Female              13
NaN                 12

# age             fasce d'età
60-70             194
70-80             128
50-60             102
40-50              46
80-90              39
NaN                29
90-100             15
30-40              12
20-30               3
10-20               1

# atb             indica se il paziente ha avuto una terapia antibiotica
no                334
NaN               208
yes                27

# cancer          tipologia di cancro
MM                292
NSCLC             239
RCC                38

# therapy         tipologia di immunoterapia ricevuta
PDL-1             231
CTLA-4+PD-1       169
PD-1              126
PD-1+PD-L1         38
CTLA-4              5

# SRA             codici degli studi da cui arrivano i dati
SRP331625         183
ERP127050         135
ERP104577          82
SRP292135          47
SRP115355          34
SRP197281          27
SRP116709          27
SRP339782          22
SRP390205          12

# Platform        indica le piattaforme di sequenziamento usate per ottenere i dati
ILLUMINA          304
ION_TORRENT       265

# BioProject      associa ciascun campione a specifici progetti di ricerca
PRJNA751792       183
PRJEB43119        135
PRJEB22863         82
PRJNA672867        47
PRJNA397906        34
PRJNA541981        27
PRJNA399742        27
PRJNA762360        22
PRJNA866654        12

# LibraryLayout
PAIRED            304
SINGLE            265

# ORR             indica la risposta tumorale ai trattamenti
PD                284
PR                183
CR                 67
Dead                8

# response        rappresenta la classificazione binaria della risposta
NR                307
R                 262

# medoids         ?
1.0               432
2.0               137

# submedoids      ?
1_2               273
1_1               159
2_2                90
2_1                47
```
Notiamo diversi NaN in campi quali `sex`, `age`, `atb` e `ORR` che vanno quindi gestiti se utilizzati.
Nel campo `sex` è presente sia "male" che "Male" e sia "female" che "Female" che va quindi gestito.

