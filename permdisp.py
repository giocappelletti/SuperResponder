import pandas as pd
import numpy as np
from skbio.diversity import beta_diversity
from skbio.stats.distance import permanova, permdisp
import matplotlib.pyplot as plt
import seaborn as sns

from utils.df_loader import DataLoader

def main():

    # Carica il dataset
    file_path = 'datasets/raw_dataset.csv'
    
    df, microbial_cols, meta_cols = DataLoader().load_dataset(file_path, drop_response=False)



    # Estrai la matrice di abbondanza
    abundance_matrix = df[microbial_cols]

    # Gestisci i valori NaN nella matrice di abbondanza (sostituisci con 0, pratica comune per dati di microbioma)
    abundance_matrix = abundance_matrix.fillna(0)

    # Variabili di raggruppamento di interesse
    grouping_variables = ['response', 'therapy', 'cancer', 'atb', 'sex', 'country', 'continent']

    # Esegui PERMANOVA e PERMDISP per ogni variabile di raggruppamento
    for g_var in grouping_variables:
        print(f"\n=====================================================================")
        print(f"Analisi per la variabile di raggruppamento: '{g_var}'")
        print(f"=====================================================================")

        # Filtra il DataFrame per rimuovere i NaN nella variabile di raggruppamento corrente
        df_current = df.dropna(subset=[g_var]).copy() # Aggiungi .copy() per evitare SettingWithCopyWarning
        
        # Assicurati che la variabile di raggruppamento sia di tipo 'category'
        df_current[g_var] = df_current[g_var].astype('category')

        # Filtra la matrice di abbondanza per corrispondere al DataFrame filtrato
        abundance_matrix_current = abundance_matrix.loc[df_current.index]

        # Verifica che ci siano almeno due gruppi unici per l'analisi
        if df_current[g_var].nunique() < 2:
            print(f"Attenzione: Meno di 2 gruppi unici in '{g_var}'. PERMANOVA/PERMDISP non possono essere eseguiti.")
            continue

        # Verifica che ci siano abbastanza campioni per gruppo per PERMDISP (almeno 3 per gruppo è consigliato)
        # PERMDISP richiede almeno 3 campioni per gruppo per calcolare la dispersione.
        group_counts = df_current[g_var].value_counts()
        if any(group_counts < 3):
            print(f"Attenzione: Alcuni gruppi in '{g_var}' hanno meno di 3 campioni. PERMDISP potrebbe non essere affidabile o non eseguibile.")
            # PERMANOVA può comunque essere eseguito se ci sono almeno 2 campioni per gruppo.

        # 1. Calcola la matrice di distanza (Bray-Curtis)
        # Utilizza 'samples' come ID univoco per i campioni
        # Poiché abundance_matrix_current ha già 'samples' come indice,
        # beta_diversity userà questi come ID per la matrice di distanza.
        dm = beta_diversity("braycurtis", abundance_matrix_current) # Rimosso ids=df_current['samples']

        # 2. Esegui PERMANOVA
        print(f"\n--- Risultati PERMANOVA per '{g_var}' ---")
        try:
            permanova_results = permanova(dm, df_current[g_var])
            print(permanova_results)
        except ValueError as e:
            print(f"Errore durante l'esecuzione di PERMANOVA: {e}")

        # 3. Esegui PERMDISP
        print(f"\n--- Risultati PERMDISP per '{g_var}' ---")
        try:
            permdisp_results = permdisp(dm, df_current[g_var], warn_neg_eigval=0.5)
            print(permdisp_results)


        except ValueError as e:
            print(f"Errore durante l'esecuzione di PERMDISP: {e}")



if __name__ == "__main__":
    main()
