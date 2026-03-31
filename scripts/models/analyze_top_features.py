import pandas as pd
import os
import sys
import ast # Per valutare in modo sicuro le rappresentazioni stringa delle liste
from itertools import combinations

# Aggiungi la root del progetto a sys.path
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def jaccard_similarity(list1, list2):
    """Calcola la similarità di Jaccard tra due liste."""
    s1 = set(list1)
    s2 = set(list2)
    if not s1 and not s2: # Gestisce il caso di liste vuote per evitare divisione per zero
        return 1.0
    return len(s1.intersection(s2)) / len(s1.union(s2))

def analyze_top_features_consistency(file_path):
    """
    Analizza la consistenza delle feature principali tra i diversi fold.

    Args:
        file_path (str): Percorso del file top_features.csv.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Errore: File non trovato in {file_path}")
        return

    print(f"Analisi della consistenza delle feature dal file: {file_path}\n")

    # Si assume che ogni cella contenga una rappresentazione stringa di una lista.
    # Dobbiamo convertire queste stringhe in liste effettive.
    # Le colonne dovrebbero essere nominate come '10', '20', '30', ecc.
    def parse_feature_list(cell_content):
        if pd.isna(cell_content) or str(cell_content).strip() == '':
            return [] # Tratta NaN o stringhe vuote come liste vuote
        try:
            # Valuta in modo sicuro il contenuto della stringa
            evaluated_content = ast.literal_eval(str(cell_content))
            
            # Se è già una lista, restituiscila
            if isinstance(evaluated_content, list):
                return evaluated_content
            # Se è un intero 0, o una stringa '0', interpretalo come una lista vuota
            elif evaluated_content == 0 or str(evaluated_content).strip() == '0':
                return []
            else:
                # Se è un altro tipo di letterale, è inatteso per una colonna di liste di feature.
                print(f"Warning: La cella '{cell_content}' è stata valutata come un tipo inatteso ({type(evaluated_content)}). Verrà trattata come lista vuota.")
                return []
        except (ValueError, SyntaxError) as e:
            # Se ast.literal_eval fallisce, la stringa non è un letterale Python valido.
            # Questo indica un'entrata malformata. Verrà trattata come lista vuota.
            print(f"Error parsing cell '{cell_content}': {e}. Verrà trattata come lista vuota.")
            return []
    for col in df.columns:
        df[col] = df[col].apply(parse_feature_list)

    results = {}

    for n_features_col in df.columns:
        print(f"--- Analisi per N={n_features_col} feature ---")
        feature_lists_for_n = df[n_features_col].tolist() # Ottiene le 5 liste per questo N

        if not feature_lists_for_n:
            print(f"Nessun dato per N={n_features_col}. Salto l'analisi.")
            continue

        # Converti le liste in set per facilitare le operazioni sugli insiemi
        feature_sets_for_n = [set(f_list) for f_list in feature_lists_for_n]

        # 1. Intersezione (feature comuni a tutti i fold)
        common_features = set.intersection(*feature_sets_for_n)
        print(f"  Feature comuni a tutti i fold ({len(common_features)})")

        # 2. Unione (tutte le feature uniche tra tutti i fold)
        all_unique_features = set.union(*feature_sets_for_n)
        print(f"  Totale feature uniche tra tutti i fold ({len(all_unique_features)})")

        # 3. Similarità di Jaccard tra i fold
        jaccard_similarities = []
        fold_pairs = list(combinations(range(len(feature_sets_for_n)), 2)) # es. (0,1), (0,2), ...
        
        for i, j in fold_pairs:
            sim = jaccard_similarity(feature_lists_for_n[i], feature_lists_for_n[j])
            jaccard_similarities.append(sim)
            #print(f"  Similarità di Jaccard (Fold {i+1} vs Fold {j+1}): {sim:.4f}")

        if jaccard_similarities:
            avg_jaccard = sum(jaccard_similarities) / len(jaccard_similarities)
            min_jaccard = min(jaccard_similarities)
            max_jaccard = max(jaccard_similarities)
            print(f"  Similarità di Jaccard media tra i fold: {avg_jaccard:.4f}")
            print(f"  Similarità di Jaccard minima tra i fold: {min_jaccard:.4f}")
            print(f"  Similarità di Jaccard massima tra i fold: {max_jaccard:.4f}")
        else:
            print("  Impossibile calcolare le similarità di Jaccard (meno di 2 fold).")

        results[n_features_col] = {
            'common_features_count': len(common_features),
            'total_unique_features_count': len(all_unique_features),
            'avg_jaccard_similarity': avg_jaccard if jaccard_similarities else None,
            'min_jaccard_similarity': min_jaccard if jaccard_similarities else None,
            'max_jaccard_similarity': max_jaccard if jaccard_similarities else None,
        }
        print("-" * 40)

if __name__ == "__main__":
    top_features_file = os.path.join(project_root, 'top_features.csv')
    analyze_top_features_consistency(top_features_file)