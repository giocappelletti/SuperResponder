import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir, os.pardir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from skbio.diversity import beta_diversity
from skbio.stats.distance import permanova, permdisp

from fileio.df_loader import DataLoader
from logger.logger import logger

def main(): 

    # Load Dataset 
    file_path = 'datasets/raw_dataset.csv' 

    df, microbial_cols, meta_cols = DataLoader().load_dataset(file_path, drop_response=False) 

    # Extract abundance matrix 
    abundance_matrix = df[microbial_cols] 

    abundance_matrix = abundance_matrix.fillna(0) 

    grouping_variables = ['response', 'therapy', 'cancer', 'atb', 'sex', 'country', 'continent']

    for g_var in grouping_variables:
        
        logger.info(f"Running PERMANOVA and PERMDISP for {g_var} variable")

        # Filter the DataFrame to remove NaNs in the current grouping variable
        # Filter DataFrame
        df_current = df.dropna(subset=[g_var]).copy()

        # Ensure the grouping variable is of type 'category'
        df_current[g_var] = df_current[g_var].astype('category')

        # Filter the abundance matrix to match the filtered DataFrame
        abundance_matrix_current = abundance_matrix.loc[df_current.index]

        # Ensure there are at least two unique groups for the analysis
        if df_current[g_var].nunique() < 2:
            logger.warning(f"Not enough unique groups for analysis, Skipping PERMANOVA and PERMDISP for {g_var}")
            continue

        # Ensure There are enough samples per group for PERMDISP (at least 3 per group is recommended).
        # PERMDISP requires at least 3 samples per group to calculate dispersion.
        group_counts = df_current[g_var].value_counts()
        if any(group_counts < 3):
            logger.warning(f"Some groups in '{g_var}' have fewer than 3 samples. PERMDISP may be unreliable or unworkable.")
        # PERMANOVA can still be run if there are at least 2 samples per group.

        # 1. Calculate the distance matrix (Bray-Curtis)
        # Use 'samples' as the unique ID for the samples.
        # Since abundance_matrix_current already has 'samples' as an index,
        # beta_diversity will use these as IDs for the distance matrix.
        dm = beta_diversity("braycurtis", abundance_matrix_current) # Removed ids=df_current['samples'] 

        #2. Run PERMANOVA 
        try: 
            permanova_results = permanova(dm, df_current[g_var]) 
            print(permanova_results) 
        except ValueError as e: 
            logger.error(f"Error executing PERMANOVA: {e}") 

        #3. Run PERMDISP 
        try: 
            permdisp_results = permdisp(dm, df_current[g_var], warn_neg_eigval=0.5) 
            print(permdisp_results) 

        except ValueError as e:
            logger.error(f"Error executing PERMDISP: {e}")


if __name__ == "__main__":
    main()