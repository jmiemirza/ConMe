import os
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import shutil


def load_metrics(path, verbose=True, num_rows=20000):
    """
    Load in all of the metrics in the directory
    from path.
    """
    # Total df 
    total_df = pd.DataFrame()

    result_folders = os.listdir(path)
    broken_dirs = []
    for rf in tqdm(result_folders):
        exp_path = os.path.join(path, rf)
        try:
            pkl_path = os.path.join(exp_path, 'results_dataframe.pkl')
            df = pd.read_pickle(pkl_path).head(num_rows)
            total_df = pd.concat([total_df, df], axis=0)
        except:
            if verbose:
                print(f"Error in {exp_path}, no valid .pkl found.")
            broken_dirs.append(exp_path)
            pass

    return total_df, broken_dirs


def clean_up_results(broken_dirs):
    if len(broken_dirs) == 0:
        print("No broken directories found.")
        return
    confirmation = input(f"Are you sure you want to delete {len(broken_dirs)} directories? (y/n)")
    if confirmation == 'y':
        for bd in broken_dirs:
            shutil.rmtree(bd)
