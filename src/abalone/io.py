import pandas as pd
from .constants import base_dir
import os

def read_data_from_csv(filename="abalone-dataset.csv"):
    df = pd.read_csv(os.path.join(base_dir, filename),
        header=None, 
        names=feature_columns_names + [label_column],
        dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype)
    )
    
    return df


def save_output_to_csv(**kwargs):
    for k,v in kwargs:
        pd.DataFrame(v).to_csv(f"{base_dir}/{k}/{k}.csv", header=False, index=False))
        