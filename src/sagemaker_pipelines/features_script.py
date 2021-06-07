import argparse
import os
import requests
import tempfile
import numpy as np
import pandas as pd
import sys
import argparse

import logging
import numpy as np
import os
import pandas as pd
import sys
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
warnings.simplefilter("ignore", category=DeprecationWarning)
from sklearn.preprocessing import StandardScaler, OneHotEncoder


logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


NUMERIC_COLS = [
    "length",
    "diameter",
    "height",
    "whole_weight",
    "shucked_weight",
    "viscera_weight",
    "shell_weight",
]

CAT_COLS = ["sex"]
LABEL = "rings"

FEATURE_COLS_DTYPE= {
    "sex": str,
    "length": np.float64,
    "diameter": np.float64,
    "height": np.float64,
    "whole_weight": np.float64,
    "shucked_weight": np.float64,
    "viscera_weight": np.float64,
    "shell_weight": np.float64
}
LABEL_DTYPE = {"rings": np.float64}



def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


def read_data_from_csv(filepath):
    names = CAT_COLS + NUMERIC_COLS + [LABEL]
    logger.info(f"setting header names as: {names}")
    df = pd.read_csv(filepath,
        header=None, 
        names=names,
        dtype=merge_two_dicts(FEATURE_COLS_DTYPE, LABEL_DTYPE)
    )
    
    return df


def save_output_to_csv(base_dir, **kwargs):
    for k,v in kwargs.items():
        par_dir = os.path.join(base_dir, k)
        if not os.path.exists(par_dir):
            logger.info('Train/Val/Test dirs do not exist so creating them')
            os.makedirs(par_dir)           
        pd.DataFrame(v).to_csv(os.path.join(par_dir, f"{k}.csv"), header=False, index=False)
        

def numerical_transformer():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )
    
def cat_transformer():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

def preprocess_pipeline(df):    
    preprocess_transformer = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer(), NUMERIC_COLS),
            ("cat", cat_transformer(), CAT_COLS)
        ]
    )
    X = df.drop(LABEL, axis=1)
    return preprocess_transformer.fit_transform(X)
    

def reshape_label_col(df):
    y = df.loc[:, LABEL]
    return y.to_numpy().reshape(len(y), 1)    
    

def concat_feat_label_after_transform(X_pre, y_pre):
    logger.info("concatenating features and label after transformation")
    return np.concatenate((y_pre, X_pre), axis=1)
    
def train_val_test_split(X, val_test_split):
    np.random.shuffle(X)
    logger.info('splitting data into train/val/test')
    train_split = 1- (2*val_test_split) 
    train, validation, test = np.split(X, [int(train_split*len(X)), int((1-val_test_split)*len(X))])
    train_val_test_dict = {"train":train, "validation": validation, "test":test}
    return train_val_test_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--input-data-path', type=str, default="/opt/ml/processing/input/abalone-dataset.csv")
    parser.add_argument('--train-val-test-dir', type=str, default="/opt/ml/processing")
    parser.add_argument('--val-test-split', type=float, default=0.15)

    args = parser.parse_args()
    
    df = read_data_from_csv(args.input_data_path)
    X_pre = preprocess_pipeline(df)
    y_pre = reshape_label_col(df)   
    X = concat_feat_label_after_transform(X_pre, y_pre)
    
    train_val_test_dict = train_val_test_split(X, args.val_test_split)
    save_output_to_csv(args.train_val_test_dir, **train_val_test_dict)
    