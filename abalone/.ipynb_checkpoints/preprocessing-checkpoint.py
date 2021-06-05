import argparse
import os
import requests
import tempfile
import numpy as np
import pandas as pd


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from .constants import  *
from .io import read_data_from_csv, save_output_to_csv


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


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
    

def train_val_test_split(X):
    np.random.shuffle(X)
    train, validation, test = np.split(X, [int(.7*len(X)), int(.85*len(X))])
    train_val_test_dict = {"train":train, "validation": validation, "test":test}
    return train_val_test_dict

    
if __name__ == "__main__":
    
    df = read_data_from_csv()

    y = df.pop(LABEL)
    X_pre = preprocess_pipeline().fit_transform(df)
   
    X = np.concatenate((y_pre, X_pre), axis=1)
    
    train_val_test_dict = train_val_test_split(X)
    save_output_to_csv(**train_val_test_dict)
    