import os
import sys
import numpy as np
import polars as pl
import argparse
import sagemaker
from sklearn.datasets import *
import sklearn.model_selection
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
from pathlib import Path
import os

# This instantiates a SageMaker session that we will be operating in.


def show_ranked_feature_importance_list(scores, data):
    """
    Prints the features ranked by their corresponding importance score.
    """
    lst = list(zip(data.columns, scores))
    ranked_lst = sorted(lst, key=lambda t: t[1], reverse=True)
    # Convert to Polars DataFrame
    return pl.DataFrame(ranked_lst, schema=["Feature", "Importance Score"])

def compute_feature_importance(X_train):
    """
    Compute feature importance using permutation importance
    """
    # Convert Polars DataFrame to numpy array for sklearn compatibility
    X_data, y_label = make_regression(
        n_samples=X_train.shape[0], 
        n_features=X_train.shape[1], 
        n_informative=10, 
        random_state=1
    )
    k_nn_model = KNeighborsRegressor()
    k_nn_model.fit(X_data, y_label)
    feature_importances_permutations = permutation_importance(
        k_nn_model, X_data, y_label, scoring="neg_mean_squared_error"
    ).importances_mean

    for index, importance_score in enumerate(feature_importances_permutations):
        print(f"Feature: {X_train.columns[index]}, Score: {importance_score}")
    return feature_importances_permutations

def remove_features(lst, data, threshold):
    """
    Remove features found in lst from data if its importance score is below threshold.
    """
    features_to_remove = []
    for _, pair in enumerate(list(zip(data.columns, lst))):
        if pair[1] <= threshold:
            features_to_remove.append(pair[0])

    if features_to_remove:
        return data.drop(features_to_remove)
    return data

def process(args):
    os.environ["AWS_DEFAULT_REGION"] = args.region
    session = sagemaker.Session()
    data_set = fetch_california_housing()
    threshold = args.threshold
    train_ratio = args.train_ratio
    validation_ratio = args.val_ratio
    test_ratio = args.test_ratio
      
    data_dir = args.data_output_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Create Polars DataFrames
    X = pl.DataFrame(
        data_set.data,
        schema=data_set.feature_names
    )
    Y = pl.DataFrame(
        data_set.target,
        schema=['target']
    )

    print("Features:", X.columns)
    print("Dataset shape:", X.shape)
    print("Dataset Type:", type(X))
    print("Label set shape:", Y.shape)
    print("Label set Type:", type(Y))

    # Convert to numpy for sklearn split then back to Polars
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
        X.to_numpy(), Y.to_numpy(), test_size=1 - train_ratio
    )
    X_val, X_test, Y_val, Y_test = sklearn.model_selection.train_test_split(
        X_test, Y_test, test_size=test_ratio/(test_ratio + validation_ratio)
    )

    # Convert numpy arrays back to Polars DataFrames
    X_train = pl.DataFrame(X_train, schema=X.columns)
    X_val = pl.DataFrame(X_val, schema=X.columns)
    X_test = pl.DataFrame(X_test, schema=X.columns)
    Y_train = pl.DataFrame(Y_train, schema=['target'])
    Y_val = pl.DataFrame(Y_val, schema=['target'])
    Y_test = pl.DataFrame(Y_test, schema=['target'])

    print(X_train.shape, X_val.shape, X_test.shape)

    feature_importances_permutations = compute_feature_importance(X_train)
    importance_df = show_ranked_feature_importance_list(feature_importances_permutations, X_train)
    print(importance_df)

    data = {"train": (X_train, Y_train), "val": (X_val, Y_val), "test": (X_test, Y_test)}
    for k, v in data.items():
        X_data, Y_data = v
        X_data = remove_features(lst=feature_importances_permutations, data=X_data, threshold=threshold)
        
        # Concatenate and write to CSV
        combined_data = pl.concat([Y_data, X_data], how="horizontal")
        combined_data.write_csv(
            os.path.join(data_dir, f"{k}.csv"),
            include_header=True
        )
        
        bucket_path = session.upload_data(f"{data_dir}/{k}.csv", key_prefix=k)
        print(f"Uploaded {k} dataset to {bucket_path}")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Script to process data for housing dataset."
    )
    parser.add_argument("--data_output_dir",
                        help="data_output_dir",
                        type=str,
                        default="data/processed")
    

    

    parser.add_argument("--test_ratio",
                        help="test_ratio",
                        type=float,
                        default=0.10)
    parser.add_argument("--val_ratio",
                        help="val_ratio",
                        type=float,
                        default=0.15)
    parser.add_argument("--train_ratio",
                    help="train_ratio",
                    type=float,
                    default=0.75)
    parser.add_argument("--threshold",
                    help="threshold",
                    type=float,
                    default=0.01)
    parser.add_argument("--region",
                    help="aws region",
                    type=str,
                    default="us-east-1")

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    process(args)
    sys.exit(0)