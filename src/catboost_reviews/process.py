import os
import sys
import numpy as np
import pandas as pd
import argparse
import sagemaker
from sklearn.datasets import *
import sklearn.model_selection
from sklearn.datasets import make_regression
import sklearn.model_selection
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance


def show_ranked_feature_importance_list(scores, data):
    """
    Prints the features ranked by their corresponding importance score.
    """
    lst = list(zip(data.columns, scores))
    ranked_lst = sorted(lst, key=lambda t: t[1], reverse=True)
    print(pd.DataFrame(ranked_lst, columns=["Feature", "Importance Score"]))

def compute_feature_importance(X_train):
    """
    """
    X_data, y_label = make_regression(
    n_samples=X_train.shape[0], n_features=X_train.shape[1], n_informative=10, random_state=1
)
    k_nn_model = KNeighborsRegressor()
    k_nn_model.fit(X_data, y_label)
    feature_importances_permutations = permutation_importance(
        k_nn_model, X_data, y_label, scoring="neg_mean_squared_error"
    ).importances_mean

    for index, importance_score in enumerate(feature_importances_permutations):
        print("Feature: {}, Score: {}".format(X_train.columns[index], importance_score))
    return feature_importances_permutations


def remove_features(lst, data, threshold):
    """
    Remove features found in lst from data iff its importance score is below threshold.
    """
    features_to_remove = []
    for _, pair in enumerate(list(zip(data.columns, lst))):
        if pair[1] <= threshold:
            features_to_remove.append(pair[0])

    if features_to_remove:
        data.drop(features_to_remove, axis=1)

def process(args):
    data_set = fetch_california_housing()
    threshold = args.threshold
    test_sizr = args.test_size
    X = pd.DataFrame(data_set.data, columns=data_set.feature_names)
    Y = pd.DataFrame(data_set.target)

    print("Features:", list(X.columns))
    print("Dataset shape:", X.shape)
    print("Dataset Type:", type(X))
    print("Label set shape:", Y.shape)
    print("Label set Type:", type(X))

    # We partition the dataset into 2/3 training and 1/3 test set.
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.33)

    # We further split the training set into a validation set i.e., 2/3 training set, and 1/3 validation set
    X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(
        X_train, Y_train, test_size=test_size
        )

    feature_importances_permutations = compute_feature_importance(X_train)
    show_ranked_feature_importance_list(feature_importances_permutations, X_train)

    remove_features(lst=feature_importances_permutations, data=X_train, threshold=threshold)
    remove_features(lst=feature_importances_permutations, data=X_val, threshold=threshold)
    remove_features(lst=feature_importances_permutations, data=X_test, threshold=threshold)
    data_dir = "data/housing"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    prefix = "catboost-regressor"

    pd.concat([Y_train, X_train], axis=1).to_csv(
        os.path.join(data_dir, "train.csv"), header=False, index=False
    )
    pd.concat([Y_val, X_val], axis=1).to_csv(
        os.path.join(data_dir, "validation.csv"), header=False, index=False
    )
    # This instantiates a SageMaker session that we will be operating in.
    session = sagemaker.Session()

    val_location = session.upload_data(os.path.join(data_dir, "validation.csv"), key_prefix=prefix)
    train_location = session.upload_data(os.path.join(data_dir, "train.csv"), key_prefix=prefix)
    print(val_location)
    print(train_location)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Script to build Q2X model."
    )

    # sample code to get a hyperparam
    parser.add_argument("--test_size",
                        help="test_size",
                        type=float,
                        default=0.33)
    parser.add_argument("--threshold",
                    help="threshold",
                    type=float,
                    default=0.01)

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    process(args)
    sys.exit(0)