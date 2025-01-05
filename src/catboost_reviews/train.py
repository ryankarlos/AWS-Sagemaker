#!/usr/bin/python3

from __future__ import print_function

import os
import sys
import argparse
import sagemaker
# CatBoost Regression Model 
from catboost import CatBoostRegressor 


def train():
    prefix = '/opt/ml/'

    input_path = prefix + 'input/data'
    output_path = os.path.join(prefix, 'output')
    model_path = os.path.join(prefix, 'model')
    # This algorithm has a single channel of input data called 'train'. Since we run in
    # File mode, the input files are copied to the directory specified here.
    train_channel_name = 'train'
    validation_chanel_name = 'validation'
    training_path = os.path.join(input_path, train_channel_name)
    validation_path = os.path.join(input_path, validation_chanel_name)
    ## added new column browse node
    # Initialize the CatBoostRegressor with RMSE as the loss function 
    model = CatBoostRegressor(loss_function='RMSE') 
    # Fit the model on the training data with verbose logging every 100 iterations 
    model.fit(X_train, Y_train, verbose=100) 


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
    train(args)
    sys.exit(0)