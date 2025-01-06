#!/usr/bin/python3

from __future__ import print_function
import polars as pl
import os
import sys
import numpy as np
import argparse
import sagemaker
import logging
from catboost import metrics
# Import the mean squared error (MSE) function from sklearn and alias it as 'mse' 
from sklearn.metrics import mean_squared_error as mse 
  
# CatBoost Regression Model 
from catboost import CatBoostRegressor 

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def train(args):
    model_dir = args.model_dir
        
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = args.model_name

    training_path = os.path.join(args.train , "train.csv")
    validation_path = os.path.join(args.test,  "val.csv")
    train = pl.read_csv(training_path)
    val = pl.read_csv(validation_path)
    print(train.head())
    print(val.head())
    Y_train =train.select(pl.col("target"))
    Y_val = val.select(pl.col("target"))
    X_train = train.select(pl.all().exclude("target"))
    X_val = val.select(pl.all().exclude("target"))

    model_params ={"iterations":args.iterations,
    "learning_rate":args.lr,
    "depth":args.depth,
    "loss_function":args.loss_function}

    # Initialize the CatBoostRegressor with RMSE as the loss function 
    model = CatBoostRegressor(**model_params) 
    # Fit the model on the training data with verbose logging every 100 iterations 
    model.fit(X_train.to_pandas(), Y_train.to_pandas(), verbose=100) 

    print(model.get_best_score())

    logger.info('validating model')
    preds = model.predict(X_val.to_pandas())
    actuals = list(Y_val.to_numpy().flatten())
    rmse = metrics.RMSE().eval(actuals, preds)
    logger.info('RMSE: {}'.format(rmse))
    path = os.path.join(model_dir, model_name)
    logging.info('saving to {}'.format(path))
    model.save_model(path)

    m = model.load_model(path)
    print(model.predict(X_val.to_pandas()))
    
def get_parser():
    parser = argparse.ArgumentParser(
        description="Script to build Q2X model."
    )

    # sample code to get a hyperparam
    parser.add_argument("--model_dir",
                        help="path to the directory to write model artifacts to",
                        type=str,
                        default=os.environ.get('SM_MODEL_DIR', "model/output/data"))
    parser.add_argument("--output_data_dir",
                    help="filesystem path to write output artifacts to.",
                    type=str,
                    default=os.environ.get('SM_OUTPUT_DIR', "model"))
    parser.add_argument('--model-name', type=str, default='catboost_model')
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', "data/processed"))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST',  "data/processed"))
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--loss_function', type=str, default="RMSE")


    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    train(args)
    sys.exit(0)