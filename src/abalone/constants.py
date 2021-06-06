import numpy as np
import os

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

INPUT_DATA_DIR = "/opt/ml/processing/input"
TRAIN_DIR = "/opt/ml/processing/train"
TEST_DIR = "/opt/ml/processing/test"


HP={ "max_depth":"5",
    "eta":"0.2",
    "gamma":"4",
    "min_child_weight":"6",
    "subsample":"0.7",
    "objective":"reg:squarederror",
    "num_round":"50"
   }