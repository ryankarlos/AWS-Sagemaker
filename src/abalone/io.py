import pandas as pd
from src.abalone.constants import *
import os
import logging
import sys
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


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
        