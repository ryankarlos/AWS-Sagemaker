import logging
import os
import sys
from sagemaker.huggingface import HuggingFace
from sagemaker_config import ROLE, Hyperparameters, InstanceConfig
from .preprocess import *


logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

