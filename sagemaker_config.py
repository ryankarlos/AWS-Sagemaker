import sagemaker.session
import sagemaker
from enum import Enum


S3_BUCKET = "sagemaker_experiments"
S3_PREFIX_VIDEOGAMES = "videogames"
S3_PREFIX_IMDB = "imdb"
VIDEOGAME_ENDPOINT_NAME = "videogames"


class InstanceConfig(Enum):
    PROCESSING = "ml.m5.xlarge"
    TRAINING = "ml.p3.2xlarge"
    INFERENCE = "ml.g4dn.xlarge"


class Hyperparameters(Enum):
    EPOCHS = 1
    BATCH_SIZE = 32
    MODEL_NAME = "distilbert-base-uncased"


def get_session(sagemaker_session_bucket):
    role = sagemaker.get_execution_role()
    sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)
    return role, sess


def delete_endpoint(predictor):
    predictor.delete_endpoint()


ROLE, SESS = get_session("sagemaker_artifact")