import sagemaker.session
import sagemaker
from enum import Enum
from sagemaker import image_uris

HUGGING_FACE_URI = image_uris.retrieve(framework='huggingface',region='us-east-1',version='4.17.0',image_scope='training',base_framework_version='pytorch1.10.2')
S3_BUCKET = "sagemaker-experiments-ml"
S3_PREFIX_VIDEOGAMES = "videogames"
S3_PREFIX_IMDB = "imdb"
VIDEOGAME_ENDPOINT_NAME = "videogames"
ROLE_NAME = "SagemakerMainRole"


class InstanceConfig(Enum):
    PROCESSING = "ml.m5.xlarge"
    TRAINING = "ml.m5.2xlarge"
    INFERENCE = "ml.m5.2xlarge"


class Hyperparameters(Enum):
    EPOCHS = 1
    BATCH_SIZE = 32
    MODEL_NAME = "distilbert-base-uncased"


def get_session(sagemaker_session_bucket):
    sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)
    return sess


def delete_endpoint(predictor):
    predictor.delete_endpoint()


SESS = get_session(S3_BUCKET)