import sagemaker.session
import sagemaker
from enum import Enum
import boto3
from sagemaker.predictor import Predictor
from sagemaker import image_uris

HUGGING_FACE_URI = image_uris.retrieve(framework='huggingface',region='us-east-1',version='4.17.0',image_scope='training',base_framework_version='pytorch1.10.2')
S3_BUCKET = "sagemaker-experiments-ml"
S3_PREFIX_VIDEOGAMES = "videogames"
S3_PREFIX_IMDB = "imdb"
VIDEOGAME_ENDPOINT_NAME = "videogames"
ROLE = "sagemaker-experiments-ml-role"
ROLE_NAME = "SagemakerMainRole"

class InstanceConfig(Enum):
    PROCESSING = "ml.m5.xlarge"
    TRAINING = "ml.m5.2xlarge"
    INFERENCE = "ml.m5.2xlarge"


class Hyperparameters(Enum):
    EPOCHS = 1
    BATCH_SIZE = 32
    MODEL_NAME = "bert-base-uncased"


def get_session():
    sess = sagemaker.Session(default_bucket=S3_BUCKET)
    return sess


def get_role(role_name):
    iam_client = boto3.client('iam')
    role_arn = iam_client.get_role(RoleName=role_name)['Role']['Arn']
    return role_arn


def delete_endpoint(*endpoints):
    for name in endpoints:
        predictor = Predictor(endpoint_name=name, sagemaker_session=get_session())
        predictor.delete_endpoint()


ROLE_ARN = get_role(ROLE_NAME)
SESS = get_session()

if __name__ == "__main__":
    print(ROLE_ARN)
    #delete_endpoint([VIDEOGAME_ENDPOINT_NAME])
