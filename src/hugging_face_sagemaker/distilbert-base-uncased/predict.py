import logging
import os
import sys
from sagemaker.s3 import S3Downloader
from sagemaker_config import SESS, ROLE
from sagemaker.huggingface.model import HuggingFaceModel


logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def download_model_from_s3(estimator):
    S3Downloader.download(
        s3_uri=estimator.model_data,
        local_path='.',                          # local path where *.targ.gz is saved
        sagemaker_session=SESS                   # SageMaker session used for training the model
    )


def deploy_model(estimator):
    huggingface_model = HuggingFaceModel(
        model_data=estimator.model_data,  # path to your trained SageMaker model
        role=ROLE,  # IAM role with permissions to create an endpoint
        transformers_version="4.6",  # Transformers version used
        pytorch_version="1.7",  # PyTorch version used
        py_version='py36',  # Python version used
    )
    # deploy model to SageMaker Inference
    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.xlarge"
    )
    return predictor


def generate_prediction(input_text, predictor):
    predictor.predict(input_text)


if __name__ == "__main__":
    predictor = deploy_model()
    input_text = {
        "inputs": "It feels like a curtain closing...there was an elegance in the way they moved toward conclusion. "
                  "No fan is going to watch and feel short-changed."}

    generate_prediction(input_text, predictor)