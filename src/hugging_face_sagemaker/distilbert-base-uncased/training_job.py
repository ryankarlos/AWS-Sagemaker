import logging
import os
import sys
from sagemaker.huggingface import HuggingFace
from sagemaker_config import ROLE, Hyperparameters, InstanceConfig
from preprocess import get_bucket_paths

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def create_hugging_face_estimator(distributed_training=False):
    hyperparameters = {
        "epochs": Hyperparameters.EPOCHS.value,  # number of training epochs
        "train_batch_size": Hyperparameters.BATCH_SIZE.value
        ,  # training batch size
        "model_name": Hyperparameters.MODEL_NAME.value  # name of pretrained model
    }

    metric_definitions = [
        {"Name": "train_runtime", "Regex": "train_runtime.*=\D*(.*?)$"},
        {"Name": "eval_accuracy", "Regex": "eval_accuracy.*=\D*(.*?)$"},
        {"Name": "eval_loss", "Regex": "eval_loss.*=\D*(.*?)$"},
    ]
    huggingface_estimator = HuggingFace(
        entry_point="train.py",  # fine-tuning script to use in training job
        source_dir="./scripts",  # directory where fine-tuning script is stored
        instance_type=InstanceConfig.TRAINING.value,  # instance type
        instance_count=1,  # number of instances
        role=ROLE,  # IAM role used in training job to acccess AWS resources (S3)
        transformers_version="4.6",  # Transformers version
        pytorch_version="1.7",  # PyTorch version
        py_version="py39",  # Python version
        metric_definitions=metric_definitions,
        hyperparameters=hyperparameters  # hyperparameters to use in training job
    )
    if distributed_training:
        # configuration for running training on smdistributed data parallel
        distribution = {'smdistributed': {'dataparallel': {'enabled': True}}}
        huggingface_estimator.distribution = distribution
    return huggingface_estimator


def train_estimator(huggingface_estimator):
    training_input_path, test_input_path = get_bucket_paths()
    huggingface_estimator.fit({"train": training_input_path, "test": test_input_path})


if __name__ == "__main__":
    estimator = create_hugging_face_estimator()
    train_estimator(estimator)
