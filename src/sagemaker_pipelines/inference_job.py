import boto3
from sagemaker.serializers import CSVSerializer
from sagemaker.predictor import Predictor
from utils.sagemaker_config import *
from utils.aws_helper import get_pre_trained_model_from_bucket
import logging
import os
import sys
import argparse



logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)




def deploy_model(estimator):
    
    predictor=estimator.deploy(
    initial_instance_count=INSTANCE_CONFIG['inference_instance_count'],
    instance_type=INSTANCE_CONFIG['inference_instance_type'],
    serializer=CSVSerializer()
    )
    
    return predictor


def real_time_inference_job(endpoint_name):
    predictor = Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=SAGEMAKER_SESSION,
        serializer=CSVSerializer()
    )
    
    logger.info(f"Endpoint context for {endpoint_name}: {predictor.endpoint_context()}\n")
    
    return predictor
    
    
    
def batch_inference_job(estimator, bucket_prefix="abalone"):

    # The location of the test dataset
    batch_input = 's3://{}/{}/test'.format(DEFAULT_BUCKET, bucket_prefix)

    # The location to store the results of the batch transform job
    batch_output = 's3://{}/{}/batch-prediction'.format(DEFAULT_BUCKET, bucket_prefix)
    
    logger.info(f"Starting batch inference job-input path:{batch_input}, output path:{batch_output}")
    
    transformer = estimator.transformer(
    instance_count=INSTANCE_CONFIG['inference_instance_count'], 
    instance_type=INSTANCE_CONFIG['batch_transform_instance_type'],
    output_path=batch_output
    )
    
    transformer.transform(
    data=batch_input, 
    data_type='S3Prefix',
    content_type='text/csv', 
    split_type='Line'
    )
    transformer.wait()

    return transformer
   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', type=str, default=None)
    parser.add_argument('--payload', type=str, default='0.3415, 0.4244, 0.1311, 0.04233, 0.2799, -0.110342, -0.099, 1.0, 0, 0')
    args = parser.parse_args()

    logger.info(f"starting runner {sys.argv[0]} ....")
    logger.info(sys.argv[1:])

    
    if args.endpoint is None:
        try:
            model_uri = os.environ['MODEL_URI']
        except:
            logger.error("endpoint not specified in command line args so env var MODEL_URI must be specified")
        estimator = get_pre_trained_model_from_bucket(model_uri, ROLE, IMAGE_METADATA["framework"], IMAGE_METADATA["region"], IMAGE_METADATA["version"])
    
        predictor = deploy_model(estimator)
        endpoint = predictor.endpoint_name
        logger.info(f"Model served by endpoint: {endpoint}")
    else:
        endpoint = args.endpoint
    predictor = real_time_inference_job(endpoint)
    logger.info(f"Prediction returned: {predictor.predict(args.payload)}")



