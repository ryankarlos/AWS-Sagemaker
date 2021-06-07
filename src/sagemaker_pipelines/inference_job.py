import boto3
from sagemaker.serializers import CSVSerializer
from sagemaker.predictor import (json_serializer, csv_serializer, 
                                 json_deserializer, RealTimePredictor)
from utils.sagemaker_config import *
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
    initial_instance_count=1,
    instance_type='ml.t2.medium',
    serializer=CSVSerializer()
    )
    
    return predictor


def real_time_inference_job(endpoint_name):
    predictor = RealTimePredictor(
        endpoint=endpoint_name,
        sagemaker_session=sagemaker_session,
        serializer=csv_serializer,
        content_type="text/csv",
        accept="application/json")
    
    return predictor
    
    
def batch_inference_job(estimator, bucket_prefix="abalone"):

    # The location of the test dataset
    batch_input = 's3://{}/{}/test'.format(DEFAULT_BUCKET, bucket_prefix)

    # The location to store the results of the batch transform job
    batch_output = 's3://{}/{}/batch-prediction'.format(DEFAULT_BUCKET, bucket_prefix)
    
    logger.info(f"Starting batch inference job-input path:{batch_input}, output path:{batch_output}")
    
    transformer = estimator.transformer(
    instance_count=1, 
    instance_type='ml.m4.xlarge', 
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
    args = parser.parse_args()
    
    if args.endpoint is not None:
        predictor = deploy_model(xgb_fitted)
        endpoint = predictor.endpoint_name
        logger.info(f"Model served by endpoint: {endpoint}")
    else:
        endpoint = args.endpoint
    predictor = real_time_inference_job(endpoint)
    payload = 'M, 0.44, 0.365, 0.125, 0.516, 0.2155, 0.114, 0.155'
    logger.info(f"Prediction returned for payload {payload}: {predictor.predict(payload)}")



