import boto3
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from utils.sagemaker_config import *
import logging
import os
import sys



logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def processor_feat_eng_job(bucket_input_data, script):
    sklearn_processor=SKLearnProcessor(role=ROLE
                                       ,framework_version="0.20.0"
                                       ,instance_type=INSTANCE_CONFIG['processing_instance_type']
                                       ,instance_count=INSTANCE_CONFIG['processing_instance_count']
                                      )
    sklearn_processor.run(
        code=script,
        inputs=[ProcessingInput(source=bucket_input_data, destination="/opt/ml/processing/input")],
        outputs=[
            ProcessingOutput(output_name="train_data", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="test_data", source="/opt/ml/processing/test"),
            ProcessingOutput(output_name="val_data", source="/opt/ml/processing/validation")
        ]
    )
    
    return sklearn_processor



if __name__ == "__main__":
    bucket_input_data = 's3://{}/abalone/abalone-dataset.csv'.format(DEFAULT_BUCKET)
    sklearn_processor = processor_feat_eng_job(bucket_input_data, script="src/sagemaker_pipelines/features_script.py")
   
