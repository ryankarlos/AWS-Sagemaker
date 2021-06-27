import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import (
    ConditionGreaterThanOrEqualTo,
)
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet,
)
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.step_collections import RegisterModel
from utils.aws_helper import get_session, get_image_uri
from utils.sagemaker_config import processing_instance_count, 
    processing_instance_type, 
    training_instance_type,
    training_instance_count,
    model_approval status 


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def processing_job(processing_instance_type, processing_instance_count, sagemaker_session, role):
    
    # Processing step for feature engineering
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-abalone-preprocess", 
        sagemaker_session=sagemaker_session,
        role=role,
    )
    
    step_process = ProcessingStep(
        name="AbaloneProcess",  
        processor=sklearn_processor,
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(
                output_name="validation", source="/opt/ml/processing/validation"
            ),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        job_arguments=["--input-data", input_data],
    )
    
    return step_process


def training_job(image_uri, 
                 training_instance_type, 
                 training_instance_count, 
                 model_path,
                 sagemaker_session, 
                 role):
    
     xgb_train = Estimator(
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=training_instance_count,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}/Abalone-train",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    xgb_train.set_hyperparameters(
        objective="binary:logistic",
        num_round=50,
        max_depth=5,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.7,
        silent=0,
    )
    step_train = TrainingStep(
        name="CustomerChurnTrain",
        estimator=xgb_train,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )
    
    return step_train

    

def get_pipeline(
    region,
    default_bucket='sagemaker-us-east-1-376337229415',
    model_package_group_name="AbaloneExample", 
    pipeline_name="AbaloneExample",  
    base_job_prefix="abalone", 
):
    """Gets a SageMaker ML Pipeline instance working with on CustomerChurn data.
    Args:
        region: AWS region to create and run the pipeline.
        default_bucket: the bucket to use for storing the artifacts
    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    role = sagemaker.session.get_execution_role(sagemaker_session)
        
    input_data = ParameterString(
    name="InputDataUrl", 
    default_value='s3://{}/abalone/abalone-dataset.csv'.format(default_bucket),  
        )
    
    step_process = processing_job(processing_instance_type, 
                                  processing_instance_count, 
                                  sagemaker_session, 
                                  role)
    # Training step for generating model artifacts
    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/Abalonetrain"
    
    image_uri = get_image_uri(framework="xgboost", 
                              region=region, 
                              version="1.0-1", 
                              py_version="py3", 
                              training_instance_type=training_instance_type)

    
    step_train = training_job(image_uri, 
                              training_instance_type, 
                              training_instance_count, 
                              model_path, 
                              sagemaker_session, role)
    # Pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            input_data,
        ],
        steps=[step_process, step_train],
        sagemaker_session=sagemaker_session,
    )
    return pipeline