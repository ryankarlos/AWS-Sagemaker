import boto3
from sagemaker.serializers import CSVSerializer
from sagemaker import image_uris
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.estimator import Estimator
from sagemaker.predictor import json_serializer, csv_serializer, json_deserializer, RealTimePredictor
from sagemaker.content_types import CONTENT_TYPE_CSV, CONTENT_TYPE_JSON
from utils.pipeline_params import training_instance_type, role, region, default_bucket, sagemaker_session
from src.abalone.constants import HP
from functools import partial

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def processor_feat_eng_job(script="preprocessing.py", *args):
    sklearn_processor = SKLearnProcessor(
        role=role, instance_type=training_instance_type, instance_count=1
    )

    sklearn_processor.run(
        code=script,
        inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
        outputs=[
            ProcessingOutput(output_name="train_data", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="test_data", source="/opt/ml/processing/test"),
        ],
        arguments=args,
    )
    
    preprocessing_job_description = sklearn_processor.jobs[-1].describe()

    output_config = preprocessing_job_description["ProcessingOutputConfig"]
    for output in output_config["Outputs"]:
        if output["OutputName"] == "train_data":
            train_bucket = output["S3Output"]["S3Uri"]
        if output["OutputName"] == "test_data":
            test_bucket = output["S3Output"]["S3Uri"]

    logger.info(f"S3 paths of preprocessing job outputs - train dataset: {train_bucket}, test_dataset:{test_bucket}")
    return sklearn_processor, train_bucket, test_bucket

    
def initialise_estimator_job(framework, bucket_prefix="abalone"):    
    # set an output path where the trained model will be saved
    output_path = 's3://{}/{}/{}/output'.format(default_bucket, bucket_prefix, 'xgb-built-in')

    # this line automatically looks for the XGBoost image URI and builds an XGBoost container.
    # specify the repo_version depending on your preference.
    container = image_uris.retrieve("xgboost", region, "1.2-2")

    # construct a SageMaker estimator that calls the xgboost-container
    estimator = Estimator(image_uri=container, 
                          hyperparameters=HP, 
                          role=role, 
                          instance_count=1, 
                          instance_type='ml.m5.2xlarge', 
                          volume_size=5, 
                          output_path=output_path)

    return estimator

    
def estimator_fit_job(func,content_type = "libsvm",  bucket_prefix="abalone"):
    # define the data type and paths to the training and validation datasets
    train_input = TrainingInput("s3://{}/{}/{}/".format(default_bucket, bucket_prefix, 'train'), content_type=content_type)
    validation_input = TrainingInput("s3://{}/{}/{}/".format(default_bucket, bucket_prefix, 'validation'),
                                     content_type=content_type)
    # execute the XGBoost training job
    estimator = func()
    estimator.fit({'train': train_input, 'validation': validation_input}, logs=True)

    return estimator


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
        content_type=CONTENT_TYPE_CSV,
        accept=CONTENT_TYPE_JSON)
    
    return predictor
    
    
def batch_inference_job(estimator, bucket_prefix="abalone"):

    # The location of the test dataset
    batch_input = 's3://{}/{}/test'.format(default_bucket, bucket_prefix)

    # The location to store the results of the batch transform job
    batch_output = 's3://{}/{}/batch-prediction'.format(default_bucket, bucket_prefix)
    
    logger.info(f"Starting batch inference job,  input path: {batch_input}, ouput path:{batch_output}")
    
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
    processor_feat_eng_job(script="preprocessing.py", *args)
    xgb_est = partial(initialise_estimator_job, framework="xgboost")
    xgb_fitted = estimator_fit_job(xgb_est)
    predictor = deploy_model(xgb_fitted)
    endpoint = predictor.endpoint_name
    logger.info(f"Model served by endpoint: {endpoint}")
    predictor = real_time_inference_job(endpoint_name)
    payload = 'M, 0.44, 0.365, 0.125, 0.516, 0.2155, 0.114, 0.155'
    logger.info(f"Prediction returned for payload {payload}: {predictor.predict(payload)}")



