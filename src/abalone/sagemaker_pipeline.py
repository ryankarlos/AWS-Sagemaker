from sagemaker.estimator import Estimator
from utils.pipeline_params import training_instance_type, role, region, default_bucket
from sagemaker.processing import ProcessingInput, ProcessingOutput
from src.abalone.constants import HP
import boto3
from sagemaker.serializers import CSVSerializer
from sagemaker import image_uris
from sagemaker.inputs import TrainingInput

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

    
def initialise_estimator_job(framework="xgboost", bucket_prefix="abalone"):    
    image_uri = sagemaker.image_uris.retrieve(
        framework=framework,
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type=training_instance_type,
    )
    train = Estimator(
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=model_path,
        role=role,
    )
    train.set_hyperparameters(**HP)
    
    # set an output path where the trained model will be saved
    output_path = 's3://{}/{}/{}/output'.format(default_bucket, bucket_prefix, 'xgb-built-in')

    # this line automatically looks for the XGBoost image URI and builds an XGBoost container.
    # specify the repo_version depending on your preference.
    container = sagemaker.image_uris.retrieve("xgboost", region, "1.2-2")

    # construct a SageMaker estimator that calls the xgboost-container
    estimator = sagemaker.estimator.Estimator(image_uri=container, 
                                              hyperparameters=HP,
                                              role=role,
                                              instance_count=1, 
                                              instance_type='ml.m5.2xlarge', 
                                              volume_size=5, # 5 GB 
                                              output_path=output_path)
    
    return estimator

    
def estimator_fit_job(initialise_estimator_job,content_type = "libsvm",  bucket_prefix="abalone"):
    # define the data type and paths to the training and validation datasets
    train_input = TrainingInput("s3://{}/{}/{}/".format(default_bucket, bucket_prefix, 'train'), content_type=content_type)
    validation_input = TrainingInput("s3://{}/{}/{}/".format(default_bucket, bucket_prefix, 'validation'),
                                     content_type=content_type)
    # execute the XGBoost training job
    estimator = initialise_estimator_job()
    estimator.fit({'train': train_input, 'validation': validation_input})

    return estimator


def deploy_model_and_create_endpoint(estimator):
    
    predictor=estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium',
    serializer=CSVSerializer()
    )
    
    logger.info(f"Model deployed with endpoint name: {predictor.endpoint_name}")
    
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
    processor_feat_eng_job(script="preprocessing.py", *args):


