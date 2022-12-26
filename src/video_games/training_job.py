import sagemaker
from sagemaker import image_uris
from sagemaker.serializers import CSVSerializer
from sagemaker.inputs import TrainingInput
from sagemaker_config import S3_BUCKET, S3_PREFIX_VIDEOGAMES, VIDEOGAME_ENDPOINT_NAME


def create_xgboost_estimator():
    # initialize hyperparameters
    hyperparameters = {
            "max_depth":"3",
            "eta":"0.1",
            "eval_metric": "auc",
            "scale_pos_weight": "2.0",
            "subsample":"0.5",
            "objective":"binary:logistic",
            "num_round":"50"}
    output_path = 's3://{}/{}/output'.format(S3_BUCKET, S3_PREFIX_VIDEOGAMES)
    xgboost_container = sagemaker.image_uris.retrieve("xgboost", "us-east-1", "1.5-1")
    # construct a SageMaker estimator that calls the xgboost-container
    estimator = sagemaker.estimator.Estimator(image_uri=xgboost_container,
                                              hyperparameters=hyperparameters,
                                              role=sagemaker.get_execution_role(),
                                              instance_count=1,
                                              instance_type='ml.m5.2xlarge',
                                              volume_size=5,  # 5 GB
                                              output_path=output_path)

    return estimator


def train_estimator(estimator):
    # define the data type and paths to the training and validation datasets
    content_type = "csv"
    train_input = TrainingInput("s3://{}/{}/{}/".format(S3_BUCKET, S3_PREFIX_VIDEOGAMES, 'train'), content_type=content_type)
    validation_input = TrainingInput("s3://{}/{}/{}/".format(S3_BUCKET, S3_PREFIX_VIDEOGAMES, 'validation'), content_type=content_type)
    # execute the XGBoost training job
    estimator.fit({'train': train_input, 'validation': validation_input})


if __name__ == "__main__":
    estimator = create_xgboost_estimator()
    train_estimator(estimator)

    serializer = CSVSerializer()
    predictor = estimator.deploy(
        initial_instance_count=1,
        endpoint_name=VIDEOGAME_ENDPOINT_NAME,
        instance_type="ml.m5.xlarge",
        serializer=serializer,
    )
