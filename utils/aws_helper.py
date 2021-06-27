import boto3
import sagemaker
import sagemaker.session


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.
    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts
    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

    
def upload_dataset_to_s3(default_bucket, local_path):
    base_uri = f"s3://{default_bucket}/abalone"
    input_data_uri = sagemaker.s3.S3Uploader.upload(
        local_path=local_path, 
        desired_s3_uri=base_uri,
    )
    print(input_data_uri)
    


def get_image_uri(framework, region, version, py_version, training_instance_type):
    """
    Retrieves docker image uri from registry. e.g framework "sklearn", region "us-east-1", version "0.23-1"
    """
    return sagemaker.image_uris.retrieve(framework, region, version, py_version, training_instance_type)

    return sagemaker.image_uris.retrieve(
        framework="xgboost",  # we are using the Sagemaker built in xgboost algorithm
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type=training_instance_type,
    )

    
def get_pre_trained_model_from_bucket(model_uri, role, framework, region, version):
    """
    Returns a sagemaker model from s3 uri which can be deployed to an endpoint
    """
    
    return sagemaker.model.Model(model_data=model_uri, image_uri=get_image_uri(framework, region, version), role=role)  
    
