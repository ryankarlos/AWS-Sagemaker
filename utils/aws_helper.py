import boto3
import sagemaker
import sagemaker.session


def get_sagemaker_session():
    region = boto3.Session().region_name
    sagemaker_session = sagemaker.session.Session()
    role = sagemaker.get_execution_role()
    default_bucket = sagemaker_session.default_bucket()
    return default_bucket, role, region, sagemaker_session

    
def upload_dataset_to_s3(default_bucket, local_path):
    base_uri = f"s3://{default_bucket}/abalone"
    input_data_uri = sagemaker.s3.S3Uploader.upload(
        local_path=local_path, 
        desired_s3_uri=base_uri,
    )
    print(input_data_uri)
    

