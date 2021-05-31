import boto3
import sagemaker
import sagemaker.session
from utils.config import local_path, abalone_dataset_default_path


def get_sagemaker_session():
    region = boto3.Session().region_name
    sagemaker_session = sagemaker.session.Session()
    role = sagemaker.get_execution_role()
    default_bucket = sagemaker_session.default_bucket()
    print(default_bucket)
    return default_bucket, role, region


def download_sample_dataset_to_local(region, local_path):
    s3 = boto3.resource("s3")
    s3.Bucket(f"sagemaker-servicecatalog-seedcode-{region}").download_file(
        abalone_dataset_default_path,
        local_path
    )

    
def upload_dataset_to_s3(default_bucket, local_path):
    base_uri = f"s3://{default_bucket}/abalone"
    input_data_uri = sagemaker.s3.S3Uploader.upload(
        local_path=local_path, 
        desired_s3_uri=base_uri,
    )
    print(input_data_uri)
    
    
if __name__ == "__main__":
    default_bucket, role, region = get_sagemaker_session()
    download_sample_dataset_to_local(region, local_path)
    upload_dataset_to_s3(default_bucket, local_path)