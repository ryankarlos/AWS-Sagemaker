from sagemaker.estimator import Estimator
import boto3
import sagemaker
import argparse 
import sys

region = boto3.Session().region_name
sess = sagemaker.session.Session()
bucket = sess.default_bucket()


def start_remote_training_job(args):
    container_image_uri = args.image_uri
    role = args.role_arn
    ecr_repository_name = args.ecr_repository_name
    train_prefix = args.s3_train_prefix
    val_prefix = args.s3_val_prefix

    output_path = 's3://' + bucket + '/' + 'training_jobs'
    print(output_path)

    account_id = role.split(':')[4]

    container_image_uri = f'{account_id}.dkr.ecr.{region}.amazonaws.com/{ecr_repository_name}:latest'
    print('ECR container ARN: {}'.format(container_image_uri))

    estimator = Estimator(image_uri=container_image_uri,
                    role=role,
                    instance_count=1,
                    instance_type='ml.m5.xlarge',
                    output_path=output_path,
                    hyperparameters={
                                    'sagemaker_submit_directory': submit_dir
                                    })
   
    estimator.fit({'train': f's3://{bucket}/{train_prefix}',
                        'test': f's3://{bucket}/{val_prefix}'})
    

def get_parser():
    parser = argparse.ArgumentParser(
        description="Script to process data for housing dataset."
    )
    parser.add_argument("--role_arn",
                        type=str
                        )
    
    parser.add_argument("--ecr_repository_name",
                    type=str,
                    default="sm-catboost-train"
                    )
    parser.add_argument("--s3_train_prefix",
                    type=str,
                    default ="train/train.csv"
                    )
    parser.add_argument("--s3_val_prefix",
                type=str,
                default ="validation/validation.csv"
                )

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    # start_remote_training_job(args)
    sys.exit(0)