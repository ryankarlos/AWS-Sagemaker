from sagemaker.estimator import Estimator
import boto3
import sagemaker
import argparse 
import sys


def start_training_job(args):
    region = boto3.Session().region_name
    sess = sagemaker.session.Session()
    bucket = sess.default_bucket()
    account_id = boto3.client('sts').get_caller_identity().get('Account')

    role = args.role_arn
    ecr_repository_name = args.ecr_repository_name
    train_prefix = args.s3_train_prefix
    val_prefix = args.s3_val_prefix
    instance_count = args.instance_count
    instance_type = args.instance_type

    output_path = 's3://' + bucket + '/' + 'training_jobs'
    print(output_path)

    if instance_type == "local":
        container_image_uri = ecr_repository_name
    else:
        container_image_uri = f'{account_id}.dkr.ecr.{region}.amazonaws.com/{ecr_repository_name}:latest'
        print('ECR container ARN: {}'.format(container_image_uri))
    print(container_image_uri)

    estimator = Estimator(image_uri=container_image_uri,
                    role=role,
                    instance_count=instance_count,
                    instance_type=instance_type,
                    output_path=output_path,
                    hyperparameters={
                            'iterations': args.iterations,
                            'lr':args.lr,
                            'depth':args.depth
                            })
    print(f's3://{bucket}/{train_prefix}')
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
                default ="val/val.csv"
                )
    parser.add_argument("--instance_count",
            type=int,
            default=1
            )
    parser.add_argument("--instance_type",
        type=str,
        default="ml.m5.xlarge"
        )
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--depth', type=int, default=2)

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    start_training_job(args)
    sys.exit(0)