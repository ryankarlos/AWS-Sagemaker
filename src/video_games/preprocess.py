import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.utils import get_repo_root
from sagemaker_config import S3_BUCKET, S3_PREFIX_VIDEOGAMES
import logging
import boto3
from botocore.exceptions import ClientError
import os
import io


basepath = get_repo_root(__file__)
data_dir = os.path.join(basepath, 'data/video_games')


def download_data_from_s3():
    raw_data_filename = 'Video_Games_Sales_as_at_22_Dec_2016.csv'
    local_dest_path = os.path.join(data_dir, 'raw_data.csv')
    data_bucket = 'sagemaker-workshop-pdx'
    s3 = boto3.resource('s3')
    s3.Bucket(data_bucket).download_file(raw_data_filename, local_dest_path)
    df = pd.read_csv(local_dest_path)
    return df


def visualise_target_imbalance(df):
    data = df.copy()
    viz = data.filter(['User_Score', 'Critic_Score', 'Global_Sales'], axis=1)
    viz['User_Score'] = pd.Series(viz['User_Score'].apply(pd.to_numeric, errors='coerce'))
    viz['User_Score'] = viz['User_Score'].mask(np.isnan(viz["User_Score"]), viz['Critic_Score'] / 10.0)
    viz.plot(kind='scatter', logx=True, logy=True, x='Critic_Score', y='Global_Sales')
    viz.plot(kind='scatter', logx=True, logy=True, x='User_Score', y='Global_Sales')
    plt.show()


def preprocess_data(df):
    data = df.copy()
    data['y'] = (data['Global_Sales'] > 1)
    data = data.drop(
        ['Name', 'Year_of_Release', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales', 'Critic_Count',
         'User_Count', 'Developer'], axis=1)
    data = data.dropna()
    data['User_Score'] = data['User_Score'].apply(pd.to_numeric, errors='coerce')
    data['User_Score'] = data['User_Score'].mask(np.isnan(data["User_Score"]), data['Critic_Score'] / 10.0)
    target = data['y'].replace({True: 1, False: 0})
    data = data.drop(['y'], axis=1)
    features = pd.get_dummies(data)
    df = pd.concat([target, features], axis=1)
    return df.rename(columns={'y': 'Y'})


def train_test_split(df):
    model_data = df.copy()
    train_data, validation_data, test_data = np.split(model_data.sample(frac=1, random_state=1729),
                                                      [int(0.75 * len(model_data)), int(0.95 * len(model_data))])
    train_data.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    validation_data.to_csv(os.path.join(data_dir, "validation.csv"), index=False)
    test_data.to_csv(os.path
                     .join(data_dir, "test.csv"), index=False)


def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def get_data_from_s3(key):
    client = boto3.client('s3')
    response = client.get_object(
        Bucket=S3_BUCKET,
        Key=key
    )

    data = response['Body'].read().decode('utf-8')
    csv_file = io.StringIO()
    return pd.read_csv(io.StringIO(data))

def upload_model_data_to_s3():
    upload_file(os.path.join(data_dir, "train.csv"), S3_BUCKET, f"{S3_PREFIX_VIDEOGAMES}/train/train.csv")
    upload_file(os.path.join(data_dir, "validation.csv"), S3_BUCKET, f"{S3_PREFIX_VIDEOGAMES}/validation/validation.csv")
    upload_file(os.path.join(data_dir, "test.csv"), S3_BUCKET, f"{S3_PREFIX_VIDEOGAMES}/test/test.csv")


if __name__ == "__main__":
    df = download_data_from_s3()
    print(df)
    visualise_target_imbalance(df)
    model_data = preprocess_data(df)
    train_test_split(model_data)
    upload_model_data_to_s3()
