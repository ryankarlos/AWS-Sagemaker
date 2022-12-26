import boto3
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils import get_repo_root
from sagemaker_config import S3_BUCKET, S3_PREFIX_VIDEOGAMES


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
    data['y'] = data['y'].apply(lambda y: 'yes' if y == True else 'no')
    model_data = pd.get_dummies(data)
    return model_data


def train_test_split(df):
    model_data = df.copy()
    train_data, validation_data, test_data = np.split(model_data.sample(frac=1, random_state=1729),
                                                      [int(0.8 * len(model_data)), int(0.995 * len(model_data))])
    return train_data, validation_data, test_data


def upload_model_data_to_s3():
    boto3.Session().resource("s3").Bucket(S3_BUCKET).Object(
        os.path.join(S3_PREFIX_VIDEOGAMES, "train/train.csv")
    ).upload_file("train.csv")
    boto3.Session().resource("s3").Bucket(S3_BUCKET).Object(
        os.path.join(S3_PREFIX_VIDEOGAMES, "validation/validation.csv")
    ).upload_file("validation.csv")


if __name__ == "__main__":
    df = download_data_from_s3()
    print(df)
    visualise_target_imbalance(df)
    model_data = preprocess_data(df)
    train_data, validation_data, test_data = train_test_split(model_data)
    #upload_model_data_to_s3()
