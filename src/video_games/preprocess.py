from sklearn.datasets import dump_svmlight_file
import boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

bucket = 'sagemaker-artifacts'
prefix = 'videogames_xgboost'


def download_data_from_s3():
    raw_data_filename = 'Video_Games_Sales_as_at_22_Dec_2016.csv'
    local_dest_path = "'data/video_games/raw_data.csv'"
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
                                                      [int(0.7 * len(model_data)), int(0.9 * len(model_data))])
    return train_data, validation_data, test_data


def convert_data_to_libsvm(train_data, validation_data, test_data):
    dump_svmlight_file(X=train_data.drop(['y_no', 'y_yes'], axis=1), y=train_data['y_yes'], f='data/video_games/train.libsvm')
    dump_svmlight_file(X=validation_data.drop(['y_no', 'y_yes'], axis=1), y=validation_data['y_yes'], f='data/video_games/validation.libsvm')
    dump_svmlight_file(X=test_data.drop(['y_no', 'y_yes'], axis=1), y=test_data['y_yes'], f='data/video_games/test.libsvm')


def upload_model_data_to_s3():
    boto3.Session().resource('s3').Bucket(bucket).Object(prefix + '/train/train.libsvm').upload_file('data/video_games/train.libsvm')
    boto3.Session().resource('s3').Bucket(bucket).Object(prefix + '/validation/validation.libsvm').upload_file('data/video_games/validation.libsvm')


if __name__ == "__main__":
    df = download_data_from_s3()
    visualise_target_imbalance(df)
    model_data = preprocess_data(df)
    train_data, validation_data, test_data = train_test_split(model_data)
    convert_data_to_libsvm(train_data, validation_data, test_data)
    upload_model_data_to_s3()
