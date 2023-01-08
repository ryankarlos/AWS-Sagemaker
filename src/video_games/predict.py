import boto3
import os
import numpy as np
from src.utils import get_repo_root
from .preprocess import get_data_from_s3
from sklearn.metrics import f1_score
import io
import pandas as pd
from sagemaker_config import (
    VIDEOGAME_ENDPOINT_NAME,
    delete_endpoint,
    S3_PREFIX_VIDEOGAMES,
)

basepath = get_repo_root(__file__)
data_dir = os.path.join(basepath, "data/video_games")
runtime = boto3.client("runtime.sagemaker")


def get_prediction(data, endpoint_name, content_type="text/csv"):
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name, ContentType=content_type, Body=data
    )
    return response["Body"].read().decode("utf-8").splitlines()


if __name__ == "__main__":
    key_test = f"{S3_PREFIX_VIDEOGAMES}/test/test.csv"
    test_df = get_data_from_s3(key_test)
    csv_file = io.StringIO()
    test_df.drop(["Y"], axis=1).to_csv(csv_file, sep=",", header=False, index=False)
    preds = get_prediction(csv_file.getvalue(), VIDEOGAME_ENDPOINT_NAME)
    test_df["predictions"] = np.where(pd.Series(preds).astype("float") > 0.5, 1, 0)
    f1_score(test_df["Y"], test_df["predictions"], average="weighted")
    delete_endpoint(*[VIDEOGAME_ENDPOINT_NAME])
