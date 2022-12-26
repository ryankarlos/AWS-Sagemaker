import boto3
import json
import os
from src.utils import get_repo_root
from sagemaker_config import VIDEOGAME_ENDPOINT_NAME

basepath = get_repo_root(__file__)
data_dir = os.path.join(basepath, 'data/video_games')

runtime = boto3.client('runtime.sagemaker')


def get_prediction(data, endpoint_name, content_type="text/csv"):
    response = runtime.invoke_endpoint(EndpointName=endpoint_name,
                                       ContentType=content_type,
                                       Body=data)
    result = json.loads(response['Body'].read().decode())
    return result


if __name__ == "__main__":
    with open(os.path.join(data_dir, 'raw_data.csv')) as f:
        payload = f.read()
    preds = get_prediction(payload, VIDEOGAME_ENDPOINT_NAME)
    print(preds)
