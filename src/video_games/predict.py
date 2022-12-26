import sys
import boto3
import pandas as pd
import numpy as np

endpoint_name = 'videogames-xgboost'
runtime = boto3.client('runtime.sagemaker')


def do_predict(data, endpoint_name, content_type):
    payload = '\n'.join(data)
    response = runtime.invoke_endpoint(EndpointName=endpoint_name,
                                       ContentType=content_type,
                                       Body=payload)
    result = response['Body'].read()
    result = result.decode("utf-8")
    result = result.split(',')
    preds = [float((num)) for num in result]
    preds = [round(num) for num in preds]
    return preds


def batch_predict(data, batch_size, endpoint_name, content_type):
    items = len(data)
    arrs = []

    for offset in range(0, items, batch_size):
        if offset + batch_size < items:
            results = do_predict(data[offset:(offset + batch_size)], endpoint_name, content_type)
            arrs.extend(results)
        else:
            arrs.extend(do_predict(data[offset:items], endpoint_name, content_type))
        sys.stdout.write('.')
    return (arrs)


if __name__ == "__main__":
    with open('data/video_games/test.libsvm', 'r') as f:
        payload = f.read().strip()

    labels = [int(line.split(' ')[0]) for line in payload.split('\n')]
    test_data = [line for line in payload.split('\n')]
    preds = batch_predict(test_data, 100, endpoint_name, 'text/x-libsvm')
    print('\nerror rate=%f' % (sum(1 for i in range(len(preds)) if preds[i] != labels[i]) / float(len(preds))))
    print(pd.crosstab(index=np.array(labels), columns=np.array(preds)))


