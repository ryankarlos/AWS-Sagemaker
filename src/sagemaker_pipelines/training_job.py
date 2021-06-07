import boto3
from sagemaker import image_uris
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
from sagemaker.tuner import HyperparameterTuner
from sagemaker import HyperparameterTuningJobAnalytics
from utils.sagemaker_config import *
from functools import partial
import logging
import os
import sys
import argparse

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def initialise_estimator_job(framework, bucket_prefix="abalone"):    
    # set an output path where the trained model will be saved
    output_path = 's3://{}/{}/{}/output'.format(DEFAULT_BUCKET, bucket_prefix, framework)

    # this line automatically looks for the XGBoost image URI and builds an XGBoost container.
    # specify the repo_version depending on your preference.
    container = image_uris.retrieve(framework, REGION, "1.2-2")

    # construct a SageMaker estimator that calls the xgboost-container
    estimator = Estimator(image_uri=container, 
                          role=ROLE, 
                          instance_count=INSTANCE_CONFIG['training_instance_count'], 
                          instance_type=INSTANCE_CONFIG['training_instance_type'], 
                          volume_size=INSTANCE_CONFIG['training_volume_size'], 
                          output_path=output_path)

    return estimator

    
def estimator_fit_job(func, ts, hp_ranges):
    train_bucket = f"s3://{DEFAULT_BUCKET}/sagemaker-scikit-learn-{ts}/output/train_data/train.csv"
    val_bucket = f"s3://{DEFAULT_BUCKET}/sagemaker-scikit-learn-{ts}/output/val_data/validation.csv"
    train_input = TrainingInput(train_bucket, content_type="text/csv")
    validation_input = TrainingInput(val_bucket, content_type="text/csv")
    estimator = func()
    # need to define these hp first if not specified in hp tuning ranges, otherwirse job will throw an error 
    estimator.set_hyperparameters(num_round=100,
                                 objective="reg:squarederror")
    
    tuner = HyperparameterTuner(estimator=estimator,
                                objective_metric_name="validation:rmse",
                                objective_type='Minimize',
                                hyperparameter_ranges=hp_ranges,
                                max_jobs=2, max_parallel_jobs=2)
    
    tuner.fit({'train': train_input, 'validation': validation_input}, logs=True)

    return tuner


def hyperparameter_metrics(tuner):
    tuner_name = tuner.latest_tuning_job.job_name
    tuner_metrics_df =HyperparameterTuningJobAnalytics(tuner_name).dataframe()
    return tuner_metrics_df.sort_values(['FinalObjectiveValue'], ascending=False)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-val-bucket-ts', type=str, default="2021-06-06-20-19-30-801")
    parser.add_argument('--framework', type=str, default="xgboost")
    args = parser.parse_args()
    framework=args.framework
    xgb_est = partial(initialise_estimator_job, framework=framework)
    hp_ranges = hp_config[framework]["hyperparameter_ranges"]
    print(hp_ranges)
    tuner = estimator_fit_job(xgb_est, args.train_val_bucket_ts, hp_ranges)
    metrics_df = hyperparameter_metrics(tuner)
    print(metrics_df)
