from sagemaker.tuner import IntegerParameter, ContinuousParameter
from utils.aws_helper import get_sagemaker_session


DEFAULT_BUCKET, ROLE, REGION, SAGEMAKER_SESSION = get_sagemaker_session()


IMAGE_METADATA = {"framework":"xgboost", 
 "region":"us-east-1", 
 "version":"1.2-2"}


INSTANCE_CONFIG = {"processing_instance_count":1, 
 "processing_instance_type":"ml.m5.xlarge", 
 "training_instance_count":1, 
 "training_instance_type":'g4ad.4xlarge',
 "training_volume_size":5,
 "inference_instance_count":1,
 "inference_instance_type":'ml.t2.medium',
 "batch_transform_instance_type":'ml.m4.xlarge'
}


hp_config ={
    "xgboost":{
            "hyperparameter_ranges": {
                "max_depth":  IntegerParameter(3, 10, scaling_type="Auto"),
                "eta": ContinuousParameter(0, 1, scaling_type="Auto"),
                "alpha":  ContinuousParameter(0, 2, scaling_type="Auto"),
                "min_child_weight":ContinuousParameter(1,10, scaling_type="Auto"),
                "subsample":ContinuousParameter(0.2, 0.9, scaling_type="Auto"),
              }
          }
    }