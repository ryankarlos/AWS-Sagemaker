from sagemaker.tuner import IntegerParameter, ContinuousParameter
from utils.aws_helper import get_sagemaker_session



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

# Parameters for pipeline execution
processing_instance_count = ParameterInteger(
    name="ProcessingInstanceCount", default_value=1
)
processing_instance_type = ParameterString(
    name="ProcessingInstanceType", default_value="ml.m5.xlarge"
)
training_instance_type = ParameterString(
    name="TrainingInstanceType", default_value="ml.m5.xlarge"
)
# Parameters for pipeline execution
training_instance_count = ParameterInteger(
    name="TrainingInstanceCount", default_value=1
)
model_approval_status = ParameterString(
    name="ModelApprovalStatus",
    default_value="PendingManualApproval",  # ModelApprovalStatus can be set to a default of "Approved" if you don't want manual approval.
)




hp_config_abalone ={
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