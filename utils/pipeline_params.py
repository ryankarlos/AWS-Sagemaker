from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from utils.session_initialisation import get_sagemaker_session


default_bucket, role, region = get_sagemaker_session()

processing_instance_count = ParameterInteger(
    name="ProcessingInstanceCount",
    default_value=1
)
processing_instance_type = ParameterString(
    name="ProcessingInstanceType",
    default_value="ml.m5.xlarge"
)
training_instance_type = ParameterString(
    name="TrainingInstanceType",
    default_value="ml.m5.xlarge"
)


