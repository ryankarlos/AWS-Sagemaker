#Download an open source TensorFlow Docker image
FROM tensorflow/tensorflow:latest-gpu-jupyter

# Install sagemaker-training toolkit that contains the common functionality necessary to create a container compatible with SageMaker AI and the Python SDK.
RUN pip3 install sagemaker-training

# Copies the code inside the container
COPY train.py /opt/ml/code/train.py
COPY train.py /opt/ml/code/process.py

# Defines process.py as script entrypoint
ENV SAGEMAKER_PROGRAM process.py