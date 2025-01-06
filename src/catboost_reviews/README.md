
## Instructions


Install the libraies in requirements.txt in local virtual env to run some of the scripts locally

```
cd src/catboost_reviews
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

### Testing the scirpts locally outside Sagemaker

Build docker train image and run container locally passing env vars for secrets


Generate temp creds for passing in as env vars to run container locally to access aws services
https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_temp_use-resources.html
make a note of access key id and secret access key.

```
aws sts assume-role --role-arn <role-arn> --role-session-name "RoleSession1" --profile IAM-user-name 
```

Build train docker images sm-catboost-train 

```bash
cd src/catboost_reviews
docker build -f Dockerfile.train -t sm-catboost-train .
```

Run the container passing the env vars and the cmd /opt/ml/code/process.py. The 
default entrypoint in dockefile is "python"

```
docker run --rm -it -e AWS_ACCESS_KEY_ID='<access-key-id>' -e AWS_SECRET_ACCESS_KEY='<secret-access-key>' -e AWS_SESSION_TOKEN='<session-token>' -e AWS_DEFAULT_REGION="us-east-1" sm-catboost-train /opt/ml/code/process.py
```

or pass in env variables from file as in format below. For example create a .env file 

```
AWS_ACCESS_KEY_ID=<access-key-id>
AWS_SECRET_ACCESS_KEY=<secret-access-key>
AWS_DEFAULT_REGION=us-east-1
AWS_SESSION_TOKEN=<session-token>
``
and then specify the path to the .env file as --env-file argument to docker run command

```
docker run -it  --env-file ./.env sm-catboost-train /opt/ml/code/process.py
```

Similarly for the train script pass in the path to the train script in the container "/opt/ml/code/train.py" and also train and test channel s3 dir path (these should be output as logs from the processing container run or check s3 for the data and get the s3 uri)

docker run -it  --env-file ./.env sm-catboost-train /opt/ml/code/train.py --train <s3-train-dir-url> --test <s3-val-dir-url>

### Running jobs in Sagemaker

First test container with training job in local mode

```
sh build_and_push.sh Dockerfile.train sm-catboost-train
```


Then start process or traing job (enter value for job arg accordingly) and set instance_type to local.


```
python sm_job_exec.py --role_arn <sagemaker-role-arn> --job <process/train> --instance_type=local
```


then start a remote sagemaker process or training job (dont pass in instance_type and it will default to "ml.m5.xlarge")

 python training_job_exec.py --role_arn <sagemaker-role-arn>  --job <process/train>