
## Instructions


### Testing the scirpts locally outside Sagemaker

Build docker train image and run container locally passing env vars for secrets


Generate temp creds for passing in as env vars to run container locally to access aws services
https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_temp_use-resources.html
make a note of access key id and secret access key.

We pass in duration seconds to determine how long the creds should be valid for. Here will 
set it to ve valid for 7200 secs (2 hrs)

```
aws sts assume-role --role-arn <role-arn> --role-session-name "RoleSession1" --profile IAM-user-name --duration-seconds 7200
```

Build train docker images sm-catboost-train 

```bash
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


Similarly for the train script pass in the path to the train script in the container "/opt/ml/code/train.py"

### Running remote jobs in Sagemaker

