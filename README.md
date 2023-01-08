## ML training and deployment in Sagemaker

A number of example experiments based on existing Sagemaker and Kaggle sample datasets to
try different things on Sagemaker.

### Setting up resources via terraform

Download the required plugins/providers

```bash
terraform init
```

Then check the .tf files pass all validation checks and check plan to see what resources will
be created

```bash
terraform validate
terraform plan
```

Now create the resources

```bash
terraform apply
```

### Cleanup resources

to delete the endpoint
https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-delete-resources.html

```bash
aws sagemaker delete-endpoint --endpoint-name <endpoint-name>
```

To delete the model resource for creating an endpoint

```bash
aws sagemaker list-models --query 'Models[0].ModelName' | xargs -I {} aws sagemaker delete-model --model-name {}
```

Delete all the other resources e.g. bucket, roles, policies and notebook instance via IAC

```
terraform destroy
```
