### Hugging Face on Sagemaker

This section follows the steps in the tutorial on the Hugging Face docs to
run preprocessing, training and inference jobs.
https://huggingface.co/docs/sagemaker/train#run-training-on-amazon-sagemaker

### Sagemaker roles

If you have created Sagemaker roles previosuly, you can list
them out on the console, filtering the output for roles with sagemaker service
principal in assumed role policy.
In my case, I have roles created for SagemakerNotebook and Sagemaker Main Role

```
 aws iam list-roles --query 'Roles[?AssumeRolePolicyDocument.Statement[0].Principal.Service==`sagemaker.amazonaws.com`].Arn'
```

![](../../../screenshots/iam_roles.png)

To get more details on the attached policies for the SagemakerMainRole.


![](../../../screenshots/sagemaker-role-policies-attached.png)

![](../../../screenshots/get_role_details.png)


### Running the steps

Run the preprocess.py script to download and preprocess  (tokenise) the IMDB
dataset for training.
This also uploads the preprocessed dataset to the S3 BUCKET.

![](../../../screenshots/hugging-face-preprocessing.png)


The training_job.py script creates a Hugging Face Estimator to handle end-to-end SageMaker
training. This uses a finetuning script in the scripts/train.py which is taken from
https://github.com/huggingface/notebooks/blob/main/sagemaker/01_getting_started_pytorch/scripts/train.py

The training job starts by creating the instances for training, downloading the ECR hugging face image uri specified
and fetching the training and test datasets from S3 for training and evaluation of metrics.

We can fetch details of the training job from cli. We can filter by create date as below.

aws sagemaker list-training-jobs --query 'TrainingJobSummaries[?CreationTime>`2023-01-01`]'


![](../../../screenshots/list-training-jobs-sagemaker.png)


Once the training job is complete, the model should be persisted to S3. Run the `predict.py` script to downaload
the fine-tuned model from S3 and deploy to a Sagemaker endpoint by providing the name of the instance type and
number of instances to run the endpoint.

The script also sends an example request to the endpoint for generating a prediction.
Finally, the endpoint is deleted by calling `predictor.delete_endpoint(<Endpoint-Name>)`
