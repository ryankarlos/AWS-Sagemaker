# Machine Learning projects/services I have played around with in AWS

Will be adding more to the list

*Sagemaker
*Deepracer
*Personalise 


# Setting up AMI (EC2) 

I normally find Sagemaker Notebook instance quote cool as easy to launch a set the kernel with preinstalled popular open-source deep learning frameworks, including TensorFlow, Apache MXNet, PyTorch. 
However, sometimes its quite useful to setup EC2 instance with Deep Learning AMI with GPU if i need to do some work on the terminal and run pythin scripts to take advantage of GPU.
I found on this amazon blog post https://aws.amazon.com/blogs/machine-learning/get-started-with-deep-learning-using-the-aws-deep-learning-ami/

After creating instance and generating ssh key pair - need to give read permissions to private key on local machine (i normally store in ssh folder)
`chmod 400 deep-learning.pem`

The can ssh into instance from terminal, with port forwarding so all requests on local port are forwarded onto port on remote ec2 insatnce. 
Need to set <instance-name> on AWS GUI if it is blank - e.g. ubuntu. The <DNS> is public DNS which can be found in description tab.

`ssh -L localhost:8888:localhost:8888 -i ~/ssh/tf-cert.pem <instance-name>@<DNS>`

Can then launch jupyter notebook once inside the container and navigate to link in browser. Can also activate required deep learning environment
once inside container e..g 

`ubuntu@ip-172-31-2-142:~$ conda activate tensorflow_p36`

# Copy files from locally to AWS instance - can use SCP protocol from terminal
`scp -i deep-learning.pem <path-to-file> user@hostname:<file-path-to-be-copied-on-remote>`


