# Deep Racer reward function and log analysis

The following training params and action space settings were used during the AWS March 2021 Qualifier in the open division-time trial on the Po-Chun Speedway circuit (68.683m)

Being a causal competitor I managed to get a rank of 150 out of 1000 with just 8 hours of training ($30 Deep Racer credits)

## Action space


Speed: {0.93, 1.87, 2.8} m/s
Steering angle: {-30, -15, 0, 15, 30} degrees

## Hyperparameters

The model was first trained with batch sizs 32 and MSE loss type for 80 mins.
Then resumed training by using Huber with larger batch size.  


| Hyperparameters                                                | Value           |
|----------------------------------------------------------------|-----------------|
| Gradient Descent Batch Size                                    | 32 or 64              |
| Entropy                                                        | 0.01            |
| Discount Factor                                                | 0.999           |
| Loss Type                                                      | MSE/Huber           |
| Learning Rate                                                  | 0.001 or 0.001 |
| No# Experience Episodes between each policy-updating iteration | 20              |
| No# of Epochs                                                  | 10         |

## Stop conditions

- 60 or 80 mins

## Updating the environment for running notebook

create and activate new environment

```
python3 -m venv venv
source venv/bin/activate
```

install/upgrade packages from requirements.txt

```
pip install --upgrade -r requirements.txt
jupyter lab
```

## References and Credits

* Reward functions: https://docs.aws.amazon.com/deepracer/latest/developerguide/deepracer-reward-function-examples.html
* Log analysis:  Training log analysis from Sagemaker and Robomaker using the notebooks provided from the official Deep Racer repo updated by the community 
 https://github.com/aws-samples/aws-deepracer-workshops/tree/master/log-analysishttps://github.com/aws-deepracer-community/deepracer-analysis
