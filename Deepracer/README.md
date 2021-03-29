
The following training params and action space settings were used during the AWS March 2021 Qualifier in the open division-time trial on the Po-Chun Speedway circuit (68.683m)


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

