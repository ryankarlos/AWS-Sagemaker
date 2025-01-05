# This is the file that implements a flask server to do inferences. It's the file that you will modify
# to implement the prediction for your own algorithm.

from __future__ import print_function
import joblib
import flask
from flask import Flask, jsonify
import os
import request

MODEL_PATH = '/opt/ml/'
MODEL_NAME = '' 


def load_model(model_dir: str):
    """
    Load the model from the specified directory.
    """
    return joblib.load(os.path.join(model_dir, "model.joblib"))


# Load the model by reading the `SM_MODEL_DIR` environment variable
# which is passed to the container by SageMaker (usually /opt/ml/model).

            
# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.
class CatBoostRegressorService(object):

    @classmethod
    def get_model(cls):
        """Get the model object for this instance."""
        return load_model(os.environ["SM_MODEL_DIR"])

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them."""
        
        model = cls.get_model()
        return model.predict(input) 

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = CatBoostRegressorService.get_model() is not None  
    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    # Do the prediction
    body = request.json
    predictions = CatBoostRegressorService.predict(body).tolist() #predict() also loads the model
    print('predictions: ' + str(predictions[0]) + ', ' + str(predictions[1]))
    return {'predictions': predictions}