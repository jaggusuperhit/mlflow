# MLflow

# Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the app.py

# How to run?

### STEPS:

Clone the repository:

'''bash
https://github.com/jaggusuperhit/MLflow.git
'''
...

### STEP 01- Create a conda environment after opening the repository

'''bash
conda create -n mlpos python=3.8 -y
'''
'''bash
conda activate mlops
'''

### STEP 02- install the required packages

'''bash
pip install -r requirements.txt
'''

'''bash

# Finally , run the app

python app.py
'''
Now , let's go through the code and see how it works.
'''bash
Open up you local host and port

### MLflow

[Documentation](https://www.mlflow.org/docs/2.1.1/index.html)

### cmd

- mlflow ui

### dagshub

[dagshub](https://dagshub.com/)

import dagshub
dagshub.init(repo_owner='jaggusuperhit', repo_name='MLflow', mlflow=True)

import mlflow
with mlflow.start_run():
mlflow.log_param('parameter name', 'value')
mlflow.log_metric('metric name', 1)

Run this to export as env variables:

'''bash

export MLFLOW_TRACKING_URI=https://dagshub.com/jaggusuperhit/MLflow
export MLFLOW_TRACKING_USERNAME=jaggusuperhit

import dagshub
dagshub.init(repo_owner='jaggusuperhit', repo_name='MLflow', mlflow=True)
