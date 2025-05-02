import dagshub
dagshub.init(repo_owner='jaggusuperhit', repo_name='MLflow', mlflow=True)

import mlflow
import os

# Set MLflow tracking URI
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/jaggusuperhit/MLflow.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'jaggusuperhit'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'your_password_here'  # Replace with your actual password

# Example MLflow logging
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)