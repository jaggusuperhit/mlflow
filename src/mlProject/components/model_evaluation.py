import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib
import mlflow
from pathlib import Path
from dotenv import load_dotenv
from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.utils.common import save_json
from mlProject.utils.dagshub_utils import setup_dagshub_auth
from mlProject import logger

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        # Evaluate the model
        predicted_qualities = model.predict(test_x)
        (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)

        # Save metrics locally
        scores = {"rmse": rmse, "mae": mae, "r2": r2}
        save_json(path=Path(self.config.metric_file_name), data=scores)

        # Load environment variables from .env file
        load_dotenv()

        try:
            # Get credentials from environment variables
            mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", self.config.mlflow_uri)
            mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
            mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")

            if not mlflow_uri or not mlflow_username or not mlflow_password:
                logger.warning("MLflow credentials not found in environment variables. Skipping MLflow logging.")
                return

            # Set environment variables for MLflow
            os.environ['MLFLOW_TRACKING_URI'] = mlflow_uri
            os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_username
            os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_password

            # Set MLflow tracking URI
            mlflow.set_tracking_uri(mlflow_uri)
            logger.info(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")

            # Start MLflow run with experiment name
            logger.info("Creating experiment: wine-quality-prediction")
            mlflow.set_experiment("wine-quality-prediction")

            logger.info("Starting MLflow run")
            with mlflow.start_run(run_name="elasticnet-model"):
                # Log metrics
                logger.info("Logging metrics and parameters")
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

                # Log parameters
                mlflow.log_params(self.config.all_params)

                # Add tags for better organization
                mlflow.set_tag("model_type", "elasticnet")
                mlflow.set_tag("data_version", "1.0")

                # Log model
                logger.info("Logging model to MLflow")
                mlflow.sklearn.log_model(model, "model")

                logger.info("Model logged to MLflow successfully")
        except Exception as e:
            logger.error(f"Error logging to MLflow: {e}")
            # Continue execution even if MLflow logging fails

        logger.info(f"Model evaluation completed. RMSE: {rmse}, MAE: {mae}, R2: {r2}")
