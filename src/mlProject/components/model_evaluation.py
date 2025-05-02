import os
import sys
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator
from dotenv import load_dotenv

from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.utils.common import save_json
from mlProject.utils.dagshub_utils import setup_dagshub_auth
from mlProject import logger


class ModelEvaluation:
    """
    Model evaluation component for evaluating model performance and logging to MLflow.
    """

    def __init__(self, config: ModelEvaluationConfig):
        """
        Initialize the ModelEvaluation component.

        Args:
            config: Configuration for model evaluation
        """
        self.config = config

    def eval_metrics(self, actual: pd.DataFrame, pred: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate evaluation metrics for the model.

        Args:
            actual: Actual target values
            pred: Predicted target values

        Returns:
            Tuple containing RMSE, MAE, and R2 score
        """
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def _setup_mlflow(self) -> bool:
        """
        Set up MLflow tracking.

        Returns:
            bool: True if setup was successful, False otherwise
        """
        # Load environment variables from .env file
        load_dotenv()

        try:
            # Get credentials from environment variables
            mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", self.config.mlflow_uri)
            mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
            mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")

            if not mlflow_uri:
                logger.warning("MLflow tracking URI not found. Skipping MLflow logging.")
                return False

            if not mlflow_username or not mlflow_password:
                logger.warning("MLflow credentials not found in environment variables. Skipping MLflow logging.")
                return False

            # Set environment variables for MLflow
            os.environ['MLFLOW_TRACKING_URI'] = mlflow_uri
            os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_username
            os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_password

            # Set MLflow tracking URI
            mlflow.set_tracking_uri(mlflow_uri)
            logger.info(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")

            return True
        except Exception as e:
            logger.error(f"Error setting up MLflow: {e}")
            return False

    def _log_model_to_mlflow(
        self,
        model: BaseEstimator,
        metrics: Dict[str, float],
        feature_names: List[str],
        run_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Log model, metrics, and parameters to MLflow.

        Args:
            model: Trained model
            metrics: Dictionary of metrics
            feature_names: List of feature names
            run_name: Name for the MLflow run

        Returns:
            Optional[str]: Run ID if successful, None otherwise
        """
        if not self._setup_mlflow():
            return None

        experiment_name = self.config.experiment_name
        run_name = run_name or "elasticnet-model"

        try:
            # Set experiment
            logger.info(f"Creating experiment: {experiment_name}")
            mlflow.set_experiment(experiment_name)

            # Create a sample input for model signature
            sample_input = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

            logger.info(f"Starting MLflow run: {run_name}")
            with mlflow.start_run(run_name=run_name) as run:
                run_id = run.info.run_id

                # Log metrics
                logger.info("Logging metrics")
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

                # Log parameters
                logger.info("Logging parameters")
                mlflow.log_params(self.config.all_params)

                # Add tags for better organization
                mlflow.set_tag("model_type", "elasticnet")
                mlflow.set_tag("data_version", "1.0")
                mlflow.set_tag("python_version", sys.version)

                # Log model with signature
                logger.info("Logging model to MLflow")
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    input_example=sample_input,
                    registered_model_name="ElasticNetWineModel"
                )

                logger.info(f"Model logged to MLflow successfully with run_id: {run_id}")
                return run_id
        except Exception as e:
            logger.error(f"Error logging to MLflow: {e}")
            return None

    def log_into_mlflow(self) -> Dict[str, float]:
        """
        Evaluate the model and log results to MLflow.

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        logger.info("Loading test data and model")
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        logger.info("Preparing test data")
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        feature_names = test_x.columns.tolist()

        # Evaluate the model
        logger.info("Evaluating model")
        predicted_qualities = model.predict(test_x)
        (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)

        # Save metrics locally
        metrics = {"rmse": rmse, "mae": mae, "r2": r2}
        save_json(path=Path(self.config.metric_file_name), data=metrics)
        logger.info(f"Metrics saved to {self.config.metric_file_name}")

        # Log to MLflow
        run_id = self._log_model_to_mlflow(model, metrics, feature_names)

        if run_id:
            logger.info(f"MLflow run created with ID: {run_id}")

        logger.info(f"Model evaluation completed. RMSE: {rmse}, MAE: {mae}, R2: {r2}")
        return metrics

    def evaluate_and_compare(self, baseline_metrics_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate the model and compare with baseline metrics.

        Args:
            baseline_metrics_path: Path to baseline metrics file

        Returns:
            Dict[str, Any]: Comparison results
        """
        # Get current model metrics
        current_metrics = self.log_into_mlflow()

        # If no baseline metrics provided, return current metrics
        if not baseline_metrics_path:
            return {
                "current_metrics": current_metrics,
                "baseline_metrics": None,
                "is_improved": None,
                "improvement": None
            }

        # Load baseline metrics
        try:
            baseline_metrics = pd.read_json(baseline_metrics_path).to_dict(orient='records')[0]

            # Compare metrics
            is_improved = current_metrics["r2"] > baseline_metrics["r2"]
            improvement = {
                "rmse": baseline_metrics["rmse"] - current_metrics["rmse"],
                "mae": baseline_metrics["mae"] - current_metrics["mae"],
                "r2": current_metrics["r2"] - baseline_metrics["r2"]
            }

            comparison = {
                "current_metrics": current_metrics,
                "baseline_metrics": baseline_metrics,
                "is_improved": is_improved,
                "improvement": improvement
            }

            logger.info(f"Model comparison: {comparison}")
            return comparison

        except Exception as e:
            logger.error(f"Error comparing with baseline: {e}")
            return {
                "current_metrics": current_metrics,
                "baseline_metrics": None,
                "is_improved": None,
                "improvement": None,
                "error": str(e)
            }
