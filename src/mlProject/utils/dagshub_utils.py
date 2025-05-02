import os
import mlflow
from mlProject import logger
from dotenv import load_dotenv

def setup_dagshub_auth():
    """
    Set up DagHub authentication using environment variables.
    Returns True if successful, False otherwise.
    """
    try:
        # Load environment variables from .env file
        load_dotenv()

        # Get credentials from environment variables
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
        mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
        mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")

        if not mlflow_uri or not mlflow_username or not mlflow_password:
            logger.warning("MLflow credentials not found in environment variables.")
            return False

        # Set environment variables for MLflow
        os.environ['MLFLOW_TRACKING_URI'] = mlflow_uri
        os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_password

        # Set MLflow tracking URI directly
        mlflow.set_tracking_uri(mlflow_uri)

        logger.info("MLflow authentication set up successfully")
        return True
    except Exception as e:
        logger.error(f"Error setting up MLflow authentication: {e}")
        return False
