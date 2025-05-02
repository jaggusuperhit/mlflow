from pathlib import Path
from typing import Dict, Any, Optional
import os

from mlProject.constants import *
from mlProject.utils.common import read_yaml, create_directories
from mlProject.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    PredictionConfig
)
from mlProject import logger


class ConfigurationManager:
    """
    Configuration manager for the ML project.

    This class is responsible for loading configuration files and creating
    configuration objects for different stages of the ML pipeline.
    """

    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
        schema_filepath: Path = SCHEMA_FILE_PATH
    ):
        """
        Initialize the ConfigurationManager.

        Args:
            config_filepath: Path to the config.yaml file
            params_filepath: Path to the params.yaml file
            schema_filepath: Path to the schema.yaml file
        """
        logger.info(f"Loading configuration from {config_filepath}")
        self.config = read_yaml(config_filepath)

        logger.info(f"Loading parameters from {params_filepath}")
        self.params = read_yaml(params_filepath)

        logger.info(f"Loading schema from {schema_filepath}")
        self.schema = read_yaml(schema_filepath)

        # Create artifacts root directory
        create_directories([self.config.artifacts_root])
        logger.info(f"Created artifacts root directory: {self.config.artifacts_root}")

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Get configuration for data ingestion stage.

        Returns:
            DataIngestionConfig: Configuration for data ingestion
        """
        logger.info("Creating data ingestion configuration")
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )

        logger.info(f"Data ingestion config: {data_ingestion_config}")
        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        """
        Get configuration for data validation stage.

        Returns:
            DataValidationConfig: Configuration for data validation
        """
        logger.info("Creating data validation configuration")
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir=config.unzip_data_dir,
            all_schema=schema
        )

        logger.info(f"Data validation config: {data_validation_config}")
        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
        Get configuration for data transformation stage.

        Returns:
            DataTransformationConfig: Configuration for data transformation
        """
        logger.info("Creating data transformation configuration")
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path
        )

        logger.info(f"Data transformation config: {data_transformation_config}")
        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        Get configuration for model trainer stage.

        Returns:
            ModelTrainerConfig: Configuration for model trainer
        """
        logger.info("Creating model trainer configuration")
        config = self.config.model_trainer
        params = self.params.ElasticNet
        schema = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            model_name=config.model_name,
            alpha=params.alpha,
            l1_ratio=params.l1_ratio,
            target_column=schema.name
        )

        logger.info(f"Model trainer config: {model_trainer_config}")
        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """
        Get configuration for model evaluation stage.

        Returns:
            ModelEvaluationConfig: Configuration for model evaluation
        """
        logger.info("Creating model evaluation configuration")
        config = self.config.model_evaluation
        params = self.params.ElasticNet
        schema = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        # Get experiment name from config or environment variable
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "wine-quality-prediction")

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path=config.model_path,
            all_params=params,
            metric_file_name=config.metric_file_name,
            target_column=schema.name,
            mlflow_uri=config.mlflow_uri,
            experiment_name=experiment_name
        )

        logger.info(f"Model evaluation config: {model_evaluation_config}")
        return model_evaluation_config

    def get_prediction_config(self, model_path: Optional[str] = None) -> PredictionConfig:
        """
        Get configuration for model prediction.

        Args:
            model_path: Optional path to the model file. If not provided,
                        the default path from model_trainer config is used.

        Returns:
            PredictionConfig: Configuration for model prediction
        """
        logger.info("Creating prediction configuration")

        if model_path is None:
            model_path = self.config.model_trainer.model_path

        # Get feature columns from schema
        feature_columns = list(self.schema.COLUMNS.keys())
        target_column = self.schema.TARGET_COLUMN.name

        # Remove target column from feature columns
        if target_column in feature_columns:
            feature_columns.remove(target_column)

        prediction_config = PredictionConfig(
            model_path=Path(model_path),
            feature_columns=feature_columns
        )

        logger.info(f"Prediction config: {prediction_config}")
        return prediction_config
