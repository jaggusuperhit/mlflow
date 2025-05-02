from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass(frozen=True)
class DataIngestionConfig:
    """Configuration for data ingestion stage.

    Attributes:
        root_dir: Root directory for data ingestion artifacts
        source_URL: URL to download the data
        local_data_file: Path to save the downloaded data
        unzip_dir: Directory to unzip the data
    """
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    """Configuration for data validation stage.

    Attributes:
        root_dir: Root directory for data validation artifacts
        unzip_data_dir: Directory containing the unzipped data
        STATUS_FILE: Path to save the validation status
        all_schema: Schema definition for data validation
    """
    root_dir: Path
    unzip_data_dir: str
    STATUS_FILE: Path
    all_schema: Dict[str, Any]


@dataclass(frozen=True)
class DataTransformationConfig:
    """Configuration for data transformation stage.

    Attributes:
        root_dir: Root directory for data transformation artifacts
        data_path: Path to the data to be transformed
    """
    root_dir: Path
    data_path: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    """Configuration for model training stage.

    Attributes:
        root_dir: Root directory for model training artifacts
        train_data_path: Path to the training data
        test_data_path: Path to the test data
        model_name: Name of the model file
        alpha: Alpha parameter for ElasticNet
        l1_ratio: L1 ratio parameter for ElasticNet
        target_column: Name of the target column
    """
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    alpha: float
    l1_ratio: float
    target_column: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    """Configuration for model evaluation stage.

    Attributes:
        root_dir: Root directory for model evaluation artifacts
        test_data_path: Path to the test data
        model_path: Path to the trained model
        all_params: All model parameters
        metric_file_name: Path to save the evaluation metrics
        target_column: Name of the target column
        mlflow_uri: URI for MLflow tracking server
        experiment_name: Name of the MLflow experiment
    """
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: Dict[str, Any]
    metric_file_name: Path
    target_column: str
    mlflow_uri: str
    experiment_name: Optional[str] = "default-experiment"


@dataclass(frozen=True)
class PredictionConfig:
    """Configuration for model prediction.

    Attributes:
        model_path: Path to the trained model
        transformer_path: Path to the data transformer
        feature_columns: List of feature column names
    """
    model_path: Path
    transformer_path: Optional[Path] = None
    feature_columns: Optional[list] = None