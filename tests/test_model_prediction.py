import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mlProject.components.model_prediction import ModelPredictor
from src.mlProject.entity.config_entity import PredictionConfig


class TestModelPredictor:
    """Test cases for ModelPredictor class."""

    @pytest.fixture
    def mock_model_path(self, tmp_path):
        """Create a mock model file."""
        import joblib
        from sklearn.linear_model import ElasticNet

        # Create a simple model
        model = ElasticNet(alpha=0.5, l1_ratio=0.7, random_state=42)
        model.coef_ = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
        model.intercept_ = 5.0

        # Save the model
        model_path = tmp_path / "model.joblib"
        joblib.dump(model, model_path)

        return model_path

    @pytest.fixture
    def feature_columns(self):
        """Return feature column names."""
        return [
            "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
            "pH", "sulphates", "alcohol"
        ]

    @pytest.fixture
    def prediction_config(self, mock_model_path, feature_columns):
        """Create a prediction configuration."""
        return PredictionConfig(
            model_path=mock_model_path,
            feature_columns=feature_columns
        )

    @pytest.fixture
    def predictor(self, prediction_config):
        """Create a model predictor instance."""
        return ModelPredictor(prediction_config)

    def test_init(self, predictor, prediction_config):
        """Test initialization of ModelPredictor."""
        assert predictor.config == prediction_config
        assert predictor.model is not None
        assert predictor.feature_columns == prediction_config.feature_columns

    def test_predict_dataframe(self, predictor, feature_columns):
        """Test prediction with DataFrame input."""
        # Create a sample DataFrame
        data = {col: [0.5] for col in feature_columns}
        df = pd.DataFrame(data)

        # Make prediction
        predictions = predictor.predict(df)

        # Check predictions
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 1

    def test_predict_dict(self, predictor, feature_columns):
        """Test prediction with dictionary input."""
        # Create a sample dictionary
        data = {col: [0.5] for col in feature_columns}

        # Make prediction
        predictions = predictor.predict(data)

        # Check predictions
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 1

    def test_predict_single(self, predictor, feature_columns):
        """Test prediction for a single sample."""
        # Create a sample dictionary
        data = {col: 0.5 for col in feature_columns}

        # Make prediction
        prediction = predictor.predict_single(data)

        # Check prediction
        assert isinstance(prediction, float)

    def test_validate_input_missing_features(self, predictor):
        """Test validation with missing features."""
        # Create a sample DataFrame with missing features
        data = pd.DataFrame({"fixed acidity": [0.5]})

        # Check that validation raises an error
        with pytest.raises(ValueError):
            predictor._validate_input(data)
