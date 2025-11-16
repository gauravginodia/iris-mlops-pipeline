"""
Model Evaluation Tests
Tests to ensure model performance meets minimum requirements
"""
import pytest
import pandas as pd
import joblib
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn import metrics

PROJECT_ROOT = Path(__file__).parent.parent

class TestModelEvaluation:
    """Test suite for model evaluation"""
    
    @pytest.fixture
    def model(self):
        """Load trained model"""
        model_path = PROJECT_ROOT / "models" / "iris_model.joblib"
        if not model_path.exists():
            pytest.skip(f"Model file not found: {model_path}")
        return joblib.load(model_path)
    
    @pytest.fixture
    def test_data(self):
        """Prepare test data"""
        data_path = PROJECT_ROOT / "data" / "iris.csv"
        if not data_path.exists():
            pytest.skip(f"Data file not found: {data_path}")
        
        data = pd.read_csv(data_path)
        _, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
        
        X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        y_test = test['species']
        
        return X_test, y_test
    
    def test_model_exists(self):
        """Test if model file exists"""
        model_path = PROJECT_ROOT / "models" / "iris_model.joblib"
        assert model_path.exists(), "Model file not found"
    
    def test_model_type(self, model):
        """Test if model is of correct type"""
        from sklearn.tree import DecisionTreeClassifier
        assert isinstance(model, DecisionTreeClassifier), "Model is not a DecisionTreeClassifier"
    
    def test_model_accuracy(self, model, test_data):
        """Test if model accuracy meets minimum threshold"""
        X_test, y_test = test_data
        predictions = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, predictions)
        
        min_accuracy = 0.90  # 90% minimum
        assert accuracy >= min_accuracy, f"Model accuracy {accuracy:.3f} below threshold {min_accuracy}"
    
    def test_model_predictions_shape(self, model, test_data):
        """Test if predictions have correct shape"""
        X_test, y_test = test_data
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(y_test), "Prediction count mismatch"
    
    def test_model_prediction_types(self, model, test_data):
        """Test if predictions are valid species"""
        X_test, y_test = test_data
        predictions = model.predict(X_test)
        
        valid_species = {'setosa', 'versicolor', 'virginica'}
        for pred in predictions:
            assert pred in valid_species, f"Invalid prediction: {pred}"
    
    def test_model_precision(self, model, test_data):
        """Test if model precision meets minimum threshold"""
        X_test, y_test = test_data
        predictions = model.predict(X_test)
        
        precision = metrics.precision_score(y_test, predictions, average='weighted')
        min_precision = 0.90
        
        assert precision >= min_precision, f"Precision {precision:.3f} below threshold {min_precision}"
    
    def test_model_recall(self, model, test_data):
        """Test if model recall meets minimum threshold"""
        X_test, y_test = test_data
        predictions = model.predict(X_test)
        
        recall = metrics.recall_score(y_test, predictions, average='weighted')
        min_recall = 0.90
        
        assert recall >= min_recall, f"Recall {recall:.3f} below threshold {min_recall}"
    
    def test_model_f1_score(self, model, test_data):
        """Test if model F1 score meets minimum threshold"""
        X_test, y_test = test_data
        predictions = model.predict(X_test)
        
        f1 = metrics.f1_score(y_test, predictions, average='weighted')
        min_f1 = 0.90
        
        assert f1 >= min_f1, f"F1 score {f1:.3f} below threshold {min_f1}"
    
    def test_no_prediction_errors(self, model, test_data):
        """Test if model can make predictions without errors"""
        X_test, _ = test_data
        
        try:
            predictions = model.predict(X_test)
            assert len(predictions) > 0
        except Exception as e:
            pytest.fail(f"Model prediction failed with error: {e}")