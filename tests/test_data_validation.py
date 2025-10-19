"""
Data Validation Tests
Tests to ensure data quality and schema correctness
"""
import pytest
import pandas as pd
import os
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

class TestDataValidation:
    """Test suite for data validation"""
    
    @pytest.fixture
    def iris_data(self):
        """Load iris dataset"""
        data_path = PROJECT_ROOT / "data" / "iris.csv"
        if not data_path.exists():
            pytest.skip(f"Data file not found: {data_path}")
        return pd.read_csv(data_path)
    
    def test_data_exists(self):
        """Test if data file exists"""
        data_path = PROJECT_ROOT / "data" / "iris.csv"
        assert data_path.exists(), "iris.csv file not found"
    
    def test_data_shape(self, iris_data):
        """Test if data has correct number of rows and columns"""
        assert iris_data.shape[0] == 150, f"Expected 150 rows, got {iris_data.shape[0]}"
        assert iris_data.shape[1] == 5, f"Expected 5 columns, got {iris_data.shape[1]}"
    
    def test_required_columns(self, iris_data):
        """Test if all required columns exist"""
        required_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        for col in required_columns:
            assert col in iris_data.columns, f"Missing required column: {col}"
    
    def test_no_missing_values(self, iris_data):
        """Test if there are no missing values"""
        missing_count = iris_data.isnull().sum().sum()
        assert missing_count == 0, f"Found {missing_count} missing values"
    
    def test_feature_types(self, iris_data):
        """Test if feature columns are numeric"""
        numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(iris_data[col]), f"{col} is not numeric"
    
    def test_target_values(self, iris_data):
        """Test if target has correct unique values"""
        expected_species = {'setosa', 'versicolor', 'virginica'}
        actual_species = set(iris_data['species'].unique())
        assert actual_species == expected_species, f"Unexpected species values: {actual_species}"
    
    def test_feature_ranges(self, iris_data):
        """Test if features are within reasonable ranges"""
        # Sepal length should be between 4 and 8 cm
        assert iris_data['sepal_length'].min() >= 4.0, "Sepal length too small"
        assert iris_data['sepal_length'].max() <= 8.0, "Sepal length too large"
        
        # All features should be positive
        numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for col in numeric_cols:
            assert (iris_data[col] > 0).all(), f"{col} contains non-positive values"
    
    def test_class_distribution(self, iris_data):
        """Test if classes are balanced"""
        class_counts = iris_data['species'].value_counts()
        assert len(class_counts) == 3, "Should have exactly 3 species"
        assert (class_counts == 50).all(), "Classes should be balanced (50 samples each)"