"""
Inference script using MLflow Model Registry
"""
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os

# MLflow configuration
EXPERIMENT_NAME = "iris-classification"
MODEL_NAME = "iris-classifier"

def get_latest_model_version(model_name):
    """Get the latest version of registered model"""
    client = mlflow.tracking.MlflowClient()
    
    try:
        # Get all versions of the model
        versions = client.search_model_versions(f"name='{model_name}'")
        
        if not versions:
            print(f"No versions found for model: {model_name}")
            return None
        
        # Sort by version number (descending) and get latest
        latest_version = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
        
        print(f"Latest model version: {latest_version.version}")
        print(f"Status: {latest_version.current_stage}")
        print(f"Run ID: {latest_version.run_id}")
        
        return latest_version
    
    except Exception as e:
        print(f"Error getting model version: {e}")
        return None

def get_best_model(model_name, metric_name='test_accuracy'):
    """Get the best model based on a metric"""
    client = mlflow.tracking.MlflowClient()
    
    try:
        # Get all versions
        versions = client.search_model_versions(f"name='{model_name}'")
        
        if not versions:
            print(f"No versions found for model: {model_name}")
            return None
        
        # Get metrics for each version
        best_version = None
        best_metric = -1
        
        for version in versions:
            run = client.get_run(version.run_id)
            metric_value = run.data.metrics.get(metric_name, 0)
            
            if metric_value > best_metric:
                best_metric = metric_value
                best_version = version
        
        print(f"Best model version: {best_version.version}")
        print(f"Best {metric_name}: {best_metric:.4f}")
        
        return best_version
    
    except Exception as e:
        print(f"Error getting best model: {e}")
        return None

def load_model_from_registry(model_name, version=None, use_best=True):
    """Load model from MLflow Model Registry"""
    
    if use_best:
        print("Loading best model based on test accuracy...")
        model_version = get_best_model(model_name)
    elif version:
        print(f"Loading model version {version}...")
        model_uri = f"models:/{model_name}/{version}"
        return mlflow.sklearn.load_model(model_uri)
    else:
        print("Loading latest model...")
        model_version = get_latest_model_version(model_name)
    
    if model_version:
        model_uri = f"models:/{model_name}/{model_version.version}"
        model = mlflow.sklearn.load_model(model_uri)
        return model, model_version
    
    return None, None

def run_inference(model, X_test, y_test):
    """Run inference and calculate metrics"""
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate metrics
    accuracy = metrics.accuracy_score(y_test, predictions)
    precision = metrics.precision_score(y_test, predictions, average='weighted')
    recall = metrics.recall_score(y_test, predictions, average='weighted')
    f1 = metrics.f1_score(y_test, predictions, average='weighted')
    
    # Print results
    print("\n" + "="*70)
    print("INFERENCE RESULTS")
    print("="*70)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # Show classification report
    print("\nClassification Report:")
    print(metrics.classification_report(y_test, predictions))
    
    # Show confusion matrix
    print("\nConfusion Matrix:")
    cm = metrics.confusion_matrix(y_test, predictions)
    print(cm)
    
    # Sample predictions
    results_df = pd.DataFrame({
        'Actual': y_test.values[:10],
        'Predicted': predictions[:10],
        'Match': (y_test.values[:10] == predictions[:10])
    })
    
    print("\nSample Predictions (first 10):")
    print(results_df.to_string(index=False))
    
    correct = (y_test.values == predictions).sum()
    print(f"\n✓ Correct: {correct}/{len(predictions)} ({correct/len(predictions)*100:.1f}%)")
    
    return predictions, accuracy

def main():
    """Main inference pipeline"""
    print("\n" + "="*70)
    print("IRIS INFERENCE WITH MLFLOW MODEL REGISTRY")
    print("="*70 + "\n")
    
    # Load test data
    print("Loading test data...")
    data = pd.read_csv('data/iris.csv')
    _, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
    X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_test = test['species']
    print(f"Test samples: {len(X_test)}")
    
    # Load best model from registry
    model, model_version = load_model_from_registry(MODEL_NAME, use_best=True)
    
    if model is None:
        print("\n❌ No model found in registry!")
        print("Run training first: python src/train_with_mlflow.py")
        return
    
    print(f"\n✓ Model loaded successfully!")
    
    # Run inference
    predictions, accuracy = run_inference(model, X_test, y_test)
    
    print("\n" + "="*70)
    print("✓ INFERENCE COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()