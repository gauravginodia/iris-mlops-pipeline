"""
Training script with MLflow experiment tracking
"""
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from datetime import datetime
import os

# MLflow configuration - LOCAL STORAGE
EXPERIMENT_NAME = "iris-classification"

# Use local tracking
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Experiment: {EXPERIMENT_NAME}\n")

def load_data(data_path='data/iris.csv'):
    """Load iris dataset"""
    print(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    print(f"Data shape: {data.shape}")
    return data

def prepare_data(data, test_size=0.4, random_state=42):
    """Split data into train and test sets"""
    train, test = train_test_split(
        data, 
        test_size=test_size, 
        stratify=data['species'], 
        random_state=random_state
    )
    
    X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_train = train['species']
    X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_test = test['species']
    
    return X_train, X_test, y_train, y_test

def train_and_log_model(X_train, X_test, y_train, y_test, 
                        max_depth, min_samples_split, min_samples_leaf,
                        criterion='gini', random_state=1):
    """Train model and log to MLflow"""
    
    run_name = f"depth{max_depth}_split{min_samples_split}_leaf{min_samples_leaf}"
    
    with mlflow.start_run(run_name=run_name):
        
        # Log parameters
        mlflow.log_param('max_depth', max_depth)
        mlflow.log_param('min_samples_split', min_samples_split)
        mlflow.log_param('min_samples_leaf', min_samples_leaf)
        mlflow.log_param('criterion', criterion)
        mlflow.log_param('random_state', random_state)
        mlflow.log_param('algorithm', 'DecisionTree')
        
        # Train model
        print(f"\nExperiment: {run_name}")
        print(f"  Parameters: depth={max_depth}, split={min_samples_split}, leaf={min_samples_leaf}")
        
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_acc = metrics.accuracy_score(y_train, y_pred_train)
        test_acc = metrics.accuracy_score(y_test, y_pred_test)
        precision = metrics.precision_score(y_test, y_pred_test, average='weighted')
        recall = metrics.recall_score(y_test, y_pred_test, average='weighted')
        f1 = metrics.f1_score(y_test, y_pred_test, average='weighted')
        
        # Log metrics
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Get run info
        run_id = mlflow.active_run().info.run_id
        
        # Print results
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy:  {test_acc:.4f}")
        print(f"  Precision:      {precision:.4f}")
        print(f"  F1 Score:       {f1:.4f}")
        print(f"  Run ID: {run_id}")
        
        return {
            'model': model,
            'accuracy': test_acc,
            'run_id': run_id,
            'run_name': run_name,
            'params': {
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf
            },
            'metrics': {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        }

def run_experiments(X_train, X_test, y_train, y_test):
    """Run multiple experiments with different hyperparameters"""
    
    print("="*70)
    print("RUNNING HYPERPARAMETER TUNING EXPERIMENTS")
    print("="*70)
    
    # Define experiments
    experiments = [
        {'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'name': 'Shallow Tree'},
        {'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 1, 'name': 'Medium Tree'},
        {'max_depth': 7, 'min_samples_split': 2, 'min_samples_leaf': 1, 'name': 'Deep Tree'},
        {'max_depth': 5, 'min_samples_split': 5, 'min_samples_leaf': 2, 'name': 'Constrained'},
        {'max_depth': 4, 'min_samples_split': 10, 'min_samples_leaf': 5, 'name': 'Very Constrained'},
    ]
    
    results = []
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/5] {exp['name']}")
        print("-" * 70)
        
        result = train_and_log_model(
            X_train, X_test, y_train, y_test,
            max_depth=exp['max_depth'],
            min_samples_split=exp['min_samples_split'],
            min_samples_leaf=exp['min_samples_leaf']
        )
        result['experiment_name'] = exp['name']
        results.append(result)
    
    # Find best
    best = max(results, key=lambda x: x['accuracy'])
    
    print("\n" + "="*70)
    print("BEST MODEL")
    print("="*70)
    print(f"Experiment: {best['experiment_name']}")
    print(f"Run ID: {best['run_id']}")
    print(f"Test Accuracy: {best['accuracy']:.4f}")
    print(f"Parameters: {best['params']}")
    print("="*70)
    
    return results, best

def save_results(results, best):
    """Save results summary"""
    import json
    
    os.makedirs('reports', exist_ok=True)
    
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_experiments': len(results),
        'best_model': {
            'run_id': best['run_id'],
            'run_name': best['run_name'],
            'experiment_name': best['experiment_name'],
            'accuracy': best['accuracy'],
            'params': best['params'],
            'metrics': best['metrics']
        },
        'all_results': [
            {
                'run_name': r['run_name'],
                'experiment_name': r['experiment_name'],
                'accuracy': r['accuracy'],
                'params': r['params']
            }
            for r in results
        ]
    }
    
    # Save as JSON
    with open('reports/mlflow_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved: reports/mlflow_summary.json")
    
    # Save as CSV for easy viewing
    import pandas as pd
    df = pd.DataFrame([
        {
            'Experiment': r['experiment_name'],
            'Run Name': r['run_name'],
            'Accuracy': f"{r['accuracy']:.4f}",
            'max_depth': r['params']['max_depth'],
            'min_samples_split': r['params']['min_samples_split'],
            'min_samples_leaf': r['params']['min_samples_leaf']
        }
        for r in results
    ])
    df = df.sort_values('Accuracy', ascending=False)
    df.to_csv('reports/mlflow_results.csv', index=False)
    
    print(f"✓ CSV saved: reports/mlflow_results.csv")

def main():
    """Main pipeline"""
    print("\n" + "="*70)
    print("IRIS CLASSIFICATION WITH MLFLOW")
    print("="*70 + "\n")
    
    # Load data
    data = load_data()
    X_train, X_test, y_train, y_test = prepare_data(data)
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}\n")
    
    # Run experiments
    results, best = run_experiments(X_train, X_test, y_train, y_test)
    
    # Save best model
    import joblib
    os.makedirs('models', exist_ok=True)
    joblib.dump(best['model'], 'models/best_model.joblib')
    print(f"\n✓ Best model saved: models/best_model.joblib")
    
    # Save results
    save_results(results, best)
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)
    print(f"\nView results:")
    print(f"  1. Open: view_mlflow_results.ipynb")
    print(f"  2. Or check: reports/mlflow_results.csv")
    print(f"  3. Or run: cat reports/mlflow_summary.json")
    
    return results, best

if __name__ == "__main__":
    main()