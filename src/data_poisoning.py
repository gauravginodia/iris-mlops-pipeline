"""
Data Poisoning Implementation for IRIS Dataset
Poisons data at different levels: 5%, 10%, 50%
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# MLflow setup
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("iris-data-poisoning")

class DataPoisoner:
    """Class to poison IRIS dataset"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.original_df = df.copy()
    
    def poison_labels(self, poison_rate=0.05, random_state=42):
        """
        Poison labels by randomly flipping them
        
        Args:
            poison_rate: Percentage of labels to flip (0.05 = 5%)
            random_state: Random seed for reproducibility
        
        Returns:
            Poisoned dataframe
        """
        np.random.seed(random_state)
        df_poisoned = self.df.copy()
        
        # Calculate number of samples to poison
        n_samples = len(df_poisoned)
        n_poison = int(n_samples * poison_rate)
        
        # Randomly select indices to poison
        poison_indices = np.random.choice(n_samples, n_poison, replace=False)
        
        # Get unique species
        species = df_poisoned['species'].unique()
        
        # Flip labels
        for idx in poison_indices:
            current_label = df_poisoned.loc[idx, 'species']
            # Choose a different random label
            possible_labels = [s for s in species if s != current_label]
            new_label = np.random.choice(possible_labels)
            df_poisoned.loc[idx, 'species'] = new_label
        
        print(f"✓ Poisoned {n_poison}/{n_samples} samples ({poison_rate*100}%)")
        
        return df_poisoned, poison_indices
    
    def poison_features(self, poison_rate=0.05, noise_level=2.0, random_state=42):
        """
        Poison features by adding random noise
        
        Args:
            poison_rate: Percentage of samples to corrupt
            noise_level: Multiplier for noise (higher = more corruption)
            random_state: Random seed
        
        Returns:
            Poisoned dataframe
        """
        np.random.seed(random_state)
        df_poisoned = self.df.copy()
        
        n_samples = len(df_poisoned)
        n_poison = int(n_samples * poison_rate)
        
        poison_indices = np.random.choice(n_samples, n_poison, replace=False)
        
        feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        
        for idx in poison_indices:
            for col in feature_cols:
                # Add significant random noise
                original_value = df_poisoned.loc[idx, col]
                noise = np.random.normal(0, original_value * noise_level)
                df_poisoned.loc[idx, col] = max(0, original_value + noise)  # Keep non-negative
        
        print(f"✓ Corrupted features in {n_poison}/{n_samples} samples ({poison_rate*100}%)")
        
        return df_poisoned, poison_indices
    
    def poison_combined(self, poison_rate=0.05, random_state=42):
        """
        Combined attack: poison both labels and features
        """
        df_poisoned = self.df.copy()
        
        # Poison labels first
        df_poisoned, label_indices = self.poison_labels(poison_rate/2, random_state)
        
        # Then poison features
        self.df = df_poisoned
        df_poisoned, feature_indices = self.poison_features(poison_rate/2, random_state=random_state+1)
        
        all_indices = np.unique(np.concatenate([label_indices, feature_indices]))
        
        print(f"✓ Combined poisoning: {len(all_indices)} samples affected")
        
        return df_poisoned, all_indices

def train_and_evaluate(df, model_name='DecisionTree', poison_rate=0.0, 
                       poison_type='none', poison_indices=None):
    """
    Train model and log to MLflow
    """
    
    run_name = f"{poison_type}_{int(poison_rate*100)}pct_{model_name}"
    
    with mlflow.start_run(run_name=run_name):
        
        # Log parameters
        mlflow.log_param("poison_rate", poison_rate)
        mlflow.log_param("poison_type", poison_type)
        mlflow.log_param("model", model_name)
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("n_poisoned", len(poison_indices) if poison_indices is not None else 0)
        
        # Prepare data
        X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        y = df['species']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train model
        if model_name == 'DecisionTree':
            model = DecisionTreeClassifier(max_depth=5, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_acc = metrics.accuracy_score(y_train, y_pred_train)
        test_acc = metrics.accuracy_score(y_test, y_pred_test)
        precision = metrics.precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
        recall = metrics.recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
        f1 = metrics.f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
        
        # Calculate degradation from baseline
        baseline_acc = 0.98  # Expected accuracy on clean data
        degradation = baseline_acc - test_acc
        
        # Log metrics
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy_degradation", degradation)
        
        # Confusion matrix
        cm = metrics.confusion_matrix(y_test, y_pred_test)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Print results
        print(f"\n{'='*70}")
        print(f"Results: {poison_type.upper()} - {poison_rate*100}% - {model_name}")
        print(f"{'='*70}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy:  {test_acc:.4f}")
        print(f"Precision:      {precision:.4f}")
        print(f"Recall:         {recall:.4f}")
        print(f"F1 Score:       {f1:.4f}")
        print(f"Degradation:    {degradation:.4f} ({degradation/baseline_acc*100:.1f}%)")
        print(f"{'='*70}\n")
        
        return {
            'model': model,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'degradation': degradation,
            'confusion_matrix': cm
        }

def run_poisoning_experiments():
    """
    Run complete poisoning experiments
    """
    print("\n" + "="*70)
    print("DATA POISONING EXPERIMENTS - IRIS DATASET")
    print("="*70 + "\n")
    
    # Load clean data
    df_clean = pd.read_csv('data/iris.csv')
    print(f"Loaded clean data: {df_clean.shape}")
    
    # Poison rates to test
    poison_rates = [0.0, 0.05, 0.10, 0.50]
    
    results = []
    
    # Experiment 1: Baseline (Clean Data)
    print("\n" + "="*70)
    print("EXPERIMENT 1: BASELINE (CLEAN DATA)")
    print("="*70)
    
    result = train_and_evaluate(
        df_clean, 
        model_name='DecisionTree',
        poison_rate=0.0,
        poison_type='clean'
    )
    results.append({
        'poison_type': 'clean',
        'poison_rate': 0.0,
        'test_acc': result['test_acc'],
        'degradation': result['degradation']
    })
    
    # Experiment 2: Label Poisoning
    print("\n" + "="*70)
    print("EXPERIMENT 2: LABEL POISONING")
    print("="*70)
    
    for rate in poison_rates[1:]:  # Skip 0%
        poisoner = DataPoisoner(df_clean)
        df_poisoned, indices = poisoner.poison_labels(rate)
        
        result = train_and_evaluate(
            df_poisoned,
            model_name='DecisionTree',
            poison_rate=rate,
            poison_type='label_flip',
            poison_indices=indices
        )
        results.append({
            'poison_type': 'label_flip',
            'poison_rate': rate,
            'test_acc': result['test_acc'],
            'degradation': result['degradation']
        })
    
    # Experiment 3: Feature Corruption
    print("\n" + "="*70)
    print("EXPERIMENT 3: FEATURE CORRUPTION")
    print("="*70)
    
    for rate in poison_rates[1:]:
        poisoner = DataPoisoner(df_clean)
        df_poisoned, indices = poisoner.poison_features(rate, noise_level=2.0)
        
        result = train_and_evaluate(
            df_poisoned,
            model_name='DecisionTree',
            poison_rate=rate,
            poison_type='feature_noise',
            poison_indices=indices
        )
        results.append({
            'poison_type': 'feature_noise',
            'poison_rate': rate,
            'test_acc': result['test_acc'],
            'degradation': result['degradation']
        })
    
    # Experiment 4: Combined Attack
    print("\n" + "="*70)
    print("EXPERIMENT 4: COMBINED ATTACK (LABELS + FEATURES)")
    print("="*70)
    
    for rate in poison_rates[1:]:
        poisoner = DataPoisoner(df_clean)
        df_poisoned, indices = poisoner.poison_combined(rate)
        
        result = train_and_evaluate(
            df_poisoned,
            model_name='DecisionTree',
            poison_rate=rate,
            poison_type='combined',
            poison_indices=indices
        )
        results.append({
            'poison_type': 'combined',
            'poison_rate': rate,
            'test_acc': result['test_acc'],
            'degradation': result['degradation']
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    os.makedirs('reports', exist_ok=True)
    results_df.to_csv('reports/poisoning_results.csv', index=False)
    
    print("\n" + "="*70)
    print("✓ ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"Results saved to: reports/poisoning_results.csv")
    print(f"View in MLflow: mlflow ui")
    
    return results_df

if __name__ == "__main__":
    results = run_poisoning_experiments()
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(results.to_string(index=False))