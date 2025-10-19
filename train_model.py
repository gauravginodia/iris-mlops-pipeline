import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import sys

data_file = sys.argv[1] if len(sys.argv) > 1 else 'data/iris.csv'
print(f"Training with: {data_file}")

data = pd.read_csv(data_file)
print(f"Data shape: {data.shape}")

train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species

print("Training model...")
model = DecisionTreeClassifier(max_depth=3, random_state=1)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.3f}")

os.makedirs('models', exist_ok=True)
model_path = 'models/iris_model.joblib'
joblib.dump(model, model_path)
print(f"✓ Model saved: {model_path}")

os.makedirs('logs', exist_ok=True)
with open('logs/metrics.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy:.3f}\n")
    f.write(f"Data: {data_file}\n")
    f.write(f"Samples: {len(data)}\n")
print("✓ Metrics saved")