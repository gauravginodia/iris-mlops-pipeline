import pandas as pd
import numpy as np

print("Loading original data...")
data = pd.read_csv('data/iris.csv')
print(f"Original data: {data.shape[0]} rows")

print("Creating augmented samples...")
augmented_samples = []

for _, row in data.sample(n=30, random_state=42).iterrows():
    new_row = row.copy()
    new_row['sepal_length'] += np.random.normal(0, 0.1)
    new_row['sepal_width'] += np.random.normal(0, 0.1)
    new_row['petal_length'] += np.random.normal(0, 0.1)
    new_row['petal_width'] += np.random.normal(0, 0.05)
    augmented_samples.append(new_row)

augmented_df = pd.DataFrame(augmented_samples)
combined_data = pd.concat([data, augmented_df], ignore_index=True)

combined_data.to_csv('data/iris_augmented.csv', index=False)
print(f"Augmented data: {combined_data.shape[0]} rows")
print("✓ Saved to: data/iris_augmented.csv")