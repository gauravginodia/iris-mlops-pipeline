import pandas as pd
from datetime import datetime, timedelta

print("Loading IRIS data...")
data = pd.read_csv('data/iris.csv')

print(f"Original shape: {data.shape}")

# Add entity ID (unique identifier for each sample)
data['iris_id'] = range(1, len(data) + 1)

# Add timestamp (required by Feast)
base_time = datetime(2024, 1, 1)
data['event_timestamp'] = [
    base_time + timedelta(hours=i) 
    for i in range(len(data))
]

# Reorder columns
data = data[[
    'iris_id',
    'event_timestamp', 
    'sepal_length',
    'sepal_width',
    'petal_length',
    'petal_width',
    'species'
]]

# Save as parquet (Feast format)
data.to_parquet('data/iris_features.parquet', index=False)

print(f"✓ Saved {len(data)} samples")
print("\nFirst 5 rows:")
print(data.head())