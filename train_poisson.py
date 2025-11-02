"""
Fixed Poisson Regression Training Script
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

print("="*60)
print("POISSON REGRESSION MODEL TRAINING")
print("="*60)

# Load data
print("\n[1/6] Loading data...")
df = pd.read_csv('data/processed/engineered_features.csv')
print(f"✓ Loaded {len(df)} records")

# Drop date column
if 'date' in df.columns:
    df = df.drop('date', axis=1)

# Encode categorical variables
print("\n[2/6] Encoding categorical variables...")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    print(f"  ✓ Encoded {col}")

# Prepare count target
print("\n[3/6] Preparing count target...")
df['carbon_emissions_count'] = df['carbon_emissions'].round().astype(int)
# Ensure non-negative
df['carbon_emissions_count'] = df['carbon_emissions_count'].abs()
print(f"✓ Target range: {df['carbon_emissions_count'].min()} to {df['carbon_emissions_count'].max()}")

# Prepare features
print("\n[4/6] Preparing features...")
exclude_cols = ['carbon_emissions', 'carbon_emissions_count']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols]
y = df['carbon_emissions_count']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✓ Training set: {len(X_train)} samples")
print(f"✓ Test set: {len(X_test)} samples")

# Train model
print("\n[5/6] Training Poisson Regression model...")
model = PoissonRegressor(alpha=1.0, max_iter=1000)
model.fit(X_train, y_train)
print("✓ Model trained successfully")

# Evaluate
print("\n[6/6] Evaluating model...")
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*60)
print("PERFORMANCE METRICS")
print("="*60)
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Save model
joblib.dump(model, 'models/saved_models/poisson_model.pkl')
print("\n✓ Model saved to: models/saved_models/poisson_model.pkl")

print("\n" + "="*60)
print("✓ TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
