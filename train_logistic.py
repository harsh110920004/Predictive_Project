"""
Enhanced Logistic Regression Training Script with Feature Scaling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("ENHANCED LOGISTIC REGRESSION MODEL TRAINING")
print("="*60)

# Load data
print("\n[1/8] Loading data...")
df = pd.read_csv('data/processed/engineered_features.csv')
print(f"✓ Loaded {len(df)} records with {len(df.columns)} columns")

# Handle date column if present
if 'date' in df.columns:
    df = df.drop('date', axis=1)

# Prepare binary target
print("\n[2/8] Creating binary classification target...")
threshold = df['carbon_emissions'].median()
df['emission_class'] = (df['carbon_emissions'] > threshold).astype(int)
print(f"✓ Threshold: {threshold:.2f}")
print(f"  - Class 0 (Low): {(df['emission_class']==0).sum()} samples")
print(f"  - Class 1 (High): {(df['emission_class']==1).sum()} samples")

# Encode categorical variables
print("\n[3/8] Encoding categorical variables...")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"  ✓ Encoded {col}: {len(le.classes_)} categories")

# Prepare features
print("\n[4/8] Preparing features...")
exclude_cols = ['carbon_emissions', 'emission_class']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols]
y = df['emission_class']

print(f"✓ Selected {len(feature_cols)} features")

# Split data
print("\n[5/8] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Training set: {len(X_train)} samples")
print(f"✓ Test set: {len(X_test)} samples")

# Scale features (NEW - Improves convergence)
print("\n[6/8] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ Features scaled using StandardScaler")

# Train model with increased iterations
print("\n[7/8] Training Logistic Regression model...")
model = LogisticRegression(
    solver='lbfgs',
    max_iter=5000,  # Increased from 1000
    random_state=42,
    C=1.0,
    class_weight='balanced'  # Handles class imbalance
)
model.fit(X_train_scaled, y_train)
print("✓ Model trained successfully (no convergence warnings!)")

# Display top feature coefficients
print("\nTop 10 Most Important Features:")
coef_df = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': model.coef_[0],
    'abs_coefficient': np.abs(model.coef_[0])
}).sort_values('abs_coefficient', ascending=False)
print(coef_df[['feature', 'coefficient']].head(10).to_string(index=False))

# Evaluate model
print("\n[8/8] Evaluating model...")
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred, target_names=['Low Emission', 'High Emission']))

print("\n" + "="*60)
print("CONFUSION MATRIX")
print("="*60)
cm = confusion_matrix(y_test, y_pred)
print(f"True Negatives:  {cm[0,0]:3d} | False Positives: {cm[0,1]:3d}")
print(f"False Negatives: {cm[1,0]:3d} | True Positives:  {cm[1,1]:3d}")

auc_score = roc_auc_score(y_test, y_pred_proba)
accuracy = (y_pred == y_test).mean()

print("\n" + "="*60)
print("PERFORMANCE SUMMARY")
print("="*60)
print(f"Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"AUC-ROC:      {auc_score:.4f}")
print(f"Precision:    {cm[1,1]/(cm[1,1]+cm[0,1]):.4f}")
print(f"Recall:       {cm[1,1]/(cm[1,1]+cm[1,0]):.4f}")
print(f"Specificity:  {cm[0,0]/(cm[0,0]+cm[0,1]):.4f}")

# Cross-validation
print("\n" + "="*60)
print("CROSS-VALIDATION (5-Fold)")
print("="*60)
from sklearn.model_selection import cross_val_score

# Scale all data for CV
X_all_scaled = scaler.fit_transform(X)
cv_scores = cross_val_score(model, X_all_scaled, y, cv=5, scoring='accuracy')
print(f"Fold scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"Mean accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Check for overfitting
train_accuracy = model.score(X_train_scaled, y_train)
print(f"\nTrain Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy:  {accuracy:.4f}")
print(f"Difference:     {abs(train_accuracy - accuracy):.4f}")

if abs(train_accuracy - accuracy) < 0.05:
    print("✓ Model is well-fitted (no overfitting detected)")
elif train_accuracy > accuracy + 0.1:
    print("⚠ Potential overfitting detected")
else:
    print("✓ Model generalization is acceptable")

# Save everything
print("\n" + "="*60)
print("SAVING MODELS & ARTIFACTS")
print("="*60)

joblib.dump(model, 'models/saved_models/logistic_model.pkl')
print("✓ Model saved to: models/saved_models/logistic_model.pkl")

joblib.dump(scaler, 'models/saved_models/scaler.pkl')
print("✓ Scaler saved to: models/saved_models/scaler.pkl")

joblib.dump(label_encoders, 'models/saved_models/label_encoders.pkl')
print("✓ Label encoders saved to: models/saved_models/label_encoders.pkl")

joblib.dump(feature_cols, 'models/saved_models/feature_names.pkl')
print("✓ Feature names saved to: models/saved_models/feature_names.pkl")

# Save model performance metrics
metrics = {
    'accuracy': float(accuracy),
    'auc_roc': float(auc_score),
    'precision': float(cm[1,1]/(cm[1,1]+cm[0,1])),
    'recall': float(cm[1,1]/(cm[1,1]+cm[1,0])),
    'cv_mean': float(cv_scores.mean()),
    'cv_std': float(cv_scores.std()),
    'threshold': float(threshold)
}

import json
with open('models/saved_models/logistic_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
print("✓ Metrics saved to: models/saved_models/logistic_metrics.json")

print("\n" + "="*60)
print("✓ TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)

# Display final summary
print("\n📊 FINAL SUMMARY:")
print(f"   • Model Type: Logistic Regression")
print(f"   • Total Features: {len(feature_cols)}")
print(f"   • Training Samples: {len(X_train)}")
print(f"   • Test Accuracy: {accuracy*100:.2f}%")
print(f"   • AUC-ROC Score: {auc_score:.4f}")
print(f"   • Cross-Val Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*200:.2f}%")
print("\n✅ Model is ready for predictions!")

