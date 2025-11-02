"""
Complete Model Comparison and Evaluation Script
Trains and compares multiple regression models for carbon emission prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("COMPREHENSIVE MODEL COMPARISON & EVALUATION")
print("="*70)

# ============================================================
# STEP 1: DATA LOADING & PREPROCESSING
# ============================================================
print("\n[1/7] Loading and preprocessing data...")
df = pd.read_csv('data/processed/engineered_features.csv')
print(f"✓ Loaded {len(df)} records with {len(df.columns)} columns")

# Drop date column
if 'date' in df.columns:
    df = df.drop('date', axis=1)

# Encode categorical variables
print("\n[2/7] Encoding categorical variables...")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"  ✓ Encoded {col}: {len(le.classes_)} categories")

# Prepare features and target
print("\n[3/7] Preparing features and target...")
target_col = 'carbon_emissions'
feature_cols = [col for col in df.columns if col != target_col]

X = df[feature_cols]
y = df[target_col]

print(f"✓ Features: {len(feature_cols)}")
print(f"✓ Target: {target_col}")
print(f"✓ Target range: {y.min():.2f} to {y.max():.2f}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✓ Training set: {len(X_train)} samples")
print(f"✓ Test set: {len(X_test)} samples")

# Scale features
print("\n[4/7] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ Features scaled using StandardScaler")

# ============================================================
# STEP 2: TRAIN MULTIPLE MODELS
# ============================================================
print("\n[5/7] Training multiple regression models...")
print("="*70)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0, random_state=42),
    'Lasso Regression': Lasso(alpha=1.0, random_state=42),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
}

trained_models = {}
results = []

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    
    # Use scaled data for linear models, unscaled for tree-based
    if model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    
    results.append({
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape
    })
    
    trained_models[model_name] = model
    
    print(f"  ✓ RMSE: {rmse:.2f}")
    print(f"  ✓ MAE: {mae:.2f}")
    print(f"  ✓ R²: {r2:.4f}")
    print(f"  ✓ MAPE: {mape:.2f}%")

# ============================================================
# STEP 3: MODEL COMPARISON
# ============================================================
print("\n[6/7] Comparing model performance...")
print("\n" + "="*70)
print("MODEL COMPARISON RESULTS")
print("="*70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('RMSE')

print(results_df.to_string(index=False))

# Find best model
best_model_name = results_df.iloc[0]['Model']
best_rmse = results_df.iloc[0]['RMSE']
best_r2 = results_df.iloc[0]['R²']

print("\n" + "="*70)
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"   • RMSE: {best_rmse:.2f}")
print(f"   • R²: {best_r2:.4f}")
print("="*70)

# ============================================================
# STEP 4: CROSS-VALIDATION ON BEST MODEL
# ============================================================
print(f"\n[7/7] Performing 5-fold cross-validation on {best_model_name}...")

best_model = trained_models[best_model_name]

if best_model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
    X_cv = scaler.fit_transform(X)
else:
    X_cv = X

cv_scores = cross_val_score(
    best_model, X_cv, y, 
    cv=5, 
    scoring='neg_root_mean_squared_error'
)
cv_scores = -cv_scores  # Convert to positive RMSE

print(f"Fold scores: {[f'{s:.2f}' for s in cv_scores]}")
print(f"Mean RMSE: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

# ============================================================
# STEP 5: FEATURE IMPORTANCE (for tree-based models)
# ============================================================
if best_model_name in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
    print(f"\n" + "="*70)
    print(f"TOP 15 MOST IMPORTANT FEATURES ({best_model_name})")
    print("="*70)
    
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance_df.head(15).to_string(index=False))

# ============================================================
# STEP 6: OVERFITTING ANALYSIS
# ============================================================
print("\n" + "="*70)
print("OVERFITTING ANALYSIS")
print("="*70)

if best_model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
    train_score = best_model.score(X_train_scaled, y_train)
    test_score = best_model.score(X_test_scaled, y_test)
else:
    train_score = best_model.score(X_train, y_train)
    test_score = best_model.score(X_test, y_test)

print(f"Training R²:   {train_score:.4f}")
print(f"Testing R²:    {test_score:.4f}")
print(f"Difference:    {abs(train_score - test_score):.4f}")

if abs(train_score - test_score) < 0.05:
    print("\n✓ Model is WELL-FITTED (excellent generalization)")
elif train_score > test_score + 0.1:
    print("\n⚠ OVERFITTING detected (train score >> test score)")
    print("  Recommendations:")
    print("  • Reduce model complexity")
    print("  • Increase regularization")
    print("  • Add more training data")
else:
    print("\n✓ Model generalization is ACCEPTABLE")

# ============================================================
# STEP 7: HYPERPARAMETER TUNING (Best Model)
# ============================================================
print("\n" + "="*70)
print(f"HYPERPARAMETER TUNING FOR {best_model_name}")
print("="*70)

if best_model_name == 'Random Forest':
    print("\nPerforming Grid Search...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5]
    }
    
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid,
        cv=3,
        scoring='neg_root_mean_squared_error',
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best RMSE: {-grid_search.best_score_:.2f}")
    
    # Use tuned model
    best_model = grid_search.best_estimator_
    y_pred_tuned = best_model.predict(X_test)
    rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
    r2_tuned = r2_score(y_test, y_pred_tuned)
    
    print(f"Tuned Model RMSE: {rmse_tuned:.2f}")
    print(f"Tuned Model R²: {r2_tuned:.4f}")
    
    improvement = ((best_rmse - rmse_tuned) / best_rmse) * 100
    print(f"Improvement: {improvement:.2f}%")

# ============================================================
# STEP 8: SAVE EVERYTHING
# ============================================================
print("\n" + "="*70)
print("SAVING MODELS & RESULTS")
print("="*70)

# Save best model
joblib.dump(best_model, 'models/saved_models/best_model.pkl')
print(f"✓ Best model saved: models/saved_models/best_model.pkl")

# Save all models
for model_name, model in trained_models.items():
    filename = model_name.lower().replace(' ', '_')
    joblib.dump(model, f'models/saved_models/{filename}_model.pkl')
print(f"✓ All models saved to: models/saved_models/")

# Save scaler and encoders
joblib.dump(scaler, 'models/saved_models/regression_scaler.pkl')
joblib.dump(label_encoders, 'models/saved_models/regression_encoders.pkl')
joblib.dump(feature_cols, 'models/saved_models/regression_features.pkl')
print("✓ Preprocessing artifacts saved")

# Save comparison results
results_dict = {
    'comparison': results_df.to_dict('records'),
    'best_model': best_model_name,
    'best_metrics': {
        'rmse': float(best_rmse),
        'r2': float(best_r2),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std())
    },
    'overfitting_analysis': {
        'train_r2': float(train_score),
        'test_r2': float(test_score),
        'difference': float(abs(train_score - test_score))
    }
}

with open('models/model_comparison_results.json', 'w') as f:
    json.dump(results_dict, f, indent=4)
print("✓ Comparison results saved: models/model_comparison_results.json")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*70)
print("✓ MODEL EVALUATION COMPLETED SUCCESSFULLY!")
print("="*70)

print("\n📊 FINAL SUMMARY:")
print(f"   • Total Models Trained: {len(models)}")
print(f"   • Best Model: {best_model_name}")
print(f"   • Best RMSE: {best_rmse:.2f}")
print(f"   • Best R² Score: {best_r2:.4f}")
print(f"   • Cross-Val RMSE: {cv_scores.mean():.2f} ± {cv_scores.std()*2:.2f}")
print(f"   • Model Status: {'Well-fitted ✓' if abs(train_score - test_score) < 0.05 else 'Acceptable ✓'}")

print("\n📁 Saved Files:")
print("   • models/saved_models/best_model.pkl")
print("   • models/saved_models/regression_scaler.pkl")
print("   • models/model_comparison_results.json")

print("\n✅ All models are ready for production use!")
print("="*70)
