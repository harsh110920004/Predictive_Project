"""
Poisson Regression Module for Count Data
Predicts emission counts/frequencies
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import yaml

class PoissonRegressionModel:
    """
    Poisson Regression for count-based carbon emission prediction
    As covered in CSE3141 syllabus
    """
    
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        params = self.config['models']['poisson_regression']
        self.model = PoissonRegressor(
            alpha=params['alpha'],
            max_iter=params['max_iter']
        )
        
        self.is_fitted = False
    
    def prepare_count_target(self, df, target_column):
        """
        Prepare target variable for Poisson regression
        Convert to counts if necessary
        """
        print("\n=== POISSON REGRESSION DATA PREPARATION ===")
        
        # Ensure target is non-negative
        if (df[target_column] < 0).any():
            print("Warning: Negative values detected. Converting to absolute values.")
            df[target_column] = df[target_column].abs()
        
        # Convert to integer counts
        df[f'{target_column}_count'] = df[target_column].round().astype(int)
        
        print(f"Target range: {df[f'{target_column}_count'].min()} to {df[f'{target_column}_count'].max()}")
        print(f"Mean count: {df[f'{target_column}_count'].mean():.2f}")
        
        return df
    
    def train(self, X_train, y_train):
        """
        Train Poisson regression model
        """
        print("\n=== TRAINING POISSON REGRESSION ===")
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        print("Model training completed")
        
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate Poisson regression model
        Metrics: RMSE, MAE, R²
        """
        print("\n=== POISSON REGRESSION EVALUATION ===")
        
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nRoot Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        return {
            'predictions': y_pred,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def save_model(self, filepath):
        """Save trained model"""
        if not self.is_fitted:
            raise ValueError("No trained model to save")
        joblib.dump(self.model, filepath)
        print(f"\nModel saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        self.model = joblib.load(filepath)
        self.is_fitted = True
        print(f"\nModel loaded from: {filepath}")


# Example usage
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/processed/engineered_features.csv')
    
    # Initialize model
    poisson_model = PoissonRegressionModel()
    
    # Prepare count target
    df = poisson_model.prepare_count_target(df, 'carbon_emissions')
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in ['carbon_emissions', 'carbon_emissions_count', 'date']]
    X = df[feature_cols]
    y = df['carbon_emissions_count']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    poisson_model.train(X_train, y_train)
    
    # Evaluate model
    results = poisson_model.evaluate(X_test, y_test)
    
    # Save model
    poisson_model.save_model('models/saved_models/poisson_model.pkl')
