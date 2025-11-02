"""
Logistic Regression Module for Binary Classification
Classifies emission levels as High/Low
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import yaml

class LogisticRegressionModel:
    """
    Logistic Regression for binary classification of carbon emissions
    Implements techniques from CSE3141 syllabus
    """
    
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        params = self.config['models']['logistic_regression']
        self.model = LogisticRegression(
            solver=params['solver'],
            max_iter=params['max_iter'],
            C=params['C'],
            random_state=self.config['data']['random_state']
        )
        
        self.is_fitted = False
    
    def prepare_binary_target(self, df, target_column, threshold='median'):
        """
        Convert continuous target to binary classification
        1 = High emissions (above threshold)
        0 = Low emissions (below threshold)
        """
        if threshold == 'median':
            threshold_value = df[target_column].median()
        elif threshold == 'mean':
            threshold_value = df[target_column].mean()
        else:
            threshold_value = threshold
        
        df['emission_class'] = (df[target_column] > threshold_value).astype(int)
        
        print(f"\n=== BINARY CLASSIFICATION SETUP ===")
        print(f"Threshold: {threshold_value:.2f}")
        print(f"Class 0 (Low): {(df['emission_class']==0).sum()} samples")
        print(f"Class 1 (High): {(df['emission_class']==1).sum()} samples")
        
        return df, threshold_value
    
    def train(self, X_train, y_train):
        """
        Train logistic regression model
        Estimation of regression coefficients
        """
        print("\n=== TRAINING LOGISTIC REGRESSION ===")
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Display coefficients
        print(f"\nIntercept: {self.model.intercept_[0]:.4f}")
        print(f"\nTop 10 Feature Coefficients:")
        if hasattr(X_train, 'columns'):
            coef_df = pd.DataFrame({
                'feature': X_train.columns,
                'coefficient': self.model.coef_[0]
            }).sort_values('coefficient', key=abs, ascending=False)
            print(coef_df.head(10))
        
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        Metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC
        """
        print("\n=== MODEL EVALUATION ===")
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # AUC-ROC Score
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"\nAUC-ROC Score: {auc_score:.4f}")
        
        return {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'auc_score': auc_score
        }
    
    def cross_validate(self, X, y):
        """
        Perform k-fold cross-validation
        """
        cv_folds = self.config['evaluation']['cross_validation_folds']
        
        print(f"\n=== CROSS-VALIDATION ({cv_folds} folds) ===")
        
        scores = cross_val_score(self.model, X, y, cv=cv_folds, scoring='accuracy')
        
        print(f"Cross-validation scores: {scores}")
        print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
    
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
    log_model = LogisticRegressionModel()
    
    # Prepare binary target
    df, threshold = log_model.prepare_binary_target(df, 'carbon_emissions')
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in ['carbon_emissions', 'emission_class', 'date']]
    X = df[feature_cols]
    y = df['emission_class']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    log_model.train(X_train, y_train)
    
    # Evaluate model
    results = log_model.evaluate(X_test, y_test)
    
    # Cross-validate
    log_model.cross_validate(X, y)
    
    # Save model
    log_model.save_model('models/saved_models/logistic_model.pkl')
