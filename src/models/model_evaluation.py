"""
Model Evaluation Module
Implements comprehensive model validation and optimization techniques
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
)
import json
import yaml

class ModelEvaluator:
    """
    Comprehensive model evaluation and optimization
    As per CSE3141 syllabus: validation, metrics, hyperparameter tuning
    """
    
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.results = {}
    
    def cross_validation(self, model, X, y, cv=5):
        """
        Perform k-fold cross-validation
        As covered in CSE3141 syllabus
        """
        print(f"\n=== {cv}-FOLD CROSS-VALIDATION ===")
        
        # For regression models
        if hasattr(model, 'predict') and not hasattr(model, 'predict_proba'):
            scoring = 'neg_mean_squared_error'
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            scores = -scores  # Convert negative MSE to positive
            
            print(f"Cross-validation MSE scores: {scores}")
            print(f"Mean MSE: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        # For classification models
        else:
            scoring = 'accuracy'
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            
            print(f"Cross-validation accuracy scores: {scores}")
            print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
    
    def calculate_regression_metrics(self, y_true, y_pred):
        """
        Calculate comprehensive regression metrics
        Metrics: RMSE, MAE, R², MAPE
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
        
        return metrics
    
    def calculate_classification_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate comprehensive classification metrics
        Metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }
        
        if y_pred_proba is not None:
            auc = roc_auc_score(y_true, y_pred_proba)
            metrics['AUC-ROC'] = auc
        
        return metrics
    
    def print_metrics(self, metrics, model_name="Model"):
        """
        Print evaluation metrics in formatted manner
        """
        print(f"\n=== {model_name.upper()} PERFORMANCE METRICS ===")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
    
    def grid_search_hyperparameters(self, model, param_grid, X_train, y_train, cv=5):
        """
        Hyperparameter tuning using Grid Search
        As covered in CSE3141 syllabus
        """
        print("\n=== GRID SEARCH HYPERPARAMETER TUNING ===")
        print(f"Parameter grid: {param_grid}")
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best score: {-grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def random_search_hyperparameters(self, model, param_distributions, X_train, y_train, 
                                     n_iter=50, cv=5):
        """
        Hyperparameter tuning using Random Search
        As covered in CSE3141 syllabus
        """
        print("\n=== RANDOM SEARCH HYPERPARAMETER TUNING ===")
        print(f"Parameter distributions: {param_distributions}")
        print(f"Number of iterations: {n_iter}")
        
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1,
            random_state=self.config['data']['random_state']
        )
        
        random_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {random_search.best_params_}")
        print(f"Best score: {-random_search.best_score_:.4f}")
        
        return random_search.best_estimator_, random_search.best_params_
    
    def detect_overfitting_underfitting(self, train_metrics, test_metrics):
        """
        Detect overfitting and underfitting
        As covered in CSE3141 syllabus
        """
        print("\n=== OVERFITTING/UNDERFITTING ANALYSIS ===")
        
        if 'R2' in train_metrics and 'R2' in test_metrics:
            train_r2 = train_metrics['R2']
            test_r2 = test_metrics['R2']
            
            print(f"Training R²: {train_r2:.4f}")
            print(f"Testing R²: {test_r2:.4f}")
            print(f"Difference: {abs(train_r2 - test_r2):.4f}")
            
            if train_r2 > 0.9 and test_r2 < 0.7:
                print("\n⚠️  WARNING: Likely OVERFITTING")
                print("Recommendations:")
                print("  - Reduce model complexity")
                print("  - Increase training data")
                print("  - Apply regularization")
                print("  - Use cross-validation")
            elif train_r2 < 0.6 and test_r2 < 0.6:
                print("\n⚠️  WARNING: Likely UNDERFITTING")
                print("Recommendations:")
                print("  - Increase model complexity")
                print("  - Add more features")
                print("  - Reduce regularization")
                print("  - Train longer")
            else:
                print("\n✓ Model appears well-fitted")
    
    def learning_curve_analysis(self, model, X, y, train_sizes=np.linspace(0.1, 1.0, 10)):
        """
        Generate learning curve to analyze model performance vs training size
        """
        from sklearn.model_selection import learning_curve
        
        print("\n=== LEARNING CURVE ANALYSIS ===")
        
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        train_scores_mean = -np.mean(train_scores, axis=1)
        test_scores_mean = -np.mean(test_scores, axis=1)
        
        print("Training sizes:", train_sizes)
        print("Training scores:", train_scores_mean)
        print("Testing scores:", test_scores_mean)
        
        return train_sizes, train_scores_mean, test_scores_mean
    
    def compare_models(self, models_dict, X_test, y_test):
        """
        Compare multiple models and rank by performance
        """
        print("\n=== MODEL COMPARISON ===")
        
        comparison_results = []
        
        for model_name, model in models_dict.items():
            y_pred = model.predict(X_test)
            metrics = self.calculate_regression_metrics(y_test, y_pred)
            metrics['Model'] = model_name
            comparison_results.append(metrics)
        
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.sort_values('RMSE')
        
        print("\n" + "="*70)
        print(comparison_df.to_string(index=False))
        print("="*70)
        
        return comparison_df
    
    def save_results(self, results, filepath='models/model_metrics.json'):
        """
        Save evaluation results to JSON file
        """
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nResults saved to: {filepath}")


# Example usage
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    
    # Load data
    df = pd.read_csv('data/processed/engineered_features.csv')
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in ['carbon_emissions', 'date']]
    X = df[feature_cols]
    y = df['carbon_emissions']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Train models
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Evaluate models
    models = {
        'Random Forest': rf_model,
        'Linear Regression': lr_model
    }
    
    comparison = evaluator.compare_models(models, X_test, y_test)
    
    # Cross-validation
    evaluator.cross_validation(rf_model, X, y)
    
    # Hyperparameter tuning example
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None]
    }
    
    best_model, best_params = evaluator.grid_search_hyperparameters(
        RandomForestRegressor(random_state=42),
        param_grid,
        X_train,
        y_train
    )
