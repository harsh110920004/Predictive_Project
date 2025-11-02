"""
Visualization Module
Creates comprehensive plots for EDA and model evaluation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class Visualizer:
    """
    Create visualizations for carbon footprint prediction project
    Implements various plot types as per CSE3141 syllabus
    """
    
    def __init__(self, style='seaborn', figsize=(12, 6), dpi=100):
        plt.style.use(style)
        self.figsize = figsize
        self.dpi = dpi
    
    def plot_histogram(self, data, column, bins=30, title=None):
        """
        Create histogram for data distribution
        As covered in CSE3141 syllabus
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.hist(data[column], bins=bins, edgecolor='black', alpha=0.7)
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.title(title or f'Distribution of {column}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt
    
    def plot_scatter(self, data, x_col, y_col, title=None):
        """
        Create scatter plot
        As covered in CSE3141 syllabus
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.scatter(data[x_col], data[y_col], alpha=0.6)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(title or f'{y_col} vs {x_col}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt
    
    def plot_heatmap(self, correlation_matrix, title='Correlation Heatmap'):
        """
        Create correlation heatmap
        As covered in CSE3141 syllabus
        """
        plt.figure(figsize=(14, 10), dpi=self.dpi)
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1)
        plt.title(title)
        plt.tight_layout()
        return plt
    
    def plot_time_series(self, data, date_col, value_col, title=None):
        """
        Create time series plot
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.plot(data[date_col], data[value_col], linewidth=2)
        plt.xlabel('Date')
        plt.ylabel(value_col)
        plt.title(title or f'Time Series: {value_col}')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt
    
    def plot_box_plot(self, data, column, title=None):
        """
        Create box plot for outlier detection
        As covered in CSE3141 syllabus
        """
        plt.figure(figsize=(8, 6), dpi=self.dpi)
        plt.boxplot(data[column].dropna())
        plt.ylabel(column)
        plt.title(title or f'Box Plot: {column}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt
    
    def plot_predictions_vs_actual(self, y_true, y_pred, title='Predictions vs Actual'):
        """
        Plot predicted vs actual values
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt
    
    def plot_residuals(self, y_true, y_pred, title='Residual Plot'):
        """
        Plot residuals for regression analysis
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=self.dpi)
        
        # Residual scatter plot
        axes[0].scatter(y_pred, residuals, alpha=0.6)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residual Plot')
        axes[0].grid(True, alpha=0.3)
        
        # Residual histogram
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residual Distribution')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return plt
    
    def plot_feature_importance(self, feature_names, importances, top_n=15):
        """
        Plot feature importance for tree-based models
        """
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8), dpi=self.dpi)
        plt.barh(range(top_n), importances[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        return plt
    
    def plot_learning_curve(self, train_sizes, train_scores, test_scores):
        """
        Plot learning curve
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.plot(train_sizes, train_scores, 'o-', label='Training Score')
        plt.plot(train_sizes, test_scores, 'o-', label='Testing Score')
        plt.xlabel('Training Set Size')
        plt.ylabel('Mean Squared Error')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt
    
    def save_plot(self, plt_obj, filepath):
        """
        Save plot to file
        """
        plt_obj.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        print(f"Plot saved to: {filepath}")
