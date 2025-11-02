"""
Data Preprocessing Module for Carbon Footprint Prediction
Handles data cleaning, missing value imputation, and outlier detection
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy import stats
import yaml

class DataPreprocessor:
    """
    Class for preprocessing supply chain emissions data
    Implements techniques covered in CSE3141 syllabus
    """
    
    def __init__(self, config_path='config/config.yaml'):
        """Initialize preprocessor with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.scaler = None
        self.feature_names = None
        
    def load_data(self, filepath):
        """Load raw data from CSV file"""
        try:
            df = pd.read_csv(filepath)
            print(f"Data loaded successfully: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def explore_data(self, df):
        """
        Perform Exploratory Data Analysis (EDA)
        Calculate mean, median, mode, variance, standard deviation
        """
        print("\n=== DATA EXPLORATION ===")
        print(f"\nDataset Shape: {df.shape}")
        print(f"\nColumn Names: {df.columns.tolist()}")
        print(f"\nData Types:\n{df.dtypes}")
        print(f"\nMissing Values:\n{df.isnull().sum()}")
        
        # Statistical Summary
        print("\n=== STATISTICAL SUMMARY ===")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            print(f"\n{col}:")
            print(f"  Mean: {df[col].mean():.2f}")
            print(f"  Median: {df[col].median():.2f}")
            print(f"  Mode: {df[col].mode().values[0] if not df[col].mode().empty else 'N/A'}")
            print(f"  Variance: {df[col].var():.2f}")
            print(f"  Std Dev: {df[col].std():.2f}")
            print(f"  Min: {df[col].min():.2f}")
            print(f"  Max: {df[col].max():.2f}")
        
        return df.describe()
    
    def handle_missing_values(self, df):
        """
        Handle missing values using various strategies
        Methods: drop, mean, median, interpolation
        """
        method = self.config['preprocessing']['handle_missing']
        
        print(f"\n=== HANDLING MISSING VALUES (Method: {method}) ===")
        missing_before = df.isnull().sum().sum()
        
        if method == 'drop':
            df = df.dropna()
        elif method == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif method == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif method == 'interpolate':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
        
        missing_after = df.isnull().sum().sum()
        print(f"Missing values before: {missing_before}")
        print(f"Missing values after: {missing_after}")
        
        return df
    
    def detect_outliers_iqr(self, df, column):
        """
        Detect outliers using Interquartile Range (IQR) method
        As covered in CSE3141 syllabus
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        threshold = self.config['preprocessing']['outlier_threshold']
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        
        return outliers, lower_bound, upper_bound
    
    def detect_outliers_zscore(self, df, column, threshold=3):
        """
        Detect outliers using Z-score method
        As covered in CSE3141 syllabus
        """
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        outliers = df[z_scores > threshold]
        
        return outliers
    
    def handle_outliers(self, df):
        """
        Handle outliers in numerical columns
        Methods: IQR, Z-score
        """
        method = self.config['preprocessing']['outlier_method']
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        print(f"\n=== HANDLING OUTLIERS (Method: {method}) ===")
        
        for col in numeric_cols:
            if method == 'iqr':
                outliers, lower, upper = self.detect_outliers_iqr(df, col)
                print(f"{col}: {len(outliers)} outliers detected")
                # Cap outliers instead of removing
                df[col] = df[col].clip(lower=lower, upper=upper)
            elif method == 'zscore':
                outliers = self.detect_outliers_zscore(df, col)
                print(f"{col}: {len(outliers)} outliers detected")
                # Remove extreme outliers
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                df = df[z_scores < 3]
        
        return df
    
    def scale_features(self, df, columns=None):
        """
        Scale numerical features
        Methods: StandardScaler, MinMaxScaler, RobustScaler
        """
        method = self.config['preprocessing']['scaling_method']
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"\n=== SCALING FEATURES (Method: {method}) ===")
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        
        df[columns] = self.scaler.fit_transform(df[columns])
        self.feature_names = columns
        
        return df
    
    def preprocess_pipeline(self, df):
        """
        Complete preprocessing pipeline
        """
        print("\n" + "="*50)
        print("STARTING DATA PREPROCESSING PIPELINE")
        print("="*50)
        
        # Step 1: Explore data
        self.explore_data(df)
        
        # Step 2: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 3: Handle outliers
        df = self.handle_outliers(df)
        
        # Step 4: Data type conversions if needed
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        print("\n" + "="*50)
        print("PREPROCESSING COMPLETED")
        print("="*50)
        
        return df
    
    def save_processed_data(self, df, filepath):
        """Save processed data to CSV"""
        df.to_csv(filepath, index=False)
        print(f"\nProcessed data saved to: {filepath}")


# Example usage
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    
    # Load data
    df = preprocessor.load_data('data/raw/supply_chain_emissions.csv')
    
    # Preprocess data
    df_processed = preprocessor.preprocess_pipeline(df)
    
    # Save processed data
    preprocessor.save_processed_data(df_processed, 'data/processed/cleaned_emissions.csv')
