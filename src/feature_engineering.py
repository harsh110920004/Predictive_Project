"""
Feature Engineering Module
Creates time-based features, lag features, and rolling statistics
"""

import numpy as np
import pandas as pd
import yaml

class FeatureEngineer:
    """
    Feature engineering for carbon footprint prediction
    Implements techniques for improved model performance
    """
    
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def create_time_features(self, df, date_column='date'):
        """
        Create time-based features from date column
        Features: year, month, day, day_of_week, quarter, is_weekend
        """
        if not self.config['feature_engineering']['create_time_features']:
            return df
        
        print("\n=== CREATING TIME FEATURES ===")
        
        df[date_column] = pd.to_datetime(df[date_column])
        
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['day'] = df[date_column].dt.day
        df['day_of_week'] = df[date_column].dt.dayofweek
        df['quarter'] = df[date_column].dt.quarter
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['week_of_year'] = df[date_column].dt.isocalendar().week
        
        # Cyclical encoding for month and day_of_week
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        print(f"Created time features: {['year', 'month', 'day', 'day_of_week', 'quarter', 'is_weekend']}")
        
        return df
    
    def create_lag_features(self, df, target_column, lag_periods=None):
        """
        Create lag features for time series prediction
        Lag periods: 1, 3, 7, 30 days
        """
        if not self.config['feature_engineering']['create_lag_features']:
            return df
        
        if lag_periods is None:
            lag_periods = self.config['feature_engineering']['lag_periods']
        
        print(f"\n=== CREATING LAG FEATURES (Periods: {lag_periods}) ===")
        
        for lag in lag_periods:
            df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
        
        return df
    
    def create_rolling_features(self, df, target_column, windows=None):
        """
        Create rolling window statistics
        Windows: 7, 14, 30 days
        Statistics: mean, std, min, max
        """
        if not self.config['feature_engineering']['create_rolling_features']:
            return df
        
        if windows is None:
            windows = self.config['feature_engineering']['rolling_windows']
        
        print(f"\n=== CREATING ROLLING FEATURES (Windows: {windows}) ===")
        
        for window in windows:
            df[f'{target_column}_rolling_mean_{window}'] = df[target_column].rolling(window=window).mean()
            df[f'{target_column}_rolling_std_{window}'] = df[target_column].rolling(window=window).std()
            df[f'{target_column}_rolling_min_{window}'] = df[target_column].rolling(window=window).min()
            df[f'{target_column}_rolling_max_{window}'] = df[target_column].rolling(window=window).max()
        
        return df
    
    def create_interaction_features(self, df, feature_pairs):
        """
        Create interaction features between specified feature pairs
        """
        print("\n=== CREATING INTERACTION FEATURES ===")
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                print(f"Created interaction: {feat1}_x_{feat2}")
        
        return df
    
    def create_domain_features(self, df):
        """
        Create domain-specific features for supply chain carbon emissions
        """
        print("\n=== CREATING DOMAIN-SPECIFIC FEATURES ===")
        
        # Example domain features
        if 'transportation_distance' in df.columns and 'fuel_consumption' in df.columns:
            df['emission_intensity'] = df['carbon_emissions'] / (df['transportation_distance'] + 1)
            df['fuel_efficiency'] = df['transportation_distance'] / (df['fuel_consumption'] + 1)
        
        if 'production_volume' in df.columns:
            df['emission_per_unit'] = df['carbon_emissions'] / (df['production_volume'] + 1)
        
        return df
    
    def feature_selection(self, df, target_column, threshold=0.05):
        """
        Perform feature selection based on correlation with target
        """
        print("\n=== FEATURE SELECTION ===")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        correlations = df[numeric_cols].corrwith(df[target_column]).abs()
        selected_features = correlations[correlations > threshold].index.tolist()
        
        print(f"Selected {len(selected_features)} features with correlation > {threshold}")
        
        return selected_features
    
    def engineer_features_pipeline(self, df, target_column='carbon_emissions'):
        """
        Complete feature engineering pipeline
        """
        print("\n" + "="*50)
        print("STARTING FEATURE ENGINEERING PIPELINE")
        print("="*50)
        
        # Create time features
        if 'date' in df.columns:
            df = self.create_time_features(df)
        
        # Create lag features
        df = self.create_lag_features(df, target_column)
        
        # Create rolling features
        df = self.create_rolling_features(df, target_column)
        
        # Create domain features
        df = self.create_domain_features(df)
        
        # Drop rows with NaN created by lag/rolling features
        df = df.dropna()
        
        print("\n" + "="*50)
        print("FEATURE ENGINEERING COMPLETED")
        print(f"Final feature count: {len(df.columns)}")
        print("="*50)
        
        return df


# Example usage
if __name__ == "__main__":
    engineer = FeatureEngineer()
    
    # Load processed data
    df = pd.read_csv('data/processed/cleaned_emissions.csv')
    
    # Engineer features
    df_engineered = engineer.engineer_features_pipeline(df)
    
    # Save
    df_engineered.to_csv('data/processed/engineered_features.csv', index=False)
