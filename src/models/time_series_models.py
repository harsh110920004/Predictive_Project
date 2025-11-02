"""
Time Series Analysis Module
Implements ARIMA, SARIMA for carbon emissions forecasting
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
import warnings
warnings.filterwarnings('ignore')
import joblib
import yaml

class TimeSeriesModels:
    """
    Time series forecasting for carbon emissions
    Implements ARIMA and SARIMA as per CSE3141 syllabus
    """
    
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.arima_model = None
        self.sarima_model = None
    
    def check_stationarity(self, timeseries):
        """
        Test stationarity using Augmented Dickey-Fuller test
        """
        print("\n=== STATIONARITY TEST (ADF) ===")
        
        result = adfuller(timeseries.dropna())
        
        print(f'ADF Statistic: {result[0]:.6f}')
        print(f'p-value: {result[1]:.6f}')
        print(f'Critical Values:')
        for key, value in result[4].items():
            print(f'\t{key}: {value:.3f}')
        
        if result[1] <= 0.05:
            print("\nResult: Series is STATIONARY")
            return True
        else:
            print("\nResult: Series is NON-STATIONARY")
            return False
    
    def decompose_series(self, timeseries, period=12):
        """
        Decompose time series into trend, seasonality, and noise
        As covered in CSE3141 syllabus
        """
        print(f"\n=== TIME SERIES DECOMPOSITION (Period={period}) ===")
        
        decomposition = seasonal_decompose(timeseries, model='additive', period=period)
        
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        
        print("Decomposition completed:")
        print(f"  - Trend component")
        print(f"  - Seasonal component")
        print(f"  - Residual (noise) component")
        
        return decomposition
    
    def find_best_arima_order(self, timeseries):
        """
        Find optimal ARIMA order (p,d,q) using AIC criterion
        """
        print("\n=== FINDING OPTIMAL ARIMA ORDER ===")
        
        best_aic = np.inf
        best_order = None
        
        # Grid search over p, d, q
        for p in range(0, 3):
            for d in range(0, 2):
                for q in range(0, 3):
                    try:
                        model = ARIMA(timeseries, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        print(f"Best ARIMA order: {best_order}")
        print(f"Best AIC: {best_aic:.2f}")
        
        return best_order
    
    def fit_arima(self, timeseries, order=None):
        """
        Fit ARIMA model for time series forecasting
        ARIMA(p, d, q) where:
        - p: autoregressive order
        - d: differencing order
        - q: moving average order
        """
        if order is None:
            order = tuple(self.config['models']['arima']['order'])
        
        print(f"\n=== FITTING ARIMA{order} MODEL ===")
        
        self.arima_model = ARIMA(timeseries, order=order)
        self.arima_fitted = self.arima_model.fit()
        
        print("\nModel Summary:")
        print(self.arima_fitted.summary())
        
        return self.arima_fitted
    
    def fit_sarima(self, timeseries, order=None, seasonal_order=None):
        """
        Fit SARIMA model for seasonal time series forecasting
        SARIMA(p,d,q)(P,D,Q,s) where:
        - (p,d,q): non-seasonal parameters
        - (P,D,Q,s): seasonal parameters
        """
        if order is None:
            order = tuple(self.config['models']['arima']['order'])
        if seasonal_order is None:
            seasonal_order = tuple(self.config['models']['arima']['seasonal_order'])
        
        print(f"\n=== FITTING SARIMA{order}x{seasonal_order} MODEL ===")
        
        self.sarima_model = SARIMAX(
            timeseries,
            order=order,
            seasonal_order=seasonal_order
        )
        self.sarima_fitted = self.sarima_model.fit(disp=False)
        
        print("\nModel Summary:")
        print(self.sarima_fitted.summary())
        
        return self.sarima_fitted
    
    def forecast(self, model, steps=30):
        """
        Generate forecasts for future time periods
        """
        print(f"\n=== FORECASTING {steps} STEPS AHEAD ===")
        
        forecast = model.forecast(steps=steps)
        forecast_df = pd.DataFrame({
            'forecast': forecast
        })
        
        print(f"\nForecast statistics:")
        print(forecast_df.describe())
        
        return forecast_df
    
    def evaluate_forecast(self, actual, predicted):
        """
        Evaluate forecast accuracy
        Metrics: RMSE, MAE, MAPE
        """
        print("\n=== FORECAST EVALUATION ===")
        
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mae = np.mean(np.abs(actual - predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
    
    def exponential_smoothing(self, timeseries, alpha=0.3):
        """
        Simple exponential smoothing for forecasting
        As covered in CSE3141 syllabus
        """
        print(f"\n=== EXPONENTIAL SMOOTHING (alpha={alpha}) ===")
        
        result = [timeseries.iloc[0]]
        for i in range(1, len(timeseries)):
            result.append(alpha * timeseries.iloc[i] + (1 - alpha) * result[i-1])
        
        smoothed = pd.Series(result, index=timeseries.index)
        
        return smoothed
    
    def save_model(self, model, filepath):
        """Save fitted time series model"""
        model.save(filepath)
        print(f"\nModel saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load saved time series model"""
        from statsmodels.iolib.smpickle import load_pickle
        model = load_pickle(filepath)
        print(f"\nModel loaded from: {filepath}")
        return model


# Example usage
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/processed/cleaned_emissions.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    # Get time series
    timeseries = df['carbon_emissions']
    
    # Initialize time series models
    ts_models = TimeSeriesModels()
    
    # Check stationarity
    ts_models.check_stationarity(timeseries)
    
    # Decompose series
    decomposition = ts_models.decompose_series(timeseries)
    
    # Fit ARIMA
    arima_fitted = ts_models.fit_arima(timeseries)
    
    # Fit SARIMA
    sarima_fitted = ts_models.fit_sarima(timeseries)
    
    # Forecast
    forecast = ts_models.forecast(sarima_fitted, steps=30)
    
    # Save model
    ts_models.save_model(sarima_fitted, 'models/saved_models/sarima_model.pkl')
