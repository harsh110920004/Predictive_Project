"""
Time Series Forecasting with ARIMA and SARIMA
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TIME SERIES FORECASTING - ARIMA & SARIMA")
print("="*70)

# Load data
print("\n[1/6] Loading time series data...")
df = pd.read_csv('data/processed/cleaned_emissions.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
df = df.set_index('date')

# Use only carbon emissions for time series
ts_data = df['carbon_emissions'].resample('D').mean()  # Daily aggregation
ts_data = ts_data.dropna()

print(f"✓ Time series length: {len(ts_data)} days")
print(f"✓ Date range: {ts_data.index.min()} to {ts_data.index.max()}")
print(f"✓ Mean emissions: {ts_data.mean():.2f}")
print(f"✓ Std emissions: {ts_data.std():.2f}")

# Stationarity test
print("\n[2/6] Testing stationarity (ADF Test)...")
adf_result = adfuller(ts_data)
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")

if adf_result[1] <= 0.05:
    print("✓ Series is STATIONARY (p-value ≤ 0.05)")
    d = 0
else:
    print("⚠ Series is NON-STATIONARY (p-value > 0.05)")
    print("  Applying first-order differencing...")
    d = 1

# Split data
print("\n[3/6] Splitting data...")
train_size = int(len(ts_data) * 0.8)
train_data = ts_data[:train_size]
test_data = ts_data[train_size:]

print(f"✓ Training set: {len(train_data)} days")
print(f"✓ Test set: {len(test_data)} days")

# Train ARIMA
print("\n[4/6] Training ARIMA model...")
print("Fitting ARIMA(1,1,1)...")

try:
    arima_model = ARIMA(train_data, order=(1, d, 1))
    arima_fitted = arima_model.fit()
    
    print("✓ ARIMA model fitted successfully")
    print(f"  AIC: {arima_fitted.aic:.2f}")
    print(f"  BIC: {arima_fitted.bic:.2f}")
    
    # Forecast
    arima_forecast = arima_fitted.forecast(steps=len(test_data))
    arima_rmse = np.sqrt(mean_squared_error(test_data, arima_forecast))
    arima_mae = mean_absolute_error(test_data, arima_forecast)
    
    print(f"  RMSE: {arima_rmse:.2f}")
    print(f"  MAE: {arima_mae:.2f}")
    
    # Save model
    arima_fitted.save('models/saved_models/arima_model.pkl')
    print("✓ ARIMA model saved")
    
except Exception as e:
    print(f"⚠ ARIMA training failed: {e}")
    arima_fitted = None
    arima_rmse = None

# Train SARIMA
print("\n[5/6] Training SARIMA model...")
print("Fitting SARIMA(1,1,1)x(1,1,1,7)...")

try:
    sarima_model = SARIMAX(train_data, order=(1, d, 1), seasonal_order=(1, 1, 1, 7))
    sarima_fitted = sarima_model.fit(disp=False)
    
    print("✓ SARIMA model fitted successfully")
    print(f"  AIC: {sarima_fitted.aic:.2f}")
    print(f"  BIC: {sarima_fitted.bic:.2f}")
    
    # Forecast
    sarima_forecast = sarima_fitted.forecast(steps=len(test_data))
    sarima_rmse = np.sqrt(mean_squared_error(test_data, sarima_forecast))
    sarima_mae = mean_absolute_error(test_data, sarima_forecast)
    
    print(f"  RMSE: {sarima_rmse:.2f}")
    print(f"  MAE: {sarima_mae:.2f}")
    
    # Save model
    sarima_fitted.save('models/saved_models/sarima_model.pkl')
    print("✓ SARIMA model saved")
    
except Exception as e:
    print(f"⚠ SARIMA training failed: {e}")
    sarima_fitted = None
    sarima_rmse = None

# Compare models
print("\n[6/6] Model comparison...")
print("\n" + "="*70)
print("TIME SERIES MODEL COMPARISON")
print("="*70)

if arima_fitted and sarima_fitted:
    comparison = pd.DataFrame({
        'Model': ['ARIMA(1,1,1)', 'SARIMA(1,1,1)x(1,1,1,7)'],
        'RMSE': [arima_rmse, sarima_rmse],
        'MAE': [arima_mae, sarima_mae],
        'AIC': [arima_fitted.aic, sarima_fitted.aic]
    })
    
    print(comparison.to_string(index=False))
    
    best_model = 'ARIMA' if arima_rmse < sarima_rmse else 'SARIMA'
    print(f"\n🏆 Best Model: {best_model}")

# Future forecast
print("\n" + "="*70)
print("30-DAY FUTURE FORECAST")
print("="*70)

if sarima_fitted:
    future_forecast = sarima_fitted.forecast(steps=30)
    print(f"Next 30 days forecast:")
    print(f"  Mean: {future_forecast.mean():.2f}")
    print(f"  Min: {future_forecast.min():.2f}")
    print(f"  Max: {future_forecast.max():.2f}")
    
    # Save forecast
    forecast_df = pd.DataFrame({
        'forecast_date': pd.date_range(start=ts_data.index[-1] + pd.Timedelta(days=1), periods=30),
        'forecasted_emissions': future_forecast
    })
    forecast_df.to_csv('models/saved_models/forecast_30days.csv', index=False)
    print("✓ 30-day forecast saved to: models/saved_models/forecast_30days.csv")

print("\n" + "="*70)
print("✓ TIME SERIES TRAINING COMPLETED SUCCESSFULLY!")
print("="*70)
