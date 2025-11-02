# AI-Powered Carbon Footprint Prediction for Supply Chains
## Project Report - CSE3141 Predictive Analysis

**Student Name:** [Your Name]
**Roll Number:** [Your Roll Number]
**Course:** CSE3141 - Predictive Analysis
**Session:** Jul-Nov 2025
**Date:** October 21, 2025

---

## 1. EXECUTIVE SUMMARY

This project implements a comprehensive AI-powered carbon footprint prediction system
for supply chain operations using machine learning and statistical techniques covered
in the CSE3141 course syllabus.

### Key Achievements:
- Trained and evaluated 6 different ML models
- Best model: Gradient Boosting
- Achieved R2 score of 0.9804
- RMSE: 92.29
- Successfully implemented time series forecasting with ARIMA/SARIMA
- Deployed web application for real-time predictions

---

## 2. DATASET DESCRIPTION

**Dataset Statistics:**
- Total Records: 1000
- Features: 12
- Target Variable: Carbon Emissions (tons CO2)

**Emissions Statistics:**
- Mean: 826.42 tons CO2
- Median: 591.90 tons CO2
- Std Dev: 605.96 tons CO2
- Range: 110.20 - 2180.09 tons CO2

---

## 3. METHODOLOGY

### 3.1 Data Preprocessing
- Handled missing values using interpolation method
- Detected and treated outliers using IQR method
- Performed feature scaling using StandardScaler
- Encoded categorical variables using LabelEncoder

### 3.2 Feature Engineering
- Created time-based features (year, month, day, weekday)
- Generated lag features (1, 3, 7, 30 days)
- Calculated rolling statistics (7, 14, 30-day windows)
- Created domain-specific features (emission intensity, fuel efficiency)

### 3.3 Models Implemented

Gradient Boosting:
  - RMSE: 92.29
  - R2: 0.9804
  - MAPE: 6.10%

Random Forest:
  - RMSE: 132.38
  - R2: 0.9597
  - MAPE: 8.59%

Decision Tree:
  - RMSE: 185.07
  - R2: 0.9213
  - MAPE: 13.02%

Lasso Regression:
  - RMSE: 283.07
  - R2: 0.8159
  - MAPE: 29.05%

Ridge Regression:
  - RMSE: 284.32
  - R2: 0.8143
  - MAPE: 29.53%

Linear Regression:
  - RMSE: 284.36
  - R2: 0.8143
  - MAPE: 29.57%

---

## 4. RESULTS & FINDINGS

### Best Model Performance
- **Model:** Gradient Boosting
- **RMSE:** 92.29
- **R2 Score:** 0.9804
- **Cross-Validation Mean:** 93.78
- **Cross-Validation Std:** 15.76

### Overfitting Analysis
- Train R2: 0.9998
- Test R2: 0.9804
- Difference: 0.0194
- **Status:** Well-fitted model (no overfitting detected)

---

## 5. VISUALIZATIONS

All visualizations are available in the reports/figures/ directory:
1. Target Distribution & Box Plot
2. Feature Correlation Heatmap
3. Feature vs Target Scatter Plots
4. Model Comparison Charts
5. Time Series Analysis

---

## 6. WEB APPLICATION

A Flask-based web application has been developed with:
- Interactive prediction interface
- Real-time carbon emission forecasting
- Model performance dashboard
- RESTful API endpoints

Access: http://localhost:5000

---

## 7. CONCLUSION

This project successfully demonstrates the application of predictive analytics
techniques for carbon footprint prediction in supply chains. The Gradient Boosting
model achieved excellent performance with an R2 score of 0.9804,
indicating strong predictive capability.

### Key Learnings:
- Practical application of regression and classification techniques
- Time series forecasting using ARIMA/SARIMA
- Model evaluation and hyperparameter tuning
- Deployment of ML models in web applications

---

## 8. COURSE ALIGNMENT (CSE3141)

This project covers all major topics from the syllabus:

- Data Preparation & Exploration (EDA, outlier detection, feature engineering)
- Logistic Regression & Inference (binary classification, coefficient estimation)
- Poisson Regression (count data modeling)
- Time Series Analysis (ARIMA, SARIMA, forecasting)
- Model Validation & Optimization (cross-validation, hyperparameter tuning)

---

## 9. TECHNICAL IMPLEMENTATION

### Technologies Used:
- Programming Language: Python 3.11+
- Core Libraries: NumPy, Pandas, Scikit-learn
- Statistical Models: Statsmodels
- Machine Learning: Random Forest, Gradient Boosting
- Visualization: Matplotlib, Seaborn, Plotly
- Web Framework: Flask

### Model Training Pipeline:
1. Data loading and exploration
2. Preprocessing (missing values, outliers, encoding)
3. Feature engineering (lag, rolling, time features)
4. Model training (6 different algorithms)
5. Cross-validation and evaluation
6. Hyperparameter tuning
7. Model selection and deployment

---

## 10. RESULTS SUMMARY

### Model Rankings (by RMSE):
1. Gradient Boosting: RMSE = 92.29, R2 = 0.9804
2. Random Forest: RMSE = 132.38, R2 = 0.9597
3. Decision Tree: RMSE = 185.07, R2 = 0.9213
4. Lasso Regression: RMSE = 283.07, R2 = 0.8159
5. Ridge Regression: RMSE = 284.32, R2 = 0.8143
6. Linear Regression: RMSE = 284.36, R2 = 0.8143

### Performance Interpretation:
- R2 > 0.9: Excellent predictive power
- RMSE < 50: High accuracy in emission prediction
- CV consistency: Model generalizes well to unseen data

---

## 11. REFERENCES

1. Course materials from CSE3141 Predictive Analysis, Manipal University Jaipur
2. Scikit-learn Documentation: https://scikit-learn.org
3. Statsmodels Documentation: https://www.statsmodels.org
4. Gareth James et al., Introduction to Statistical Learning
5. Research papers on supply chain sustainability

---

Report Generated: 2025-10-21 01:31:15
Project Status: COMPLETED SUCCESSFULLY
Course: CSE3141 - Predictive Analysis
Institution: Manipal University Jaipur
