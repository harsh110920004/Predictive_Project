"""
Complete Flask Web Application for Carbon Footprint Prediction
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json
import os

app = Flask(__name__)
CORS(app)

print("="*70)
print("LOADING MODELS...")
print("="*70)

# Load all models
try:
    best_model = joblib.load('../models/saved_models/best_model.pkl')
    scaler = joblib.load('../models/saved_models/regression_scaler.pkl')
    label_encoders = joblib.load('../models/saved_models/regression_encoders.pkl')
    feature_names = joblib.load('../models/saved_models/regression_features.pkl')
    print("✓ Regression models loaded successfully")
    models_loaded = True
except Exception as e:
    print(f"⚠ Error loading models: {e}")
    models_loaded = False

# Load model comparison results
try:
    with open('../models/model_comparison_results.json', 'r') as f:
        comparison_results = json.load(f)
    print("✓ Model metrics loaded")
except:
    comparison_results = None

print("="*70)

@app.route('/')
def index():
    """Home page with prediction form"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard with model performance metrics"""
    return render_template('dashboard.html', results=comparison_results)

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for carbon emission prediction"""
    if not models_loaded:
        return jsonify({'success': False, 'error': 'Models not loaded'}), 500
    
    try:
        data = request.get_json()
        
        # Create input dataframe with all required features
        input_data = {
            'transportation_distance': float(data.get('transportation_distance', 0)),
            'fuel_consumption': float(data.get('fuel_consumption', 0)),
            'production_volume': float(data.get('production_volume', 0)),
            'energy_usage': float(data.get('energy_usage', 0)),
            'warehouse_area': float(data.get('warehouse_area', 5000)),
            'num_suppliers': int(data.get('num_suppliers', 10)),
            'vehicle_type': data.get('vehicle_type', 'diesel_truck'),
            'transportation_mode': data.get('transportation_mode', 'road'),
            'product_category': data.get('product_category', 'electronics'),
            'region': data.get('region', 'north')
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical variables
        for col in label_encoders.keys():
            if col in input_df.columns:
                le = label_encoders[col]
                try:
                    input_df[col] = le.transform([input_df[col].values[0]])
                except:
                    input_df[col] = 0  # Default encoding for unknown categories
        
        # Add missing features with zeros
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Reorder columns to match training
        input_df = input_df[feature_names]
        
        # Make prediction
        prediction = best_model.predict(input_df)[0]
        
        # Calculate emission level
        if prediction < 300:
            level = 'Low'
            color = 'green'
        elif prediction < 600:
            level = 'Medium'
            color = 'orange'
        else:
            level = 'High'
            color = 'red'
        
        return jsonify({
            'success': True,
            'predicted_emissions': float(prediction),
            'emission_level': level,
            'level_color': color,
            'unit': 'tons CO₂'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/statistics', methods=['GET'])
def statistics():
    """Get dataset and model statistics"""
    try:
        df = pd.read_csv('../data/processed/cleaned_emissions.csv')
        
        stats = {
            'dataset': {
                'total_records': int(len(df)),
                'mean_emissions': float(df['carbon_emissions'].mean()),
                'median_emissions': float(df['carbon_emissions'].median()),
                'std_emissions': float(df['carbon_emissions'].std()),
                'min_emissions': float(df['carbon_emissions'].min()),
                'max_emissions': float(df['carbon_emissions'].max())
            }
        }
        
        if comparison_results:
            stats['models'] = comparison_results
        
        return jsonify({'success': True, 'statistics': stats})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of trained models and their performance"""
    if comparison_results:
        return jsonify({'success': True, 'data': comparison_results})
    else:
        return jsonify({'success': False, 'error': 'No model data available'}), 404

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 STARTING FLASK APPLICATION")
    print("="*70)
    print("Access the application at: http://localhost:5000")
    print("API Documentation:")
    print("  • POST /api/predict - Make predictions")
    print("  • GET /api/statistics - Get dataset stats")
    print("  • GET /api/models - Get model comparison")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
