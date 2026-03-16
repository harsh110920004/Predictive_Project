"""
Complete Flask Web Application for Carbon Footprint Prediction
Production Ready Version
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json
import os

# ---------------------------------------------------
# Initialize Flask App
# ---------------------------------------------------

app = Flask(__name__)
CORS(app)

print("=" * 70)
print("LOADING MODELS...")
print("=" * 70)

# ---------------------------------------------------
# Define Base Directory
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "..", "models", "saved_models")
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")

# ---------------------------------------------------
# Load Machine Learning Models
# ---------------------------------------------------

try:

    best_model = joblib.load(
        os.path.join(MODEL_DIR, "best_model.pkl")
    )

    scaler = joblib.load(
        os.path.join(MODEL_DIR, "regression_scaler.pkl")
    )

    label_encoders = joblib.load(
        os.path.join(MODEL_DIR, "regression_encoders.pkl")
    )

    feature_names = joblib.load(
        os.path.join(MODEL_DIR, "regression_features.pkl")
    )

    print("✓ Regression models loaded successfully")

    models_loaded = True

except Exception as e:

    print("⚠ Error loading models:", str(e))

    models_loaded = False


# ---------------------------------------------------
# Load Model Comparison Results
# ---------------------------------------------------

try:

    comparison_file = os.path.join(
        BASE_DIR, "..", "models", "model_comparison_results.json"
    )

    with open(comparison_file, "r") as f:

        comparison_results = json.load(f)

    print("✓ Model metrics loaded")

except Exception:

    comparison_results = None


print("=" * 70)

# ---------------------------------------------------
# Home Page
# ---------------------------------------------------


@app.route("/")
def index():
    """Home page with prediction form"""
    return render_template("index.html")


# ---------------------------------------------------
# Dashboard Page
# ---------------------------------------------------


@app.route("/dashboard")
def dashboard():
    """Dashboard with model performance metrics"""
    return render_template("dashboard.html", results=comparison_results)


# ---------------------------------------------------
# Prediction API
# ---------------------------------------------------


@app.route("/api/predict", methods=["POST"])
def predict():

    if not models_loaded:

        return jsonify({
            "success": False,
            "error": "Models failed to load"
        }), 500

    try:

        data = request.get_json()

        # ---------------------------------------------------
        # Prepare Input Data
        # ---------------------------------------------------

        input_data = {
            "transportation_distance": float(data.get("transportation_distance", 0)),
            "fuel_consumption": float(data.get("fuel_consumption", 0)),
            "production_volume": float(data.get("production_volume", 0)),
            "energy_usage": float(data.get("energy_usage", 0)),
            "warehouse_area": float(data.get("warehouse_area", 5000)),
            "num_suppliers": int(data.get("num_suppliers", 10)),
            "vehicle_type": data.get("vehicle_type", "diesel_truck"),
            "transportation_mode": data.get("transportation_mode", "road"),
            "product_category": data.get("product_category", "electronics"),
            "region": data.get("region", "north")
        }

        input_df = pd.DataFrame([input_data])

        # ---------------------------------------------------
        # Encode Categorical Variables
        # ---------------------------------------------------

        for col in label_encoders.keys():

            if col in input_df.columns:

                encoder = label_encoders[col]

                try:
                    input_df[col] = encoder.transform(
                        [input_df[col].values[0]]
                    )

                except Exception:
                    input_df[col] = 0

        # ---------------------------------------------------
        # Add Missing Features
        # ---------------------------------------------------

        for feature in feature_names:

            if feature not in input_df.columns:

                input_df[feature] = 0

        # ---------------------------------------------------
        # Correct Column Order
        # ---------------------------------------------------

        input_df = input_df[feature_names]

        # ---------------------------------------------------
        # Prediction
        # ---------------------------------------------------

        prediction = best_model.predict(input_df)[0]

        # ---------------------------------------------------
        # Emission Level Classification
        # ---------------------------------------------------

        if prediction < 300:
            level = "Low"
            color = "green"

        elif prediction < 600:
            level = "Medium"
            color = "orange"

        else:
            level = "High"
            color = "red"

        return jsonify({

            "success": True,
            "predicted_emissions": float(prediction),
            "emission_level": level,
            "level_color": color,
            "unit": "tons CO₂"

        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


# ---------------------------------------------------
# Dataset Statistics API
# ---------------------------------------------------


@app.route("/api/statistics", methods=["GET"])
def statistics():

    try:

        data_path = os.path.join(
            DATA_DIR,
            "cleaned_emissions.csv"
        )

        df = pd.read_csv(data_path)

        stats = {

            "dataset": {

                "total_records": int(len(df)),

                "mean_emissions": float(
                    df["carbon_emissions"].mean()
                ),

                "median_emissions": float(
                    df["carbon_emissions"].median()
                ),

                "std_emissions": float(
                    df["carbon_emissions"].std()
                ),

                "min_emissions": float(
                    df["carbon_emissions"].min()
                ),

                "max_emissions": float(
                    df["carbon_emissions"].max()
                )

            }

        }

        if comparison_results:
            stats["models"] = comparison_results

        return jsonify({
            "success": True,
            "statistics": stats
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ---------------------------------------------------
# Get Model Performance
# ---------------------------------------------------


@app.route("/api/models", methods=["GET"])
def get_models():

    if comparison_results:

        return jsonify({
            "success": True,
            "data": comparison_results
        })

    return jsonify({
        "success": False,
        "error": "No model data available"
    }), 404


# ---------------------------------------------------
# Run Flask App
# ---------------------------------------------------


if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("🚀 STARTING FLASK APPLICATION")
    print("=" * 70)

    print("Access the application at:")
    print("http://localhost:5000")

    print("\nAvailable APIs:")

    print("POST  /api/predict")
    print("GET   /api/statistics")
    print("GET   /api/models")

    print("=" * 70 + "\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )