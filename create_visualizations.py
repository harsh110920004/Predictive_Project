"""
Generate Comprehensive Visualizations for Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("GENERATING PROJECT VISUALIZATIONS")
print("="*70)

# Load data
print("\n[1/5] Loading data...")
df = pd.read_csv('data/processed/cleaned_emissions.csv')
print(f"✓ Loaded {len(df)} records")

# Create output directory
import os
os.makedirs('reports/figures', exist_ok=True)

# 1. Target Distribution
print("\n[2/5] Creating target distribution plot...")
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(df['carbon_emissions'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Carbon Emissions (tons CO₂)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Carbon Emissions', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot(df['carbon_emissions'])
plt.ylabel('Carbon Emissions (tons CO₂)', fontsize=12)
plt.title('Box Plot - Outlier Detection', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reports/figures/1_target_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: reports/figures/1_target_distribution.png")
plt.close()

# 2. Correlation Heatmap
print("\n[3/5] Creating correlation heatmap...")
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('reports/figures/2_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: reports/figures/2_correlation_heatmap.png")
plt.close()

# 3. Feature vs Target Scatter Plots
print("\n[4/5] Creating feature scatter plots...")
features = ['transportation_distance', 'fuel_consumption', 'production_volume', 'energy_usage']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, feature in enumerate(features):
    axes[idx].scatter(df[feature], df['carbon_emissions'], alpha=0.5)
    axes[idx].set_xlabel(feature.replace('_', ' ').title(), fontsize=11)
    axes[idx].set_ylabel('Carbon Emissions', fontsize=11)
    axes[idx].set_title(f'Emissions vs {feature.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reports/figures/3_feature_scatter_plots.png', dpi=300, bbox_inches='tight')
print("✓ Saved: reports/figures/3_feature_scatter_plots.png")
plt.close()

# 4. Model Comparison
print("\n[5/5] Creating model comparison chart...")
try:
    import json
    with open('models/model_comparison_results.json', 'r') as f:
        results = json.load(f)
    
    comparison_df = pd.DataFrame(results['comparison'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # RMSE comparison
    axes[0].barh(comparison_df['Model'], comparison_df['RMSE'], color='skyblue')
    axes[0].set_xlabel('RMSE (Lower is Better)', fontsize=12)
    axes[0].set_title('Model Comparison - RMSE', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # R² comparison
    axes[1].barh(comparison_df['Model'], comparison_df['R²'], color='lightgreen')
    axes[1].set_xlabel('R² Score (Higher is Better)', fontsize=12)
    axes[1].set_title('Model Comparison - R² Score', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('reports/figures/4_model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: reports/figures/4_model_comparison.png")
    plt.close()
except:
    print("⚠ Model comparison results not found. Run train_model_comparison.py first.")

# 5. Time Series Plot
if 'date' in df.columns:
    print("\n[6/5] Creating time series plot...")
    df['date'] = pd.to_datetime(df['date'])
    df_sorted = df.sort_values('date')
    
    plt.figure(figsize=(14, 6))
    plt.plot(df_sorted['date'], df_sorted['carbon_emissions'], linewidth=1.5, alpha=0.7)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Carbon Emissions (tons CO₂)', fontsize=12)
    plt.title('Carbon Emissions Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('reports/figures/5_time_series.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: reports/figures/5_time_series.png")
    plt.close()

print("\n" + "="*70)
print("✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("="*70)
print("\n📁 Generated files:")
print("   • reports/figures/1_target_distribution.png")
print("   • reports/figures/2_correlation_heatmap.png")
print("   • reports/figures/3_feature_scatter_plots.png")
print("   • reports/figures/4_model_comparison.png")
print("   • reports/figures/5_time_series.png")
