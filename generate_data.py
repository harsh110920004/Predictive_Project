import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed
np.random.seed(42)

# Parameters
n_records = 1000
start_date = datetime(2023, 1, 1)

# Generate dates
dates = [start_date + timedelta(days=i) for i in range(n_records)]

# Categories
vehicle_types = ['diesel_truck', 'electric_van', 'hybrid_truck']
transportation_modes = ['road', 'rail', 'air', 'sea']
product_categories = ['electronics', 'textiles', 'automotive', 'food', 'machinery', 
                      'pharmaceuticals', 'consumer_goods', 'chemicals']
regions = ['north', 'south', 'east', 'west']

# Generate features
data = {
    'date': dates,
    'transportation_distance': np.random.uniform(100, 1000, n_records),
    'fuel_consumption': np.random.uniform(50, 500, n_records),
    'production_volume': np.random.randint(500, 4000, n_records),
    'energy_usage': np.random.uniform(300, 2500, n_records),
    'warehouse_area': np.random.randint(3000, 12000, n_records),
    'num_suppliers': np.random.randint(4, 35, n_records),
    'vehicle_type': np.random.choice(vehicle_types, n_records),
    'transportation_mode': np.random.choice(transportation_modes, n_records),
    'product_category': np.random.choice(product_categories, n_records),
    'region': np.random.choice(regions, n_records)
}

df = pd.DataFrame(data)

# Calculate carbon emissions
def calculate_emissions(row):
    base_emission = (
        row['transportation_distance'] * 0.5 +
        row['fuel_consumption'] * 0.8 +
        row['production_volume'] * 0.05 +
        row['energy_usage'] * 0.15
    )
    
    vehicle_multipliers = {'diesel_truck': 1.2, 'electric_van': 0.6, 'hybrid_truck': 0.9}
    transport_multipliers = {'road': 1.0, 'rail': 0.7, 'air': 2.5, 'sea': 0.5}
    
    emission = base_emission * vehicle_multipliers[row['vehicle_type']] * \
               transport_multipliers[row['transportation_mode']]
    emission += np.random.normal(0, 20)
    
    return max(emission, 50)

df['carbon_emissions'] = df.apply(calculate_emissions, axis=1)

# Round values
df['transportation_distance'] = df['transportation_distance'].round(1)
df['fuel_consumption'] = df['fuel_consumption'].round(1)
df['energy_usage'] = df['energy_usage'].round(1)
df['carbon_emissions'] = df['carbon_emissions'].round(1)

# Save
df.to_csv('data/raw/supply_chain_emissions.csv', index=False)
print(f"✓ Generated {n_records} records")
print(f"✓ Saved to: data/raw/supply_chain_emissions.csv")
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Statistics:")
print(df.describe())
