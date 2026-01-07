import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Define possible values for categorical features
product_types = ['Dairy', 'Bakery', 'Beverage', 'Snack']
packaging_types = ['Plastic', 'Glass', 'Cardboard', 'Can']

# Define base shelf life for each product type (in days)
base_shelf_life = {
    'Dairy': 14,
    'Bakery': 7,
    'Beverage': 90,
    'Snack': 120
}

# Define packaging bonus (in days)
packaging_bonus = {
    'Plastic': 0,
    'Glass': 10,
    'Cardboard': -3,
    'Can': 7
}

data = []
for i in range(1000):
    product_id = i + 1
    product_type = np.random.choice(product_types)
    storage_temperature = np.round(np.random.uniform(2, 35), 1)  # Celsius
    storage_humidity = np.round(np.random.uniform(20, 90), 1)    # Percent
    packaging_type = np.random.choice(packaging_types)
    initial_quality_score = np.random.randint(1, 11)

    # Calculate shelf life
    shelf_life = base_shelf_life[product_type]
    shelf_life -= (storage_temperature - 4) * 0.7  # Higher temp reduces shelf life
    shelf_life -= (storage_humidity - 40) * 0.15   # Higher humidity reduces shelf life
    shelf_life += packaging_bonus[packaging_type]
    shelf_life += (initial_quality_score - 5) * 0.8  # Better initial quality increases shelf life
    shelf_life += np.random.normal(0, 2)  # Add random noise
    shelf_life = max(1, int(round(shelf_life)))  # Shelf life can't be less than 1 day

    data.append([
        product_id,
        product_type,
        storage_temperature,
        storage_humidity,
        packaging_type,
        initial_quality_score,
        shelf_life
    ])

columns = [
    'product_id',
    'product_type',
    'storage_temperature_celsius',
    'storage_humidity_percent',
    'packaging_type',
    'initial_quality_score',
    'shelf_life_days'
]

df = pd.DataFrame(data, columns=columns)
df.to_csv('product_data.csv', index=False)

print('Synthetic product data generated and saved to product_data.csv')
