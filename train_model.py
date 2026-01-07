import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
import joblib

# Load data
df = pd.read_csv('product_data.csv')

# Features and target
categorical_features = ['product_type', 'packaging_type']
numerical_features = [
    'storage_temperature_celsius',
    'storage_humidity_percent',
    'initial_quality_score'
]
X_cat = df[categorical_features]
X_num = df[numerical_features]
y = df['shelf_life_days']

# One-hot encoding for categorical features
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat_encoded = encoder.fit_transform(X_cat)
encoded_feature_names = encoder.get_feature_names_out(categorical_features)

# Combine features
X_all = np.concatenate([X_num.values, X_cat_encoded], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, random_state=42
)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f'R-squared: {r2:.3f}')
print(f'Mean Absolute Error: {mae:.2f} days')

# Save model and encoder
joblib.dump(model, 'model.joblib')
joblib.dump(encoder, 'encoder.joblib')
print('Model and encoder saved as model.joblib and encoder.joblib')
