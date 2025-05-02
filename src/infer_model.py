import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Paths
MODEL_PATH = "models/logistic_model.pkl"
SCALER_PATH = "data/processed/scaler.pkl"
INPUT_PATH = "data/raw_data/new_claim.csv"
OUTPUT_PATH = "data/predictions/predictions.csv"

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Load files
new_data = pd.read_csv(INPUT_PATH)
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Preprocessing same as training
cols_to_drop = ['policy_number', 'policy_bind_date', 'incident_location', 
                'incident_date', 'auto_model', '_c39']
new_data.drop(columns=cols_to_drop, inplace=True, errors='ignore')
new_data.ffill(inplace=True)

# Identify categorical columns
categorical_cols = new_data.select_dtypes(include='object').columns.tolist()
one_hot_cols = [col for col in categorical_cols if new_data[col].nunique() <= 10]
high_card_cols = [col for col in categorical_cols if col not in one_hot_cols]

# One-hot encoding
new_data = pd.get_dummies(new_data, columns=one_hot_cols, drop_first=True)

# Label encoding for high cardinality columns
le = LabelEncoder()
for col in high_card_cols:
    if new_data[col].nunique() > 0:
        new_data[col] = le.fit_transform(new_data[col])

# Align columns with training data (fix any column mismatch)
X_train_columns = joblib.load("data/processed/x_train_columns.pkl")
for col in X_train_columns:
    if col not in new_data.columns:
        new_data[col] = 0       
new_data = new_data[X_train_columns]

# Load column names used during training scaling
scaler_numeric_cols = joblib.load("data/processed/scaler_numeric_columns.pkl")

# Ensure all those columns exist in new_data
missing_scaler_cols = [col for col in scaler_numeric_cols if col not in new_data.columns]
if missing_scaler_cols:
    for col in missing_scaler_cols:
        new_data[col] = 0  # Add missing columns with zeros

new_data[scaler_numeric_cols] = scaler.transform(new_data[scaler_numeric_cols])

# Predict
predictions = model.predict(new_data)

# Save predictions
output_df = pd.DataFrame(predictions, columns=["fraud_prediction"])
output_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Predictions saved to: {OUTPUT_PATH}")
