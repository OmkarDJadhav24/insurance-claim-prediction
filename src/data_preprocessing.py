import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os


def handle_outliers_iqr(df, numeric_cols):
    """Clip outliers using the IQR method."""
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)
    return df

def preprocess_data(input_path, output_path):
    """Preprocess data, handling missing values, encoding, and outliers."""

    """
    In below code we are going to use one hot encoding and Label encoding for numerical columns.
    One hot encoding is for columns which have <=10 non-NA unique values, because one hot encoding creates new columns for each unique value of that column.
    Label encoding is for columns which have >10 non-NA values, because if we use one hot encoding here it will create multiple new columns as per unique value which will lead to high memory usage and also because of multiple columns models accuracy may decrease.
    """
    # Load the raw data
    df = pd.read_csv(input_path)

    # Drop unnecessary columns
    drop_cols = ['policy_number', 'policy_bind_date', 'incident_location', 
                 'incident_date', 'auto_model', '_c39']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Forward fill missing values
    df.ffill(inplace=True)

    # Convert the target variable 'fraud_reported' to binary (1, 0)
    df['fraud_reported'] = df['fraud_reported'].map({'Y': 1, 'N': 0})

    # One-hot encoding for low-cardinality categorical features
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    one_hot_cols = [col for col in categorical_cols if df[col].nunique() <= 10]
    df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)

    # Label encoding for high-cardinality categorical features
    high_card_cols = [col for col in categorical_cols if col not in one_hot_cols]
    le = LabelEncoder()
    for col in high_card_cols:
        if df[col].nunique() > 0:
            df[col] = le.fit_transform(df[col])

    # Handle outliers using IQR for numeric columns
    cols_to_clip = [
        'months_as_customer', 'age', 'policy_annual_premium',
        'umbrella_limit', 'capital-gains', 'capital-loss',
        'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim'
    ]
    df = handle_outliers_iqr(df, cols_to_clip)

    # Save the preprocessed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Preprocessed data saved to: {output_path}")

# Execute preprocessing on the raw data
if __name__ == "__main__":
    preprocess_data("data/raw_data/insurance_fraud_claims.csv", "data/processed/cleaned_claims.csv")
