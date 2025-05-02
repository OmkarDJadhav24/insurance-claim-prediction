import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib

def split_and_scale(input_path, output_dir, test_size=0.2, random_state=42):
    # Load the cleaned data
    df = pd.read_csv(input_path)

    # Split features and target
    X = df.drop(columns=["fraud_reported"])
    y = df["fraud_reported"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale numerical features only
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    
    # Save the scaler
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    
    # Save the column order of X_train
    joblib.dump(X_train.columns.tolist(), os.path.join(output_dir, "x_train_columns.pkl"))
    
    # Save the numeric columns for reference in scaling
    joblib.dump(numeric_cols.tolist(), os.path.join(output_dir, "scaler_numeric_columns.pkl"))

    print(f"✅ Data split and scaled. Files saved in: {output_dir}")

if __name__ == "__main__":
    split_and_scale(
        input_path="data/processed/cleaned_claims.csv",
        output_dir="data/processed"
    )
