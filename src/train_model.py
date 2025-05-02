import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

# Paths
DATA_DIR = "data/processed/"
MODEL_DIR = "models/"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_evaluate():
    # Load the data
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

    # Train Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Accuracy for test data: ", model.score(X_test, y_test))
    print("✅ Classification Report:\n", classification_report(y_test, y_pred))
    print("✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Cross-validation (optional, extra insight)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    print(f"✅ 5-Fold CV F1 Score: {scores.mean():.4f}")

    # Save the trained model
    joblib.dump(model, os.path.join(MODEL_DIR, "logistic_model.pkl"))
    print(f"✅ Model saved to: {os.path.join(MODEL_DIR, 'logistic_model.pkl')}")

if __name__ == "__main__":
    train_and_evaluate()
