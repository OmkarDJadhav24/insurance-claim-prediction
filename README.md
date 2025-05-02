Insurance Claims Fraud Prediction

This project predicts insurance fraud using machine learning. It contains scripts for data preprocessing, model training, and inference, which can be run both manually and via an API.


Scripts Overview:

1. Data Preprocessing (data_preprocessing.py)
    This script performs the following operations:

* Cleans the raw claims data by removing unnecessary columns.
* Fills missing values.
* Encodes categorical variables.
* Handles outliers using the IQR method.
* Saves the cleaned data to ../../data/processed/cleaned_claims.csv.

How to Run:
    python src/data_preprocessing.py


2. Data Splitting and Scaling (split_and_scale.py)
    This script performs the following operations:

* Splits the data into training and testing sets.
* Scales the features using StandardScaler.

How to Run:
    python src/split_and_scale.py



3. Model Training (train_model.py)
    This script trains a logistic regression model using the preprocessed data and evaluates it using classification metrics (accuracy, F1 score, etc.).

How to Run:
    python src/train_model.py


4. Inference (infer_model.py)
    This script predicts whether a new insurance claim is fraudulent based on a trained model and processed input data.

How to Run:
    python src/infer_model.py