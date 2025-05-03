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



Flask API Setup

Endpoints:

1. Preprocessing Data (/preprocess)
    Triggers the data_preprocessing.py script to clean the raw data and then runs the split_and_scale.py script to preprocess the data.
    Method: POST
    Response: 200 OK upon success, with a message indicating completion.


2. Train Model (/train)
    Triggers the train_model.py script to train the logistic regression model.
    Method: POST
    Response: 200 OK upon success, with a message indicating the model has been trained.


3. Make Predictions (/predict)
    This endpoint allows you to upload a new claim file and returns fraud predictions based on the trained model.
    Method: POST
    Request: A file (CSV) with new claims data.
    Response: 200 OK with a message indicating the predictions and output file location.



How to Run the API:
1. Flask Installation: Ensure you have Flask installed:
    pip install flask

2. Start the Flask API Server: Run the Flask application to start the API server:
    python app.py


The server will be available at http://127.0.0.1:5000/


Example API Requests:

1. Preprocessing:
    URL: http://127.0.0.1:5000/preprocess
    Method: POST
    Response: "Data preprocessing and scaling completed successfully!"

2. Training:
    URL: http://127.0.0.1:5000/train
    Method: POST
    Response: "Model trained successfully!"

3. Prediction:
    URL: http://127.0.0.1:5000/predict
    Method: POST
    Body: Form-data with file (CSV file)
    Response: "Predictions completed successfully!", with a link to the output file.



Folder Structure:

insurance-claim-prediction/
├── src/
│   ├── data_preprocessing.py
│   ├── split_and_scale.py
│   ├── train_model.py
│   ├── infer_model.py
├── app.py (Flask API)
├── data/
├── models/
└── README.md