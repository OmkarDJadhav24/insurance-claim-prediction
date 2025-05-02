from flask import Flask, jsonify, request
import os
import subprocess

app = Flask(__name__)

# Ensure the directories exist
os.makedirs("../data/processed", exist_ok=True)
os.makedirs("../models", exist_ok=True)
os.makedirs("../data/raw_data", exist_ok=True)
os.makedirs("../data/predictions", exist_ok=True)

@app.route('/preprocess', methods=['POST'])
def preprocess():
    try:
        # Step 1: Run the data preprocessing script (data_preprocessing.py)
        subprocess.run(["python", "src/data_preprocessing.py"], check=True)

        # Step 2: Run the split and scale script (split_and_scale.py)
        subprocess.run(["python", "src/split_and_scale.py"], check=True)
        
        return jsonify({"message": "Data preprocessing and scaling completed successfully!"}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    try:
        # Run the training script (train_model.py)
        subprocess.run(["python", "src/train_model.py"], check=True)
        return jsonify({"message": "Model trained successfully!"}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input file path from the request
        file = request.files['file']
        input_path = "data/raw_data/new_claim.csv"
        file.save(input_path)
        
        # Run the inference script (infer_model.py)
        subprocess.run(["python", "src/infer_model.py"], check=True)
        
        # Provide the output file path as the result
        output_file = "data/predictions/predictions.csv"
        return jsonify({"message": "Predictions completed successfully!", "output_file": output_file}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
