import pandas as pd # pandas is not strictly needed for the app, but often imported for ML projects
from flask import Flask, render_template, request
import pickle
import os # For path checking and debugging

app = Flask(__name__)

# Define the path to your model file
MODEL_PATH = 'NEWS.pkl' # Make sure this path is correct relative to app.py

# Load the trained pipeline from the pickle file
# This will be attempted only once when the Flask app starts
try:
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
    print(f"Model (pipeline) loaded successfully from {MODEL_PATH}.")
except FileNotFoundError:
    print(f"Error: {MODEL_PATH} not found. Ensure it's in the same directory as app.py and was created by your training script.")
    model = None # Set model to None if loading fails
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    print("This might happen if the .pkl file is corrupted or saved incorrectly.")
    model = None # Set model to None if loading fails

@app.route('/')
def home():
    # Pass a message to the template if the model failed to load during app startup
    if model is None:
        return render_template('index.html', prediction_text="Server Error: Model could not be loaded. Please contact support.", result_class="fake")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Before processing, check if the model object was successfully loaded
    if model is None:
        return render_template('index.html', prediction_text="Prediction not possible: Model is not loaded. Check server logs.", result_class="fake")

    if request.method == 'POST':
        news_text = request.form['news_text']

        if not news_text:
            return render_template('index.html', prediction_text="Please enter some news text to analyze.", result_class="not-fake") # No input isn't "fake" or "not fake"

        try:
            # The 'model' object, being a scikit-learn Pipeline, correctly handles
            # both the transformation (vectorization) and the classification.
            # You pass the raw text string directly to model.predict().
            prediction = model.predict([news_text])[0]

            # Assuming 0 for Fake and 1 for Not Fake (adjust if your model uses different labels)
            if prediction == 0:
                result_text = "Fake News"
                result_class = "fake" # For styling
            else:
                result_text = "Not Fake News"
                result_class = "not-fake" # For styling

            return render_template('index.html', prediction_text=f"The news is: {result_text}", result_class=result_class)
        except Exception as e:
            # Catch any error during prediction (e.g., if input format is unexpected by model, though less likely with Pipeline)
            print(f"Error during prediction: {e}") # Log the error on the server side
            return render_template('index.html', prediction_text=f"An internal error occurred during prediction. Please try again. ({e})", result_class="fake")

if __name__ == '__main__':
    # Add a pre-check to inform user if model file is missing before running Flask
    if not os.path.exists(MODEL_PATH):
        print(f"\nWARNING: Model file '{MODEL_PATH}' not found. Please ensure your training script has created it in the correct directory.")
        print("The Flask application will start, but predictions will fail until the model file is present.")
    app.run(debug=True) # debug=True is good for development, but set to False in production