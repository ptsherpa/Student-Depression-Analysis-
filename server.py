from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Paths to the saved model and preprocessors
model_path = 'C:\\Users\\DEVIL\\Desktop\\6th project\\model\\website\\backend\\logistic_model.pkl'
encoder_path = 'C:\\Users\\DEVIL\\Desktop\\6th project\\model\\website\\backend\\encoder.pkl'
scaler_path = 'C:\\Users\\DEVIL\\Desktop\\6th project\\model\\website\\backend\\scaler.pkl'


# Load the logistic regression model and preprocessors
logistic_model = joblib.load(model_path)
encoder = joblib.load(encoder_path)
scaler = joblib.load(scaler_path)

# Explicitly set feature names for the encoder
encoder.feature_names_in_ = ['gender', 'sleepDuration', 'dietaryHabits', 'degree', 'suicidalThoughts', 'familyHistory']

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Log incoming request
        print("Received request:", request.json)

        # Input received from the frontend
        data = request.json

        # Extract features from the input
        categorical_inputs = [[
            data.get("gender"),
            data.get("sleepDuration"),
            data.get("dietaryHabits"),
            data.get("degree"),
            data.get("suicidalThoughts"),
            data.get("familyHistory")
        ]]

        numerical_inputs = [[
            float(data.get("age", 0)),
            float(data.get("academicPressure", 0)),
            float(data.get("cgpa", 0)),
            float(data.get("studySatisfaction", 0)),
            float(data.get("workStudyHours", 0))
        ]]

        # Transform inputs
        categorical_data = encoder.transform(categorical_inputs)
        numerical_data = scaler.transform(numerical_inputs)

        # Combine features
        processed_features = np.hstack((categorical_data, numerical_data))

        # Log processed features
        print("Processed features:", processed_features)

        # Make prediction
        prediction = logistic_model.predict(processed_features)[0]
        prediction_probability = logistic_model.predict_proba(processed_features)[0][1]

        # Log and return the prediction
        print("Prediction:", prediction, "Probability:", prediction_probability)
        return jsonify({'prediction': int(prediction), 'probability': prediction_probability})

    except Exception as e:
        # Log the error
        print("Error:", e)
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
