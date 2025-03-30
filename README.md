Overview
This project focuses on analyzing depression risk among college students using machine learning techniques.
The analysis explores various factors such as sleep quality, financial stress and academic pressure.

Features
Data analysis and visualization of key factors contributing to depression.
Interpretation of patterns using machine learning techniques.
Develope model and deploy in web interface.

Dataset
The dataset used in this project is imported from Kaggle. Please ensure you download the appropriate dataset from Kaggle and place it in the project directory for the analysis to work correctly.

Installation
To avoid errors, please install the necessary libraries. You can do so by running the following command:
pip install -r requirements.txt
Note: Make sure to create a requirements.txt file in your project directory with the list of required libraries.

Usage
Clone this repository to your local machine.
Download the dataset from Kaggle and place it in the specified folder.
Install the required libraries using the command above.
Run the scripts to perform analysis and visualization.

License
This project is open source and available under the MIT License.

folder dictory:
website/
├── backend/
│   ├── server.py
│   ├── preprocessors/
│   │   ├── encoder.joblib
│   │   ├── scaler.joblib
│   └── models/
│       └── logistic_model.pkl
├── frontend/
│   ├── depression-prediction/
│       ├── node_modules/
│       ├── public/
│       ├── src/
│           ├── App.js
│           ├── index.js
│           └── components/
│               ├── PredictionForm.js
│               └── Results.js

