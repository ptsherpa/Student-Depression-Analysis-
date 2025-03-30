**Overview**
```
This project focuses on analyzing depression risk among college students using machine learning techniques.
The analysis explores various factors such as sleep quality, financial stress and academic pressure.
```
#Features
```
Data analysis and visualization of key factors contributing to depression.
Interpretation of patterns using machine learning techniques.
Develope model and deploy in web interface.
```
#Dataset
```
The dataset used in this project is imported from Kaggle. Please ensure you download the appropriate dataset from Kaggle and place it in the project directory for the analysis to work correctly.
```
#Installation
```
To avoid errors, please install the necessary libraries. You can do so by running the following command:
pip install -r requirements.txt
Note: Make sure to create a requirements.txt file in your project directory with the list of required libraries.
```
#Usage
```
Clone this repository to your local machine.
Download the dataset from Kaggle and place it in the specified folder.
Install the required libraries using the command above.
Run the scripts to perform analysis and visualization.
```
#License
```
This project is open source and available under the MIT License.
```
#folder dictory:
```
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
```
**Steps to Run Your Project**
```
**1. Run the Backend (Flask Server)**
Prerequisites:
Ensure Python and pip are properly installed.

Install required dependencies (Flask, Flask-CORS, scikit-learn, joblib, etc.).

Steps:
Navigate to the Backend Directory:
cd "C:\Users\Desktop\model\website\backend"

Run the Flask Server:
python server.py

If successful, you’ll see something like:
Running on http://127.0.0.1:5000/
Keep the Flask Server Running:

Leave the terminal open to keep the server running while you work with the frontend.

**2. Run the Frontend (React Application)**

Prerequisites:
Node.js and npm should be installed.

Ensure React dependencies are installed.
Steps:

Navigate to the Frontend Directory:
cd "C:\Users\Desktop\model\website\frontend\depression-prediction"

Install Dependencies (if not already installed):
npm install

Start the React Development Server:
npm start

This should open a browser window with your React app running at:
http://localhost:3000/

3. Test the Application
Access the Frontend:
Open your browser and go to http://localhost:3000.
Submit a Prediction Request:
Fill out the form with valid inputs and submit.
Check the Backend Logs:

In the Flask terminal, you should see the processed request and the prediction result.
```
