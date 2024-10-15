from flask import Flask, render_template, request, jsonify
from prediction import load_trained_model, preprocess_input, predict_diabetes
import numpy as np
import warnings

app = Flask(__name__)

# Load model and scaler
loaded_model, scaler = load_trained_model('Model_3_0.001_Bn_relu_.4_0.00001_40_512_LaReDrLrEpBs.h5', 'scaler.pkl')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/pred')
def index():
    return render_template('predictor.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/bmi')
def bmi():
    return render_template('bmi.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    user_input = []
    for key in ['age', 'gender', 'bloodGlucose', 'hyperTension', 'bmi', 'smokingHistory', 'hba1cLevel', 'heartDisease']:
        user_input.append(float(request.form[key]))
    
    # Preprocess user input
    user_input_np = np.array([user_input])
    
    # Suppress the specific warning related to feature names
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
        user_input_scaled = preprocess_input(user_input_np, scaler)
    
    # Predict diabetes
    prediction_percentage = predict_diabetes(loaded_model, user_input_scaled)
    
    # Return prediction as JSON
    return jsonify({'prediction': prediction_percentage})

if __name__ == '__main__':
    app.run(debug=True)
