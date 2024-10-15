# prediction.py

from keras.models import load_model
import joblib
import numpy as np

def load_trained_model(model_path, scaler_path):
    # Load the trained model
    loaded_model = load_model(model_path)
    
    # Load scaling parameters
    scaler = joblib.load(scaler_path)
    scaler.feature_names = ['age', 'gender', 'bloodGlucose', 'hyperTension', 'bmi', 'smokingHistory', 'hba1cLevel', 'heartDisease']

    return loaded_model, scaler

def preprocess_input(input_data, scaler):
    # Preprocess input data using the scaler
    input_scaled = scaler.transform(input_data)
    return input_scaled

def predict_diabetes(model, input_scaled):
    # Make predictions using the loaded model
    prediction = model.predict(input_scaled)[0][0] * 100
    return prediction

# Example function to get user input
def get_user_input():

    # Gather user input for various health parameters


    # Format user input into a list
    user_input = [[0, 0, 0, 0, 0, 0, 0, 0]]
    return np.array(user_input)

# Load model and scaler
loaded_model, scaler = load_trained_model('Model_3_0.001_Bn_relu_.4_0.00001_40_512_LaReDrLrEpBs.h5', 'scaler.pkl')

# Get user input and preprocess it
user_input = get_user_input()
user_input_scaled = preprocess_input(user_input, scaler)

# Predict diabetes based on user input and print the prediction percentage
prediction_percentage = predict_diabetes(loaded_model, user_input_scaled)
print("Diabetes Prediction Percentage:", prediction_percentage)
