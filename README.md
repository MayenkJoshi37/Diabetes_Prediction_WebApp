# 🩺 Diabetes Prediction Web App

A Flask‑based app that estimates a user's risk of type‑2 diabetes based on 8 health markers. Trained using Keras with dropout/L2 regularization and pipelined using `model.py` and `prediction.py`, it provides a percentage-based prediction in real time.

---

## 🎯 Purpose

Allows users to enter:
- **age**
- **gender**
- **bloodGlucose level**
- **hyperTension status**
- **BMI**
- **smokingHistory**
- **HbA1c level**
- **heartDisease history**

and receive an estimated **% diabetes risk**, powered by a trained neural network. Designed for health awareness and screening purposes—not as medical advice.

Similar Flask apps (e.g., logistic‑regression or ensemble‑based) have used the Pima Indians Diabetes Dataset for similar purposes :contentReference[oaicite:0]{index=0}.

---

## 🧰 Tech Stack

- **Flask** — Web framework serving multiple routes and JSON prediction API  
- **Keras / TensorFlow** — Training and loading of the neural network  
- **scikit-learn / joblib** — Feature scaling via `StandardScaler`  
- **NumPy**, **Pandas** — Data loading & manipulation  
- **HTML** templates (`index.html`, `predictor.html`, `about.html`, `bmi.html`) as frontend

---

## ⚙️ Setup & Run

```bash
git clone https://github.com/MayenkJoshi37/Diabetes_Prediction_WebApp.git
cd Diabetes_Prediction_WebApp
pip install -r requirements.txt
python model.py                     # Optional—trains the model
python app.py
```

## Access the interface at:

```bash
http://127.0.0.1:5000/home
```


---

## 🗂️ Project Structure
```bash
.
├─ app.py                 # Flask app routing and predict API
├─ model.py               # Neural network training script
├─ prediction.py          # Pre-trained model inference
├─ App/
│   ├─ Final_Dataset.xlsx # Dataset for training (8 input features + Outcome)
│   ├─ scaler.pkl         # Saved scaler for input normalization
│   └─ [model].h5         # Saved Keras model file
├─ templates/             # HTML templates for various pages
├─ requirements.txt
└─ README.md              # This file
```



## 🚀 Core Endpoints

| Endpoint   | Method | Purpose                                          |
|------------|--------|--------------------------------------------------|
| `/home`    | GET    | Homepage (`index.html`)                          |
| `/pred`    | GET    | Predict form input page (`predictor.html`)       |
| `/about`   | GET    | About the app (`about.html`)                     |
| `/bmi`     | GET    | BMI trend page (`bmi.html`, optional)            |
| `/predict` | POST   | Receives health inputs → returns JSON risk %     |


## 🧠 How It Works

1. **Training (`model.py`)**  
   - Load `Final_Dataset.xlsx` containing 8 features + Outcome.  
   - Split into train/test with `train_test_split`.  
   - Scale features with pre-saved scaler.  
   - Build a neural network:  
     - 64 → 32 dense units with ReLU  
     - BatchNormalization → Dropout(0.4) → L2 regularization (l2=0.001)  
     - Sigmoid output for binary classification (% risk computed as prediction × 100)  
   - Trained with Adam (`learning_rate=1e-5`), `batch_size=512`, `epochs=40`.  
   - Dropout and L2 combat overfitting, following best practices in diabetes NN modeling.

2. **Inference (`prediction.py`)**  
   - `load_trained_model(model_path, scaler_path)` loads the `.h5` model and `scaler.pkl`.  
   - `preprocess_input(user_array, scaler)` normalizes raw user input.  
   - `predict_diabetes(model, input_scaled)` returns a % risk estimate.

3. **Interactive API (`app.py`)**  
   - Handles requests via the `/predict` POST route.  
   - Uses NumPy to format 8 input values into an array.  
   - Suppresses warnings about feature names during scaling.  
   - Returns a JSON response with the prediction percentage.

---

## 💬 Response Style

- Only returns JSON for seamless frontend integration.  
- `predict_diabetes` outputs a number between 0.0–100.0 representing % risk.

---

## 🛠️ Customization Suggestions

- Introduce additional features (e.g., insulin levels, pedigree function) for finer accuracy using Pima dataset insights.  
- Switch the neural architecture to tree-based models (XGBoost, Random Forest) for enhanced interpretability.  
- Add input validation/UX checks for missing or invalid data.  
- Improve the frontend with charts or confidence bands.  
- Deploy using Docker, Heroku, or Flask–Gunicorn in production settings.

---

## ⚠️ Caution & Medical Disclaimer

🚫 **Not medical advice**: Informational only.  
Predictions rely entirely on user-entered data and model training.  
Health data is sensitive—implement HTTPS and data encryption in any deployment.  
It is strongly advised to include a disclaimer, for example:

> _“This tool is for informational purposes only and is not a substitute for professional medical diagnosis or treatment.”_

---

## ⚖️ Contributions & License

📝 The source code is available under an open-source license (e.g., MIT, Apache 2.0) of your choosing.

**Contributing ideas:**
- Add public diabetes datasets (e.g., Pima Indian dataset variants).  
- Implement interpretability tools like SHAP or LIME to explain predictions.  
- Extend with mobile or offline UI support.

---

## ✅ Quick Summary

- `model.py`: trains and saves a dropout-regularized neural network model.  
- `prediction.py`: loads saved model and scaler to make percentage-based diabetes predictions.  
- `app.py`: Flask web app with routes for home, prediction form, and `/predict` JSON API.  
- Provides a structured workflow: dataset ingestion → model training → input scaling → online inference.  

