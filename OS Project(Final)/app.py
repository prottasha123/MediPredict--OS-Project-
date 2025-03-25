from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load trained models and scaler
lung_cancer_model = pickle.load(open("lung_cancer_model.pkl", "rb"))
diabetes_model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)

# Homepage route
@app.route('/')
@app.route('/home')
def home():
    return render_template("home.html")

@app.route('/health_info')
def health_info():
    return render_template("health_info.html")

@app.route('/disease_prediction')
def disease_prediction():
    return render_template("disease_prediction.html")

@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/cancer')
def cancer():
    return render_template("cancer.html")

@app.route('/lung')
def lung():
    return render_template("lung.html")

@app.route('/diabetes')
def diabetes():
    return render_template("diabetes.html")

# Diabetes Prediction Route
@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    try:
        # Get form data
        pregnancies = float(request.form["pregnancies"])
        glucose = float(request.form["glucose"])
        blood_pressure = float(request.form["blood_pressure"])
        skin_thickness = float(request.form["skin_thickness"])
        insulin = float(request.form["insulin"])
        bmi = float(request.form["bmi"])
        pedigree_function = float(request.form["diabetes_pedigree_function"])
        age = float(request.form["age"])

        # Prepare the data (reshape and scale)
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree_function, age]])
        features_scaled = scaler.transform(features)  # Apply scaling

        # Make prediction
        prediction = diabetes_model.predict(features_scaled)[0]

        # Format prediction result
        prediction_text = "The patient is likely to have diabetes." if prediction == 1 else "The patient is likely not to have diabetes."

        return render_template("diabetes.html", prediction_text=prediction_text)
    except Exception as e:
        return jsonify({'error': str(e)})

# Lung Cancer Prediction Route
@app.route('/predict_lung', methods=['POST'])
def predict_lung():
    try:
        # Get form data for lung cancer prediction
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        smoke = int(request.form['smoke'])
        yellow_fingers = int(request.form['yellow_fingers'])
        anxiety = int(request.form['anxiety'])
        peer_pressure = int(request.form['peer_pressure'])
        chronic_disease = int(request.form['chronic_disease'])
        fatigue = int(request.form['fatigue'])
        allergy = int(request.form['allergy'])
        wheezing = int(request.form['wheezing'])
        alcohol = int(request.form['alcohol'])
        coughing = int(request.form['coughing'])
        shortness_of_breath = int(request.form['shortness_of_breath'])
        swallowing_difficulty = int(request.form['swallowing_difficulty'])
        chest_pain = int(request.form['chest_pain'])

        # Prepare input data for lung cancer prediction
        features = np.array([[age, gender, smoke, yellow_fingers, anxiety, peer_pressure,
                              chronic_disease, fatigue, allergy, wheezing, alcohol, coughing,
                              shortness_of_breath, swallowing_difficulty, chest_pain]])

        # Predict lung cancer risk
        prediction = lung_cancer_model.predict(features)[0]
        result = "Positive" if prediction == 1 else "Negative"

        return render_template("lung.html", prediction_text=f'Lung Cancer Prediction: {result}')
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == "__main__":
    app.run(debug=True, port=5001)
