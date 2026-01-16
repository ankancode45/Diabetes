from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = ""
    probability_text = ""

    if request.method == "POST":
        try:
            pregnancies = float(request.form["Pregnancies"])
            glucose = float(request.form["Glucose"])
            blood_pressure = float(request.form["BloodPressure"])
            skin_thickness = float(request.form["SkinThickness"])
            insulin = float(request.form["Insulin"])
            bmi = float(request.form["BMI"])
            dpf = float(request.form["DiabetesPedigreeFunction"])
            age = float(request.form["Age"])

            input_data = np.array([[
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                dpf,
                age
            ]])

            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            if prediction == 1:
                prediction_text = "Diabetes Detected"
            else:
                prediction_text = "No Diabetes Detected"

            probability_text = f"Diabetes Probability: {probability * 100:.2f}%"

        except Exception as e:
            prediction_text = "Invalid input values"
            probability_text = str(e)

    return render_template(
        "index.html",
        prediction=prediction_text,
        probability=probability_text
    )

if __name__ == "__main__":
    app.run(debug=True)
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

