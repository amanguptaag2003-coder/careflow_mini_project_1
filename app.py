from flask import Flask, render_template, request
import pandas as pd
import pickle


# Create Flask application
app = Flask(__name__)


# Load trained machine learning model
model = pickle.load(
    open("model/cancer_model.pkl", "rb")
)


# Load label encoder
label_encoder = pickle.load(
    open("model/label_encoder.pkl", "rb")
)


# Home page
@app.route("/")
def home():

    return render_template("index.html")


# Prediction page
@app.route("/predict", methods=["POST"])
def predict():

    # Get values from HTML form
    age = int(request.form["age"])
    gender = int(request.form["gender"])

    air_pollution = int(request.form["air_pollution"])
    alcohol_use = int(request.form["alcohol_use"])
    dust_allergy = int(request.form["dust_allergy"])
    occupational_hazards = int(request.form["occupational_hazards"])
    genetic_risk = int(request.form["genetic_risk"])
    chronic_lung_disease = int(request.form["chronic_lung_disease"])

    balanced_diet = int(request.form["balanced_diet"])
    obesity = int(request.form["obesity"])
    smoking = int(request.form["smoking"])
    passive_smoker = int(request.form["passive_smoker"])

    chest_pain = int(request.form["chest_pain"])
    coughing_of_blood = int(request.form["coughing_of_blood"])
    fatigue = int(request.form["fatigue"])
    weight_loss = int(request.form["weight_loss"])

    shortness_of_breath = int(
        request.form["shortness_of_breath"]
    )

    wheezing = int(request.form["wheezing"])

    swallowing_difficulty = int(
        request.form["swallowing_difficulty"]
    )

    clubbing_of_finger_nails = int(
        request.form["clubbing_of_finger_nails"]
    )

    frequent_cold = int(
        request.form["frequent_cold"]
    )

    dry_cough = int(
        request.form["dry_cough"]
    )

    snoring = int(
        request.form["snoring"]
    )


    # Create input DataFrame
    input_data = pd.DataFrame([[
        age,
        gender,
        air_pollution,
        alcohol_use,
        dust_allergy,
        occupational_hazards,
        genetic_risk,
        chronic_lung_disease,
        balanced_diet,
        obesity,
        smoking,
        passive_smoker,
        chest_pain,
        coughing_of_blood,
        fatigue,
        weight_loss,
        shortness_of_breath,
        wheezing,
        swallowing_difficulty,
        clubbing_of_finger_nails,
        frequent_cold,
        dry_cough,
        snoring
    ]])


    # Make prediction
    prediction = model.predict(input_data)


    # Convert numerical prediction back to text
    result = label_encoder.inverse_transform(prediction)[0]


    # Send result to result.html
    return render_template(
        "result.html",
        prediction=result
    )


# Run Flask application
if __name__ == "__main__":

    app.run(debug=True)