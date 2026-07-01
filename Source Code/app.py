import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/findyourcrop")
def findyourcrop():
    return render_template("findyourcrop.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input
        N = float(request.form["nitrogen"])
        P = float(request.form["phosphorous"])
        K = float(request.form["potassium"])

        # Apply the same transformation used during training
        K = np.log1p(K)

        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        # Create input dataframe
        input_data = pd.DataFrame(
            [[N, P, K, temperature, humidity, ph, rainfall]],
            columns=[
                "N",
                "P",
                "K",
                "temperature",
                "humidity",
                "ph",
                "rainfall"
            ]
        )

        # Predict
        prediction = model.predict(input_data)

        return render_template(
            "findyourcrop.html",
            prediction_text=f"Recommended Crop: {prediction[0]}"
        )

    except Exception as e:
        return render_template(
            "findyourcrop.html",
            prediction_text=f"Error: {e}"
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
