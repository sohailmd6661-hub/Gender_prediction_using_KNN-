from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


app = Flask(__name__)

# Load model and scaler
with open("breast_cancer.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Collect form inputs
            features = [
                float(request.form["Long hair"]),
                float(request.form["forehead_width_cm"]),
                float(request.form["forehead_height_cm"]),
                float(request.form["nose_wide"]),
                float(request.form["nose_long"]),
                float(request.form["lips_thin"]),
                float(request.form["distance_nose_to_lip_long"]),

            ]

            # Convert to numpy array
            features_array = np.array([features])


            # Make prediction
            prediction = model.predict(features_array)[0]
            if prediction == 0:
                return render_template("index.html", prediction='Gender:Male')
            else:
                return render_template("index.html", prediction="Gender:Female")

        except Exception as e:
            # If an error occurs during POST, render the template with the error message
            prediction = f"Error: {str(e)}"
            return render_template("index.html", prediction=prediction) # <--- Added return here for error handling

    # This handles the initial GET request (when the page is loaded)
    # and any scenario where the POST block didn't execute or encountered an error.
    return render_template("index.html", prediction=prediction) # <--- ADDED REQUIRED RETURN STATEMENT

if __name__ == "__main__":
    app.run(debug=True)