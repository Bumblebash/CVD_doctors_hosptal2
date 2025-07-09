from flask import Flask, request, jsonify
import joblib  # Or pickle, depending on how you saved your ML model
import numpy as np
#
app = Flask(__name__)

# Load your trained model
model = joblib.load("XGBClassifier_model.pkl")  # Ensure your model file is in the project directory

@app.route("/")
def home():
    return jsonify({"message": "Flask API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()  # Expecting JSON input
        features = np.array(data["features"]).reshape(1, -1)  # Convert input to NumPy array
        prediction = model.predict(features)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
