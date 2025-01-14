from flask import Flask, request, jsonify
import joblib  # For loading the model
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load(r"C:\Users\user\Projects\Sales-Forecast-For-Rossmann-Pharmaceuticals\model\sales_model_2025-01-14-22-44-36.pkl")  # Replace with your model's path

# Define an endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        input_data = request.get_json()

        # Convert input data to a DataFrame
        input_df = pd.DataFrame([input_data])

        # Preprocess the input if necessary (e.g., scaling)
        # Example: scaled_input = scaler.transform(input_df)

        # Generate predictions
        prediction = model.predict(input_df)

        # Return the prediction as JSON
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=5000)
