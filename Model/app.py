from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the model
with open('crop_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    temperature = data['temperature']
    humidity = data['humidity']
    soil_type = data['soil_type']

    # Create a NumPy array for the model input
    features = np.array([[temperature, humidity, soil_type]])

    # Make prediction
    prediction = model.predict(features)
    return jsonify({'predicted_crop': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
