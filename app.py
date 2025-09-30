from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model_filename = "fake_news_ensemble_model.joblib"
model = joblib.load(model_filename)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.get_json(force=True)
        input_data = np.array(data['features']).reshape(1, -1)
        
        # Make predictions
        prediction = model.predict(input_data)
        
        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)