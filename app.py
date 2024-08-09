from django.shortcuts import render
import pickle
from flask import Flask, request, app,jsonify, url_for, render_template

import numpy as np
import pandas as pd

app=Flask(__name__)
regmodel=pickle.load(open('regmodel.pkl','rb'))
model=pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
# def predict_api():
#     data=request.json['data']
#     print(data)
#     print(np.array(list(data.values())).reshape(1,-1))
#     new_data=model.transform(np.array(list(data.values())).reshape(1,-1))
#     output=regmodel.predict(new_data)
#     print(output[0])
#     return jsonify(output[0])
def predict_api():
    try:
        # Get JSON data from request
        data = request.json['data']
        print("Received data:", data)
        
        # Convert data to numpy array and reshape for prediction
        features = np.array(list(data.values())).reshape(1, -1)
        print("Features for prediction:", features)
        
        # Make prediction using the model
        prediction = model.predict(features)
        print("Prediction:", prediction[0])
        
        # Return the prediction as JSON
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        # Handle exceptions and return an error message
        return jsonify({'error': str(e)}), 500

    if __name__=="__main__":
        app.run(debug=True)