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

# @app.route('/predict',methods=['POST'])
# def predict():
#     try:
#         data = [float(x) for x in request.form.values()]
#         final_input = np.array(data).reshape(1, -1)
#         output = regmodel.predict(final_input)[0]
#         return render_template("home.html", prediction_text=f"The predicted wine quality is {output}")
#     except Exception as e:
#         return render_template("home.html", prediction_text="An error occurred during prediction.")
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extracting each input value from the form
        fixed_acidity = float(request.form['fixed_acidity'])
        volatile_acidity = float(request.form['volatile_acidity'])
        citric_acid = float(request.form['citric_acid'])
        residual_sugar = float(request.form['residual_sugar'])
        chlorides = float(request.form['chlorides'])
        free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
        total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
        density = float(request.form['density'])
        pH = float(request.form['pH'])
        sulphates = float(request.form['sulphates'])
        alcohol = float(request.form['alcohol'])

        # Creating a numpy array with all the inputs
        final_input = np.array([
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
            density, pH, sulphates, alcohol
        ]).reshape(1, -1)

        # Making the prediction
        output = regmodel.predict(final_input)[0]

        # Rendering the template with the prediction result
        return render_template("home.html", prediction_text=f"The predicted wine quality is {output}")
    
    except Exception as e:
        # Handling any errors that occur during prediction
        return render_template("home.html", prediction_text="An error occurred during prediction.")

if __name__=="__main__":
        app.run(debug=True)