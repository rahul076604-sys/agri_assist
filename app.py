
from flask import Flask, render_template, request
from train_model import predict_from_form

import pickle

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/form.html')
def form():
    return render_template('form.html')


@app.route('/form.html/predict', methods=['POST'])
def predict():

    form_data= [
    str(request.form['region']),
    str(request.form['soil_type']),
    str(request.form['crop']),
    float(request.form['rainfall']),
    float(request.form['temperature']),
    request.form['fertilizer_used'].lower() == 'true',
    request.form['irrigation_used'].lower() == 'true',
    str(request.form['weather_condition']),
    float(request.form['days_to_harvest'])
]




    raw_result = predict_from_form(form_data)
    result=round(raw_result[0],3)
    return render_template(
        'form.html',
        prediction_text=f"predicted {str(request.form['crop'])} yield: {result} tons per hectare"
    )

if __name__ == '__main__':
    app.run(debug=True)

