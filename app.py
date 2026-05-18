from flask import Flask, render_template, request
import os
import time

from utils.serial_predict import serial_prediction
from utils.parallel_predict import parallel_prediction

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    files = request.files.getlist('images')

    method = request.form.get('method')

    image_paths = []

    for file in files:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        image_paths.append(filepath)

    start = time.time()

    if method == 'serial':
        predictions = serial_prediction(image_paths)

    else:
        predictions = parallel_prediction(image_paths)

    end = time.time()

    execution_time = end - start

    return render_template(
        'result.html',
        predictions=predictions,
        execution_time=execution_time,
        method=method
    )

if __name__ == '__main__':
    app.run(debug=True)