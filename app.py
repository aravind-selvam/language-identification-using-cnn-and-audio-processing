from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from from_root import from_root
from src.pipe.prediction_pipeline import LanguageData, SinglePrediction

app = Flask(__name__)
CORS(app)


prediction_artifacts_path = os.join.path(from_root(), APPLICATION_ARTIFACTS_DIR)


@app.route('/', methods=['GET'])
@cross_origin()
def home():

@app.route('/predict', methods=['POST'])
@cross_origin()
def predictroute():
    if request.method == 'POST':
        image = request.json['sound']
        decodeSound(image, "inputSound.wav")
