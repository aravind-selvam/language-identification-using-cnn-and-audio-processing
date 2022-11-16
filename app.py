from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from from_root import from_root
from src.pipe.prediction_pipeline import LanguageData, SinglePrediction
from src.entity.config_entity import PredictionPipelineConfig
from src.utils import decodesound

app = Flask(__name__)
CORS(app)

prediction_config = PredictionPipelineConfig()
predictor = SinglePrediction()

@app.route('/', methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@cross_origin()
def predictroute():
    input_file_path = prediction_config.input_sounds_path
    if request.method == 'POST':
        base_64 = request.json['sound']
        decodesound(base_64, input_file_path)
        signal = LanguageData().load_data(input_file_path)
        result = predictor.predict_language(input_signal=signal)
        return jsonify({"Result" : result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
        
