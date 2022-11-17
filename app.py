from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from src.pipe.prediction_pipeline import LanguageData, SinglePrediction
from src.entity.config_entity import PredictionPipelineConfig
from src.utils import decodesound
from src.pipe.training_pipeline import TrainingPipeline

app = Flask(__name__)
CORS(app)

predictor = SinglePrediction(PredictionPipelineConfig())

@app.route('/', methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/train", methods=['GET'])
@cross_origin()
def train():
    train_pipeline = TrainingPipeline()

    train_pipeline.run_pipeline()

    return jsonify(train_pipeline.get_result())

@app.route('/predict', methods=['POST'])
@cross_origin()
def predictroute():
    config = PredictionPipelineConfig()
    input_sounds_path = config.input_sounds_path
    app_artifacts = config.app_artifacts
    os.makedirs(app_artifacts, exist_ok=True)
    if request.method == 'POST':
        base_64 = request.json['sound']
        decodesound(base_64, input_sounds_path)
        signal = LanguageData().load_data(input_sounds_path)
        signal.unsqueeze_(0)
        result = predictor.predict_language(input_signal=signal)
        return jsonify({"Result" : result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)