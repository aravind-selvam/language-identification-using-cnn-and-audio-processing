from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from src.pipe.prediction_pipeline import LanguageData, SinglePrediction
from src.entity.config_entity import PredictionPipelineConfig
from pydub import AudioSegment
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
    os.makedirs(config.prediction_artifact_dir, exist_ok=True)
    input_sounds_path = config.input_sounds_path
    wave_sounds_path = config.wave_sounds_path
    app_artifacts = config.app_artifacts
    os.makedirs(app_artifacts, exist_ok=True)
    if request.method == 'POST':
        base_64 = request.json['sound']
        decodesound(base_64, input_sounds_path)
        sound = AudioSegment.from_mp3(input_sounds_path)
        sound.export(wave_sounds_path, format="wav")
        signal = LanguageData().load_data(wave_sounds_path)
        signal.unsqueeze_(0)
        result = predictor.predict_language(input_signal=signal)
        return jsonify({"Result" : result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)