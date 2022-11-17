from src.logger import logging
import os, sys
import torch
import torchaudio
from src.cloud_storage.s3_operations import S3Sync

from src.exceptions import CustomException
from src.entity.config_entity import PredictionPipelineConfig, CustomDatasetConfig, DataPreprocessingConfig
from src.constants import *
from src.models.final_model import CNNNetwork
from src.utils import load_object


class LanguageData:
    def __init__(self):
        self.dataset_config = CustomDatasetConfig()
        self.data_preprocessing_config = DataPreprocessingConfig()
    
    def _cut_if_necessary(self, signal):
        num_samples = self.dataset_config.num_samples
        if signal.shape[1] > num_samples:
            signal = signal[:, :num_samples]
        return signal
    
    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        num_samples = self.dataset_config.num_samples
        if length_signal < num_samples:
            num_missing = num_samples - length_signal
            last_dim_padding = (0, num_missing)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
        
    def _resample_if_necessary(self, signal, sr):
        target_sample_rate = self.dataset_config.sample_rate
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
            signal = resampler(signal)
        return signal
    
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim = 0, keepdim=True)
        return signal
    
    def load_data(self, audio_data_path):
        transformation_obj_path = self.data_preprocessing_config.transformations_object_path
        transformation = load_object(transformation_obj_path)
        signal, sr = torchaudio.load(audio_data_path)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = transformation(signal)
        return signal
     

class SinglePrediction:
    def __init__(self, prediction_pipeline_config: PredictionPipelineConfig):
        self.prediction_pipeline_config = prediction_pipeline_config
        self.data_preprocessing_config = DataPreprocessingConfig()

    def _get_model_in_production(self):
        s3_model_path = self.prediction_pipeline_config.s3_model_path
        model_download_path = self.prediction_pipeline_config.model_download_path
        os.makedirs(model_download_path, exist_ok=True)
        s3_sync = S3Sync()
        s3_sync.sync_folder_from_s3(folder=model_download_path, aws_bucket_url=s3_model_path)
        for file in os.listdir(model_download_path):
            if file.endswith(".pt"):
                prediction_model_path = os.path.join(model_download_path, file)
                logging.info(f"Production model for prediction found in {prediction_model_path}")
                break
            else:
                logging.info("Model is not available in Prediction artifacts")
                prediction_model_path = None
        return prediction_model_path

    @staticmethod
    def prediction_step(model, input_signal, class_mapping):
        model.eval()
        with torch.no_grad():
            prediction = model(input_signal)
            prediction_index = prediction[0].argmax(0)
            predicted = class_mapping[prediction_index]
        return predicted
    
    def get_model(self):
        prediction_model_path = self._get_model_in_production()
        if prediction_model_path is None:
            return None
        else:
            num_classes = NUM_CLASSES
            in_channels = IN_CHANNELS
            prediction_model = CNNNetwork(in_channels=in_channels, num_classes=num_classes)
            model_state_dict = torch.load(prediction_model_path)
            prediction_model.load_state_dict(model_state_dict[0]['model_state_dict'])
            prediction_model.eval()
        return prediction_model
    
    def predict_language(self, input_signal):
        prediction_model = self.get_model()
        class_mapping = self.data_preprocessing_config.class_mappings_object_path
        class_mapping = load_object(file_path=class_mapping)
        if prediction_model is not None:
            output = self.prediction_step(prediction_model, input_signal, class_mapping)
            return output
        else:
            raise CustomException("Model not Found in production", sys)

