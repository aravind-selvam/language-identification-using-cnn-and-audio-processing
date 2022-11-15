import pandas as pd
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import Dataset

from src.entity.artifact_entity import DataPreprocessingArtifacts
from src.entity.config_entity import CustomDatasetConfig
from src.exceptions import CustomException
from src.logger import logging
import os, sys

class IndianLanguageDataset(Dataset):
    try: 
        def __init__(self, dataset_config: CustomDatasetConfig, transformations: MelSpectrogram,
                    preprocessing_artifacts: DataPreprocessingArtifacts, validation: False):

            self.dataset_config = dataset_config
            self.transformations = transformations
            self.preprocessing_artifacts = preprocessing_artifacts
            if validation:
                self.annotations = pd.read_csv(self.preprocessing_artifacts.test_metadata_path)
            else:
                self.annotations = pd.read_csv(self.preprocessing_artifacts.train_metadata_path)
            self.audio_dir = self.dataset_config.audio_dir
            self.num_samples = self.dataset_config.num_samples
            self.target_sample_rate = self.dataset_config.sample_rate
        
        def __len__(self):
            return len(self.annotations)
        
        def __getitem__(self, idx):
            audio_sample_path = self._get_audio_sample_path(idx)
            label = self._get_audio_sample_label(idx)
            signal, sr = torchaudio.load(audio_sample_path)
            signal = self._resample_if_necessary(signal, sr)
            signal = self._mix_down_if_necessary(signal)
            signal = self._cut_if_necessary(signal)
            signal = self._right_pad_if_necessary(signal)
            signal = self.transformations(signal)
            return signal, label
        
        def _get_audio_sample_path(self, idx):
            class_name = f"{self.annotations.iloc[idx, 1]}"
            path = os.path.join(self.audio_dir, class_name, self.annotations.iloc[idx, 0])
            return path

        def _get_audio_sample_label(self, idx):
            return self.annotations.iloc[idx, 2]
        
        def _cut_if_necessary(self, signal):
            if signal.shape[1] > self.num_samples:
                signal = signal[:, :self.num_samples]
            return signal
        
        def _right_pad_if_necessary(self, signal):
            length_signal = signal.shape[1]
            if length_signal < self.num_samples:
                num_missing = self.num_samples - length_signal
                last_dim_padding = (0, num_missing)
                signal = torch.nn.functional.pad(signal, last_dim_padding)
            return signal
            
        def _resample_if_necessary(self, signal, sr):
            if sr != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
                signal = resampler(signal)
            return signal
        
        def _mix_down_if_necessary(self, signal):
            if signal.shape[0] > 1:
                signal = torch.mean(signal, dim = 0, keepdim=True)
            return signal

    except Exception as e:
        raise CustomException(e, sys)
