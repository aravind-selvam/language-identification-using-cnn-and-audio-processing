import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit

from src.entity.artifact_entity import DataIngestionArtifacts
from src.entity.config_entity import DataPreprocessingConfig
from src.exceptions import CustomException
from src.logger import logging
from src.constants import FFT_SIZE,HOP_LENGTH,N_MELS


class DataPreprocessing:
    "preprocess metadata training"

    def __init__(self, data_preprocessing_config: DataPreprocessingConfig,
                 data_ingestion_artifacts: DataIngestionArtifacts):
        try:
            self.data_preprocessing_config = data_preprocessing_config
            self.data_ingestion_artifacts = data_ingestion_artifacts
        except Exception as e:
            raise CustomException(e, sys)

    def get_meta_data(self) -> (pd.DataFrame, dict):
        try:
            audio_dir = self.data_ingestion_artifacts.extracted_data_path
            metadata = {}
            for label in os.listdir(audio_dir):
                class_path = audio_dir + '/' + str(label)
                audio_clips = os.listdir(class_path)
                for filename in audio_clips:
                    metadata[filename] = label

            metadata = pd.DataFrame.from_dict(
                metadata, orient='index').reset_index().sort_values(by=0)
            metadata.columns = ['filename', 'foldername']
            le = preprocessing.LabelEncoder()
            metadata['labels'] = le.fit_transform(metadata.foldername)
            le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            return metadata, le_name_mapping
        except Exception as e:
            raise CustomException(e, sys)

    def train_test_split(self, metadata: DataFrame) -> None:
        try:
            split = StratifiedShuffleSplit(
                n_splits=5, train_size=0.7, test_size=0.3, random_state=42)
            for train_index, test_index in split.split(metadata, metadata['labels']):
                strat_train_set = metadata.loc[train_index]
                strat_val_set = metadata.loc[test_index]

            train_file_path = self.data_preprocessing_config.train_file_path
            test_file_path = self.data_preprocessing_config.test_file_path

            strat_train_set.to_csv(train_file_path, index=False)
            strat_val_set.to_csv(test_file_path, index=False)

        except Exception as e:
            raise CustomException(e, sys)
    
    def audio_transformations(self) -> MelSpectrogram:
        mel_spectrogram = torchaudio.transforms.Mealspectrogram(
        sample_rate= self.data_preprocessing_config.sample_rate,
        n_fft= FFT_SIZE,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
        ) 

    def initiate_data_preprocessing(self) -> DataPreprocessingArtifacts:
        try:
            metadata, mappings = get_meta_data()
            train_test_split(metadata)
            data_preprocessing_artifacts = DataPreprocessingArtifacts(train_metadata_path=self.data_preprocessing_config.train_file_path,
                                                                      test_metadata_path=self.data_preprocessing_config.test_file_path)
            return data_preprocessing_artifacts
        except Exception as e:
            raise CustomException(e, sys)
