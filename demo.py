from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig, DataPreprocessingConfig
from src.components.data_preprocessing import DataPreprocessing

di = DataIngestion(data_ingestion_config=DataIngestionConfig)
data_ingestion_artifacts = di.initiate_data_ingestion()
dp = DataPreprocessing(data_preprocessing_config=DataPreprocessingConfig, data_ingestion_artifacts=data_ingestion_artifacts)
data_preprocessing_artifacts = dp.initiate_data_preprocessing()
print(data_preprocessing_artifacts)

# D:\iNeuron\Project_Neuron\language_recognition\language-detection-using-cnn-pytroch\data\download_data\language-audio-data.zip