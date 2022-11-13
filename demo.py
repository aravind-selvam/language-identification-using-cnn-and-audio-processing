from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig

di = DataIngestion(data_ingestion_config=DataIngestionConfig)
di.initiate_data_ingestion()
# di.initiate_data_ingestion()


# D:\iNeuron\Project_Neuron\language_recognition\language-detection-using-cnn-pytroch\data\download_data\language-audio-data.zip