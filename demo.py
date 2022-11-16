from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig, DataPreprocessingConfig
from src.components.data_preprocessing import DataPreprocessing
from src.components.model_trainer import ModelTrainer
from src.components.model_pusher import ModelPusher
from src.components.model_evaluation import ModelEvaluation
from src.entity.config_entity import *
from src.components.dataset_custom import IndianLanguageDataset
from src.models.final_model import CNNNetwork
import torch

di = DataIngestion(data_ingestion_config=DataIngestionConfig)
data_ingestion_artifacts = di.initiate_data_ingestion()
print(data_ingestion_artifacts.extracted_data_path)
dp = DataPreprocessing(data_preprocessing_config=DataPreprocessingConfig, data_ingestion_artifacts=data_ingestion_artifacts)
data_preprocessing_artifacts = dp.initiate_data_preprocessing()


train_data = IndianLanguageDataset(dataset_config=CustomDatasetConfig, 
                                    transformations=data_preprocessing_artifacts.transformation_object,
                                    validation=False, 
                                    preprocessing_artifacts=data_preprocessing_artifacts)

val_data = IndianLanguageDataset(dataset_config=CustomDatasetConfig, 
                                    transformations=data_preprocessing_artifacts.transformation_object,
                                    validation=True, 
                                    preprocessing_artifacts=data_preprocessing_artifacts)


model = CNNNetwork(in_channels=1, num_classes=data_preprocessing_artifacts.num_classes)

mt = ModelTrainer(modeltrainer_config=ModelTrainerConfig, data_preprocessing_artifacts = data_preprocessing_artifacts,
train_data=train_data , test_data=val_data, model=model, optimizer_func=torch.optim.Adam)
print(mt.device)
model_trainer_artifacts = mt.initiate_model_trainer()
print(model_trainer_artifacts)
me = ModelEvaluation(data_preprocessing_artifacts=data_preprocessing_artifacts, model_evaluation_config=ModelEvaluationConfig, model_trainer_artifacts=model_trainer_artifacts,
                    optimizer=torch.optim.Adam, train_data=train_data, val_data=val_data)
model_eval_artifacts = me.initiate_evaluation()
mp = ModelPusher(model_evaluation_artifacts= model_eval_artifacts)
mp.initiate_model_pusher()