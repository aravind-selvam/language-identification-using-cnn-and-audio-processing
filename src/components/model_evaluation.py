import os, sys
import torch
from src.entity.config_entity import ModelEvaluationConfig, ModelTrainerConfig
from src.entity.artifact_entity import DataPreprocessingArtifacts, ModelTrainerArtifacts, ModelEvaluationArtifacts
from src.cloud_storage.s3_operations import S3Sync
from src.logger import logging
from src.models.final_model import CNNNetwork
from src.exceptions import CustomException
import numpy as np


class ModelEvaluation:
    def __init__(self, data_preprocessing_artifacts: DataPreprocessingArtifacts, 
                model_evaluation_config: ModelEvaluationConfig,
                model_trainer_artifacts: ModelTrainerArtifacts):
        self.data_preprocessing_artifacts = data_preprocessing_artifacts
        self.model_evaluation_config = model_evaluation_config
        self.trainer_artifacts = model_trainer_artifacts
    
    def get_best_model_path(self):
        try:
            model_evaluation_artifacts_dir = self.model_evaluation_config.model_evaluation_artifacts_dir
            os.makedirs(model_evaluation_artifacts_dir, exist_ok=True)
            model_path = self.model_evaluation_config.s3_model_path
            best_model_dir = self.model_evaluation_config.best_model_dir
            s3_sync = S3Sync()
            best_model_path = None
            s3_sync.sync_folder_from_s3(folder=best_model_dir, aws_bucket_url=model_path)
            for file in os.listdir(best_model_dir):
                if file.endswith(".pt"):
                    best_model_path = os.path.join(best_model_dir, file)
                    logging.info(f"Best model found in {best_model_path}")
                    break
                else:
                    logging.info("Model is not available in best_model_directory")
            return best_model_path
        except Exception as e:
            raise CustomException(e,sys)
    
    def evaluate_model(self):
        best_model_path = self.get_best_model_path()
        if best_model_path is not None:
            in_channels = self.model_evaluation_config.in_channels
            num_classes = self.data_preprocessing_artifacts.num_classes
            model = CNNNetwork(in_channels, num_classes)
            # load back the model
            state_dict = torch.load(best_model_path, map_location='cpu')
            model.load_state_dict(state_dict['model_state_dict'])
            model.eval()
            accuracy = state_dict['accuracy']
            loss = state_dict['loss']
            logging.info(f"S3 Model Validation accuracy is {accuracy}")
            logging.info(f"S3 Model Validation loss is {loss}")
            logging.info(f"Locally trained accuracy is {self.trainer_artifacts.model_accuracy}")
            s3_model_accuracy = accuracy
            s3_model_loss = loss
        else:
            logging.info("Model is not found on production server, So couldn't evaluate")
            s3_model_accuracy = None
            s3_model_loss = None
        return s3_model_accuracy, s3_model_loss
        
    def initiate_evaluation(self):
        s3_model_accuracy, s3_model_loss = self.evaluate_model()
        tmp_best_model_accuracy = 0 if s3_model_accuracy is None else s3_model_accuracy
        tmp_best_model_loss = np.inf if s3_model_loss is None else s3_model_loss
        trained_model_accuracy = self.trainer_artifacts.model_accuracy
        trained_model_loss = self.trainer_artifacts.model_loss
        evaluation_response = trained_model_accuracy > tmp_best_model_accuracy and tmp_best_model_loss > trained_model_loss
        model_evaluation_artifacts = ModelEvaluationArtifacts(trained_model_accuracy = trained_model_accuracy,
                                                            s3_model_accuracy = s3_model_accuracy,
                                                            is_model_accepted = evaluation_response,
                                                            trained_model_path = self.trainer_artifacts.trained_model_path,
                                                            s3_model_path = self.model_evaluation_config.s3_model_path
                                                            )
        logging.info(f"Model evaluation completed! Artifacts: {model_evaluation_artifacts}")
        return model_evaluation_artifacts
    
