import os, sys
import torch
from src.entity.config_entity import ModelEvaluationConfig, ModelTrainerConfig
from src.entity.artifact_entity import DataPreprocessingArtifacts, ModelTrainerArtifacts, ModelEvaluationArtifacts
from src.cloud_storage.s3_operations import S3Sync
from src.logger import logging
from src.models.final_model import CNNNetwork
from src.utils.gpu_functions import to_device, get_default_device
from src.exceptions import CustomException
from src.components.model_trainer import ModelTrainer


class ModelEvaluation:
    def __init__(self, data_preprocessing_artifacts: DataPreprocessingArtifacts, 
                model_evaluation_config: ModelEvaluationConfig,
                model_trainer_artifacts: ModelTrainerArtifacts,
                optimizer, train_data, val_data):
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.data_preprocessing_artifacts = data_preprocessing_artifacts
        self.model_evaluation_config = model_evaluation_config
        self.trainer_artifacts = model_trainer_artifacts
    
    def get_best_model_path(self):
        """
        Method Name :   get_best_model
        Description :   This function is used to get model in production
        
        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            model_path = self.model_evaluation_config.s3_model_path
            best_model_dir = self.model_evaluation_config.best_model_dir
            s3_sync = S3Sync()
            s3_sync.sync_folder_from_s3(folder=best_model_dir, aws_bucket_url=model_path)
            for file in os.listdir(best_model_dir):
                if file.endswith(".pth"):
                    best_model_path = os.path.join(best_model_dir, file)
                    logging.info(f"Best model found in {best_model_path}")
                    break
                else:
                    logging.info("Model is not available in best_model_directory")
                    best_model_path = None
            return best_model_path
        except Exception as e:
            raise CustomException(e,sys)
    
    def evaluate_model(self):
        best_model_path = self.get_best_model_path()
        if best_model_path is not None:
            device = get_default_device()
            in_channels = self.model_evaluation_config.in_channels
            num_classes = self.data_preprocessing_artifacts.num_classes
            model = to_device(CNNNetwork(in_channels, num_classes), device)
            # load back the model
            state_dict = torch.load(best_model_path)
            model = model.load_state_dict(state_dict)
            model_trainer = ModelTrainer(modeltrainer_config=ModelTrainerConfig(),
                                        data_preprocessing_artifacts=self.data_preprocessing_artifacts,
                                        train_data=self.train_data, val_data=self.val_data,
                                        model = model,
                                        optimizer_func= self.optimizer)
            _ , val_loader = model_trainer.get_dataloader()
            result = model_trainer.evaluate(model, val_loader)
            val_accuracy = result['val_acc']
            logging.info(f"S3 Validation accuracy is {val_accuracy}")
            logging.info(f"Locally trained accuracy is {self.trainer_artifacts.model_accuracy}")
            s3_model_accuracy = val_accuracy
        else:
            logging.info("Model is not found on production server, So couldn't evaluate")
            s3_model_accuracy = None
        return s3_model_accuracy
        
    def initiate_evaluation(self):
        s3_model_accuracy = self.evaluate_model()
        tmp_best_model_accuracy = 0 if s3_model_accuracy is None else s3_model_accuracy
        trained_model_accuracy = self.trainer_artifacts.model_accuracy
        model_evaluation_artifacts = ModelEvaluationArtifacts(trained_model_accuracy = trained_model_accuracy,
                                                            s3_model_accuracy = s3_model_accuracy,
                                                            is_model_accepted = trained_model_accuracy > tmp_best_model_accuracy,
                                                            trained_model_path = self.trainer_artifacts.trained_model_path,
                                                            s3_model_path = self.get_best_model_path
                                                            )
        logging.info(f"Model evaludation completed! Artifacts: {model_evaluation_artifacts}")
        return model_evaluation_artifacts
    
