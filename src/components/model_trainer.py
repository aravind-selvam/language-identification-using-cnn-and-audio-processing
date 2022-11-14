import torch
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataPreprocessingArtifacts, ModelTrainerArtifacts
from src.models.final_model import CNNNetwork
from src.components.dataset_custom import IndianLanguageDataset
from src.utils.gpu_functions import DeviceDataLoader, get_default_device, to_device
from src.exceptions import CustomException
from torch.optim.lr_scheduler import StepLR 


class ModelTrainer:
    """
    Model Trainer
    """
    def __init__(self, modeltrainer_config: ModelTrainerConfig,
                data_preprocessing_artifacts: DataPreprocessingArtifacts,
                optimizer_func,
                model,
                train_data,
                val_data):
            try: 
                self.model_trainer_config = modeltrainer_config
                self.data_preprocessing_artifacts = data_preprocessing_artifacts
                self.learning_rate = modeltrainer_config.learning_rate
                self.model = model
                self.epochs = modeltrainer_config.epochs
                self.optimizer_func = optimizer_func
                self.train_data = train_data
                self.val_data = val_data
            except Exception as e:
                raise CustomException(e, sys)
    
    @torch.no_grad()
    def evaluate(model, val_loader):
        try:
            self.model.eval()
            outputs = [self.model.validation_step(batch) for batch in val_loader]
            return self.model.validation_epoch_end(outputs)
        except Exception as e:
            raise CustomException(e, sys)

    def fit(self, train_loader, val_loader):
        try:
            history = []
            self.model.train()
            optimizer = self.optimizer_func(model.parameters(), self.learning_rate)
            scheduler = StepLR(optimizer= optimizer, 
                               step_size= self.model_trainer_config.stepsize, 
                               gamma= self.model_trainer_config.gamma
                               )
            for epoch in range(1, self.epochs + 1):
                # Training
                scheduler.step()
                print('Epoch:', epoch,'LR:', scheduler.get_lr())
                train_losses = []
                for batch in train_loader:
                    optimizer.zero_grad()
                    loss = model.training_step(batch)
                    train_losses.append(loss)
                    loss.backward()
                    optimizer.step()
                # Validation
                result = evaluate(self.model, val_loader)
                result['train_loss'] = torch.stack(train_losses).mean().item()
                self.model.epoch_end(epoch, result)
                history.append(result)
            return history
        except Exception as e:
            raise CustomException(e, sys)

    def get_dataloader(self,) -> Dataloader:
        try:
            train_loader = DataLoader(self.train_data,
                                    batch_size=self.model_trainer_config.batch_size,
                                    shuffle=True,
                                    num_workers=self.model_trainer_config.num_workers,
                                    pin_memory=True)
            
            val_loader = DataLoader(self.test_data,
                                    batch_size=self.model_trainer_config.batch_size * 2,
                                    shuffle=False,
                                    num_workers=self.model_trainer_config.num_workers,
                                    pin_memory=True)
            return train_loader, val_loader
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifacts:
        try:
            logging.info("Starting model trainer component...")

            device = get_default_device()
            num_classes = self.data_preprocessing_artifacts.num_classes
            model = to_device(model, device=device)
            train_loader, val_loader = self.get_dataloader()

            # use the wrapper class to load the data to device
            train_dataloader = DeviceDataLoader(train_loader, device)
            val_dataloader = DeviceDataLoader(val_loader, device)

            # training the model for defined epochs
            history = self.fit(train_dataloader, val_dataloader)
            trained_model_path = self.model_trainer_config.trained_model_dir
            os.makedirs(trained_model_path, exist_ok=True)
            torch.save(self.model.state_dict(),trained_model_path)

            model_trainer_artifacts = ModelTrainerArtifacts(trained_model_path=trained_model_path)
            logging.info("Model Trainer class completed")
            return model_trainer_artifacts
        except Exception as e:
            raise CustomException(e, sys)




