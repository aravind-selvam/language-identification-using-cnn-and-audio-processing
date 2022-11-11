import os, sys
from zipfile import Path, ZipFile
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifacts
from src.exceptions import CustomException
from src.cloud_storage.s3_operations import S3Sync
from src.constants import *

class DataIngestion:
    """Ingest the data to the pipeline."""
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.s3_sync = S3Sync()
        except Exception as e:
            raise CustomException(e, sys)
        
    def get_data_from_cloud(self) -> None:
        try:
            logger.info("Initiating data download from s3 bucket...")
            download_dir = self.data_ingestion_config.download_dir
            zip_file_path = self.data_ingestion_config.zip_file_path
            if os.path.isfile(zip_file_path):
                logger.info(f"Data is already present in {download_dir}, So skipping download step.")
                return None
            else:
                os.makedirs(download_dir, exist_ok=True)
                self.s3_sync.sync_folder_from_s3(self.data_ingestion_config.download_dir,S3_DATA_URI)
                logger.info(f"Data is downloaded from s3 bucket to Download directory: {download_dir}.")
        except Exception as e:
            raise CustomException(e, sys)
    
    def unzip_data(self) -> None:
        try:
            logger.info("Unzipping the downloaded zip file from download directory...")
            zip_file_path = self.data_ingestion_config.zip_file_path
            unzip_data_path = self.data_ingestion_config.unzip_data_dir_path
            if os.listdir(unzip_data_path):
                logger.info("File already exists in unzip directory, so skipping unzip operation.")
            else:
                os.makedirs(unzip_data_path)
            with ZipFile(self.data_ingestion_config.zip_file_path, 'r') as zip_file_ref:
                zip_file_ref.extractall(unzip_data_path)
            logger.info(f"Unzipped file exists in unzip directory: {unzip_data_path}.")
        except Exception as e:
            raise CustomException(e, sys)
    
    def rename_files(self) -> None:
        try:
            logger.info("Renaming files in unzip directory to single format...")
            unzip_data_dir_path = self.data_ingestion_config.unzip_data_dir_path
            for folder in os.listdir(unzip_data_dir_path):
                class_path = unzip_data_dir_path + '/' + str(folder)
                for count, files in enumerate(os.listdir(class_path)):
                    try:
                        dst = f"{folder}-{str(count)}.mp3"
                        src =f"{audio_dir1}/{folder}/{files}"  
                        dst =f"{audio_dir1}/{folder}/{dst}"
                        os.rename(src, dst)
                    except FileExistsError:
                        pass
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        logger.info("Starting data ingestion component...")
        try:
            self.get_data_from_cloud()
            self.unzip_data()
            logger.info("Data ingestion completed successfully.")
        except Exception as e:
            logger.error("Error in data ingestion component.")
            raise CustomException(e, sys)