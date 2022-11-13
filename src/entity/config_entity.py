import os
import sys
from zipfile import Path, ZipFile
from dataclasses import dataclass
from src.constants import *
from datetime import datetime
from from_root import from_root


TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = SOURCE_DIR_NAME
    artifact_dir: str = os.path.join(ARTIFACTS_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP

training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    bucket_name: str = BUCKET_NAME
    zip_file_name: str = ZIP_FILE_NAME
    download_dir: str = os.path.join(from_root(), DATA_DIR_NAME, DOWNLOAD_DIR)
    zip_file_path: str = os.path.join(from_root(), download_dir, ZIP_FILE_NAME)
    unzip_data_dir_path: str = os.path.join(from_root(), DATA_DIR_NAME, EXTRACTED_DATA_DIR)
    data_ingestion_artifacts_dir: str = os.path.join(from_root(), training_pipeline_config.artifact_dir, DATA_INGESTION_ARTIFACTS_DIR)
    train_file_path: str = os.path.join(data_ingestion_artifacts_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    test_file_path: str = os.path.join(data_ingestion_artifacts_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    