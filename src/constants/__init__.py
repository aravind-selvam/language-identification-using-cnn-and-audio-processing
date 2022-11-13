import os

ARTIFACTS_DIR: str = "artifacts"
SOURCE_DIR_NAME: str = "src"
LOGS_DIR: str = "logs"
LOGS_FILE: str = "language_detector.log"

BUCKET_NAME: str = "spoken-language-data"
ZIP_FILE_NAME: str = "language-audio-data.zip"
UNZIPPED_FOLDER_NAME: str = "language-audio-data"
S3_BUCKET_URI = "s3://spoken-language-data/data/"

# common files
FILE_NAME: str = "metadata.csv"
TRAIN_FILE_NAME: str = "metadata_train.csv"
TEST_FILE_NAME: str = "metadata_test.csv"
MODEL_FILE_NAME: str = "language_model.pth"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

# constants related to data ingestion
DATA_DIR_NAME: str = "data"
DOWNLOAD_DIR: str = "download_data"
EXTRACTED_DATA_DIR: str = "final_data"
DATA_INGESTION_ARTIFACTS_DIR: str = "data_ingestion_artifacts"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_DIR: str = "train"
DATA_INGESTION_TEST_DIR: str = "test"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.3

# constants related to data preprocessing

