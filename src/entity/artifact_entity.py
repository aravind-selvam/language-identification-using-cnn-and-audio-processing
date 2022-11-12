from dataclasses import dataclass

# Data ingestion artifacts
@dataclass
class DataIngestionArtifacts:
    downloaded_data_path: str 
    extracted_data_path: str

@dataclass
class DataPreprocessingArtifacts:
    train_metadata_path: str
    test_metadata_path: str
    class_mappings: dict