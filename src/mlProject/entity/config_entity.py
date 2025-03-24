from pathlib import Path
from dataclasses import dataclass

# Data class for data ingestion configuration
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path