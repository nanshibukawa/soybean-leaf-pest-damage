from dataclasses import dataclass, field
from pathlib import Path
from ..config import constants

@dataclass(frozen=True, kw_only=True)
class DataIngestionConfig:
    root_dir: Path = field(default=constants.ROOT_DIR)
    source_URL: str = field(default=constants.SOURCE_URL)
    local_datafile: Path = field(default=constants.DATA_ZIP_FILE)
    unzip_dir: Path = field(default=constants.DATA_EXTRACT_DIR)
