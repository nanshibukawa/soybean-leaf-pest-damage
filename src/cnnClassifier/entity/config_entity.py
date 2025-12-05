from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from pydantic import BaseModel, Field
from ..config import constants

@dataclass(frozen=True, kw_only=True)
class DataIngestionConfig:
    root_dir: Path = field(default=constants.ROOT_DIR)
    source_URL: str = field(default=constants.SOURCE_URL)
    local_datafile: Path = field(default=constants.DATA_ZIP_FILE)
    unzip_dir: Path = field(default=constants.DATA_EXTRACT_DIR)

class ImageConfig(BaseModel):
    altura: int = Field(256, description="Altura da imagem")
    largura: int = Field(256, description="Largura da imagem")
    canais: int = Field(3, description="Número de canais da imagem (RGB=3)")
    data_dir: Path = Field(..., description="Diretório dos dados de imagem")

class DataSplitterConfig(BaseModel):
    batch_size: int = Field(32, description="Tamanho do lote para treinamento")
    random_seed: int = Field(42, description="Semente para embaralhamento aleatório")
    train_ratio: float = Field(0.8, description="Proporção dos dados para treino")
    val_ratio: float = Field(0.20, description="Proporção dos dados para validação")
    test_ratio: float = Field(0.20, description="Proporção dos dados para teste")


class DataSubsetType(Enum):
    TRAIN = 'training'
    VALIDATION = 'validation'
    TEST = 'test'
