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
    data_dir: Path = Field(default=Path(constants.LOCAL_DATA_DIR), description="Diretório dos dados de imagem")

    @property
    def size_tuple(self):
        """Retorna o tamanho da imagem como tupla (altura, largura, canais)"""
        return (self.altura, self.largura, self.canais)

class DataSplitterConfig(BaseModel):
    batch_size: int = Field(32, description="Tamanho do lote para treinamento")
    random_seed: int = Field(42, description="Semente para embaralhamento aleatório")
    train_ratio: float = Field(0.8, description="Proporção dos dados para treino")
    val_ratio: float = Field(0.20, description="Proporção dos dados para validação")
    test_ratio: float = Field(0.20, description="Proporção dos dados para teste")

class ModelConfig(BaseModel):
    model_name: str = Field("MobileNetV3", description="Nome do modelo CNN")
    batch_size: int = Field(32, description="Tamanho do lote para treinamento")
    epochs: int = Field(10, description="Número de épocas para treinamento")
    learning_rate: float = Field(0.001, description="Taxa de aprendizado")
    num_classes: int = Field(4, description="Número de classes de saída")
    dropout_rate: float = Field(0.2, description="Taxa de dropout para regularização")

    def get_image_size(self, image_config: 'ImageConfig') -> tuple:
        """Retorna o tamanho da imagem baseado na configuração fornecida"""
        return image_config.size_tuple


class DataSubsetType(Enum):
    TRAIN = 'training'
    VALIDATION = 'validation'
    TEST = 'test'



