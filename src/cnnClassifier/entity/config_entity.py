from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import yaml

from cnnClassifier.utils.logger import configure_logger
from ..config import constants

logger = configure_logger(__name__)


@dataclass(frozen=True, kw_only=True)
class DataIngestionConfig:
    root_dir: Path = field(default=constants.ROOT_DIR)
    source_URL: str = field(default=constants.SOURCE_URL)
    local_datafile: Path = field(default=constants.DATA_ZIP_FILE)
    unzip_dir: Path = field(default=constants.DATA_EXTRACT_DIR)


class ImageConfig(BaseModel):
    altura: int = Field(..., description="Altura da imagem")
    largura: int = Field(..., description="Largura da imagem")
    canais: int = Field(..., description="Número de canais da imagem (RGB=3)")
    data_dir: Path = Field(
        default=Path(constants.LOCAL_DATA_DIR),
        description="Diretório dos dados de imagem",
    )

    @property
    def size_tuple(self):
        """Retorna o tamanho da imagem como tupla (altura, largura, canais)"""
        return (self.altura, self.largura, self.canais)


class DataSplitterConfig(BaseModel):
    batch_size: int = Field(..., description="Tamanho do lote para treinamento")
    random_seed: int = Field(..., description="Semente para embaralhamento aleatório")
    train_ratio: float = Field(..., description="Proporção dos dados para treino")
    val_ratio: float = Field(..., description="Proporção dos dados para validação")


class ModelConfig(BaseModel):
    # Global
    random_seed: int
    # Model architecture
    model_name: str = Field(..., description="Nome do modelo CNN")
    weights: str = Field(..., description="Pesos pré-treinados")
    include_top: bool = Field(..., description="Incluir camadas de classificação")

    # Image configuration
    image_size: List[int] = Field(..., description="Dimensões da imagem")

    # Training parameters
    batch_size: int = Field(..., description="Tamanho do lote")
    epochs: int = Field(..., description="Número de épocas")
    learning_rate: float = Field(..., description="Taxa de aprendizado")
    dropout_rate: float = Field(..., description="Taxa de dropout")
    loss_function: str = Field(..., description="Função de loss")
    metrics: List[str] = Field(..., description="Métricas de avaliação")
    class_weights: Optional[Dict[int, float]] = None  # ← ADICIONAR
    use_pretrained: bool = Field(default=True, description="Usar modelo pré-treinado")

    # Otimizador e compilação
    optimizer_name: str = Field(..., description="Nome do otimizador")
    optimizer_params: Optional[dict] = Field(default_factory=dict, description="Parâmetros específicos do otimizador")

    # Dataset
    num_classes: int
    train_ratio: float
    val_ratio: float

    # Data augmentation
    augmentation_enabled: bool = Field(True, description="Ativar data augmentation")
    horizontal_flip: bool = Field(True, description="Flip horizontal")
    rotation_factor: float = Field(0.05, description="Fator de rotação")
    zoom_factor: float = Field(..., description="Fator de zoom")

    @classmethod
    def from_yaml(cls, config_path: str, experiment: Optional[str] = None):
        """Carrega configuração do arquivo YAML."""
        logger.info(f"📄 Carregando YAML: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Se especificar experimento, merge
        if experiment and experiment in config.get("experiments", {}):
            exp_config = config["experiments"][experiment]
            for section, values in exp_config.items():
                if section in config:
                    config[section].update(values)
                else:
                    config[section] = values

        # Class weights - CORRIGIDO
        class_weights = config.get("training", {}).get("class_weights")
        if class_weights:
            # Converter strings para int (YAML pode ler como string)
            class_weights = {int(k): v for k, v in class_weights.items()}
        try:
            return cls(
                # Model
                model_name=config["model"]["name"],
                weights=config["model"]["weights"],
                include_top=config["model"]["include_top"],
                # Image
                image_size=config["image"]["size"],
                # Training
                batch_size=config["training"]["batch_size"],
                epochs=config["training"]["epochs"],
                learning_rate=config["optimizer"]["learning_rate"],
                dropout_rate=config["training"]["dropout_rate"],
                loss_function=config["training"]["loss"],
                metrics=config["training"]["metrics"],
                random_seed=config["random_seed"],
                class_weights=class_weights,
                use_pretrained=config["model"].get("use_pretrained", True),
                # Optimizer
                optimizer_name=config["optimizer"]["name"],
                optimizer_params={
                    k: v for k, v in config["optimizer"].items() 
                    if k not in ['name', 'learning_rate']
                },
 
                # Dataset
                num_classes=config["dataset"]["classes"],
                train_ratio=config["dataset"]["train_ratio"],
                val_ratio=config["dataset"]["val_ratio"],
                # Augmentation
                augmentation_enabled=config["augmentation"]["enabled"],
                horizontal_flip=config["augmentation"]["horizontal_flip"],
                rotation_factor=config["augmentation"].get("rotation_factor", 0.05),
                zoom_factor=config["augmentation"].get("zoom_factor", 0.05),
                zoom_range=config["augmentation"].get("zoom_range"),
                brightness_range=config["augmentation"].get("brightness_range"),
                rotation_range=config["augmentation"].get("rotation_range"),
            )
        except KeyError as e:
            raise ValueError(f"🚨 Campo obrigatório faltando no YAML: {e}")


class DataSubsetType(Enum):
    TRAIN = "training"
    VALIDATION = "validation"
    TEST = "test"
