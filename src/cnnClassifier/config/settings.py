import os
from dataclasses import dataclass
from cnnClassifier.config import constants

@dataclass
class EnvironmentSettings:
    """Configurações que podem mudar por ambiente"""
    debug_mode: bool = os.getenv("DEBUG", "False").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    device: str = os.getenv("DEVICE", "auto")  # auto, cpu, cuda
    
@dataclass 
class RuntimeSettings:
    """Configurações de runtime que podem ser ajustadas"""
    batch_size: int = constants.BATCH_SIZE
    epochs: int = constants.EPOCHS
    learning_rate: float = constants.LEARNING_RATE
    
    def __post_init__(self):
        # Ajustes baseados no ambiente
        if os.getenv("QUICK_TEST"):
            self.epochs = 2
            self.batch_size = 8