from pathlib import Path

# =============================================================================
# CAMINHOS PRINCIPAIS
# =============================================================================
ROOT_DIR = Path(__file__).parent.parent.parent.parent
ARTIFACTS_DIR = Path("artifacts")
LOGS_DIR = ARTIFACTS_DIR / "logs"
MODELS_DIR = ARTIFACTS_DIR / "models"
LOCAL_DATA_DIR = ARTIFACTS_DIR / "data"

# =============================================================================
# DATA INGESTION
# =============================================================================
SOURCE_URL = (
    # "https://drive.google.com/file/d/1tQc-PGonSVSDtfL4HEsd30klq0gXxjqP/view?usp=sharing" #Soybean leaf pest damage
    "https://drive.google.com/file/d/1fTCJtVkgFO5eUvjVNnbsYri5RCDYVRdW/view?usp=drive_link"  # Pest image
)
DATA_INGESTION_DIR = ARTIFACTS_DIR / "data_ingestion"
DATA_ZIP_FILE = DATA_INGESTION_DIR / "data.zip"
DATA_EXTRACT_DIR = DATA_INGESTION_DIR


# =============================================================================
# DATA SPLIT
# =============================================================================
# DATA_SOURCE_DIR = "artifacts/data_ingestion/bycbh73438-1"
# DATA_SOURCE_DIR = "artifacts/data_ingestion/DatasetPests/Classes/Rotuladas"
DATA_SOURCE_DIR = "artifacts/data/INSECT12C-cropped-10-classes"
DATA_SPLIT_DIR = "artifacts/data_split"

# Proporções
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.1

# Parâmetros
RANDOM_STATE = 42
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]

# =============================================================================
# MODELO
# =============================================================================
MODEL_NAME = "MobileNetV3"
MODELS_DIR = ARTIFACTS_DIR / "models"
IMAGE_SIZE = [224, 224, 3]
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 10
DROPOUT_RATE = 0.2


# =============================================================================
# VALIDAÇÕES
# =============================================================================
def validate_split_ratios():
    """Valida se as proporções de divisão somam 1.0"""
    total = TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Proporções devem somar 1.0, atual: {total:.3f}")
    return True


# Executar validação na importação
validate_split_ratios()
