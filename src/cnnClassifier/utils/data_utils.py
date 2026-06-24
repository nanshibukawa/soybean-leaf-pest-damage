import zipfile
from pathlib import Path
import gdown
from cnnClassifier.utils.logger import configure_logger
import urllib.request
import tensorflow as tf

logger = configure_logger(__name__)


def download_file(url: str, output_path: Path) -> Path:
    """Baixa um arquivo de qualquer URL (Google Drive via gdown ou HTTP direto via urllib)"""
    if output_path.exists():
        logger.debug(f"Arquivo já existe: {output_path.name}")
        return output_path
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if "drive.google.com" in url or "/file/d/" in url:
        return download_from_gdrive(url, output_path)
    else:
        logger.info(f"Baixando link direto: {url}")
        try:
            req = urllib.request.Request(
                url, 
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req) as response, open(output_path, "wb") as out_file:
                out_file.write(response.read())
            logger.info("✅ Download completo!")
            return output_path
        except Exception as e:
            logger.error(f"Erro ao baixar do link direto {url}: {e}")
            raise e


def download_from_gdrive(url: str, output_path: Path) -> Path:
    """Baixa arquivo do Google Drive"""
    if output_path.exists():
        non_zip_items = [
            item for item in output_path.parent.iterdir() if item.name.endswith(".zip")
        ]

        if non_zip_items:
            logger.debug(f"Arquivo já existe: {output_path.name})")
            return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Baixando: {url}")

    try:
        # Método 1: URL direta com fuzzy matching
        gdown.download(url, str(output_path), quiet=False, fuzzy=True)
    except Exception as e1:
        logger.warning(f"Método 1 falhou: {e1}")
        try:
            # Método 2: Extrai ID e tenta novamente
            if "/file/d/" in url:
                file_id = url.split("/file/d/")[1].split("/")[0]
                logger.info(f"Tentando com ID extraído: {file_id}")
                gdown.download(id=file_id, output=str(output_path), quiet=False)
            else:
                raise e1
        except Exception as e2:
            logger.error(f"Ambos os métodos falharam: {e2}")
            raise e2

    logger.info("✅ Download completo!")
    return output_path


def extract_zip(zip_path: Path, extract_to: Path) -> Path:
    sentinel = extract_to / f".{zip_path.name}.extracted"
    if sentinel.exists():
        logger.debug(f"{zip_path.name} já extraído (marcador encontrado)")
        return extract_to

    try:
        logger.info(f"Extraindo arquivo de {zip_path} para {extract_to}...")
        extract_to.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        sentinel.touch()
        logger.info("✅ Extração completa!")
        return extract_to
    except Exception as e:
        logger.error(f"Erro ao extrair ZIP: {e}")
        raise


def create_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def log_gpu_info():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.debug("✅ GPU detectada")
        for gpu in gpus:
            logger.debug(f" - {gpu}")
    else:
        logger.debug("❌ Nenhuma GPU detectada")


def register_preprocess_input(model_name: str):
    import tensorflow as tf

    model_name_lower = model_name.lower()

    preprocess_funcs = get_preprocess_input_dict()

    # Se o modelo for None (como EfficientNetV2) ou não listado, usamos um fallback inofensivo lambda x: x
    preprocess_func = preprocess_funcs.get(model_name_lower, None)
    if preprocess_func is None:
        preprocess_func = lambda x: x

    tf.keras.utils.get_custom_objects()["preprocess_input"] = preprocess_func
    logger.info(f"✅ Registered preprocess_input for model: {model_name}")


def get_preprocess_input_dict() -> dict:
    """
    Retorna o dicionário de mapeamento de funções de pré-processamento
    nativas de cada arquitetura do Keras.
    """
    return {
        "mobilenetv3large": tf.keras.applications.mobilenet_v3.preprocess_input,
        "mobilenetv3small": tf.keras.applications.mobilenet_v3.preprocess_input,
        "inceptionv3": tf.keras.applications.inception_v3.preprocess_input,
        "vgg19": tf.keras.applications.vgg19.preprocess_input,
        "nasnetlarge": tf.keras.applications.nasnet.preprocess_input,
        "nasnetmobile": tf.keras.applications.nasnet.preprocess_input,
        "efficientnetb0": tf.keras.applications.efficientnet.preprocess_input,
        "efficientnetb1": tf.keras.applications.efficientnet.preprocess_input,
        "efficientnetb2": tf.keras.applications.efficientnet.preprocess_input,
        "efficientnetb3": tf.keras.applications.efficientnet.preprocess_input,
        "efficientnetb7": tf.keras.applications.efficientnet.preprocess_input,
        "efficientnetv2b0": None,
        "efficientnetv2b1": None,
        "efficientnetv2b2": None,
        "efficientnetv2b3": None,
        "convnexttiny": tf.keras.applications.convnext.preprocess_input,
        "convnextsmall": tf.keras.applications.convnext.preprocess_input,
    }
