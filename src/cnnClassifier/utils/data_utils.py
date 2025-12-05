import zipfile
from pathlib import Path
import gdown
import tensorflow as tf
from cnnClassifier.utils.logger import configure_logger

logger = configure_logger(__name__)


def download_from_gdrive(url: str, output_path: Path) -> Path:
    """Baixa arquivo do Google Drive"""
    if output_path.exists():
        non_zip_items = [item for item in output_path.parent.iterdir() 
                if item.name.endswith('.zip')]

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
    if extract_to.exists():
        # Lista apenas arquivos/pastas que NÃO sejam .zip
        non_zip_items = [item for item in extract_to.iterdir() 
                        if not item.name.endswith('.zip')]
        
        if non_zip_items:
            logger.debug(f"{zip_path.name} já extraído ({len(non_zip_items)} items)")
            return extract_to
    
    try:
        logger.info(f"Extraindo arquivo de {zip_path} para {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info("✅ Extração completa!")
        return extract_to
    except Exception as e:
        logger.error(f"Erro ao extrair ZIP: {e}")
        raise

def create_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

def print_gpu_info():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("✅ GPU detectada")
        for gpu in gpus:
            print(f" - {gpu}")
    else:
        print("❌ Nenhuma GPU detectada")