import os
import tarfile
import urllib.request
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from loguru import logger

# Configurações de caminhos
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CAT_DOGS_DIR = ROOT_DIR / "artifacts" / "data" / "cat_dogs"
IMAGES_DIR = CAT_DOGS_DIR / "images"
ANNOTATIONS_TAR = CAT_DOGS_DIR / "annotations.tar.gz"
ANNOTATIONS_DIR = CAT_DOGS_DIR / "annotations"

OUTPUT_DIR = ROOT_DIR / "artifacts" / "data" / "final" / "oxford_pets-split"
TRAIN_OUT = OUTPUT_DIR / "train"
VAL_OUT = OUTPUT_DIR / "val"
TEST_OUT = OUTPUT_DIR / "test"

ANNOTATIONS_URL = "https://thor.robots.ox.ac.uk/pets/annotations.tar.gz"

def download_annotations():
    """Baixa as anotações oficiais se não existirem."""
    if ANNOTATIONS_TAR.exists():
        logger.info(f"📝 Arquivo de anotações {ANNOTATIONS_TAR.name} já existe. Pulando download.")
        return
    
    logger.info(f"📥 Baixando anotações oficiais de: {ANNOTATIONS_URL}")
    CAT_DOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Callback de progresso simples
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        if percent % 10 == 0:
            logger.info(f"📊 Download progresso: {min(percent, 100)}%")
            
    urllib.request.urlretrieve(ANNOTATIONS_URL, ANNOTATIONS_TAR, progress_hook)
    logger.info("✅ Download de anotações concluído!")

def extract_annotations():
    """Extrai o arquivo tar.gz das anotações."""
    if ANNOTATIONS_DIR.exists() and (ANNOTATIONS_DIR / "trainval.txt").exists():
        logger.info("📝 Pasta de anotações já extraída. Pulando extração.")
        return
        
    logger.info(f"📦 Extraindo {ANNOTATIONS_TAR}...")
    with tarfile.open(ANNOTATIONS_TAR, "r:gz") as tar:
        tar.extractall(path=CAT_DOGS_DIR)
    logger.info("✅ Extração concluída!")

def parse_annotation_file(file_path: Path):
    """Lê o arquivo de anotação oficial e retorna listas de imagens e classes correspondentes."""
    images = []
    classes = []
    
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            image_name = parts[0]
            # No Oxford Pet, o nome da classe é derivado do nome da imagem (removendo o número final)
            class_name = "_".join(image_name.split("_")[:-1])
            
            # Verificar se a imagem física existe antes de adicionar
            img_file = IMAGES_DIR / f"{image_name}.jpg"
            if img_file.exists():
                images.append(image_name)
                classes.append(class_name)
            else:
                logger.warning(f"⚠️ Imagem física não encontrada: {img_file}")
                
    return images, classes

def create_symlinks(image_names, classes, target_dir: Path):
    """Cria links simbólicos organizados em subpastas de classes."""
    target_dir.mkdir(parents=True, exist_ok=True)
    
    for img_name, class_name in zip(image_names, classes):
        class_dir = target_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        src_path = (IMAGES_DIR / f"{img_name}.jpg").resolve()
        dest_path = class_dir / f"{img_name}.jpg"
        
        if not dest_path.exists():
            os.symlink(src_path, dest_path)

def main():
    logger.info("🐾 Iniciando Preparação do Oxford-IIIT Pet Dataset...")
    
    # 1. Downloads e Extrações
    download_annotations()
    extract_annotations()
    
    # 2. Ler splits oficiais
    trainval_txt = ANNOTATIONS_DIR / "trainval.txt"
    test_txt = ANNOTATIONS_DIR / "test.txt"
    
    logger.info("📖 Lendo arquivos de anotação oficiais...")
    trainval_images, trainval_classes = parse_annotation_file(trainval_txt)
    test_images, test_classes = parse_annotation_file(test_txt)
    
    logger.info(f"📊 Total trainval oficial: {len(trainval_images)} imagens")
    logger.info(f"📊 Total test oficial: {len(test_images)} imagens")
    
    # 3. Dividir trainval em 90% treino e 10% validação de forma estratificada
    logger.info("⚖️ Dividindo trainval em treino (90%) e validação (10%) de forma estratificada...")
    train_images, val_images, train_classes, val_classes = train_test_split(
        trainval_images,
        trainval_classes,
        test_size=0.1,
        random_state=42,
        stratify=trainval_classes
    )
    
    logger.info(f"📈 Conjunto de Treino: {len(train_images)} imagens")
    logger.info(f"📈 Conjunto de Validação: {len(val_images)} imagens")
    logger.info(f"📈 Conjunto de Teste: {len(test_images)} imagens")
    
    # 4. Limpar diretório de saída se necessário
    if OUTPUT_DIR.exists():
        logger.info(f"🧹 Limpando links simbólicos antigos de {OUTPUT_DIR.name}...")
        # Remove os links e diretórios
        import shutil
        shutil.rmtree(OUTPUT_DIR)
        
    # 5. Criar os links simbólicos organizados
    logger.info("🔗 Criando links simbólicos organizados por pastas de classes...")
    create_symlinks(train_images, train_classes, TRAIN_OUT)
    create_symlinks(val_images, val_classes, VAL_OUT)
    create_symlinks(test_images, test_classes, TEST_OUT)
    
    # Mostrar distribuição
    unique_classes = sorted(list(set(train_classes)))
    logger.info(f"✅ Dataset pronto com {len(unique_classes)} classes!")
    logger.info(f"📁 Localização: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
