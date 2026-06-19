#!/usr/bin/env python3
import os
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Add root directory to python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cnnClassifier.utils.logger import configure_logger

logger = configure_logger("prepare_insect12c")

# Base directories
ROOT_DIR = Path(__file__).parent.parent
DATASET_BASE_DIR = ROOT_DIR / "artifacts/data_ingestion/INSECT12C-Dataset-main"
OUTPUT_12_CLASSES_DIR = ROOT_DIR / "artifacts/data/INSECT12C-cropped-12-classes"
OUTPUT_10_CLASSES_DIR = ROOT_DIR / "artifacts/data/INSECT12C-cropped-10-classes"

# 10-class mapping to match the current pipeline (DatasetPests structure)
# Maps the 12 classes of INSECT12C to the 10 folders expected in the current pipeline.
CLASS_MAPPING_10 = {
    "Anticarsia_gemmatalis": "anticarsia_gemmatalis-larva",
    "Coccinellidae": "coccinellidae",
    "Diabrotica_speciosa": "diabrotica_speciosa",
    "Edessa_meditabunda": "edessa_meditabunda",
    "Euschistus_heros_adulto": "euschistus_heros",
    "Euschistus_heros_ninfa": "euschistus_heros",
    "Gastropoda": "gastropoda",
    "Lagria_villosa": "lagria_villosa",
    "Nezara_viridula_adulto": "nezara_viridula",
    "Nezara_viridula_ninfa": "nezara_viridula",
    "Rhammatocerus_schistocercoides": "rhammatocerus_schistocercoides",
    "Spodoptera_albula": "spodoptera_albula",
}

def extract_zips():
    """Extract all zip files in the dataset folder."""
    zip_files = list(DATASET_BASE_DIR.glob("*.zip"))
    logger.info(f"Encontrados {len(zip_files)} arquivos zip para extrair.")
    
    for zip_path in sorted(zip_files):
        folder_name = zip_path.stem
        extract_to = DATASET_BASE_DIR / folder_name
        
        if extract_to.exists():
            logger.info(f"Skipping extraction for {zip_path.name} (diretório já existe).")
            continue
            
        logger.info(f"Extraindo {zip_path.name} para {extract_to.name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"Concluído: {zip_path.name}")

def crop_objects(use_10_classes: bool = True):
    """Parse XML annotations and crop objects into class-based directories."""
    output_dir = OUTPUT_10_CLASSES_DIR if use_10_classes else OUTPUT_12_CLASSES_DIR
    logger.info(f"Iniciando recorte dos insetos. Destino: {output_dir}")
    
    # Create output dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all extracted directories
    extracted_dirs = [d for d in DATASET_BASE_DIR.iterdir() if d.is_dir() and d.name.startswith("part-")]
    
    total_images_processed = 0
    total_crops_saved = 0
    
    for part_dir in sorted(extracted_dirs):
        logger.info(f"Processando diretório: {part_dir.name}")
        xml_files = list(part_dir.rglob("*.xml"))
        
        for xml_path in tqdm(xml_files, desc=f"Lendo XMLs de {part_dir.name}"):
            try:
                # Parse XML
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # Encontrar imagem correspondente no mesmo diretório do XML
                filename_xml = root.find("filename").text
                img_parent_dir = xml_path.parent
                img_path = img_parent_dir / filename_xml
                if not img_path.exists():
                    # Tentar case-insensitive para a extensão
                    base_name = Path(filename_xml).stem
                    matching_images = list(img_parent_dir.glob(f"{base_name}.*"))
                    # Filtra apenas formatos de imagem comuns
                    matching_images = [img for img in matching_images if img.suffix.lower() in [".jpg", ".jpeg", ".png"]]
                    if matching_images:
                        img_path = matching_images[0]
                    else:
                        logger.warning(f"Imagem correspondente não encontrada para {xml_path.name} (esperado: {filename_xml})")
                        continue
                
                # Abrir imagem
                with Image.open(img_path) as img:
                    width, height = img.size
                    
                    # Processar objetos no XML
                    for obj_idx, obj in enumerate(root.findall("object")):
                        class_name = obj.find("name").text.strip()
                        
                        # Bounding box
                        bndbox = obj.find("bndbox")
                        xmin = int(float(bndbox.find("xmin").text))
                        ymin = int(float(bndbox.find("ymin").text))
                        xmax = int(float(bndbox.find("xmax").text))
                        ymax = int(float(bndbox.find("ymax").text))
                        
                        # Garantir limites da imagem
                        xmin = max(0, xmin)
                        ymin = max(0, ymin)
                        xmax = min(width, xmax)
                        ymax = min(height, ymax)
                        
                        # Pular se box for inválida ou sem área
                        if xmax <= xmin or ymax <= ymin:
                            continue
                            
                        # Recortar
                        cropped_img = img.crop((xmin, ymin, xmax, ymax))
                        
                        # Definir diretório de destino
                        if use_10_classes:
                            mapped_class = CLASS_MAPPING_10.get(class_name)
                            if not mapped_class:
                                logger.warning(f"Classe desconhecida '{class_name}' em {xml_path.name}, pulando mapeamento.")
                                continue
                            class_output_dir = output_dir / mapped_class
                        else:
                            class_output_dir = output_dir / class_name
                            
                        class_output_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Salvar imagem recortada
                        # Nome único baseado no nome da imagem original e index do objeto
                        output_filename = f"{img_path.stem}_crop_{obj_idx}.jpg"
                        cropped_img.save(class_output_dir / output_filename, "JPEG")
                        total_crops_saved += 1
                        
                total_images_processed += 1
                
            except Exception as e:
                logger.error(f"Erro ao processar {xml_path.name}: {str(e)}")
                
    logger.info(f"Concluído! Processadas {total_images_processed} imagens, gerando {total_crops_saved} recortes de insetos.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extrai e recorta o dataset INSECT12C baseando-se nos XMLs Pascal VOC.")
    parser.add_argument("--classes-12", action="store_true", help="Gera o dataset com 12 classes originais (separa ninfas e adultos).")
    args = parser.parse_args()
    
    # 1. Extrair zips se necessário
    extract_zips()
    
    # 2. Recortar
    use_10_classes = not args.classes_12
    crop_objects(use_10_classes=use_10_classes)
