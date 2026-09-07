#!/usr/bin/env python3
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from loguru import logger

# Add root directory to python path
# Busca dinâmica da localização da fonte oficial do IP102
CANDIDATE_DIRS = [
    ROOT_DIR / "artifacts" / "data" / "ip102",
    ROOT_DIR / "artifacts" / "data_ingestion" / "ip102",
    Path.home() / "Downloads" / "IP102_v1.1-20260704T201959Z-3-001" / "IP102_v1.1",
]

IP102_OFFICIAL_DIR = None
for candidate in CANDIDATE_DIRS:
    if (candidate / "Detection" / "VOC2007" / "Annotations").exists():
        IP102_OFFICIAL_DIR = candidate
        break

OUTPUT_DIR = ROOT_DIR / "artifacts" / "data" / "ip102_cropped"

def crop_official_ip102():
    if not IP102_OFFICIAL_DIR:
        logger.error(f"❌ Pastas de anotações XML do IP102 não encontradas. Tentou em: {CANDIDATE_DIRS}")
        return

    annotations_dir = IP102_OFFICIAL_DIR / "Detection" / "VOC2007" / "Annotations"
    images_dir = IP102_OFFICIAL_DIR / "Detection" / "VOC2007" / "JPEGImages"

    if not annotations_dir.exists() or not images_dir.exists():
        logger.error(f"❌ Pastas de anotações ou imagens não encontradas em: {IP102_OFFICIAL_DIR}")
        return

    logger.info(f"📂 Fonte de anotações do IP102 localizada em: {IP102_OFFICIAL_DIR}")

    logger.info("📂 Escaneando arquivos XML de anotação do IP102...")
    xml_files = list(annotations_dir.glob("*.xml"))
    
    crops_info = [] # Lista de tuplas (crop_pil_image, class_name, filename_stem)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}

    logger.info("✂️ Processando e recortando imagens baseando-se nas anotações XML...")
    for xml_path in tqdm(xml_files, desc="Processando XMLs"):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Obter nome do arquivo de imagem correspondente
            filename_xml = root.find("filename").text
            img_path = images_dir / filename_xml
            if not img_path.exists():
                # Fallback case-insensitive
                base_name = Path(filename_xml).stem
                img_path = None
                for ext in image_extensions:
                    temp_path = images_dir / f"{base_name}{ext}"
                    if temp_path.exists():
                        img_path = temp_path
                        break
            
            if not img_path or not img_path.exists():
                continue

            with Image.open(img_path) as img:
                W, H = img.size
                
                # Obter objetos (bounding boxes)
                for obj_idx, obj in enumerate(root.findall("object")):
                    class_name = obj.find("name").text.strip()
                    
                    bndbox = obj.find("bndbox")
                    xmin = int(float(bndbox.find("xmin").text))
                    ymin = int(float(bndbox.find("ymin").text))
                    xmax = int(float(bndbox.find("xmax").text))
                    ymax = int(float(bndbox.find("ymax").text))
                    
                    # Truncar limites da imagem
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(W, xmax)
                    ymax = min(H, ymax)
                    
                    if (xmax - xmin) < 5 or (ymax - ymin) < 5:
                        continue
                        
                    # Fazer o crop temporariamente na memória para split stratified subsequente
                    # Ou salvar em uma lista para processar depois
                    crops_info.append({
                        "img_path": img_path,
                        "box": (xmin, ymin, xmax, ymax),
                        "class_name": class_name,
                        "obj_idx": obj_idx
                    })
                    
        except Exception as e:
            logger.error(f"❌ Erro ao processar XML {xml_path.name}: {e}")

    logger.info(f"📊 Total de recortes (crops) identificados: {len(crops_info)}")
    if not crops_info:
        logger.error("❌ Nenhum recorte encontrado para processar.")
        return

    # Extrair classes para estratificação
    classes = [c["class_name"] for c in crops_info]
    
    logger.info("⚖️ Dividindo os recortes em treino (90%) e validação (10%) de forma estratificada...")
    train_info, val_info = train_test_split(
        crops_info,
        test_size=0.1,
        random_state=42,
        stratify=classes
    )
    
    # Função auxiliar para salvar os crops fisicamente
    def save_subset_crops(subset_info, subset_name):
        subset_dir = OUTPUT_DIR / subset_name
        subset_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Salvando recortes do subset '{subset_name}'...")
        for item in tqdm(subset_info, desc=f"Salvando {subset_name}"):
            try:
                with Image.open(item["img_path"]) as img:
                    cropped = img.crop(item["box"])
                    
                    # Criar subpasta da classe
                    class_dir = subset_dir / item["class_name"]
                    class_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Nome único
                    output_name = f"{item['img_path'].stem}_crop_{item['obj_idx']}.jpg"
                    cropped.save(class_dir / output_name, "JPEG")
            except Exception as e:
                logger.error(f"❌ Erro ao salvar crop de {item['img_path'].name}: {e}")

    save_subset_crops(train_info, "train")
    save_subset_crops(val_info, "val")
    logger.info(f"🎉 Dataset de classificação oficial do IP102 (102 classes) criado em: {OUTPUT_DIR}")

def main():
    logger.info("🚀 Iniciando processamento do dataset oficial do IP102...")
    if not IP102_OFFICIAL_DIR.exists():
        logger.error(f"❌ Diretório IP102 não encontrado em: {IP102_OFFICIAL_DIR}")
        return

    crop_official_ip102()

if __name__ == "__main__":
    main()
