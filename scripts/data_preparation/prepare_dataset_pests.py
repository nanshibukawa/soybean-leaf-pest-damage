#!/usr/bin/env python3
import os
import sys
import csv
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import re

# Add root directory to python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cnnClassifier.utils.logger import configure_logger

logger = configure_logger("prepare_dataset_pests")

# Base directories
ROOT_DIR = Path(__file__).parent.parent.parent
DATASET_BASE_DIR = ROOT_DIR / "artifacts/data_ingestion/DatasetPests/Classes/Rotuladas"
OUTPUT_DIR = ROOT_DIR / "artifacts/data/processed/DatasetPests-cropped"

def normalize_name(name: str) -> str:
    """Normaliza nomes de arquivo removendo case, hífens, underscores, espaços e o sufixo 'orig'."""
    name = name.lower()
    name = name.replace("orig_", "").replace("_orig", "").replace("orig", "")
    for char in [" ", "_", "-", "/"]:
        name = name.replace(char, "")
    return name

def clean_class_name(name: str) -> str:
    """Remove sufixos como '_final_shido', '_shido' e '_final' do nome da classe."""
    for suffix in ["_final_shido", "_shido", "_final"]:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def find_fallback_match(csv_name: str, disk_files: dict) -> Path:
    csv_clean = csv_name.lower()
    if "-" in csv_clean:
        parts = csv_clean.split("-", 1)
        if len(parts[0]) <= 8:  # Se tiver prefixo de hash (ex: "30f12e82-nome.jpg")
            csv_clean = parts[1]
            
    csv_tokens = set(re.findall(r"[a-z0-9]+", csv_clean))
    csv_tokens = {t for t in csv_tokens if len(t) >= 3 and t not in ("jpg", "png", "orig")}
    
    best_match = None
    best_score = 0
    
    for disk_norm, disk_path in disk_files.items():
        disk_name_lower = disk_path.name.lower()
        disk_tokens = set(re.findall(r"[a-z0-9]+", disk_name_lower))
        disk_tokens = {t for t in disk_tokens if len(t) >= 3 and t not in ("jpg", "png", "orig")}
        
        intersection = csv_tokens.intersection(disk_tokens)
        score = len(intersection)
        
        # Verificar correspondência de augmentação
        csv_aug = re.search(r"aug\d+", csv_clean)
        disk_aug = re.search(r"aug\d+", disk_name_lower)
        if csv_aug or disk_aug:
            if (csv_aug.group() if csv_aug else None) != (disk_aug.group() if disk_aug else None):
                continue
                
        # Verificar correspondência de números longos (IDs)
        csv_long_digits = set(re.findall(r"\d{6,}", csv_clean))
        disk_long_digits = set(re.findall(r"\d{6,}", disk_name_lower))
        if csv_long_digits and disk_long_digits:
            if not csv_long_digits.intersection(disk_long_digits):
                continue
                
        if score > best_score:
            best_score = score
            best_match = disk_path
            
    if best_score >= 3:
        return best_match
    return None

def crop_dataset_pests():
    logger.info(f"Iniciando recorte do DatasetPests. Origem: {DATASET_BASE_DIR}")
    logger.info(f"Destino: {OUTPUT_DIR}")

    if not DATASET_BASE_DIR.exists():
        logger.error(f"Diretório de origem não existe: {DATASET_BASE_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Buscar pastas de classes
    class_folders = [d for d in DATASET_BASE_DIR.iterdir() if d.is_dir()]
    logger.info(f"Encontradas {len(class_folders)} pastas de classes.")

    total_images_processed = 0
    total_crops_saved = 0

    for folder in sorted(class_folders):
        class_name = folder.name
        logger.info(f"Processando classe: {class_name}")

        # Encontrar arquivo CSV de anotação
        csv_files = list(folder.glob("*.csv"))
        if not csv_files:
            logger.warning(f"Nenhum arquivo CSV de anotações encontrado em {folder.name}. Pulando...")
            continue

        # Usar o primeiro CSV encontrado
        csv_path = csv_files[0]
        logger.info(f"Lendo anotações de {csv_path.name}")

        # Criar dicionário de busca normalizada para os arquivos físicos
        disk_files = {normalize_name(f.name): f for f in folder.iterdir() if f.is_file()}

        # Criar pasta de saída da classe
        cleaned_class_name = clean_class_name(class_name)
        class_output_dir = OUTPUT_DIR / cleaned_class_name
        class_output_dir.mkdir(parents=True, exist_ok=True)

        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Converter linhas do leitor em lista para usar tqdm
            rows = list(reader)
            
            for row in tqdm(rows, desc=f"Copiando crops de {cleaned_class_name}"):
                img_name = row.get("image")
                if not img_name:
                    continue
                
                # 1. Tentar encontrar a imagem diretamente
                img_path = folder / img_name
                
                # 2. Se não existir, tentar resolver caminhos/nomes flexíveis usando normalização
                if not img_path.exists():
                    base_name = Path(img_name).name
                    
                    # Tentar encontrar a imagem normalizada
                    norm_key = normalize_name(base_name)
                    
                    # Se tiver prefixo de hash (ex: "30f12e82-nome.jpg"), obter também a versão limpa
                    norm_key_clean = None
                    if "-" in base_name:
                        parts = base_name.split("-", 1)
                        if len(parts) > 1:
                            norm_key_clean = normalize_name(parts[1])
                    
                    # 1. Tentar busca exata por chave normalizada
                    if norm_key in disk_files:
                        img_path = disk_files[norm_key]
                    elif norm_key_clean and norm_key_clean in disk_files:
                        img_path = disk_files[norm_key_clean]
                    else:
                        # 2. Tentar busca parcial (sufixo/prefixo), útil se no disco a imagem tiver um prefixo extra de classe
                        # Ex: no CSV "bulimulus_..." mas no disco "gastropoda_bulimulus_..."
                        for disk_norm_key, disk_file_path in disk_files.items():
                            if disk_norm_key.endswith(norm_key) or norm_key.endswith(disk_norm_key):
                                img_path = disk_file_path
                                break
                            elif norm_key_clean and (disk_norm_key.endswith(norm_key_clean) or norm_key_clean.endswith(disk_norm_key)):
                                img_path = disk_file_path
                                break
                    
                    # 3. Se ainda não existir, tentar o casamento de fallback fuzzy (tratando variações e truncamentos do Label Studio)
                    if not img_path.exists():
                        fallback_path = find_fallback_match(base_name, disk_files)
                        if fallback_path and fallback_path.exists():
                            img_path = fallback_path
                
                if not img_path.exists():
                    logger.warning(f"Imagem não encontrada: {img_name} (caminho tentado: {img_path})")
                    continue

                bbox_str = row.get("bbox")
                if not bbox_str:
                    # Sem anotação nesta linha
                    continue

                try:
                    # O formato de bbox no CSV é uma string JSON de uma lista de dicts
                    bboxes = json.loads(bbox_str)
                    
                    if not bboxes:
                        continue

                    with Image.open(img_path) as img:
                        w, h = img.size

                        for idx, bbox in enumerate(bboxes):
                            # Obter coordenadas em porcentagem
                            x = bbox.get("x")
                            y = bbox.get("y")
                            width = bbox.get("width")
                            height = bbox.get("height")

                            if x is None or y is None or width is None or height is None:
                                continue

                            # Calcular pixels
                            xmin = int((x / 100.0) * w)
                            ymin = int((y / 100.0) * h)
                            xmax = int(((x + width) / 100.0) * w)
                            ymax = int(((y + height) / 100.0) * h)

                            # Limitar coordenadas às dimensões da imagem
                            xmin = max(0, xmin)
                            ymin = max(0, ymin)
                            xmax = min(w, xmax)
                            ymax = min(h, ymax)

                            # Pular se área for inválida
                            if xmax <= xmin or ymax <= ymin:
                                continue

                            # Cortar
                            cropped_img = img.crop((xmin, ymin, xmax, ymax))

                            # Converter para RGB se a imagem tiver transparência ou paleta (JPEG não suporta RGBA)
                            if cropped_img.mode in ("RGBA", "P"):
                                cropped_img = cropped_img.convert("RGB")

                            # Salvar crop
                            output_filename = f"{img_path.stem}_crop_{idx}.jpg"
                            cropped_img.save(class_output_dir / output_filename, "JPEG")
                            total_crops_saved += 1

                    total_images_processed += 1
                except Exception as e:
                    logger.error(f"Erro ao processar imagem {img_name} no CSV: {str(e)}")

    logger.info(f"Concluído! Processadas {total_images_processed} imagens originais, gerando {total_crops_saved} recortes no total.")

if __name__ == "__main__":
    crop_dataset_pests()
