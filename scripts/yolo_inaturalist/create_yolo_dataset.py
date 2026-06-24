#!/usr/bin/env python3
"""
Script para converter as anotações do Label-Studio (CSV) em formato YOLO (txt)
e criar a estrutura de pastas pronta para o treinamento do YOLOv8.
Evita vazamento de dados agrupando por imagem-mãe.
"""

import os
import csv
import json
import shutil
import random
import re
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Configurações globais
RANDOM_SEED = 42
TRAIN_RATIO = 0.9  # 90% para treino, 10% para validação

ROOT_DIR = Path(__file__).parent.parent.parent
DATASET_BASE_DIR = ROOT_DIR / "artifacts/data_ingestion/DatasetPests/Classes/Rotuladas"
YOLO_DIR = ROOT_DIR / "artifacts/data/yolo/yolo_dataset"

def get_group_id(filename: str) -> str:
    """Extrai o ID da imagem original (imagem-mãe) para evitar vazamento."""
    name = Path(filename).stem
    name = name.replace("_orig", "")
    name = re.sub(r'_aug\d+', '', name)
    name = re.sub(r'\d{4}-\d{2}-\d{2}\s*', '', name)
    return name.lower().strip()

def clean_class_name(name: str) -> str:
    """Remove sufixos do nome da classe."""
    for suffix in ["_final_shido", "_shido", "_final"]:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name

def normalize_name(name: str) -> str:
    """Normaliza nomes de arquivo para comparação."""
    name = name.lower()
    name = name.replace("orig_", "").replace("_orig", "").replace("orig", "")
    for char in [" ", "_", "-", "/"]:
        name = name.replace(char, "")
    return name

def find_image_path(img_name: str, disk_files: dict, folder_path: Path) -> Path:
    """Tenta encontrar a imagem no disco resolvendo variações de nome."""
    # 1. Tentar caminho direto
    img_path = folder_path / img_name
    if img_path.exists():
        return img_path
        
    # 2. Tentar busca normalizada
    base_name = Path(img_name).name
    norm_key = normalize_name(base_name)
    if norm_key in disk_files:
        return disk_files[norm_key]
        
    # 3. Tentar busca sem o prefixo de hash do Label-Studio
    if "-" in base_name:
        parts = base_name.split("-", 1)
        if len(parts) > 1:
            norm_key_clean = normalize_name(parts[1])
            if norm_key_clean in disk_files:
                return disk_files[norm_key_clean]
                
    # 4. Tentar busca de sufixo parcial
    for disk_norm, disk_path in disk_files.items():
        if disk_norm.endswith(norm_key) or norm_key.endswith(disk_norm):
            return disk_path
            
    return None

def prepare_yolo_dataset():
    print(f"📁 Origem: {DATASET_BASE_DIR}")
    print(f"📁 Destino YOLO: {YOLO_DIR}")
    
    if not DATASET_BASE_DIR.exists():
        print(f"❌ Diretório não encontrado: {DATASET_BASE_DIR}")
        return
        
    # Limpar/Criar pastas do YOLO
    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)
        
    for split in ["train", "val"]:
        (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    class_folders = [d for d in DATASET_BASE_DIR.iterdir() if d.is_dir()]
    
    # Coletar todas as observações com caixas por imagem-mãe
    grouped_data = defaultdict(list)
    
    for folder in sorted(class_folders):
        csv_files = list(folder.glob("*.csv"))
        if not csv_files:
            continue
            
        csv_path = csv_files[0]
        disk_files = {normalize_name(f.name): f for f in folder.iterdir() if f.is_file()}
        
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row.get("image")
                bbox_str = row.get("bbox")
                if not img_name or not bbox_str:
                    continue
                    
                img_path = find_image_path(img_name, disk_files, folder)
                if not img_path or not img_path.exists():
                    continue
                    
                try:
                    bboxes = json.loads(bbox_str)
                    if not bboxes:
                        continue
                        
                    # Converter bboxes para formato YOLO (Single class: 0 - pest)
                    yolo_labels = []
                    for bbox in bboxes:
                        x = bbox.get("x")
                        y = bbox.get("y")
                        w = bbox.get("width")
                        h = bbox.get("height")
                        
                        if x is None or y is None or w is None or h is None:
                            continue
                            
                        # iNaturalist/Label-studio usa coordenadas de 0 a 100 (porcentagem)
                        # YOLO precisa do centro (x, y) e largura/altura normatizados de 0 a 1
                        x_center = (x + w / 2.0) / 100.0
                        y_center = (y + h / 2.0) / 100.0
                        w_norm = w / 100.0
                        h_norm = h / 100.0
                        
                        # Limitar limites de segurança
                        x_center = max(0.0, min(1.0, x_center))
                        y_center = max(0.0, min(1.0, y_center))
                        w_norm = max(0.0, min(1.0, w_norm))
                        h_norm = max(0.0, min(1.0, h_norm))
                        
                        yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                        
                    if yolo_labels:
                        group_id = get_group_id(img_path.name)
                        grouped_data[group_id].append({
                            "img_path": img_path,
                            "labels": yolo_labels
                        })
                except Exception as e:
                    print(f"Erro ao parsear bounding box da imagem {img_name}: {e}")
                    
    # Fazer split seguro por imagens-mãe
    parent_ids = sorted(list(grouped_data.keys()))
    random.seed(RANDOM_SEED)
    random.shuffle(parent_ids)
    
    split_idx = int(len(parent_ids) * TRAIN_RATIO)
    train_parents = parent_ids[:split_idx]
    val_parents = parent_ids[split_idx:]
    
    # Copiar imagens e salvar arquivos .txt
    def process_split(parents, split_name):
        count = 0
        for pid in tqdm(parents, desc=f"Copiando split {split_name}"):
            for item in grouped_data[pid]:
                img_path = item["img_path"]
                labels = item["labels"]
                
                # Definir nomes dos arquivos de destino
                dest_img_path = YOLO_DIR / "images" / split_name / img_path.name
                dest_lbl_path = YOLO_DIR / "labels" / split_name / f"{img_path.stem}.txt"
                
                # Copiar imagem
                shutil.copy2(img_path, dest_img_path)
                
                # Salvar labels
                dest_lbl_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
                count += 1
        return count

    train_count = process_split(train_parents, "train")
    val_count = process_split(val_parents, "val")
    
    # Criar arquivo de configuração dataset.yaml
    yaml_content = f"""# Configuração do Dataset de Detector de Pragas (Single Class)
path: {YOLO_DIR.absolute()}
train: images/train
val: images/val

names:
  0: pest
"""
    (YOLO_DIR / "dataset.yaml").write_text(yaml_content, encoding="utf-8")
    
    print("\n--- Estatísticas Finais ---")
    print(f"Total de Imagens de Treino: {train_count}")
    print(f"Total de Imagens de Validação: {val_count}")
    print(f"Arquivo de configuração gerado em: {YOLO_DIR / 'dataset.yaml'}")

if __name__ == "__main__":
    prepare_yolo_dataset()
