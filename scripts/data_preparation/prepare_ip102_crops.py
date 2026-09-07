#!/usr/bin/env python3
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from loguru import logger

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
IP102_YOLO_DIR = ROOT_DIR / "artifacts" / "data" / "ip102"
OUTPUT_DIR = ROOT_DIR / "artifacts" / "data" / "ip102_cropped"

def crop_yolo_dataset(subset: str):
    images_dir = IP102_YOLO_DIR / subset / "images"
    labels_dir = IP102_YOLO_DIR / subset / "labels"
    dest_dir = OUTPUT_DIR / subset

    if not images_dir.exists() or not labels_dir.exists():
        logger.warning(f"⚠️ Diretórios para o subset '{subset}' não encontrados. Pulando.")
        return

    logger.info(f"✂️ Processando recortes para o subset '{subset}'...")
    
    # Extensões comuns de imagem
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}
    
    # Listar arquivos de labels
    label_files = list(labels_dir.glob("*.txt"))
    total_crops = 0

    for label_path in tqdm(label_files, desc=f"Recortando {subset}"):
        # Encontrar imagem correspondente
        base_name = label_path.stem
        img_path = None
        for ext in image_extensions:
            temp_path = images_dir / f"{base_name}{ext}"
            if temp_path.exists():
                img_path = temp_path
                break
        
        if not img_path:
            logger.warning(f"⚠️ Imagem não encontrada para o label: {label_path.name}")
            continue

        try:
            with Image.open(img_path) as img:
                W, H = img.size
                
                # Ler anotações do arquivo YOLO
                with open(label_path, "r") as f:
                    lines = f.readlines()
                
                for idx, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = parts[0]
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    w_norm = float(parts[3])
                    h_norm = float(parts[4])
                    
                    # Converter coordenadas normalizadas para coordenadas de pixel absolutas
                    w_box = w_norm * W
                    h_box = h_norm * H
                    x_center_abs = x_center * W
                    y_center_abs = y_center * H
                    
                    xmin = max(0, int(x_center_abs - w_box / 2))
                    ymin = max(0, int(y_center_abs - h_box / 2))
                    xmax = min(W, int(x_center_abs + w_box / 2))
                    ymax = min(H, int(y_center_abs + h_box / 2))
                    
                    # Evitar recortes com tamanho inválido
                    if (xmax - xmin) < 5 or (ymax - ymin) < 5:
                        continue
                        
                    # Recortar e salvar
                    cropped_img = img.crop((xmin, ymin, xmax, ymax))
                    class_dir = dest_dir / class_id
                    class_dir.mkdir(parents=True, exist_ok=True)
                    
                    output_name = f"{base_name}_crop_{idx}.jpg"
                    cropped_img.save(class_dir / output_name, "JPEG")
                    total_crops += 1
                    
        except Exception as e:
            logger.error(f"❌ Erro ao processar {img_path.name}: {e}")

    logger.info(f"✅ Subset '{subset}' concluído: {total_crops} recortes salvos.")

def main():
    logger.info("🚀 Iniciando crop do dataset IP102 (formato YOLO) para classificação...")
    if not IP102_YOLO_DIR.exists():
        logger.error(f"❌ Diretório IP102 não encontrado em: {IP102_YOLO_DIR}")
        return

    crop_yolo_dataset("train")
    crop_yolo_dataset("val")
    crop_yolo_dataset("test")
    logger.info(f"🎉 Dataset de classificação criado com sucesso em: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
