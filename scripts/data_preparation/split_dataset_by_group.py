#!/usr/bin/env python3
import os
import sys
import shutil
import random
import re
from pathlib import Path
from collections import defaultdict

# Configurações globais
RANDOM_SEED = 42
TRAIN_RATIO = 0.9  # 90% para treino, 10% para validação

ROOT_DIR = Path(__file__).parent.parent.parent
INPUT_DIR = ROOT_DIR / "artifacts/data/processed/DatasetPests-cropped"
OUTPUT_DIR = ROOT_DIR / "artifacts/data/final/DatasetPests-split"

def get_group_id(filename: str) -> str:
    """
    Limpa o nome do arquivo de crop para obter o ID único da imagem original (imagem-mãe).
    Remove sufixos de crop, marcadores de original/aumentado, e prefixos de data/hora.
    """
    # 1. Remover extensão
    name = Path(filename).stem
    
    # 2. Remover crop (_crop_X)
    if "_crop_" in name:
        name = name.split("_crop_")[0]
        
    # 3. Remover marcador de original (_orig)
    name = name.replace("_orig", "")
    
    # 4. Remover marcadores de augmentação offline (_augXXXX)
    name = re.sub(r'_aug\d+', '', name)
    
    # 5. Remover datas/horas como prefixo (ex: "2026-06-02 ")
    name = re.sub(r'\d{4}-\d{2}-\d{2}\s*', '', name)
    
    # 6. Normalizar
    return name.lower().strip()

def main():
    print(f"📁 Diretório de origem: {INPUT_DIR}")
    print(f"📁 Diretório de destino: {OUTPUT_DIR}")

    if not INPUT_DIR.exists():
        print(f"❌ Erro: Diretório de origem {INPUT_DIR} não encontrado.")
        return

    # Limpar/Criar diretório de saída
    if OUTPUT_DIR.exists():
        print(f"🧹 Limpando diretório existente: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    
    (OUTPUT_DIR / "train").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "val").mkdir(parents=True, exist_ok=True)

    # Buscar pastas de classes
    class_folders = [d for d in INPUT_DIR.iterdir() if d.is_dir()]
    print(f"🔍 Encontradas {len(class_folders)} classes.")

    total_train_images = 0
    total_val_images = 0
    total_parent_images = 0

    print("\n--- Iniciando Split por Grupos Seguros (Imagem-Mãe + Augmentações Offline) ---")

    for folder in sorted(class_folders):
        class_name = folder.name
        
        # Encontrar todas as imagens de crops da classe
        crop_files = [
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]

        if not crop_files:
            print(f"⚠️ Classe {class_name} não possui imagens. Pulando...")
            continue

        # Agrupar crops pelo ID único da imagem-mãe original
        grouped_crops = defaultdict(list)
        for crop_path in crop_files:
            group_id = get_group_id(crop_path.name)
            grouped_crops[group_id].append(crop_path)

        parent_stems = sorted(list(grouped_crops.keys()))
        total_parent_images += len(parent_stems)

        # Embaralhar com seed fixa para reprodutibilidade
        rng = random.Random(RANDOM_SEED)
        rng.shuffle(parent_stems)

        # Split 90/10 dos grupos (imagens-mãe)
        split_idx = int(len(parent_stems) * TRAIN_RATIO)
        # Garantir que pelo menos 1 imagem-mãe vá para validação
        if split_idx == len(parent_stems) and len(parent_stems) > 1:
            split_idx = len(parent_stems) - 1

        train_parents = parent_stems[:split_idx]
        val_parents = parent_stems[split_idx:]

        # Criar subpastas de classe no destino
        train_class_dir = OUTPUT_DIR / "train" / class_name
        val_class_dir = OUTPUT_DIR / "val" / class_name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)

        class_train_count = 0
        class_val_count = 0

        # Copiar arquivos de treino
        for parent in train_parents:
            for crop_path in grouped_crops[parent]:
                shutil.copy2(crop_path, train_class_dir / crop_path.name)
                class_train_count += 1

        # Copiar arquivos de validação
        for parent in val_parents:
            for crop_path in grouped_crops[parent]:
                shutil.copy2(crop_path, val_class_dir / crop_path.name)
                class_val_count += 1

        total_train_images += class_train_count
        total_val_images += class_val_count

        print(f"📊 Classe: {class_name:30} | Imagens-Mãe Reais: {len(parent_stems):3d} | Treino: {class_train_count:4d} crops | Val: {class_val_count:4d} crops")

    print("\n--- Estatísticas Finais ---")
    print(f"Total de Imagens-Mãe Reais: {total_parent_images}")
    print(f"Total de Crops de Treino (Sem Vazamento): {total_train_images}")
    print(f"Total de Crops de Validação (Sem Vazamento): {total_val_images}")
    print(f"Proporção real: {total_train_images/(total_train_images+total_val_images)*100:.2f}% / {total_val_images/(total_train_images+total_val_images)*100:.2f}%")
    print(f"\n✅ Conjunto de dados dividido com sucesso em: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
