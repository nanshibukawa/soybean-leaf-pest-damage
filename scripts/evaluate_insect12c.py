#!/usr/bin/env python3
"""
Script de Avaliação Zero-Shot de Robustez no Dataset INSECT12C
Avalia o modelo treinado (Passo 2) no dataset de teste externo INSECT12C em 3 cenários de resolução:
1. Dataset Completo (Sem filtro)
2. Imagens Médias (> 60x60 px)
3. Imagens Grandes (> 100x100 px)
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from cnnClassifier.utils.logger import configure_logger
from cnnClassifier.components.prepare_model import PrepareModel, TopKGlobalAveragePooling2D
from cnnClassifier.models.custom_blocks import SpatialAttentionFeatureReduction, ResidualSRCNNBlock

logger = configure_logger("evaluate_insect12c")

from cnnClassifier.utils.data_utils import register_preprocess_input

DEFAULT_MODEL_PATH = ROOT_DIR / "artifacts/models/efficientnetv2b1_trained.keras"
INSECT12C_TEST_DIR = ROOT_DIR / "artifacts/data/final/INSECT12C-test"
OUTPUT_REPORT_PATH = ROOT_DIR / "artifacts/model_evaluation/insect12c_benchmark_report.json"

def load_and_preprocess_image(img_path: Path, target_size=(240, 240)):
    """Carrega e pré-processa uma imagem para inferência (faixa [0, 255] original)."""
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img_resized = img.resize(target_size)
        img_array = np.array(img_resized, dtype=np.float32)
        # Mantém a faixa [0, 255] pois o preprocess_input da EfficientNetV2 trata internamente
        return img_array

def evaluate_scenario(model, samples, class_names, target_size=(240, 240)):
    """Executa a avaliação para uma lista de amostras (caminho_imagem, rotulo_idx)."""
    if not samples:
        return {"error": "Nenhuma amostra encontrada para este cenário."}

    y_true = []
    y_pred = []

    logger.info(f"⏳ Processando inferência em {len(samples)} imagens...")
    batch_images = []
    batch_labels = []
    batch_size = 32

    for img_path, label_idx in tqdm(samples, desc="Avaliando"):
        try:
            img_array = load_and_preprocess_image(img_path, target_size=target_size)
            batch_images.append(img_array)
            batch_labels.append(label_idx)

            if len(batch_images) == batch_size:
                preds = model.predict(np.array(batch_images), verbose=0)
                pred_labels = np.argmax(preds, axis=1)
                y_pred.extend(pred_labels)
                y_true.extend(batch_labels)
                batch_images = []
                batch_labels = []
        except Exception as e:
            logger.warning(f"⚠️ Erro ao processar {img_path.name}: {e}")

    if batch_images:
        preds = model.predict(np.array(batch_images), verbose=0)
        pred_labels = np.argmax(preds, axis=1)
        y_pred.extend(pred_labels)
        y_true.extend(batch_labels)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    return {
        "total_samples": len(samples),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": report
    }

def main():
    parser = argparse.ArgumentParser(description="Avaliação Zero-Shot de Robustez no INSECT12C")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Caminho para o modelo .keras treinado"
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"❌ Modelo não encontrado em: {model_path}")
        sys.exit(1)

    if not INSECT12C_TEST_DIR.exists():
        logger.error(f"❌ Diretório de teste INSECT12C não encontrado em: {INSECT12C_TEST_DIR}")
        sys.exit(1)

    logger.info(f"🚀 Carregando modelo treinado: {model_path.name}...")
    register_preprocess_input("efficientnetv2b1")
    custom_objects = {
        "TopKGlobalAveragePooling2D": TopKGlobalAveragePooling2D,
        "SpatialAttentionFeatureReduction": SpatialAttentionFeatureReduction,
        "ResidualSRCNNBlock": ResidualSRCNNBlock,
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

    # Identificar classes em ordem alfabética (mesmo padrão do Keras DirectoryIterator)
    class_folders = sorted([d for d in INSECT12C_TEST_DIR.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_folders]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    logger.info(f"🏷️ Classes identificadas ({len(class_names)}): {class_names}")

    # Coletar todas as imagens com dimensões (W, H)
    all_samples = []
    medium_samples = [] # > 60x60 px
    large_samples = []  # > 100x100 px

    for folder in class_folders:
        label_idx = class_to_idx[folder.name]
        for img_file in folder.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                try:
                    with Image.open(img_file) as img:
                        w, h = img.size
                    sample = (img_file, label_idx)
                    all_samples.append(sample)
                    if w > 60 and h > 60:
                        medium_samples.append(sample)
                    if w > 100 and h > 100:
                        large_samples.append(sample)
                except Exception as e:
                    logger.warning(f"⚠️ Erro ao ler imagem {img_file.name}: {e}")

    logger.info(f"📊 Total de imagens encontradas: {len(all_samples)}")
    logger.info(f"📏 Imagens Médias (> 60x60 px): {len(medium_samples)}")
    logger.info(f"📏 Imagens Grandes (> 100x100 px): {len(large_samples)}")

    # Detectar dinamicamente a resolução esperada pelo modelo
    in_shape = model.input_shape
    if in_shape and in_shape[1] is not None and in_shape[2] is not None:
        target_size = (int(in_shape[1]), int(in_shape[2]))
    else:
        target_size = (224, 224)
    logger.info(f"📐 Resolução de entrada detectada do modelo: {target_size}")

    # Executar avaliações nos 3 cenários
    results = {}
    
    logger.info("\n--- 1. Avaliando Dataset Completo (Sem Filtros) ---")
    results["all"] = evaluate_scenario(model, all_samples, class_names, target_size=target_size)

    logger.info("\n--- 2. Avaliando Imagens Médias (> 60x60 px) ---")
    results["medium_gt_60"] = evaluate_scenario(model, medium_samples, class_names, target_size=target_size)

    logger.info("\n--- 3. Avaliando Imagens Grandes (> 100x100 px) ---")
    results["large_gt_100"] = evaluate_scenario(model, large_samples, class_names, target_size=target_size)

    # Imprimir Tabela Resumo no Terminal
    print("\n" + "=" * 80)
    print(" 📊 RESUMO DO BENCHMARK DE GENERALIZAÇÃO EXTERNA (INSECT12C)")
    print("=" * 80)
    print(f"{'Cenário de Avaliação':<30} | {'Amostras':<10} | {'Acurácia Global':<16} | {'Macro F1-Score':<16}")
    print("-" * 80)
    
    for key, name in [("all", "Dataset Completo (Sem filtro)"), ("medium_gt_60", "Médias (> 60x60 px)"), ("large_gt_100", "Grandes (> 100x100 px)")]:
        res = results.get(key, {})
        acc_str = f"{res.get('accuracy', 0)*100:.2f}%" if "accuracy" in res else "N/A"
        f1_str = f"{res.get('macro_f1', 0)*100:.2f}%" if "macro_f1" in res else "N/A"
        print(f"{name:<30} | {res.get('total_samples', 0):<10} | {acc_str:<16} | {f1_str:<16}")
        
    print("=" * 80 + "\n")

    # Salvamento do Relatório JSON
    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"📄 Relatório completo de benchmark salvo em: {OUTPUT_REPORT_PATH}")

if __name__ == "__main__":
    main()
