#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

# Add root directory to python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from cnnClassifier.entity.config_entity import ModelConfig, ImageConfig
from cnnClassifier.components.prepare_model import TopKGlobalAveragePooling2D
from cnnClassifier.utils.logger import configure_logger
from cnnClassifier.utils.data_utils import register_preprocess_input

logger = configure_logger("evaluate_resolution_impact")

def evaluate_resolution_impact(model_path: Path, val_dir: Path, experiment: str):
    logger.info("📋 Carregando configurações...")
    model_config = ModelConfig.from_yaml("model_params.yaml", experiment=experiment)
    
    # Obter dimensões do modelo
    target_height = model_config.image_size[0]
    target_width = model_config.image_size[1]
    
    logger.info(f"📥 Carregando classes do diretório: {val_dir}")
    class_names = sorted([d.name for d in val_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    logger.info(f"Mapeamento de classes: {class_to_idx}")
    
    logger.info(f"📥 Carregando modelo treinado: {model_path}")
    register_preprocess_input(model_config.model_name)
    
    from cnnClassifier.models.custom_blocks import SpatialAttentionFeatureReduction, ResidualSRCNNBlock
    from cnnClassifier.components.prepare_model import PreprocessingLayer
    
    try:
        model = tf.keras.models.load_model(
            str(model_path),
            custom_objects={
                "TopKGlobalAveragePooling2D": TopKGlobalAveragePooling2D,
                "SpatialAttentionFeatureReduction": SpatialAttentionFeatureReduction,
                "PreprocessingLayer": PreprocessingLayer,
                "ResidualSRCNNBlock": ResidualSRCNNBlock
            }
        )
    except Exception as e:
        logger.error(f"❌ Erro ao carregar o modelo: {e}")
        sys.exit(1)
        
    # Listar todas as imagens da pasta de validação
    image_paths = []
    y_true = []
    
    for class_name in class_names:
        class_idx = class_to_idx[class_name]
        class_path = val_dir / class_name
        for img_file in class_path.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                image_paths.append(img_file)
                y_true.append(class_idx)
                
    if not image_paths:
        logger.error("❌ Nenhuma imagem de validação encontrada.")
        sys.exit(1)
        
    logger.info(f"🔍 Encontradas {len(image_paths)} imagens para avaliação.")
    
    # Buckets de resolução
    # Pequena: < 60px
    # Média: 60px - 120px
    # Alta: > 120px
    buckets = {
        "pequena": {"max_size": 60, "true": [], "pred": [], "paths": [], "sizes": []},
        "media": {"min_size": 60, "max_size": 120, "true": [], "pred": [], "paths": [], "sizes": []},
        "alta": {"min_size": 120, "true": [], "pred": [], "paths": [], "sizes": []}
    }
    
    # Iterar e classificar
    logger.info("⚡ Executando predições e analisando resoluções originais...")
    for idx, (img_path, true_label) in enumerate(zip(image_paths, y_true)):
        if idx > 0 and idx % 100 == 0:
            logger.info(f"   Processadas {idx}/{len(image_paths)} imagens...")
            
        # 1. Obter resolução original usando PIL
        try:
            with Image.open(img_path) as pil_img:
                w, h = pil_img.size
        except Exception as e:
            logger.warning(f"⚠️ Erro ao ler metadados de {img_path}: {e}")
            continue
            
        # Determinar tamanho do crop (usamos o maior lado para categorizar o crop)
        crop_size = max(w, h)
        
        # 2. Pré-processar a imagem usando a mesma lógica com Letterbox padding do DataSplitter
        try:
            img_bytes = tf.io.read_file(str(img_path))
            img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
            img = tf.cast(img, tf.float32)
            img = tf.image.resize_with_pad(img, target_height, target_width)
            img_array = tf.expand_dims(img, axis=0)
        except Exception as e:
            logger.warning(f"⚠️ Erro no pré-processamento de {img_path}: {e}")
            continue
            
        # 3. Predição do modelo
        preds = model.predict(img_array, verbose=0)
        pred_label = np.argmax(preds[0])
        
        # 4. Alocar nos buckets
        if crop_size < buckets["pequena"]["max_size"]:
            bucket = "pequena"
        elif crop_size <= buckets["media"]["max_size"]:
            bucket = "media"
        else:
            bucket = "alta"
            
        buckets[bucket]["true"].append(true_label)
        buckets[bucket]["pred"].append(pred_label)
        buckets[bucket]["paths"].append(img_path)
        buckets[bucket]["sizes"].append((w, h))

    # Imprimir relatórios
    print("\n" + "="*80)
    print("📊 RELATÓRIO DE IMPACTO DA RESOLUÇÃO DO CROP NA PERFORMANCE")
    print("="*80)
    
    total_samples = len(image_paths)
    
    for name, data in buckets.items():
        count = len(data["true"])
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        
        print(f"\n📂 Categoria: {name.upper()}")
        if name == "pequena":
            print(f"   Faixa: Crop < 60px (Maior lado)")
        elif name == "media":
            print(f"   Faixa: Crop entre 60px e 120px (Maior lado)")
        else:
            print(f"   Faixa: Crop > 120px (Maior lado)")
            
        print(f"   Amostras: {count} ({percentage:.2f}% do dataset de validação)")
        
        if count == 0:
            print("   ⚠️ Nenhuma amostra nesta categoria.")
            continue
            
        # Calcular acurácia
        true_arr = np.array(data["true"])
        pred_arr = np.array(data["pred"])
        correct = np.sum(true_arr == pred_arr)
        accuracy = correct / count
        
        print(f"   Acurácia: {accuracy:.4f} ({correct}/{count} acertos)")
        
        # Relatório de classificação detalhado
        print("\n   Detalhamento de Métricas:")
        # Identificar classes presentes neste subconjunto para evitar erros do classification_report
        unique_classes = np.unique(true_arr)
        present_class_names = [class_names[c] for c in unique_classes]
        
        report = classification_report(
            true_arr, 
            pred_arr, 
            labels=unique_classes, 
            target_names=present_class_names, 
            output_dict=True,
            zero_division=0
        )
        
        # Exibir macro e weighted avg f1-score
        macro_f1 = report["macro avg"]["f1-score"]
        weighted_f1 = report["weighted avg"]["f1-score"]
        print(f"     F1-Score Macro: {macro_f1:.2f}")
        print(f"     F1-Score Weighted: {weighted_f1:.2f}")
        
        # Detalhe por classe
        print("     F1-Score por Classe:")
        for c_idx in unique_classes:
            c_name = class_names[c_idx]
            c_f1 = report[c_name]["f1-score"]
            c_support = report[c_name]["support"]
            print(f"       * {c_name: <25} -> F1: {c_f1:.4f} (suporte: {c_support})")
            
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avalia o impacto da resolução original dos crops na performance do modelo.")
    parser.add_argument("--model-path", type=str, default="artifacts/models/best_model.keras", help="Caminho do arquivo de modelo treinado (.keras)")
    parser.add_argument("--val-dir", type=str, default="artifacts/data/final/DatasetPests-split/val", help="Caminho do diretório de validação com crops de imagem.")
    parser.add_argument("--experiment", type=str, default="mobilenetv3large", help="Experimento correspondente no model_params.yaml.")
    
    args = parser.parse_args()
    
    # Se o modelo default não existe, tenta buscar na pasta de modelos
    model_path = Path(args.model_path)
    if not model_path.exists():
        alternatives = [
            Path("artifacts/models/mobile/MobileNetV3Large_keras_tuner_best.keras"),
            Path("artifacts/models/mobile/EfficientNetV2B1_keras_tuner_best.keras"),
            Path("artifacts/models/mobile/EfficientNetV2B0_keras_tuner_best.keras"),
            Path("artifacts/models/mobilenetv3large_trained.keras"),
            Path("artifacts/models/efficientnetv2b1_trained.keras"),
            Path("artifacts/models/best_model.keras"),
        ]
        for alt in alternatives:
            if alt.exists():
                model_path = alt
                logger.info(f"📍 Usando modelo alternativo encontrado em: {model_path}")
                break
                
    val_dir = Path(args.val_dir)
    
    if not model_path.exists():
        logger.error(f"❌ Modelo {model_path} não encontrado. Por favor, especifique o caminho correto via --model-path")
        sys.exit(1)
        
    if not val_dir.exists():
        logger.error(f"❌ Diretório de validação {val_dir} não encontrado. Execute a etapa de split/preparação de dados primeiro.")
        sys.exit(1)
        
    evaluate_resolution_impact(model_path, val_dir, args.experiment)
