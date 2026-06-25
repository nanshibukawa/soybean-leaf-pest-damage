#!/usr/bin/env python3
"""
Script para treinar o detector YOLOv8 (Single Class)
para encontrar pragas em imagens de campo.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
YAML_PATH = ROOT_DIR / "artifacts/data/yolo/yolo_dataset/dataset.yaml"

def main():
    try:
        import torch
        torch.backends.cudnn.enabled = False
        print("⚡ cuDNN desabilitado para evitar conflito de versão (mismatch).")
    except ImportError:
        pass

    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Biblioteca 'ultralytics' não encontrada.")
        print("Instale rodando:")
        print("   pip install ultralytics")
        sys.exit(1)
        
    if not YAML_PATH.exists():
        print(f"❌ Arquivo de configuração do dataset não encontrado em: {YAML_PATH}")
        print("Por favor, execute o script anterior primeiro:")
        print("   python scripts/yolo_inaturalist/create_yolo_dataset.py")
        sys.exit(1)
        
    print("🚀 Carregando modelo pré-treinado YOLOv8n...")
    model = YOLO("yolov8n.pt")
    
    print("\n--- Iniciando Treinamento do Detector ---")
    print(f"Dataset config: {YAML_PATH.absolute()}")
    
    # Treinar o modelo
    # epochs=30: Suficiente para um detector simples de bounding boxes (single class)
    # imgsz=640: Resolução padrão do YOLOv8
    # device=0 ou device='cpu' dependendo de GPU
    results = model.train(
        data=str(YAML_PATH.absolute()),
        epochs=30,
        imgsz=640,
        device=0, # Defina device=0 para usar GPU Nvidia se disponível!
        workers=2,
        project="artifacts/yolo_runs",
        name="pest_detector"
    )
    
    print("\n✅ Treinamento Concluído!")
    print("O modelo de melhor performance (best.pt) foi salvo em:")
    print("   artifacts/yolo_runs/pest_detector/weights/best.pt")

if __name__ == "__main__":
    main()
