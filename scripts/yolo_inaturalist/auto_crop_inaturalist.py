#!/usr/bin/env python3
"""
Script para recortar automaticamente os insetos das fotos brutas do iNaturalist
usando o detector YOLOv8 treinado no próprio dataset de pragas.
Salva os crops diretamente na pasta de classes do classificador.
"""

import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent.parent.parent
INAT_BASE_DIR = ROOT_DIR / "artifacts/data_ingestion/inaturalist"
MODEL_PATH = ROOT_DIR / "artifacts/yolo_runs/pest_detector/weights/best.pt"
OUTPUT_BASE_DIR = ROOT_DIR / "artifacts/data/processed/DatasetPests-cropped"

# Mapeamento do nome da pasta iNaturalist para o nome da pasta do DatasetPests-cropped
FOLDER_MAPPING = {
    "rhammatocerus_schistocercoides": "rhammatocerus_schistocercoides",
    "anticarsia_gemmatalis": "anticarsia_gemmatalis-larva",
    "euschistus_heros": "euschistus_heros",
    "gastropoda": "gastropoda",
    "spodoptera_albula": "spodoptera_albula"
}

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
        print("❌ Biblioteca 'ultralytics' não encontrada. Instale rodando:")
        print("   pip install ultralytics")
        sys.exit(1)
        
    if not MODEL_PATH.exists():
        print(f"❌ Modelo treinado não encontrado em: {MODEL_PATH}")
        print("Por favor, execute primeiro o treinamento do detector:")
        print("   python scripts/yolo_inaturalist/train_yolo.py")
        sys.exit(1)
        
    print(f"🚀 Carregando detector treinado de: {MODEL_PATH.name}...")
    model = YOLO(MODEL_PATH)
    
    # Criar pasta de saída se necessário
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n--- Iniciando Recorte Automático (Auto-Crop) das Imagens do iNaturalist ---")
    
    for inat_folder_name, output_folder_name in FOLDER_MAPPING.items():
        inat_dir = INAT_BASE_DIR / inat_folder_name / "images"
        dest_dir = OUTPUT_BASE_DIR / output_folder_name
        
        if not inat_dir.exists():
            print(f"⚠️ Pasta de imagens não encontrada para {inat_folder_name}. Pulando...")
            continue
            
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Coletar todas as imagens brutas
        img_files = [
            f for f in inat_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
        
        print(f"\n🔍 Processando classe: {inat_folder_name} ({len(img_files)} imagens)")
        
        crops_saved_class = 0
        
        for img_path in tqdm(img_files, desc=f"Recortando {inat_folder_name}"):
            try:
                # Rodar inferência do YOLO
                results = model(img_path, verbose=False)
                
                # Se nenhuma caixa for encontrada, não salvamos nada (evita fundo sem inseto)
                if not results or not results[0].boxes:
                    continue
                    
                # Abrir imagem com PIL
                with Image.open(img_path) as img:
                    w, h = img.size
                    
                    # Pegar as detecções ordenadas por confiança decrescente
                    boxes = results[0].boxes
                    
                    for idx, box in enumerate(boxes):
                        # Se a confiança for muito baixa, descartamos
                        conf = float(box.conf[0])
                        if conf < 0.25:
                            continue
                            
                        # Coordenadas [xmin, ymin, xmax, ymax] em pixels
                        coords = box.xyxy[0].tolist()
                        xmin, ymin, xmax, ymax = map(int, coords)
                        
                        # Limitar limites da imagem
                        xmin = max(0, xmin)
                        ymin = max(0, ymin)
                        xmax = min(w, xmax)
                        ymax = min(h, ymax)
                        
                        if xmax <= xmin or ymax <= ymin:
                            continue
                            
                        # Realizar o crop
                        cropped_img = img.crop((xmin, ymin, xmax, ymax))
                        
                        # Converter para RGB se necessário
                        if cropped_img.mode in ("RGBA", "P"):
                            cropped_img = cropped_img.convert("RGB")
                            
                        # Salvar na pasta final do DatasetPests com prefixo inat_ para fácil rastreabilidade
                        output_filename = f"inat_{img_path.stem}_crop_{idx}.jpg"
                        cropped_img.save(dest_dir / output_filename, "JPEG")
                        crops_saved_class += 1
                        
            except Exception as e:
                # Continuar para as próximas se falhar em uma imagem
                print(f"\n❌ Erro ao processar {img_path.name}: {str(e)}")
                # Se for erro crítico (como CUDA/Memory/etc), interrompe para não dar loop infinito de erro
                if "CUDA" in str(e) or "CUDNN" in str(e) or "out of memory" in str(e).lower():
                    raise e
                
        print(f"✅ {crops_saved_class} crops salvos em: {dest_dir.name}/")
        
    print("\n🎉 Processamento concluído com sucesso!")
    print("Agora você pode re-rodar o split para reorganizar as pastas de treino/val:")
    print("   python scripts/data_preparation/split_dataset_by_group.py")

if __name__ == "__main__":
    main()
