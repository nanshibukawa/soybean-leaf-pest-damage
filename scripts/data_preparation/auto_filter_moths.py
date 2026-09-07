#!/usr/bin/env python3
"""
Script para filtrar automaticamente mariposas adultas do dataset iNaturalist de Anticarsia gemmatalis.
Utiliza a API do iNaturalist para identificar observações anotadas como 'Adult' (ID 2)
e as move para uma pasta de backup, limpando o dataset bruto de imagens.
"""

import os
import json
import shutil
import urllib.request
import time
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
ANTICARSIA_DIR = ROOT_DIR / "artifacts/data_ingestion/inaturalist/anticarsia_gemmatalis"
IMAGES_DIR = ANTICARSIA_DIR / "images"
BACKUP_DIR = ANTICARSIA_DIR / "adult_moths_backup"
METADATA_FILE = ANTICARSIA_DIR / "metadata.json"

def fetch_batch(obs_ids):
    ids_str = ",".join(obs_ids)
    url = f"https://api.inaturalist.org/v1/observations?id={ids_str}"
    req = urllib.request.Request(url, headers={"User-Agent": "SoybeanPestCleanBot/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception as e:
        print(f"\n❌ Erro de conexão com a API: {e}")
        return None

def main():
    if not IMAGES_DIR.exists() or not METADATA_FILE.exists():
        print(f"❌ Diretório ou arquivo de metadados não encontrado em: {ANTICARSIA_DIR}")
        return

    # Criar pasta de backup se não existir
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    obs_ids = list(metadata.keys())
    print(f"🔍 Carregados {len(obs_ids)} metadados de observações.")
    print("🚀 Consultando a API do iNaturalist para identificar estágios de vida...")

    adult_ids = []
    larva_ids = []
    unannotated_ids = []

    batch_size = 50
    for i in range(0, len(obs_ids), batch_size):
        batch = obs_ids[i:i+batch_size]
        print(f"   Processando lote {i // batch_size + 1}/{(len(obs_ids) - 1) // batch_size + 1}...", end="\r")
        
        data = fetch_batch(batch)
        if not data or "results" not in data:
            print(f"\n⚠️ Pulando lote a partir do índice {i} devido a falha na API.")
            continue
            
        for obs in data["results"]:
            obs_id_str = str(obs["id"])
            annotations = obs.get("annotations", [])
            life_stage_found = False
            for ann in annotations:
                if ann.get("controlled_attribute_id") == 1:
                    life_stage_found = True
                    val_id = ann.get("controlled_value_id")
                    if val_id == 2:  # 2 = Adult (Mariposa)
                        adult_ids.append(obs_id_str)
                    elif val_id == 6:  # 6 = Larva (Lagarta)
                        larva_ids.append(obs_id_str)
                    else:
                        unannotated_ids.append(obs_id_str)
                    break
                    
            if not life_stage_found:
                unannotated_ids.append(obs_id_str)
                
        # Delay de cortesia para a API
        time.sleep(0.3)

    print("\n\n--- 📊 ESTATÍSTICAS DE FILTRAGEM ---")
    print(f"🦋 Mariposas Adultas identificadas (para mover): {len(adult_ids)}")
    print(f"🐛 Lagartas (Larvas) identificadas (manter): {len(larva_ids)}")
    print(f"❓ Sem anotação clara / Outros (manter para revisão): {len(unannotated_ids)}")

    # Mover arquivos de mariposas adultas e atualizar metadados
    moved_count = 0
    clean_metadata = {}

    for obs_id in obs_ids:
        entry = metadata[obs_id]
        filename = entry.get("file_name")
        src_path = IMAGES_DIR / filename
        
        if obs_id in adult_ids:
            # É mariposa adulta, movemos para o backup
            if src_path.exists():
                shutil.move(src_path, BACKUP_DIR / filename)
                moved_count += 1
        else:
            # Mantemos no dataset limpo
            clean_metadata[obs_id] = entry

    # Salvar metadados limpos
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(clean_metadata, f, indent=2, ensure_ascii=False)

    print("\n--- 💾 RESULTADO ---")
    print(f"✅ {moved_count} arquivos de mariposas adultas movidos para: {BACKUP_DIR.relative_to(ROOT_DIR)}")
    print(f"📝 Metadados limpos salvos em: {METADATA_FILE.relative_to(ROOT_DIR)} (restaram {len(clean_metadata)} registros)")
    print("\n👉 Por favor, revise visualmente as imagens restantes em 'images/' e rode:")
    print("   python scripts/yolo_inaturalist/auto_crop_inaturalist.py")

if __name__ == "__main__":
    main()
