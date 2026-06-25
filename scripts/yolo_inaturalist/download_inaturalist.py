#!/usr/bin/env python3
"""
Script para baixar imagens reais de pragas da soja diretamente da API do iNaturalist.
Focado nas classes minoritárias: Rhammatocerus schistocercoides, Anticarsia gemmatalis e Euschistus heros.
"""

import os
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

# Configuração das classes prioritárias
TARGET_SPECIES = {
    "rhammatocerus_schistocercoides": {
        "taxon_id": 211764,  # ID do Gênero Rhammatocerus (1600+ obs) para maior volume, em vez de 758127 (apenas 10 obs)
        "display_name": "Gafanhoto do Cerrado",
        "quality_grade": "research,needs_id"
    },
    "anticarsia_gemmatalis": {
        "taxon_id": 213809,  # ID real de Anticarsia gemmatalis (10.000+ obs)
        "display_name": "Lagarta-da-soja",
        "quality_grade": "research"
    },
    "euschistus_heros": {
        "taxon_id": 544061,  # ID real de Euschistus heros (430+ obs)
        "display_name": "Percevejo-marrom",
        "quality_grade": "research"
    },
    "gastropoda": {
        "taxon_id": 47114,  # ID da Classe Gastropoda
        "display_name": "Gastropoda (Lesmas/Caracóis)",
        "quality_grade": "research,needs_id"
    },
    "spodoptera_albula": {
        "taxon_id": 424930,  # ID real de Spodoptera albula
        "display_name": "Lagarta-das-vagens (Spodoptera albula)",
        "quality_grade": "research"
    }
}

# Configurações globais
BASE_OUTPUT_DIR = Path(__file__).parent.parent.parent / "artifacts/data_ingestion/inaturalist"
USER_AGENT = "SoybeanPestResearchBot/1.0 (contact: nanshibukawa@example.com)" # iNaturalist solicita User-Agent descritivo

def fetch_json(url):
    """Realiza requisição HTTP GET e retorna o JSON usando a biblioteca padrão (sem dependências externas)."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json"
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"Erro ao acessar {url}: {e}")
        return None

def download_image(url, output_path):
    """Baixa um arquivo de imagem da URL especificada."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            output_path.write_bytes(response.read())
        return True
    except Exception as e:
        print(f"\nErro ao baixar imagem de {url}: {e}")
        return False

def download_species_images(name, info, limit=400):
    """Consulta a API do iNaturalist e faz o download das imagens da espécie."""
    taxon_id = info["taxon_id"]
    display_name = info["display_name"]
    quality_grade = info.get("quality_grade", "research")
    
    print("\n" + "="*60)
    print(f"Iniciando busca para: {display_name} ({name}) - Taxon ID: {taxon_id}")
    print("="*60)
    
    species_dir = BASE_OUTPUT_DIR / name
    images_dir = species_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_file = species_dir / "metadata.json"
    
    # Carregar metadados existentes, se houver
    existing_metadata = {}
    if metadata_file.exists():
        try:
            existing_metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    page = 1
    downloaded_count = 0
    metadata_to_save = existing_metadata.copy()
    
    # Parâmetros da API
    # quality_grade: Filtra por status de identificação (research e/ou needs_id)
    # photos=true: Filtra apenas observações que contêm fotos
    
    while downloaded_count < limit:
        params = {
            "taxon_id": taxon_id,
            "quality_grade": quality_grade,
            "photos": "true",
            "per_page": 100,
            "page": page
        }
        
        query_string = urllib.parse.urlencode(params)
        api_url = f"https://api.inaturalist.org/v1/observations?{query_string}"
        
        data = fetch_json(api_url)
        if not data or "results" not in data or not data["results"]:
            break
            
        results = data["results"]
        total_results = data.get("total_results", 0)
        print(f"Página {page}: Encontradas {len(results)} observações (Total na API: {total_results})")
        
        for obs in results:
            if downloaded_count >= limit:
                break
                
            obs_id = obs["id"]
            photos = obs.get("photos", [])
            if not photos:
                continue
                
            # Dados de atribuição
            user_name = obs.get("user", {}).get("name") or obs.get("user", {}).get("login", "Desconhecido")
            
            # Filtro de licença Creative Commons (verificação robusta no cliente)
            license_code = photos[0].get("license_code")
            if not license_code:
                # Foto sem licença é tratada como "All Rights Reserved", pulamos
                continue
                
            allowed_licenses = {"cc0", "cc-by", "cc-by-sa", "cc-by-nc", "cc-by-nc-sa"}
            if license_code.lower() not in allowed_licenses:
                continue
            
            # iNaturalist armazena várias resoluções. Vamos pegar a versão 'medium' (largura ~1024px) ou 'large'
            # A URL padrão normalmente termina com '/square.jpg'. Substituímos por '/medium.jpg'
            photo_url = photos[0].get("url")
            if not photo_url:
                continue
            
            # Converter URL para tamanho médio (ideal para treinamento de visão computacional)
            medium_url = photo_url.replace("square.jpg", "medium.jpg").replace("square.jpeg", "medium.jpeg")
            
            img_filename = f"obs_{obs_id}.jpg"
            img_path = images_dir / img_filename
            
            # Pula se a imagem já foi baixada
            if img_path.exists() and str(obs_id) in metadata_to_save:
                continue
                
            print(f"Baixando imagem {downloaded_count + 1}/{limit} (Obs ID: {obs_id})...", end="\r")
            
            if download_image(medium_url, img_path):
                metadata_to_save[str(obs_id)] = {
                    "file_name": img_filename,
                    "author": user_name,
                    "license": license_code.upper(),
                    "observation_url": f"https://www.inaturalist.org/observations/{obs_id}",
                    "original_photo_url": medium_url,
                    "date_observed": obs.get("observed_on_string")
                }
                downloaded_count += 1
                
                # Evitar sobrecarregar a API com requisições rápidas de imagem
                time.sleep(0.3)
                
        page += 1
        # Delay de cortesia entre requisições de página da API
        time.sleep(1.0)
        
    # Salvar metadados atualizados
    metadata_file.write_text(json.dumps(metadata_to_save, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nConcluído! Total de imagens baixadas nesta execução para {display_name}: {downloaded_count}")
    print(f"Diretório de saída: {images_dir}")
    print(f"Metadados salvos em: {metadata_file}")

def main():
    print("Iniciando o downloader do iNaturalist para Pragas da Soja...")
    
    # Criar pasta base de saída
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Executar o download para cada espécie
    for name, info in TARGET_SPECIES.items():
        # Vamos definir um limite de 1500 imagens por espécie
        download_species_images(name, info, limit=1500)

if __name__ == "__main__":
    main()
