#!/usr/bin/env bash
# =============================================================================
# PIPELINE DE PRÉ-PROCESSAMENTO DE DADOS (SOYBEAN PEST DAMAGE DETECTION)
# =============================================================================
# Este script automatiza o fluxo completo de preparação dos dados do zero:
# 1. Treinamento do detector YOLOv8 e auto-recorte do iNaturalist
# 2. Recorte do DatasetPests com margem configurável (+20%)
# 3. Divisão do dataset por Imagem-Mãe sem data leakage
# 4. Preparação do dataset agrícola IP102 (102 classes) para pré-treinamento
# 5. Preparação do dataset de teste externo INSECT12C
# =============================================================================

set -e  # Interrompe o script se qualquer comando falhar

# Ativar ambiente virtual se disponível
if [ -d ".venv" ]; then
    source .venv/bin/activate 2>/dev/null || true
fi

# Cores para formatação de saída no terminal
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # Sem cor

# Parâmetros Padrão
MARGIN=${1:-"0.20"}  # Margem extra padrão: 20% (0.20)
ONLY_TEACHERS=${2:-"false"} # Setar para "true" se quiser apenas dados dos professores

echo -e "${BLUE}=====================================================================${NC}"
echo -e "${BLUE} 🚀 INICIANDO PIPELINE DE PRÉ-PROCESSAMENTO COMPLETO ${NC}"
echo -e "${BLUE}=====================================================================${NC}"
echo -e "${YELLOW}📌 Margem de expansão de crop: ${MARGIN} (padding contras distorção)${NC}"
echo -e "${YELLOW}📌 Modo apenas professores: ${ONLY_TEACHERS}${NC}"
echo ""

# -----------------------------------------------------------------------------
# PASSO 1: DETECTOR YOLOV8 & AUTO-CROP DO INATURALIST
# -----------------------------------------------------------------------------
echo -e "${GREEN}---------------------------------------------------------------------${NC}"
echo -e "${GREEN} [PASSO 1/5] Detecção YOLOv8 & Auto-Crop do iNaturalist${NC}"
echo -e "${GREEN}---------------------------------------------------------------------${NC}"

echo -e "${BLUE}1.1 Criando dataset no formato YOLO a partir dos dados dos professores...${NC}"
python scripts/yolo_inaturalist/create_yolo_dataset.py

echo -e "${BLUE}1.2 Treinando detector YOLOv8n (Pest Detector)...${NC}"
python scripts/yolo_inaturalist/train_yolo.py

echo -e "${BLUE}1.3 Auto-recortando imagens mineradas do iNaturalist com YOLOv8...${NC}"
python scripts/yolo_inaturalist/auto_crop_inaturalist.py

# -----------------------------------------------------------------------------
# PASSO 2: RECORTE DO DATASETPESSTS (PROFESSORES) COM MARGEM (+20%)
# -----------------------------------------------------------------------------
echo -e "${GREEN}---------------------------------------------------------------------${NC}"
echo -e "${GREEN} [PASSO 2/5] Recorte do DatasetPests dos Professores com Margem${NC}"
echo -e "${GREEN}---------------------------------------------------------------------${NC}"

echo -e "${BLUE}2.1 Recortando bounding boxes com margem de ${MARGIN}...${NC}"
python scripts/data_preparation/prepare_dataset_pests.py --margin "${MARGIN}"

# -----------------------------------------------------------------------------
# PASSO 3: DIVISÃO ESTRATIFICADA POR IMAGEM-MÃE (SEM VAZAMENTO)
# -----------------------------------------------------------------------------
echo -e "${GREEN}---------------------------------------------------------------------${NC}"
echo -e "${GREEN} [PASSO 3/5] Divisão por Imagem-Mãe (Group-based Split)${NC}"
echo -e "${GREEN}---------------------------------------------------------------------${NC}"

if [ "${ONLY_TEACHERS}" = "true" ]; then
    echo -e "${YELLOW}3.1 Gerando split puramente do dataset dos Professores (--only-teachers)...${NC}"
    python scripts/data_preparation/split_dataset_by_group.py --only-teachers
else
    echo -e "${BLUE}3.1 Gerando split misto (Professores + iNaturalist)...${NC}"
    python scripts/data_preparation/split_dataset_by_group.py
fi

# -----------------------------------------------------------------------------
# PASSO 4: PREPARAÇÃO DO DATASET DE PRÉ-TREINAMENTO (IP102)
# -----------------------------------------------------------------------------
echo -e "${GREEN}---------------------------------------------------------------------${NC}"
echo -e "${GREEN} [PASSO 4/5] Preparação do Dataset IP102 (102 Classes)${NC}"
echo -e "${GREEN}---------------------------------------------------------------------${NC}"

echo -e "${BLUE}4.1 Recortando anotações do IP102 para pré-treinamento de domínio...${NC}"
python scripts/data_preparation/prepare_ip102.py

# -----------------------------------------------------------------------------
# PASSO 5: PREPARAÇÃO DO DATASET DE BENCHMARK EXTERNO (INSECT12C)
# -----------------------------------------------------------------------------
echo -e "${GREEN}---------------------------------------------------------------------${NC}"
echo -e "${GREEN} [PASSO 5/5] Preparação do Dataset de Teste INSECT12C${NC}"
echo -e "${GREEN}---------------------------------------------------------------------${NC}"

echo -e "${BLUE}5.1 Extraindo e recortando imagens da base externa INSECT12C...${NC}"
python scripts/data_preparation/prepare_insect12c.py

echo ""
echo -e "${BLUE}=====================================================================${NC}"
echo -e "${GREEN} 🎉 PIPELINE DE PRÉ-PROCESSAMENTO CONCLUÍDO COM SUCESSO! ${NC}"
echo -e "${BLUE}=====================================================================${NC}"
echo -e "${YELLOW}Próximo passo: Rodar o pré-treino do IP102 com 'python scripts/train_ip102.py'${NC}"
