#!/usr/bin/env bash
# =============================================================================
# EXECUTOR DO BENCHMARK DE GENERALIZAÇÃO EXTERNA NO INSECT12C
# =============================================================================
# Avalia o modelo treinado (Passo 2) no dataset de teste externo INSECT12C
# em 3 faixas de resolução (Completo, > 60x60 px, > 100x100 px).
# =============================================================================

set -e

# Ativar ambiente virtual se disponível
if [ -d ".venv" ]; then
    source .venv/bin/activate 2>/dev/null || true
fi

MODEL_PATH=${1:-"artifacts/models/efficientnetv2b1_trained.keras"}

echo "====================================================================="
echo " 🧪 EXECUTANDO BENCHMARK ZERO-SHOT NO INSECT12C"
echo "====================================================================="
echo "📌 Modelo a ser avaliado: ${MODEL_PATH}"
echo ""

python scripts/evaluate_insect12c.py --model_path "${MODEL_PATH}"
