#!/bin/bash
set -e

# Base dirs
DATA_DIR="artifacts/data/final"
ORIGINAL_SPLIT="$DATA_DIR/DatasetPests-split"
BACKUP_SPLIT="$DATA_DIR/DatasetPests-split-backup"
ORIGINAL_TEST="$DATA_DIR/INSECT12C-test"
BACKUP_TEST="$DATA_DIR/INSECT12C-test-backup"
FLOWER_DATA="artifacts/data/flower_photos"

echo "=== 🔄 Iniciando Teste de Validação do Pipeline ==="

if [ ! -d "$FLOWER_DATA" ]; then
    echo "❌ Erro: O dataset de flores não foi encontrado em $FLOWER_DATA."
    exit 1
fi

# 1. Fazer backup temporário do dataset original
if [ -d "$ORIGINAL_SPLIT" ]; then
    echo "📦 Criando backup temporário de $ORIGINAL_SPLIT..."
    mv "$ORIGINAL_SPLIT" "$BACKUP_SPLIT"
fi
if [ -d "$ORIGINAL_TEST" ]; then
    echo "📦 Criando backup temporário de $ORIGINAL_TEST..."
    mv "$ORIGINAL_TEST" "$BACKUP_TEST"
fi

# 2. Criar links simbólicos para as flores
echo "🔗 Apontando o pipeline temporariamente para o dataset de flores..."
ln -s "../flower_photos" "$ORIGINAL_SPLIT"
ln -s "../flower_photos/val" "$ORIGINAL_TEST"

# 3. Rodar o pipeline de treinamento para 5 épocas
echo "🚀 Executando o pipeline (5 épocas)..."
cleanup() {
    echo "🧹 Restaurando o dataset original..."
    rm -f "$ORIGINAL_SPLIT"
    rm -f "$ORIGINAL_TEST"
    if [ -d "$BACKUP_SPLIT" ]; then
        mv "$BACKUP_SPLIT" "$ORIGINAL_SPLIT"
    fi
    if [ -d "$BACKUP_TEST" ]; then
        mv "$BACKUP_TEST" "$ORIGINAL_TEST"
    fi
    echo "✅ Concluído!"
}
trap cleanup EXIT INT TERM

.venv/bin/python scripts/main_tuning.py --mode retrain --experiment flower_test --final-epochs 50

echo "✨ Teste do pipeline concluído com sucesso!"
