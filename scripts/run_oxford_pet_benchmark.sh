#!/bin/bash
set -e

# Base dirs
DATA_DIR="artifacts/data/final"
ORIGINAL_SPLIT="$DATA_DIR/DatasetPests-split"
BACKUP_SPLIT="$DATA_DIR/DatasetPests-split-backup"
ORIGINAL_TEST="$DATA_DIR/INSECT12C-test"
BACKUP_TEST="$DATA_DIR/INSECT12C-test-backup"
PET_DATA="artifacts/data/final/oxford_pets-split"

# Verificar argumentos
EXPERIMENT=${1:-oxford_pet_baseline}
EPOCHS=${2:-30}
HP_PATH=${3:-artifacts/tuning/best_hyperparameters_oxford_pet.json}

echo "=== 🔄 Iniciando Benchmark do Oxford-IIIT Pet Dataset ==="
echo "📊 Experimento selecionado: $EXPERIMENT"
echo "📈 Épocas: $EPOCHS"
echo "📝 Hiperparâmetros: $HP_PATH"

if [ ! -d "$PET_DATA" ]; then
    echo "❌ Erro: O dataset Oxford Pets não está preparado em $PET_DATA."
    echo "Por favor, rode primeiro: .venv/bin/python scripts/data_preparation/prepare_oxford_pets.py"
    exit 1
fi

# 1. Fazer backup temporário do dataset original
if [ -d "$ORIGINAL_SPLIT" ] && [ ! -L "$ORIGINAL_SPLIT" ]; then
    echo "📦 Criando backup temporário de $ORIGINAL_SPLIT..."
    mv "$ORIGINAL_SPLIT" "$BACKUP_SPLIT"
fi
if [ -d "$ORIGINAL_TEST" ] && [ ! -L "$ORIGINAL_TEST" ]; then
    echo "📦 Criando backup temporário de $ORIGINAL_TEST..."
    mv "$ORIGINAL_TEST" "$BACKUP_TEST"
fi

# Remover links antigos se existirem
rm -f "$ORIGINAL_SPLIT"
rm -f "$ORIGINAL_TEST"

# 2. Criar links simbólicos para o Oxford Pet
echo "🔗 Apontando o pipeline temporariamente para o Oxford Pet Dataset..."
ln -s "oxford_pets-split" "$ORIGINAL_SPLIT"
ln -s "oxford_pets-split/test" "$ORIGINAL_TEST"

# 3. Rodar o pipeline de treinamento
echo "🚀 Executando treinamento ($EXPERIMENT)..."
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

.venv/bin/python scripts/main_tuning.py --mode retrain --experiment "$EXPERIMENT" --final-epochs "$EPOCHS" --best-hp-path "$HP_PATH"

echo "✨ Benchmark de $EXPERIMENT concluído com sucesso!"
