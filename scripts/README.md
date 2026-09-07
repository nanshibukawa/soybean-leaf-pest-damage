# Scripts e Pipelines de Preparação de Dados e Análise

Este diretório contém os scripts desenvolvidos para preparar, processar, analisar e organizar as bases de dados utilizadas no treinamento dos classificadores de danos e detectores de pragas em folhas de soja.

---

## 🏗️ Estrutura do Diretório

```text
scripts/
├── run_preprocessing_pipeline.sh # AUTOMATIZADOR DO PIPELINE DE PRÉ-PROCESSAMENTO
├── main.py                     # Pipeline principal de treinamento do classificador (--experiment)
├── main_tuning.py              # Pipeline de busca de hiperparâmetros (Keras Tuner)
├── train_ip102.py              # Passo 1: Pre-training de domínio no dataset IP102 (102 classes)
├── evaluate_test.py            # Avaliação dedicada do modelo no conjunto de teste final
├── data_preparation/           # Scripts para processamento e recorte de bases
│   ├── prepare_dataset_pests.py# Recorte do DatasetPests com opção de margem (--margin 0.20)
│   ├── prepare_ip102.py        # Preparação do dataset oficial IP102 para pré-treinamento
│   ├── prepare_insect12c.py    # Preparação e recorte da base de teste externa INSECT12C
│   ├── prepare_oxford_pets.py  # Preparação do dataset Oxford-IIIT Pet para benchmarking SOTA
│   ├── split_dataset_by_group.py # Divisão por imagem-mãe (flag --only-teachers)
│   ├── auto_filter_moths.py    # Filtro automático de fases de vida (lagartas vs mariposas)
│   └── parse_pretrain_metrics.py
├── yolo_inaturalist/           # Scripts do detector YOLO e dados do iNaturalist
│   ├── download_inaturalist.py # Download com filtros de Life Stage (larvas/ninfas)
│   ├── create_yolo_dataset.py
│   ├── train_yolo.py
│   └── auto_crop_inaturalist.py
└── analysis/                   # Scripts de análise de explicabilidade e impacto de resolução
    ├── gradcam_analysis.py     # Análise de explicabilidade por Grad-CAM
    └── evaluate_resolution_impact.py # Avaliação do impacto da resolução da imagem
```

---

## ⚡ Automação Rápida (Execução Completa do Pré-Processamento)

Para executar todo o pipeline de pré-processamento do zero em um único comando:

```bash
# Execução padrão (margem 20%, dataset misto)
bash scripts/run_preprocessing_pipeline.sh

# Execução apenas com dados dos professores (margem 20%)
bash scripts/run_preprocessing_pipeline.sh 0.20 true
```

---

## 📋 Guia de Execução dos Experimentos Metodológicos

### 1️⃣ Pré-Treinamento de Domínio (IP102 ➜ DatasetPests)

1. **Preparar recortes do IP102:**
   ```bash
   python scripts/data_preparation/prepare_ip102.py
   ```
2. **Passo 1: Pré-treinar backbone no IP102 (102 classes de pragas):**
   ```bash
   python scripts/train_ip102.py
   ```
   *Saída:* Salva o extrator de características em `artifacts/models/ip102_pretrained_extractor.keras`.
3. **Passo 2: Fine-tuning no DatasetPests:**
   ```bash
   python scripts/main.py --experiment efficientnetv2b1_finetune
   ```

---

### 2️⃣ Experimento: Recorte Padrão vs. Recorte com Margem (+20%)

Para testar a hipótese de que o recorte ajustado causa distorções em pragas pequenas:
```bash
# Recorte com 20% de margem extra ao redor da bounding box
python scripts/data_preparation/prepare_dataset_pests.py --margin 0.20

# Dividir sem vazamento de dados
python scripts/data_preparation/split_dataset_by_group.py
```

---

### 3️⃣ Experimento: Dataset Apenas Professores vs. Dataset Misto (+iNaturalist)

Para isolar o impacto do dataset minerado do iNaturalist vs. apenas dados do laboratório:
```bash
# Gerar split apenas com imagens dos professores
python scripts/data_preparation/split_dataset_by_group.py --only-teachers

# Treinar modelo baseline
python scripts/main.py --experiment mobilenetv3large
```

---

## 🔬 Scripts de Benchmarking de Blocos Customizados (CBAM & Super-Resolution)

- **Oxford-IIIT Pet (37 classes):** `bash scripts/run_oxford_pet_benchmark.sh`
- **INSECT12C (10 classes):** `bash scripts/run_insect12c_benchmark.sh`

  ```bash
  python scripts/analysis/gradcam_analysis.py
  ```
