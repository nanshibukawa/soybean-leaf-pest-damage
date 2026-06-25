# Scripts e Pipelines de Preparação de Dados e Análise

Este diretório contém os scripts desenvolvidos para preparar, processar, analisar e organizar as bases de dados utilizadas no treinamento dos classificadores de danos e detectores de pragas em folhas de soja.

---

## 🏗️ Estrutura do Diretório

```text
scripts/
├── main.py                     # Pipeline principal de treinamento do classificador
├── main_tuning.py              # Pipeline de busca de hiperparâmetros (Keras Tuner)
├── data_preparation/           # Scripts para processamento e recorte de bases
│   ├── prepare_dataset_pests.py
│   ├── prepare_insect12c.py
│   └── split_dataset_by_group.py
├── yolo_inaturalist/           # Scripts do detector YOLO e dados do iNaturalist
│   ├── download_inaturalist.py
│   ├── create_yolo_dataset.py
│   ├── train_yolo.py
│   └── auto_crop_inaturalist.py
└── analysis/                   # Scripts de análise de explicabilidade e Grad-CAM
    └── gradcam_analysis.py
```

---

## 📋 Categorias de Scripts

### 1️⃣ Principais de Execução (ML Pipelines)

* **`scripts/main.py`**:
  Executa o pipeline completo de machine learning usando as configurações padrão do arquivo `model_params.yaml`. Cobre a ingestão de dados, divisão, preparação dos dados de treinamento/validação, treinamento do modelo (ex: EfficientNet, MobileNet) e avaliação das métricas.
  ```bash
  python scripts/main.py
  ```

* **`scripts/main_tuning.py`**:
  Executa o pipeline de otimização de hiperparâmetros usando Keras Tuner (Bayesian Optimization) integrado com MLflow para rastreamento de experimentos.
  ```bash
  python scripts/main_tuning.py --experiment mobilenetv3large --max-trials 30
  ```

### 2️⃣ Preparação e Recorte de Dados (`scripts/data_preparation/`)

* **`prepare_dataset_pests.py`**:
  Realiza o recorte (*crop*) das imagens originais anotadas do DatasetPests com base nas coordenadas de bounding boxes contidas nos arquivos CSV de anotação (exportados do Label-Studio).
  * **Saída:** Crops salvos em `artifacts/data/processed/DatasetPests-cropped/<classe>/`.
  ```bash
  python scripts/data_preparation/prepare_dataset_pests.py
  ```

* **`prepare_insect12c.py`**:
  Extrai e processa a base de dados externa `INSECT12C`. Analisa as anotações no formato Pascal VOC (XML) para recortar as imagens e opcionalmente as mapeia de 12 para 10 classes para manter compatibilidade com o classificador de pragas do projeto.
  * **Saída:** Crops do conjunto de teste salvos em `artifacts/data/final/INSECT12C-test/`.
  ```bash
  python scripts/data_preparation/prepare_insect12c.py
  ```

* **`split_dataset_by_group.py`**:
  Realiza a divisão final do dataset em treino (90%) e validação (10%), agrupando os recortes pelo ID único da imagem-mãe. Esse agrupamento é crucial para evitar o vazamento de dados (*data leakage*) decorrente de augmentações offline ou múltiplos recortes da mesma folha.
  * **Saída:** Organiza o conjunto final estruturado em `artifacts/data/final/DatasetPests-split/`.
  ```bash
  python scripts/data_preparation/split_dataset_by_group.py
  ```

### 3️⃣ Pipeline do Detector YOLO e iNaturalist (`scripts/yolo_inaturalist/`)

* **`download_inaturalist.py`**:
  Minerador de imagens reais de campo diretamente da API pública do iNaturalist. É focado nas espécies e classes de pragas minoritárias do projeto para equilibrar o conjunto de dados.
  * **Saída:** Imagens brutas e metadados em `artifacts/data_ingestion/inaturalist/`.
  ```bash
  python scripts/yolo_inaturalist/download_inaturalist.py
  ```

* **`create_yolo_dataset.py`**:
  Lê as anotações do CSV do DatasetPests e as converte no formato YOLO (coordenadas normalizadas com classe única). Faz uma divisão 90/10 baseada na imagem-mãe.
  * **Saída:** Estrutura pronta para treino em `artifacts/data/yolo/yolo_dataset/`.
  ```bash
  python scripts/yolo_inaturalist/create_yolo_dataset.py
  ```

* **`train_yolo.py`**:
  Treina um modelo YOLOv8n de classe única (`pest`) usando o dataset estruturado no passo anterior para localizar pragas de soja de forma automatizada.
  * **Saída:** O melhor modelo (`best.pt`) é salvos em `artifacts/yolo_runs/pest_detector/weights/best.pt`.
  ```bash
  python scripts/yolo_inaturalist/train_yolo.py
  ```

* **`auto_crop_inaturalist.py`**:
  Utiliza o detector YOLOv8 treinado no passo anterior para recortar de forma inteligente as pragas presentes nas imagens reais baixadas do iNaturalist. Isso automatiza a pseudo-anotação e o recorte de milhares de novas imagens.
  * **Saída:** Salva os crops com o prefixo `inat_` em `artifacts/data/processed/DatasetPests-cropped/<classe>/`.
  ```bash
  python scripts/yolo_inaturalist/auto_crop_inaturalist.py
  ```

### 4️⃣ Análise de Modelos (`scripts/analysis/`)

* **`gradcam_analysis.py`**:
  Gera visualizações baseadas em Grad-CAM (Gradient-weighted Class Activation Mapping) em imagens de folhas de soja, mostrando quais regiões foram mais influentes nas predições do modelo de classificação.
  ```bash
  python scripts/analysis/gradcam_analysis.py
  ```
