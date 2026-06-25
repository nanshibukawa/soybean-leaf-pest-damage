# Soybean Leaf Pest Damage Detection

Sistema de detecção de danos causados por pragas em folhas de soja utilizando Deep Learning, com módulo complementar de RAG (Retrieval-Augmented Generation) para consulta em documentos técnicos.

## 🔎 Sumário

- [📋 Descrição](#descricao)
- [🚀 Tecnologias](#tecnologias)
- [📦 Instalação](#instalacao)
- [🏗️ Estrutura do Projeto](#estrutura)
- [📝 Uso](#uso)
- [📊 Dataset](#dataset)
- [🤖 Módulo RAG (Retrieval-Augmented Generation)](#rag)
- [🎓 Projeto](#projeto)
- [📄 Licença](#licenca)

<a id="descricao"></a>
## 📋 Descrição

Este projeto implementa:

1. Classificação de imagens para identificar danos em folhas de soja, usando CNNs e arquiteturas baseadas em Transformers (ex.: MobileViT, ViT).
2. Perguntas e respostas com RAG, integrando recuperação semântica de documentos e geração de respostas estruturadas com LLM.

<a id="tecnologias"></a>
## 🚀 Tecnologias

### ML / Visão Computacional
- Python 3.12+
- TensorFlow 2.17.1
- Keras (incluindo Keras 3)
- Keras Tuner
- MLflow e DagsHub (rastreamento de experimentos)
- scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- MobileViT, Vision Transformer (ViT)

### API / RAG
- FastAPI
- Pydantic
- pydantic-ai
- Groq (LLM)
- Qdrant (vector database)
- Busca semântica com embeddings
- Chunking semântico com Sentence Transformers + HDBSCAN
- Estratégia híbrida de recuperação (sparse + dense + fusão RRF + reranking ColBERT)

<a id="instalacao"></a>
## 📦 Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/soybean-leaf-pest-damage.git
cd soybean-leaf-pest-damage

# Instale via pyproject (modo padrão)
pip install .

# Ou modo desenvolvimento com extras
pip install -e .[dev]

# Extras de processamento de imagem (opcional)
pip install .[image-processing]
```

<a id="estrutura"></a>
## 🏗️ Estrutura do Projeto

```
├── src/
│   ├── cnnClassifier/          # Pipeline de classificação de imagens
│   │   ├── components/
│   │   ├── config/
│   │   ├── entity/
│   │   ├── models/
│   │   ├── pipeline/
│   │   └── utils/
│   ├── api/                    # API de busca + geração (RAG)
│   ├── rag/                    # Engine e ingestão de documentos para RAG
│   └── ui/                     # Interface de usuário
├── scripts/                    # Scripts utilitários e de execução
│   ├── data_preparation/       # Processamento e recorte de dados
│   ├── yolo_inaturalist/       # Mineração iNaturalist e pipeline YOLO
│   ├── analysis/               # Grad-CAM e explicabilidade
│   ├── main.py                 # Pipeline principal do classificador
│   └── main_tuning.py          # Busca de hiperparâmetros (Keras Tuner)
├── artifacts/                  # Artefatos do projeto
│   ├── data/                   # Datasets organizados (raw, processed, final, yolo)
│   ├── data_ingestion/         # Ingestão bruta de dados
│   ├── models/                 # Modelos CNN salvos
│   └── yolo_runs/              # Treinamentos do detector YOLOv8
└── logs/                       # Logs de execução
```

<a id="uso"></a>
## 📝 Uso

### Pipeline Completo
```bash
# Executar pipeline completo com configuração padrão
python scripts/main.py
```

### Tuning de Hiperparâmetros

**Testar diferentes modelos:**
```bash
# MobileNetV3 Large (padrão - rápido e eficiente)
python scripts/main_tuning.py --experiment mobilenetv3large

# MobileNetV3 Small (ultra-leve para mobile)
python scripts/main_tuning.py --experiment mobilenetv3small

# InceptionV3 (multi-escala, 299x299)
python scripts/main_tuning.py --experiment inceptionv3

# VGG19 (clássico, preciso mas pesado)
python scripts/main_tuning.py --experiment vgg19

# EfficientNet B0 (balanço performance/eficiência)
python scripts/main_tuning.py --experiment efficientnetb0

# EfficientNet B1 (240x240, balanceado)
python scripts/main_tuning.py --experiment efficientnetb1

# EfficientNet B2 (260x260, melhor precisão)
python scripts/main_tuning.py --experiment efficientnetb2

# EfficientNet B3 (300x300, alta precisão)
python scripts/main_tuning.py --experiment efficientnetb3

# EfficientNet B7 (máxima precisão, mais pesado)
python scripts/main_tuning.py --experiment efficientnetb7

# EfficientNetV2 B0 (versão V2, rápida e eficiente)
python scripts/main_tuning.py --experiment efficientnetv2b0

# EfficientNetV2 B1 (versão V2 com maior capacidade)
python scripts/main_tuning.py --experiment efficientnetv2b1

# EfficientNetV2 B2 (versão V2 com maior capacidade)
python scripts/main_tuning.py --experiment efficientnetv2b2

# EfficientNetV2 B3 (versão V2 com maior capacidade)
python scripts/main_tuning.py --experiment efficientnetv2b3

# ConvNeXtTiny
python scripts/main_tuning.py --experiment convnexttiny

# ConvNeXtSmall 
python scripts/main_tuning.py --experiment convnextsmall

# NASNet Mobile (AutoML, mobile-friendly)
python scripts/main_tuning.py --experiment nasnetmobile

# NASNet Large (AutoML, alta performance)
python scripts/main_tuning.py --experiment nasnetlarge

# MobileNet com compression blocks + SE
python scripts/main_tuning.py --experiment mobilenet_advanced
 
# MobileViT (Transformer para mobile, eficiente e moderno)
python scripts/main_tuning.py --experiment mobilevit-small

# MobileViT Customizado (versão aprimorada)
python scripts/main_tuning.py --experiment mobilevit-custom

# Vision Transformer (ViT) para datasets pequenos
python scripts/main_tuning.py --experiment vit_small_ds_v2
```

**Ajustar parâmetros de busca:**
```bash
python scripts/main_tuning.py \
    --experiment efficientnetb0 \
    --max-trials 50 \
    --epochs-per-trial 20 \
    --final-epochs 150
```

**Retreinar com hiperparâmetros salvos:**
```bash
# Retreinar com o mesmo experimento
python scripts/main_tuning.py \
    --mode retrain \
    --experiment mobilenetv3large \
    --best-hp-path artifacts/tuning/best_hyperparameters.json
```

<a id="dataset"></a>
## 📊 Dataset

O pipeline de dados do projeto é composto por três fontes de imagens reais de pragas agrícolas da soja:

1. **DatasetPests (GDrive)**: Base principal contendo fotos de pragas e anotações manuais de bounding boxes (Label-Studio CSV) feitas pelos professores.
2. **iNaturalist API**: Imagens de campo adicionais mineradas de forma automatizada para mitigar o desbalanceamento severo de classes.
3. **INSECT12C**: Base de dados externa usada para validação cruzada e teste de robustez independente.

### Organização de Diretórios sob `artifacts/data/`

* `artifacts/data/raw/`: Contém os pacotes compactados originais baixados das fontes.
* `artifacts/data/processed/`: Contém os recortes (*crops*) extraídos das imagens antes da divisão final.
  * `DatasetPests-cropped/`: Fusão de crops originais manuais com os crops gerados pelo detector automático YOLOv8 nas imagens brutas do iNaturalist.
* `artifacts/data/final/`: Contém as divisões finais prontas para modelagem:
  * `DatasetPests-split/`: Divisão segura de Treino (90%) e Validação (10%) agrupada por imagem-mãe (ID único da folha original) para evitar vazamento de dados (*data leakage*).
  * `INSECT12C-test/`: Conjunto de teste contendo recortes da base externa, com as 12 classes originais mapeadas para as 10 classes do classificador.
* `artifacts/data/yolo/`: Dataset com anotações convertidas em coordenadas normatizadas YOLO para treinamento do detector de pragas.

<a id="rag"></a>
## 🤖 Módulo RAG (Retrieval-Augmented Generation)

Além da classificação de imagens, o projeto inclui um módulo de **RAG** para perguntas e respostas com base em documentos técnicos sobre pragas e danos em folhas de soja.

### Componentes
- **Ingestão de documentos**: src/rag/ingestion/
- **Engine de consulta RAG**: src/rag/
- **API (busca + geração de resposta)**: src/api/

### Como executar (visão rápida)
```bash
# API
cd src
uvicorn api.main:app --reload

# (Opcional) pipeline de ingestão RAG
python -m rag.ingestion.main
```

> Configure as variáveis de ambiente necessárias em `src/api/config/settings.py` antes de executar.

### Configuração necessária

Crie um arquivo `.env` na raiz do projeto com, no mínimo, as variáveis abaixo:

```env
qdrant_url=https://<sua-instancia-qdrant>
qdrant_api_key=<sua-chave-qdrant>
groq_api_key=<sua-chave-groq>
```

Variáveis opcionais, com valores padrão definidos em `src/api/config/settings.py`:

```env
collection_name=agronomia-soja
dense_model=intfloat/multilingual-e5-large
sparse_model=Qdrant/bm25
colbert_model=colbert-ir/colbertv2.0
groq_base_url=https://api.groq.com/openai/v1
groq_model=llama-3.3-70b-versatile
```

### Documentação detalhada
- [README da API](src/api/README.md)
- [README do módulo RAG](src/rag/README.md)

## Arquitetura e Tecnologias de RAG

### 1) Chunking semântico (Semantic Chunking)
 Os documentos são segmentados com base em similaridade semântica, usando embeddings, HDBSCAN e controle de tamanho por tokens. Isso preserva melhor o contexto do que um chunking puramente fixo.

### 2) Embeddings híbridos
A busca combina múltiplos sinais de relevância, por exemplo:
- similaridade vetorial (semântica)
- correspondência lexical/palavra-chave
- reranking (quando aplicável)

Isso reduz falsos positivos e melhora cobertura de consultas técnicas.

### 2.1) Indexação vetorial com Qdrant
Os embeddings são armazenados e consultados no Qdrant, que atua como base vetorial para a recuperação híbrida do RAG.

### 3) Recuperação + Geração (RAG)
Fluxo principal:
1. recuperar chunks relevantes (SearchService)
2. montar contexto com fonte e página
3. gerar resposta estruturada com LLM (RAGService)
4. retornar resposta + metadados de rastreabilidade

### 4) Saída estruturada
A resposta é validada por schema (RAGOutput/RAGResponse), garantindo formato consistente para consumo pela API/UI.


<a id="projeto"></a>
## 🎓 Projeto

Desenvolvido como parte de pesquisa de mestrado.

<a id="licenca"></a>
## 📄 Licença

Este projeto está sob licença UTFPR.

---

**Status:** Em desenvolvimento 🚧
