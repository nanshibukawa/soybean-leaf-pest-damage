# Soybean Leaf Pest Damage Detection

Sistema de detecção de danos causados por pragas em folhas de soja utilizando deep learning.

## 📋 Descrição

Este projeto implementa um sistema de classificação de imagens para identificar danos causados por pragas em folhas de soja, utilizando redes neurais convolucionais (CNN) com TensorFlow/Keras.

## 🚀 Tecnologias

- Python 3.12+
- TensorFlow 2.17.1
- Keras
- scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn

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

## 🏗️ Estrutura do Projeto

```
├── src/cnnClassifier/     # Código fonte principal
│   ├── components/        # Componentes do pipeline
│   ├── config/           # Configurações
│   ├── entity/           # Entidades e modelos
│   ├── pipeline/         # Pipelines de treinamento e predição
│   └── utils/            # Funções utilitárias
├── notebooks/            # Jupyter notebooks para análise
├── scripts/              # Scripts de execução
├── artifacts/            # Artefatos gerados (modelos, dados)
└── logs/                 # Logs de execução
```

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

# EfficientNet B7 (máxima precisão, mais pesado)
python scripts/main_tuning.py --experiment efficientnetb7

# NASNet Mobile (AutoML, mobile-friendly)
python scripts/main_tuning.py --experiment nasnetmobile

# NASNet Large (AutoML, alta performance)
python scripts/main_tuning.py --experiment nasnetlarge

# MobileNet com compression blocks + SE
python scripts/main_tuning.py --experiment mobilenet_advanced
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

## 📊 Dataset

O projeto utiliza imagens de folhas de soja com diferentes níveis de danos causados por pragas.

## 🎓 Projeto

Desenvolvido como parte de pesquisa de mestrado.

## 📄 Licença

Este projeto está sob licença [especificar licença].

---

**Status:** Em desenvolvimento 🚧