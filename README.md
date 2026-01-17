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

```bash
# Executar pipeline completo
python scripts/main.py

# Tuning de hiperparâmetros (com opções)
python scripts/main_tuning.py --mode tune --max-trials 30 --epochs-per-trial 30 --final-epochs 100

# Retreinar modelo com hiperparâmetros já encontrados
python scripts/main_tuning.py --mode retrain --best-hp-path artifacts/tuning/best_hyperparameters.json
```

## 📊 Dataset

O projeto utiliza imagens de folhas de soja com diferentes níveis de danos causados por pragas.

## 🎓 Projeto

Desenvolvido como parte de pesquisa de mestrado.

## 📄 Licença

Este projeto está sob licença [especificar licença].

---

**Status:** Em desenvolvimento 🚧