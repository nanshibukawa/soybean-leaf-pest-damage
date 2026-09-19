# 📚 Metodologia, Decisões Experimentais e Controle de Hiperparâmetros

Este documento centraliza todas as decisões metodológicas, critérios de rigor científico e o histórico de configurações dos experimentos da pesquisa científica em classificação de pragas de folhas de soja.

---

## 1. 🔬 Rigor Científico e Controle Experimental (*Ceteris Paribus*)

Para que as comparações entre o **Baseline (ImageNet)** e a **Proposta (IP102 + Fine-Tuning)** sejam cientificamente válidas e rigorosamente comparáveis, adota-se o princípio de controle estrito de variáveis.

### 1.1. O que é Mantido Estritamente Idêntico:
* **Dados e Particionamento (*Splits*)**:
  - Exatamente as mesmas imagens no conjunto de treino (90%) e de validação (10%).
  - Particionamento estratificado por **Imagem-Mãe (*Group-based Split*)** para evitar vazamento de dados (*data leakage*) entre recortes da mesma folha.
  - Semente aleatória fixa: `seed=42`.
* **Resolução de Entrada**:
  - Fixa por família arquitetural: $224 \times 224$ (MobileNetV3, ConvNeXt, EfficientNetV2-B0) e $240 \times 240$ (EfficientNetV2-B1).
* **Pipeline de Data Augmentation**:
  - Idêntico para todos os modelos: *CutMix*, *RandomFlip* (horizontal e vertical), *RandomTranslation* (15%), *RandomZoom* (20%) e *Gaussian Noise* (0.03).
* **Função de Perda (*Loss Function*)**:
  - *Categorical Focal Loss* com $\gamma=1.5$ e ponderação dinâmica $\alpha$ calculada por classe para equilibrar classes raras.

---

## 2. ⚖️ Decisão Metodológica: Otimização de Hiperparâmetros (Keras Tuner)

### Dúvida Histórica da Pesquisa:
> *"Para comparação do baseline e do IP102, deve-se utilizar exatamente os mesmos hiperparâmetros fixos, ou cada um deve ser tunado?"*

### Decisão Adotada: **Tuning Bayesiano Independente**
* **Justificativa Teórica**:
  - A superfície de erro (*loss landscape*) de uma rede que parte dos pesos gerais do **ImageNet** é fundamentalmente diferente de uma rede que já foi pré-treinada no domínio entomológico do **IP102**.
  - Se forçássemos os mesmos hiperparâmetros (ex: mesma taxa de aprendizado ou mesmo número de camadas descongeladas), introduziríamos um **viés de seleção**: a banca poderia apontar que o Baseline perdeu não pela limitação do ImageNet, mas porque a taxa de aprendizado escolhida favorecia a convergência do modelo IP102.
  - Ao executar uma **Otimização Bayesiana (Keras Tuner com 30 trials)** em ambos, garantimos uma comparação do **potencial máximo de cada abordagem** (*fair apple-to-apple comparison*).

### Espaço de Busca Otimizado:
* **Taxa de Aprendizado ($lr$)**: Busca logarítmica entre $10^{-6}$ e $10^{-3}$.
* **Dropout**: Entre $0.1$ e $0.6$ (passo $0.1$).
* **Camadas Descongeladas (*Unfreeze Last N Layers*)**: Entre 16 e 128 camadas (com Batch Normalization mantido em modo inferência para estabilidade).
* **Regularização $L_2$**: Entre $10^{-5}$ e $5 \times 10^{-2}$.
* **Top-K Pooling Ratio**: Percentual de retenção de ativações espaciais (5% a 25%).

---

## 3. 🏷️ Padrão Oficial de Nomenclatura dos Modelos

Para clareza em gráficos, tabelas e no texto do manuscrito, adotou-se a seguinte nomenclatura padronizada:

| Estratégia | Nomenclatura nos Gráficos/Tabelas | Descrição Metodológica |
| :--- | :--- | :--- |
| **Baseline Tradicional** | `<Modelo> (ImageNet)` | Pesos pré-treinados no ImageNet ➔ Fine-tuning no DatasetPests. |
| **Proposta da Pesquisa** | `<Modelo> (IP102)` | ImageNet ➔ Pré-treino no IP102 (102 classes) ➔ Fine-tuning no DatasetPests. |
| **Sem Pré-Treino** | `<Modelo> (From Scratch)` | Inicialização aleatória dos pesos ➔ Treino direto no DatasetPests. |

---

## 4. 📊 Matriz de Modelos e Status Atual

| Arquitetura | Estratégia | DatasetPests (Val Acc) | INSECT12C (Test Acc) | Latência GPU | Status / Observação |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **MobileNetV3-Large** | Baseline (ImageNet) | 91.27% | 61.88% | 55.23 ms | ✅ Concluído |
| **MobileNetV3-Large** | Proposta (IP102) | **95.28%** | **62.38%** | 54.18 ms | ✅ Concluído (Líder em Validação) |
| **MobileNetV3-Small** | Baseline (ImageNet) | 87.18% | 48.59% | **52.80 ms** | ✅ Concluído |
| **MobileNetV3-Small** | Proposta (IP102) | **91.10%** | **62.03%** | **52.78 ms** | ✅ Concluído (**Salto de +13.44%** no teste externo) |
| **EfficientNetV2-B1** | Baseline (ImageNet) | 90.03% | 57.94% | 58.01 ms | ✅ Concluído |
| **EfficientNetV2-B1** | Proposta (IP102) | **94.30%** | <span style="color:green">**64.59%**</span> | 59.36 ms | ✅ Concluído (**Líder em Robustez Externa**) |
| **EfficientNetV2-B0** | Proposta (IP102) | **94.84%** | **64.13%** | 58.09 ms | ✅ Concluído (**Maior F1-Score: 0.9559**) |
| **EfficientNetV2-B0** | Baseline (ImageNet) | 91.18% | 61.27% | 60.12 ms | ✅ Concluído (F1 Soja: 0.8917 / INSECT12C: 0.6301) |
| **MobileViT** | From Scratch | **52.27%** | **40.03%** | 56.90 ms | ✅ Concluído (Estudo de caso: ausência de viés indutivo) |
| **ConvNeXt-Tiny** | Proposta (IP102) | **93.32%** | **59.63%** | **58.17 ms** | ✅ Concluído (**80.81%** em imagens grandes >100px) |
| **ConvNeXt-Tiny** | Baseline (ImageNet) | **94.39%** | **60.39%** | **57.89 ms** | ✅ Concluído (Modelo de alta capacidade ImageNet) |

> 📌 **Status do Benchmark Oficial:** Todos os modelos acima foram avaliados de forma unificada e compõem as curvas de Pareto e a Tabela 1 oficial do artigo científico.

---

## 5. 📦 Estrutura dos Datasets e Ingestão Automática

Para garantir reprodutibilidade sem necessidade de downloads manuais pesados, o projeto conta com ingestão automática configurada no Google Drive:

* **Link Google Drive do `data_final.zip`**:  
  `https://drive.google.com/file/d/1Af5dJrsr8ovDDkhBszBge2LzQcxm1895/view?usp=sharing` (ID: `1Af5dJrsr8ovDDkhBszBge2LzQcxm1895`)
* **Conteúdo do pacote (~516 MB compactado)**:
  - `artifacts/data/final/DatasetPests-split/` (11.468 fotos recortadas com margem +20%, divididas por imagem-mãe).
  - `artifacts/data/final/INSECT12C-test/` (2.618 recortes anotados para teste externo estratificado por resolução).
  - `artifacts/data/ip102_cropped/` (22.282 recortes cobrindo as 102 classes agrícolas do IP102).

### Como inicializar em qualquer máquina nova:
```bash
.venv/bin/python src/cnnClassifier/pipeline/stage_01_data_ingestion.py
```
*(O script detecta automaticamente se os dados já existem localmente; se não existirem, baixa do Drive e extrai nas pastas corretas).*

---

## 6. 🚀 Guia de Comandos para Treinamento e Benchmarking

### 6.1. Treinar / Tunar Baseline (ImageNet)
```bash
.venv/bin/python scripts/main_tuning.py --experiment <nome_do_experimento>_imagenet_baseline --max-trials 30 --epochs-per-trial 40 --final-epochs 80
```

### 6.2. Treinar Proposta (IP102 + Fine-Tuning)
```bash
# Passo 1: Pré-treinamento no IP102
.venv/bin/python scripts/train_ip102.py --model <nome_do_modelo> --experiment ip102_<nome_do_modelo>_pretrain

# Passo 2: Fine-Tuning Bayesiano no DatasetPests
.venv/bin/python scripts/main_tuning.py --experiment <nome_do_modelo>_finetune --max-trials 30 --epochs-per-trial 40 --final-epochs 80
```

### 6.3. Atualizar Dados do Benchmark e Gráficos de Pareto
```bash
# 1. Gera predições em lote e mede latência GPU/CPU
.venv/bin/python scripts/generate_benchmark_data.py

# 2. Gera os PDFs e PNGs da Fronteira de Pareto, Heatmaps e Análise de Resolução
.venv/bin/python graphics/benchmark_graficos_avaliacao_predicao.py
```
Os arquivos gerados são salvos automaticamente em `graphics/output/`.
