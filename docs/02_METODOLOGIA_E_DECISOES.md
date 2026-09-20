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
  - *Categorical Focal Loss* com $\gamma=1.5$ e ponderação dinâmica $\alpha$ calculada por classe via Número Efetivo de Amostras (Cui et al., CVPR 2019) para equilibrar classes raras.
* **Estratégia de Otimização no Pré-treino de Domínio (Passo 1 — IP102)**:
  - **SGD com Momentum (0.9) e Nesterov** (Wilson et al., 2017): Adotado no pré-treinamento em larga escala por induzir superfícies de perda com maior capacidade de generalização em visão computacional quando comparado a otimizadores adaptativos sem regularização estrita.
  - **Linear Warmup (5 épocas)** (Goyal et al., 2017): Inicialização suave em $10^{-4}$ com elevação linear até o pico de $10^{-2}$, impedindo que os gradientes elevados da cabeça densa recém-inicializada desestabilizem os pesos convolucionais do backbone (*catastrophic forgetting*).
  - **Cosine Decay Learning Rate Schedule** (Loshchilov & Hutter, 2017): Desaceleração contínua e suave da taxa de atualização até a época 50 ($10^{-4}$), guiando os pesos a mínimos planos estáveis (*flat minima*).

---

## 2. ⚖️ Decisão Metodológica: Otimização de Hiperparâmetros (Keras Tuner)

### Abordagem Adotada: **Otimização Bayesiana Independente**
* **Justificativa Metodológica**:
  - A superfície de perda (*loss landscape*) de uma arquitetura que parte dos pesos genéricos do **ImageNet** é fundamentalmente diferente de uma arquitetura previamente adaptada ao domínio entomológico do **IP102**.
  - A imposição de hiperparâmetros fixos compartilhados (ex.: mesma taxa de aprendizado ou número de camadas descongeladas) introduziria um **viés de seleção**: o baseline poderia apresentar desempenho inferior não por deficiência representacional dos pesos ImageNet, mas pela subotimalidade da taxa de aprendizado para a sua dinâmica de convergência específica.
  - Ao executar **Otimização Bayesiana (Keras Tuner com 30 trials)** de forma independente para cada configuração, viabilizou-se a comparação do **potencial ótimo de cada abordagem** (*fair comparison*).

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

---

## 7. 📚 Referências Bibliográficas

1. **CUI, Yin; JIA, Menglin; LIN, Tsung-Yi; SONG, Yang; BELONGIE, Serge.** *Class-Balanced Loss Based on Effective Number of Samples*. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 9268-9277. DOI: [10.1109/CVPR.2019.00949](https://doi.org/10.1109/CVPR.2019.00949).
2. **GOYAL, Priya; DOLLÁR, Piotr; GIRSHICK, Ross; NOORDHUIS, Pieter; WESOLOWSKI, Lukasz; KYROLA, Aapo; TULLOCH, Andrew; JIA, Yangqing; HE, Kaiming.** *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*. arXiv preprint arXiv:1706.02677, 2017. DOI: [10.48550/arXiv.1706.02677](https://doi.org/10.48550/arXiv.1706.02677).
3. **LIN, Tsung-Yi; GOYAL, Priya; GIRSHICK, Ross; HE, Kaiming; DOLLÁR, Piotr.** *Focal Loss for Dense Object Detection*. In: Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017, pp. 2980-2988. DOI: [10.1109/ICCV.2017.324](https://doi.org/10.1109/ICCV.2017.324).
4. **LOSHCHILOV, Ilya; HUTTER, Frank.** *SGDR: Stochastic Gradient Descent with Warm Restarts*. In: International Conference on Learning Representations (ICLR), 2017. DOI: [10.48550/arXiv.1608.03983](https://doi.org/10.48550/arXiv.1608.03983).
5. **WILSON, Ashia C.; ROELOFS, Rebecca; STERN, Mitchell; SREBRO, Nathan; RECHT, Benjamin.** *The Marginal Value of Adaptive Gradient Methods in Machine Learning*. In: Advances in Neural Information Processing Systems (NeurIPS), v. 30, 2017, pp. 4148-4158.
