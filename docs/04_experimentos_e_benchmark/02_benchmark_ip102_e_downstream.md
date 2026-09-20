# Relatório de Resultados — Benchmark IP102 (102 Classes) e Fine-Tuning Downstream

Este documento consolida os resultados quantitativos completos obtidos com todas as arquiteturas treinadas via pipeline `python scripts/train_ip102.py` (Passo 1: Pré-Treinamento de Domínio no dataset IP102 - 102 classes) e seu subsequente fine-tuning no dataset-alvo de pragas de insetos na soja (INSECT12C / DatasetPests), em comparação direta com a evolução do Estado da Arte (SOTA) da literatura (2019–2026).

---

## 1. Arquitetura da Estratégia de Treinamento

A abordagem desenvolvida para superar os desafios do dataset IP102 (variação intra-classe por metamorfose, desbalanceamento extremo de dados long-tail e ruído de fundo) é baseada em dois passos:

1. **Passo 1 — Domain-Specific Pre-training (IP102):**
   * **Dataset:** IP102 (102 classes oficiais de insetos/pragas, 2.229 imagens de validação).
   * **Função de Perda:** Categorical Focal Loss ($\gamma = 1.5$, $\alpha$ ponderado por frequência de classe inversa) para mitigar o desbalanceamento desproporcional entre classes.
   * **Mecanismo de Atenção/Pooling:** **Top-K Adaptive Pooling** ($k = 0.15$) para focar nos $15\%$ das regiões de maior relevância espacial e ignorar fundos ruidosos.
   * **Otimizador:** SGD com Cosine Decay Learning Rate Schedule e Warmup.
   * **Saída:** Extrator de características reutilizável (`ip102_<modelo>_extractor.keras`) e modelo completo (`ip102_<modelo>_full_model.keras`).

2. **Passo 2 — Fine-Tuning Downstream (INSECT12C / Pragas da Soja):**
   * Transferência dos pesos aprendidos no IP102 para o dataset de teste/validação de pragas de insetos na cultura da soja.

---

## 2. Resultados Empíricos de Todos os Modelos no IP102 (102 Classes)

Abaixo estão os resultados oficiais de inferência calculados sobre as 2.229 imagens do conjunto de validação do IP102 (102 classes):

| Modelo / Arquitetura | Parâmetros | Top-1 Accuracy | Top-5 Accuracy | Macro F1-Score | Weighted F1-Score | Extrator Gerado | Status IP102 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **MobileNetV3-Small** | ~2,5 M | **48,63%** | **83,22%** | 0,3046 | 0,4767 | `ip102_mobilenetv3small_extractor.keras` | Concluído |
| **EfficientNetV2-B1** | ~8,1 M | **57,47%** | **86,00%** | 0,3477 | 0,5428 | `ip102_pretrained_extractor.keras` | Concluído |
| **EfficientNetV2-B0** | ~7,1 M | **57,96%** | **88,07%** | 0,3905 | 0,5588 | `ip102_efficientnetv2b0_extractor.keras` | Concluído |
| **ConvNeXt-Tiny** | ~28,6 M | **57,96%** | **90,22%** 🏆 | **0,3712** | **0,5685** | `ip102_convnexttiny_extractor.keras` | Concluído |
| **MobileNetV3-Large** | ~5,4 M | **58,23%** 🏆 | **88,69%** | **0,4057** 🏆 | **0,5765** 🏆 | `ip102_mobilenetv3large_extractor.keras` | Concluído |

---

## 3. Desempenho Downstream no Fine-Tuning (DatasetPests & Teste Zero-Shot INSECT12C)

Após a transferência do aprendizado do IP102, os modelos foram submetidos a fine-tuning com Keras Tuner no **DatasetPests** (10 classes) e avaliados em teste independente zero-shot no **INSECT12C**:

| Modelo / Arquitetura | Val Acc (DatasetPests) | Test Acc (INSECT12C Geral) | Test Acc ($GT > 60\text{px}$) | Test Acc ($GT > 100\text{px}$) | Val Macro F1 (Soja) | Test Macro F1 (INSECT12C) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Small (IP102)** | 91,10% | 62,03% | 70,49% | 75,38% | 0,8984 | 0,5959 |
| **MobileNetV3-Large (IP102)** | **95,28%** 🏆 | 62,38% | 74,47% | 82,68% | 0,9432 | 0,6444 |
| **EfficientNetV2-B0 (IP102)** | 94,84% | 64,13% | 76,28% | 83,70% | **0,9559** 🏆 | 0,6748 |
| **EfficientNetV2-B1 (IP102)** | 94,30% | **64,59%** 🏆 | **78,02%** 🏆 | **85,91%** 🏆 | 0,9345 | **0,6806** 🏆 (0,8215 >100px) |
| **ConvNeXt-Tiny (IP102)** | 93,32% | 59,63% | 73,04% | 80,81% | 0,9348 | 0,6247 |
| **ConvNeXt-Tiny (ImageNet)** | 94,39% | 60,39% | 74,35% | 83,02% | 0,9504 | 0,6429 |

> 📌 **Nota Metodológica:** A coluna *Val Acc* refere-se ao particionamento de validação sem vazamento do **DatasetPests** (amostras em condições reais do iNaturalist com anotações manuais via Label-Studio e mineração complementar). O **INSECT12C** atua estritamente como a base de teste externa de robustez (zero-shot, sem fine-tuning).

---

## 4. Comparativo Completo com o Estado da Arte (SOTA) da Literatura (IP102 — 102 Classes)

| Ano | Modelo / Abordagem | Top-1 Accuracy | Foco / Mecanismo da Técnica | Categoria de Complexidade |
| :---: | :--- | :---: | :--- | :--- |
| **2019** | **ResNet-50 (Baseline CVPR — Wu et al.)** | **49,40%** | Fine-tuning padrão a partir de ImageNet. | Monoestágio Basal |
| **2019** | **MobileNetV2 (Baseline CVPR — Wu et al.)** | **50,20%** | Classificação leve padrão. | Monoestágio Basal |
| **2019** | **DenseNet121 (Baseline CVPR — Wu et al.)** | **53,10%** | Conexões densas sem atenção. | Monoestágio Basal |
| **Nosso** | **MobileNetV3-Small (Nosso Modelo)** | **48,63%** | **Top-K Pooling ($k=0.15$) + Focal Loss.** | **Monoestágio Ultraleve (~2.5M)** |
| **Nosso** | **EfficientNetV2-B1 (Nosso Modelo)** | **57,47%** | **Top-K Pooling ($k=0.15$) + Focal Loss.** | **Monoestágio Especializado (~8.1M)** |
| **Nosso** | **EfficientNetV2-B0 (Nosso Modelo)** | **57,96%** | **Top-K Pooling ($k=0.15$) + Focal Loss.** | **Monoestágio Especializado (~7.1M)** |
| **Nosso** | **ConvNeXt-Tiny (Nosso Modelo)** | **57,96%** | **Kernels 7x7 + Top-K Pooling ($k=0.15$) + Focal Loss.** | **Monoestágio Moderno (~28.6M)** |
| **Nosso** | **MobileNetV3-Large (Nosso Modelo)** | **58,23%** 🏆 | **Top-K Pooling ($k=0.15$) + Focal Loss.** | **Monoestágio Especializado (~5.4M)** |
| **2020–2021** | GAEnsemble / MnasNet-DenseNet | 67,10% | Combinação de múltiplas CNNs e data augmentation. | Ensemble Multi-Modelo |
| **2022–2023** | ViT-Base / Swin Transformer | 69,00% – 73,00% | Mecanismos de autoatenção para ignorar fundos ruidosos. | Vision Transformer |
| **2023–2024** | Faster-PestNet / Refined ResNet-50 | 76,00% – 82,40% | Otimizadores avançados (SWA/EMA), regularização e Mixup. | CNN Otimizada + SWA |
| **2024–2026** | CT-PestNet | 77,32% | Destilação cruzada + ViT com interação de tokens locais. | Híbrido CNN-ViT |
| **2025–2026** | Coarse-to-Fine / Multi-stage Networks | 85,00% – 91,00%* | Decomposição hierárquica por subcentros (fases de vida). | Multi-Estágio Hierárquico |

*\*Resultados acima de 85% utilizam estratégias de múltiplos estágios (supervisão auxiliar por estágio de vida do inseto - ovo/larva/adulto), e não modelos ponta a ponta puramente monoestágio.*

---

## 5. Principais Conclusões e Insights para Benchmarking

1. **Recorde de Top-5 Accuracy do ConvNeXt-Tiny (90.22%):**
   * O **ConvNeXt-Tiny** atingiu o **maior Top-5 Accuracy de todo o benchmark ($90.22\%$)**, demonstrando que os convolucionais modernos de kernel largo ($7 \times 7$) constroem um espaço latente de representação morfológica extraordinariamente rico, onde a classe correta está no top-5 em 9 de cada 10 imagens.
2. **Campeão de Top-1 Accuracy e Eficiência (MobileNetV3-Large):**
   * O **MobileNetV3-Large** consolidou a liderança em **Top-1 Acc ($58.23\%$)** e **Macro F1 ($0.4057$)**, com uma excelente relação de parâmetros (~5.4M).
