# 📊 Pré-Treinamento no IP102 (102 Classes) e Transferência Downstream

Este documento consolida os resultados quantitativos obtidos com as cinco arquiteturas monoestágio de borda pré-treinadas via pipeline `python scripts/train_ip102.py` (Passo 1: Pré-Treinamento de Domínio no dataset IP102 - 102 classes) e seu subsequente fine-tuning no catálogo de pragas da soja (**DatasetPests**) com avaliação cega zero-shot no benchmark independente (**INSECT12C**).

---

## 1. Arquitetura da Estratégia de Treinamento

A abordagem desenvolvida para superar os desafios do dataset IP102 (variação intra-classe por metamorfose, desbalanceamento extremo de dados long-tail e ruído de fundo) é baseada em dois passos:

1. **Passo 1 — Domain-Specific Pre-training (IP102):**
   * **Dataset:** IP102 (102 classes oficiais de insetos/pragas, 2.229 imagens de validação).
   * **Alinhamento de Domínio por Recortes:** O treinamento foi conduzido diretamente sobre os recortes focados das caixas delimitadoras (`artifacts/data/ip102_cropped/`), alinhando intencionalmente a distribuição espacial de entrada com a saída do detector (YOLOv8) do sistema final de dois estágios.
   * **Função de Perda:** [**Categorical Focal Loss**](01_analise_focal_loss_soja.md) ($\gamma = 1.5$, vetor $\alpha$ suavizado pelo Número Efetivo de Amostras — Cui et al., CVPR 2019, com $\beta=0.999$) para mitigar o desbalanceamento desproporcional entre classes sem desestabilizar os gradientes.
   * **Mecanismo de Atenção/Pooling:** [**Top-K Adaptive Pooling**](../01_ROTEIRO_TECNICO_E_METODOLOGICO.md) ($k = 0.15$) para focar nos 15% de ativações espaciais mais intensas da praga, descartando o ruído de fundo sem introduzir parâmetros treináveis adicionais.
   * **Estratégia de Otimização:** **SGD com Momentum ($0.9$) e Nesterov** (Wilson et al., 2017), *Linear Warmup* de 5 épocas (Goyal et al., 2017) para prevenir que gradientes iniciais instáveis destruam os pesos do backbone, e *Cosine Decay Schedule* (Loshchilov & Hutter, 2017) desacelerando suavemente a taxa de aprendizado até a época 50 para guiar os parâmetros a bacias estáveis de convergência (*flat minima*).
   * **Saída:** Extrator de características reutilizável (`ip102_<modelo>_extractor.keras`) e modelo completo (`ip102_<modelo>_full_model.keras`).

2. **Passo 2 — Fine-Tuning Downstream (INSECT12C / Pragas da Soja):**
   * Transferência dos pesos aprendidos no IP102 para o dataset de teste/validação de pragas de insetos na cultura da soja.

---

## 2. Resultados Empíricos dos Modelos no IP102 (102 Classes)

Abaixo estão os resultados oficiais de inferência calculados sobre as 2.229 imagens do conjunto de validação do IP102 (102 classes recortadas):

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

> 📌 **Nota Metodológica:** A coluna *Val Acc* refere-se ao particionamento de validação sem vazamento do **DatasetPests** (amostras em condições reais do iNaturalist com anotações manuais e mineração complementar). O **INSECT12C** atua estritamente como a base de teste externa de robustez (zero-shot, sem fine-tuning).

---

## 4. Síntese Comparativa das Arquiteturas de Borda

1. **Eficiência Paramétrica da MobileNetV3-Large:**
   * Atingiu a maior acurácia Top-1 no IP102 (**58,23%**) e o maior Macro F1 (**0,4057**), demandando apenas **5,4M de parâmetros**, demonstrando excelente adequação para dispositivos com restrição severa de memória.
2. **Capacidade Representacional da ConvNeXt-Tiny:**
   * Alcançou o maior Top-5 de todo o benchmark (**90,22%**), comprovando que convoluções modernas com kernels largos ($7 \times 7$) estruturam um espaço latente morfológico robusto, capturando variações taxonômicas complexas.
3. **Resiliência e Generalização Externa da EfficientNetV2-B1:**
   * Na etapa downstream, a **EfficientNetV2-B1** confirmou-se como a arquitetura mais equilibrada para aplicação prática: liderou o teste cego zero-shot no INSECT12C em todos os cenários de resolução (**64,59%** geral, **78,02%** em $>60\text{px}$ e **85,91%** em $>100\text{px}$), justificando sua seleção como o modelo de referência do projeto.

---

## 5. Referências Bibliográficas

1. **CUI, Yin; JIA, Menglin; LIN, Tsung-Yi; SONG, Yang; BELONGIE, Serge.** *Class-Balanced Loss Based on Effective Number of Samples*. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 9268-9277. DOI: [10.1109/CVPR.2019.00949](https://doi.org/10.1109/CVPR.2019.00949).
2. **GOYAL, Priya; DOLLÁR, Piotr; GIRSHICK, Ross; NOORDHUIS, Pieter; WESOLOWSKI, Lukasz; KYROLA, Aapo; TULLOCH, Andrew; JIA, Yangqing; HE, Kaiming.** *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*. arXiv preprint arXiv:1706.02677, 2017. DOI: [10.48550/arXiv.1706.02677](https://doi.org/10.48550/arXiv.1706.02677).
3. **LOSHCHILOV, Ilya; HUTTER, Frank.** *SGDR: Stochastic Gradient Descent with Warm Restarts*. In: International Conference on Learning Representations (ICLR), 2017. DOI: [10.48550/arXiv.1608.03983](https://doi.org/10.48550/arXiv.1608.03983).
4. **WILSON, Ashia C.; ROELOFS, Rebecca; STERN, Mitchell; SREBRO, Nathan; RECHT, Benjamin.** *The Marginal Value of Adaptive Gradient Methods in Machine Learning*. In: Advances in Neural Information Processing Systems (NeurIPS), v. 30, 2017, pp. 4148-4158.
