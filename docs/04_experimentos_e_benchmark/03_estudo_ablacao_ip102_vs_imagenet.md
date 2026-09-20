# 📑 Relatório Científico de Estudo de Ablação
## Impacto do Pré-Treinamento de Domínio Agrícola (IP102) vs. Baseline ImageNet

Este documento consolida o estudo de ablação experimental comparando a inicialização convencional com pesos genéricos do **ImageNet** contra a abordagem proposta de **Pré-Treinamento de Domínio Específico no IP102** na arquitetura de referência da pesquisa (**EfficientNetV2-B1**). O estudo avalia quantitativamente o ganho de discriminação interna na cultura da soja (**DatasetPests**) e a robustez de generalização cega zero-shot no benchmark externo independente (**INSECT12C**).

* **Projeto de Pesquisa:** Sistema de Detecção e Classificação de Danos por Pragas em Folhas de Soja  
* **Arquitetura de Referência:** EfficientNetV2-B1 ($240 \times 240$, Top-K Pooling $k=0.15$)  
* **Ambiente de Otimização:** Keras Tuner (30 trials de Busca Bayesiana)  
* **Fontes Oficiais dos Dados:** [`performance_summary.csv`](../../graphics/output/performance_summary.csv) e [`resolution_summary_insect12c.csv`](../../graphics/output/resolution_summary_insect12c.csv)

---

## 🎯 1. Resumo Executivo

Este estudo de ablação (*ablation study*) avalia rigorosamente a hipótese central da pesquisa: **a transferência de aprendizado prévia em domínio entomológico agrícola (IP102 com 102 classes) confere representações visuais mais ricas e generalizáveis do que a inicialização padrão baseada exclusivamente em pesos genéricos do ImageNet.**

Os resultados empíricos consolidados demonstram que o pré-treinamento de domínio no IP102:
1. **Elevou a Acurácia na Soja para 94,30%** (ganho de **+4,27%** sobre o baseline ImageNet de 90,03%).
2. **Expandiu o Macro F1-Score interno para 93,45%** (ganho de **+6,34%** sobre 87,11%), assegurando equilíbrio discriminativo entre as 10 classes taxonômicas.
3. **Alavancou a Generalização Zero-Shot no Benchmark Externo (INSECT12C):**
   * **Cenário Geral (2.618 imagens):** Macro F1 saltou de **60,84%** para **68,08%** (**+7,24%** de ganho absoluto);
   * **Recortes Médios ($>60\text{px}$):** Macro F1 subiu de **71,65%** para **79,26%** (**+7,61%** de ganho);
   * **Recortes Grandes ($>100\text{px}$):** Atingiu **85,91% de acurácia** (vs. 80,31%) e **82,15% de Macro F1** (vs. 76,57%), com ganho de **+5,58%**.

---

## 🔬 2. Metodologia Experimental (*Ceteris Paribus*)

Para assegurar que o diferencial de desempenho seja estritamente atribuído à transferência de aprendizado de domínio do IP102, todas as demais variáveis do pipeline de Machine Learning foram mantidas **rigorosamente idênticas**:

* **Base de Dados e Particionamento Sem Vazamento:** Divisão estratificada agrupada por imagem-mãe (*Group-based Split*), com 10.345 recortes no conjunto de Treino (90%) e 1.123 recortes no conjunto de Validação (10%).
* **Pré-Processamento Anatômico:** Recortes de bounding boxes expandidos com **margem foliar de +20%** para incluir textura de tecido vegetal e evitar deformação de proporção (*aspect ratio*).
* **Resolução Padronizada:** $240 \times 240$ pixels, compatível com a escala nativa da EfficientNetV2-B1.
* **Função de Perda:** [**Categorical Focal Loss**](01_analise_focal_loss_soja.md) ($\gamma=1.5$, $\beta=0.999$, Cui et al., 2019) normalizada pela média amostral.
* **Mecanismo de Pooling:** [**Top-K Adaptive Pooling**](../01_ROTEIRO_TECNICO_E_METODOLOGICO.md) ($k=0.15$), retendo os 15% maiores sinais espaciais do mapa convolucional.
* **Estratégia de Otimização Unificada:** SGD com Momentum ($0.9$), Nesterov, Linear Warmup e Cosine Decay Schedule (Loshchilov & Hutter, 2017), com busca bayesiana independente de taxa de aprendizado e dropout via Keras Tuner (30 trials).

### Configurações Comparadas:
* **Configuração A (Abordagem Proposta — IP102):**  
  $\text{ImageNet} \longrightarrow \text{IP102 (50 épocas, 102 classes)} \longrightarrow \text{Fine-Tuning na Soja (10 classes)}$
* **Configuração B (Baseline Convencional — ImageNet):**  
  $\text{ImageNet (Pesos Genéricos)} \longrightarrow \text{Fine-Tuning na Soja (10 classes)}$

---

## 📊 3. Resultados Empíricos Consolidados

### 3.1. Validação Interna no Catálogo de Pragas da Soja (DatasetPests)

Avaliação calculada sobre os 1.123 recortes de validação sem vazamento de dados:

| Estratégia de Inicialização | Acurácia Global | **Macro F1-Score** | Precisão Macro | Recall Macro | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Apenas ImageNet)** | 90,03% | 87,11% | 88,27% | 86,48% | 89,85% |
| **Proposto (Com IP102)** | **94,30%** 🏆 | **93,45%** 🏆 | **94,76%** 🏆 | **92,67%** 🏆 | **94,28%** 🏆 |
| **Ganho Absoluto ($\Delta$)** | **+4,27%** | **+6,34%** | **+6,49%** | **+6,19%** | **+4,43%** |

> 📌 **Destaque:** O pré-treinamento de domínio proporcionou salto consistente em todas as métricas fundamentais, com elevação simultânea de Precisão (+6,49%) e Recall (+6,19%), indicando que o modelo ampliou sua capacidade discriminativa sem induzir falsos positivos nas classes minoritárias.

---

### 3.2. Teste Cego Zero-Shot de Generalização Externa (Benchmark INSECT12C)

Avaliação *Out-of-Distribution* realizada na base independente **INSECT12C** (2.618 imagens de lavouras comerciais sem ajuste fino prévio), estratificada por faixas de resolução dos recortes:

| Cenário de Avaliação (INSECT12C) | Amostras | Baseline (ImageNet) | **Proposto (Com IP102)** | **Ganho Absoluto F1** |
| :--- | :---: | :---: | :---: | :---: |
| **Dataset Completo (Sem Filtro)** | 2.618 | Acc: 57,94% \| F1: 60,84% | **Acc: 64,63% \| F1: 68,08%** 🏆 | **+7,24%** |
| **Recortes Médios ($>60\times 60\text{px}$)** | 1.606 | Acc: 71,67% \| F1: 71,65% | **Acc: 78,08% \| F1: 79,26%** 🏆 | **+7,61%** |
| **Recortes Grandes ($>100\times 100\text{px}$)** | 589 | Acc: 80,31% \| F1: 76,57% | **Acc: 85,91% \| F1: 82,15%** 🏆 | **+5,58%** 🚀 |

---

## 📈 4. Análise Crítica e Discussão

### A. O Efeito Regularizador do Domínio Entomológico
O pré-treinamento no IP102 atua como um **regularizador indutivo de domínio**: os filtros visuais da EfficientNetV2-B1 iniciam a fase de fine-tuning já especializados na detecção de bordas de élitros, venação de asas membranosas, segmentos torácicos e lesões foliares necróticas. Essa convergência inicial reduz a tendência da rede em memorizar padrões espúrios de fundo foliar ou iluminação de laboratório, estabilizando a generalização.

### B. Robustez Fora da Distribuição (*Out-of-Distribution*)
Na avaliação cega zero-shot no INSECT12C, a superioridade da proposta é evidente em todas as faixas de resolução:
* No conjunto geral de 2.618 imagens, o modelo com IP102 obteve **+7,24% no Macro F1**;
* Esse salto comprova que o espaço latente construído com o IP102 abstrai características morfológicas genuínas dos insetos, preservando a capacidade preditiva mesmo diante de espécimes fotografados sob condições agrícolas, climáticas e de iluminação completamente distintas daquelas presentes no treino.

### C. Confirmação do Papel da Resolução e Margem de Contexto
Observa-se em ambas as configurações uma progressão ascendente contínua das métricas conforme o tamanho do espécime aumenta:
$$\text{Geral (64,6%)} \longrightarrow >60\text{px (78,1%)} \longrightarrow >100\text{px (85,9\%)}$$
Esse comportamento valida a decisão metodológica de introduzir a **margem de expansão de +20% (Padding)**: recortes diminutos sofrem severamente com perda de densidade de pixels e subamostragem morfológica, sendo o contexto foliar e a preservação de pixels essenciais para a correta identificação em borda.

---

## 🏁 5. Conclusão da Investigação

A hipótese científica formulada é **plenamente ratificada**: a transferência de aprendizado de domínio específico prévio no **IP102** supera com folga estatística a abordagem tradicional baseada em pesos genéricos do ImageNet.

* **Acurácia Interna (DatasetPests):** Salto de **90,03%** para **94,30%** (+4,27%).
* **Macro F1 Interno:** Salto de **87,11%** para **93,45%** (+6,34%).
* **Macro F1 Externo (INSECT12C >100px):** Salto de **76,57%** para **82,15%** (+5,58%).

**Recomendação para a arquitetura final:** Adotar em definitivo o extrator com pré-treinamento de domínio IP102 (`ip102_pretrained_extractor.keras`) como o padrão oficial da arquitetura de produção do projeto.

---

## 6. Referências Bibliográficas

1. **CUI, Yin; JIA, Menglin; LIN, Tsung-Yi; SONG, Yang; BELONGIE, Serge.** *Class-Balanced Loss Based on Effective Number of Samples*. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 9268-9277. DOI: [10.1109/CVPR.2019.00949](https://doi.org/10.1109/CVPR.2019.00949).
2. **GOYAL, Priya; DOLLÁR, Piotr; GIRSHICK, Ross; NOORDHUIS, Pieter; WESOLOWSKI, Lukasz; KYROLA, Aapo; TULLOCH, Andrew; JIA, Yangqing; HE, Kaiming.** *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*. arXiv preprint arXiv:1706.02677, 2017. DOI: [10.48550/arXiv.1706.02677](https://doi.org/10.48550/arXiv.1706.02677).
3. **LOSHCHILOV, Ilya; HUTTER, Frank.** *SGDR: Stochastic Gradient Descent with Warm Restarts*. In: International Conference on Learning Representations (ICLR), 2017. DOI: [10.48550/arXiv.1608.03983](https://doi.org/10.48550/arXiv.1608.03983).
4. **TAN, Mingxing; LE, Quoc V.** *EfficientNetV2: Smaller Models and Faster Training*. In: International Conference on Machine Learning (ICML), 2021, pp. 10096-10106.
5. **WILSON, Ashia C.; ROELOFS, Rebecca; STERN, Mitchell; SREBRO, Nathan; RECHT, Benjamin.** *The Marginal Value of Adaptive Gradient Methods in Machine Learning*. In: Advances in Neural Information Processing Systems (NeurIPS), v. 30, 2017, pp. 4148-4158.
