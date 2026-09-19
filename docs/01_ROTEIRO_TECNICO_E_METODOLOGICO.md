# 🧭 Roteiro Técnico e Metodológico de Desenvolvimento
## Sistema de Classificação de Pragas da Soja & Borda

> **Propósito deste documento:**  
> Centralizar a especificação técnica, os parâmetros experimentais, o pipeline de dados e os resultados consolidados do sistema de classificação de pragas agrícolas em dispositivos de borda. Serve como **guia oficial de arquitetura e reprodutibilidade científica** do projeto.

---

## 📌 1. Escopo Técnico da Pesquisa: Borda vs. Pipeline Experimental Completo

| Dimensão Técnica | Módulo de Aplicação em Borda (Mobile Edge) | Pipeline de Pesquisa e Benchmarking Completo |
| :--- | :--- | :--- |
| **Foco Central** | Aplicação móvel offline (PestClassifier) e inferência em tempo real com o modelo ótimo global. | Avaliação empírica: mineração de dados, benchmark de 11 configurações (9 leves + 2 referência), ablações e profiling multi-tier. |
| **Arquitetura Alvo** | EfficientNetV2-B1 otimizada e quantizada em TFLite FP16 (~16.5 MB). | Estudo comparativo sistemático entre MobileNetV3 (Small/Large), EfficientNetV2 (B0/B1), MobileViT e ConvNeXt-Tiny. |
| **Estratégia de Domínio** | Pré-treinamento entomológico no IP102 seguido de fine-tuning especializado. | Confronto sistemático: ImageNet (baseline genérico) vs. IP102 (domínio entomológico) vs. Treinamento From Scratch. |
| **Pipeline de Dados** | Captura local, pré-processamento anatômico com margem foliar de +20% e inferência sem nuvem. | Ingestão multi-fonte (DatasetPests, iNaturalist, IP102, INSECT12C) e particionamento agrupado por imagem-mãe (anti-leakage). |
| **Métricas Críticas** | Latência de inferência (ms) e consumo de recursos por profiling tier. | Macro-F1, Precision, Recall, Acurácia estratificada por resolução espacial (>60px, >100px) e fronteira de Pareto multi-objetivo. |

---

## 🔬 2. Dados e Pré-Processamento

### 2.1. Fontes de Imagens
1. **DatasetPests (Anotação Manual no Label-Studio):** Base primária contendo fotografias de campo em condições reais brasileiras obtidas a partir da plataforma iNaturalist, com anotações manuais de caixas delimitadoras (*bounding boxes*) realizadas pela equipe do laboratório e exportadas via Label-Studio (CSV).
2. **Mineração Automatizada iNaturalist API:** Mineração complementar automatizada via API do iNaturalist, com filtros de fase biológica (larvas/ninfas via `term_id=1`, `term_value_id=6`) para balancear espécies desfolhadoras e sugadoras, recortadas autonomamente pelo detector YOLOv8.
3. **IP102 Dataset:** 102 classes de pragas agrícolas gerais (XML Pascal VOC) utilizado exclusivamente para o **pré-treinamento de domínio entomológico**.
4. **INSECT12C Dataset:** Base externa independente (12 classes originais mapeadas para as 10 classes do projeto), utilizada exclusivamente para **avaliação zero-shot de robustez cruzada**.

### 2.2. Taxonomia Oficial (10 Classes)
| Índice | Nome Científico | Nome Comum | Danos Principais |
| :---: | :--- | :--- | :--- |
| `0` | *Anticarsia gemmatalis* | Lagarta-da-soja | Desfolha severa em estágios vegetativo e reprodutivo |
| `1` | Coccinellidae | Joaninha | Inseto benéfico / predador (controle biológico) |
| `2` | *Diabrotica speciosa* | Vaquinha-verde-amarela | Perfuração foliar e danos radiculares pelas larvas |
| `3` | *Edessa meditabunda* | Percevejo-asa-verde | Sucção de seiva e danos a vagens e grãos |
| `4` | *Euschistus heros* | Percevejo-marrom | Principal praga sugadora de grãos e vagens |
| `5` | Gastropoda | Lesmas e Caracóis | Raspagem foliar e destruição de plântulas |
| `6` | *Lagria villosa* | Besouro-da-couve / Bicho-capixaba | Desfolha e ataque a flores e brotações novas |
| `7` | *Nezara viridula* | Percevejo-verde | Sucção de vagens provocando retenção foliar |
| `8` | *Radacridium schistocercoides*| Gafanhoto | Desfolha voraz e corte de hastes |
| `9` | *Spodoptera albula* | Lagarta-das-folhas | Desfolhamento agressivo e ataque a vagens |

### 2.3. Decisões de Pré-Processamento
* **Recorte Anatômico com Margem Foliar (`--margin 0.20`):** Caixas delimitadoras muito justas (*tight crops*) deformavam o inseto ao serem interpoladas para $224\times224$ ou $240\times240$. A expansão de 20% preservou a morfologia do inseto e incluiu o contexto foliar imediato (tipo de perfuração/dano).
* **Auto-Crop via YOLOv8:** Treinou-se um detector `YOLOv8n` de classe única (`pest`) nas anotações manuais do DatasetPests para localizar e recortar autonomamente as pragas das imagens brutas mineradas via API do iNaturalist (`auto_crop_inaturalist.py`).
* **Group-based Stratified Split (Sem *Data Leakage*):**
  - **Divisão Real:** **90% Treino e 10% Validação** (`TRAIN_RATIO = 0.9`, `VALIDATION_RATIO = 0.1`, semente fixa `seed=42`).
  - **Critério de Agrupamento:** Todas as caixas delimitadoras recortadas da mesma imagem-mãe (folha original) foram forçadas a pertencer integralmente ou ao treino ou à validação (`split_dataset_by_group.py`). Isso impediu que o modelo memorizasse o fundo da folha.
  - **Conjunto de Teste:** O teste intra-domínio oficial do benchmark é a partição de validação isolada (1.123 amostras), e o teste inter-domínio de generalização externa zero-shot é a base independente INSECT12C (2.618 amostras).

---

## 🧠 3. Modelagem e Benchmark das Arquiteturas Leves

O benchmark avaliou **9 configurações principais** (e ConvNeXt-Tiny como modelo complementar) divididas em três estratégias:
1. **Baseline Tradicional:** Pré-treino ImageNet $\to$ Fine-tuning no DatasetPests.
2. **Proposta da Pesquisa:** ImageNet $\to$ Pré-treino entomológico no IP102 $\to$ Fine-tuning no DatasetPests.
3. **From Scratch:** Inicialização aleatória $\to$ Treino direto no DatasetPests (MobileViT).

### 3.1. Resolução e Especificações Técnicas por Modelo

| Modelo | Pré-Treinamento | Resolução Alvo | Parâmetros Nominais (Ref.) | Parâmetros Reais em Disco (10 classes + Top-K) | Papel no Benchmark |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **EfficientNetV2-B1** | IP102 | **$240 \times 240$** | ~8.1M | **~7.1M** (7.096.894) | **Modelo Selecionado para Borda** (Líder em robustez zero-shot) |
| **EfficientNetV2-B1** | ImageNet | **$240 \times 240$** | ~8.1M | ~7.1M | Comparativo direto de ablação |
| **EfficientNetV2-B0** | IP102 | **$224 \times 224$** | ~7.1M | **~6.1M** (6.085.082) | **Líder Intra-Domínio** (Maior F1 no DatasetPests: 0.956) |
| **EfficientNetV2-B0** | ImageNet | **$224 \times 224$** | ~7.1M | ~6.1M | Comparativo direto de ablação |
| **MobileNetV3-Large** | IP102 | **$224 \times 224$** | ~5.4M | **~3.1M** (3.121.162) | Alto balanço acurácia/velocidade |
| **MobileNetV3-Large** | ImageNet | **$224 \times 224$** | ~5.4M | ~3.1M | Comparativo direto de ablação |
| **MobileNetV3-Small** | IP102 | **$224 \times 224$** | ~2.9M | **~1.0M** (1.014.778) | Ultra-leve (Maior salto relativo com IP102: +13.4% em teste externo) |
| **MobileNetV3-Small** | ImageNet | **$224 \times 224$** | ~2.9M | ~1.0M | Comparativo direto de ablação |
| **MobileViT (Custom)**| From Scratch | **$256 \times 256$** | ~5.6M | ~5.6M | Estudo de caso: ausência de viés indutivo em datasets pequenos |
| *ConvNeXt-Tiny* | IP102 | **$224 \times 224$** | ~28.6M | ~28.6M | Modelo de alta capacidade (analisado na dissertação) |
| *ConvNeXt-Tiny* | ImageNet | **$224 \times 224$** | ~28.6M | ~28.6M | Baseline de alta capacidade (analisado na dissertação) |

> 📌 **Nota sobre Contagem de Parâmetros:** Os valores nominais referem-se à arquitetura padrão com cabeça de 1.000 classes do ImageNet. Ao substituir a cabeça padrão pelo bloco enxuto `TopKGlobalAveragePooling2D` + `Dense(10)`, os modelos finais implantados em disco tornam-se consideravelmente mais leves (ex.: MobileNetV3-Large cai de 5.4M para 3.1M; B1 cai de 8.1M para 7.1M), o que reforça ainda mais sua adequação para dispositivos móveis.
>
> 📌 **Especificação de Resoluções:** O modelo implantado em produção (**EfficientNetV2-B1**) opera nativamente a **$240 \times 240$**, enquanto os backbones MobileNetV3, ConvNeXt e EfficientNetV2-B0 operam a $224 \times 224$, e o MobileViT a $256 \times 256$.

### 3.2. Status das Camadas e Blocos Customizados (Inspeção no Grafo Salvo)
Para manter o modelo leve, enxuto e reprodutível, a arquitetura final adotada e salva em disco (`artifacts/models/mobile/`):
* **`CBAM` (Attention Block):** **DESATIVADO** (`use_cbam: False`). Foi testado na fase exploratória, mas não foi necessário para atingir os resultados campeões.
* **`ResidualSRCNNBlock` (Super-Resolução):** **DESATIVADO** (`use_sr_block: False`).
* **`Squeeze-and-Excitation` e `Compression Blocks` extras:** **DESATIVADOS** (`use_se_block: False`, `use_compression_blocks: False`).
* **`TopKGlobalAveragePooling2D`:** **ATIVO**. É a única camada customizada que atua sobre o mapa de características do backbone, selecionando as $k\%$ ativações espaciais mais fortes da praga antes da cabeça densa.
* **Reprodutibilidade:** Essa configuração enxuta assegura simplicidade de implantação, mantendo as backbones limpas e de fácil reprodução.

---

## ⚙️ 4. Treinamento, Otimização e Controle Experimental

### 4.1. Rigor Experimental (*Ceteris Paribus*)
Para que a comparação ImageNet vs. IP102 fosse irrefutável perante revisores e banca examinadora:
* **Mesmo particionamento** e mesma semente aleatória (`seed=42`).
* **Mesmo pipeline de Data Augmentation:** Inspecionado diretamente no grafo salvo dos modelos, composto por:
  1. *Gaussian Noise* (ruído de sensor móvel);
  2. *RandomFlip* horizontal e vertical (captura em ângulos livres e face inferior da folha);
  3. *RandomTranslation* (15% com modo `reflect`): originalmente concebido para evitar que pragas na borda do recorte ficassem cortadas, mantido no dataset de 10 classes para conferir invariância à posição espacial de pequenos insetos periféricos;
  4. *RandomZoom* (variação de distância da câmera);
  5. *RandomBrightness* e *RandomContrast* (variações de luminosidade do sol e sombra no campo);
  6. *RandomRotation* (rotação natural);
  7. *CutMix* (regularização de lotes aplicada via `tf.data`).
* **Mesma métrica de parada antecipada:** *EarlyStopping* monitorando `val_loss` com paciência de 15 épocas.

### 4.2. Função de Perda: *Categorical Focal Crossentropy*
* **Problema:** A ponderação simples de classes ($w_c \propto 1/N_c$) na Cross-Entropy padrão gera excesso de penalidade em classes raras, fazendo a rede prever a classe rara como rota de fuga, destruindo a **Precision**.
* **Solução:** Adoção da **Focal Loss** com $\gamma = 1.5$ e vetor $\alpha$ de pesos normalizados:
  $$\mathcal{L}_{FL} = - \alpha_t (1 - p_t)^\gamma \log(p_t)$$
  O fator de modulação $(1 - p_t)^\gamma$ anula gradientes de exemplos fáceis e repetitivos ($p_t \to 1$), concentrando o aprendizado nos exemplos limítrofes e difíceis.

### 4.3. Otimização Bayesiana Independente (Keras Tuner)
* **Justificativa:** A superfície de perda (*loss landscape*) de uma rede inicializada no ImageNet difere radicalmente de uma rede pré-condicionada no IP102. Forçar hiperparâmetros fixos introduziria viés de seleção.
* **Protocolo:** Executaram-se **30 tentativas (*trials*)** de Otimização Bayesiana por modelo (`scripts/main_tuning.py`):
  - *Learning Rate ($lr$):* Busca logarítmica entre $10^{-6}$ e $10^{-3}$.
  - *Dropout:* $0.1$ a $0.6$ (passo $0.1$).
  - *Camadas Descongeladas (*Unfreeze Last N*):* 16 a 128 camadas (com Batch Normalization mantido congelado em modo inferência).
  - *Regularização $L_2$:* $10^{-5}$ a $5 \times 10^{-2}$.
  - *Top-K Pooling Ratio:* 5% a 25%.

---

## 📱 5. Implantação em Borda e Desempenho Mobile

### 5.1. Exportação TFLite
* **Quantização:** Pós-treinamento com quantização flutuante de 16 bits (**FP16**), via `converter.target_spec.supported_types = [tf.float16]`.
* **Benefícios:**
  - Redução de ~50% no tamanho em disco (EfficientNetV2-B1 caiu de ~33 MB para ~16.5 MB).
  - Preservação de 99.9% da precisão numérica em comparação com FP32.
  - Compatibilidade com aceleradores neurais (GPU / NPU móveis).

### 5.2. Requisitos Técnicos de Execução Android
* **Nível Mínimo de API:** **Android 5.0 (API nível 21)** ou recomendada **Android 7.0 (API nível 24)** para total compatibilidade com o runtime C++/Java do TensorFlow Lite.

### 5.3. Latência e Categorias de Hardware (*Profiling Tiers*)
As latências do benchmark consolidado (`scripts/generate_benchmark_data.py`) foram calculadas a partir de medição empírica direta e escalonamento por profiling:
1. **High-End Tier (GPU de referência):** ~59 a 63 ms por inferência (permite classificação contínua em tempo real).
2. **Mid-End Tier (Profiling Scale $2.5\times$):** ~148 a 158 ms por inferência (equivalente a processadores intermediários Snapdragon 7-series / Dimensity).
3. **Low-End Tier (Profiling Scale $6.5\times$):** ~386 a 410 ms por inferência (viável para SoCs de entrada e dispositivos legados no campo).

> 📌 **Metodologia de Profiling:** As categorias de hardware representam *Profiling Hardware Tiers* obtidos a partir de medições de referência em hardware acelerado, escalonadas pelos fatores médios de capacidade computacional dos processadores de borda.

---

## 📊 6. Tabela Oficial de Desempenho Consolidado

Dados extraídos com rigor dos arquivos [`performance_summary.csv`](../graphics/output/performance_summary.csv) e [`resolution_summary_insect12c.csv`](../graphics/output/resolution_summary_insect12c.csv):

| Modelo | Pré-Treinamento | DatasetPests (Val) <br> **Macro-F1** | INSECT12C (All) <br> **Macro-F1** | INSECT12C (>60px) <br> **Macro-F1** | INSECT12C (>100px) <br> **Macro-F1** | Pareto Frontier (Borda) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileViT** | From Scratch | 0.426 | 0.260 | 0.287 | 0.258 | Não |
| **MobileNetV3-Small** | ImageNet | 0.857 | 0.506 | 0.598 | 0.600 | Sim (Velocidade) |
| **MobileNetV3-Small** | IP102 | 0.898 | 0.596 | 0.680 | 0.704 | Não |
| **MobileNetV3-Large** | ImageNet | 0.890 | 0.624 | 0.718 | 0.745 | Sim |
| **MobileNetV3-Large** | IP102 | **0.943** | 0.644 | 0.746 | 0.803 | Sim |
| **EfficientNetV2-B1** | ImageNet | 0.871 | 0.608 | 0.717 | 0.766 | Não |
| **EfficientNetV2-B1** (Vencedor) | **IP102** | **0.934** | <span style="color:green">**0.681**</span> | <span style="color:green">**0.793**</span> | <span style="color:green">**0.822**</span> | **Sim (Ótimo Global)** |
| **EfficientNetV2-B0** | ImageNet | 0.892 | 0.630 | 0.723 | 0.763 | Não |
| **EfficientNetV2-B0** | IP102 | <span style="color:blue">**0.956**</span> | 0.675 | 0.769 | 0.804 | Não |
| *Média Geral* | — | **0.852** | **0.580** | **0.670** | **0.696** | — |

### Principais Conclusões Científicas:
1. **Domínio Entomológico Vence em 100% dos Casos:** O pré-treino IP102 superou o ImageNet em todas as 12 comparações diretas. O MobileNetV3-Small (IP102) (0.898) superou inclusive o MobileNetV3-Large (ImageNet) (0.890), provando que representações de domínio importam mais que capacidade de parâmetros.
2. **Por que a EfficientNetV2-B1 foi a escolhida:** Embora a B0 (IP102) tenha vencido no intra-domínio (0.956 vs 0.934), a **B1 (IP102) foi superior em todos os cenários zero-shot reais do INSECT12C** (0.681 vs 0.675 geral, 0.793 vs 0.769 em recortes médios, e 0.822 vs 0.804 em recortes grandes), garantindo maior resiliência a variações de iluminação e campo.
3. **Fracasso do MobileViT Sem Pré-treino:** Confirmou empiricamente que arquiteturas baseadas em atenção pura sofrem com a falta de viés indutivo e colapsam quando treinadas do zero em bases agronômicas restritas (Macro-F1 de apenas 0.260 no INSECT12C).

---

## 🛠️ 7. Mapa de Scripts de Execução e Reprodutibilidade

| Etapa do Trabalho | Script Principal | Parâmetros / Observações |
| :--- | :--- | :--- |
| **Ingestão Automática** | `src/cnnClassifier/pipeline/stage_01_data_ingestion.py` | Baixa do Google Drive e extrai em `artifacts/data/` |
| **Recorte DatasetPests** | `scripts/data_preparation/prepare_dataset_pests.py` | `--margin 0.20` (expansão de contexto foliar) |
| **Divisão Sem Vazamento** | `scripts/data_preparation/split_dataset_by_group.py` | Agrupamento por ID da imagem-mãe (90/10) |
| **Pré-Treino IP102** | `scripts/train_ip102.py` | `--model efficientnetv2b1 --experiment ip102_pretrain` |
| **Tuning Bayesiano** | `scripts/main_tuning.py` | `--experiment efficientnetv2b1_finetune --max-trials 30` |
| **Geração de Métricas** | `scripts/generate_benchmark_data.py` | Avalia modelos `.keras` nos datasets e gera CSVs |
| **Gráficos e Pareto** | `graphics/benchmark_graficos_avaliacao_predicao.py` | Gera os PNGs/PDFs da curva de Pareto e F1-Heatmaps |
