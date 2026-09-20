# 📑 Fundamentação da Categorical Focal Loss e Particionamento Anti-Vazamento

Este documento fundamenta a escolha da **Categorical Focal Crossentropy** em substituição ao sobreaumento artificial (*oversampling*), detalhando a dinâmica de gradientes, a calibração para o catálogo de 10 classes e a estratégia de particionamento estratificado sem vazamento de dados (*Group-based Split*), incorporando as duas réguas de avaliação do projeto: **DatasetPests** (treino e validação) e **INSECT12C** (teste independente).

---

## 1. Focal Loss vs. Sobreaumento de Dados (Oversampling)

### Limitações Estruturais da Geração de Dados Sintéticos e Sobreaumento
Tentar balancear classes minoritárias no espaço de entrada gerando dados sintéticos adicionais ou duplicando imagens geometricamente apresenta restrições metodológicas críticas:
1. **Redundância de Informação e Colapso de Variabilidade:** A repetição ou interpolação geométrica não adiciona informação estatística nova à distribuição da classe. Ela apenas replica padrões já presentes nas amostras existentes.
2. **Sensibilidade a Texturas Artificiais:** Redes neurais convolucionais e Vision Transformers são altamente sensíveis a frequências espaciais e texturas locais, correndo o risco de aprender artefatos de replicação em vez de atributos morfológicos entomológicos reais.
3. **Diluição do Gradiente Útil:** Um volume desproporcional de amostras artificiais fáceis satura o treinamento, consumindo a maior parte do orçamento de otimização em exemplos redundantes.

### A Superioridade Estratégica da Focal Loss
Em vez de forçar um equilíbrio artificial no conjunto de dados, a **Focal Loss** atua no espaço de otimização da função de perda, reequilibrando a contribuição de cada amostra dinamicamente:
* **Foco na Dificuldade Amostral:** Amostras de classificação fácil geram perdas residuais próximas de zero, permitindo que os gradientes priorizem os exemplos de fronteira morfológica mais desafiadores.
* **Eficiência Computacional:** Preserva a distribuição real de captura de campo sem sobrecarregar o pipeline com milhares de cópias redundantes.

---

## 2. Dinâmica de Gradientes: Cross Entropy Ponderada vs. Focal Loss

### A Limitação de Cross Entropy com Pesos Simples ($w_c \propto 1/N_c$)
A aplicação de pesos inversos lineares na Cross Entropy padrão tende a penalizar desproporcionalmente a precisão das classes minoritárias:

$$\mathcal{L}_{WCE} = - w_c \log(p_c)$$

Onde $w_c \propto 1/N_c$. O gradiente em relação ao logit pré-ativação $z_j$ da classe $j$ é:
$$\frac{\partial \mathcal{L}_{WCE}}{\partial z_j} = w_c (p_j - y_j)$$

Se o modelo classificar incorretamente uma amostra minoritária ($c=\text{minoria}$), a penalidade por falso negativo é multiplicada por um peso elevado ($w_{\text{minoria}}$). Contudo, a penalidade por prever erroneamente a classe minoritária em uma amostra majoritária é atenuada por $w_{\text{maioria}}$, que é muito pequeno. 

> [!IMPORTANT]
> **O Efeito Colateral:** O modelo é incentivado a emitir previsões para a classe minoritária diante de qualquer incerteza probabilística, elevando os falsos positivos e degradando severamente a **Precision** da classe.

---

### A Modulação Adaptativa da Categorical Focal Loss
A Focal Loss mitiga esse comportamento ao introduzir o fator de modulação $(1 - p_t)^\gamma$:
$$\mathcal{L}_{FL} = - \alpha_t (1 - p_t)^\gamma \log(p_t)$$

Onde $p_t$ é a probabilidade atribuída pelo modelo à classe correta. O gradiente da perda em relação ao logit correto torna-se:
$$\frac{\partial \mathcal{L}_{FL}}{\partial z} \approx (1 - p_t)^\gamma \left( \gamma p_t \log(p_t) - (1 - p_t) \right)$$

* **Amostras Fáceis ($p_t \to 1$):** O fator $(1 - p_t)^\gamma \to 0$. O gradiente é fortemente atenuado, impedindo que classes frequentes saturem o aprendizado.
* **Amostras Difíceis ($p_t \to 0$):** O fator $(1 - p_t)^\gamma \to 1$. O gradiente retém sua magnitude nominal, direcionando a atualização dos filtros convolucionais para padrões de grão fino.

Essa ponderação é **adaptativa por instância** (*sample-wise*), tratando cada espécime pela sua complexidade intrínseca.

---

## 3. Arquitetura de Particionamento e Mitigação de Vazamento (Anti-Leakage)

A esteira de dados foi desenhada para assegurar controle experimental rigoroso em dois níveis complementares:

```
[DatasetPests] ──> [Recortes com Margem +20%] ──> [split_dataset_by_group.py] ──> Partição Sem Vazamento
                                                                                   ├── Treino (90% das Imagens-Mãe)
                                                                                   └── Validação (10% das Imagens-Mãe) 🟢 LIMPO

[INSECT12C] ───> [Recortes com Bounding Boxes] ──────────────────────────────────> Teste Independente Zero-Shot 🟢 LIMPO
```

### 1. Independência do Teste Externo (INSECT12C)
O conjunto de teste é baseado no benchmark **INSECT12C**, uma base externa independente de fotografias agrícolas. Como não há compartilhamento de fotos com o `DatasetPests`, os resultados reportados avaliam a capacidade real de generalização da arquitetura em cenários não vistos durante o treinamento.

### 2. Particionamento Agrupado por Imagem-Mãe (Group-based Split)
Múltiplos recortes de insetos podem provir de uma mesma folha fotografada em lavoura. Um particionamento puramente aleatório poderia alocar um recorte da folha no treino e outro recorte da mesma folha na validação, permitindo que a rede memorize a textura do fundo foliar em vez da morfologia da praga.

* **Solução Adotada:** O script `split_dataset_by_group.py` agrupa os recortes pelo identificador único da foto original (imagem-mãe). Garante-se que 100% dos recortes derivados de uma foto pertençam estritamente ao conjunto de **Treino** ou de **Validação**.

### 3. Preservação da Distribuição Natural de Amostras
Ao adotar a `CategoricalFocalCrossentropy`, elimina-se a necessidade de duplicar amostras para balancear classes minoritárias. O modelo treina diretamente sobre a distribuição natural de recortes, prevenindo sobreajuste (*overfitting*) em cópias artificiais.

---

## 4. Calibração de Hiperparâmetros no Cenário de 10 Classes

Em classificação multiclasse de 10 classes, os parâmetros $\gamma$ e $\alpha$ requerem calibração cuidadosa:

### O Comportamento do Parâmetro `alpha`
No `tf.keras.losses.CategoricalFocalCrossentropy`:
* **Escalar (ex: `0.25`):** Atua como multiplicador global homogêneo sobre todas as classes, reduzindo a magnitude geral dos gradientes.
* **Vetor (tamanho 10):** Atribui pesos estáticos específicos por categoria taxonômica.

### Estratégia de Suavização via Número Efetivo de Amostras
Para evitar instabilidade numérica decorrente da combinação de $\gamma$ alto com pesos inversos lineares ($1/N_c$), adota-se a formulação de **Class-Balanced Loss** fundamentada no **Número Efetivo de Amostras** (Cui et al., CVPR 2019):

$$w_y = \frac{1 - \beta}{1 - \beta^{n_y}}$$

Onde $n_y$ é a contagem de amostras da classe $y$, e $\beta = 0.999$ atua como fator de amortecimento, impedindo que classes minoritárias gerem gradientes desproporcionais nas primeiras épocas de convergência.

---

## 5. Dinâmica de Convergência e Monitoramento Experimental

### A. Escala da Perda Focal
O valor numérico absoluto da perda Focal inicial é tipicamente inferior ao de uma Cross Entropy convencional. Para 10 classes, uma Cross Entropy inicial inicia próxima de $-\ln(0.1) \approx 2.3$, enquanto com $\gamma=1.5$, a Focal Loss inicial situa-se na faixa de $0.15$ a $0.40$.

### B. Curvas de Aprendizado (Treino vs. Validação)
Com o particionamento agrupado por imagem-mãe e a eliminação de vazamento, a curva de perda de validação descreve uma trajetória suave e representativa da generalização, estabilizando de forma consistente sem indicar memorização espúria.

### C. Equilíbrio entre Precisão e Cobertura (Precision vs. Recall)
O balanceamento adequado de $\gamma$ e do vetor $\alpha$ assegura que as classes minoritárias obtenham crescimento contínuo de Recall sem provocar a proliferação de falsos positivos, mantendo elevado o Macro F1-Score do benchmark.
