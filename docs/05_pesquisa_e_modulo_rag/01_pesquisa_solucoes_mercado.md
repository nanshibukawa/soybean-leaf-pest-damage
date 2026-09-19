# Guia de Pesquisa Científica: Classificação de Insetos em Baixa Resolução

Este documento serve como um guia metodológico para buscar artigos científicos e soluções na literatura para o problema identificado no nosso pipeline: **a queda de desempenho do classificador ao lidar com recortes (crops) pequenos/de baixa resolução (ex: 32x32 a 60x60 pixels)**, especialmente na classe majoritária *Euschistus heros*.

---

## 🎯 1. Resumo do Desafio Científico

O problema enfrentado não é um bug de código, mas sim um desafio clássico de **Visão Computacional Aplicada**:
1.  **Distorção por Interpolação**: Crops pequenos (< 60x60 px) ampliados para a resolução de entrada do modelo (240x240 px) perdem padrões de alta frequência (texturas, bordas nítidas), resultando em ruído visual.
2.  **Classificação de Grão Fino (Fine-Grained)**: Insetos na mesma família (ex: percevejos) compartilham morfologias muito parecidas, exigindo detalhes de alta resolução para serem distinguidos.
3.  **Desbalanceamento de Suporte**: A classe *Euschistus* domina o conjunto de testes, fazendo com que sua baixa revocação em resoluções pequenas derrube drasticamente a acurácia global (weighted), enquanto a revocação macro (macro avg) permanece alta (~83%).

---

## 🔍 2. Termos de Pesquisa (Keywords para Google Scholar / IEEE Xplore)

Como a literatura científica relevante é majoritariamente em inglês, utilize as seguintes combinações de palavras-chave:

### Grupo A: Resolução e Objetos Pequenos
*   `"low resolution image classification" AND "deep learning"`
*   `"small object classification" AND "convolutional neural networks"`
*   `"resolution-aware image classification"`
*   `"sub-pixel classification" AND "fine-grained"`

### Grupo B: Soluções com Super-Resolução (Pre-processing)
*   `"super-resolution" AND "image classification" AND "joint training"`
*   `"super-resolution pre-processing for image classification"`
*   `"SRGAN" OR "ESRGAN" AND "object classification pipeline"`

### Grupo C: Aplicação Agrícola e Pragas
*   `"fine-grained insect classification" AND "low resolution"`
*   `"agricultural pest recognition" AND "small scale dataset"`
*   `"IP102 dataset" AND "small object detection classification"`

---

## 💡 3. Direções de Solução Encontradas na Literatura

Ao pesquisar, você encontrará três grandes vertentes de soluções. Abaixo estão descritas as principais técnicas para você procurar nos artigos:

### Solução 1: Preprocessamento com Super-Resolução (Super-Resolution - SR)
*   **Como funciona**: Em vez de usar interpolação bilinear/bicúbica para ampliar o crop de 32x32 para 240x240, usa-se uma rede neural de super-resolução leve (ex: **SRCNN**, **ESRGAN** ou **Real-ESRGAN**) para reconstruir as texturas e detalhes ausentes do inseto antes de enviá-lo ao classificador.
*   **O que buscar**: Artigos que tratem de *"Super-resolution as preprocessing for classification"*. Muitos artigos provam que treinar a rede de SR de forma conjunta (end-to-end) com o classificador aumenta a acurácia em até 15% em objetos pequenos.

### Solução 2: Fusão de Features Multi-Escala (Multi-Scale Feature Fusion)
*   **Como funciona**: Redes neurais profundas perdem informações espaciais de objetos pequenos nas camadas finais. Essa técnica combina mapas de características das camadas iniciais (que possuem alta resolução espacial) com as camadas finais antes da classificação.
*   **O que buscar**: *"Feature Pyramid Networks (FPN) for classification"* ou *"Multi-scale feature fusion for fine-grained classification"*.

### Solução 3: Mecanismos de Atenção Localizada (Local Attention Mechanisms)
*   **Como funciona**: Adiciona blocos de atenção (como **CBAM** - *Convolutional Block Attention Module* ou **Coordinate Attention**) para forçar o modelo a focar estritamente nas regiões morfológicas chave do inseto, ignorando as bordas borradas geradas pelo redimensionamento.
*   **O que buscar**: *"Attention mechanisms for small object classification"*.

### Solução 4: Aprendizado Multi-Tarefa (Multi-Task Learning)
*   **Como funciona**: Treinar o modelo para prever simultaneamente a **Espécie** (ex: *Euschistus*) e o **Estágio de Vida** (ex: *Ninfa* vs *Adulto*). A literatura mostra que prever o estágio ajuda o modelo a aprender representações internas mais robustas sobre o tamanho e o formato do inseto.
*   **O que buscar**: *"Multi-task learning for insect classification"*.

---

## 🗂️ 4. Bases de Dados Recomendadas para Busca
1.  **Google Scholar (Acadêmico)**: Excelente ponto de partida geral.
2.  **IEEE Xplore**: Focado em engenharia e tecnologia (ótimo para arquiteturas de redes).
3.  **ScienceDirect / Elsevier**: Artigos de alta qualidade em agronomia computacional.
4.  **ArXiv / Papers With Code**: Para encontrar as implementações em Python/PyTorch/TensorFlow mais recentes do estado da arte.
