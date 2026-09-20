# 📋 Roteiro de Treinamento do Modelo (IP102 ➔ Fine-Tuning Soja)

Este guia detalha o fluxo de treinamento em dois passos (Two-Step Training) para o classificador de pragas da soja utilizando transferência de aprendizado específica de domínio.

---

## 🏗️ Passo 1: Pré-Treinamento (Domain-Specific Pre-training)
*Já Concluído com Sucesso!* ✅

O modelo `EfficientNetV2B1` foi pré-treinado no dataset oficial **IP102** (102 classes de insetos). Isso ensina ao extrator de características do modelo a reconhecer a morfologia de insetos (patas, asas, antenas) antes de olhar para as fotos específicas de soja.

*   **Comando executado:**
    ```bash
    .venv/bin/python scripts/train_ip102.py --data_dir artifacts/data/ip102
    ```
*   **Resultados gerados:**
    *   `artifacts/models/ip102_pretrained_extractor.keras` (Pesos do extrator/backbone congelados).
    *   `artifacts/models/ip102_full_pretrained_model.keras` (Modelo completo com cabeça de 102 classes).

---

## ⚡ Passo 2: Fine-Tuning nas Pragas da Soja
Para executar o ajuste fino nas 10 classes de interesse, o pipeline carrega os pesos do extrator pré-treinado no IP102, acopla a cabeça de classificação adaptada e executa o fine-tuning:

*   **Comando para executar:**
    ```bash
    .venv/bin/python scripts/main.py --experiment efficientnetv2b1_finetune
    ```
*   **O que ele faz:**
    *   Carrega a divisão de treino/validação de `artifacts/data/final/DatasetPests-split/`.
    *   Descongela cirurgicamente as últimas **20 camadas** do backbone.
    *   Aplica **Categorical Focal Loss** com pesos suavizados e **CutMix**.
    *   Treina por 80 épocas usando o otimizador SGD com Cosine Decay.

---

## 📊 Passo 3: Avaliação Independente (INSECT12C)
Para avaliar o modelo em um dataset de teste de campo completamente independente (*INSECT12C*):

*   **Comando para executar:**
    ```bash
    .venv/bin/python scripts/evaluate_test.py
    ```
*   **O que ele faz:**
    *   Avalia o modelo nas imagens da pasta `artifacts/data/final/INSECT12C-test`.
    *   Calcula métricas como Acurácia, F1-Score Macro, Precision e Recall.
    *   Salva a Matriz de Confusão para análise de erros.

---

## ⚙️ Passo 4: Otimização de Hiperparâmetros (Opcional)
Para executar a busca bayesiana de hiperparâmetros (taxas de aprendizado, dropout, etc.) usando o Keras Tuner integrado ao MLflow:

*   **Comando para executar:**
    ```bash
    .venv/bin/python scripts/main_tuning.py --mode tune --experiment efficientnetv2b1_finetune --max-trials 30 --epochs-per-trial 50
    ```

---

## 🧠 Dúvida: "Mesmo com val_accuracy de ~54%, devo usar o modelo?"

**Sim, com certeza!** 

Pode parecer que uma acurácia de validação de **53.92%** é baixa se comparada a classificadores binários simples, mas no contexto do IP102 ela é **excelente**:

1.  **Complexidade das Classes (102 categorias):** Em um dataset com 102 classes, um chute aleatório acertaria apenas **0,98%** das vezes. Acertar **53.92%** de forma "Top-1" significa que o modelo aprendeu recursos morfológicos extremamente fortes para distinguir uma enorme variedade de pragas (muitas delas visualmente quase idênticas).
2.  **Objetivo do Pré-Treino:** O objetivo aqui não era obter um classificador perfeito de 102 classes, mas sim **ensinar conceitos gerais de insetos** à rede. Esse "extrator de características" pré-treinado agora é infinitamente mais especializado do que o ImageNet padrão (que é treinado em gatos, carros e aviões).
3.  **Transferência para a Soja (10 classes):** Ao fazer o fine-tuning para apenas 10 classes no **Passo 2**, o espaço de busca de decisão diminui drasticamente, e o classificador conseguirá mapear essas características refinadas com muito mais facilidade. A acurácia final do fine-tuning na soja deve alcançar níveis muito maiores (comumente acima de 80-90%).
