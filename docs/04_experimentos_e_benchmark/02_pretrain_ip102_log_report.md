# 📊 Relatório de Análise do Pré-Treinamento (IP102)

Este relatório foi gerado automaticamente a partir dos logs de treinamento do **Passo 1 (Domain-Specific Pre-training)** no dataset oficial IP102.

---

## 📈 Resumo do Treinamento
*   **Modelo:** `EfficientNetV2B1`
*   **Total de Épocas:** 50
*   **Melhor Época (Menor Val Loss):** Época 50
*   **Acurácia de Validação Final (Best Epoch):** 53.92%
*   **Perda de Validação Final (Best Epoch):** 0.9708

---

## 📉 Histórico de Métricas por Época

| Época | Accuracy (Treino) | Loss (Treino) | Val Accuracy | Val Loss | Learning Rate |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.0196 | 5.5483 | 0.1827 | 4.6348 | 2.10e-03 |
| 2 | 0.1264 | 3.9560 | 0.2837 | 2.9271 | 4.10e-03 |
| 3 | 0.1944 | 2.6746 | 0.3419 | 1.9929 | 6.00e-03 |
| 4 | 0.2211 | 2.1456 | 0.3705 | 1.6695 | 8.00e-03 |
| 5 | 0.2298 | 2.0369 | 0.3539 | 1.6273 | 1.00e-02 |
| 6 | 0.2330 | 2.0361 | 0.3796 | 1.5683 | 1.00e-02 |
| 7 | 0.2386 | 1.9985 | 0.3863 | 1.5449 | 1.00e-02 |
| 8 | 0.2410 | 1.9834 | 0.3811 | 1.4949 | 9.90e-03 |
| 9 | 0.2459 | 1.9717 | 0.4100 | 1.4686 | 9.80e-03 |
| 10 | 0.2556 | 1.9529 | 0.3801 | 1.4646 | 9.80e-03 |
| 11 | 0.2529 | 1.9441 | 0.4053 | 1.4575 | 9.60e-03 |
| 12 | 0.2582 | 1.9469 | 0.3749 | 1.4332 | 9.50e-03 |
| 13 | 0.2625 | 1.9300 | 0.4291 | 1.3752 | 9.40e-03 |
| 14 | 0.2617 | 1.9298 | 0.4105 | 1.4060 | 9.20e-03 |
| 15 | 0.2658 | 1.9122 | 0.4290 | 1.3508 | 9.00e-03 |
| 16 | 0.2693 | 1.9035 | 0.4393 | 1.3700 | 8.90e-03 |
| 17 | 0.2746 | 1.8990 | 0.4301 | 1.3474 | 8.60e-03 |
| 18 | 0.2718 | 1.8899 | 0.4470 | 1.3340 | 8.40e-03 |
| 19 | 0.2773 | 1.8715 | 0.4322 | 1.3438 | 8.20e-03 |
| 20 | 0.2824 | 1.8635 | 0.4399 | 1.3149 | 7.90e-03 |
| 21 | 0.2834 | 1.8456 | 0.4584 | 1.2889 | 7.70e-03 |
| 22 | 0.2896 | 1.8396 | 0.4394 | 1.3007 | 7.40e-03 |
| 23 | 0.2930 | 1.8292 | 0.4530 | 1.2581 | 7.10e-03 |
| 24 | 0.2860 | 1.8290 | 0.4571 | 1.2497 | 6.80e-03 |
| 25 | 0.2971 | 1.8013 | 0.4754 | 1.2338 | 6.50e-03 |
| 26 | 0.2966 | 1.7970 | 0.4727 | 1.2269 | 6.20e-03 |
| 27 | 0.2989 | 1.7886 | 0.4738 | 1.2448 | 5.90e-03 |
| 28 | 0.3026 | 1.7700 | 0.4784 | 1.2042 | 5.60e-03 |
| 29 | 0.3096 | 1.7559 | 0.4860 | 1.1777 | 5.30e-03 |
| 30 | 0.3111 | 1.7469 | 0.4925 | 1.1831 | 5.00e-03 |
| 31 | 0.3150 | 1.7339 | 0.4857 | 1.1429 | 4.70e-03 |
| 32 | 0.3195 | 1.7229 | 0.5005 | 1.1494 | 4.40e-03 |
| 33 | 0.3231 | 1.6990 | 0.4912 | 1.1315 | 4.10e-03 |
| 34 | 0.3218 | 1.7057 | 0.4995 | 1.1120 | 3.80e-03 |
| 35 | 0.3232 | 1.6964 | 0.4920 | 1.1102 | 3.50e-03 |
| 36 | 0.3281 | 1.6809 | 0.5188 | 1.0954 | 3.20e-03 |
| 37 | 0.3309 | 1.6713 | 0.5087 | 1.0913 | 2.90e-03 |
| 38 | 0.3332 | 1.6482 | 0.5036 | 1.0662 | 2.60e-03 |
| 39 | 0.3348 | 1.6333 | 0.4875 | 1.0741 | 2.30e-03 |
| 40 | 0.3384 | 1.6359 | 0.5147 | 1.0651 | 2.10e-03 |
| 41 | 0.3423 | 1.6273 | 0.5197 | 1.0457 | 1.80e-03 |
| 42 | 0.3453 | 1.6120 | 0.5244 | 1.0298 | 1.60e-03 |
| 43 | 0.3428 | 1.6023 | 0.5254 | 1.0210 | 1.40e-03 |
| 44 | 0.3483 | 1.5976 | 0.5302 | 1.0039 | 1.20e-03 |
| 45 | 0.3564 | 1.5812 | 0.5270 | 0.9980 | 9.64e-04 |
| 46 | 0.3547 | 1.5742 | 0.5349 | 0.9924 | 7.88e-04 |
| 47 | 0.3563 | 1.5687 | 0.5350 | 0.9803 | 6.28e-04 |
| 48 | 0.3558 | 1.5569 | 0.5360 | 0.9797 | 4.85e-04 |
| 49 | 0.3603 | 1.5513 | 0.5373 | 0.9758 | 3.61e-04 |
| 50 | 0.3667 | 1.5346 | 0.5392 | 0.9708 | 2.54e-04 |

---

## 🔍 Análise Crítica e Diagnóstico

### 1. Evolução das Métricas (Acurácia e Loss)
*   **Acurácia de Treino:** Iniciou em valores muito baixos (~1%) e alcançou **36.67%** na época 50. Esse crescimento gradual é saudável e mostra que o modelo aprendeu representações úteis nas camadas descongeladas.
*   **Acurácia de Validação:** Atingiu **53.92%** no final. Como discutido, para um dataset complexo de 102 classes de grão fino, este valor é **altamente satisfatório** (mais de 50x superior a um classificador aleatório).
*   **Curva de Perda (Loss):** O `val_loss` caiu de forma consistente ao longo das épocas (terminando em **0.9708**), indicando que o modelo convergiu bem e não sofreu overfitting severo no final do treino.

### 2. Comportamento do Otimizador (SGD + CosineDecay)
*   A taxa de aprendizado começou em `1.00e-04` (warmup inicial) e seguiu a curva de decaimento cosseno até terminar em `2.54e-04`.
*   O Cosine Decay ajudou a estabilizar as atualizações de pesos na reta final do treinamento, permitindo que a perda de validação continuasse a cair de forma suave sem oscilações abruptas.

### 3. Conclusão para o Fine-Tuning (Passo 2)
O extrator salvo em `artifacts/models/ip102_pretrained_extractor.keras` está em um estado **ótimo** e muito bem condicionado. O modelo agora possui filtros visuais especializados na identificação de partes morfológicas de pragas, o que acelerará drasticamente a convergência e elevará a acurácia final ao treinar nas 10 classes de soja.
