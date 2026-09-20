# 🔍 Análise Qualitativa de Ruído Amostral, Polimorfismo e Falsos Positivos da Mineração

Este documento formaliza o diagnóstico qualitativo dos recortes (*crops*) gerados pelo pipeline de mineração automatizada via API do iNaturalist e localização pelo detector YOLOv8. Serve como registro técnico e fundamentação empírica para a compreensão dos modos de falha e da discrepância de generalização observada no benchmark externo (*INSECT12C*).

---

## 🔬 1. Contexto e Motivação da Mineração Automatizada

Para contornar o desbalanceamento severo de classes em pragas desfolhadoras e sugadoras na cultura da soja, desenvolveu-se uma esteira automatizada composta por:
1. Ingestão de observações via API do iNaturalist com filtros taxonômicos e de fase de vida (`term_id=1`, `term_value_id=6` para Larva);
2. Detecção e recorte autônomo de caixas delimitadoras via modelo `YOLOv8n` treinado na classe genérica `pest`.

Embora a esteira tenha viabilizado o aumento do volume de dados para 11.468 recortes no total, a auditoria visual pós-treinamento revelou fontes críticas de variabilidade intra-classe que oferecem fortes indícios para compreender a discrepância de desempenho entre a validação interna (*DatasetPests*) e a avaliação de robustez externa (*INSECT12C*).

---

## 📸 2. Diagnóstico Empírico das Amostras Auditadas (*Spodoptera albula*)

A inspeção detalhada da classe `spodoptera_albula` (localizada em `artifacts/data/processed/DatasetPests-cropped/spodoptera_albula/`) identificou três perfis morfológicos distintos presentes no dataset de treino:

### A. Amostra Canônica Ideal (Fase Larval em Tecido Vegetal)
<img src="images/inat_obs_244363539_crop_1.jpg" alt="Amostra Canônica Larval" width="220" />

* **Arquivo:** [`inat_obs_244363539_crop_1.jpg`](images/inat_obs_244363539_crop_1.jpg)
* **Descrição Visual:** Lagarta íntegra com padrão listrado longitudinal nítido, cápsula cefálica preservada, em postura ativa de alimentação sobre tecido vegetal.
* **Hipótese de Impacto no Modelo:** Fornece os gradientes convolucionais desejados para reconhecimento de danos de desfolha no campo.

### B. Polimorfismo de Fase de Vida (Espécimes Adultos / Mariposas)
<p align="left">
  <img src="images/inat_obs_42444465_crop_0.jpg" alt="Mariposa em repouso" width="180" />
  <img src="images/inat_obs_244826083_crop_0.jpg" alt="Mariposa adulta em pano" width="180" />
  <img src="images/inat_obs_247051453_crop_0.jpg" alt="Mariposa noturna" width="180" />
</p>

* **Arquivos Auditados:**
  * [`inat_obs_42444465_crop_0.jpg`](images/inat_obs_42444465_crop_0.jpg) (Mariposa em repouso com asas dobradas sobre superfície plana);
  * [`inat_obs_244826083_crop_0.jpg`](images/inat_obs_244826083_crop_0.jpg) (Mariposa adulta com vista dorsal e asas triangulares sobre pano de amostragem);
  * [`inat_obs_247051453_crop_0.jpg`](images/inat_obs_247051453_crop_0.jpg) (Mariposa adulta cinzenta de grande porte em pano de captura noturna).
* **Causa-Raiz:** No iNaturalist (plataforma colaborativa de ciência cidadã), metadados de ciclo de vida frequentemente são marcados a nível de observação geral por voluntários. Usuários que monitoram o ciclo da praga anexam fotos do adulto emergido, ou marcam erradamente a fase larval em observações de espécimes adultos.
* **Hipótese de Impacto no Modelo:** A classe passa a induzir potencialmente uma **distribuição bimodal no espaço latente** (corpos cilíndricos/vermiformes vs. corpos triangulares/alados com escamas), dispersando a capacidade do extrator de características com padrões morfológicos não associados ao dano foliar direto.

### C. Falsos Positivos de Detecção (Fragmentos Anatômicos e Artefatos)
<img src="images/inat_obs_247158339_crop_4.jpg" alt="Fragmento ou Apêndice Isolado" width="180" />

* **Arquivo:** [`inat_obs_247158339_crop_4.jpg`](images/inat_obs_247158339_crop_4.jpg)
* **Descrição Visual:** Antena / apêndice isolado sobre tecido branco, sem corpo ou morfologia discernível de lagarta.
* **Causa-Raiz:** O detector YOLOv8n, calibrado para sensibilidade alta em caixas de pequeno porte, ocasionalmente detecta segmentos corporais ou detritos como pragas individuais (`pest`).
* **Hipótese de Impacto no Modelo:** Ruído estocástico de fundo que atua como regularizador indesejado ou vetor de confusão em classes com suporte reduzido.

---

## 📊 3. Correlação Empírica e Hipóteses Explicativas

A identificação dessas fontes de variabilidade amostral fornece indícios fundamentados para formular hipóteses explicativas sobre dois fenômenos mensurados no benchmark:

1. **Hipótese da Divergência entre Validação Interna e Teste Zero-Shot:**
   * **Fato Observado:** O F1-score de *S. albula* alcançou **0.903 a 0.963** na validação interna (*DatasetPests*), mas apresentou queda sensível para a faixa de **0.672 a 0.778** no benchmark externo (*INSECT12C*).
   * **Hipótese Explicativa:** Como a partição de validação foi particionada da mesma base minerada via API, ela preservava proporções similares de adultos e lagartas. Em contraste, o benchmark externo (*INSECT12C*) é composto **estritamente por lagartas fotografadas em lavouras reais**. Levanta-se a hipótese de que os extratores convolucionais despenderam capacidade representacional codificando morfologias aladas adultas, reduzindo a densidade discriminativa específica para formas larvais desfolhadoras em condições de campo.

2. **Hipótese da Resiliência por Capacidade Arquitetural:**
   * **Fato Observado:** Arquiteturas com maior contagem de parâmetros e mecanismos de atenção (e.g., `EfficientNetV2-B1`, F1=0.778 no INSECT12C) mantiveram desempenho substancialmente superior a redes ultraleves (e.g., `MobileNetV3-Small`, F1=0.481 no INSECT12C).
   * **Hipótese Explicativa:** Modelos com maior capacidade de canal aparentam possuir flexibilidade para acomodar distribuições multimodais no espaço latente sem sofrer interferência catastrófica entre subclasses morfológicas, enquanto redes compactas sofrem saturação na presença de padrões conflitantes.

---

## 📌 4. Conclusões Técnicas e Hipóteses para Trabalhos Futuros

### Síntese do Diagnóstico Qualitativo
* **Fatos Empíricos Confirmados:** A inspeção direta dos dados comprovou a ocorrência de polimorfismo de estágio de vida (mariposas) e falsos positivos de detecção (antenas/apêndices) associados aos metadados de ciência cidadã.
* **Hipótese Causal Formulada:** A penalidade de generalização externa em *Spodoptera albula* decorre, como hipótese provável, da bimodalidade distributiva no espaço de características morfológicas.

### Propostas de Validação Experimental (Trabalhos Futuros)
* **Validação Causal Controlada:** Conduzir um experimento controlado de ablação retreinando os modelos com a exclusão manual de 100% dos espécimes adultos e fragmentos, mensurando isoladamente o impacto no ganho de F1 no INSECT12C.
* **Classificação Hierárquica:** Propor pipeline em dois estágios (*Estágio Biológico: Larva vs. Adulto* ➔ *Espécie*), condicionando a classificação taxonômica à confirmação prévia da fase larval de interesse agronômico.
* **Calibração de Detector:** Otimizar o limiar de confiança (*confidence threshold*) e incorporar restrições de área mínima de caixa delimitadora no detector YOLO para mitigar a extração de fragmentos anatômicos desconexos.

---

## 📚 5. Referências Bibliográficas

* **JOCHER, G.; CHAURASIA, A.; QIU, J.** *Ultralytics YOLOv8*. Versão 8.0.0, 2023. Disponível em: <https://github.com/ultralytics/ultralytics>.
* **UNGER, S. et al.** *iNaturalist as a tool for citizen science and biodiversity monitoring: quality, biases and potential*. Biological Conservation, v. 257, p. 109099, 2021.
* **WU, X. et al.** *IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition*. In: **IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)**, 2019, p. 8787-8796. DOI: [10.1109/CVPR.2019.00899](https://doi.org/10.1109/CVPR.2019.00899).

