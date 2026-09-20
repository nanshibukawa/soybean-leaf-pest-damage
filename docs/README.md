# 📚 Documentação Técnica e Científica do Projeto
## Classificação de Pragas da Soja em Borda

Este diretório reúne todos os relatórios, decisões de arquitetura, guias operacionais de dados e análises experimentais do projeto de pesquisa científica.

---

## 🗺️ Guia de Leitura Cronológica Recomendada

Para compreender a evolução e a fundamentação do projeto do início ao fim, recomenda-se a seguinte ordem de leitura:

```mermaid
flowchart TD
    A["01. Roteiro Técnico e Metodológico<br/>(01_ROTEIRO_TECNICO_E_METODOLOGICO.md)"] --> B["02. Metodologia e Decisões<br/>(02_METODOLOGIA_E_DECISOES.md)"]
    B --> C["03. Preparação e Limpeza de Dados<br/>(03_preparacao_dados/)"]
    C --> D["04. Experimentos, Perda e Benchmarks<br/>(04_experimentos_e_benchmark/)"]
    D --> E["05. Pesquisa de Campo & Módulo RAG<br/>(05_pesquisa_e_modulo_rag/)"]
```

---

### Fase 1: Visão Geral e Rigor Metodológico
1. [**`01_ROTEIRO_TECNICO_E_METODOLOGICO.md`**](01_ROTEIRO_TECNICO_E_METODOLOGICO.md):  
   *A fonte única da verdade.* Consolida as 10 classes de pragas, o particionamento 90/10 sem vazamento por imagem-mãe, as resoluções de entrada ($240\times240$ no B1), confirmação de desativação de CBAM/SRCNN e ativação de TopK Pooling, exportação TFLite FP16 e a tabela de resultados finais.
2. [**`02_METODOLOGIA_E_DECISOES.md`**](02_METODOLOGIA_E_DECISOES.md):  
   *Controle experimental rigoroso (Ceteris Paribus).* Justifica a otimização bayesiana independente (Keras Tuner) para baseline vs. proposta e detalha a matriz de modelos.

---

### Fase 2: Engenharia e Preparação de Dados ([`03_preparacao_dados/`](03_preparacao_dados/))
3. [**`03_preparacao_dados/01_pipeline_preprocessamento.md`**](03_preparacao_dados/01_pipeline_preprocessamento.md):  
   Fluxo ponta a ponta: extração de dados brutos, margem foliar de +20%, detector YOLOv8 para auto-crop no iNaturalist (licenças Creative Commons) e divisão por grupo.
4. [**`03_preparacao_dados/02_guia_execucao_treinamento.md`**](03_preparacao_dados/02_guia_execucao_treinamento.md):  
   Roteiro prático passo a passo de comandos CLI para treinar o IP102, rodar o fine-tuning e avaliar na base independente INSECT12C.

---

### Fase 3: Experimentos, Perda e Benchmarks ([`04_experimentos_e_benchmark/`](04_experimentos_e_benchmark/))
5. [**`04_experimentos_e_benchmark/01_analise_focal_loss_soja.md`**](04_experimentos_e_benchmark/01_analise_focal_loss_soja.md):  
   Dedução matemática da *Categorical Focal Loss* ($\gamma = 1.5$) demonstrando por que a Cross-Entropy com pesos simples destruía a precisão das classes raras.
6. [**`04_experimentos_e_benchmark/02_pretrain_ip102_log_report.md`**](04_experimentos_e_benchmark/02_pretrain_ip102_log_report.md):  
   Log e métricas de convergência das 50 épocas do pré-treinamento da EfficientNetV2-B1 no benchmark IP102.
7. [**`04_experimentos_e_benchmark/03_benchmark_ip102_e_downstream.md`**](04_experimentos_e_benchmark/03_benchmark_ip102_e_downstream.md):  
   Comparação detalhada com o Estado da Arte (SOTA 2019–2026) da literatura no IP102 e resultados downstream do fine-tuning.
8. [**`04_experimentos_e_benchmark/04_estudo_ablacao_ip102_vs_imagenet.md`**](04_experimentos_e_benchmark/04_estudo_ablacao_ip102_vs_imagenet.md):  
   Estudo preliminar de ablação (rodada de Julho/2026 com SGD fixo) que comprovou o salto de acurácia conferido pelo pré-treino de domínio.
9. [**`04_experimentos_e_benchmark/05_historico_refactoring_3classes.md`**](04_experimentos_e_benchmark/05_historico_refactoring_3classes.md):  
   *Documento histórico:* Memória técnica do planejamento de desacoplamento do extrator quando o projeto ainda abordava 3 classes preliminares.
10. [**`04_experimentos_e_benchmark/06_analise_qualitativa_ruido_mineracao_e_polimorfismo.md`**](04_experimentos_e_benchmark/06_analise_qualitativa_ruido_mineracao_e_polimorfismo.md):  
    Diagnóstico qualitativo de ruído amostral (mariposas adultas e artefatos de detecção YOLO) na classe *Spodoptera albula* e seu impacto no gap de generalização com o teste zero-shot (*INSECT12C*). Base textual pronta para a discussão de erros do manuscrito científico.

---

### Fase 4: Pesquisa de Campo & Módulo RAG (Projeto Pessoal) ([`05_pesquisa_e_modulo_rag/`](05_pesquisa_e_modulo_rag/))
11. [**`05_pesquisa_e_modulo_rag/01_pesquisa_solucoes_mercado.md`**](05_pesquisa_e_modulo_rag/01_pesquisa_solucoes_mercado.md):  
    Levantamento de aplicações e ferramentas concorrentes existentes para identificação de pragas no agronegócio.
12. [**`05_pesquisa_e_modulo_rag/02_roteiro_entrevista_campo.md`**](05_pesquisa_e_modulo_rag/02_roteiro_entrevista_campo.md):  
    Roteiro de entrevistas com agrônomos, produtores e cooperativas para validação de usabilidade da solução em campo.
13. [**`05_pesquisa_e_modulo_rag/03_arquitetura_sistema_rag.md`**](05_pesquisa_e_modulo_rag/03_arquitetura_sistema_rag.md):  
    Documentação técnica da prova de conceito da API de RAG: *Docling*, *Semantic Chunking* (HDBSCAN), representação híbrida (*Dense E5* + *Sparse BM25* + *ColBERT*), banco vetorial *Qdrant* e LLM *Llama 3.3*.
