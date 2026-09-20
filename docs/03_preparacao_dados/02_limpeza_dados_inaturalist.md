# Guia de Limpeza e Preparação do Dataset (iNaturalist & DatasetPests)

Este guia detalha o procedimento para limpar as imagens brutas mineradas da API do iNaturalist, remover ruídos (fase adulta/mariposa), reexecutar o processamento local e preparar o arquivo final para upload no Google Drive.

---

## 🛠️ Passo 1: Filtragem Manual de Mariposas Adultas (*Anticarsia gemmatalis*)

Como a API do iNaturalist busca por espécie e retorna tanto a **lagarta (larva)** quanto a **mariposa (adulto)**, precisamos remover as mariposas adultas para evitar falsos positivos no classificador.

1. Navegue até a pasta de imagens brutas de Anticarsia:
   👉 [artifacts/data_ingestion/inaturalist/anticarsia_gemmatalis/images/](../../artifacts/data_ingestion/inaturalist/anticarsia_gemmatalis/images/)
2. Visualize as imagens e **delete todas que mostrarem a mariposa adulta** (inseto com asas abertas marrons/acinzentadas, ex: `obs_194509610.jpg`).
3. Mantenha apenas as fotos que mostram a **lagarta** (caterpillar).

---

## ✂️ Passo 2: Regenerar os Recortes (Auto-Crop) com YOLOv8

Após remover as mariposas adultas das fotos brutas, você deve apagar os recortes antigos da pasta processada e rodar o detector para gerar recortes limpos:

1. Delete a pasta processada antiga de *Anticarsia* para garantir que nenhum crop de mariposa antiga permaneça:
   ```bash
   rm -rf artifacts/data/processed/DatasetPests-cropped/anticarsia_gemmatalis-larva
   ```
2. Execute o script de recorte automático. Ele lerá as imagens brutas restantes e usará o detector YOLOv8 para recortar as lagartas:
   ```bash
   python scripts/yolo_inaturalist/auto_crop_inaturalist.py
   ```
   *Os novos recortes limpos serão salvos em `artifacts/data/processed/DatasetPests-cropped/`.*

---

## 📊 Passo 3: Executar a Divisão do Dataset (Split Seguro)

Com os recortes atualizados e limpos na pasta `processed`, rode o script de divisão para recriar as partições de treino e validação agrupadas de forma segura por imagem-mãe (evitando data leakage):

```bash
python scripts/data_preparation/split_dataset_by_group.py
```
*O resultado final limpo estará pronto para treinamento em `artifacts/data/final/DatasetPests-split/`.*

---

## 📦 Passo 4: Empacotar e Fazer Upload no Google Drive

Para disponibilizar as imagens **brutas (raw)** limpas para outros ambientes ou execuções do pipeline (evitando ter que baixar tudo via API novamente):

1. Vá para a pasta de ingestão:
   👉 [artifacts/data_ingestion/inaturalist/](../../artifacts/data_ingestion/inaturalist/)
2. Compacte as seguintes pastas brutas de espécies (incluindo as imagens e o `metadata.json` de cada uma):
   * `anticarsia_gemmatalis/` (agora limpa sem mariposas)
   * `euschistus_heros/`
   * `gastropoda/`
   * `rhammatocerus_schistocercoides/`
   * `spodoptera_albula/`
3. Salve o arquivo compactado como **`inaturalist_raw.zip`**.
4. Faça o upload desse arquivo para o seu Google Drive.
5. Se o ID do link de compartilhamento mudar, atualize a constante `INAT_RAW_URL` no arquivo de configuração do projeto:
   👉 [src/cnnClassifier/config/constants.py](../../src/cnnClassifier/config/constants.py#L23)

---

## 💡 Sobre os arquivos com sufixo `_augXXXX`

* **Origem:** Esses arquivos (ex: `..._aug9204_crop_0.jpg`) existem apenas no arquivo antigo baixado do Drive (`DatasetPests-cropped-10-classes`). Eles foram gerados por um processo anterior de *Data Augmentation offline* (rotação, espelhamento, etc.).
* **Processamento Local:** Ao rodar os scripts localmente, a pasta processada `artifacts/data/processed/DatasetPests-cropped` conterá apenas os crops originais (`orig_...`) e os crops do iNaturalist (`inat_...`). Os scripts ignoram as entradas aumentadas que não existem fisicamente no disco.
* **Segurança:** O pipeline agrupa os dados pelo nome base original do arquivo (removendo `_aug`), garantindo que nenhuma imagem derivada/aumentada seja dividida entre treino e validação ao mesmo tempo.
