# 🛠️ Plano de Correção do Dataset (Remoção de Mariposas Adultas)

Este documento serve como um guia passo a passo e checklist para corrigir a contaminação da classe `anticarsia_gemmatalis-larva` com imagens de mariposas adultas (fase alada). 

---

## 📋 Resumo do Problema
O downloader do iNaturalist buscou imagens pela espécie *Anticarsia gemmatalis* e retornou **99,84% de mariposas adultas** (1.894 de 1.897 imagens anotadas). A classe do classificador é destinada exclusivamente a identificar a **lagarta (fase larval)**, que é a causadora do dano físico às folhas de soja. 

---

## ⚡ Solução Semi-Automatizada com Script Auxiliar
Para evitar que você tenha que inspecionar manualmente 1.900 imagens na pasta do iNaturalist, criamos um script automatizado que consulta a API do iNaturalist em lotes e move todas as imagens anotadas oficialmente como **Adulto (ID 2)** para uma pasta de backup temporária.

Após o script rodar, você precisará revisar manualmente **apenas as imagens não anotadas ou anotadas como larvas (menos de 10 imagens)**.

---

## 🗺️ Passo a Passo para Correção

### [ ] Passo 1: Executar a Filtragem Automática de Mariposas
Execute o script auxiliar para mover as mariposas adultas para uma pasta de backup externa:
```bash
python scripts/data_preparation/auto_filter_moths.py
```
* **O que o script faz:** Cria a pasta `artifacts/data_ingestion/inaturalist/anticarsia_gemmatalis/adult_moths_backup/` e move todas as imagens de mariposas adultas para lá.

### [ ] Passo 2: Revisão Manual Restante
Navegue até a pasta de imagens brutas de *Anticarsia*:
👉 [artifacts/data_ingestion/inaturalist/anticarsia_gemmatalis/images/](../../artifacts/data_ingestion/inaturalist/anticarsia_gemmatalis/images/)
1. O diretório deve conter pouquíssimas imagens agora.
2. Inspecione visualmente as imagens restantes.
3. Se houver alguma mariposa adulta residual (que não estava anotada na API), **delete-a**.
4. Certifique-se de que restaram apenas as fotos que mostram a **lagarta** (caterpillar).

### [ ] Passo 3: Regenerar os Crops com YOLOv8
Limpe os crops antigos de *Anticarsia* (para garantir que nenhuma mariposa antiga permaneça na pasta processada) e rode o auto-crop novamente:
```bash
# Remover crops antigos
rm -rf artifacts/data/processed/DatasetPests-cropped/anticarsia_gemmatalis-larva

# Gerar novos crops usando o detector treinado
python scripts/yolo_inaturalist/auto_crop_inaturalist.py
```
👉 Script de recorte: [auto_crop_inaturalist.py](../../scripts/yolo_inaturalist/auto_crop_inaturalist.py)

### [ ] Passo 4: Atualizar as Partições (Split)
Execute o script de divisão para recriar as pastas `train` e `val` sem vazamento de dados agrupadas por imagem-mãe:
```bash
python scripts/data_preparation/split_dataset_by_group.py
```
👉 Script de split: [split_dataset_by_group.py](../../scripts/data_preparation/split_dataset_by_group.py)

### [ ] Passo 5: Compactar e Atualizar Backup (Opcional)
Se desejar manter o backup das imagens brutas limpas no Google Drive:
1. Compacte a pasta `artifacts/data_ingestion/inaturalist/` (agora contendo apenas as imagens limpas).
2. Salve o arquivo como `inaturalist_raw.zip` e suba no Google Drive.

### [ ] Passo 6: Executar Novo Treinamento/Tuning
Agora que os dados estão limpos e representam realmente lagartas, re-execute o treinamento para obter métricas de validação e teste reais:
```bash
python scripts/main.py --experiment efficientnetv2b1
```

---

## 🛠️ Script de Filtragem (`scripts/data_preparation/auto_filter_moths.py`)
O código do script auxiliar já foi criado em [auto_filter_moths.py](../../scripts/data_preparation/auto_filter_moths.py). Ele utiliza a API do iNaturalist para identificar e mover os adultos de forma segura.
