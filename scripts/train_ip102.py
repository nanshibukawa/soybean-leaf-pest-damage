import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

import argparse
from pathlib import Path
import warnings
import tensorflow as tf

from cnnClassifier.pipeline.stage_02_data_splitting import DataSplittingPipeline
from cnnClassifier.components.prepare_model import PrepareModel
from cnnClassifier.components.model_training import ModelTraining
from cnnClassifier.utils.logger import configure_logger
from cnnClassifier.entity.config_entity import ImageConfig, ModelConfig
from cnnClassifier.utils.data_utils import create_dirs
from cnnClassifier.config.constants import MODELS_DIR

warnings.filterwarnings("ignore")
logger = configure_logger(__name__)
ROOT_DIR = Path(__file__).resolve().parent.parent

def main():
    parser = argparse.ArgumentParser(description="Passo 1: Domain-Specific Pre-training (IP102)")
    default_dir = "artifacts/data/ip102_cropped" if (ROOT_DIR / "artifacts/data/ip102_cropped").exists() else "artifacts/data/ip102"
    parser.add_argument(
        "--data_dir",
        type=str,
        default=default_dir,
        help="Diretório contendo os recortes do dataset IP102"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenetv3small",
        help="Arquitetura para pré-treinamento no IP102 (ex: mobilenetv3small, mobilenetv3large, efficientnetv2b1)"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Nome do experimento no model_params.yaml (sobrescreve o default baseado em --model)"
    )
    args = parser.parse_args()

    # Mapear experimento
    if args.experiment:
        experiment_name = args.experiment
    else:
        experiment_name = f"ip102_{args.model.lower()}_pretrain"

    logger.info(f"🚀 Iniciando Passo 1: Domain-Specific Pre-training (IP102) para '{args.model}' (exp: {experiment_name})...")
    
    try:
        # 1. Carregar Configurações do Experimento IP102
        logger.info(f"📋 Carregando configurações do experimento '{experiment_name}'...")
        model_config = ModelConfig.from_yaml("model_params.yaml", experiment=experiment_name)
        
        # 2. Configurar diretório de imagens do IP102
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"❌ Diretório do dataset IP102 não encontrado em: {data_dir}")
            
        image_config = ImageConfig(
            altura=model_config.image_size[0],
            largura=model_config.image_size[1],
            canais=model_config.image_size[2],
            data_dir=data_dir,
        )

        # 3. Divisão e Carregamento de Dados
        logger.info("🔄 === Divisão e Carregamento de Dados (IP102) ===")
        data_splitting = DataSplittingPipeline(
            config=model_config, image_config=image_config
        )
        stage2_result = data_splitting.main()
        
        if not stage2_result.get("success"):
            raise Exception(f"Falha na divisão dos dados: {stage2_result.get('error')}")

        # 4. Construção do Modelo (Classes detectadas dinamicamente)
        train_ds = stage2_result["train_data"]
        discovered_classes = train_ds.element_spec[1].shape[-1]
        logger.info(f"🔍 Classes detectadas no dataset IP102: {discovered_classes}")
        model_config.num_classes = discovered_classes

        logger.info(f"🏗️ === Preparação do Modelo ({discovered_classes} Classes) ===")
        create_dirs(MODELS_DIR)
        
        preparador_modelo = PrepareModel(
            model_config=model_config,
            image_config=image_config
        )
        model = preparador_modelo.build_model(num_classes=discovered_classes)

        # 5. Treinamento
        logger.info("🔥 === Iniciando Treinamento ===")
        # train_dir é passado para cálculo opcional de pesos das classes no dataset de pre-treino
        train_dir = data_dir / "train" if (data_dir / "train").exists() else data_dir
        trainer = ModelTraining(model, model_config, train_dir=train_dir)
        
        history = trainer.train_model(
            train_data=stage2_result["train_data"],
            validation_data=stage2_result["validation_data"]
        )

        # 6. Salvamento do Extrator de Características (Backbone + Atenção + Pooling)
        logger.info("💾 === Exportando Extrator de Características (Passo 1 concluído) ===")
        
        try:
            # Pegamos a entrada da camada 'classification_head' para reter todo o topo intermediário
            classification_layer = model.get_layer("classification_head")
            feature_output = classification_layer.input
            logger.info("✅ Camada 'classification_head' identificada. Mantendo todas as camadas intermediárias.")
        except ValueError:
            # Fallback caso a camada de classificação final não possua esse nome exato
            feature_output = model.layers[-2].output
            logger.warning("⚠️ Camada 'classification_head' não encontrada. Usando penúltima camada como fallback.")
            
        model_tag = model_config.model_name.lower()
        extractor_model = tf.keras.Model(
            inputs=model.input,
            outputs=feature_output,
            name=f"ip102_{model_tag}_extractor"
        )
        
        extractor_filename = f"ip102_{model_tag}_extractor.keras"
        extractor_path = MODELS_DIR / extractor_filename
        extractor_model.save(str(extractor_path))
        logger.info(f"🎉 Extrator de características salvo com sucesso em: {extractor_path}")

        # Também salvamos o modelo completo de classificação (102 classes) para fins de log/auditoria
        full_model_filename = f"ip102_{model_tag}_full_model.keras"
        full_model_path = MODELS_DIR / full_model_filename
        model.save(str(full_model_path))
        logger.info(f"📊 Modelo completo do Passo 1 (102 classes) salvo em: {full_model_path}")

    except Exception as e:
        logger.exception(f"💥 Pipeline do Passo 1 falhou: {e}")
        exit(1)

if __name__ == "__main__":
    main()
