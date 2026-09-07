#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import tensorflow as tf

# Add root directory to python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cnnClassifier.entity.config_entity import ModelConfig, ImageConfig, DataSplitterConfig, DataSubsetType
from cnnClassifier.components.data_splitter import DataSplitter
from cnnClassifier.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline
from cnnClassifier.utils.logger import configure_logger
from cnnClassifier.utils.data_utils import register_preprocess_input

logger = configure_logger("evaluate_test")

def main(experiment: str, model_path_override: str = None):
    logger.info(f"🔍 Iniciando avaliação standalone para o experimento: {experiment}")
    
    # Configure GPU memory growth
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("✅ GPU Memory Growth ativado")
        except RuntimeError as e:
            logger.warning(f"⚠️ Erro ao configurar Memory Growth: {e}")

    # 1. Carregar Configurações
    try:
        model_config = ModelConfig.from_yaml("model_params.yaml", experiment=experiment)
        image_config = ImageConfig(
            altura=model_config.image_size[0],
            largura=model_config.image_size[1],
            canais=model_config.image_size[2],
            data_dir=Path("artifacts/data/final/DatasetPests-split")
        )
    except Exception as e:
        logger.error(f"❌ Erro ao carregar configurações: {e}")
        sys.exit(1)

    # 2. Carregar dados de teste
    logger.info("📦 Carregando dados de teste...")
    try:
        data_splitter_config = DataSplitterConfig(
            batch_size=model_config.batch_size or 32,
            random_seed=model_config.random_seed,
            train_ratio=model_config.train_ratio,
            val_ratio=model_config.val_ratio,
            test_ratio=model_config.test_ratio,
        )
        data_splitter = DataSplitter(
            data_split_config=data_splitter_config,
            image_config=image_config,
            subset=DataSubsetType.TEST,
        )
        test_data = data_splitter.load_test_data()
    except Exception as e:
        logger.error(f"❌ Erro ao carregar dados de teste: {e}")
        sys.exit(1)

    # 3. Carregar modelo
    if model_path_override:
        model_path = Path(model_path_override)
    else:
        # Priorizar o modelo treinado recentemente pelo main.py
        model_path = Path(f"artifacts/models/{model_config.model_name.lower()}_trained.keras")
        if not model_path.exists():
            # Tentar caminhos alternativos se não achar
            alternatives = [
                Path(f"artifacts/models/mobile/{model_config.model_name}_keras_tuner_best.keras"),
                Path("artifacts/models/best_model.keras")
            ]
            for alt in alternatives:
                if alt.exists():
                    model_path = alt
                    break
            else:
                logger.error(f"❌ Modelo não encontrado nos caminhos padrão. Use --model_path para especificar.")
                sys.exit(1)
            
    logger.info(f"📥 Carregando modelo treinado de: {model_path}")
    try:
        register_preprocess_input(model_config.model_name)
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo: {e}")
        sys.exit(1)

    # 4. Executar avaliação
    logger.info("🎯 Executando etapa de teste final...")
    try:
        # Passamos test_data no lugar de validation_data nos campos da pipeline
        eval_pipeline = ModelEvaluationPipeline(
            model_config=model_config,
            image_config=image_config,
            data={"validation_data": test_data}
        )
        
        test_result = eval_pipeline.main(
            validation_data=test_data,
            model=model,
            history=None,
            prefix="test_evaluation_standalone"
        )
        
        if test_result["success"]:
            logger.info("=" * 80)
            logger.info("✨ AVALIAÇÃO DE TESTE REALIZADA COM SUCESSO!")
            logger.info("=" * 80)
            logger.info(f"📊 Acurácia: {test_result['metrics']['accuracy']:.4f}")
            logger.info(f"🎯 F1-Score (Macro): {test_result['metrics']['f1_macro']:.4f}")
            logger.info(f"📁 Resultados salvos em: {test_result['evaluation_dir']}")
            logger.info("=" * 80)
        else:
            logger.error(f"❌ Avaliação falhou: {test_result.get('error')}")
            
    except Exception as e:
        logger.exception(f"💥 Falha ao executar avaliação: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Roda a avaliação final do modelo treinado no conjunto de teste externo.")
    parser.add_argument("--experiment", type=str, default="efficientnetv2b1", help="Nome do experimento no YAML.")
    parser.add_argument("--model_path", type=str, default=None, help="Caminho direto para o arquivo de pesos do modelo .keras.")
    args = parser.parse_args()
    main(args.experiment, args.model_path)
