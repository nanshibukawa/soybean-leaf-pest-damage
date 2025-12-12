from datetime import datetime
from pathlib import Path
import traceback
import tensorflow as tf

from cnnClassifier.config.constants import MODELS_DIR
from cnnClassifier.entity.config_entity import ModelConfig, ImageConfig
from cnnClassifier.components.model_evaluation import ModelEvaluator  # ← IMPORTA COMPONENTE
from cnnClassifier.pipeline.stage_02_data_splitting import DataSplittingPipeline
from cnnClassifier.utils.logger import configure_logger
from cnnClassifier.utils.data_utils import create_dirs

logger = configure_logger(__name__)

STAGE_NAME = "Model Evaluation"

class ModelEvaluationPipeline:
    """Pipeline que orquestra a avaliação de modelos"""
    
    def __init__(
            self, model_config: ModelConfig, 
            image_config: ImageConfig,
            data= None
                 ):
        self.data = data
        self.model_config = model_config
        self.image_config = image_config
        self.model_dir = MODELS_DIR
        self.evaluation_dir = Path("artifacts/evaluation")
        
        self.evaluator = ModelEvaluator(model_config, image_config, class_names=['Caterpillar', 'Diabrotica speciosa', 'Healthy'])
    
    def load_trained_model(self):
        """Carrega modelo treinado do disco"""
        possible_paths = [
            self.model_dir / "best_model.keras",
            self.model_dir / f"{self.model_config.model_name.lower()}_trained.keras",
            self.model_dir / "cnn_model.keras",
            self.model_dir / "cnn_model.h5"
        ]
        
        model_path = None
        for path in possible_paths:
            if path.exists():
                model_path = path
                break
        
        if not model_path:
            raise FileNotFoundError(f"Modelo não encontrado em {self.model_dir}")
        
        logger.info(f"📥 Carregando modelo: {model_path}")
        model = tf.keras.models.load_model(model_path)
        return model, model_path

    def _create_results_dir(self):
        """🔧 MÉTODO QUE ESTAVA FALTANDO!"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path("artifacts") / "model_evaluation" / f"evaluation_{timestamp}"
        create_dirs(save_dir)
        logger.info(f"📁 Diretório de resultados: {save_dir}")
        return save_dir

    def main(self, validation_data=None, model=None, history=None):
        """Pipeline único - carrega do disco OU usa modelo fornecido"""
        try:
            logger.info(f"🎯 Iniciando {STAGE_NAME}...")
            
            # Criar diretório com timestamp
            save_dir = self._create_results_dir()
            
            # Carregar dados
            if validation_data is None:
                validation_data = self.data['validation_data']
            
            if model is not None:
                logger.info("📥 Usando modelo fornecido (Stage 4)")
                used_model = model

                #TODO: ajustar para receber model path e history
                model_path = "add path model"
            else:
                # Carregar modelo do disco
                logger.info("💾 Carregando modelo do disco...")
                used_model, model_path = self.load_trained_model()
                history = None  # Sem histórico para modelos do disco
            
            result = self.evaluator.evaluate(
                model=used_model,
                validation_data=validation_data,
                save_dir=save_dir,
                history=history  # None para modelos do disco, real para Stage 4
            )
            
            result.update({
                "model_path": str(model_path),
                "evaluation_dir": str(save_dir),
                "message": f"Modelo avaliado com accuracy {result['metrics']['accuracy']:.4f}"
            })
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro no {STAGE_NAME}",
                         error=str(e),
                         traceback=traceback.format_exc()
            )
            return {"success": False, "error": str(e)}


if __name__ == "__main__":
    # Execução standalone
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
        # Usar configuração padrão
        model_config = ModelConfig()
        image_config = ImageConfig()
        
        pipeline = ModelEvaluationPipeline(model_config, image_config)
        result = pipeline.main()
        
        if result["success"]:
            logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
            logger.info(f"📊 Accuracy: {result['metrics']['accuracy']:.4f}")
            logger.info(f"🎯 F1-Score: {result['metrics']['f1_macro']:.4f}")
        else:
            logger.error(f">>>>>> stage {STAGE_NAME} failed <<<<<<")
            
    except Exception as e:
        logger.exception(e)
        raise e