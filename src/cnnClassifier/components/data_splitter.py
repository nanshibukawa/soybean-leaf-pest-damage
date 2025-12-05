import os
# 🔇 SILENCIA LOGS VERBOSOS DO TENSORFLOW 0=all, 1=info, 2=warning, 3=error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Silencia logs verbosos
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # Remove warnings oneDNN
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 



from dataclasses import dataclass
from pathlib import Path
import tensorflow as tf

from cnnClassifier.config.constants import DATA_SOURCE_DIR
from cnnClassifier.entity.config_entity import (
    DataSplitterConfig, 
    ImageConfig, 
    DataSubsetType)
from cnnClassifier.utils.logger import configure_logger

logger = configure_logger(__name__)

@dataclass
class DataSplitter:
    """Divide dados de imagem em conjuntos de treino e validação.
    
    Atributos:
        data_split_config (DataSplitterConfig): Configuração para proporções de divisão 
            de dados, tamanho do batch e random seed para divisões reproduzíveis
        image_config (ImageConfig): Configuração para dimensões de imagem (altura/largura)
            e o caminho do diretório de dados
        subset (DataSubsetType): Definições de tipo para subconjuntos de dados (TRAIN/VALIDATION)
            usado para especificar qual porção dos dados carregar
    """
    data_split_config: DataSplitterConfig
    image_config: ImageConfig
    subset: DataSubsetType

    def load_train_data(self):
        """
        Carrega dados de treino do diretório configurado.
        
        Cria um dataset TensorFlow para treino carregando imagens do diretório especificado
        e aplicando a proporção de divisão de treino configurada. O dataset é automaticamente
        embaralhado e agrupado em lotes de acordo com a configuração.
        
        Returns:
            tf.data.Dataset: Dataset de treino contendo imagens agrupadas em batchs e
                pré-processadas com seus rótulos correspondentes. Cada batch contém imagens
                redimensionadas para as dimensões configuradas (altura x largura) e o
                dataset usa o subconjunto de treino baseado na proporção de divisão.
        """
        treino = tf.keras.utils.image_dataset_from_directory(
            directory=self.image_config.data_dir,
            validation_split=1 - self.data_split_config.train_ratio,
            shuffle=True,
            subset=self.subset.TRAIN.value,
            seed=self.data_split_config.random_seed,
            image_size=(self.image_config.altura, self.image_config.largura),
            batch_size=self.data_split_config.batch_size
        )
        return treino

    def load_validation_data(self):
        """Carrega dados de validação do diretório configurado.
        
        Cria um dataset TensorFlow para validação carregando imagens do diretório 
        especificado e aplicando a proporção de divisão de validação configurada. 
        O dataset é automaticamente embaralhado e agrupado em batchs de acordo com a 
        configuração, usando a mesma semente dos dados de treino para divisões consistentes.
        
        Returns:
            tf.data.Dataset: Dataset de validação contendo imagens agrupadas em batchs e
                pré-processadas com seus rótulos correspondentes. Cada batch contém imagens
                redimensionadas para as dimensões configuradas (altura x largura) e o
                dataset usa o subconjunto de validação baseado na proporção de divisão.
        """
        validation = tf.keras.utils.image_dataset_from_directory(
            directory=self.image_config.data_dir,
            validation_split=self.data_split_config.val_ratio,
            shuffle=True,
            subset=self.subset.VALIDATION.value,
            seed=self.data_split_config.random_seed,
            image_size=(self.image_config.altura, self.image_config.largura),
            batch_size=self.data_split_config.batch_size,

        )
        return validation

if __name__ == "__main__":
    logger.info("Iniciando o DataSplitter...")
    
    data_split_config = DataSplitterConfig()
    image_config = ImageConfig(data_dir=Path(DATA_SOURCE_DIR))
    
    data_splitter = DataSplitter(
        data_split_config=data_split_config,
        image_config=image_config,
        subset=DataSubsetType
    )
    
    train_data = data_splitter.load_train_data()
    validation_data = data_splitter.load_validation_data()

    logger.info("Dados de treino e validação carregados com sucesso.")