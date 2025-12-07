
from typing import Optional
import tensorflow as tf

from cnnClassifier.utils.logger import configure_logger
from cnnClassifier.entity.config_entity import ModelConfig, ImageConfig

logger = configure_logger(__name__)

class PrepareModel:
    """
    Componente para preparar o modelo de classificação CNN.
    """
    
    def __init__(self, model_config: ModelConfig = None, image_config: ImageConfig = None):
        """
        Inicializa o componente de preparação de modelo.
        
        Args:
            model_config: Configuração do modelo
            image_config: Configuração da imagem
        """
 
        if model_config is None:
            raise ValueError(
            " ModelConfig é obrigatória! Use ModelConfig.from_yaml() "
            "ou forneça configuração explícita."
            )
        
        if image_config is None:
            raise ValueError(
            "ImageConfig é obrigatória! Forneça configuração explícita."

            )
        
        self.model_config = model_config
        self.image_config = image_config

        if not hasattr(self.model_config, 'augmentation_enabled'):
            raise ValueError("'augmentation_enabled' não encontrado no YAML!")
        self.data_augmentation = self._data_augmentation()

        
    def build_model(self) -> tf.keras.Model:
        """
        Constrói e retorna um modelo Keras para classificação de imagens.
        
        Returns:
            tf.keras.Model: Modelo compilado pronto para treinamento
        """
        logger.info(f"🏗️ Construindo modelo {self.model_config.model_name}")
        logger.info(f"📐 Input shape: {self.image_config.size_tuple}")
        logger.info(f"🎯 Número de classes: {self.model_config.num_classes}")
        
        modelo = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=self.image_config.size_tuple),
            self.data_augmentation,
            tf.keras.layers.Rescaling(1./255),
            
            # Primeira camada convolucional
            tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            
            # Segunda camada convolucional
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            
            # Terceira camada convolucional
            tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            
            # Flatten e Dense layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(self.model_config.dropout_rate),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(self.model_config.dropout_rate),
            tf.keras.layers.Dense(self.model_config.num_classes, activation='softmax')
        ])
        

        # Compilar modelo
        modelo.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.model_config.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"✅ Modelo {self.model_config.model_name} construído com sucesso")
        return modelo
    
    def _data_augmentation(self):
        """
        Cria pipeline de data augmentation baseado na configuração.
        
        Returns:
            tf.keras.Sequential: Pipeline de augmentação
        """
        layers = []
        
        # Verificar se model_config tem configurações de augmentation
        if hasattr(self.model_config, 'horizontal_flip') and self.model_config.horizontal_flip:
            layers.append(tf.keras.layers.RandomFlip("horizontal"))
        else:
            layers.append(tf.keras.layers.RandomFlip("horizontal"))  # default
            
        if hasattr(self.model_config, 'rotation_factor'):
            layers.append(tf.keras.layers.RandomRotation(self.model_config.rotation_factor))
        else:
            layers.append(tf.keras.layers.RandomRotation(0.05))  # default
            
        if hasattr(self.model_config, 'zoom_factor'):
            layers.append(tf.keras.layers.RandomZoom(self.model_config.zoom_factor))
        else:
            layers.append(tf.keras.layers.RandomZoom(0.05))  # default
        
        data_augmentation = tf.keras.Sequential(layers)
        logger.info(f"🔄 Data augmentation configurado com {len(layers)} transformações")
        return data_augmentation
    
    def _pre_trained_model(self, include_top: bool = False, weights: str = "imagenet"):
        """
        Cria modelo pré-treinado baseado na configuração.
        
        Args:
            include_top: Se incluir camadas de classificação
            weights: Pesos pré-treinados para usar
            
        Returns:
            tf.keras.Model: Modelo base pré-treinado
        """
        model_name = self.model_config.model_name.lower()
        
        if "mobilenet" in model_name:
            modelo_base = tf.keras.applications.MobileNetV3Large(
                input_shape=self.image_config.size_tuple,
                include_top=include_top,
                weights=weights
            )
        elif "inception" in model_name:
            modelo_base = tf.keras.applications.InceptionV3(
                input_shape=self.image_config.size_tuple,
                include_top=include_top,
                weights=weights
            )
        elif "vgg" in model_name:
            modelo_base = tf.keras.applications.VGG19(
                input_shape=self.image_config.size_tuple,
                include_top=include_top,
                weights=weights
            )
        else:
            # Default para MobileNet
            modelo_base = tf.keras.applications.MobileNetV3Large(
                input_shape=self.image_config.size_tuple,
                include_top=include_top,
                weights=weights
            )
        
        logger.info(f"🏗️ Modelo base {model_name} carregado")
        return modelo_base

    def get_model_summary(self):
        """
        Retorna resumo das configurações do modelo.
        
        Returns:
            dict: Dicionário com informações do modelo
        """
        return {
            "model_name": self.model_config.model_name,
            "input_shape": self.image_config.size_tuple,
            "num_classes": self.model_config.num_classes,
            "dropout_rate": self.model_config.dropout_rate,
            "learning_rate": self.model_config.learning_rate,
        }
