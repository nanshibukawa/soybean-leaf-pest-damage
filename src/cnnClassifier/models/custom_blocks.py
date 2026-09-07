import tensorflow as tf


def compression_block(filters, kernel_size=3, strides=1, l2_reg=0.01):
    return tf.keras.Sequential(
        [
            tf.keras.layers.DepthwiseConv2D(
                kernel_size, strides=strides, padding="same"
            ),
            tf.keras.layers.Conv2D(
                filters,
                1,
                padding="same",
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(l2_reg),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ]
    )


def se_block(input_tensor, ratio=8):
    """
    Implementa bloco Squeeze-and-Excitation (SE).

    Este bloco realiza recalibração de canais adaptativa através de dois passos:
    1. Squeeze: Comprime informações espaciais usando GlobalAveragePooling2D
    2. Excitation: Modela relacionamentos inter-canais com duas camadas Dense

    O resultado é multiplicado element-wise com o tensor de entrada original.

    Args:
        input_tensor (tf.Tensor): Tensor de entrada com shape (batch, height, width, channels)
        ratio (int): Fator de redução para o gargalo (bottleneck) no bloco SE.
                    Default: 8. O número de neurônios na primeira Dense será
                    filters // ratio. Deve ser >= 1.

    Returns:
        tf.Tensor: Tensor recalibrado com mesmo shape que input_tensor

    Raises:
        ValueError: Se ratio <= 0 ou se filters // ratio < 1

    Nota:
        A validação garante que filters // ratio >= 1 para evitar dimensões inválidas.
    """
    filters = input_tensor.shape[-1]

    # Validação: garantir que ratio é válido
    if ratio <= 0:
        raise ValueError(f"ratio deve ser positivo, recebido: {ratio}")

    reduced_filters = filters // ratio
    if reduced_filters < 1:
        raise ValueError(
            f"Número de filtros ({filters}) muito pequeno para ratio ({ratio}). "
            f"filters // ratio deve ser >= 1, obteve {reduced_filters}. "
            f"Considere usar ratio <= {filters}."
        )

    # Squeeze: comprime informações espaciais
    se = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)

    # Excitation: modela dependências inter-canais
    se = tf.keras.layers.Dense(reduced_filters, activation="relu")(se)
    se = tf.keras.layers.Dense(filters, activation="sigmoid")(se)

    # Reshape para compatibilidade com multiplicação
    se = tf.keras.layers.Reshape((1, 1, filters))(se)

    # Recalibração: multiplicação element-wise
    return tf.keras.layers.multiply([input_tensor, se])


def channel_attention(input_tensor, ratio=8):
    """Atenção de canal para o bloco CBAM."""
    channels = input_tensor.shape[-1]
    
    # Garantir que o canal reduzido seja de pelo menos 1
    reduced_channels = channels // ratio
    if reduced_channels < 1:
        reduced_channels = 1

    # Compartilhamento do MLP (Dense layers)
    shared_layer_one = tf.keras.layers.Dense(
        reduced_channels,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=True,
        bias_initializer="zeros",
    )
    shared_layer_two = tf.keras.layers.Dense(
        channels,
        kernel_initializer="he_normal",
        use_bias=True,
        bias_initializer="zeros",
    )
    
    # Caminho AvgPool
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    avg_pool = tf.keras.layers.Reshape((1, 1, channels))(avg_pool)
    
    # Caminho MaxPool
    max_pool = tf.keras.layers.GlobalMaxPooling2D()(input_tensor)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    max_pool = tf.keras.layers.Reshape((1, 1, channels))(max_pool)
    
    # Fusão de caminhos e aplicação de ativação sigmoid
    cbam_feature = tf.keras.layers.add([avg_pool, max_pool])
    cbam_feature = tf.keras.layers.Activation("sigmoid")(cbam_feature)
    
    # Multiplicação element-wise
    return tf.keras.layers.multiply([input_tensor, cbam_feature])


@tf.keras.utils.register_keras_serializable(package="Custom", name="SpatialAttentionFeatureReduction")
class SpatialAttentionFeatureReduction(tf.keras.layers.Layer):
    """
    Camada customizada para extrair médias e máximos ao longo do canal 
    e concatená-los, servindo de entrada para o Spatial Attention.
    """
    def __init__(self, **kwargs):
        super(SpatialAttentionFeatureReduction, self).__init__(**kwargs)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        return tf.keras.layers.concatenate([avg_pool, max_pool], axis=-1)


def spatial_attention(input_tensor, kernel_size=7):
    """Atenção espacial para o bloco CBAM."""
    # Concatenar ao longo do canal usando camada customizada serializável
    concat = SpatialAttentionFeatureReduction()(input_tensor)
    
    # Convolução 2D para gerar o mapa de atenção espacial de canal único (1)
    cbam_feature = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=kernel_size,
        strides=1,
        padding="same",
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=False,
    )(concat)
    
    # Multiplicação element-wise
    return tf.keras.layers.multiply([input_tensor, cbam_feature])


def cbam_block(input_tensor, ratio=8, kernel_size=7):
    """
    Implementa o bloco CBAM (Convolutional Block Attention Module).
    Combina a recalibração de canais com o mapeamento de relevância espacial.
    """
    x = channel_attention(input_tensor, ratio=ratio)
    x = spatial_attention(x, kernel_size=kernel_size)
    return x


@tf.keras.utils.register_keras_serializable(package="Custom", name="ResidualSRCNNBlock")
class ResidualSRCNNBlock(tf.keras.layers.Layer):
    """
    Bloco de Super-Resolução Residual (SRCNN modificada) para ser integrado
    no início da rede, aprendendo a reconstruir/aguçar texturas de forma end-to-end.
    """
    def __init__(self, filters_1=64, filters_2=32, kernel_size_1=9, kernel_size_2=5, kernel_size_3=5, **kwargs):
        super(ResidualSRCNNBlock, self).__init__(**kwargs)
        self.filters_1 = filters_1
        self.filters_2 = filters_2
        self.kernel_size_1 = kernel_size_1
        self.kernel_size_2 = kernel_size_2
        self.kernel_size_3 = kernel_size_3

    def build(self, input_shape):
        self.conv1 = tf.keras.layers.Conv2D(
            self.filters_1, 
            self.kernel_size_1, 
            padding="same", 
            activation="relu", 
            kernel_initializer="he_normal",
            name="srcnn_conv1"
        )
        self.conv2 = tf.keras.layers.Conv2D(
            self.filters_2, 
            self.kernel_size_2, 
            padding="same", 
            activation="relu", 
            kernel_initializer="he_normal",
            name="srcnn_conv2"
        )
        # Saída com 3 canais correspondentes aos canais de cor da imagem
        # Inicializado com zeros para que comece como uma transformação identidade
        self.conv3 = tf.keras.layers.Conv2D(
            3, 
            self.kernel_size_3, 
            padding="same", 
            kernel_initializer="zeros",
            name="srcnn_conv3"
        )
        super(ResidualSRCNNBlock, self).build(input_shape)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        residual = self.conv3(x)
        # Conexão residual
        return tf.keras.layers.add([inputs, residual])

    def get_config(self):
        config = super(ResidualSRCNNBlock, self).get_config()
        config.update({
            "filters_1": self.filters_1,
            "filters_2": self.filters_2,
            "kernel_size_1": self.kernel_size_1,
            "kernel_size_2": self.kernel_size_2,
            "kernel_size_3": self.kernel_size_3
        })
        return config

