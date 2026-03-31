import tensorflow as tf


def compression_block(filters, kernel_size=3, strides=1):
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
                kernel_regularizer=tf.keras.regularizers.L2(0.01),
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
