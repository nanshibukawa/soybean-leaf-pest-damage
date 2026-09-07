import tensorflow as tf
from cnnClassifier.utils.logger import configure_logger

logger = configure_logger(__name__)

def configure_backbone_trainability(model: tf.keras.Model, unfreeze_last_n: int, backbone_name: str = "core_backbone"):
    """
    Configura quais camadas do backbone estão congeladas ou liberadas.
    Mantém as camadas BatchNormalization sempre congeladas para preservar estatísticas.
    Suporta modelos com backbones aninhados dentro de outros submodelos (ex: extrator do Passo 1).
    
    Args:
        model: Modelo Keras funcional ou sequencial.
        unfreeze_last_n: Número de camadas finais do backbone a descongelar. 
                         Se for <= 0, congela o backbone inteiro.
        backbone_name: Nome da camada do backbone no modelo.
    """
    backbone = None
    parent_models = []

    def find_layer_recursive(current_model):
        nonlocal backbone
        try:
            layer = current_model.get_layer(backbone_name)
            backbone = layer
            return True
        except ValueError:
            pass

        for layer in current_model.layers:
            if isinstance(layer, tf.keras.Model) or hasattr(layer, "layers"):
                if find_layer_recursive(layer):
                    parent_models.append(layer)
                    return True
        return False

    find_layer_recursive(model)

    if backbone is None:
        logger.error(f"❌ Layer '{backbone_name}' não encontrada no modelo!")
        raise ValueError(f"Layer '{backbone_name}' não encontrada no modelo!")

    # Garantir que todos os containers pais estejam com trainable=True
    for p in parent_models:
        p.trainable = True
        logger.info(f"🔓 Ativando trainable=True no container pai aninhado: {p.name}")

    total_layers = len(backbone.layers)
    
    if unfreeze_last_n <= 0:
        backbone.trainable = False
        logger.info(f"🔒 Todo o backbone '{backbone_name}' foi congelado ({total_layers} camadas).")
        return

    # Habilita trainable no backbone, mas congela camadas individuais
    backbone.trainable = True
    layers_to_unfreeze = min(unfreeze_last_n, total_layers)
    
    # Congelar as primeiras camadas, descongelar as últimas N (exceto BN)
    frozen_count = total_layers - layers_to_unfreeze
    
    for layer in backbone.layers[:frozen_count]:
        layer.trainable = False
        
    unfrozen_count = 0
    bn_count = 0
    for layer in backbone.layers[frozen_count:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
            unfrozen_count += 1
        else:
            layer.trainable = False
            bn_count += 1
            
    logger.info(
        f"🔓 Fine-tuning: {unfrozen_count} camadas liberadas, {bn_count} camadas BN mantidas "
        f"congeladas de um total de {total_layers} camadas no backbone '{backbone_name}'."
    )


def create_sgd_optimizer(
    initial_lr: float,
    peak_lr: float,
    epochs: int,
    steps_per_epoch: int,
    momentum: float = 0.9,
    nesterov: bool = True,
    alpha: float = 0.001,
):
    """
    Cria um otimizador SGD acoplado com CosineDecay learning rate schedule e warmup.
    
    Args:
        initial_lr: Taxa de aprendizado inicial (início do warmup).
        peak_lr: Taxa de aprendizado máxima (fim do warmup, início do decaimento).
        epochs: Número total de épocas de treino.
        steps_per_epoch: Número de batches (passos) por época.
        momentum: Momentum do SGD (padrão 0.9).
        nesterov: Utilizar gradiente acelerado de Nesterov (padrão True).
        alpha: Fração mínima do peak_lr no fim do decaimento para evitar estagnação.
    """
    total_decay_steps = max(1, epochs * steps_per_epoch)
    # Warmup dura 10% do total de passos
    warmup_steps = max(1, int(0.1 * total_decay_steps))
    
    logger.info(
        f"📈 Configurando CosineDecay: warmup_steps={warmup_steps}, "
        f"decay_steps={total_decay_steps}, initial_lr={initial_lr}, peak_lr={peak_lr}"
    )
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=total_decay_steps,
        alpha=alpha,
        warmup_target=peak_lr,
        warmup_steps=warmup_steps,
    )
    
    return tf.keras.optimizers.SGD(
        learning_rate=lr_schedule,
        momentum=momentum,
        nesterov=nesterov
    )
