import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import os
from cnnClassifier.components.prepare_model import TopKGlobalAveragePooling2D


@keras.saving.register_keras_serializable(package="builtins")
def preprocess_input(x, **kwargs):
    return tf.keras.applications.mobilenet_v3.preprocess_input(x)


@keras.saving.register_keras_serializable(package="builtins", name="function")
def function(x, **kwargs):
    return tf.keras.applications.mobilenet_v3.preprocess_input(x)


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # 1. Identificar onde está a camada e isolar o submodelo se necessário
    submodel = None
    target_layer = None

    # Procura no modelo principal
    for layer in model.layers:
        if layer.name == last_conv_layer_name:
            target_layer = layer
            break
        # Se estiver dentro de um submodelo (ex: 'core_backbone')
        if hasattr(layer, "layers"):
            for sub_layer in layer.layers:
                if sub_layer.name == last_conv_layer_name:
                    submodel = layer
                    target_layer = sub_layer
                    break
            if target_layer:
                break

    if target_layer is None:
        raise ValueError(f"Camada {last_conv_layer_name} não foi encontrada.")

    # 2. Se a camada está num submodelo, criamos o Grad-CAM olhando para dentro dele
    if submodel is not None:
        # Cria um modelo interno que vai da entrada do submodelo até a camada conv e a saída do submodelo
        grad_model = tf.keras.models.Model(
            submodel.inputs, [target_layer.output, submodel.output]
        )

        # Passamos a imagem primeiro pelas camadas anteriores ao submodelo (se houver)
        # No seu caso, a imagem vai direto para o submodelo da MobileNet
        with tf.GradientTape() as tape:
            # Pega as ativações e a saída do backbone
            last_conv_layer_output, submodel_features = grad_model(img_array)

            # Passa a saída do backbone pelas camadas finais do modelo principal (Pooling, Dense...)
            # Para isso, reconstruímos o caminho do meio para o fim:
            x = submodel_features

            # Encontra o índice de onde o submodel estava para rodar o resto da rede
            submodel_index = model.layers.index(submodel)
            for layer in model.layers[submodel_index + 1 :]:
                x = layer(x)

            preds = x
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

    else:
        # Caso padrão (se o modelo fosse totalmente plano)
        grad_model = tf.keras.models.Model(
            model.inputs, [target_layer.output, model.output]
        )
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

    # 3. Cálculo normal do Grad-CAM (daqui para baixo permanece igual)
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path="cam_output.jpg", alpha=0.4):
    # image_array = cv2.imread(img_path)
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)
    jet = plt.colormaps.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)
    print(f"Salvo em {cam_path}")


# USO
if __name__ == "__main__":
    # Caminho do seu modelo final (.keras, não .tflite)

    # No bloco de carregamento:
    model_path = os.getenv(
        "MODEL_PATH",
        "artifacts/models/mobile/MobileNetV3Large_keras_tuner_best.keras"
    )
    # Fallback para o caminho local do usuário se o padrão não existir e o local existir
    if not os.path.exists(model_path):
        local_user_path = "/home/nanshibukawa/Documents/teste_images/health/MobileNetV3Large_keras_tuner_best (1).keras"
        if os.path.exists(local_user_path):
            model_path = local_user_path

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"TopKGlobalAveragePooling2D": TopKGlobalAveragePooling2D},
    )

    # Encontrar automaticamente a última camada convolucional 4D (mesmo em modelos aninhados)
    last_conv_layer_name = None

    def find_last_conv(layers):
        for layer in reversed(layers):
            # Se for uma subcamada/modelo aninhado (comum em modelos do Keras Tuner)
            if hasattr(layer, "layers"):
                res = find_last_conv(layer.layers)
                if res:
                    return res
            try:
                if hasattr(layer, "output") and len(layer.output.shape) == 4:
                    if any(
                        x in layer.name.lower()
                        for x in ["conv", "activation", "re_lu", "expanded_conv"]
                    ):
                        return layer.name
            except (AttributeError, RuntimeError):
                continue
        return None

    last_conv_layer_name = find_last_conv(model.layers)
    print(f"Última camada convolucional usada: {last_conv_layer_name}")

    if last_conv_layer_name is None:
        raise ValueError(
            "Não foi possível encontrar uma camada 4D válida. Verifique a estrutura do modelo."
        )

    # Lista das duas imagens problemáticas
    img_paths = [
        "/home/nanshibukawa/Documents/teste_images/mobile_test/images (3).jpeg",  # Imagem 1
        "/home/nanshibukawa/Documents/teste_images/health/Foco-folha-soja-Vantagens-e-desvantagens-da-producao-de-soja-no-Brasil.jpg",  # Imagem 2
    ]

    for path in img_paths:
        if not os.path.exists(path):
            continue

        img = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        # Atenção: Dependendo do modelo (ex: MobileNetV3), pode ser necessário aplicar funções de prepocessing.
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        pred_idx = np.argmax(preds[0])
        print(
            f"\nAnalisando {path} | Classe Predita: {pred_idx} | Confiança: {preds[0][pred_idx]:.2f}"
        )

        # Gera o Heatmap
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

        # Salva o resultado
        base_name = os.path.basename(path).split(".")[0]
        save_and_display_gradcam(path, heatmap, f"gradcam_{base_name}.jpg")
