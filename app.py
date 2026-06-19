"""
Classificador de Danos em Folhas de Soja - Streamlit App

IMPORTANTE - Configuração necessária no model_params.yaml:
  model:
    normalize_input: false   (para MobileNetV3, EfficientNetV2, ViT)
    normalize_input: true    (para InceptionV3, VGG19, NASNet, ConvNeXt)

A normalização é feita automaticamente via prepare_model.py
Modelos com preprocessing nativo não devem ser normalizados aqui.
"""

import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import re
import ast
import time
import os


@tf.keras.utils.register_keras_serializable(package="Custom", name="TopKGlobalAveragePooling2D")
class TopKGlobalAveragePooling2D(tf.keras.layers.Layer):
    def __init__(self, k_percent=0.1, **kwargs):
        super(TopKGlobalAveragePooling2D, self).__init__(**kwargs)
        self.k_percent = k_percent

    def call(self, inputs):
        shape = tf.shape(inputs)
        b, h, w, c = shape[0], shape[1], shape[2], shape[3]
        flattened = tf.reshape(inputs, [b, h * w, c])
        k = tf.cast(tf.cast(h * w, tf.float32) * self.k_percent, tf.int32)
        top_k_values, _ = tf.math.top_k(flattened, k=k, sorted=False)
        return tf.reduce_mean(top_k_values, axis=1)

    def get_config(self):
        config = super(TopKGlobalAveragePooling2D, self).get_config()
        config.update({"k_percent": self.k_percent})
        return config


@st.cache_resource
def carrega_modelo(path_url, formato_modelo):
    url = converte_url_drive(path_url)

    if url is None:
        raise ValueError("❌ URL do Google Drive inválida. Verifique o link.")

    if formato_modelo == "tflite":
        filename = "modelo_temp.tflite"
        # Remove arquivo antigo se existir
        if os.path.exists(filename):
            os.remove(filename)
        try:
            gdown.download(url, filename, quiet=False)
        except Exception as e:
            raise ValueError(f"Erro ao baixar modelo TFLite: {str(e)}")
        interpreter = tf.lite.Interpreter(model_path=filename)
        interpreter.allocate_tensors()
        return {"tipo": "tflite", "modelo": interpreter}

    elif formato_modelo in ["keras", "h5"]:
        filename = f"modelo_temp.{formato_modelo}"
        # Remove arquivo antigo se existir
        if os.path.exists(filename):
            os.remove(filename)
        try:
            gdown.download(url, filename, quiet=False)
        except Exception as e:
            raise ValueError(f"Erro ao baixar modelo Keras: {str(e)}")
        # Registrar preprocess_input globalmente com base na URL para evitar erro de desserialização
        url_lower = str(path_url).lower()
        if "inception" in url_lower:
            func = tf.keras.applications.inception_v3.preprocess_input
        elif "vgg" in url_lower:
            func = tf.keras.applications.vgg19.preprocess_input
        elif "nasnet" in url_lower:
            func = tf.keras.applications.nasnet.preprocess_input
        elif "efficientnet" in url_lower:
            func = tf.keras.applications.efficientnet.preprocess_input
        elif "convnext" in url_lower:
            func = tf.keras.applications.convnext.preprocess_input
        else:
            func = tf.keras.applications.mobilenet_v3.preprocess_input
            
        tf.keras.utils.get_custom_objects()["preprocess_input"] = func

        model = tf.keras.models.load_model(filename)
        return {"tipo": "keras", "modelo": model}

    else:
        raise ValueError(
            f"Formato '{formato_modelo}' não suportado. Use 'tflite', 'keras' ou 'h5'."
        )


def carrega_imagem():
    uploaded_file = st.file_uploader(
        "Arraste e solte uma imagem ou clique para selecionar uma",
        type=["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"],
    )

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Imagem Original", use_container_width=True)

        TENSOR_SIZE = 224
        image_resized = image.resize((TENSOR_SIZE, TENSOR_SIZE))

        with col2:
            st.image(
                image_resized,
                caption="Imagem Redimensionada (224x224)",
                use_container_width=True,
            )

        st.success("✅ Imagem carregada com sucesso")

        image_converted = image_resized.convert("RGB")
        image_array = np.array(image_converted, dtype=np.float32)
        # NÃO normalizar aqui! O modelo faz preprocessing internamente ou está configurado no prepare_model.py
        image_array = np.expand_dims(image_array, axis=0)

        return image_array


def converte_url_drive(url_compartilhamento):
    """
    Extrai o ID do arquivo do link de compartilhamento do Google Drive
    e o converte para o formato de download direto.
    """
    # Expressão regular para encontrar o ID. Ela busca o ID que vem depois de '/d/' ou 'id='
    match = re.search(r"id=([a-zA-Z0-9_-]+)|/d/([a-zA-Z0-9_-]+)", url_compartilhamento)

    if match:
        # match.group(1) captura o ID se for encontrado após 'id='
        # match.group(2) captura o ID se for encontrado após '/d/'
        file_id = match.group(1) if match.group(1) else match.group(2)

        # Constrói a URL de download no formato desejado
        url_download_direto = f"https://drive.google.com/uc?id={file_id}"
        return url_download_direto
    else:
        return None


def previsao(modelo_dict, image, classes):
    tipo_modelo = modelo_dict["tipo"]
    modelo = modelo_dict["modelo"]

    # Medir tempo de inferência
    tempo_inicio = time.time()

    if tipo_modelo == "tflite":
        input_details = modelo.get_input_details()
        output_details = modelo.get_output_details()
        modelo.set_tensor(input_details[0]["index"], image)
        modelo.invoke()
        output_data = modelo.get_tensor(output_details[0]["index"])
    
    elif tipo_modelo == "keras":
        output_data = modelo.predict(image, verbose=0)

    tempo_fim = time.time()
    tempo_inferencia_ms = (tempo_fim - tempo_inicio) * 1000

    df = pd.DataFrame()
    df["classes"] = classes

    probabilidades = np.squeeze(output_data)
    df["probabilidades (%)"] = 100 * probabilidades

    # Encontrar classe com maior probabilidade
    classe_predita = df.loc[df["probabilidades (%)"].idxmax(), "classes"]
    confianca_max = df["probabilidades (%)"].max()

    fig = px.bar(
        df,
        y="classes",
        x="probabilidades (%)",
        orientation="h",
        text="probabilidades (%)",
        title="Classificação de Danos em Folhas de Soja",
    )
    st.plotly_chart(fig)

    # Exibir informações de desempenho
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="⚡ Tempo de Inferência", value=f"{tempo_inferencia_ms:.2f} ms")

    with col2:
        st.metric(label="🎯 Classe Predita", value=classe_predita)

    with col3:
        st.metric(label="📊 Confiança", value=f"{confianca_max:.2f}%")


def main():
    st.set_page_config(
        page_title="Classifica folha de Soja",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🌱 Classificador de Danos em Folhas de Soja")
    st.markdown("---")

    # Sidebar para configurações
    with st.sidebar:
        st.header("⚙️ Configurações")
        st.markdown("### Modelo")

        formato_modelo = st.selectbox(
            "Formato do modelo:",
            options=["tflite", "keras", "h5"],
            help="Selecione o formato do modelo que você vai usar",
        )

        model_url_input = st.text_input(
            "Cole a URL do modelo (Google Drive ou Link Direto):",
            value="",
            help=f"Use um modelo em formato .{formato_modelo}",
        )

        st.markdown("### Classes")
        classes_input_string = st.text_area(
            "Lista de classes:",
            value="['Caterpillar', 'Diabrotica speciosa', 'Healthy']",
            height=100,
            help="Formato: ['classe1', 'classe2', 'classe3']",
        )

    # Validação e carregamento do modelo
    modelo_dict = None
    classes_list = None

    if classes_input_string:
        try:
            classes_list = ast.literal_eval(classes_input_string)
            if not isinstance(classes_list, list):
                st.error(
                    "❌ Erro na lista de classes: A entrada não é uma lista válida."
                )
                classes_list = None
        except Exception:
            st.error("❌ Erro na lista de classes: Verifique a sintaxe.")
            classes_list = None

    if model_url_input and classes_list is not None:
        try:
            with st.spinner("⏳ Carregando modelo..."):
                modelo_dict = carrega_modelo(model_url_input, formato_modelo)
            st.success(f"✅ Modelo {formato_modelo.upper()} carregado com sucesso!")
        except Exception as e:
            st.error(f"❌ Erro ao carregar o modelo: {str(e)}")
            modelo_dict = None
    elif model_url_input or classes_list is not None:
        st.info("ℹ️ Preencha os campos de URL do modelo e classes para começar")

    # Seção de classificação
    if modelo_dict is not None:
        st.markdown("---")
        st.header("📸 Classificação de Imagem")

        col_upload, col_result = st.columns([1, 1])

        with col_upload:
            st.subheader("Carregue uma imagem")
            image = carrega_imagem()

        if image is not None:
            with col_result:
                st.subheader("Resultados")
                with st.spinner("🔍 Analisando imagem..."):
                    previsao(modelo_dict, image, classes_list)


if __name__ == "__main__":
    main()
