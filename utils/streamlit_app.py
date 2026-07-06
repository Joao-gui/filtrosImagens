import streamlit as st
import cv2
import numpy as np

from utils.image_processing import *
from utils.aprimoramento import *

def run():
    st.set_page_config(
        page_title="Processamento de Imagens",
        layout="wide"
    )

    st.title("🖼️ Processamento de Imagens")

    arquivo = st.file_uploader(
        "Escolha uma imagem",
        type=["jpg", "jpeg", "png"]
    )

    if arquivo is None:
        return

    bytes_data = np.asarray(bytearray(arquivo.read()), dtype=np.uint8)

    image = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    imagem_original = image.copy()
    #resultado = image.copy()
    # ===========================
    # Barra lateral
    # ===========================

    st.sidebar.header("Redimensionamento")

    escala = st.sidebar.slider(
        "Escala da imagem (%)",
        min_value=10,
        max_value=200,
        value=100,
        step=10
    )

    # ===========================
    # Escala da imagem
    # ===========================
    image = resize_image(image, escala, escala)

    # ===========================
    # Aplicando ruído na imagem
    # ===========================
    st.sidebar.header("Ruído")

    usar_ruido = st.sidebar.checkbox(
        "Adicionar ruído 'Sal e Pimenta' na imagem"
    )

    if usar_ruido:
        image = salt_and_pepper_noise(image)
        imagem_ruido = image.copy()

    # ===========================
    # Sidebar dos filtros
    # ===========================
    st.sidebar.header("Filtros")

    filtro = st.sidebar.selectbox(
        "Escolha um Filtro",
        [
            "Original",
            "Cinza",
            "Média",
            "Gaussiano",
            "Mediana",
            "Sobel",
            "Laplaciano",
            "High Boost"
        ]
    )

    #resultado = image.copy()

    if filtro == "Cinza":
        image = greyscale_image(image)

    elif filtro == "Média":
        k = st.sidebar.slider("Kernel", 3, 15, 3, step=2)
        image = average_filter(image, k)

    elif filtro == "Gaussiano":
        k = st.sidebar.slider("Kernel", 3, 15, 3, step=2)
        image = gaussian_filter(image, k)

    elif filtro == "Mediana":
        k = st.sidebar.slider("Kernel", 3, 15, 3, step=2)
        image = median_filter(image, k)

    elif filtro == "Sobel":
        image = sobel_filter(image)

    elif filtro == "Laplaciano":
        image = laplacian_filter(image)

    elif filtro == "High Boost":
        a = st.sidebar.slider("A", 1, 10, 1)
        image = highboost_filter(image, a)


    # ===========================
    # Sidebar Aprimoramento da imagem
    # ===========================
    st.sidebar.header("Aprimoramento")

    aprimoramento = st.sidebar.selectbox(
        "Escolha um Aprimoramento",
        [
            "Original",
            "Negativo",
            "Transformação Logarítmica",
            "Transformação Gama"
        ]
    )

    #resultado = image.copy()

    if aprimoramento == "Negativo":
        image = negative_image(image)

    elif aprimoramento == "Transformação Logarítmica":
        image = log_transform(image)

    elif aprimoramento == "Transformação Gama":
        g = st.sidebar.slider("Gama", 0.0, 2.0, 1.0, step=0.1)
        image = gamma_correction(image, g)

    # ===========================
    # Aplicando histograma na imagem
    # ===========================
    st.sidebar.header("Histograma")

    ativar_histograma = st.sidebar.checkbox(
        "Mostrar Histograma da imagem"
    )

    histogram = None

    if ativar_histograma:
        histogram = show_histogram(image)

    # ===========================
    # Informações da imagem
    # ===========================

    st.write(f"**Dimensões da imagem:** {imagem_original.shape[1]} x {imagem_original.shape[0]} pixels")

    # ===========================
    # Exibição das imagens
    # ===========================

    # ===========================
    # Layout
    # ===========================
    if ativar_histograma:
        if usar_ruido:
            col1, col2, col3, col4 = st.columns(4)
        else:
            col1, col2, col3 = st.columns(3)

    else:
        if usar_ruido:
            col1, col2, col3 = st.columns(3)
        else:
            col1, col2 = st.columns(2)

    # ===========================
    # Imagem Original
    # ===========================
    with col1:

        st.subheader('Imagem Original')

        st.write(f'{imagem_original.shape[1]} x {imagem_original.shape[0]}')

        st.image(imagem_original)

    # ===========================
    # Imagem com Ruído
    # ===========================
    if usar_ruido:
        with col2:
            st.subheader('Imagem com Ruído')

            st.write(f'{imagem_ruido.shape[1]} x {imagem_ruido.shape[0]}')

            st.image(imagem_ruido)

        coluna_resultado = col3

        if ativar_histograma:
            coluna_hist = col4

    else:
        coluna_resultado = col2

        if ativar_histograma:
            coluna_hist = col3

    # ===========================
    # Resultado
    # ===========================
    with coluna_resultado:
        st.subheader("Resultado Final")

        st.write(f'{image.shape[1]} x {image.shape[0]}')

        if len(image.shape) == 2:
            st.image(image, clamp=True)
        else:
            st.image(image)

    # ===========================
    # Histograma
    # ===========================
    if ativar_histograma:
        with coluna_hist:
            st.subheader("Histograma")

            st.pyplot(histogram)