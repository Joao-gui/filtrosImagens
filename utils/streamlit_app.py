import streamlit as st
import cv2
import numpy as np

from utils.image_processing import *
from utils.aprimoramento import *
from utils.segmentacao import *

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
            "Transformação Gama",
            "Ajuste de Contraste"
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

    elif aprimoramento == "Ajuste de Contraste":
        min_value = st.sidebar.slider("Mín", 0, 255, 0, step=1)
        max_value = st.sidebar.slider("Máx", 0, 255, 255, step=1)
        image = contrast_stretch(image, max_value, min_value)

    # ===========================
    # Sidebar Segmentação da imagem
    # ===========================
    st.sidebar.header("Segmentação")

    segmentacao = st.sidebar.selectbox(
        "Tipos de segmentação da imagem",
        [
            "Original",
            "Limiarização (Thresholding)",
            "Limiarização (Método Otsu)",
            "Segmentação por Cor (HSV)",
            "Detecção de bordas (Canny)"
        ]
    )

    # Mascara da segmentação
    mask = None

    # Valor limiar usado no Thresholding e Otsu
    t = None

    if segmentacao == 'Limiarização (Thresholding)':
        t = st.sidebar.slider("Limiar", 0, 255, 125, step=1)
        image, mask = threshold(image, t)

    elif segmentacao == 'Limiarização (Método Otsu)':
        image, mask, t = otsu_threshold(image)

    elif segmentacao == 'Segmentação por Cor (HSV)':
        st.sidebar.markdown('### 🎨 Cor de Referência')

        cor = st.sidebar.color_picker("Escolha uma cor", "#ff0000")

        # RGB -> HSV
        rgb = tuple(
            int(cor[i:i+2], 16)
            for i in (1, 3, 5)
        )

        hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]

        h = int(hsv[0])
        s = int(hsv[1])
        v = int(hsv[2])

        # Intervaço automático
        h_min = max(0, h)
        h_max = min(179, h)

        s_min = max(0, s - 60)
        s_max = 255

        v_min = max(0, v - 60)
        v_max = 255

        # Configuração avançada
        with st.sidebar.expander("Configuração Avançada (HSV)"):
            h_min = st.slider("H min", 0, 179, h_min)
            h_max = st.slider("H max", 0, 179, h_max)

            s_min = st.slider("S min", 0, 255, s_min)
            s_max = st.slider("S max", 0, 255, s_max)

            v_min = st.slider("V min", 0, 255, v_min)
            v_max = st.slider("V max", 0, 255, v_max)

        image, mask = color_segmentation(
            image,
            h_min,
            h_max,
            s_min,
            s_max,
            v_min,
            v_max
        )

    elif segmentacao == 'Detecção de bordas (Canny)':
        lower_thresh_rate = st.sidebar.slider("Proporção do Limiar inferior", 0.0, 1.0, 0.5, step=0.1)
        image, mask, num_counters = canny(image, lower_thresh_rate)
        
    # ===========================
    # Aplicando Equalização de Histograma
    # ===========================
    st.sidebar.header("Equalização de Histograma")

    ativar_equalizacao_histograma = st.sidebar.checkbox(
        "Fazer a equalização do histograma da imagem"
    )

    if ativar_equalizacao_histograma:
        image = histogram_equalization(image)

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

    st.write(f"**Dimensões da imagem importada:** {imagem_original.shape[1]} x {imagem_original.shape[0]} pixels")

    # ===========================
    # Exibição das imagens
    # ===========================

    # ===========================
    # Layout
    # ===========================
    num_colunas = 2     # Original + Resultado

    if usar_ruido:
        num_colunas += 1

    if mask is not None:
        num_colunas += 1

    if ativar_histograma:
        num_colunas += 1

    colunas = st.columns(num_colunas)

    indice = 0

    col_original = colunas[indice]
    indice += 1

    if usar_ruido:
        col_ruido = colunas[indice]
        indice += 1

    col_resultado = colunas[indice]
    indice += 1

    if mask is not None:
        col_mask = colunas[indice]
        indice += 1

    if ativar_histograma:
        col_hist = colunas[indice]

    # ===========================
    # Imagem Original
    # ===========================
    with col_original:

        st.subheader('Imagem Original')

        st.write(f'{imagem_original.shape[1]} x {imagem_original.shape[0]}')

        st.image(imagem_original)

    # ===========================
    # Imagem com Ruído
    # ===========================
    if usar_ruido:
        with col_ruido:
            st.subheader('Imagem com Ruído')

            st.write(f'{imagem_ruido.shape[1]} x {imagem_ruido.shape[0]}')

            st.image(imagem_ruido)

    # ===========================
    # Resultado
    # ===========================
    with col_resultado:
        st.subheader("Resultado Final")

        st.write(f'{image.shape[1]} x {image.shape[0]}')

        if len(image.shape) == 2:
            st.image(image, clamp=True)
        else:
            st.image(image)

    # ===========================
    # Máscara
    # ===========================
    if mask is not None:
        if segmentacao == 'Detecção de bordas (Canny)':
            with col_mask:
                st.subheader("Máscara")

                st.write(f'**Número de regiões =** {num_counters}')

                st.image(mask, clamp=True)

        else:
            with col_mask:
                st.subheader("Máscara")

                st.write(f"**Valor do limiar aplicado:** {t}")

                st.image(mask, clamp=True)

    # ===========================
    # Histograma
    # ===========================
    if ativar_histograma:
        with col_hist:
            st.subheader("Histograma")

            st.write('Gráfico 3D')

            st.pyplot(histogram)