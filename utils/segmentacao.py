from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops
from skimage.morphology import dilation, disk, remove_small_objects, label
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Segmentação por cor
def color_segmentation(image, h_min, h_max, s_min, s_max, v_min, v_max):
    '''
    Aplica segmentação de uma imagem apartir de uma cor específica.

    Args:
        image (numpy.ndarray): A imagem para qual será feito aplicação de segmentação por cor.
        h_min (int): valor mínimo de h
        h_max (int): valor máximo de h
        s_min (int): valor mínimo de s
        s_max (int): valor máximo de s
        v_min (int): valor mínimo de v
        v_max (int): valor máximo de v

    Returns:
        result (numpy.ndarray): A imagem resultada após a segmentação.
        mask (numpy.ndarray): A mascara de aplicação para a segmentação da imagem.
    '''
    # Define os limites inferiores e superiores para a segmentação de cores no espaço dee cores HSV
    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])

    # Verifica se a imagem é colorida (tem mais de 2 dimensões)
    # Se a imagem for em escala de cinza não faz nada
    if (len(image.shape) > 2):
        # Aplica um filtro de mediana para suavizar a imagem e reduzir o ruído
        blur_image = cv2.medianBlur(image, 5)

        # Converte a imagem do espaço de cores RGB para HSV
        hsv = cv2.cvtColor(blur_image, cv2.COLOR_RGB2HSV)

        # Cria uma máscara binária onde os pixels dentro do intervalo especificado são brancos (255) e o resto é preto (0)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Preenchee buracos na máscara binária para garantir que as áreas segmentadas sejam contínuas
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

        # Correção para mostrar no streamlit para sair do 0 ou 1 e mostrar 0 ou 255
        mask *= 255

        # Aplica a máscara à imagem original, mantendo apenas os pixels dentro do
        # intervalo de cores especificado
        # A máscara é uma imagem binária (contendo valores 0 a 255).
        # A opreação AND será aplicada apenas onde a máscara tem valor 255 (branco),
        # enquanto os valores 0 (preto) na máscara irão resultar em zeros na iamgme resultante.
        result = cv2.bitwise_and(image, image, mask=mask)

        # Retorna a imagm segmentada e a máscara binária
        return result, mask

# Limiarização (Thresholding)
def threshold(image, t):
    '''
    Divide a imagem em regiões de acordo com o valor de intensidade de cada pixel em comparação com um limiar (t).
    
    Args:
        image (numpy.ndarray): A imagem para qual será feito aplicação de segmentação por Limiarização.
        t (int): Valor inteiro do limiar.

    Returns:
        result (numpy.ndarray): A imagem resultada após a segmentação.
        mask (numpy.ndarray): A mascara de aplicação para a segmentação da imagem.
    '''
    # Verifica se a imagem é colorida (tem mais de 2 dimensões)
    if (len(image.shape) > 2):
        # Converte a imagem de RGB para escala de cinza
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        # Se a imagem já estiver em escala de cinza, a usa diretamente
        gray_image = image

    # Aplica um filtro de mediana para suavizar a imagem e reduzir ruído
    blur_image = cv2.medianBlur(gray_image, 5)

    # Aplica um limiar binário inverso na imagem suavizada
    # Os pixels acima do limiar 't' são definidos como 0 (preto)
    # Os pixels abaixo do limiar 't' são definidos como 255 (branco)
    t, mask = cv2.threshold(blur_image, t, 255, cv2.THRESH_BINARY_INV)

    # Preenche buracos na máscara binária para garantir que as áreas segmentadas sejam contínuas
    mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

    # Correção para mostrar no streamlit para sair do 0 ou 1 e mostrar 0 ou 255
    mask *= 255

    # Aplica a máscara à imagem original, mantendo apenas os pixels dentro do
    # intervalo de cores especificado.
    # A máscara é uma imagem binária (contendo valores 0 a 255).
    # A opreação AND será aplicada apenas onde a máscara tem valor 255 (branco),
    # enquanto os valores 0 (preto) na máscara irão resultar em zeros na imagem resultante.
    result = cv2.bitwise_and(image, image, mask=mask)

    # Retorna a imagem segmentada e a máscara binária
    return result, mask

# Limiarizzação (Método de Otsu)
def otsu_threshold(image):
    '''
    Parecido com o método de Thresholding apenas diferenciando que o limiar (t) é calculado automatticamente a partir do histograma da imagem.

    Args:
        image (numpy.ndarray): A imagem para qual será feito aplicação de segmentação por Otsu.

    Returns:
        result (numpy.ndarray): A imagem resultando após a segmentação.
        mask (numpy.ndarray): A mascara de aplicação para a segmentação da imagem.
        t (int): Valor do limiar cálculado pelo método de Otsu.
    '''
    # Verifica se a imagem é colorida (tem mais de 2 dimensões)
    if (len(image.shape) > 2):
        # Converte a imagem de RGB para escala de cinza
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        # Se a imagem já estiver em escala de cinza, a usa diretamente
        gray_image = image

    # Aplica um filtro de mediana para suavizar a iamgem e reduzir o ruído
    blur_image = cv2.medianBlur(gray_image, 5)

    # Aplica o limiar de Otsu na imagem suavizada com limiar binário inverso
    # O limiar de Otsu dtermina automaticamente o valor de limiar 't'
    # Pixels acima do limiar 't' são definidos como 0 (preto)
    # Pixels abaixo do limiar 't' são definidos como 255 (branco)
    t, mask = cv2.threshold(blur_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Preenche buracos na máscara binária para garantir quee as áreas segmentadas sejam contínuas
    mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

    # Correção para mostrar no streamlit para sair do 0 ou 1 e mostrar 0 ou 255
    mask *= 255

    # Aplica a máscara à imagem original, mantendo apenas os pixels dentro do
    # intervalo de cores especificado.
    # A máscara é uma imagem binária (contendo valores 0 e 255).
    # A operalção AND será aplicada apenas onde a máscara tem o valor 255 (branco),
    # enquanto os valore 0 (preto) na máscara irão resultar em zeros na imagem resultante.
    result = cv2.bitwise_and(image, image, mask=mask)

    # Retorna a iamgem segmentada, a máscara binária e o valor de limiar determinado por Otsu.
    return result, mask, t

# Detecção de Bordas - Método de Canny
def canny(image, lower_thresh_rate):
    '''
    É um dos principais métodos de detecção de bordas por permitir a detecção de bordas em toda a imagem,
    incluindo regiões de baixo contraste.

    Args:
        image (numpy.ndarray): A imagem para qual será feito aplicação de Canny.
        lower_thresh_rate (float): Proporção do limiar inferior

    Returns:
        result (numpy.ndarray): A imagem resultando após a segmentação.
        edges (numpy.ndarray): Bordas da imagem
        num_coutours (int): Número de contornos encontrados
    '''
    # Verifica se a imagem é colorida (tem mais de 2 dimensões)
    if(len(image.shape) > 2):
        # Converte a imagem de RGB para escala de cinza
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        # Se a imagem já estiver em escala de cinza, a usa diretamente
        gray_image = image

    # Aplica um filtro de mediana para suavizar a imagem e reduzir o ruído
    gray_image = cv2.medianBlur(gray_image, 3)

    # Aplica o limiar de Otsu na imagem suavizada com limiar binário inverso
    # O limiar de Otsu determina automaticamente o valor de limiar 't'
    # Pixels acima do limiar 't' são definidos como 0 (preto)
    # Pixels abaixo do limiar 't' são definidos como 255 (branco)
    t, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Calcula o limiar inferior como uma proporção do limiar superior determinado por Otsu
    lower = int(t * lower_thresh_rate)
    upper = t

    # Aplica o detector de bordas Canny usando os limiares inferiore e superior
    edges = cv2.Canny(gray_image, lower, upper)

    # Encontra os contornos nas bordas detectadas
    # cv2.findContours retorna uma lista de contornos encontrados e a
    # hierarquia dos contornos.
    # O modo cv2.RETR_EXTERNAL recupera apenas os contornos externos
    # O método cv2.CHAIN_APPROX_SIMPLE comprime segmentos horizontais, verticais e diagonais
    # e deixa apenas seus pontos finais.
    # h = hierarquia dos contornos
    (contours, h) = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Desenha os contornos encontrados na imagem original, colorindo-os de verde e
    # com espessura de 3 pixels
    # -1: Indica que todos os contornos da lista contours devem ser desenhados.
    result = cv2.drawContours(image.copy(), contours, -1, color=(0, 255, 0), thickness=3)

    # Conta o número de contornos encontrados
    num_contours = len(contours)

    # Retorna a imagem com os contornos desenhados, a imagem de bordas e o número dee contornos
    return result, edges, num_contours