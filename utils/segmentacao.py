from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops
from skimage.morphology import dilation, disk, remove_small_objects, label
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import cv2

def color_segmentation(_):
    pass

# Limiarização (Thresholding)
def threshold(image, t):
    '''
    Divide a imagem em regiões de acordo com o valor de intensidade de cada pizel em comparação com um limiar (t).
    
    Args:
        image (numpy.ndarray): A imagem para qual será feito aplicação de segmentação por Limiarização.
        t (int): Valor inteiro do limiar.

    Returns:
        result (numpy.ndarray): A imagem resultando após a segmentação.
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