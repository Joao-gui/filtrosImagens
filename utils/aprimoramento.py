import cv2
import numpy as np
import matplotlib.pyplot as plt

# Filtro Negativo
def negative_image(image):
    '''
    Converte uma imagem para seu negativo.

    Args:
        image (numpy.ndarray): A imagem a ser convertida. Pode ser uma imagem em escala de cinza ou colorida.

    Returns:
        negative (numpy.ndarray): A imagem negativa resultande da conversão.
    '''
    # Calcula o negativo da imagem subtraindo cada valor de pixel de 255.
    # Para imagens em escala de cinza, o valor de cada pixel é invertido.
    # Para imagens coloridas, o mesmo processo é aplicado a cada canal de cor.
    negative = 255 - image

    # Retorna a iamgem negativa
    return negative

# Transformação Logarítmica
def log_transform(image):
    '''
    Aplica uma transformação loratítmica a uma imagem.

    Args:
        image (numpy.ndarray): A imagem a ser transformada. Pode ser uma imagem em escala de cinza ou colorida

    Returns:
        log_image (numpy.ndarray): A imagem após a transformação logarítmica
    '''
    # Converte a imagem para o tipo de dado float para realizar operações matemáticas
    image = image.astype(float)

    # Calcula o valor de c usando o valor máximo da imagem.
    # `np.log(1 + np.max(image))` é o valor máximo no domínio logarítmico.
    # O fator de escala c é ajustado para que o valor máximo da imagem transformada seja 255.
    c = 255 / np.log(1 + np.max(image))

    # Aplica a transformação logarítmica a cada pixel da imagem.
    # `np.log(1 + imagem)` realiza a transformação logatítmica para cada pixel.
    # Multiplica o resultado pelo fator de escala c.
    log_image = c * np.log(1 + image)

    # Converte a imagem trsnformada de volta para o tipo de dado uint8 (0-255).
    # Isso garante que os valores dos pixels estejjam no intervalo correto para exibição.
    log_image = log_image.astype(np.uint8)

    # Retorna a imagem transformada
    return log_image