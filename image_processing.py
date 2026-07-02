import math
import numpy as np
import cv2

# Ler Imagem
def read_image(filename):
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    # Verifica se a imagem é colorida
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # OpenCV le imagem em BGR
    return image

# Redimensionar Imagem
def resize_image(image, width, height):
    # Verifica se a imagem é colorida, se sim pego linhas, colunas e canais
    if len(image.shape) > 2:
        rows, cols, channels = image.shape
    # Se for em escala de cinza, pego apenas linhas e colunas
    else:
        rows, cols = image.shape
    # Calcula a nova largura e arredonda para cima
    w = int(math.ceil(cols*width/100))
    # Calcula a nova altura e arredonda para cima
    h = int(math.ceil(rows*height/100))
    # Define o novo tamanho como uma tupla
    new_size = (w,h)
    # Redimensiona a imagem com base o novo tamanho
    image = cv2.resize(image, new_size)
    return image

# Converter para escala de cinza
def greyscale_image(image):
    # Verifica se a imagem é colorida
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

# Filtro de Média
    '''
    Os ﬁltros espaciais lineares de suavização também são chamados de ﬁltros de média: 
    - O resultado da convolução é simplesmente a média dos pixels correspondentes à máscara.
    - Quanto maior a máscara maior o efeito blur.
    '''
def average_filter(image, kernel_size): # Carrega a imagem e o tamanho do kernel para a mascara do filtro (3x3, 5x5... sempre ímpar)
    image = cv2.blur(image, (kernel_size, kernel_size)) # Método blur do kernel tem q ser passado em forma de tupla
    return image

# Filtro Gaussiano
    '''
    Utiliza uma máscara cujos coeﬁcientes são obtidos a partir de uma aproximação discreta da função Gaussiana bidimensional.
        O resultado da convolução do filtro de suavização Gaussiano é uma média ponderada, onde:
        - Seu grau de suavização está relacionado com o parâmetro 𝜎.
            - Quanto maior o valor de 𝜎 maior será a suavização da imagem.
        - O valor do coeficiente central da máscara é maior que o valor dos seus vizinhos, 
        cujos valores são reduzidos em função do aumento da distância do coeficiente central, isso permite:
            - Uma redução do borramento (blur) no processo de suavização.
            - A obtenção de uma suavização mais sutil/delicada que o filtro da média aritmética simples.
    '''
def gaussian_filter(image, kernel__size):
    standard_deviation = 0 # Usa o método cv2.GaussianBlur, quando passamos desvio padrão = 0, ela calcula o desvio padrão automaticamente baseada no Kernel
    image = cv2.GaussianBlur(image, (kernel__size, kernel__size), standard_deviation)
    return image

# Filtro Mediana
    '''
    Filtro não linear.
    - Substitui o valor do pixel central pela mediana dos valores na vizinhança do pixel central (inclui o pixel central).
        - Ordena-se os valores dos pixels cobertos pela máscara e obtém-se o valor da mediana.
    - Eﬁciente para remover ruídos impulsivos (como sal e pimenta).
    - Mantém bordas e detalhes importantes.
        - Produzindo um borramento ou blur reduzido em relação aos ﬁltros lineares (média) utilizando máscaras de tamanho similar.
    '''
def median_filter(image, kerne_size):
    image = cv2.medianBlur(image, kerne_size) # Kernel_size não precisa ser em tupla
    return image

# Ruído sal e pimenta
    '''
    Para simular uma imagem com ruído, foi feito esta função.
    '''
def salt_and_pepper_noise(image):
    image_noisy = image.copy()
    if len(image.shape) > 2: # Se a imagem for colorida
        rows, cols, channels = image_noisy.shape
    else:
        rows, cols = image_noisy.shape
    noise = np.zeros((rows,cols), np.uint8)
    cv2.randu(noise,0,255)
    image_noisy[noise <= 5] = 0
    image_noisy[noise >= 250] = 255
    return image_noisy

# Filtro de Sobel
    '''
    - É utilizado para detecção de bordas horizontais e verticais em uma imagem.
    - O ﬁltro de Sobel utiliza a operação de convolução utilizando dois kernels:
        - Um kernel de Sobel para detectar bordas horizontais:
        [-1 -2 -1]
        [ 0  0  0]
        [ 1  2  1]
        - Um kernel de Sobel para detectar bordas verticais:
        [-1  0  1]
        [-2  0  2]
        [-1  0  1]
    - A convolução com 2 kernels resulta em duas imagens chamadas de gradiente,
    uma representando as bordas horizontais e outra as bordas verticais.
    - A magnitude do gradiente, calculada como a combinação das duas imagens
    gradiente, é frequentemente usada para identiﬁcar bordas fortes na imagem.
        - A imagem resultante pode ser mesclada com a imagem de entrada para realçar as bordas existentes.
        - As bordas detectadas podem ser utilizadas para segmentação de objetos de interesse.
    '''
def sobel_filter(image):
    # Verifica se a imagem é colorida
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Se ja for cinza
    else:
        gray_image = image
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 0,1, ksize=3) # O 0,1 é para detectar bordas 0=X e 1=Y
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 1,0, ksize=3) # O 1,0 é para detectar bordas 1=X e 0=Y
    # Combinando 2 imagens
    gradient_magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    # Faz com que os valores fiquem entre 0 e 255
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    # Converter a imagem de float 64 bits para int
    edge_image = np.uint8(gradient_magnitude)
    # Verifica se a imagem é colorida
    if len(image.shape) > 2:
        edge_image = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2RGB) # Transforma imagem de escala de cinza 2 canais para 3 canais.
    imagem_add = cv2.add(image, edge_image) # Métoddo do OpenCV para fazer soma das 2 imagens
    return imagem_add

# Filtro Laplacianlo (Gera ruído, recomendado usar algum filtro antes do laplaciano)
    '''
    - O ﬁltro de realce laplaciano realiza a operação de convolução em uma imagem de
    entrada utilizando um kernel Laplaciano.
    - O kernel Laplaciano é uma máscara de convolução que realça mudanças abruptas na
    intensidade dos pixels.
    - A aplicação do ﬁltro de realce laplaciano resulta em uma imagem que destaca as bordas
    e os detalhes ﬁnos, aumentando a nitidez da imagem.
    - O kernel de Laplaciano comumente usado para realce é:
        [ 0 -1  0]
        [-1  4 -1]
        [ 0 -1  0]
    '''
def laplacian_filter(image):
    # Aplicando o filtro da mediana antes
    image = median_filter(image, 3)
    # Passando o valor e os coeficientes do Kernel
    kernel = np.array([[0,-1,0],
                    [-1,4,-1],
                    [0,-1,0]])
    laplacian = cv2.filter2D(image, -1, kernel) # Realiza uma concolução, -1 -> a profundidade da saída será a mesma da entrada, mesma quantidade de bits
    image_add = cv2.add(image, laplacian)
    return image_add

# Filtro High Boost (variação do filtro laplaciano)
def highboost_filter(image,a):
    # Aplicando o filtro da mediana
    image = median_filter(image, 3)
    # Passando o valor e os coeficientes do Kernel
    kernel = np.array([[0,-1,0],
                    [-1,4+a,-1],
                    [0,-1,0]])
    highboost = cv2.filter2D(image, -1, kernel)
    image_add = cv2.add(image, highboost)
    return image_add