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

# Transfomração de Potênciia ou Gamma
def gamma_correction(image, gamma):
    '''
    Aplica a correção gamma a uma imagem.

    Args:
        image (numpy.ndarray): A iamgem a ser corrigida. Pode ser uma imagem em escala de cinza ou colorida.
        gamma (float): O valor do parâmetro gama que controla a intesidade da correção.

    Returns:
        gamma_image (numpy.ndarray): A imagem após a aplicação da correção gama.
    '''
    # Converte a imagem para o tipo de dado float para realizar operações matemáticas.
    image = image.astype(float)

    # Normaliza os valores dos pixels para o intervalo [0, 1], aplica a correção gama e reescalona o intevalo [0, 255]
    # image/255 normaliza a imagem
    # pow((image/255), gamma) aplica a função porência com o valor gama.
    # Multiplica por 255 para reescalar os valores dos pixels para o intevalo [0, 255].
    gamma_image = pow((image/255), gamma) * 255

    # Converte a imagem corrigiida de volta para o tipo de dado uint8 (0-255).
    # Isso garante que os valores dos pixels estejam no intervalo correto para exibição.
    gamma_image = gamma_image.astype(np.uint8)

    # Retorna a imagem após a aplicação da correção gama.
    return gamma_image

# Histograma Cinza
def gray_histogram(image):
    '''
    Plota o histograma de instensidade em escala de cinza de uma imagem

    Args:
        image (numpy.ndarray): A imagem em escala de cinza para a qual o histograma será gerado. 
    '''
    # Define o número de bins (intervalos) para o histograma. Para imagens em escala de cinza, há 256 níveis de intensidade.
    num_bins = 256

    # Calcula o histograma da imagem usando a função cv2.calcHist do OpenCV
    # [image]: A imagem de entrada.
    # [0]: Canal de cor para imagens em escala de cinza é 0.
    # None: Máscara (não é usada aqui, então é None).
    # [num_bins]: Número de bins para o histograma.
    # [0, num_bins]: Intervalo dos níveis de intensidade (0, a 256)
    hist = cv2.calcHist([image], [0], None, [num_bins], [0, num_bins])

    # Cria uma nova figura com tamanho especificado.
    fig = plt.figure(figsize=(10,8))

    # Adiciona um subplot 3D à figura
    ax = fig.add_subplot(projection='3d')

    # Cria um array de índices para os bins do histograma.
    xs = np.arange(0, num_bins, 1)

    # Plota um gráfico de barras 3D do histograma.
    # xs: Posições ao longo do eixo x (níveis de intensidade).
    # hist.flatten(): Quantidade de picels para cada nível de intensidade (plano z).
    # zs=0: Posição ao longo do eixo z (aqui fixada em 0).
    # zdir='y': Direção do eixo z (aqui está configurado para o eixo y).
    # color='black': Cor das barras.
    # ec='black': Cor das bordas das barras.
    # alpha=0.8: Transparência das barras.
    ax.bar(xs, hist.flatten(), zs=0, zdir='y', color='black', ec='black', alpha=0.8)

    # Define o rótulo do eixo x.
    ax.set_xlabel('Níveis de intensidade')

    # Define as marcações e rótulos do eixo y.
    ax.set_yticks([0])
    ax.set_yticklabels(['Gray'])

    # Define o rótulo do eixo z e adiciona um espaçamento ao rótulo.
    ax.set_zlabel('Quantidade de pixels', labelpad=3)

    # Ajusta o espaçamento da subfigura manualmente para garantir que os rótulos não sejam cortados.
    fig.subplots_adjust(left=0, right=1, top=0.8, bottom=0.2)

    # Exibe o histograma.
    return fig

# Histograma Colorido
def color_histogram(image):
    '''
    Plota o Histograma de intensidade para cada canal de cor (R, G, B) de uma imagem colorida.

    Args:
        image (numpy.ndarray): A imagem coloria para a qual o histograma será gerado. Deve estar no formato BGR (como é padrão no OpenCV)
    '''
    # Define o número de bins (intervalos) para o histograma. Para imagens coloridas, há 256 níveis de intensiade por canal de cor.
    num_bins = 256

    # Calcula o hisrograma para cada canal de cor: R(vermelho), G(verde) e B(azul).
    # cv2.calcHist é usado para calcular o histograma dos canais individuais da imagem.
    # [0]: Canal vermelho
    hist_r = cv2.calcHist([image], [0], None, [num_bins], [0, num_bins])
    # [1]: Canal verde
    hist_g = cv2.calcHist([image], [1], None, [num_bins], [0, num_bins])
    # [2]: Canal azul
    hist_b = cv2.calcHist([image], [2], None, [num_bins], [0, num_bins])

    # Cria uma nova figura com tamanho especificado
    fig = plt.figure(figsize=(10,8))

    # Adiciona um subplot 3D à figura
    ax = fig.add_subplot(projection='3d')

    # Cria um array de índices para os bins do histograma
    xs = np.arange(0, num_bins, 1)

    # Plota o gráfico de barras 3D para cada canal de cor
    # Canal vermelho
    ax.bar(xs, hist_r.flatten(), zs=0, zdir='y', color='red', ec='red', alpha=0.8)
    # Canal verde
    ax.bar(xs, hist_g.flatten(), zs=10, zdir='y', color='green', ec='green', alpha=0.8)
    # Canal azul
    ax.bar(xs, hist_b.flatten(), zs=20, zdir='y', color='blue', ec='blue', alpha=0.8)

    # Define o rótulo do eixo x
    ax.set_xlabel("Níveis de intensidade")

    # Define as marcações e rótulos do eixo y para cada canal de cor
    ax.set_yticks([0, 10, 20])
    ax.set_yticklabels(['Red', 'Green', 'Blue'])

    # Define o rótulo do eixo z e aiciona um espaçamento ao rótulo
    ax.set_zlabel('Quantidade de pixels', labelpad=3)

    # Ajusta o espaçamento da subfigura manualmente para garantir que os rótulos não sejam cortados
    fig.subplots_adjust(left=0, right=1, top=0.8, bottom=0.2)

    # Exibe o histograma
    return fig

# Função para chamar mostrar os histogramas
def show_histogram(image):
    '''
    Exibe o histograma da imagem, diferenciando entre imagens coloridas e em escala de cinza

    Args:
        image (numpy.ndarray): A imagem para a qual o histograma será exibido. Pode ser uma imagem colorida ou em escala de cinza.
    '''
    # Verifica se a imagem é colorida.
    # Se o número de dimensões da imagem é maior que 2, isso indica que a imagem tem canais de cor (colorida).
    if len(image.shape) > 2:
        # Chama a função color_histogram para exibir o histograma dos canais de cor da imagem
        return color_histogram(image)
    else:
        # Cado contrário, a imagem é em escala de cinza.
        # Chama a função gray_histogram para exibir o histograma da imagem em escala de cinza.
        return gray_histogram(image)