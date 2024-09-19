import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, InputLayer, Dense

(X_treinamento, _), (X_teste, _) = mnist.load_data() #Como o objetivo dos autoencoders é trabalhar diretamente com os pixels da imagem, não precisa de uma var que armazena as classes

# print(X_treinamento.shape, X_teste.shape)
# print(X_treinamento[0])

#Aplicando a normalização para os pixels, pois estão em números inteiros e precisam estar em decimais ('float32') depois, são divididos por 255 para ficarem entre 0 e 1.
X_treinamento = X_treinamento.astype('float32') / 255
X_teste = X_teste.astype('float32') / 255

# print(X_treinamento.shape, len(X_treinamento, np.prod(X_treinamento.shape[1:]))) np.prod é o produto, ou multiplicação
X_treinamento = X_treinamento.reshape((len(X_treinamento), np.prod(X_treinamento.shape[1:])))
X_teste = X_teste.reshape((len(X_teste), np.prod(X_teste.shape[1:])))

#Como entrada: 784 pixels -> 32 -> 784 #Reduz a dimensionalidade da imagem(codifica) e depois volta ao que era na entrada(decodifica)
autoencoder = Sequential()

#entrada
autoencoder.add(InputLayer(shape=(784,)))
#camada do meio
autoencoder.add(Dense(units=32, activation='relu'))
#processo de decodificação
autoencoder.add(Dense(units=784, activation='sigmoid'))

# autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

autoencoder.fit(X_treinamento, X_treinamento, epochs=50, batch_size=256, validation_data=(X_teste, X_teste)) #Como precisa fazer o comparativo entre a entrada e a entrada recriada,enviamos o X_treinamento duas vezes

#Fazendo a ligação de 784px com 32px da camada oculta
dimensao_original = Input(shape=(784,))
camada_encoder = autoencoder.layers[0] #Aqui temos a imagem compactada
encoder = Model(dimensao_original, camada_encoder(dimensao_original))
encoder.summary()

imagens_codificadas = encoder.predict(X_teste)

numero_imagens = 3
imagens_teste = np.random.randint(X_teste.shape[0], size=numero_imagens)

plt.figure(figsize=(18,18))
for i, indice_imagem in enumerate(imagens_teste):
    # print(i)
    # print(indice_imagem)

    #Imagem original:
    eixo = plt.subplot(10, 10, i+1)
    plt.imshow(X_teste[indice_imagem].reshape(28, 28)) #(28,28) pois 28*28 = 784
    plt.xticks()
    plt.yticks()
    plt.show()

    #Imagem codificada:
    eixo = plt.subplot(10, 10, i + 1 + numero_imagens)
    plt.imshow(imagens_codificadas[indice_imagem].reshape(8,4)) #(8,4 pois 8*4 = 32)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    
    #Imagem reconstruída:
    eixo = plt.subplot(10, 10, i + 1 + numero_imagens * 2)
    plt.imshow(imagens_codificadas[indice_imagem].reshape(28,28))
    plt.xticks(())
    plt.yticks(())
    plt.show()
    