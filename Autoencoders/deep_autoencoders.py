import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import InputLayer, Input, Dense

(X_treinamento, _), (X_teste, _) = mnist.load_data()
#Fazendo o tratamento da base de dados
X_treinamento = X_treinamento.astype('float32') / 255
X_teste = X_teste.astype('float32') / 255
X_treinamento = X_treinamento.reshape((len(X_treinamento), np.prod(X_treinamento.shape[1:])))
X_teste = X_teste.reshape((len(X_teste), np.prod(X_teste.shape[1:])))

#Criação do autoencoder
#Estrutura: 784 -> 128 -> 64 -> 32 -> 64 -> 128 -> 784
autoencoder = Sequential()
autoencoder.add(InputLayer(shape=(784,)))
autoencoder.add(Dense(units=128, activation='relu'))
autoencoder.add(Dense(units=64, activation='relu'))
autoencoder.add(Dense(units=32, activation='relu'))
#Decodificador
autoencoder.add(Dense(units=64, activation='relu'))
autoencoder.add(Dense(units=128, activation='relu'))
autoencoder.add(Dense(units=784, activation='sigmoid'))

autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#autoencoder.fit(X_treinamento, X_treinamento, epochs=50, batch_size=256, validation_data=(X_teste, X_teste))

#784 -> 128 -> 64 -> 32
dimensao_original = Input(shape=(784,))
camada_encoder1 = autoencoder.layers[0] #conversão para 128px
camada_encoder2 = autoencoder.layers[1] #conversão para 64px
camada_encoder3 = autoencoder.layers[2] #conversão para 32px
encoder = Model(dimensao_original, camada_encoder3(camada_encoder2(camada_encoder1(dimensao_original)))) #conexão das camadas
# encoder.summary()
imagens_codificadas = encoder.predict(X_teste)

numero_imagens = 10
imagens_teste = np.random.randint(X_teste.shape[0], size=numero_imagens)
imagens_decodificadas = autoencoder.predict(X_teste)
plt.figure(figsize=(18,18))
for i, indice_imagem in enumerate(imagens_teste):
    eixo = plt.subplot(10, 10, i + 1)
    plt.imshow(X_teste[indice_imagem].reshape(28, 28))
    plt.xticks(())
    plt.yticks(())
