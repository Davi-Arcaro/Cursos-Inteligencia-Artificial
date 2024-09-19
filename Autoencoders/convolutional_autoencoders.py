import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import InputLayer, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape

(X_treinamento, _), (X_teste, _) = mnist.load_data()
# print(X_treinamento.shape, X_teste.shape)
X_treinamento = X_treinamento.reshape(len(X_treinamento), 28, 28, 1)
X_teste = X_teste.reshape(len(X_teste), 28, 28, 1)
# print(X_treinamento.shape, X_teste.shape)

X_treinamento = X_treinamento.astype('float32') / 255
X_teste = X_teste.astype('float32') / 255

autoencoder = Sequential()
#Codificador
autoencoder.add(InputLayer(shape=(28,28,1)))
autoencoder.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
autoencoder.add(MaxPooling2D(pool_size=(2,2), padding='same'))

autoencoder.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same'))
autoencoder.add(MaxPooling2D(pool_size=(2,2), padding='same'))

autoencoder.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2)))
autoencoder.add(Flatten())

#Decodificador
autoencoder.add(Reshape((4,4,8)))
autoencoder.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D(size=(2,2)))

autoencoder.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D(size=(2,2)))

autoencoder.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
autoencoder.add(UpSampling2D(size=(2,2)))

autoencoder.add(Conv2D(filters=1, kernel_size=(3,3), activation='sigmoid', padding='same'))
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
autoencoder.fit(X_treinamento, X_treinamento, epochs=10, batch_size=256, validation_data=(X_teste, X_teste))

encoder = Model(inputs=autoencoder.get_layer('conv2d').input, outputs=autoencoder.get_layer('flatten').output)

imagens_codificadas = encoder.predict(X_teste)

imagens_decodificadas = autoencoder.predict(X_teste)
print(imagens_codificadas.shape, imagens_decodificadas.shape)

numero_imagens = 10
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