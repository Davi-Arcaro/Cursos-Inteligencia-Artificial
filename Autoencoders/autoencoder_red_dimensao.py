import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import InputLayer, Input, Dense
from keras.utils import to_categorical

(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()

#Tratamento da base de dados, para ficarem na escala enre 0 e 1
X_treinamento = X_treinamento.astype('float32') / 255
X_teste = X_teste.astype('float32') / 255

#Passando as classes para o formato OneHotEncoder
y_treinamento = to_categorical(y_treinamento)
y_teste = to_categorical(y_teste)

# print(X_treinamento.shape)
X_treinamento = X_treinamento.reshape((len(X_treinamento), np.prod(X_treinamento.shape[1:])))
X_teste = X_teste.reshape((len(X_teste), np.prod(X_teste.shape[1:])))
# print(X_treinamento.shape, X_teste.shape)

#Criação do autoencoder
autoencoder = Sequential()
autoencoder.add(InputLayer(shape=(784,)))
autoencoder.add(Dense(units=32, activation='relu'))
autoencoder.add(Dense(units=784, activation='sigmoid'))
# autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
autoencoder.fit(X_treinamento, X_treinamento, epochs=50, batch_size=256, validation_data= (X_teste,X_teste))

dimensao_original = Input(shape=(784,))
camada_encoder = autoencoder.layers[0]
encoder = Model(dimensao_original, camada_encoder(dimensao_original))

treinamento_codificado = encoder.predict(X_treinamento)
teste_codificado = encoder.predict(X_teste)

#Criando classificadores. (Sem redução de dimensionalidade)
#Nesse classificador tem-se 784n -> 397 -> 397 -> 10
#Neurônios nas camadas ocultas: (784 + 10) / 2 (neuronios da entrada + os da saída / 2)
c1 = Sequential()
c1.add(InputLayer(shape=(784,)))
c1.add(Dense(units=397, activation='relu'))
c1.add(Dense(units=397, activation='relu'))
c1.add(Dense(units=10, activation='softmax')) #softmax pois será gerada uma probabilidade para cada uma das classes

c1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
c1.fit(X_treinamento, y_treinamento, batch_size=256, epochs=100, validation_data=(X_teste, y_teste))

#Criando classificador (Com redução de dimensionalidade)
# 32 -> 21 -> 21 -> 10
c1 = Sequential()
c1.add(InputLayer(shape=(32,)))
c1.add(Dense(units=21, activation='relu'))
c1.add(Dense(units=21, activation='relu'))
c1.add(Dense(units=10, activation='softmax')) #softmax pois será gerada uma probabilidade para cada uma das classes

c1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
c1.fit(treinamento_codificado, y_treinamento, batch_size=256, epochs=100, validation_data=(teste_codificado, y_teste))