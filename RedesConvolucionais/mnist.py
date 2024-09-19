import matplotlib.pyplot as plt
from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, InputLayer, Dropout, BatchNormalization
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
import os

os.chdir('C:/Users/fanta/OneDrive/Documentos/Programming/Python projects/Cursos/Deep Learning/RedesConvolucionais')

(X_treinamento, Y_treinamento), (X_teste, Y_teste) = mnist.load_data()
plt.imshow(X_treinamento[0], cmap='gray')
plt.title('Classe' + str(Y_treinamento[0]))
previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0], 28, 28, 1)  
previsores_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')

previsores_treinamento /= 255
previsores_teste /= 255

classe_treinamento = to_categorical(Y_treinamento, 10)
classe_teste = to_categorical(Y_teste, 10)

rede_neural = Sequential()
rede_neural.add(InputLayer(shape=(28,28,1)))
rede_neural.add(Conv2D(filters=32, kernel_size = (3, 3), activation='relu'))
rede_neural.add(BatchNormalization())
rede_neural.add(MaxPooling2D(pool_size= (2, 2)))

rede_neural.add(Conv2D(filters=32, kernel_size = (3, 3), activation='relu'))
rede_neural.add(BatchNormalization())
rede_neural.add(MaxPooling2D(pool_size= (2, 2)))


rede_neural.add(Flatten())
rede_neural.add(Dense(units=128, activation='relu'))
rede_neural.add(Dropout(0.2))
rede_neural.add(Dense(units=128, activation='relu'))
rede_neural.add(Dropout(0.2))

rede_neural.add(Dense(units=10, activation='softmax'))

rede_neural.summary()

rede_neural.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
rede_neural.fit(previsores_treinamento, classe_treinamento, batch_size=128, epochs=5, validation_data= (previsores_teste, classe_teste))

resultado = rede_neural.evaluate(X_teste, Y_teste)