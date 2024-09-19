import matplotlib.pyplot as plt
from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, InputLayer, Dropout, BatchNormalization
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import numpy as np
import os

os.chdir('C:/Users/fanta/OneDrive/Documentos/Programming/PythonProjects/Cursos/DeepLearning/RedesConvolucionais')

(X,y), (X_Teste, y_teste), = mnist.load_data()
X = X.reshape(X.shape[0], 28, 28, 1)
X = X.astype('float32')
X /= 255
y = to_categorical(y,10)

seed = 5
np.random.seed(seed)

kfold = StratifiedKFold(n_splits= 5, shuffle=True, random_state=seed)
resultados = []
for indice_treinamento, indice_teste in kfold.split(X, np.zeros(shape= (y.shape[0], 1))):
    print('Índices Treinamento: '), indice_treinamento, 'Índices Teste: ', indice_teste
    classificador = Sequential()
    classificador.add(InputLayer(shape=(28,28,1)))
    classificador.add(Conv2D(32, (3,3), activation='relu'))
    classificador.add(MaxPooling2D(pool_size=(2,2)))
    classificador.add(Flatten())
    classificador.add(Dense(units=128, activation='relu'))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units=64, activation='relu'))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units=32, activation='relu'))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units=16, activation='relu'))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units=10, activation='softmax'))
    classificador.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    classificador.fit(X[indice_treinamento], y[indice_treinamento], batch_size=128, epochs=5)
    precisao = classificador.evaluate(X[indice_teste], y[indice_teste])
    resultados.append(precisao[1])  
np.array(resultados).mean()
np.array(resultados).std()
print(resultados)

classificador.summary()