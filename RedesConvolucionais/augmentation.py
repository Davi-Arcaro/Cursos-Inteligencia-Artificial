from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, InputLayer, Dropout, BatchNormalization
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os

os.chdir('C:/Users/fanta/OneDrive/Documentos/Programming/Python projects/Cursos/Deep Learning/RedesConvolucionais')

(X,y), (X_Teste, y_teste), = mnist.load_data()
X = X.reshape(X.shape[0], 28, 28, 1) # O 1 ali no final é os canais de entrada (RGB), trabalhando com cinza apenas no momento. Se for trabalhar com imagens coloridas, usar 3
X = X.astype('float32')
X /= 255
y = to_categorical(y,10)

classificador = Sequential()
classificador.add(InputLayer(shape=(28,28,1)))
classificador.add(Conv2D(32, (3, 3), activation='relu'))
classificador.add(MaxPooling2D(pool_size=(2, 2)))
classificador.add(Flatten())
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=64, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=32, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=10, activation='softmax'))
classificador.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Aumentando a quantidade de imagens aplicando pequenas alterações
gerador_treinamento = ImageDataGenerator(rotation_range=7, horizontal_flip=True, shear_range=0.2, height_shift_range=0.07, zoom_range=0.2)

#Não fazer nenhum tipo de pre processamento, pois essa classe gera novas versões da imagem, e precisamos das novas versões para aumentar a quantidade de amostras
gerador_teste = ImageDataGenerator()                                         

base_treinamento = gerador_treinamento.flow(X, y, batch_size=128)
base_teste = gerador_teste.flow(X, y, batch_size=128)

classificador.fit(base_treinamento, epochs=10, validation_data=base_teste)

