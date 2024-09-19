import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import os

os.chdir('C:/Users/fanta/OneDrive/Documentos/Programming/Python projects/Cursos/Deep Learning/BreastCancer')
previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

classificador = Sequential()
classificador.add(Dense(units= 8, activation= 'relu', kernel_initializer= 'normal', input_dim = 30)) #Neuronios na camada oculta. (Nº entradas + nº saídas) / 2. input_dim é só para a primeira camada oculta
classificador.add(Dropout(0.2))
classificador.add(Dense(units= 8, activation= 'relu', kernel_initializer= 'normal',)) #Adicionando mais uma camada oculta, sem o input dim
classificador.add(Dropout(0.2))
classificador.add(Dense(units= 1, activation = 'sigmoid')) #Camada saída

classificador.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['binary_accuracy'])

classificador.fit(previsores, classe, batch_size= 10, epochs= 100)
#Após rodar com vários testes, é interessante mudar os parâmetros para os melhores parâmetros encontrados nos testes.
novo = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178, 0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005,
                 0.04, 0.05, 0.015, 0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185, 0.84, 158, 0.363 ]])

previsao = classificador.predict(novo)
previsao = (previsao > 0.5) 
