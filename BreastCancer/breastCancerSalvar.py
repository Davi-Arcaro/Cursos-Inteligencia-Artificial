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

# classificador_json = classificador.to_json()
# with open('classificadorBreast.json', 'w') as json_file:
#     json_file.write(classificador_json)
classificador.save_weights('classificador_breast.weights.h5')