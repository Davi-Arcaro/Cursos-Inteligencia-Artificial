import os
import pandas as pd

# Mude o diretório de trabalho para o local onde o arquivo CSV está localizado
os.chdir('C:/Users/fanta/OneDrive/Documentos/Programming/Python projects/Cursos/Deep Learning/BreastCancer')

# Agora leia o arquivo CSV
previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

from sklearn.model_selection import train_test_split
all = previsoresTreinamento, previsoresTeste, classeTreinamento, classeTeste = train_test_split(previsores, classe, test_size=0.25)

import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

classificador = Sequential()
classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform', input_dim = 30)) #Neuronios na camada oculta. (Nº entradas + nº saídas) / 2. input_dim é só para a primeira camada oculta
classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform',)) #Adicionando mais uma camada oculta, sem o input dim
classificador.add(Dense(units = 1, activation = 'sigmoid'))

otimizador = keras.optimizers.Adam(learning_rate= 0.001, weight_decay= 0.0001, clipvalue = 0.5) #Ajustando parametros do otimizador Adam   
classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
#classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

classificador.fit(previsoresTreinamento, classeTreinamento, batch_size = 10, epochs = 100)

pesos0 = classificador.layers[0].get_weights() #Ligando os pesos da camada de entrada na primeira oculta
# print(pesos0)
# print(len(pesos0))
pesos1 = classificador.layers[1].get_weights() #Ligando os pesos da primeira camada oculta com a segunda
pesos2 = classificador.layers[2].get_weights() #Ligando os pesos da segunda camada oculta com a camada de saída
#Por padrão, sempre haverá a unidade de bias, como se fosse uma segunda camada extra junta com as camadas criadas.

previsoes = classificador.predict(previsoresTeste)
previsoes = previsoes >= 0.5

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classeTeste, previsoes)
matriz = confusion_matrix(classeTeste, previsoes)

resultado = classificador.evaluate(previsoresTeste, classeTeste)
print(resultado)