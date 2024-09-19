import pandas as pd
import tensorflow as tf
import keras
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score

os.chdir('C:/Users/fanta/OneDrive/Documentos/Programming/Python projects/Cursos/Deep Learning/BreastCancer')

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

def criarRede():
    classificador = Sequential()
    classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform', input_dim = 30)) #Neuronios na camada oculta. (Nº entradas + nº saídas) / 2. input_dim é só para a primeira camada oculta
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform',)) #Adicionando mais uma camada oculta, sem o input dim
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 1, activation = 'sigmoid'))

    otimizador = keras.optimizers.Adam(learning_rate= 0.001, weight_decay= 0.0001, clipvalue = 0.5) #Ajustando parametros do otimizador Adam   
    classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn= criarRede, epochs= 100, batch_size= 10)
resultados = cross_val_score(estimator= classificador, X= previsores, y= classe, cv= 10, scoring= 'accuracy')
media = resultados.mean()
desvio = resultados.std()
print(resultados)
print(media)
print(desvio)