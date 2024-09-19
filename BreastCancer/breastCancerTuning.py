import pandas as pd
import tensorflow as tf
import keras
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

os.chdir('C:/Users/fanta/OneDrive/Documentos/Programming/Python projects/Cursos/Deep Learning/BreastCancer')

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

def criarRede(optimizer, loss, kernel_initializer, activation, neurons):
    classificador = Sequential()
    classificador.add(Dense(units= neurons, activation= activation, kernel_initializer= kernel_initializer, input_dim = 30)) #Neuronios na camada oculta. (Nº entradas + nº saídas) / 2. input_dim é só para a primeira camada oculta
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units= neurons, activation= activation, kernel_initializer= kernel_initializer,)) #Adicionando mais uma camada oculta, sem o input dim
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units= 1, activation = 'sigmoid')) #Camada saída

    classificador.compile(optimizer= optimizer, loss= loss, metrics= ['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn= criarRede)
parametros = {'batch_size' : [10, 30],
              'epochs' : [50, 100],
              'optimizer' : ['adam, sgd'],
              'loss' : ['binary_crossentropy', 'hinge'],
              'kernel_initializer' : ['random_uniform', 'normal'],
              'activation' : ['relu', 'tanh'],
              'neurons' : [16, 8]}
gridSearch = GridSearchCV(estimator= classificador, param_grid= parametros, scoring= 'accuracy', cv= 5)
gridSearch = gridSearch.fit(previsores, classe)

melhoresParametros = gridSearch.best_params_
melhorPrecisao = gridSearch.best_score_