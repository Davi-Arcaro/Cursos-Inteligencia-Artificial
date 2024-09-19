import numpy as np
import pandas as pd
from keras.models import model_from_json
import os

os.chdir('C:/Users/fanta/OneDrive/Documentos/Programming/Python projects/Cursos/Deep Learning/BreastCancer')

arquivo = open('classificadorBreast.json', 'r')
estruturaRede = arquivo.read() #estrutura = funções, neuronios, biases...
arquivo.close()

classificador = model_from_json(estruturaRede)
classificador.load_weights('classificador_breast.weights.h5')

novo = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178, 0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005,
                 0.04, 0.05, 0.015, 0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185, 0.84, 158, 0.363 ]])

previsao = classificador.predict(novo)
previsao = (previsao > 0.5)

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')
classificador.compile(loss= 'binary_crossentropy', optimizer= 'adam', metrics= ['binary_accuracy'])
resultado = classificador.evaluate(previsores, classe)
print(resultado) #Primeira linha = valor da loss function; segunda linha = porcentagem de acerto