import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import sklearn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

os.chdir('C:/Users/fanta/OneDrive/Documentos/Programming/PythonProjects/Cursos/DeepLearning/RedesRecorrentes')

base = pd.read_csv('petr4_treinamento.csv')
base = base.dropna()

base_treinamento = base.iloc[:, 1:7] #Todas as linhas da coluna 1 à 6
print(base_treinamento)

normalizador = MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

X = [] #previsores
y= [] #preço real
for i in range(90, 1242):
    X.append(base_treinamento_normalizada[i-90 : i,0:6])
    y.append(base_treinamento_normalizada[i,0])
X, y = np.array(X), np.array(y)

regressor = Sequential()

regressor.add(LSTM(units=100, return_sequences=True, input_shape=(X.shape[1], X.shape[2]))) #X shape na posição 1 representa 90, que são os 90 dias anteriores
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.3))

regressor.add(Dense(units=1, activation='linear'))

regressor.summary()

regressor.compile('adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

es = EarlyStopping(monitor='loss', min_delta=1e-10, patience=10, verbose=True) #pacience se o valor da loss não diminuir de acordo com o min_delta, indica que o treinamento será interrompido
rlr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, verbose=True) #ReduceLearningRatePlateau. Diminui a taxa de aprendizagem baseado no factor. Se após 5 epochs o erro não diminuir, o valor da LR será diminuido
mcp = ModelCheckpoint(filepath='pesos.keras', monitor='loss', save_best_only=True, verbose=True)

regressor.fit(X,y, epochs=100, batch_size=32, callbacks=[es, rlr, mcp])

base_teste = pd.read_csv('petr4_teste.csv')

y_teste = base_teste.iloc[: 1:2].values

frames = [base, base_teste]

base_completa = pd.concat(frames)
base_completa = base_completa.drop('Date', axis=1) #Apagando a coluna que possui a data, não é necessária na previsão

entradas = base_completa[len(base_completa) - len(base_teste) - 90].values
entradas = normalizador.transform(entradas)

X_teste = []
for i in range(90, 112):
    X_teste.append(entradas[i-90 : i,0])
X_teste = np.array(X_teste)

normalizador_previsao = MinMaxScaler(feature_range=(0,1))
normalizador_previsao.fit_transform(base_treinamento[:, 0:1]) #Criando o normalizador para as previsoes

previsoes = regressor.predict(X_teste)
previsoes = normalizador_previsao.inverse_transform(previsoes)

print(previsoes) #fazendo a comparação entre as previsoes da rede com os preços reais
print(y_teste)

print(previsoes.mean()) #extraindo as médias
print(y_teste.mean())

from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_teste, previsoes)) #média de erro do preço da rede pra cima ou pra baixo

plt.plot(y_teste, color='red', label='Preço real')
plt.plot(previsoes, color='blue', label='Previsões')
plt.title('Previsão do preço das ações')
plt.xlabel('Tempo')
plt.ylabel('Valor')
plt.legend();
