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

base_treinamento = base.iloc[:, 1:2] #Todas as linhas da coluna 1
base_valor_maximo = base.iloc[:, 2:3].values #Selecionando o valor máximo na base de dados. (High na base)

normalizador = MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)
base_valor_maximo_normalizada = normalizador.fit_transform(base_valor_maximo)

X = [] 
y1= [] 
y2 = []
for i in range(90, 1242):
    X.append(base_treinamento_normalizada[i-90 : i,0:6])
    y1.append(base_treinamento_normalizada[i,0])
    y2.append(base_valor_maximo_normalizada[i,0])
X, y1, y2 = np.array(X), np.array(y1), np.array(y2)

y = np.column_stack((y1, y2)) #unificando a var y1 e y2 (Open & High)

regressor = Sequential()

regressor.add(InputLayer(shape=(X.shape[1], 1)))
regressor.add(LSTM(units=100, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.3))

regressor.add(Dense(units=2, activation='linear'))

regressor.compile('rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])

es = EarlyStopping(monitor='loss', min_delta=1e-10, patience=10, verbose=True) #pacience se o valor da loss não diminuir de acordo com o min_delta, indica que o treinamento será interrompido
rlr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, verbose=True) #ReduceLearningRatePlateau. Diminui a taxa de aprendizagem baseado no factor. Se após 5 epochs o erro não diminuir, o valor da LR será diminuido
mcp = ModelCheckpoint(filepath='pesos.keras', monitor='loss', save_best_only=True, verbose=True)

regressor.fit(X,y, epochs=100, batch_size=32, callbacks=[es, rlr, mcp])

base_teste = pd.read_csv('petr4_teste.csv')
y_open = base_teste.iloc[:, 1:2].values
y_high = base_teste.iloc[:, 2:3].values

base_completa = pd.concat((base['Open'], base_teste['Open']), axis=0)

entradas = base_completa[len(base_completa) - len(base_teste) - 90].values
entradas = normalizador.transform(entradas)

X_teste = []
for i in range(90, 112):
    X_teste.append(entradas[i-90 : i,0])
X_teste = np.array(X_teste)
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))

previsoes = regressor.predict(X_teste)
previsoes = normalizador.inverse_transform(previsoes)

print(previsoes) #fazendo a comparação entre as previsoes da rede com os preços reais

print(previsoes.mean()) #extraindo as médias

from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_open.ravel(), previsoes[:,0])) #média de erro do preço da rede pra cima ou pra baixo
print(mean_absolute_error(y_high.ravel(), previsoes[:,1])) #média de erro do preço da rede pra cima ou pra baixo

plt.plot(y_open, color='red', label='Preço real')
plt.plot(y_high, color='black', label='Preço alta real')

plt.plot(previsoes[:, 0], color='blue', label='Previsões')
plt.plot(previsoes[:, 1], color='orange', label='Previsões alta')

plt.title('Previsão do preço das ações')
plt.xlabel('Tempo')
plt.ylabel('Valor')
plt.legend();
