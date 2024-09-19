import tensorflow as tf
from sklearn.metrics import mean_absolute_error
import matplotlib
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM #Long Short Term Memory
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

os.chdir('C:/Users/fanta/OneDrive/Documentos/Programming/Python projects/Cursos/Deep Learning/RedesRecorrentes')

base = pd.read_csv('petr4_treinamento.csv')
base = base.dropna() #Remove os registros sem nenhum dado


base_treinamento = base.iloc[:,1:2].values #por enquanto pegar apenas a coluna com os valores de abertura das ações da petro

normalizador = MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

X = [] #Previsores
Y = [] #Preço Real das Ações
for i in range(90, 1242): #1242 pq tem 1241 preços de abertura na database, 90 pq vou usar os 90 registros anteriores para prever o próximo. Começa no 90 pq preciso de 3 meses (90 dias)
    X.append(base_treinamento_normalizada[i - 90 : i, 0])
    Y.append(base_treinamento_normalizada[i, 0])

X,Y = np.array(X), np.array(Y)

regressor = Sequential()

regressor.add(LSTM(units= 100, return_sequences=True, input_shape=(X.shape[1], 1)))
#LSTM(units=100) basicamente, vai indicar a quantidade de vezes que um neurônio manda a informação pra ele mesmo
#Não há uma regra definida para escolher o número de units, deve realizar testes
#return_sequences=True indica que o neurônio vai passar a informação para o próximo neurônio e ele mesmo

regressor.add(Dropout(0.3)) #30% dos neurônios serão zerados, para evitar Overfitting

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50,)) #Na última camada, os dados NÃO serão enviados para o próprio neurônio, removendo o return_sequences
regressor.add(Dropout(0.3))

regressor.add(Dense(units=1, activation='linear')) #Camada de saída. Para problemas de regressão, usa-se a ativação linear. Ao invés de LSTM, usa-se a camada Dense

#regressor.summary()

regressor.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])

regressor.fit(X, Y, epochs=100, batch_size=32)#var X são os 90 dias anteriores, Y os preços reais

base_teste = pd.read_csv('petr4_teste.csv')

y_teste = base_teste.iloc[:, 1:2].values #Selecionando todos os registros da coluna 1

base_completa = pd.concat((base['Open'], base_teste['Open']), axis=0)

entradas = base_completa[len(base_completa) - len(base_teste)- 90:].values #Tem a base de dados completa, remove os 22 registros de teste e também remove 90 registros

entradas = entradas.reshape(-1, 1) #Convertendo o vetor entradas para uma matriz
entradas = normalizador.transform(entradas) #Colocando os valores entre 0 e 1

X_teste = []
for i in range(90, 112):
    X_teste.append(entradas[i-90 : i,0])

X_teste = np.array(X_teste) #Transformando a lista X_teste em um array
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))

previsoes = regressor.predict(X_teste)
previsoes = normalizador.inverse_transform(previsoes)

print(previsoes)
print(f'Média das previsões: {previsoes.mean()}')

mean_absolute_error(y_teste, previsoes) #quanto a rede neural vai errar em centavos

plt.plot(y_teste, color= 'red', label='Preço real') #Gráfico para os preços reais
plt.plot(previsoes, color='blue', label='Previsões') #Gráfico para os preços previstos
plt.title("Previsão do preço das ações")
plt.xlabel('Tempo')
plt.ylabel('Valor')
plt.legend();