import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Activation, Input
from tensorflow.keras.models import Model
import os

os.chdir('C:/Users/fanta/OneDrive/Documentos/Programming/Python projects/Cursos/Deep Learning/BaseGames')

base = pd.read_csv('games.csv')
base = base.drop('Other_Sales', axis=1) #axis=1 quero apagar a coluna
base = base.drop('Global_Sales', axis=1)
base = base.drop('Developer', axis=1)

base = base.dropna(axis=0) #axis=0 quer dizer que eu quero apagar a linha
base = base.loc[base['NA_Sales'] > 1]
base = base.loc[base['EU_Sales'] > 1]


nome_jogos = base.Name
base = base.drop('Name', axis=1)

previsores = base.iloc[:, [0,1,2,3,7,8,9,10,11]].values

vendaNA = base.iloc[:, 4].values #Todas as linhas e pego a coluna 4
vendaEU = base.iloc[: 5].values
vendaJP = base.iloc[: 6].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
label_encoder = LabelEncoder()

previsores[:,0] = label_encoder.fit_transform(previsores[:,0]) 
previsores[:,2] = label_encoder.fit_transform(previsores[:,2]) 
previsores[:,3] = label_encoder.fit_transform(previsores[:,3]) 
previsores[:,8] = label_encoder.fit_transform(previsores[:,8])

categorical_features = [0,2,3,8]
column_transformer = ColumnTransformer(transformers=[('onehot', OneHotEncoder(), categorical_features,)])
previsores = column_transformer.fit_transform(previsores).toarray()

camadaEntrada = Input(shape=(61,))
camadaOculta1 = Dense(units=32, activation='sigmoid')(camadaEntrada) #Difere do modelo sequencial.
camadaOculta2 = Dense(units=32, activation='sigmoid')(camadaOculta1) #esse outro parentese quer dizer 'depois de qual camada essa deve estar'
camadaSaida1 = Dense(units=1, activation='linear')(camadaOculta2) #Camada para vendas na NA
camadaSaida2 = Dense(units=1, activation='linear')(camadaOculta2) #Camada para vendas na EU
camadaSaida3 = Dense(units=1, activation='linear')(camadaOculta2) #Camada para vendas no JP

regressor = Model(inputs= camadaEntrada, outputs=[camadaSaida1, camadaSaida2,camadaSaida3])
regressor.compile(optimizer='adam', loss='mse')
regressor.fit(previsores, [vendaNA, vendaEU, vendaJP], epochs= 5000, batch_size=100)

previsaoNA = regressor.predict(previsores)
previsaoEU = regressor.predict(previsores)
previsaoJP = regressor.predict(previsores)