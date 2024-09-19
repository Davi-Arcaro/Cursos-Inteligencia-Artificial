import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import cross_val_score
from scikeras.wrappers import KerasRegressor

os.chdir('C:/Users/fanta/OneDrive/Documentos/Programming/Python projects/Cursos/Deep Learning/Carros')

base = pd.read_csv('autos.csv', encoding= 'ISO-8859-1')
base = base.drop('dateCrawled', axis= 1) #Tratamento da base de dados. axis=1 significa que quero apagar a coluna inteira
base = base.drop('nrOfPictures', axis=1)
base = base.drop('postalCode', axis=1)
base = base.drop('dateCreated', axis=1)
base = base.drop('lastSeen', axis=1)

base['name'].value_counts()
base = base.drop('name', axis=1)
base['seller'].value_counts()
base = base.drop('seller', axis=1)
base['offerType'].value_counts()
base = base.drop('offerType', axis=1)

i1 = base.loc[base.price <= 10] #i1 = preços incosistentes decorrentes da extração dos dados da internet
# # print(len(i1))
base = base[base.price > 10] #vai jogar pra dentro da base somente os veículos com preço maior que 10
i2 = base.loc[base.price > 350000]
# # print(len(i2))
base = base.loc[base.price < 350000]

#Segundo tratamento da base de dados. Substitui valores nulos pelos que mais aparecem

base.loc[pd.isnull(base['vehicleType'])] #verifica se os valores de vehicleType são nulos, ou seja, ñ foi cadastrado
# print(base['vehicleType'].value_counts()) #verifica qual o veículo que mais aparece. #limousine
base.loc[pd.isnull(base['gearbox'])] 
# print(base['gearbox'].value_counts()) #manuell
base.loc[pd.isnull(base['model'])] 
# print(base['model'].value_counts()) #golf
base.loc[pd.isnull(base['fuelType'])] 
# print(base['fuelType'].value_counts()) #benzin(gasolina)
base.loc[pd.isnull(base['notRepairedDamage'])] 
# print(base['notRepairedDamage'].value_counts()) #nein

#Trocando os valores nulos pelos que mais aparecem:
valores = {'vehicleType' : 'limousine',
           'gearbox' : 'manuell',
           'model' : 'golf',
           'fuelType' : 'benzin',
           'notRepairedDamage' : 'nein'}
base = base.fillna(value=valores) #faz a substituição na base de dados

previsores = base.iloc[:, 1:13].values #pegando os atributos previsores. passo o 0(1:13 não 0:13) pq esse é o objetivo, que é o preço
preco_real = base.iloc[:, 0].values #[:, 0 antes dos dois pontos sem ter nada quer dizer que quero pegar todas as linhas]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_previsores = LabelEncoder()
#Transformando os valores em string(categóricos) em valores numéricos. As redes neurais só trabalham com valores numéricos

previsores[:, 0] = label_encoder_previsores.fit_transform(previsores[:, 0])
previsores[:, 1] = label_encoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = label_encoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = label_encoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 8] = label_encoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = label_encoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 10] = label_encoder_previsores.fit_transform(previsores[:, 10])

#Importação do ColumnTransformer para a implementação do OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = [0, 1, 3, 5, 8, 9, 10]

column_transformer = ColumnTransformer(transformers=[('onehot', OneHotEncoder(), categorical_features,)])

previsores = column_transformer.fit_transform(previsores).toarray()

def criar_rede():
    regressor = Sequential()
    regressor.add(Dense(units= 158, activation='relu', input_dim= 312))
    regressor.add(Dense(units= 158, activation='relu'))
    regressor.add(Dense(units=1, activation='linear'))
    regressor.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    return regressor
regressor = KerasRegressor(build_fn=criar_rede, epochs=10, batch_size=300)

resultados = cross_val_score(estimator= regressor, X= previsores, y= preco_real, cv= 10, scoring= 'neg_mean_absolute_error')
media = resultados.mean()
desvio = resultados.std()
print(media,desvio)