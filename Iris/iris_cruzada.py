import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score 
import os

os.chdir('C:/Users/fanta/OneDrive/Documentos/Programming/Python projects/Cursos/Deep Learning/Iris')

base = pd.read_csv('iris.csv') #Essa base de dados é diferente da breast cancer, tendo os valores previsores e as respostas todos juntos
previsores = base.iloc[:, 0:4].values #iloc é uma função que divide a base de dados iris.csv e recebe como parâmetro as linhas(registros) e o intervalo. .values converte para o formato do np

classe = base.iloc[:, 4].values #De novo, .values para trabalhar com o numpy

from sklearn.preprocessing import LabelEncoder #LabelEncoder = transforma a base de dados categórica(string) em numérica
label_encoder = LabelEncoder()
classe = label_encoder.fit_transform(classe) #os valores Iris Setosa... foram substituídos por 0, 1 e 2
classe_dummy = to_categorical(classe)

def criar_rede():
    classificador = Sequential()
    classificador.add(Dense(units=4, activation='relu', input_dim= 4)) #units= são os neuronios na 1ª camada oculta. Cálculo: entradas + saídas / 2 arrendondando sempre pra cima
    classificador.add(Dense(units=4, activation='relu')) #2ª camada oculta, input_dim = quantos neuronios tem na camada de entrada. Ñ precisa nessa, apenas na 1ª
    classificador.add(Dense(units=3, activation='softmax')) #Camada de saída. softmax é uma função para problemas com mais de duas classes de saída.
    classificador.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn= criar_rede, epochs=100, batch_size= 10)
resultados = cross_val_score(estimator= classificador, X= previsores, y= classe_dummy, cv= 10, scoring='accuracy')
media = resultados.mean() #faz a média dos resultados
desvio = resultados.std() #quantos valores estão variando de acordo com a média
print(media, desvio)