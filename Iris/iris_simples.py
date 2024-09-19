import pandas as pd
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.utils import to_categorical

os.chdir('C:/Users/fanta/OneDrive/Documentos/Programming/Python projects/Cursos/Deep Learning/Iris')

base = pd.read_csv('iris.csv') #Essa base de dados é diferente da breast cancer, tendo os valores previsores e as respostas todos juntos
previsores = base.iloc[:, 0:4].values #iloc é uma função que divide a base de dados iris.csv e recebe como parâmetro as linhas(registros) e o intervalo. .values converte para o formato do np
# # print(previsores) #Imprime somente os atributos, sem as classes(respostas)
classe = base.iloc[:, 4].values #De novo, .values para trabalhar com o numpy
# # print(classe) mostra as classes separadas dos previsores

#Atributos previsores = sepal_lenght, sepal_width, petal_lenght, petal_width (4)
#Atributos classificadores = Iris Setosa, Iris Versicolour, Iris Virginica (3) 

from sklearn.preprocessing import LabelEncoder #LabelEncoder = transforma a base de dados categórica(string) em numérica
label_encoder = LabelEncoder()
classe = label_encoder.fit_transform(classe) #os valores Iris Setosa... foram substituídos por 0, 1 e 2
classe_dummy = to_categorical(classe)
print(classe_dummy)
# iris setosa 1 0 0
# iris virginica 0 1 0
# iris versicolour 0 0 1

from sklearn.model_selection import train_test_split

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_dummy, test_size=0.25) #test_size= 25% da base será usada para teste, o restante é treinamento
# # train_test_split() sempre recebe 3 parâmetros: os previsores, as classes e o test_size

classificador = Sequential()
classificador.add(Dense(units=4, activation='relu', input_dim= 4)) #units= são os neuronios na 1ª camada oculta. Cálculo: entradas + saídas / 2 arrendondando sempre pra cima
classificador.add(Dense(units=4, activation='relu')) #2ª camada oculta, input_dim = quantos neuronios tem na camada de entrada. Ñ precisa nessa, apenas na 1ª
classificador.add(Dense(units=3, activation='softmax')) #Camada de saída. softmax é uma função para problemas com mais de duas classes de saída.
classificador.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

classificador.fit(previsores_treinamento, classe_treinamento, batch_size= 10, epochs=100)

resultado = classificador.evaluate(previsores_teste, classe_teste) #Pega os registros dos previsores e faz o comparativo do resultado com o classe_teste
# # O primeiro valor é a loss function e o segundo o valor de acerto (%)
previsoes = classificador.predict(previsores_teste) 
previsoes = previsoes > 0.5
# # print(previsoes)

classe_teste2 = [np.argmax(t) for t in classe_teste] #Alteração para printar a matriz de confusão. Retorna o índice de maior valor
previsoes2 = [np.argmax(t) for t in previsoes]

# # print(classe_teste2)
from sklearn.metrics import confusion_matrix
matriz = confusion_matrix(previsoes2, classe_teste2) #Matriz de confusão. Analiza facilmente quais classes estão errando mais
# # print(matriz)