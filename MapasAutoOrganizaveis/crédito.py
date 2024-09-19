import minisom
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.pylab import pcolor, colorbar, plot
from minisom import MiniSom
import os

os.chdir('C:/Users/fanta/OneDrive/Documentos/Programming/PythonProjects/Cursos/DeepLearning/MapasAutoOrganizaveis')

base = pd.read_csv('credit_data.csv')
#print(base)

#Executando alguns preprocessamentos
#print(base.isna().sum()) #Somar a quantidade de registros que possuem valores nulos
base = base.dropna() #Apagando os registros nulos
#print(base.shape)
#print(base.loc[base['age'] > 0].mean()) #Visualizando a média de tudo
#print(base['age'])
base.loc[base.age < 0, 'age'] = 40.92 #Algumas idades constavam menores que 0. Trocar elas por 40.92 que é a média das idades.
#print(base.loc[base['age'] < 0]) #Agora nenhum registro vai ser encontrado

X = base.iloc[:, 0:4].values
y = base.iloc[:, 4].values
#print(X) Essa variável consta todos os registros, desde o id até o valor da dívida
#print(y) Essa outra possui valores de 0 e 1, indicando 0 = não pagou o empréstimo, 1 = pagou

normalizador = MinMaxScaler(feature_range=(0,1))
X = normalizador.fit_transform(X)
#print(X) #Todos os valores agora estão nos valores entre 0 e 1. Necessário para o treinamento das redes neurais

print(X.shape) #1997 registros, fazer a raíz disso e depois multiplicar por 5. Para ter o tamanho da matriz na próxima etapa. 4 são as colunas, usadas no input_len abaixo

som = MiniSom(x=15, y=15, input_len=4, random_seed=0)
som.random_weights_init(X)
som.train_random(data= X, num_iteration=100)

pcolor(som.distance_map().T)
colorbar()

markers = ['o', 's']
colors = ['r','g']
for i,x in enumerate(X):
    w = som.winner(x) #Retorna qual é o neurônio vencedor considerando os 1997 registros. Cada um deles escolhe o BMU
    plot(w[0] + 0.5, w[1] + 0.5, markers[int(y[i])], markerfacecolor ='None',
          markersize=10, markeredgecolor=colors[y[i] - 1], markeredgewidth = 2)

plt.show()

mapeamento = som.win_map(X)
#print(mapeamento) #Quais foram os registros que fizeram a seleção dos BMUs em específico

suspeitos = np.concatenate((mapeamento[(4, 5)], mapeamento[(6, 13)]), axis=0) #axis = 0 representa linhas enquanto =1 representa colunas. Primeiro a coluna (4, 5) depois a linha
suspeitos = normalizador.inverse_transform(suspeitos)
#print(suspeitos)
classe = []
for i in range (len(base)):
    for j in range(len(suspeitos)):
        if base.iloc[i, 0] == int(round(suspeitos [j, 0])): # Se os IDs forem iguais, armazenar a classe
            classe.append(base.iloc[i, 4])

classe = np.array(classe)
print(classe)

suspeitos_final = np.column_stack((suspeitos, classe))
print(suspeitos_final)
suspeitos_final = suspeitos_final[suspeitos_final[:, 4].argsort()]
suspeitos_final[:, 0] = np.round(suspeitos_final[:, 0]).astype(int)
print(suspeitos_final)
