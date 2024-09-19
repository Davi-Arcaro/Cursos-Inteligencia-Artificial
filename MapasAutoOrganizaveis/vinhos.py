import minisom
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import os

from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from matplotlib.pylab import pcolor, colorbar, plot


os.chdir('C:/Users/fanta/OneDrive/Documentos/Programming/PythonProjects/Cursos/DeepLearning/MapasAutoOrganizaveis')
base = pd.read_csv('wines.csv')
#print(base)

X = base.iloc[:, 1:14].values #Variável que vai receber os atributos previsores
y = base.iloc[:, 0].values

normalizador = MinMaxScaler(feature_range=(0,1)) 
X = normalizador.fit_transform(X) #Transformando todos os dados (números) da base em valores entre 0 e 1. Em redes neurais, todos os registros devem estar nessa faixa.
#print(X)
#print(X.shape) #(178,13) 13 registros serão colocados no input_len abaixo

som = MiniSom(x=8, y=8, input_len=13, sigma=1, learning_rate=0.5, random_seed=2) #x,y equivalem ao tamanho do mapa. Nessa base de dados, há 178 registros, que de acordo com a fórmula do tamanho (5x a raiz do número de registros) ficaria 65,65. A matriz 8x8 tem 64, a mais próxima dessa base de dados. Sigma é o processo de aprendizagem, o padrão é 1.

som.random_weights_init(X) #Inicializando os pesos na base de dados X de forma aleatória

som.train_random(data= X, num_iteration=15000) #Fazendo o treinamento. data é a base de dados tratata(X) e num_iteration são as epochs.

print(som._weights) #Visualizando os pesos
print(som._activation_map.shape) #Visualizando as dimensões (8,8). 8*8 = 64, que representam o número de neurônios na camada de saída.
print(som.activation_response(X)) #Visualizando quantas vezes cada um dos neurônios foi selecionado como BMU (Best Matching Unit)

pcolor(som.distance_map().T) #Visualizando todos os cálculos de distâncias considerando todos os neurônios nas camadas de saída. T no final indica matriz transposta
colorbar(); #MID - Mean Inter-Neuron Distance
w = som.winner(X[0]) #Selecionando somente o neurônio vencedor (BMU) para cada registro. Teremos 178 BMUs, um para cada um dos registros. Ir mudando a posição do X[0] -> X[1]... X[178]
print(w)

markers = ['o', 's', 'D']
colors = ['r', 'g', 'b']

pcolor(som.distance_map().T)
colorbar()

for i, x in enumerate(X): #percorrendo a base de dados e adicionando um índice nela
    #print(i)
    #print(x)
    w = som.winner(x)
    #print(w) #Visualizando os neurônios vencedores de cada registro
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i] - 1], markerfacecolor = 'None', markeredgecolor = colors[y[i] -1], markeredgewidth = 2, markersize = 10) #Fazendo os ajustes para a melhor visualização do mapa auto organizável
plt.show()
#Se o mapa não ficar muito bem agrupado, realizar treinamento com mais epochs