import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM #Restricted Boltzmann Machines na biblioteca sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

base = datasets.load_digits() #Base de dados de dígitos escritos a mão
# print(base)

X = np.asarray(base.data, 'float32') #Previsores
# print(X.shape)
#plt.imshow(X[0].reshape((8,8)), cmap='gray')
# plt.show()

y = base.target #Classes

normalizador = MinMaxScaler(feature_range=(0,1))
X = normalizador.fit_transform(X)

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X,y, test_size=0.2, random_state=0) #20% para testar e 80% para treinar o algoritmo
# print(X_treinamento.shape, X_teste.shape)
# print(y_treinamento.shape, y_teste.shape)

#Nesse exemplo, se transforma os 64 pixels das imagens da base de dados em 50px, depois são enviados para o algoritmo naive_bayes para que consiga mensurar a taxa de acerto do algoritmo
rbm = BernoulliRBM(random_state=0)
rbm.n_iter = 25 #epochs
rbm.n_components = 50 #número de nós ocultos
naive_rbm = GaussianNB()

classificador_rbm = Pipeline(steps=[('rbm', rbm), ('naive', naive_rbm)]) #Sequência de tarefas que são executadas uma após a outra
classificador_rbm.fit(X_treinamento, y_treinamento) #Aplicando o treinamento

plt.figure(figsize=(20,20))
for i, componente in enumerate(rbm.components_):
    # print(i)
    # print(componente)
    plt.subplot(10,10, i + 1)
    plt.imshow(componente.reshape((8,8)), cmap=plt.cm.gray_r)
    plt.xticks(())
    plt.yticks(())
# plt.show()

previsoes_rbm = classificador_rbm.predict(X_teste)
previsao = metrics.accuracy_score(y_teste, previsoes_rbm)
print(previsao)

naive_simples = GaussianNB()
naive_simples.fit(X_treinamento, y_treinamento)

previsoes_naive = naive_simples.predict(X_teste)
print(previsoes_naive)

comparacao = metrics.accuracy_score(y_teste, previsoes_naive)
print(comparacao)