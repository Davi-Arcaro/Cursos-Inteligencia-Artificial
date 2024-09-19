import numpy as np
import rbm

rbm = rbm.RBM(num_visible= 6, num_hidden= 2) #Definindo os neurônios visíveis e escondidos

base = np.array([[1,1,1,0,0,0],
                 [1,0,1,0,0,0],
                 [1,1,1,0,0,0],
                 [0,0,1,1,1,1],
                 [0,0,1,1,0,1],
                 [0,0,1,1,0,1]])
#print(base.shape)
filmes = ['A bruxa', 'Invocação do mal', 'O chamado',
          'Se beber não case', 'Gente grande', 'American pie']

#Realizando o treinamento:
rbm.train(base, max_epochs=5000)
print(rbm.weights, rbm.weights.shape)

usuario1 = np.array([[1, 1, 0, 1, 0, 0]])

print(rbm.run_visible(usuario1)) #Fazendo a recomendação. O segundo neurônio se tornou especialista em filmes de terror enquanto o primeiro, especialista em comédia

usuario2 = np.array([[0,0,0,1,1,1]])
print(rbm.run_visible(usuario2)) #Mostrando a especialidade do primeiro neurônio

camada_escondida = np.array([[0,1]])
recomendacao = rbm.run_hidden(camada_escondida)

for i in range(len(usuario2[0])):
    if usuario1[0, i] == 0 and recomendacao[0, i] == 1:
        print(filmes[i])