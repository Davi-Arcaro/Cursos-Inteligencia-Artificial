#A rede neural funciona assim: Tem uma entrada -> Inicializa os pesos -> Calcula saídas -> Calcula o erro -> Calcula os pesos -> Atualiza os pesos -> O erro é pequeno? se sim -> Termina programa. Se não -> volta pra Calcula Saídas
#Quantos neurônios eu preciso na minha camada de saída? Tem uma fórmula -> Neurônios = (Entradas + saídas) / 2. Depois, fazer os devidos ajustes
import numpy as np

def sigmoid(soma):
    return 1 / (1 + np.exp(-soma)) #exp no numpy faz o exponencial. Essa é a fórmula da função sigmoid

def sigmoidDerivada(sig): #Essa é a fórmula da derivada parcial, necessária para o Gradient Descent
    return sig * (1 -sig) #retorna o valor final da função sigmoid e faz os devidos cálculos
#Esse cálculo é usado para dar o direcionamento para o algoritmo pra qual lado do gradiente fazer a atualização dos pesos

# a = sigmoid(0.5) Testes da video aula do Gradient Descent, implementação V
# print(a)
# b = sigmoidDerivada(a)
# print(b)

entradas = np.array([[0,0], [0,1], [1,0], [1,1]]) #Quando for [0,0] a resposta é [0], [0,1] a resp é [1]...
saidas = np.array([[0], [1], [1], [0]]) #Operador XOR

# Pesos da camada de entrada para a oculta
#pesos0 = np.array([[-0.424, -0.740, -0.961],
#                   [0.358, -0.577, -0.469]])

#pesos1 = np.array([[-0.017], [-0.893], [0.148]]) #Pesos da camada de entrada para a camada oculta. Esses pesos estão na aula 24

pesos0 = 2*np.random.random((2,3)) -1 #2,3 porque são 2 neurônios na camada de entrada e 3 na oculta
pesos1 = 2*np.random.random((3,1)) -1 #3 neurônios na oculta e 1 na de saida
# 2*np.random.random((2,3)) -1 é o jeito de atribuir valores aleatórios para os pesos quando se inicia a rede neural.
# 2* é usado no começo para mesclar valores negativos e positivos. Deve usar o -1 no final para os valores finais ficarem entre -1 e 1

epocas = 1000 #Quantidade de vezes(rodadas) que os pesos serão atualizados. Também chamada de training time. Quanto maior, mais a rede neural aprende e mais precisa ela fica
taxaAprendizagem = 0.3 #Define o quão rápido a rede neural vai aprender. Valores maiores demoram menos, mas podem ocasionar maiores erros.
momento = 1 #Algoritmo que ajuda a evitar o mínimo local. Quanto menor, melhor é a evasão dos minímos, porém demora mais
#Ambos taxa de aprendizagem e momento seguem a mesma ideia: Maior valor, mais rápido, menos eficiente. Menor valor, mais demorado, mais eficiente.

#Processo FeedForward
for j in range(epocas): #Todo o código dos ajustes dos pesos fica dentro desse for
    camadaEntrada = entradas #variável auxiliar para n precisar usar a matriz das entradas
    somaSinapse0 = np.dot(camadaEntrada, pesos0) #Faz a multiplicação entre a camada de entrada com a oculta
    camadaOculta = sigmoid(somaSinapse0) #Faz a aplicação da função sigmoid para cada um dos dados da somaSinapse0
    
    somaSinapse1 = np.dot(camadaOculta,pesos1) #Faz a multiplicação entre a camada oculta e a de saída
    camadaSaida = sigmoid(somaSinapse1)
    
    erroCamadaSaida = saidas - camadaSaida #Cálculo do erro: respostaCorreta - respostaObtida. Usar a métrica abs(respostaCorreta - respostaObtida)
    mediaAbsoluta = np.mean(np.abs((erroCamadaSaida))) #mean() significa média, considera os sinais negativos. np.abs() pega os valores absolutos, assim, da pra fazer a média correta
#    print('Erro: ' + str(mediaAbsoluta))

    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = erroCamadaSaida * derivadaSaida #Fórmula do delta da camada de saída

    pesos1Transposta = pesos1.T #T é uma função para a matriz pesos1 ficar transposta, o que possibilita o cálculo na variável abaixo
    deltaSaidaXPeso = deltaSaida.dot(pesos1Transposta) #Somatório da multiplicação dos pesos1 pelo deltaSaida
    deltaCamadaOculta = deltaSaidaXPeso * sigmoidDerivada(camadaOculta) #Fórmula do delta da camada oculta
    
    camadaOcultaTransposta = camadaOculta.T #Tenho que fazer a transposta para fazer os cálculos abaixo
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida) #Multiplicação e somatório do deltaSaida pela camadaOcultaTransposta, atualizando o valor dos pesos da camada oculta para a de saida
    pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaAprendizagem) #Fórmula do Backpropagation, isso atualiza os pesos da camada oculta para a de saída para novos valores
    #Fórmula de atualização do Backpropagation: peso(n+1) = (peso(n) * momento) + (entrada * delta * taxa de aprendizagem)
    #Fórmula acima lê-se: peso atual = peso novo * momento + entrada * delta * tx aprendizagem
    #camadaOcultaTransposta e pesosNovo1 são a entrada*delta na operação.

    camadaEntradaTransposta = camadaEntrada.T #Fazer a transposta para fazer os cálculos abaixo
    pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta) #novo valor dos pesos da camada de entrada para a oculta
    pesos0 = (pesos0 * momento) + (pesosNovo0 * taxaAprendizagem)
print(camadaSaida)
print(pesos0)
print(pesos1)
