#Mesmo código da rede neural multicamada, porém com a base de dados do breast cancer
import numpy as np
from sklearn import datasets #importando os essenciais para a implementação da base de dados

def sigmoid(soma):
    return 1 / (1 + np.exp(-soma)) 

def sigmoidDerivada(sig):
    return sig * (1 -sig) 

base = datasets.load_breast_cancer() #carregando a base de dados


entradas = base.data #camada de entrada recebe os dados 
valoresSaida = base.target #camada de saida recebe os valores 0 ou 1. 0 para benigno e 1 para maligno
saidas = np.empty([569, 1], dtype=int)
for i in range(569):
    saidas[i] = valoresSaida[i]

pesos0 = 2*np.random.random((30,5)) -1 #30 neuronios na camada de entrada, 5 na camada oculta
pesos1 = 2*np.random.random((5,1)) -1 #5 neuronios na camada oculta, 1 na de saida

epocas = 10000 
taxaAprendizagem = 0.3 
momento = 1 


#Processo FeedForward
for j in range(epocas): 
    camadaEntrada = entradas 
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0) 
    
    somaSinapse1 = np.dot(camadaOculta,pesos1) 
    camadaSaida = sigmoid(somaSinapse1)
    
    erroCamadaSaida = saidas - camadaSaida 
    mediaAbsoluta = np.mean(np.abs((erroCamadaSaida))) 
    print('Erro: ' + str(mediaAbsoluta))

    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = erroCamadaSaida * derivadaSaida 

    pesos1Transposta = pesos1.T 
    deltaSaidaXPeso = deltaSaida.dot(pesos1Transposta) 
    deltaCamadaOculta = deltaSaidaXPeso * sigmoidDerivada(camadaOculta) 
    
    camadaOcultaTransposta = camadaOculta.T 
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida) 
    pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaAprendizagem) 

    camadaEntradaTransposta = camadaEntrada.T 
    pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 = (pesos0 * momento) + (pesosNovo0 * taxaAprendizagem)

