import numpy as np

entradas = np.array([-1, 7, 5]) #Criando um array do numpy, otimizando o processo
pesos = np.array([0.8, 0.1, 0])

def soma(e,p): #e,p são entradas e pesos
    return e.dot(p) #dot = dot product ou produto escalar. Faz automaticamente todos os cálculos. Faz a multiplicação e o somatório
                    #Sempre usar .dot para fazer essas operações em redes neurais

s = soma(entradas,pesos)

def stepFunction(soma):
    if soma >= 1:
        return 1
    return 0 

resultado = stepFunction(s) #Aqui vai retornar o valor 1 ou 0 dependendo da função stepFunction()
print(s)
print(resultado)

#Este código é mais otimizado que o perceptron1, pois usa o numpy para otimizar o processo.