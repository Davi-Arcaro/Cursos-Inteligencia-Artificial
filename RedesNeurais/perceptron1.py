entradas = [1, 7, 5]
pesos = [0.8, 0.1, 0]

def soma(e,p): #e,p são entradas e pesos
    soma = 0
    for i in range(3): #range(3) pois a lista de entradas e pesos tem 3 elementos
        #print(entradas[i]) #print só pra ver se está imprimindo os dados certos das listas
        #print(pesos[i])
        soma += e[i] * p[i] #fazendo a multiplicação e automaticamente fazendo o somatório
    return soma


s = soma(entradas,pesos)

def stepFunction(soma):
    if soma >= 1:
        return 1
    return 0 

resultado = stepFunction(s) #Aqui vai retornar o valor 1 ou 0 dependendo da função stepFunction()
print(s)
print(resultado)

