import numpy as np

entradas = np.array([[0,0], [0,1], [1,0], [1,1]]) #Uma matriz, pensando no operador lógico AND. Para o [0,0] a saída é 0, [0,1] é 0 e assim por diante
saidas = np.array([0, 0, 0, 1]) #Saídas possíveis do operador lógico AND, onde 0=F e 1=V

# entradas = np.array([[0,0], [0,1], [1,0], [1,1]]) #Essa parte é a simulação com o operador OR ao invés do AND 
# saidas = np.array([0, 1, 1, 1]) #Aqui os valores mudaram se comparados ao operador AND

# entradas = np.array([[0,0], [0,1], [1,0], [1,1]]) #Essa parte é a simulação com o operador XOR. Valores iguais = 0, diferentes = 1
# saidas = np.array([0, 1, 1, 0]) #Mudando os valores para se adequar ao XOR.
#NÃO RODAR looping infinito

pesos = np.array([0.0, 0.0]) #Valores iniciais dos pesos
taxaApendizagem = 0.1 #Os valores são os mesmos da parte manual, disponíveis no vídeo 13 do curso Iniciante. A taxaApendizagem é um valor fixo.

def stepFuncion(soma):
    if soma >= 1:
        return 1
    return 0

def calculaSaida(registro): #Um registro é um conjunto [0,0], [0,1], [1,0] ou [1,1]. Vai tirar o produto escalar desses pesos e aplica a stepFunction
    s = registro.dot(pesos) #Pega esse registro e faz a multiplicação e depois o somatório, conforme a fórmula.
    return stepFuncion(s) #Dessa forma ele retorna ou 1 ou 0, de acordo com o valor de s pós somatório.

def treinar(): #Faz toda uma interação nos registros(entradas,saídas,pesos) e faz todo o processo de AJUSTE DE PESOS
    erroTotal = 1 #Inicializo erroTotal com valor 1 só para entrar no laço while abaixo
    while erroTotal != 0:
        erroTotal = 0
        for i in range(len(saidas)): #Pega o tamanho da array saidas e percorre todos os registros dela
            saidaCalculada = calculaSaida(np.asarray(entradas[i])) #Tenho uma matriz que é a entradas, e quero registro por registro. np.asarray() faz isso. O parâmetro é entradas[i] entradas na posição i, pega preimeiro o [0,0] depois faz o dot product e retorna 1 ou 0
            erro = abs(saidas[i] - saidaCalculada) #saidas[i] em parelho com o índice das entradas. Cada registro tem sua respectiva saída.
            erroTotal += erro #Atualiza o valor de erro total para o somatório do erro
            for j in range(len(pesos)): #Atualização nos pesos
                pesos[j] = pesos[j] + (taxaApendizagem * entradas[i][j] * erro) #essa é a fórmula para o ajuste dos pesos
                print('Peso atualizado: ' + str(pesos[j]))
        print('Total de erros: ' + str(erroTotal))

treinar()
print('Rede neural treinada')
print(calculaSaida(entradas[0]))
print(calculaSaida(entradas[1]))
print(calculaSaida(entradas[2]))
print(calculaSaida(entradas[3]))
#Basicamente essa é a aprendizagem em uma rede neural, vai fazendo laços de repetição while/for até que a máquina alcançe o meta