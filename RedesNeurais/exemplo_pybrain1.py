import pybrain

rede = pybrain.FeedForwardNetwork()

camadaEntrada = pybrain.LinearLayer(2) #2 Neuronios na camada de entrada
camadaOculta = pybrain.SigmoidLayer(3) #3 Neuronios na camada oculta
CamadaSaida = pybrain.SigmoidLayer(1) #1 Neuronio na camada de saída
bias1 = pybrain.BiasUnit()
bias2 = pybrain.BiasUnit()

#Adicionando as camadas e bias na rede neural
rede.addModule(camadaEntrada)
rede.addModule(camadaOculta)
rede.addModule(CamadaSaida)
rede.addModule(bias1)
rede.addModule(bias2)

#Fazendo as ligações
entradaOculta = pybrain.FullConection(camadaEntrada, camadaOculta) #FullConection = faz a ligação do 1 neuronio com todos os outros, 2 com todos...
ocultaSaida = pybrain.FullConection(camadaOculta, CamadaSaida)
biasOculta = pybrain.FullConection(bias1, camadaOculta)
biasSaida = pybrain.FullConection(bias2, CamadaSaida)

rede.sortModules() #Construindo a rede