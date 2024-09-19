import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Flatten, Reshape
from tensorflow.keras.regularizers import L1L2

(X_treinamento, _), (_, _) = mnist.load_data() #O objetivo é enviar os pixels para que a rede aprenda a gerar novas imagens, por isso, não tem x_teste nem y
X_treinamento = X_treinamento.astype('float32') /255

#Na linha abaixo, indica que tem-se 60000 imagens dividas em 256 batches. Em cada batch tem 2343 imagens
X_treinamento = tf.data.Dataset.from_tensor_slices(X_treinamento).shuffle(buffer_size=60000).batch(batch_size=256)
# print(type(X_treinamento))

#Criação do Gerador
#Estrutura: 100 -> 500 -> 500
gerador = Sequential()
gerador.add(Dense(units=500, input_dim=100, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5))) #kernel_regularizer evita o Overfitting. Muito importante utilizar nessa arquitetura mais complexa
gerador.add(Dense(units=500, input_dim=100, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)))
#camada saída
gerador.add(Dense(units=784, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5))) #função de ativação poderia ser a sigmoid. Se não utilizar a sigmoid, serão retornados valores na escala dos pixels, 0 e 255
#Reshape de um vetor para uma matriz
gerador.add(Reshape((28,28)))
gerador.summary()

#Criação do Discriminador
discriminador = Sequential()
discriminador.add(InputLayer(shape=(28,28))) #São as imagens que a estrutura vai receber
#Transformando a imagem, que está em formato de matriz, em um vetor
discriminador.add(Flatten())
#Camadas ocultas
#Estrutura: 784 (imagens anteriores) -> 500 -> 500 -> 1
discriminador.add(Dense(units=500, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)))
discriminador.add(Dense(units=500, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)))
#Camada de saída
discriminador.add(Dense(units=1, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)))

discriminador.summary()

#Cálculo do Erro
cross_entropy = tf.keras.losses.BinaryCrossentropy()

def generator_loss(fake_output): #erro do gerador, vai receber as saídas falsas
    return cross_entropy(tf.ones_like(fake_output), fake_output) #Função ones_like vai retornar uma matriz de acordo com o tamanho de fake_output compostos somente por 1. Representa a comparação com o número 1 (verificar quanto as previsões do discriminador estão parecidas com 1). 1=Imagens reais
    #var fake_output indica as previsões da rede neural. Se for retornado 0.7 será comparado com 1. 0.8 compara com 1...

def discriminator_loss(real_outputs, fake_outputs):
    real_loss = cross_entropy(tf.ones_like(real_outputs), real_outputs) #Compara as imagens reais e compara com o valor 1, que é a saída esperada para imagens reais
    fake_loss = cross_entropy(tf.zeros_like(fake_outputs), fake_outputs) #Compara as imagens falsas e compara com a saída esperada para elas, que é = 0. Esses valores são as imagens geradas pela rede neural
    total_loss = real_loss + fake_loss
    return total_loss

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
discriminador_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

epochs = 100
noise_dim = 100

def train_step(images):
    noise = tf.random.normal([256, noise_dim]) #Gerando 100 números aleatórios. noise_dim precisa ter o mesmo tamanho do batch, que é 256
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = gerador(noise, training=True)
        
        #Em real_output e fake_output tem-se o resultado das probabilidades da função sigmoid. Probabilidade da imagem ser um cachorro ou não
        real_output = discriminador(images, training=True)
        fake_output = discriminador(generated_images, training=True)

        gen_loss = generator_loss(fake_output) #Enviando as probabilidades retornadas pelo discriminador para o gerador quanto as imagens geradas para que este aprenda a melhorar as imagens
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, gerador.trainable_variables) #Atualização do gradiente. Direção que os pesos serão atualizados
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminador.trainable_variables) 

    generator_optimizer.apply_gradients(zip(gradients_of_generator, gerador.trainable_variables))
    discriminador_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminador.trainable_variables))

    return gen_loss, disc_loss

for epoch in range(epochs): #percorrendo cada uma das épocas
    for image_batch in X_treinamento: #percorrendo todas as imagens
        gen_loss_batch, disc_loss_batch = train_step(image_batch)
        print(f'Época: {epoch} | gen_loss: {gen_loss_batch} | disc_loss: {disc_loss_batch}')

amostras = np.random.normal(size=(20, 100))
previsao = gerador.predict(amostras)
for i in range(previsao.shape[0]):
    plt.imshow(previsao[i,:], cmap='gray')
    plt.show()
        
