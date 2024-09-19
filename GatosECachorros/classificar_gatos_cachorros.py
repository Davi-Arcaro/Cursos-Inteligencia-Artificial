import tensorflow as tf
import tempfile
import zipfile
import os
import numpy as np

os.chdir('C:/Users/fanta/OneDrive/Documentos/Programming/PythonProjects/Cursos/DeepLearning/GatosECachorros')

temp_dir = tempfile.TemporaryDirectory()
print(temp_dir)

with zipfile.ZipFile('dataset.zip') as zip:
    zip.extractall(temp_dir.name)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

classificador = Sequential()
classificador.add(InputLayer(shape= (64, 64, 3)))

classificador.add(Conv2D(32, (3, 3), activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2, 2)))

classificador.add(Flatten())

classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=64, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=32, activation='relu'))
classificador.add(Dropout(0.2))

classificador.add(Dense(units=1, activation='sigmoid')) #units 1 pelo fato de ter um problema de classificação binária (ou é gato ou é cachorro 0 ou 1)

classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

gerador_treinamento = ImageDataGenerator(rescale=1./255, rotation_range=7, horizontal_flip=True, shear_range=0.2, height_shift_range=0.07, zoom_range=0.2)

gerador_teste = ImageDataGenerator(rescale=1./255) #rescale faz uma re escala dos pixels para q fiquem com valor entre 0 e 1

base_treinamento = gerador_treinamento.flow_from_directory(f'{temp_dir.name}/dataset/training_set', target_size=(64,64), batch_size=32, class_mode='binary')

base_teste = gerador_treinamento.flow_from_directory(f'{temp_dir.name}/dataset/test_set', target_size=(64,64), batch_size=32, class_mode='binary')

classificador.fit(base_treinamento, epochs=10, validation_data=base_teste)

classificador.summary()

imagem_teste = image.load_img(f'{temp_dir.name}/dataset/test_set/cachorro/dog.3500.jpg', target_size=(64,64))
imagem_teste = image.img_to_array(imagem_teste)
imagem_teste /=255

imagem_teste = np.expand_dims(imagem_teste, axis=0)

previsao = classificador.predict(imagem_teste)
print(type(previsao))

previsao = previsao > 0.5
print(previsao)

print(base_treinamento.class_indices)

classificador.save('classificador_gatos_cachorros.keras')
classificador_novo = tf.keras.models.load_model('C:/Users/fanta/OneDrive/Documentos/Programming/PythonProjects/Cursos/DeepLearning/GatosECachorros/classificador_gatos_cachorros.keras')