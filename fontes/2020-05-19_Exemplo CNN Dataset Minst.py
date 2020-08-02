from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import TensorBoard as tb
from keras.preprocessing.image import img_to_array, load_img
import numpy as np

#  carrega os dados do MINST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Plota 4 imagens de exemplo em escala de cinza

plt.figure(0)
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
#Mostra o as imagens
plt.show()


# Apenas ajusta a matriz para as dimensões esperadas do TensorFlow
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
# Normaliza as entradas de 0-255 para 0-1
X_train = X_train / 255
X_test = X_test / 255
# Gera os vetores com as classes do conjunto de dados de treinamento e teste
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# Cria o modelo
model = Sequential()
#Convolução 2D com função de ativação Rectified Linear Units 32 kernels/Pesos (filtros) 
model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu', data_format='channels_first'))
print( model.output_shape)
#Camada de Pooling 	    
model.add(MaxPooling2D(pool_size=(2, 2)))
	
#Convolução 2D com função de ativação Rectified Linear Units 64 kernels/Pesos (filtros) 
model.add(Conv2D(64, (5, 5), activation='relu'))
print( model.output_shape)
#Camada de Pooling 	
model.add(MaxPooling2D(pool_size=(2, 2)))
	
#Remove 20% das ativações de entrada aleatoriamente 
model.add(Dropout(0.2))
#Converte o conjunto de imagens e um vetor unidimensional para a entrada da rede neural totalmente conectada
model.add(Flatten())
print( model.output_shape)
	
model.add(Dense(32, activation='sigmoid'))
print( model.output_shape)
model.add(Dense(16, activation='sigmoid'))
print( model.output_shape)
model.add(Dense(num_classes, activation='softmax'))
print( model.output_shape)

# compilando o modelo escolhendo como se dará nossa perda, otimização e métricas (parâmetros do Keras)
# mais informações em https://keras.io/losses/
# mais informações em https://keras.io/optimizers/
# mais informações em https://keras.io/metrics/
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# Treinamento do modelo
history = model.fit(X_train, y_train, validation_split = 0.2, epochs=3, batch_size=500)

# Plotagem da acurácia de Treinamento e Validação
plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Avaliação final do modelo no conjunto de testes
scores = model.evaluate(X_test, y_test, verbose=0)

#Salva o mode
model.save('MNIST.hdf5')

model  = load_model('MNIST.hdf5')

print("Erro da CNN: %.2f%%" % (100-scores[1]*100))