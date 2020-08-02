# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 03:35:54 2019

@author: felip
"""

from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D,AveragePooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import TensorBoard as tb
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from keras import preprocessing
from keras import optimizers
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
import keras
import tensorflow as tf
import cv2 as cv2

#print(keras.__version__)

#print(tf.__version__)

train_datagen = preprocessing.image.ImageDataGenerator(validation_split=0.30)
train_generator = train_datagen.flow_from_directory(
    'Datasets/Digitos/Treinamento',
    batch_size=200,
    color_mode = 'grayscale',
    subset='training',
    target_size=(30, 30))


valid_generator = train_datagen.flow_from_directory(
    'Datasets/Digitos/Treinamento',
    batch_size=200,
    color_mode = 'grayscale',
    subset='validation',
    target_size=(30, 30))

#Viasualizando 
x,y = train_generator.next()
for i in range(0,1):
    image = x[i]
    #cv2.imshow('30_30', image[:,:,0])
    plt.imshow(image[:,:,0])
    plt.show()

num_classes = 10

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

# Cria o modelo
model = Sequential()
#Convolução 2D com função de ativação Rectified Linear Units 32 kernels/Pesos (filtros) 
model.add(Conv2D(32, (7, 7), input_shape=(30,30,1), activation='relu')) #, data_format='channels_first'
#Camada de Pooling 	    
model.add(AveragePooling2D(pool_size=(2, 2)))
	
#Convolução 2D com função de ativação Rectified Linear Units 64 kernels/Pesos (filtros) 
model.add(Conv2D(64, (5, 5), activation='relu'))
#Camada de Pooling 	
model.add(AveragePooling2D(pool_size=(2, 2)))

#Convolução 2D com função de ativação Rectified Linear Units 64 kernels/Pesos (filtros) 
model.add(Conv2D(128, (3, 3), activation='relu'))
#Camada de Pooling 	
model.add(AveragePooling2D(pool_size=(2, 2)))
	
  #Remove 20% dos dados de entrada aleatoriamente 
model.add(Dropout(0.3))
#Converte o conjunto de imagens e um vetor unidimensional para a entrada da rede neural totalmente conectada
model.add(Flatten())

print( model.output_shape)
model.add(Dense(512, activation='relu'))
print( model.output_shape)
model.add(Dense(128, activation='relu'))
print( model.output_shape)
model.add(Dense(64, activation='relu'))
print( model.output_shape)
model.add(Dense(num_classes, activation='softmax'))
print( model.output_shape)


# Compila o modelo definindo o Adam optimization como algoritmo de atualização dos pesos dos neurônios  
#opt = optimizers.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


#earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('model_-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.h5', monitor='val_accuracy', mode= 'auto', save_weights_only=True, save_best_only=True)
#reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

model.summary()

model.fit_generator(train_generator, 
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10, callbacks=[mcp_save])


#TODO carregar o melhor modelo treinado


#Testando uma imagens
fileimg = 'Datasets\\Digitos\\Teste\\7\\digito_2_36-49-513863.png'

img = load_img(fileimg , color_mode = "grayscale", target_size=(30, 30))
#
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
#
y_pred = model.predict(x)
y_prob = model.predict_proba(x)

res = np.argmax(y_pred)

test_datagen = preprocessing.image.ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
    'Datasets/Digitos/Teste',
    batch_size=1,
    color_mode = 'grayscale',    
    target_size=(30, 30))


scores = model.evaluate_generator(valid_generator, steps=1)   
 
predictions = model.predict_generator(valid_generator, steps=1)


