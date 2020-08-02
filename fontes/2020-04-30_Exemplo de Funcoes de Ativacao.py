# -*- coding: utf-8 -*-
"""
Created on Wed May 29 18:12:32 2019

@author: felip
"""

import numpy as np
import matplotlib.pyplot as plt
import math

#Funções
def parabola(z):
    return (z ** 2) +1

def binary(z, threashold):
    return 1 if z > threashold else 0

def piecewise(z, threashold):
    #return np.piecewise(x, [x < threashold, x >= threashold], [-1, 1])
    if z < threashold:
        return 0
    elif z > threashold:
        return 1
    else:
        return (z + threashold)/ threashold /2

def linear(z, x):
    return x*z

def leakyrelu(z, alpha):
	return max(alpha * z, z)

def relu(w):
  return max(0, w)

def sigmoid(z):
  return 1.0 / (1 + np.exp(-z))

def tanh(z):
	return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def elu(z,alpha):
	return z if z >= 0 else alpha*(np.exp(z) -1)

#Derivadas
def parabola_deriv(z):
    return 2*z;

def linear_prime(m):
	return m

def elu_prime(z,alpha):
	return 1 if z > 0 else alpha * np.exp(z)

def leakyrelu_prime(z, alpha):
	return 1 if z > 0 else alpha

def sigmoid_prime(z):
  return sigmoid(z) * (1-sigmoid(z))

def relu_prime(z):
  return 1 if z > 0 else 0

def tanh_prime(z):
	return 1 - np.power(tanh(z), 2)

f, (axes) = plt.subplots(3, 3, sharey=False)

#cria um vetor x entre -5 e 5 variando 0,1 a cada amostra gerada
x = np.arange(-10, 10, 0.1);
l = len(x)

#Função Parabola X2 +1

y1 = np.zeros(l)
y1_deriv = np.zeros(l)
for i in range(len(x)):
    y1[i] = parabola(x[i])
    y1_deriv[i] = parabola_deriv(x[i])

axes[0,0].set_title('Função Parabola (Apenas para exemplo)' )
axes[0,0].plot(x,y1, label='Função')
axes[0,0].plot(x,y1_deriv, label='Derivada')
axes[0,0].legend(loc="upper right")
axes[0,0].grid(True)

#Função Linear
y2 = linear(1, x)
y2_deriv = np.zeros(l);
y2_deriv + 1
axes[0,1].set_title('Função Linear' )
axes[0,1].plot(x,y2, label='Função')
axes[0,1].plot(x,y2_deriv, label='Derivada')
axes[0,1].legend(loc="upper right")
axes[0,1].grid(True)

#Função RelU
y3 = np.zeros(l)
y3_deriv = np.zeros(l)
for i in range(len(x)):
    y3[i] = relu(x[i])
    y3_deriv[i] = relu_prime(x[i])
axes[0,2].set_title('Função RelU' )
axes[0,2].plot(x,y3, label='Função')
axes[0,2].plot(x,y3_deriv, label='Derivada')
axes[0,2].legend(loc="upper right")
axes[0,2].grid(True)


#Função LeakyReLU    
y4 = np.zeros(l)
y4_deriv = np.zeros(l)
for i in range(len(x)):
    y4[i] = leakyrelu(x[i], 0.7)
    y4_deriv[i] = leakyrelu_prime(x[i], 0.7)
axes[1,0].set_title('Função LeakyReLU' )
axes[1,0].plot(x,y4, label='Função')
axes[1,0].plot(x,y4_deriv, label='Derivada')
axes[1,0].legend(loc="upper right")
axes[1,0].grid(True)


#Função Sigmoid
y5 = np.zeros(l)
y5_deriv = np.zeros(l)
for i in range(len(x)):
    y5[i] = sigmoid(x[i])
    y5_deriv[i] = sigmoid_prime(x[i])
axes[1,1].set_title('Função Sigmoid' )
axes[1,1].plot(x,y5, label='Função')
axes[1,1].plot(x,y5_deriv, label='Derivada')
axes[1,1].legend(loc="upper right")
axes[1,1].grid(True)


#Função Tangente Hiperbólica
y6 = np.zeros(l)
y6_deriv = np.zeros(l)
for i in range(len(x)):
    y6[i] = tanh(x[i])
    y6_deriv[i] = tanh_prime(x[i])
axes[1,2].set_title('Função Tangente Hiperbólica' )
axes[1,2].plot(x,y6, label='Função')
axes[1,2].plot(x,y6_deriv, label='Derivada')
axes[1,2].legend(loc="upper right")
axes[1,2].grid(True)

#Função ELU
y7 = np.zeros(l)
y7_deriv = np.zeros(l)
for i in range(len(x)):
    y7[i] = elu(x[i], 0.7)
    y7_deriv[i] = elu_prime(x[i], 0.7)
#plt.set_title('Função ELU' )
axes[2,0].set_title('Função ELU' )
axes[2,0].plot(x,y7, label='Função')
axes[2,0].plot(x,y7_deriv, label='Derivada')
axes[2,0].legend(loc="upper right")
axes[2,0].grid(True)

#Função Binária
y8 = np.zeros(l)
for i in range(len(x)):
    y8[i] = binary(x[i], 0)    

axes[2,1].set_title('Função Binária' )
axes[2,1].plot(x,y8, label='Função')
axes[2,1].legend(loc="upper right")
axes[2,1].grid(True)

#Função Piecewise-Linear
y9 = np.zeros(l)
for i in range(len(x)):
    y9[i] = piecewise(x[i], 4)    
axes[2,2].set_title('Função Piecewise-Linear' )
axes[2,2].plot(x,y9, label='Função')
axes[2,2].legend(loc="upper right")
axes[2,2].grid(True)