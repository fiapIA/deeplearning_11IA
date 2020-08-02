# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:36:53 2019

@author: felip
"""
import sklearn.datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics 

class Perceptron:
  
  #constructor
  def __init__ (self):
    self.w = None
    self.th = None
    
  #model  
  def model(self, x):
      return 1 if (np.dot(self.w, x) >= self.th) else 0
      #return 1 if (np.dot(self.w, x) == 0 ) else 0
  
  #predictor to predict on the data based on w
  def predict(self, X):
    Y = []
    for x in X:
      result = self.model(x)
      Y.append(result)
    return np.array(Y)
    
  def fit(self, X, Y, epochs = 1, lr = 1):
    self.w = np.ones(X.shape[1])
    self.th = 0
    accuracy = {}
    max_accuracy = 0
    wt_matrix = []
    #for all epochs
    for i in range(epochs):
      for x, y in zip(X, Y):
        y_pred = self.model(x)
        
        if y == 1 and y_pred == 0:
          self.w = self.w + lr * x
          self.th = self.th - lr * 1
          
        elif y == 0 and y_pred == 1:
          self.w = self.w - lr * x
          self.th = self.th + lr * 1
          
      wt_matrix.append(self.w)    
      accuracy[i] = metrics.accuracy_score(self.predict(X), Y)
      if (accuracy[i] > max_accuracy):
        max_accuracy = accuracy[i]
        chkptw = self.w
        chkptb = self.th    
    print(accuracy[i])
    print("Acurácia na ultima época {}".format( accuracy[i]))
    #checkpoint (Save the weights and b value)    
    self.w = chkptw
    self.th = chkptb
        
    print("Maior Acurácia encontrada {}".format(max_accuracy))
    #plot the accuracy values over epochs
    plt.plot(list(accuracy.values()))
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    plt.figure(1)
    plt.show()
    
    #return the weight matrix, that contains weights over all epochs
    return np.array(wt_matrix)

#load the breast cancer data
breast_cancer = sklearn.datasets.load_breast_cancer()

data = pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)
data["class"] = breast_cancer.target
data.head()
data.describe()

#plotting a graph to see class imbalance
data['class'].value_counts().plot(kind = "barh")
plt.xlabel("Count")
plt.ylabel("Classes")
plt.figure(0)
plt.show()



X = data.drop("class", axis = 1)
Y = data["class"]
mnscaler = MinMaxScaler()
X = mnscaler.fit_transform(X)
X = pd.DataFrame(X, columns=data.drop("class",axis = 1).columns)

#train test split.
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1, stratify = Y, random_state = 1)


perceptron = Perceptron()

X_train = X_train.values

#epochs = 10000 and lr = 0.3
wt_matrix = perceptron.fit(X_train, Y_train, 1000, 0.3 )

X_test = X_test.values

#Realiza as predições no Conjunto de Testes
Y_pred_test = perceptron.predict(X_test)

#Checa a acu´racia do Modelo
print("Acurácia no conjunto de Teste {}".format( metrics.accuracy_score(Y_pred_test, Y_test)))