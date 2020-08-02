# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 13:40:34 2019

@author: felip
"""
#Para instalar o sklearn-extensions
#pip install sklearn-extensions

import pandas as pd

from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer


url = 'https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv'
titanic = pd.read_csv(url)
titanic.columns = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
titanic_data = titanic.drop(['PassengerId','Name','Ticket','Cabin'], 1)

def age_approx(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

titanic_data['Age'] = titanic_data[['Age', 'Pclass']].apply(age_approx, axis=1)
titanic_data.dropna(inplace=True)
gender = pd.get_dummies(titanic_data['Sex'],drop_first=True)
embark_location = pd.get_dummies(titanic_data['Embarked'],drop_first=True)
titanic_data.drop(['Sex', 'Embarked'],axis=1,inplace=True)
titanic_dmy = pd.concat([titanic_data,gender,embark_location],axis=1)
titanic_dmy.drop(['Pclass'],axis=1,inplace=True)
titanic_dmy.drop(['Q'],axis=1,inplace=True)

X = titanic_dmy.iloc[:,[1,2,3,4,5,6]].values
y = titanic_dmy.iloc[:,0].values


X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)
 
nh = 10

#Primeiro EML baseado em MPL com função de ativação sigmoid
srhl_sigmoid = MLPRandomLayer(n_hidden=nh, activation_func='sigmoid')
elm_model = GenELMClassifier(hidden_layer=srhl_sigmoid)
elm_model.fit(X_train, y_train)
score = elm_model.score(X_test, y_test)
print(score)

#Primeiro EML baseado em rede RBF
srhl_rbf = RBFRandomLayer(n_hidden=nh*2, rbf_width=0.1, random_state=0)
elm_model = GenELMClassifier(hidden_layer=srhl_rbf)
elm_model.fit(X_train, y_train)
score = elm_model.score(X_test, y_test)
print(score)
