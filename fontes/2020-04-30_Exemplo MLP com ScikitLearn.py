# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:05:38 2019

@author: felip
"""
import pandas as pd
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler



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

#Parâmetros do MLP
# activation : {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’ Activation function for the hidden layer.

# ‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x
# ‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
# ‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
# ‘relu’, the rectified linear unit function, returns f(x) = max(0, x)
# solver : {‘lbfgs’, ‘sgd’, ‘adam’}, default ‘adam’
# --> Limited-memory BFGS = quasi-Newton methods


# The solver for weight optimization.
# ‘lbfgs’ is an optimizer in the family of quasi-Newton methods.
# ‘sgd’ refers to stochastic gradient descent.
# ‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba


mlp =MLPClassifier( max_iter=2000,learning_rate_init=0.01, hidden_layer_sizes=(5,5), activation='logistic', solver='sgd' )
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
print(metrics.accuracy_score(y_test, y_pred))