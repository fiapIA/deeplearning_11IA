# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 03:25:32 2019

@author: felip
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:05:38 2019

@author: felip
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics 

from keras.models import Sequential, Model
from keras.layers import Input, Dense
#necessita do pydot pip install pydot
from keras.utils import plot_model
import pydot
import matplotlib.pyplot as plt


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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)

# Cria o Modelo Sequencial
model = Sequential()
model.add(Dense(32, input_dim=6, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history  = model.fit(X_train, y_train, epochs=300, batch_size=10)

scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


print(model.summary())
# plot graph
plot_model(model, to_file='mlp-seq.png', show_shapes=True, show_layer_names=True)


# Plot training & validation accuracy values
plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# Cria o Modelo Funcional
inputs = Input(shape=(6,))
x = Dense(32,  activation='relu')(inputs)
x = Dense(32,  activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

modelFunc = Model(inputs =inputs, outputs =  predictions)
modelFunc.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
historyFunc  = modelFunc.fit(X_train, y_train, epochs=300, batch_size=10)

scores = modelFunc.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

print(modelFunc.summary())
plot_model(modelFunc, to_file='mlp-func.png', show_shapes=True, show_layer_names=True)


plt.plot(historyFunc.history['accuracy'])
#plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()