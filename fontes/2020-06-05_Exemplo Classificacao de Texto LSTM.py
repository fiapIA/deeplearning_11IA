# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:43:06 2020

@author: felip
"""

# LSTM with Dropout for sequence classification in the IMDB dataset
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# fix random seed for reproducibility
np.random.seed(7)
#O argumento num_words=10000 mantém as top 5000 ocorrências de palavras nos dados de treinamento.
top_words = 5000

# 0	One of the other reviewers has mentioned that ...	positive
# 1	A wonderful little production. <br /><br />The...	positive
# 2	I thought this was a wonderful way to spend ti...	positive
# 3	Basically there's a family where a little boy ...	negative
# 4	Petter Mattei's "Love in the Time of Money" is...	positive
# 5	Probably my all-time favorite movie, a story o...	positive
# 6	I sure would like to see a resurrection of a u...	positive
# 7	This show was an amazing, fresh & innovative i...	negative
# 8	Encouraged by the positive comments about this...	negative
# 9	If you like original gut wrenching laughter yo...	positive
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

print("Training entries: {}, labels: {}".format(len(X_train), len(y_test)))

# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
model = Sequential()

#A primeira camada é do tipo Embedding. Esta camada toma a palavra codificada e mapeia um vetor embutido (embedding) para cada índice-palavra. Esses vetores embutidos são aprimorados a medida que o modelo treina. 
#Essa camada adiciona uma dimensão extra no formato da saída (batch, sequence, embedding). Portanto, podemos dizer que o objetivo dessa camada é criar uma representação menor que max_len dos dados de entrada.
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))