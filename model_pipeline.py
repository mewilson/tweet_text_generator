import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
import re

# TODO: make dynamic
# Currently Adapted from:
#   https://www.analyticsvidhya.com/blog/2018/03/text-generation-using-python-nlp/
def construct_model(X_set, Y_set, verbose = True):
    # Make model
    print("Making model...")

    model = Sequential()
    model.add(LSTM(400, input_shape = (X_set.shape[1], X_set.shape[2]), return_sequences = True))
    model.add(Dropout(0.25))
    model.add(LSTM(400))
    model.add(Dropout(0.25))
    model.add(Dense(Y_set.shape[1], activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

    return model


# Adapted from:
#   https://www.analyticsvidhya.com/blog/2018/03/text-generation-using-python-nlp/
# Generating Text
def generate_prediction(model, X, seq_length):
    string_mapped = X
    # generating characters
    for i in range(seq_length):
        x = np.reshape(string_mapped,(1,len(string_mapped), 1))
        x = x / float(len(characters))
        pred_index = np.argmax(model.predict(x, verbose=0))
        seq = [n_to_char[value] for value in string_mapped]
        string_mapped.append(pred_index)
        string_mapped = string_mapped[1:len(string_mapped)]

    print([n_to_char[value] for value in X])
    print([n_to_char[value] for value in string_mapped])

if __name__ == "__main__":

    X_set = np.load("data/sequences.npy")
    Y_set = np.load("data/next_char.npy")

    characters = np.load("data/characters.npy")
    characters = characters.tolist()

    n_to_char = {n:char for n, char in enumerate(characters)}
    char_to_n = {char:n for n, char in enumerate(characters)}

    # Make model
    model = construct_model(X_set, Y_set)

    # Model training
    model.fit(X_set, Y_set, epochs = 10)

    X = np.array([50, 68, 65, 1, 62, 78, 61, 74, 64, 1, 74, 65, 83, 1, 73, 61,
    74, 81, 79]).tolist()

    # Generate prediction
    generate_prediction(model, X, 19)
