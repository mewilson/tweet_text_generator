import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
import re

# Constructs a LSTM Model based on hyperparamters
# Parameters:
#   @input_shape
#   @output_shape
#   @layers is the number of LSTM layers that will be added to the model.
#   @units is the number of units per layer stored in a list.
#   @dropout is the dropout value per layer stored in a list.
#   @activation is the activation function for the last dense layer.
#   @optimizer is the optimizer to use when compiling the model.
#   @verbose adds print statements to track progress of model.
#
# Return:
#   @model
# Adapted from:
#   https://www.analyticsvidhya.com/blog/2018/03/text-generation-using-python-nlp/
def construct_lstm_model(input_shape, output_shape, layers = 1, units = [400], dropout = [0.25],
    activation = 'softmax', optimizer = "adam", verbose = True):

    # Make model
    if verbose:
        print("Making model...")

    if (layers < 1):
        print("ERROR construct_lstm_model: must have at least one layer.")
        return

    if (layers != len(units) or layers != len(dropout)):
        print("ERROR construct_lstm_model: units per layer are unspecified.")
        return

    model = Sequential()

    # Add first layer
    model.add(LSTM(units[0], input_shape = input_shape, return_sequences = True))

    if (len(dropout) > 1):
        model.add(Dropout(dropout[0]))

    # Add remaining layers if specified
    for index in np.arange(1, layers):
        model.add(LSTM(units[index]))
        model.add(Dropout(dropout[index]))

    # Add the final layer to make prediction
    model.add(Dense(output_shape, activation = activation))

    # Add loss and optimizer
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer)

    if verbose:
        print(model.summary())

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

def log():
    return

if __name__ == "__main__":

    X_set = np.load("data/sequences.npy")
    Y_set = np.load("data/next_char.npy")

    characters = np.load("data/characters.npy")
    characters = characters.tolist()

    n_to_char = {n:char for n, char in enumerate(characters)}
    char_to_n = {char:n for n, char in enumerate(characters)}

    # Make model
    model = construct_lstm_model((X_set.shape[1], X_set.shape[2]),
        Y_set.shape[1], layers = 2, units = [200, 200], dropout = [0.25, 0.25])

    # Model training
    model.fit(X_set, Y_set, epochs = 1)

    X = np.array([50, 68, 65, 1, 62, 78, 61, 74, 64, 1, 74, 65, 83, 1, 73, 61,
    74, 81, 79]).tolist()

    # Generate prediction
    generate_prediction(model, X, 19)

    # Log result
