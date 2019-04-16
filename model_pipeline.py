import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
import re
import sys
import time
import datetime
import pickle

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
        model.add(LSTM(units[index], return_sequences = True if (index != layers - 1) else False))
        model.add(Dropout(dropout[index]))

    # Add the final layer to make prediction
    model.add(Dense(output_shape, activation = activation))

    # Add loss and optimizer
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer,
                    metrics = ['acc','mae'])

    if verbose:
        print(model.summary())

    return model

def create_log(layers, unit, dropout, output, extended_output, epochs):

    entry = str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')) + \
            "," + str(layers) + "," + str(unit) + "," + str(dropout) + "," + str(output) + \
            "," + str(extended_output) + "," + str(epochs) + "\n"

    return entry


# Adapted from:
#   https://www.analyticsvidhya.com/blog/2018/03/text-generation-using-python-nlp/
# Generating Text
def generate_prediction(model, X, seq_length):
    string_mapped = X
    # generating characters
    for i in range(seq_length):
        print(string_mapped)
        x = np.reshape(string_mapped,(1,len(string_mapped), 1))
        x = x / float(len(characters))
        pred_index = np.argmax(model.predict(x, verbose=0))
        seq = [n_to_char[value] for value in string_mapped]
        string_mapped.append(pred_index)
        string_mapped = string_mapped[1:len(string_mapped)]

    print([n_to_char[value] for value in X])
    print([n_to_char[value] for value in string_mapped])

    return [n_to_char[value] for value in string_mapped]

if __name__ == "__main__":

    if (len(sys.argv) <= 1):
        print("Error: Path of logfile is unspecified.")
        exit()

    log = open(sys.argv[1], "a+")

    X_set = np.load("data/sequences.npy")
    Y_set = np.load("data/next_char.npy")

    characters = np.load("data/characters.npy")
    characters = characters.tolist()

    n_to_char = {n:char for n, char in enumerate(characters)}
    char_to_n = {char:n for n, char in enumerate(characters)}

    # X = np.array([50, 68, 65, 1, 62, 78, 61, 74, 64, 1, 74, 65, 83, 1, 73, 61,
    #     74, 81, 79, 63, 78, 69, 76, 80, 1, 66, 75, 78, 1, 61, 1, 74, 65, 83, 1,
    #     62, 75, 75, 71, 1, 62, 85, 1, 66, 61, 69, 72, 65, 64]).tolist()

    # Hyperparameter sets
    # !!!! NEVER SET layers_set = [1] , will break.
    # .... need to change logic in construct_lstm_model() to accomodate single layer set
    # .... this would be cool so that we can train models on just one layer to save time
    # .... when testing a new feature or some system configuration
    layers_set = [8]#[6, 8]
    units = [100]#[200, 400, 600]
    dropouts = [0.333] #[0.15, 0.20, 0.25]
    epochs = 10 #20

    # Call Back Functions
    i = time.strftime("%Y-%m-%d__%H-%M-%S")
    filepath = "logs/epoch-{epoch:02d}--loss-{loss:.5f}--time-%s-weights.hdf5" % i
    model_checkpoint = ModelCheckpoint(filepath = filepath, verbose = 1, monitor = 'loss')
    early_stopping = EarlyStopping(monitor = 'loss', patience = 5)
    call_backs = [model_checkpoint, early_stopping]

    for layers in layers_set:
        for unit in units:
            for dropout in dropouts:

                print("MODEL layers: " + str(layers) + ", units: " + str(unit)
                    + ", dropout: " + str(dropout) + "............................")

                # Make model
                model = construct_lstm_model((X_set.shape[1], X_set.shape[2]),
                    Y_set.shape[1], layers = layers, units = np.repeat(unit, layers),
                    dropout = np.repeat(dropout, layers))
                # Model training
                history = model.fit(X_set, Y_set, epochs = epochs, callbacks=call_backs, verbose = 1)

                # Save & Backup the Model (Architecture + Weights: Batteries Included)
                i = time.strftime("%Y-%m-%d__%H-%M-%S")
                print("the time is: %s" % i)
                model.save("models/%s.h5" % i)

                # Save the Model History for plotting later
                with open("pickle_barrel/model-history_%s.pickle" % i, "wb") as out_pickle:
                    pickle.dump(history, out_pickle, pickle.HIGHEST_PROTOCOL)

                X = np.array([50, 68, 65, 1, 62, 78, 61, 74, 64, 1, 74, 65, 83, 1, 73, 61,
                    74, 81, 79, 63, 78, 69, 76, 80, 1, 66, 75, 78, 1, 61, 1, 74, 65, 83, 1,
                    62, 75, 75, 71, 1, 62, 85, 1, 66, 61, 69, 72, 65, 64]).tolist()

                # Generate prediction
                output = generate_prediction(model, X, len(X))
                #extended_output = generate_prediction(model, X, 141)

                #datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                log.write(create_log(layers, unit, dropout, output, "extended_output", epochs))

    log.close()

    # # Make model
    # model = construct_lstm_model((X_set.shape[1], X_set.shape[2]),
    #     Y_set.shape[1], layers = 2, units = [2, 2], dropout = [0.30, 0.30])
    #
    # # Model training
    # print(Y_set.shape)
    # model.fit(X_set, Y_set, epochs = 1)
    #
    # X = np.array([50, 68, 65, 1, 62, 78, 61, 74, 64, 1, 74, 65, 83, 1, 73, 61,
    #     74, 81, 79, 63, 78, 69, 76, 80, 1, 66, 75, 78, 1, 61, 1, 74, 65, 83, 1,
    #     62, 75, 75, 71, 1, 62, 85, 1, 66, 61, 69, 72, 65, 64]).tolist()
    #
    # # Generate prediction
    # generate_prediction(model, X, len(X))
    # generate_prediction(model, X, 141)

    # Log result
