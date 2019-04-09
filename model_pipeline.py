import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
import re

# Reads in tweets from a csv file
#
# Parameters:
#   @filename title of csv filename
#   @fraction is the fraction of the data set to return [not randomized]
#   @verbose adds print messages to track process
# Returns:
#   @tweets is a list of the tweets
def read_tweets(filename, fraction = 1, verbose = True):
    # Read in tweets.
    if verbose:
        print("Reading in tweets...")

    tweets = []
    bad_tweets = []
    tweet_lengths = []

    file = open(filename, "r")
    line_count = 0

    for line in file:
        line_count += 1

        row = line.split(",")

        # row = line.split(",")

        if line_count == 0:
            print(f'Columns: {", ".join(row)}')
            columns = row
            line_count += 1
        else:
            line_count += 1

            row = row[1].replace("&amp;", "&")
            row = re.sub(r' http://.*$', "", row)
            row = re.sub(r' https://.*$', "", row)

            if len(row) > 280:
                bad_tweets.append([row])
            else:
                tweets.append([row])
                tweet_lengths.append(len(row))

    cut_off = int(len(tweets) * fraction)

    return tweets[0:cut_off], bad_tweets, tweet_lengths

# Pads each tweet with "<", which does not show up in any of the tweets, to
# make each tweet by default 280 characters, which is the maximum allowed by twitter
#
# Parameters:
#   @tweets the list of tweets made my funciton read_tweets
#   @pad_length which is the length each tweet will be padded to
#   @pad_character is the character that will be appended to the end of each tweet
#   @verbose adds print messages to track process
# Returns:
#   @padded_tweets the resulting padded tweets.
def pad_tweets(tweets, pad_length = 281, pad_character = "<", verbose = True):
    # Pad each tweet to 280 characters.
    if verbose:
        print("Padding tweets...")

    padded_tweets = []
    for tweet in tweets:
        length = len(tweet[0])

        padded_tweet = tweet[0]

        for i in range(pad_length - length):
            padded_tweet += pad_character

        padded_tweets.append([padded_tweet])

    return padded_tweets

# Define the mapping from characeters to numbers and vice versa
#
# Parameters:
#   @padded_tweets the final tweets paddded via function pad_tweets
# Returns:
#   @n_to_char is the mapping from numbers to characeters
#   @char_to_n is the mapping from characters to numbers
#
# Adapted from:
#   https://www.analyticsvidhya.com/blog/2018/03/text-generation-using-python-nlp/
def define_char_map(padded_tweets, verbose = True):
    if verbose:
        print("Defining Character/Number mappings...")

    tweet_text = ""

    for tweet in padded_tweets:
        tweet_text += tweet[0]

    characters = sorted(list(set(tweet_text)))
    n_to_char = {n:char for n, char in enumerate(characters)}
    char_to_n = {char:n for n, char in enumerate(characters)}

    return n_to_char, char_to_n, characters

# Adapted from:
#   https://www.analyticsvidhya.com/blog/2018/03/text-generation-using-python-nlp/
def create_dataset(padded_tweets, n_to_char, char_to_n, characters, seq_length, pad_length = 281, verbose = True):
    if verbose:
        print("Creating and Encoding data sets...")

    # Make the X and Y then encode
    X = []
    Y = []
    seq_length = 10

    for tweet in padded_tweets:
        for i in range(0, pad_length - seq_length, 1):
            sequence = tweet[0][i:i + seq_length]
            label = tweet[0][i + seq_length]
            X.append([char_to_n[char] for char in sequence])
            Y.append(char_to_n[label])

    X_modified = np.reshape(X, (len(X), seq_length, 1))
    X_modified = X_modified / float(len(characters))
    Y_modified = np_utils.to_categorical(Y)

    return X_modified, Y_modified, X

# TODO: make dynamic
# Currently Adapted from:
#   https://www.analyticsvidhya.com/blog/2018/03/text-generation-using-python-nlp/
def construct_model(X_set, Y_set, verbose = True):
    # Make model
    print("Making model...")

    model = Sequential()
    model.add(LSTM(500, input_shape = (X_set.shape[1], X_set.shape[2]), return_sequences = True))
    model.add(Dropout(0.15))
    model.add(LSTM(500))
    model.add(Dropout(0.15))
    model.add(Dense(Y_set.shape[1], activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

    return model


# Adapted from:
#   https://www.analyticsvidhya.com/blog/2018/03/text-generation-using-python-nlp/
# Generating Text
def generate_prediction(model, X, seq_length):
    string_mapped = X[59]
    # generating characters
    for i in range(seq_length):
        x = np.reshape(string_mapped,(1,len(string_mapped), 1))
        x = x / float(len(characters))
        pred_index = np.argmax(model.predict(x, verbose=0))
        seq = [n_to_char[value] for value in string_mapped]
        string_mapped.append(pred_index)
        string_mapped = string_mapped[1:len(string_mapped)]

    print(seq)

if __name__ == "__main__":

    tweets, bad_tweets, tweet_lengths = read_tweets("trump_tweets_copy.csv", fraction = 0.5)
    print("Number of tweets: ", len(tweets))
    tweet_lengths = np.reshape(tweet_lengths, (len(tweet_lengths), 1))


    # plt.hist(tweet_lengths, bins='auto')
    # plt.title("Tweet Length Histogram")
    # plt.xlabel("Number of Characters")
    # plt.ylabel("Instances")
    # plt.show()

    # Statistics
    print("Total num: ", len(tweet_lengths))
    print("Tweet min: ", np.min(tweet_lengths))
    print("Tweet max: ", np.max(tweet_lengths))
    print("Tweet ave: ", np.mean(tweet_lengths))

    padded_tweets = pad_tweets(tweets)
    n_to_char, char_to_n, characters = define_char_map(padded_tweets)
    X_set, Y_set, X = create_dataset(padded_tweets, n_to_char, char_to_n, characters, seq_length = 25)

    print([n_to_char[value] for value in X[59]])

    # Make model
    model = construct_model(X_set, Y_set)

    # Model training
    model.fit(X_set, Y_set, epochs = 7)

    # Generate prediction
    generate_prediction(model, X, 25)
