import matplotlib.pyplot as plt
import numpy as np
import sys
from keras.utils import np_utils
import re

# Note pad_length must be at least 1 greater than seq_length

min_length = 50
seq_length = 49

# Global Variables
global_verbose = True # adds print statements to show the data processing

# Reads in tweets from a csv file
# Processes some symbols and removes hyperlinks
#
# Parameters:
#   @filename title of csv filename
#   @fraction is the fraction of the data set to return [not randomized]
#   @verbose adds print messages to track process
# Returns:
#   @tweets is a list of the tweets
def read_tweets(filename, fraction = 1, min_length = min_length, verbose = True):
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

            if len(row) > 280 or len(row) < min_length:
                bad_tweets.append([row])
            else:
                tweets.append([row])
                tweet_lengths.append(len(row))

    cut_off = int(len(tweets) * fraction)

    return tweets[0:cut_off], bad_tweets, tweet_lengths

# Creates statistics and graph of tweet lengths
def tweet_lengths_histogram(tweet_lengths, verbose = True):
    tweet_lengths = np.reshape(tweet_lengths, (len(tweet_lengths), 1))

    # Statistics
    if verbose:
        print("Total number of valid tweets: ", len(tweet_lengths))
        print("Tweet length minimum: ", np.min(tweet_lengths))
        print("Tweet length maximum: ", np.max(tweet_lengths))
        print("Tweet length average: ", np.mean(tweet_lengths))

    plt.hist(tweet_lengths, bins='auto')
    plt.title("Tweet Length Histogram")
    plt.xlabel("Number of Characters")
    plt.ylabel("Instances")
    plt.show()

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
def create_dataset(padded_tweets, n_to_char, char_to_n, characters,
    seq_length = seq_length, pad_length = min_length, verbose = True):

    if verbose:
        print("Creating and Encoding data sets...")

    # Make the X and Y then encode
    X = []
    Y = []

    for tweet in padded_tweets:
        for i in range(0, pad_length - seq_length, 1):
            sequence = tweet[0][i:i + seq_length]
            label = tweet[0][i + seq_length]
            X.append([char_to_n[char] for char in sequence])
            Y.append(char_to_n[label])

    X_modified = np.reshape(X, (len(X), seq_length, 1))
    X_modified = X_modified / float(len(characters))
    Y_modified = np_utils.to_categorical(Y)

    print(X[1])

    return X_modified, Y_modified, X

def main():
    if (len(sys.argv) <= 1):
        print("Error: Name of csv file to read from is unspecified.")
        return

    # Read in tweets
    tweets, bad_tweets, tweet_lengths = read_tweets(sys.argv[1], fraction = 1, verbose = global_verbose)

    if global_verbose:
        print("Number of valid tweets: ", len(tweets))


    # padded_tweets = pad_tweets(tweets, pad_length = 141, verbose = global_verbose)
    n_to_char, char_to_n, characters = define_char_map(tweets, verbose = global_verbose)

    X_set, Y_set, X = create_dataset(tweets, n_to_char, char_to_n, characters, verbose = global_verbose)

    np.save("data/sequences", X_set)
    np.save("data/next_char", Y_set)
    np.save("data/characters", np.asarray(characters))

    print(len(X[1]))
    print(X[1])
    print([n_to_char[value] for value in X[1]])



    return

if __name__ == "__main__" :
    main()
