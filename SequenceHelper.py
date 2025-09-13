import numpy as np
import pandas as pd
from nltk.util import ngrams
from tensorflow.keras.preprocessing.sequence import pad_sequences as k_pad

number_to_alpha = {1: 'A', 2: 'C', 3: 'T', 4: 'G'}

def parse_alpha_to_seq(sequence):
    output = np.arange(len(sequence))
    for i in range(0, len(sequence)):
        snippet = sequence[i]
        if snippet == 'A':
            output[i] = 1       #0
        elif snippet == 'C':
            output[i] = 2       #1
        elif snippet == 'T':
            output[i] = 3       #2
        elif snippet == 'G':
            output[i] = 4       #3
        elif snippet == '0':        #new
            output[i] = 0
        elif snippet == 'N':
            output[i] = -1
        else:
            raise AssertionError("Cannot handle snippet: " + snippet)
    return output

def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy
    '''
    y = np.asarray(y, dtype='int32')
    if not nb_classes:
        nb_classes = np.max(y) + 1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        if y[i] != -1:
            Y[i, y[i]] = 1.
    return Y


def do_one_hot_encoding(sequence, seq_length):
    # X = np.zeros((sequence.shape[0], seq_length, 4))
    X = np.zeros((sequence.shape[0], seq_length, 5))
    for idx in range(0, len(sequence)):
        X[idx] = to_categorical(sequence[idx], 5)  #X[idx]
    return X

def do_dinucleotide_shuffling(X_df):
    X_shuffled_list = list()
    for x in range(0, X_df.shape[0]):
        x_as_list = list(X_df[x])
        random_index = np.arange(0, len(x_as_list)//2)
        np.random.shuffle(random_index)
        inital_base = number_to_alpha[np.random.randint(1, 5)]
        x_shuffled = [inital_base]*(len(x_as_list))
        # Shuffle odd and even positions
        for y in range(0, len(x_as_list)//2):
            x_shuffled[y*2] = x_as_list[random_index[y]*2]
            x_shuffled[(y*2)+1] = x_as_list[(random_index[y]*2)+1]

        X_shuffled_list.append("".join(x_shuffled))

    _pd = pd.DataFrame(X_shuffled_list)[0]
    return _pd


def augment_noise(df_column, noise, max_index=4):  #, quantity=10
    def add_noise(seq):
        mask = np.random.rand(len(seq)) < noise
        new_seq = [seq[i] if m == False else number_to_alpha[np.random.randint(1, max_index + 1)] for i, m in enumerate(mask)]
        return "".join(new_seq)

    new_df = df_column.copy()
    new_df = new_df.apply(add_noise)
    return new_df

def augment_grams(df_column, n_gram):
    def flatten_lol(lol):
        flat_l = list()
        for l in lol:
            flat_l += l
        return flat_l

    grams_extract = [["".join(g) for g in ngrams(seq, n_gram)] for seq in df_column]
    grams_extract = np.array(list(set(flatten_lol(grams_extract))))

    def sample_rnd_seq():
        rnd_indices = np.random.randint(0, len(grams_extract), np.random.randint(45 // n_gram, 64 // n_gram + 1, 1))
        return "".join(grams_extract[rnd_indices])

    def create_seq_from_ngrams(row):
        new_seq = sample_rnd_seq()
        return "".join(new_seq)

    new_df = df_column.copy()
    new_df = new_df.apply(create_seq_from_ngrams)
    return new_df, grams_extract

def merge_shuffle_neg_samples(neg_data_frames, target_n_samples):
    _merged = list()
    for df in neg_data_frames:
        _merged.append(df[np.random.randint(0, target_n_samples, target_n_samples//len(neg_data_frames))])
    _merged = pd.concat(_merged, ignore_index=True)
    _neg_rnd_indices = np.arange(_merged.values.shape[0])
    np.random.shuffle(_neg_rnd_indices)
    _merged = _merged[_neg_rnd_indices]
    _merged.reset_index(drop=True, inplace=True)
    return _merged


def parse_pad_onehot_encode(sequence_df_column):
    _int_enc = sequence_df_column.apply(parse_alpha_to_seq)
    _padded = k_pad(_int_enc.values.tolist(), padding='post')
    _one_hot_encoded = do_one_hot_encoding(_padded, _padded.shape[1])
    return _one_hot_encoded


