import os
import librosa
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.optimizers import Adam


def extract_features(f):
    try:
        y, sr = librosa.load(f)
        mfcc = librosa.feature.mfcc(y, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        return np.vstack([mfcc, delta])
    except:
        print(f"{f} failed")

def get_filepaths(dirs):
    filepaths = []
    for d in dirs:
        filepaths += [os.path.join(d, f) for f in os.listdir(d)]
    return filepaths

def extract_features_from_files(dirs):
    files = get_filepaths(dirs)
    return [extract_features(f) for f in files]

def get_feats_with_window(S, window_size):
    features = []
    for i in range(window_size, S.shape[1]-window_size):
        feature = S[:, i-window_size:i+window_size]
        features.append(feature.reshape((-1)))
    return features

def read_labels(filename):
    df = pd.read_csv(filename, header=None)
    df.columns = ['File', 'Start', 'End']
    return df

def get_data_of_a_label(feats, label):
    X = [np.array(get_feats_with_window(f, window_size)) for f in feats]
    X = np.vstack(X)
    y = np.ones(len(X)) if label else np.zeros(len(X))
    return X, y

def get_data(applause_feats, non_applause_feats):
    applause_feats, applause_labels = get_data_of_a_label(
        applause_feats, 1)
    non_applause_feats, non_applause_labels = get_data_of_a_label(
        non_applause_feats, 0)
    X = np.vstack([applause_feats, non_applause_feats])
    y = np.concatenate([applause_labels, non_applause_labels])
    X, y = shuffle(X, y)
    return X, y

def calc_mean_std(X):
    means = np.zeros(X.shape[1])
    std_devs = np.zeros(X.shape[1])

    window_start = 0
    while window_start < X.shape[1]:
        mean = np.mean(X[:, window_start:window_start + 2*window_size])
        std = np.std(X[:, window_start:window_start + 2*window_size])
        means[window_start:window_start + 2*window_size] = mean
        std_devs[window_start:window_start + 2*window_size] = std
        window_start += 2*window_size

    return means, std_devs

def normalize_X(X, means, std_devs):
    for i in range(X.shape[1]):
        X[:, i] -= means[i]
        X[:, i] /= std_devs[i]
    return X

def initialize_ff_model():
    model = Sequential()
    model.add(Dense(1, input_dim=260))
    model.add(Activation('sigmoid'))
    model.compile(
        optimizer=Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return model


if __name__ == '__main__':
    applause_dirs = [
        '../data/applause-training-data/applause_pt1/',
        '../data/applause-training-data/applause_pt2/']
    non_applause_dirs = [
        '../data/applause-training-data/non_applause_pt1/',
        '../data/applause-training-data/non_applause_pt2/']

    applause_labels = read_labels(
        '../data/applause-training-data/PennSound_applause_labels.csv')
    non_applause_labels = read_labels(
        '../data/applause-training-data/PennSound_non_applause_labels.csv')

    applause_feats = extract_features_from_files(applause_dirs)
    applause_feats = [feat for feat in applause_feats if feat is not None]
    non_applause_feats = extract_features_from_files(non_applause_dirs)

    test_set_size = int(len(applause_feats) * 0.2)

    window_size = 5

    X_train, y_train = get_data(
        applause_feats[test_set_size:], non_applause_feats[test_set_size:])
    X_test, y_test = get_data(
        applause_feats[0:test_set_size], non_applause_feats[0:test_set_size])

    means, std_devs = calc_mean_std(X_train)
    # X_train = normalize_X(X_train, means, std_devs)
    # X_test = normalize_X(X_test, means, std_devs)

    model = initialize_ff_model()
    model.fit(X_train, y_train, epochs=1, batch_size=256, shuffle=True)
    model.evaluate(X_test, y_test, batch_size=256)
    model.save('models/applause-model.h5')