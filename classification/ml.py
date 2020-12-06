import os
import sys
import pickle
import argparse
import textwrap
import numpy as np
from argparse import RawTextHelpFormatter
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import RidgeClassifier
sys.path.append("..")
from clipping.audio2mfcc import AudioDataset


def parse_arg():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--classifier', type=str, default='knn',
        help="available classifiers: knn, nb, rf, svm")
    parser.add_argument('--target', type=str, default='dis_flag',
        help="available targets: player_flag, hand_flag, dis_flag, serve_flag")
    parser.add_argument('--mode', type=str, default='mfcc-avg')
    parser.add_argument('--normalize', action='store_true')
    args = parser.parse_args()
    return args

def load_datasets(mode):
    audio_dir = '../data/complete_audio/'
    audio_files = ['berrettini_nadal', 'cilic_nadal', 'federer_dimitrov']
    # audio_files = ['zverev_thiem-2020']
    label_dir = '../data/label/'

    datasets = []
    for audio_file in audio_files:
        dataset = AudioDataset(audio_dir, label_dir, audio_file, args.mode)
        print(audio_file)
        datasets.append(dataset)
    print('audio feat: {}'.format(datasets[0][0]['audio'].shape))
    return datasets

def get_data(args):
    filename = '../cached/data-{}.p'.format(args.mode)
    if os.path.exists(filename):
        print('loading data from cache: {}'.format(filename))
        [X_train, X_test, y_train, y_test] = pickle.load(
            open(filename, 'rb'))
    else:
        datasets = load_datasets(args.mode)
        X, y = [], []
        for dataset in datasets:
            for i in range(len(dataset)):
                X.append(dataset[i]['audio'].ravel())
                y.append(dataset[i][args.target])
        X = np.array(X)
        y = np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=0)
        if args.normalize:
            X_train = normalize(X_train)
            X_test = normalize(X_test)
        print('X_train:{}, X_test:{}'.format(X_train.shape, X_test.shape))
        print('y_train:{}, y_test:{}'.format(y_train.shape, y_test.shape))
        print('dumping data to cache: {}'.format(filename))
        pickle.dump([X_train, X_test, y_train, y_test],
            open(filename, 'wb'))
    return X_train, X_test, y_train, y_test

def normalize(X):
    eps = 0.001
    normalized_dataset = []
    for img in X:
        if np.std(img) != 0:
            img = (img - np.mean(img)) / np.std(img)
        else:
            img = (img - np.mean(img)) / eps
        normalized_dataset.append(img)
    return np.array(normalized_dataset)


if __name__ == '__main__':
    args = parse_arg()
    X_train, X_test, y_train, y_test = get_data(args)
    if args.classifier == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=3)
    elif args.classifier == 'nb':
        classifier = GaussianNB()
    elif args.classifier == 'rf':
        classifier = RandomForestClassifier(max_depth=5, random_state=0)
    elif args.classifier == 'svm':
        classifier = svm.SVC()
    elif args.classifier == 'svm-linear':
        classifier = svm.SVC(kernel='linear')
    elif args.classifier == 'svm-poly':
        classifier = svm.SVC(kernel='poly')
    elif args.classifier == 'ridge':
        classifier = RidgeClassifier()
    else:
        raise NotImplementedError
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print('confusion_matrix:')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))