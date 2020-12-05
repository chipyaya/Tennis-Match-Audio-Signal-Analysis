import os
import sys
import pickle
import argparse
import textwrap
import numpy as np
from argparse import RawTextHelpFormatter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
sys.path.append("..")
from clipping.audio2mfcc import AudioDataset


def parse_arg():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--target', type=str, default='dis_flag',
        help="possible targets: player_flag, hand_flag, dis_flag, serve_flag")
    parser.add_argument('--mode', type=str, default='avg',
        help=textwrap.dedent('''\
        pure: use pure mfcc wo taking average;
        delta: use pure mfcc plus delta features;
        avg: taking average of mfcc features'''))
    args = parser.parse_args()
    return args

def load_datasets(mode):
    datasets = pickle.load(open('../cached/datasets-{}.p'.format(mode), 'rb'))
    print(datasets[0][0]['audio'].shape, datasets[0][0]['player_flag'], 
        datasets[0][0]['hand_flag'] , datasets[0][0]['dis_flag'], 
        datasets[0][0]['serve_flag'])
    return datasets

def get_data(mode, target):
    datasets = load_datasets(mode)
    X, y = [], []
    for dataset in datasets:
        for i in range(len(dataset)):
            X.append(dataset[i]['audio'].ravel())
            y.append(dataset[i][target])
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    print('X_train:{}, X_test:{}'.format(X_train.shape, X_test.shape))
    print('y_train:{}, y_test:{}'.format(y_train.shape, y_test.shape))
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    args = parse_arg()
    X_train, X_test, y_train, y_test = get_data(args.mode, args.target)
    k_list = [3];
    for k in k_list:
        print('k={}'.format(k))
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))