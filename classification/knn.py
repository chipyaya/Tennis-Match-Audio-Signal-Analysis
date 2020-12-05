import os
import sys
import pickle
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
sys.path.append("..")
from clipping.audio2mfcc import AudioDataset


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='dis_flag',
        help="possible targets: player_flag, hand_flag, dis_flag, serve_flag")
    args = parser.parse_args()
    return args

def load_datasets():
    datasets = pickle.load(open('../cached/datasets.p', 'rb'))
    print(datasets[0][0]['audio'].shape, datasets[0][0]['player_flag'], 
        datasets[0][0]['hand_flag'] , datasets[0][0]['dis_flag'], 
        datasets[0][0]['serve_flag'])
    return datasets

def get_data(target):
    datasets = load_datasets()
    X, y = [], []
    for dataset in datasets:
        for i in range(len(dataset)):
            X.append(dataset[i]['audio'])
            y.append(dataset[i][target])
    X = np.array(X)
    y = np.array(y)
    print(X.shape)
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    args = parse_arg()
    X_train, X_test, y_train, y_test = get_data(args.target)
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))