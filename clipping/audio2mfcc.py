import os
import pickle
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, audio_dir, label_dir, audio_file):
        self.audio_dir = audio_dir
        self.label_dir = label_dir
        self.audio_file = audio_file
        self.load_csv()

    def load_csv(self):
        audio_labels_list = pd.read_csv(self.label_dir+self.audio_file+'.csv', header=None, sep='\t', skiprows=1)
        #self.audio_labels_list = list(audio_labels_list.iloc[:, 0])
        self.audio_labels_list = audio_labels_list
        '''
        print(self.audio_labels_list)
        print(self.audio_labels_list[0])
        print(self.audio_labels_list[0][0])
        '''
    def __len__(self):
        return len(self.audio_labels_list)

    def __getitem__(self, idx):
        audio = extract_features(self.audio_dir+self.audio_file+'_500.wav', self.audio_labels_list[1][idx], self.audio_labels_list[2][idx])

        player_flag = self.audio_labels_list[0][idx]
        hand_flag = self.audio_labels_list[3][idx]
        dis_flag = self.audio_labels_list[4][idx]
        serve_flag = self.audio_labels_list[5][idx]

        #print(self.audio_labels_list[:][idx], audio.shape, label)
        return {"audio": audio, "player_flag": player_flag,
                "hand_flag": hand_flag, "dis_flag": dis_flag, "serve_flag": serve_flag}

def extract_features(f, start, end):
    try:
        start = sum(x * int(t) for x, t in zip([60, 1], start.split(":")))
        end = sum(x * int(t) for x, t in zip([60, 1], end.split(":")))
        #y, sr = librosa.load(f, offset=start, duration=1)
        y, sr = librosa.load(f, offset=start, duration=end-start+1)
        mfcc = librosa.feature.mfcc(y, n_mfcc=13)
        return np.mean(mfcc, axis=1)
        '''
        delta = librosa.feature.delta(mfcc)
        print(delta.shape)
        return np.vstack([mfcc, delta])
        '''
    except Exception as e:
        print(e)
        print(f"{f} failed")


if __name__ == '__main__':
    audio_dir = '../data/audio/'
    audio_files = ['berrettini_nadal', 'cilic_nadal', 'federer_dimitrov']
    label_dir = '../data/label/'

    datasets = []
    for audio_file in audio_files:
        dataset = AudioDataset(audio_dir, label_dir, audio_file)
        print(audio_file)
        print(dataset[0]['audio'].shape, dataset[0]['player_flag'],
            dataset[0]['hand_flag'], dataset[0]['dis_flag'], dataset[0]['serve_flag'])
        datasets.append(dataset)
    pickle.dump(datasets, open('../cached/datasets.p', 'wb'))