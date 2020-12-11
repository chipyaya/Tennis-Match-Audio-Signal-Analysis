import os
import librosa
import argparse
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
from torch.utils.data import Dataset
from argparse import RawTextHelpFormatter
from spafe.features.lfcc import lfcc


class AudioDataset(Dataset):
    def __init__(self, audio_dir, label_dir, audio_file, mode):
        self.audio_dir = audio_dir
        self.label_dir = label_dir
        self.audio_file = audio_file
        self.mode = mode
        self.load_csv()

    def load_csv(self):
        audio_labels_list = pd.read_csv(self.label_dir+self.audio_file+'.csv',
            header=None, sep='\t', skiprows=1)
        self.audio_labels_list = audio_labels_list

    def __len__(self):
        return len(self.audio_labels_list)

    def __getitem__(self, idx):
        audio = extract_features(self.audio_dir+self.audio_file+'.wav',
            self.audio_labels_list[1][idx], self.audio_labels_list[2][idx],
            self.mode)

        player_flag = self.audio_labels_list[0][idx]
        hand_flag = self.audio_labels_list[3][idx]
        dis_flag = self.audio_labels_list[4][idx]
        serve_flag = self.audio_labels_list[5][idx]

        return {"audio": audio, "player_flag": player_flag,
                "hand_flag": hand_flag, "dis_flag": dis_flag, "serve_flag": serve_flag}

def extract_features(f, start, end, mode):
    start = sum(x * int(t) for x, t in zip([60, 1], start.split(":")))
    end = sum(x * int(t) for x, t in zip([60, 1], end.split(":")))
    d = 2
    if mode == 'mfcc-avg' or mode == 'mfcc' or mode == 'mfcc-delta':
        y, sr = librosa.load(f, offset=start, duration=end-start+1)
        mfccs = librosa.feature.mfcc(y, n_mfcc=13)
        if mode == 'mfcc-avg':
            return np.mean(mfccs, axis=1)
        elif mode == 'mfcc-delta':
            delta = librosa.feature.delta(mfccs)
            return np.vstack([mfccs, delta])
        else:
            return mfccs
    elif mode == 'mel':
        y, sr = librosa.load(f, offset=start, duration=end-start+1)
        s = librosa.feature.melspectrogram(y, sr=sr)
        s = librosa.amplitude_to_db(s)
        # s = librosa.power_to_db(s)
        s = s.astype(np.float32)
        # plot_mel_spectrogram(s)
        return s
    elif mode == 'mfcc-4sec':
        y, sr = librosa.load(f, offset=max(0, end-d), duration=2*d)
        mfccs = librosa.feature.mfcc(y, n_mfcc=13)
        # plot_mfcc(mfccs)
        return mfccs
    elif mode == 'lfcc-4sec':
        y, sr = librosa.load(f, offset=max(0, end-d), duration=2*d)
        lfccs = np.swapaxes(lfcc(y, fs=50400, num_ceps=13), 0, 1)
        return lfccs
    elif mode == 'mfcc-lfcc-4sec':
        y, sr = librosa.load(f, offset=max(0, end-d), duration=2*d)
        mfccs = librosa.feature.mfcc(y, n_mfcc=13)
        lfccs = np.swapaxes(lfcc(y, fs=50400, num_ceps=13), 0, 1)
        return np.vstack([mfccs, lfccs])
    else:
        raise NotImplementedError

def plot_mfcc(mfcc):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mfcc, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title='MFCC (n_mfcc=13)')
    plt.savefig('../img/mfcc.png')
    print('saved')
    input()

def plot_mel_spectrogram(s):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(s,
        y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.savefig('../img/mel-spectrogram.png')
    print('saved')
    input()

def parse_arg():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--mode', type=str, default='mfcc-avg',
        help=textwrap.dedent('''\
        mfcc: use original mfcc;
        mfcc-avg: taking average of mfcc features;
        mfcc-4sec: use 4sec mfcc;
        mfcc-delta: use pure mfcc plus delta features;
        lfcc-4sec: use 4sec lfcc;
        mfcc-lfcc-4sec: use 4sec mfcc plus lfcc;
        mel: use melspectrogram;''')
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arg()
    audio_dir = '../data/complete_audio/'
    audio_files = ['berrettini_nadal', 'cilic_nadal', 'federer_dimitrov']
    # audio_files_2020 = ['zverev_thiem-2020']
    label_dir = '../data/label/'

    datasets = []
    for audio_file in audio_files:
        dataset = AudioDataset(audio_dir, label_dir, audio_file, args.mode)
        print(audio_file)
        print(dataset[0]['audio'].shape, dataset[0]['player_flag'],
            dataset[0]['hand_flag'], dataset[0]['dis_flag'], dataset[0]['serve_flag'])
        datasets.append(dataset)